"""V0.2 confidence-based state analysis with project field names."""
from core import schema
from .trajectory import forward_max
from typing import Optional, cast
import numpy as np
import pandas as pd

class AnaesthesiaAnalyser:

    def __init__(self, fps: float=30.0, still_window: float=120.0, still_speed: float=0.1, baseline_quantile: float=0.75, sedation_ratio: float=0.05, floor_height: float=0.05, ceiling_height: float=0.95, drop_window: float=1.0, drop_height: float=0.25, anaesthetic_onset: float=300.0, mixing_window: float=10.0, confidence: float=0.6, rebound_window: float=50.0, **kwargs):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        self.fps = float(fps)
        self.still_window = float(still_window)
        self.still_speed = float(still_speed)
        self.baseline_quantile = float(baseline_quantile)
        self.sedation_ratio = float(sedation_ratio)
        self.floor_height = float(floor_height)
        self.ceiling_height = float(ceiling_height)
        self.drop_window = float(drop_window)
        self.drop_height = float(drop_height)
        self.anaesthetic_onset = float(anaesthetic_onset)
        self.mixing_window = float(mixing_window)
        self.confidence = float(confidence)
        self.rebound_window = float(rebound_window)

    def evaluate_states(self, cleaned_df: pd.DataFrame, anaesthetic_onset: Optional[float]=None, *, _prepared=False) -> pd.DataFrame:
        cleaned_df = schema.normalise(cleaned_df) if cleaned_df is not None and not _prepared else cleaned_df
        if cleaned_df is None or cleaned_df.empty:
            return pd.DataFrame()
        anaesthetic_onset = float(anaesthetic_onset if anaesthetic_onset is not None else self.anaesthetic_onset)
        step_drop_frames = max(1, int(round(self.drop_window * self.fps)))
        smooth_3s_frames = max(1, int(round(3.0 * self.fps)))
        sustain_2s_frames = max(1, int(round(2.0 * self.fps)))
        rebound_frames = max(1, int(round(self.rebound_window * self.fps)))
        result_dfs = []
        for cid, grp in cleaned_df.groupby('chamber_id'):
            g = grp.copy().reset_index(drop=True) if _prepared else grp.sort_values('frame').reset_index(drop=True)
            n = len(g)
            if 'time' in g.columns:
                t = g['time'].to_numpy()
            else:
                t = g['frame'].to_numpy() / self.fps
                g['time'] = t
            heights = g['height'].to_numpy() if 'height' in g.columns else g['position'].to_numpy()
            speeds = g['speed'].to_numpy() if 'speed' in g.columns else np.zeros(n)
            pre_gas_mask = t < anaesthetic_onset
            pre_speeds = speeds[pre_gas_mask] if np.any(pre_gas_mask) else speeds[:int(60 * self.fps)]
            pre_speeds = pre_speeds[np.isfinite(pre_speeds)]
            moving_speeds = pre_speeds[pre_speeds > self.still_speed]
            if len(moving_speeds) >= int(self.fps):
                baseline_speed = float(np.percentile(moving_speeds, self.baseline_quantile * 100))
            else:
                baseline_speed = float(np.percentile(pre_speeds, 85)) if len(pre_speeds) > 0 else 50.0
            baseline_speed = max(baseline_speed, 30.0)
            is_drop_event = np.zeros(n, dtype=bool)
            if step_drop_frames < n:
                previous, current = heights[:-step_drop_frames], heights[step_drop_frames:]
                is_drop_event[step_drop_frames:] = (
                    (~(t[step_drop_frames:] < anaesthetic_onset)) & (previous > .35)
                    & (current <= self.floor_height) & ((previous - current) >= self.drop_height))
                if 'segment_id' in g:
                    segments = g['segment_id'].to_numpy()
                    is_drop_event[step_drop_frames:] &= segments[step_drop_frames:] == segments[:-step_drop_frames]
            post_gas_mask = t >= anaesthetic_onset
            post_indices = np.where(post_gas_mask)[0]
            at_floor = heights <= self.floor_height
            is_still = speeds <= self.still_speed
            quiescent_at_floor = post_gas_mask & at_floor & is_still
            is_anaesthesia = np.zeros(n, dtype=bool)
            first_ana_idx = None
            if len(post_indices) > 0:
                # Honour the configured window (120 s by default), even for
                # short recordings; insufficient observation is not an event.
                still_win_frames = max(1, int(round(self.still_window * self.fps)))
                rolling_ratio = np.asarray(pd.Series(quiescent_at_floor[post_indices].astype(float)).rolling(window=still_win_frames, min_periods=still_win_frames).mean())
                candidate_ends = np.where(rolling_ratio >= 0.9)[0]
                # Sparse candidates use NumPy directly; dense candidates share linear-time maxima.
                use_cache = len(candidate_ends) * min(n, rebound_frames) > 8 * n
                future_max = forward_max(heights, rebound_frames) if use_cache else None
                for q_end in candidate_ends:
                    cand_idx = post_indices[q_end - still_win_frames + 1]
                    maximum = (future_max[cand_idx] if future_max is not None
                               else np.max(heights[cand_idx:min(n, cand_idx + rebound_frames)]))
                    if maximum <= self.floor_height * 1.6:
                        first_ana_idx = cand_idx
                        is_anaesthesia[cand_idx:] = True
                        break
            t_mid = anaesthetic_onset + self.mixing_window * 0.5
            t_scale = max(1.0, self.mixing_window * 0.3)
            p_gas = 1.0 / (1.0 + np.exp(-np.clip((t - t_mid) / t_scale, -10.0, 10.0)))
            p_gas[~post_gas_mask] = 0.0
            v_smooth = np.asarray(pd.Series(speeds).rolling(smooth_3s_frames, center=True, min_periods=1).mean())
            v_sedate_target = baseline_speed * self.sedation_ratio
            e_speed = np.clip((baseline_speed * 0.5 - v_smooth) / max(0.001, baseline_speed * 0.5 - v_sedate_target), 0.0, 1.0)
            e_drop = np.asarray(pd.Series(is_drop_event.astype(float)).rolling(max(1, int(4.0 * self.fps)), min_periods=1).max())
            confidence = p_gas * np.clip(np.maximum(e_speed, e_drop * 1.25), 0.0, 1.0)
            invalid = ~(np.isfinite(speeds) & np.isfinite(heights))
            confidence[invalid] = 0.
            search_end_idx = first_ana_idx if first_ana_idx is not None else n
            first_sedate_idx = None
            conf_candidates = np.where(post_gas_mask & (confidence >= self.confidence))[0]
            conf_candidates = conf_candidates[conf_candidates < search_end_idx]
            if len(conf_candidates) > 0:
                rolling_conf_pass = np.asarray(pd.Series(confidence >= self.confidence).rolling(window=sustain_2s_frames, min_periods=sustain_2s_frames).sum() == sustain_2s_frames)
                c_ends = np.where(rolling_conf_pass)[0]
                c_ends_post = [idx for idx in c_ends if idx >= post_indices[0] and idx < search_end_idx]
                if len(c_ends_post) > 0:
                    first_sedate_idx = c_ends_post[0] - sustain_2s_frames + 1
            if first_sedate_idx is None and first_ana_idx is not None:
                first_sedate_idx = first_ana_idx
            is_sedate = np.zeros(n, dtype=bool)
            if first_sedate_idx is not None:
                end_sedate_idx = first_ana_idx if first_ana_idx is not None else n
                is_sedate[first_sedate_idx:end_sedate_idx] = True
            states = np.full(n, 'Active', dtype=object)
            states[is_sedate] = 'Sedate'
            states[is_anaesthesia] = 'Anaesthesia'
            states[invalid] = 'Unknown'
            g['state'] = states
            g['is_drop_event'] = is_drop_event.astype(int)
            g['baseline_speed'] = round(baseline_speed, 2)
            g['sedation_confidence'] = np.round(confidence, 3)
            result_dfs.append(g)
        return pd.concat(result_dfs, ignore_index=True)

    def extract_summary(self, df_with_states: pd.DataFrame, anaesthetic_onset: Optional[float]=None, *, _prepared=False) -> pd.DataFrame:
        df_with_states = schema.normalise(df_with_states) if df_with_states is not None and not _prepared else df_with_states
        if df_with_states is None or df_with_states.empty:
            return pd.DataFrame()
        anaesthetic_onset = float(anaesthetic_onset if anaesthetic_onset is not None else self.anaesthetic_onset)
        summaries = []
        for cid, grp in df_with_states.groupby('chamber_id'):
            g = grp.reset_index(drop=True) if _prepared else grp.sort_values('frame').reset_index(drop=True)
            if 'time' not in g.columns:
                g['time'] = g['time'] if 'time' in g.columns else g['frame'] / self.fps
            time_col = 'time'
            post_df = cast(pd.DataFrame, g[g[time_col] >= anaesthetic_onset])
            sed_rows = cast(pd.DataFrame, post_df[post_df['state'] == 'Sedate'])
            first_sed_time = float(sed_rows[time_col].iloc[0]) if not sed_rows.empty else None
            ana_rows = cast(pd.DataFrame, post_df[post_df['state'] == 'Anaesthesia'])
            first_ana_time = float(ana_rows[time_col].iloc[0]) if not ana_rows.empty else None
            if first_sed_time is None and first_ana_time is not None:
                first_sed_time = first_ana_time
            elif first_sed_time is not None and first_ana_time is not None:
                first_sed_time = min(first_sed_time, first_ana_time)
            sed_latency = round(first_sed_time - anaesthetic_onset, 2) if first_sed_time is not None else None
            ana_latency = round(first_ana_time - anaesthetic_onset, 2) if first_ana_time is not None else None
            total_drops = int(post_df['is_drop_event'].sum()) if 'is_drop_event' in post_df.columns else 0
            baseline_spd = float(g['baseline_speed'].iloc[0]) if 'baseline_speed' in g.columns else 0.0
            if first_sed_time is not None and 'sedation_confidence' in post_df.columns:
                sed_idx = np.searchsorted(post_df[time_col].to_numpy(), first_sed_time)
                conf_slice = post_df['sedation_confidence'].iloc[sed_idx:sed_idx + int(2.0 * self.fps)]
                confidence_score = float(np.mean(conf_slice)) if len(conf_slice) > 0 else 0.0
                confidence_score = min(1.0, max(0.0, round(confidence_score, 2)))
            else:
                confidence_score = 0.0
            summaries.append({'chamber_id': int(g['chamber_id'].iloc[0]), 'anaesthetic_onset': anaesthetic_onset, 'sedation_onset': round(first_sed_time, 2) if first_sed_time is not None else None, 'anaesthesia_onset': round(first_ana_time, 2) if first_ana_time is not None else None, 'sedation_latency': sed_latency, 'anaesthesia_latency': ana_latency, 'sedation_confidence': confidence_score, 'is_sedated': first_sed_time is not None, 'is_anaesthetised': first_ana_time is not None, 'drops': total_drops, 'baseline_speed': baseline_spd})
        return pd.DataFrame(summaries)
