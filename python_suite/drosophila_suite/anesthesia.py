"""
anesthesia.py
=================================================
基于果蝇受药动力学重构：
1. 自适应时序窗口 (Adaptive Temporal Window): 自动适应 500s 短序列与末尾击倒
2. 容错静止检测 (Tolerant Quiescence): 免疫麻醉后微抽搐与追踪噪点，杜绝硬重置
3. 局部反弹校验: 排除假性沉底，锁定真正不可逆深度麻醉
"""

import os
from typing import Optional, Dict, Any, List, Tuple, cast
import numpy as np
import pandas as pd


class AnesthesiaAnalyzer:
    def __init__(
        self,
        fps: float = 30.0,
        anesthesia_still_sec: float = 20.0,       # 沉底静止确认基准窗口 (短序列推荐 15~20s)
        speed_still_thresh: float = 2.0,          # 绝对静止速度门槛 (px/s)
        baseline_quantile: float = 0.75,          # 给药前运动期基线速度分位数
        sedate_speed_ratio: float = 0.05,         # 镇静减速比率 (低于基线 25%)
        floor_height_thresh: float = 0.15,        # 底部地面判定门槛 (Floor)
        ceiling_height_thresh: float = 0.80,      # 顶部天花板阈值 (Ceiling)
        drop_dt_sec: float = 1.0,                 # 跌落检测时间差 (s)
        drop_delta_h: float = 0.30,               # 跌落绝对下落高度幅度 (ΔH >= 0.30)
        anesthesia_onset_time: float = 300.0,     # 电磁阀/气路开启时刻 (s)
        gas_mixing_window_sec: float = 10.0,      # 气路混匀参考时间 (s)
        sedation_conf_thresh: float = 0.60,       # 镇静置信度门槛
        rebound_horizon_sec: float = 50.0,        # 局部抗反弹验证窗 (s)
        **kwargs
    ):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self.fps = float(fps)
        self.anesthesia_still_sec = float(anesthesia_still_sec)
        self.speed_still_thresh = float(speed_still_thresh)
        self.baseline_quantile = float(baseline_quantile)
        self.sedate_speed_ratio = float(sedate_speed_ratio)
        self.floor_height_thresh = float(floor_height_thresh)
        self.ceiling_height_thresh = float(ceiling_height_thresh)
        self.drop_dt_sec = float(drop_dt_sec)
        self.drop_delta_h = float(drop_delta_h)
        self.anesthesia_onset_time = float(anesthesia_onset_time)
        self.gas_mixing_window_sec = float(gas_mixing_window_sec)
        self.sedation_conf_thresh = float(sedation_conf_thresh)
        self.rebound_horizon_sec = float(rebound_horizon_sec)

    def evaluate_states(
        self,
        cleaned_df: pd.DataFrame,
        anesthesia_onset_time: Optional[float] = None
    ) -> pd.DataFrame:
        if cleaned_df is None or cleaned_df.empty:
            return pd.DataFrame()

        gas_onset = float(anesthesia_onset_time if anesthesia_onset_time is not None else self.anesthesia_onset_time)

        step_drop_frames = max(1, int(round(self.drop_dt_sec * self.fps)))
        smooth_3s_frames = max(1, int(round(3.0 * self.fps)))
        sustain_2s_frames = max(1, int(round(2.0 * self.fps)))
        rebound_frames = max(1, int(round(self.rebound_horizon_sec * self.fps)))

        result_dfs = []

        for cid, grp in cleaned_df.groupby("chamber_id"):
            g = grp.copy().sort_values("frame").reset_index(drop=True)
            n = len(g)

            # 1. 提取时间戳、高度与滤波速度
            if "timestamp_s" in g.columns:
                t = g["timestamp_s"].to_numpy()
            elif "timestamp" in g.columns:
                t = g["timestamp"].to_numpy()
                g["timestamp_s"] = t
            else:
                t = g["frame"].to_numpy() / self.fps
                g["timestamp_s"] = t

            heights = g["norm_height"].to_numpy() if "norm_height" in g.columns else g["norm_pos"].to_numpy()
            speeds = g["speed"].to_numpy() if "speed" in g.columns else np.zeros(n)

            # 2. 估计健康给药前基线移速
            pre_gas_mask = t < gas_onset
            pre_speeds = speeds[pre_gas_mask] if np.any(pre_gas_mask) else speeds[:int(60 * self.fps)]
            moving_speeds = pre_speeds[pre_speeds > self.speed_still_thresh]
            if len(moving_speeds) >= int(self.fps):
                baseline_speed = float(np.percentile(moving_speeds, self.baseline_quantile * 100))
            else:
                baseline_speed = float(np.percentile(pre_speeds, 85)) if len(pre_speeds) > 0 else 50.0
            baseline_speed = max(baseline_speed, 30.0)

            # 3. 标记跌落动作 (Knockdown Drop)
            is_drop_event = np.zeros(n, dtype=bool)
            for i in range(step_drop_frames, n):
                if t[i] < gas_onset:
                    continue
                h_pre = heights[i - step_drop_frames]
                h_now = heights[i]
                if (h_pre > 0.35) and (h_now <= self.floor_height_thresh) and ((h_pre - h_now) >= self.drop_delta_h):
                    is_drop_event[i] = True

            # 4. 深度麻醉判定 (自适应短序列 + 容错静止率)
            post_gas_mask = t >= gas_onset
            post_indices = np.where(post_gas_mask)[0]

            at_floor = heights <= self.floor_height_thresh
            is_still = speeds <= self.speed_still_thresh
            quiescent_at_floor = post_gas_mask & at_floor & is_still

            is_anaesthesia = np.zeros(n, dtype=bool)
            first_ana_idx = None

            if len(post_indices) > 0:
                # 动态自适应窗口：根据视频总长自动调节窗口，短视频采用 15~20s
                effective_still_sec = min(self.anesthesia_still_sec, max(10.0, (t[-1] - gas_onset) * 0.15))
                still_win_frames = max(1, int(round(effective_still_sec * self.fps)))

                # 容错静止积分 (允许 <= 10% 的微抽搐噪点)
                rolling_ratio = np.asarray(
                    pd.Series(quiescent_at_floor[post_indices].astype(float))
                    .rolling(window=still_win_frames, min_periods=still_win_frames)
                    .mean()
                )

                candidate_ends = np.where(rolling_ratio >= 0.90)[0]

                for q_end in candidate_ends:
                    cand_idx = post_indices[q_end - still_win_frames + 1]

                    # 局部反弹校验 (60s 内不得出现明显爬升 > 0.25)
                    eval_end = min(n, cand_idx + rebound_frames)
                    future_h = heights[cand_idx:eval_end]

                    if np.max(future_h) <= (self.floor_height_thresh * 1.6):
                        first_ana_idx = cand_idx
                        is_anaesthesia[cand_idx:] = True
                        break

                # 录像末尾终态沉底兜底 (Terminal Arrest: 针对 8 号这类在视频最后十几秒彻底倒地的果蝇)
                if first_ana_idx is None:
                    tail_dur_frames = int(10.0 * self.fps)
                    if n > tail_dur_frames:
                        tail_h = heights[-tail_dur_frames:]
                        tail_v = speeds[-tail_dur_frames:]
                        if np.all(tail_h <= self.floor_height_thresh) and np.mean(tail_v <= self.speed_still_thresh) >= 0.85:
                            # 逆向寻找最后一次进入管底不再起来的起点
                            for rev_i in range(n - 1, post_indices[0], -1):
                                if heights[rev_i] > self.floor_height_thresh:
                                    terminal_cand = rev_i + 1
                                    first_ana_idx = terminal_cand
                                    is_anaesthesia[terminal_cand:] = True
                                    break

            # 5. 镇静起始点 (Sedation Onset) 动态反推
            t_mid = gas_onset + (self.gas_mixing_window_sec * 0.5)
            t_scale = max(1.0, self.gas_mixing_window_sec * 0.30)
            p_gas = 1.0 / (1.0 + np.exp(-np.clip((t - t_mid) / t_scale, -10.0, 10.0)))
            p_gas[~post_gas_mask] = 0.0

            v_smooth = np.asarray(pd.Series(speeds).rolling(smooth_3s_frames, center=True, min_periods=1).mean())
            v_sedate_target = baseline_speed * self.sedate_speed_ratio
            e_speed = np.clip((baseline_speed * 0.50 - v_smooth) / max(1e-3, (baseline_speed * 0.50 - v_sedate_target)), 0.0, 1.0)
            e_drop = np.asarray(pd.Series(is_drop_event.astype(float)).rolling(max(1, int(4.0 * self.fps)), min_periods=1).max())
            confidence = p_gas * np.clip(np.maximum(e_speed, e_drop * 1.25), 0.0, 1.0)

            search_end_idx = first_ana_idx if first_ana_idx is not None else n
            first_sedate_idx = None

            conf_candidates = np.where(post_gas_mask & (confidence >= self.sedation_conf_thresh))[0]
            conf_candidates = conf_candidates[conf_candidates < search_end_idx]

            if len(conf_candidates) > 0:
                rolling_conf_pass = np.asarray(
                    pd.Series(confidence >= self.sedation_conf_thresh)
                    .rolling(window=sustain_2s_frames, min_periods=sustain_2s_frames)
                    .sum() == sustain_2s_frames
                )

                c_ends = np.where(rolling_conf_pass)[0]
                c_ends_post = [idx for idx in c_ends if (idx >= post_indices[0]) and (idx < search_end_idx)]
                if len(c_ends_post) > 0:
                    first_sedate_idx = c_ends_post[0] - sustain_2s_frames + 1

            if first_sedate_idx is None and first_ana_idx is not None:
                first_sedate_idx = first_ana_idx

            # 6. 整合状态输出
            is_sedate = np.zeros(n, dtype=bool)
            if first_sedate_idx is not None:
                end_sedate_idx = first_ana_idx if first_ana_idx is not None else n
                is_sedate[first_sedate_idx:end_sedate_idx] = True

            states = np.full(n, "Active", dtype=object)
            states[is_sedate] = "Sedate"
            states[is_anaesthesia] = "Anaesthesia"

            g["state"] = states
            g["is_drop_event"] = is_drop_event.astype(int)
            g["baseline_speed"] = round(baseline_speed, 2)
            g["sedation_confidence"] = np.round(confidence, 3)
            result_dfs.append(g)

        return pd.concat(result_dfs, ignore_index=True)

    def extract_summary(
        self,
        df_with_states: pd.DataFrame,
        anesthesia_onset_time: Optional[float] = None
    ) -> pd.DataFrame:
        """导出各小室生理动力学潜伏期统计表"""
        if df_with_states is None or df_with_states.empty:
            return pd.DataFrame()

        gas_onset = float(anesthesia_onset_time if anesthesia_onset_time is not None else self.anesthesia_onset_time)
        summaries = []

        for cid, grp in df_with_states.groupby("chamber_id"):
            g = grp.sort_values("frame").reset_index(drop=True)
            if "timestamp_s" not in g.columns:
                g["timestamp_s"] = g["timestamp"] if "timestamp" in g.columns else g["frame"] / self.fps
            time_col = "timestamp_s"
            post_df = cast(pd.DataFrame, g[g[time_col] >= gas_onset])

            sed_rows = cast(pd.DataFrame, post_df[post_df["state"] == "Sedate"])
            first_sed_time = float(sed_rows[time_col].iloc[0]) if not sed_rows.empty else None

            ana_rows = cast(pd.DataFrame, post_df[post_df["state"] == "Anaesthesia"])
            first_ana_time = float(ana_rows[time_col].iloc[0]) if not ana_rows.empty else None

            if first_sed_time is None and first_ana_time is not None:
                first_sed_time = first_ana_time
            elif first_sed_time is not None and first_ana_time is not None:
                first_sed_time = min(first_sed_time, first_ana_time)

            sed_latency = round(first_sed_time - gas_onset, 2) if first_sed_time is not None else None
            ana_latency = round(first_ana_time - gas_onset, 2) if first_ana_time is not None else None
            total_drops = int(post_df["is_drop_event"].sum()) if "is_drop_event" in post_df.columns else 0
            baseline_spd = float(g["baseline_speed"].iloc[0]) if "baseline_speed" in g.columns else 0.0

            if first_sed_time is not None and "sedation_confidence" in post_df.columns:
                sed_idx = np.searchsorted(post_df[time_col].to_numpy(), first_sed_time)
                conf_slice = post_df["sedation_confidence"].iloc[sed_idx:sed_idx + int(2.0 * self.fps)]
                confidence_score = float(np.mean(conf_slice)) if len(conf_slice) > 0 else 0.0
                confidence_score = min(1.0, max(0.0, round(confidence_score, 2)))
            else:
                confidence_score = 0.0

            summaries.append({
                "chamber_id": int(g["chamber_id"].iloc[0]),
                "anesthesia_gas_onset_sec": gas_onset,
                "sedation_onset_sec": round(first_sed_time, 2) if first_sed_time is not None else None,
                "anesthesia_onset_sec": round(first_ana_time, 2) if first_ana_time is not None else None,
                "sedation_latency_sec": sed_latency,
                "anesthesia_latency_sec": ana_latency,
                "sedation_confidence": confidence_score,
                "is_sedated": first_sed_time is not None,
                "is_anesthetized": first_ana_time is not None,
                "total_drop_events": total_drops,
                "baseline_speed": baseline_spd,
            })

        return pd.DataFrame(summaries)