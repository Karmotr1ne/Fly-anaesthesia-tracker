"""
Module 4: Anesthesia & Sedation Kinetics Analyzer
=================================================
Quantifies drug/anesthetic behavioral kinetics (latency to sedation/knockdown,
inactivity onset, baseline locomotion) using sliding window max filter operators.
"""

import os
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from .stationary_engine import StationaryDetectionEngine
from .models import PipelineConfig, AnesthesiaSummary


class AnesthesiaAnalyzer:
    def __init__(
        self,
        fps: float = 30.0,
        anesthesia_still_sec: float = 120.0,
        anesthesia_speed_thresh: float = 0.1,
        sedate_speed_ratio: float = 0.35,
        sedate_drop_speed: float = 0.25,
        **kwargs
    ):
        self.fps = fps
        self.anesthesia_still_sec = anesthesia_still_sec
        self.anesthesia_speed_thresh = anesthesia_speed_thresh
        self.sedate_speed_ratio = sedate_speed_ratio
        self.sedate_drop_speed = sedate_drop_speed

    def evaluate_states(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """
        三阶段动态状态机逐帧判定：
        Active -> Sedate (速度衰减或坠落) -> Anaesthesia (全数据集扫描 120s 极值静止)
        """
        if cleaned_df.empty:
            return pd.DataFrame()

        still_win_frames = int(round(self.anesthesia_still_sec * self.fps))
        smooth_frames = max(1, int(round(5.0 * self.fps)))
        result_dfs = []

        for cid, grp in cleaned_df.groupby("chamber_id"):
            g = grp.copy().sort_values("frame").reset_index(drop=True)
            n = len(g)
            speeds = g["speed"].to_numpy()
            heights = g["norm_height"].to_numpy() if "norm_height" in g.columns else g["norm_pos"].to_numpy()

            # 1. 全数据集滑动窗口扫描寻找深度麻醉区间
            is_anaesthesia = np.zeros(n, dtype=bool)
            if n >= still_win_frames:
                speed_series = pd.Series(speeds)
                window_max = speed_series.rolling(window=still_win_frames, min_periods=still_win_frames).max()
                still_ends = np.where(window_max.to_numpy() < self.anesthesia_speed_thresh)[0]
                for end_idx in still_ends:
                    start_idx = end_idx - still_win_frames + 1
                    is_anaesthesia[start_idx : end_idx + 1] = True

            # 2. 基线速度计算
            active_speeds = speeds[~is_anaesthesia]
            if len(active_speeds) > 30:
                baseline_speed = max(1.0, float(np.percentile(active_speeds, 75)))
            else:
                baseline_speed = 15.0

            rolling_speed = pd.Series(speeds).rolling(smooth_frames, center=True, min_periods=1).mean().to_numpy()

            # 3. 跌落事件检测（从上部快速下坠）
            step_1s = int(self.fps)
            is_dropping = np.zeros(n, dtype=bool)
            if n > step_1s:
                dh_1s = np.zeros(n)
                dh_1s[step_1s:] = heights[:-step_1s] - heights[step_1s:]
                was_at_top = heights[:-step_1s] > 0.55
                is_dropping[step_1s:] = (dh_1s[step_1s:] > self.sedate_drop_speed) & was_at_top

            # 4. 状态机分层判定
            states = np.full(n, "Active", dtype=object)
            sedate_mask = ((rolling_speed < baseline_speed * self.sedate_speed_ratio) | is_dropping) & (~is_anaesthesia)
            states[sedate_mask] = "Sedate"
            states[is_anaesthesia] = "Anaesthesia"

            g["state"] = states
            g["is_drop_event"] = is_dropping.astype(int)
            g["baseline_speed"] = round(baseline_speed, 2)
            result_dfs.append(g)

        return pd.concat(result_dfs, ignore_index=True)

    def extract_summary(self, df_with_states: pd.DataFrame) -> pd.DataFrame:
        """从状态数据表中聚合提炼每个通道的麻醉与镇静统计指标"""
        summaries = []
        for cid, grp in df_with_states.groupby("chamber_id"):
            g = grp.sort_values("frame").reset_index(drop=True)
            ana_rows = g[g["state"] == "Anaesthesia"]
            first_ana_time = ana_rows["timestamp_s"].iloc[0] if not ana_rows.empty else None
            
            sed_rows = g[g["state"] == "Sedate"]
            first_sed_time = sed_rows["timestamp_s"].iloc[0] if not sed_rows.empty else None

            summaries.append({
                "chamber_id": int(cid),
                "sedation_onset_sec": round(first_sed_time, 2) if first_sed_time is not None else None,
                "anesthesia_onset_sec": round(first_ana_time, 2) if first_ana_time is not None else None,
                "is_sedated": first_sed_time is not None or first_ana_time is not None,
                "is_anesthetized": first_ana_time is not None,
                "total_drop_events": int(g["is_drop_event"].sum()),
                "baseline_speed": g["baseline_speed"].iloc[0] if "baseline_speed" in g.columns else 0.0,
            })
        return pd.DataFrame(summaries)
