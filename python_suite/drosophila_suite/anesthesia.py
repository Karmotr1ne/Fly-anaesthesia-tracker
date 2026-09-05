"""
Module 4: Anesthesia & Sedation Kinetics Analyzer
=================================================
Quantifies drug/anesthetic behavioral kinetics (latency to sedation/knockdown,
inactivity onset, baseline locomotion, drop events) using sliding window operators
and a 3-state machine (Active -> Sedate -> Anaesthesia).
"""

import os
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd


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
        """
        参数配置:
        :param fps: 视频采样率
        :param anesthesia_still_sec: 深度麻醉持续极值静止时长 (默认 120s)
        :param anesthesia_speed_thresh: 判定静止的绝对速度门限 (px/s)
        :param sedate_speed_ratio: 速度衰减比例 (低于基准速度的比值则判定镇静)
        :param sedate_drop_speed: 1秒单步垂直跌落高度差门限 (0.0~1.0)
        """
        self.fps = float(fps)
        self.anesthesia_still_sec = float(anesthesia_still_sec)
        self.anesthesia_speed_thresh = float(anesthesia_speed_thresh)
        self.sedate_speed_ratio = float(sedate_speed_ratio)
        self.sedate_drop_speed = float(sedate_drop_speed)

    def evaluate_states(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """
        逐通道进行三阶段动态状态机判定与跌落检测:
        Active -> Sedate (速度衰减或管顶坠落) -> Anaesthesia (全数据段滑动极值静止)
        兼容保留双 Y 轴图和 Kymograph 所需的 speed、norm_height/norm_pos 字段。
        """
        if cleaned_df is None or cleaned_df.empty:
            return pd.DataFrame()

        still_win_frames = int(round(self.anesthesia_still_sec * self.fps))
        smooth_frames = max(1, int(round(5.0 * self.fps)))
        step_1s = max(1, int(round(self.fps)))
        result_dfs = []

        for cid, grp in cleaned_df.groupby("chamber_id"):
            g = grp.copy().sort_values("frame").reset_index(drop=True)
            n = len(g)

            # 兼容读取瞬时速度与高度字段
            speeds = g["speed"].to_numpy() if "speed" in g.columns else np.zeros(n)
            if "norm_height" in g.columns:
                heights = g["norm_height"].to_numpy()
            elif "norm_pos" in g.columns:
                heights = g["norm_pos"].to_numpy()
            else:
                heights = np.zeros(n)

            # 1. 全数据集滑动窗口扫描寻找深度麻醉区间 (持续静止仍满足 < anesthesia_speed_thresh)
            is_anaesthesia = np.zeros(n, dtype=bool)
            if n >= still_win_frames and still_win_frames > 0:
                speed_series = pd.Series(speeds)
                window_max = speed_series.rolling(
                    window=still_win_frames, min_periods=still_win_frames
                ).max()
                still_ends = np.where(window_max.to_numpy() < self.anesthesia_speed_thresh)[0]
                for end_idx in still_ends:
                    start_idx = end_idx - still_win_frames + 1
                    is_anaesthesia[start_idx : end_idx + 1] = True

            # 2. 健壮的基线速度估计 (提取非麻醉活跃阶段 75% 分位数)
            active_speeds = speeds[~is_anaesthesia]
            if len(active_speeds) > int(self.fps):
                baseline_speed = max(1.0, float(np.percentile(active_speeds, 75)))
            else:
                baseline_speed = 15.0

            # 5秒移动平均速度 (平滑短期微动抖动)
            rolling_speed = (
                pd.Series(speeds)
                .rolling(smooth_frames, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            # 3. 1 秒单步垂直跌落检测 (从管腔上部 >0.55 骤降)
            is_dropping = np.zeros(n, dtype=bool)
            if n > step_1s:
                dh_1s = np.zeros(n)
                dh_1s[step_1s:] = heights[:-step_1s] - heights[step_1s:]
                was_at_top = heights[:-step_1s] > 0.55
                is_dropping[step_1s:] = (dh_1s[step_1s:] > self.sedate_drop_speed) & was_at_top

            # 4. 三阶段状态机分层判定 (允许非强制阶跃跳变与缺省)
            states = np.full(n, "Active", dtype=object)
            sedate_mask = (
                (rolling_speed < baseline_speed * self.sedate_speed_ratio) | is_dropping
            ) & (~is_anaesthesia)
            states[sedate_mask] = "Sedate"
            states[is_anaesthesia] = "Anaesthesia"

            # 5. 写入与保留兼容字段 (给下游双 Y 轴图和 Kymograph 使用)
            g["state"] = states
            g["is_drop_event"] = is_dropping.astype(int)
            g["baseline_speed"] = round(baseline_speed, 2)
            g["norm_height"] = heights
            g["norm_pos"] = heights  # 保留 norm_pos 兼容历史双 Y 轴与 Kymograph
            
            result_dfs.append(g)

        if not result_dfs:
            return pd.DataFrame()

        return pd.concat(result_dfs, ignore_index=True)

    def extract_summary(self, df_with_states: pd.DataFrame) -> pd.DataFrame:
        """
        从带状态判定的数据表中聚合提取每个通道的关键生理药效学动力学指标:
        包含镇静潜伏期、麻醉击倒潜伏期、坠落频次与基线运动能力。
        """
        if df_with_states is None or df_with_states.empty:
            return pd.DataFrame()

        summaries = []
        for cid, grp in df_with_states.groupby("chamber_id"):
            g = grp.sort_values("frame").reset_index(drop=True)

            # 深度麻醉起始时间 (首次满足 Anaesthesia 判定的时间戳)
            ana_rows = g[g["state"] == "Anaesthesia"]
            first_ana_time = ana_rows["timestamp_s"].iloc[0] if not ana_rows.empty else None

            # 镇静起始时间 (首次触发 Sedate 或直接麻醉的时间戳)
            sed_rows = g[g["state"] == "Sedate"]
            first_sed_time = sed_rows["timestamp_s"].iloc[0] if not sed_rows.empty else None

            # 如果直接从 Active 跳入 Anaesthesia，则镇静潜伏期对齐到麻醉起始点
            if first_sed_time is None and first_ana_time is not None:
                first_sed_time = first_ana_time
            elif first_sed_time is not None and first_ana_time is not None:
                first_sed_time = min(first_sed_time, first_ana_time)

            baseline_spd = g["baseline_speed"].iloc[0] if "baseline_speed" in g.columns else 0.0
            total_drops = int(g["is_drop_event"].sum()) if "is_drop_event" in g.columns else 0

            summaries.append({
                "chamber_id": int(cid),
                "sedation_onset_sec": round(first_sed_time, 2) if first_sed_time is not None else None,
                "anesthesia_onset_sec": round(first_ana_time, 2) if first_ana_time is not None else None,
                "is_sedated": first_sed_time is not None,
                "is_anesthetized": first_ana_time is not None,
                "total_drop_events": total_drops,
                "baseline_speed": float(baseline_spd),
            })

        return pd.DataFrame(summaries)
