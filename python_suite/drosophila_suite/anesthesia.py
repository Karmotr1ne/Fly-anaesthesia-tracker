import os
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd


class AnesthesiaAnalyzer:
    def __init__(
        self,
        fps: float = 30.0,
        anesthesia_still_sec: float = 120.0,
        anesthesia_speed_thresh: float = 0.5,
        sedate_speed_ratio: float = 0.35,
        sedate_min_sec: float = 3.0,
        sedate_drop_speed: float = 0.25,
        anesthesia_onset_time: float = 0.0,
        **kwargs
    ):
        """
        :param fps: 视频采样率
        :param anesthesia_still_sec: 给药后需持续静止的时长门限 (默认 120s)
        :param anesthesia_speed_thresh: 判定静止的绝对速度门限 (px/s，建议设为 0.5~1.0 以抵消 Tracking 噪点)
        :param sedate_speed_ratio: 速度衰减比例 (低于基线运动速度此比例则进入候选镇静)
        :param sedate_min_sec: 镇静所需持续的最短时间 (秒，防止单次生理短暂停歇误触发)
        :param sedate_drop_speed: 1秒单步垂直跌落高度差门限 (0.0~1.0)
        :param anesthesia_onset_time: 麻醉气体释放起始时间戳 (秒，默认 0.0)
        """
        self.fps = float(fps)
        self.anesthesia_still_sec = float(anesthesia_still_sec)
        self.anesthesia_speed_thresh = float(anesthesia_speed_thresh)
        self.sedate_speed_ratio = float(sedate_speed_ratio)
        self.sedate_min_sec = float(sedate_min_sec)
        self.sedate_drop_speed = float(sedate_drop_speed)
        self.anesthesia_onset_time = float(anesthesia_onset_time)

    def evaluate_states(
        self,
        cleaned_df: pd.DataFrame,
        anesthesia_onset_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        逐通道进行三阶段状态判定:
        - 给药前严禁标记为 Sedate 或 Anaesthesia，仅用于评估基线运动与停顿特征。
        - 麻醉静止窗口严密限定在给药后独立计算，杜绝回溯透支。
        """
        if cleaned_df is None or cleaned_df.empty:
            return pd.DataFrame()

        gas_onset = float(anesthesia_onset_time if anesthesia_onset_time is not None else self.anesthesia_onset_time)
        still_win_frames = max(1, int(round(self.anesthesia_still_sec * self.fps)))
        sedate_win_frames = max(1, int(round(self.sedate_min_sec * self.fps)))
        smooth_frames = max(1, int(round(3.0 * self.fps)))
        step_1s = max(1, int(round(self.fps)))
        result_dfs = []

        for cid, grp in cleaned_df.groupby("chamber_id"):
            g = grp.copy().sort_values("frame").reset_index(drop=True)
            n = len(g)

            # 1. 提取时间、速度与高度
            if "timestamp_s" in g.columns:
                timestamps = g["timestamp_s"].to_numpy()
            elif "timestamp" in g.columns:
                timestamps = g["timestamp"].to_numpy()
            else:
                timestamps = g["frame"].to_numpy() / self.fps
                g["timestamp_s"] = timestamps

            speeds = g["speed"].to_numpy() if "speed" in g.columns else np.zeros(n)
            heights = g["norm_height"].to_numpy() if "norm_height" in g.columns else np.zeros(n)

            # 2. 划分给药前/后时间段
            post_gas_mask = timestamps >= gas_onset
            pre_gas_mask = ~post_gas_mask

            # 3. 基线行为校准（仅利用给药前数据）
            pre_gas_speeds = speeds[pre_gas_mask]
            if len(pre_gas_speeds) >= int(self.fps):
                # 采用给药前运动段的 85 分位数作为运动活性基准，排除基线静止期的拉低效应
                moving_speeds = pre_gas_speeds[pre_gas_speeds > self.anesthesia_speed_thresh]
                if len(moving_speeds) >= int(self.fps):
                    baseline_speed = float(np.percentile(moving_speeds, 75))
                else:
                    baseline_speed = float(np.percentile(pre_gas_speeds, 85))
                baseline_speed = max(baseline_speed, 2.0)
            else:
                baseline_speed = 15.0

            # 4. 跌落事件检测（仅在给药后有效）
            is_dropping = np.zeros(n, dtype=bool)
            if n > step_1s:
                dh_1s = np.zeros(n)
                dh_1s[step_1s:] = heights[:-step_1s] - heights[step_1s:]
                was_at_top = heights[:-step_1s] > 0.55
                drop_cond = (dh_1s[step_1s:] > self.sedate_drop_speed) & was_at_top
                is_dropping[step_1s:] = drop_cond & post_gas_mask[step_1s:]

            # 5. 深度麻醉判定（严格仅在 post_gas 序列中滑动，禁止回溯借用给药前静止）
            is_anaesthesia = np.zeros(n, dtype=bool)
            post_indices = np.where(post_gas_mask)[0]
            if len(post_indices) >= still_win_frames:
                post_speeds = speeds[post_indices]
                # 仅对给药后的速度进行静止窗检测
                post_is_still = post_speeds < self.anesthesia_speed_thresh
                post_rolling_still = (
                    pd.Series(post_is_still)
                    .rolling(window=still_win_frames, min_periods=still_win_frames)
                    .sum()
                    == still_win_frames
                ).to_numpy()

                still_ends = np.where(post_rolling_still)[0]
                for end_idx in still_ends:
                    start_idx = end_idx - still_win_frames + 1
                    # 映射回全局索引
                    is_anaesthesia[post_indices[start_idx] : post_indices[end_idx] + 1] = True

            # 一旦进入深度麻醉，其后所有帧均锁定为麻醉状态
            ana_indices = np.where(is_anaesthesia)[0]
            if len(ana_indices) > 0:
                first_ana_idx = ana_indices[0]
                is_anaesthesia[first_ana_idx:] = True

            # 6. 速度平滑与镇静（Sedate）判定
            rolling_speed = (
                pd.Series(speeds)
                .rolling(smooth_frames, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )

            # 触发镇静候选条件：速度衰减或跌落
            raw_sedate_cond = post_gas_mask & (
                (rolling_speed < baseline_speed * self.sedate_speed_ratio) | is_dropping
            )

            # 增加时延滤波（Debounce）：连续满足持续时长才认定为真正的镇静启动
            is_sedate = np.zeros(n, dtype=bool)
            if len(post_indices) >= sedate_win_frames:
                post_raw_sedate = raw_sedate_cond[post_indices]
                post_rolling_sedate = (
                    pd.Series(post_raw_sedate)
                    .rolling(window=sedate_win_frames, min_periods=sedate_win_frames)
                    .sum()
                    == sedate_win_frames
                ).to_numpy()

                sed_ends = np.where(post_rolling_sedate)[0]
                for end_idx in sed_ends:
                    start_idx = end_idx - sedate_win_frames + 1
                    is_sedate[post_indices[start_idx] : post_indices[end_idx] + 1] = True

            # 7. 综合三阶段状态机赋值
            states = np.full(n, "Active", dtype=object)
            states[is_sedate & (~is_anaesthesia)] = "Sedate"
            states[is_anaesthesia] = "Anaesthesia"

            g["state"] = states
            g["is_drop_event"] = is_dropping.astype(int)
            g["baseline_speed"] = round(baseline_speed, 2)
            g["norm_height"] = heights
            g["norm_pos"] = heights

            result_dfs.append(g)

        if not result_dfs:
            return pd.DataFrame()

        return pd.concat(result_dfs, ignore_index=True)

    def extract_summary(
        self,
        df_with_states: pd.DataFrame,
        anesthesia_onset_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        聚合提取各通道统计特征与潜伏期
        """
        if df_with_states is None or df_with_states.empty:
            return pd.DataFrame()

        gas_onset = float(anesthesia_onset_time if anesthesia_onset_time is not None else self.anesthesia_onset_time)
        summaries = []

        for cid, grp in df_with_states.groupby("chamber_id"):
            g = grp.sort_values("frame").reset_index(drop=True)
            time_col = "timestamp_s" if "timestamp_s" in g.columns else "timestamp"
            if time_col not in g.columns:
                g[time_col] = g["frame"] / self.fps

            post_gas_df = g[g[time_col] >= gas_onset]

            ana_rows = post_gas_df[post_gas_df["state"] == "Anaesthesia"]
            first_ana_time = ana_rows[time_col].iloc[0] if not ana_rows.empty else None

            sed_rows = post_gas_df[post_gas_df["state"] == "Sedate"]
            first_sed_time = sed_rows[time_col].iloc[0] if not sed_rows.empty else None

            # 保持层级一致：若先触发麻醉，则镇静潜伏期不晚于麻醉
            if first_sed_time is None and first_ana_time is not None:
                first_sed_time = first_ana_time
            elif first_sed_time is not None and first_ana_time is not None:
                first_sed_time = min(first_sed_time, first_ana_time)

            baseline_spd = g["baseline_speed"].iloc[0] if "baseline_speed" in g.columns else 0.0
            total_drops = int(post_gas_df["is_drop_event"].sum()) if "is_drop_event" in post_gas_df.columns else 0

            sed_latency = round(first_sed_time - gas_onset, 2) if first_sed_time is not None else None
            ana_latency = round(first_ana_time - gas_onset, 2) if first_ana_time is not None else None

            summaries.append({
                "chamber_id": int(cid),
                "anesthesia_gas_onset_sec": gas_onset,
                "sedation_onset_sec": round(first_sed_time, 2) if first_sed_time is not None else None,
                "anesthesia_onset_sec": round(first_ana_time, 2) if first_ana_time is not None else None,
                "sedation_latency_sec": sed_latency,
                "anesthesia_latency_sec": ana_latency,
                "is_sedated": first_sed_time is not None,
                "is_anesthetized": first_ana_time is not None,
                "total_drop_events": total_drops,
                "baseline_speed": float(baseline_spd),
            })

        return pd.DataFrame(summaries)