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
        drop_dt_sec: float = 0.35,
        drop_height_delta: float = 0.30,
        drop_top_thresh: float = 0.45,
        drop_bottom_thresh: float = 0.20,
        drop_sustain_sec: float = 4.0,
        bottom_quiescent_sec: float = 10.0,
        anesthesia_onset_time: float = 0.0,
        **kwargs
    ):
        """
        :param fps: 视频采样率
        :param anesthesia_still_sec: 深度麻醉持续静止时长 (秒，默认 120s)
        :param anesthesia_speed_thresh: 判定静止的绝对速度门限 (px/s，抗抖动建议 0.5~1.0)
        :param drop_dt_sec: 物理急剧跌落的时间窗口 (秒，默认 0.35s，避免与自主下爬混淆)
        :param drop_height_delta: 跌落过程的垂直高度差门限 (归一化高度 0.0~1.0)
        :param drop_top_thresh: 跌落起始的中高位门限 (高于此高度视为攀爬状态)
        :param drop_bottom_thresh: 底部高度门限 (落入此高度以下视为触底)
        :param drop_sustain_sec: 摔落后低位滞留观察窗口 (秒，此时间内不得重新爬起)
        :param bottom_quiescent_sec: 针对未爬高个体的底部持续低活动兜底时长 (秒)
        :param anesthesia_onset_time: 麻醉气体释放起始时间戳 (秒)
        """
        self.fps = float(fps)
        self.anesthesia_still_sec = float(anesthesia_still_sec)
        self.anesthesia_speed_thresh = float(anesthesia_speed_thresh)
        self.drop_dt_sec = float(drop_dt_sec)
        self.drop_height_delta = float(drop_height_delta)
        self.drop_top_thresh = float(drop_top_thresh)
        self.drop_bottom_thresh = float(drop_bottom_thresh)
        self.drop_sustain_sec = float(drop_sustain_sec)
        self.bottom_quiescent_sec = float(bottom_quiescent_sec)
        self.anesthesia_onset_time = float(anesthesia_onset_time)

    def evaluate_states(
        self,
        cleaned_df: pd.DataFrame,
        anesthesia_onset_time: Optional[float] = None
    ) -> pd.DataFrame:
        """
        以物理跌落（Knockdown）为主导的动态状态机判定:
        1. 给药前数据作为基线期，严禁被判定为镇静或麻醉。
        2. 麻醉时间窗仅从 gas_onset 开始往后扫描，严禁透支给药前的静止时长。
        3. 镇静首选由“失控坠落箱底并不再复爬”触发；底栖未攀爬个体由“底部持续静息”兜底触发。
        """
        if cleaned_df is None or cleaned_df.empty:
            return pd.DataFrame()

        gas_onset = float(anesthesia_onset_time if anesthesia_onset_time is not None else self.anesthesia_onset_time)

        still_win_frames = max(1, int(round(self.anesthesia_still_sec * self.fps)))
        drop_dt_frames = max(1, int(round(self.drop_dt_sec * self.fps)))
        drop_sustain_frames = max(1, int(round(self.drop_sustain_sec * self.fps)))
        bottom_quiescent_frames = max(1, int(round(self.bottom_quiescent_sec * self.fps)))
        smooth_frames = max(1, int(round(2.0 * self.fps)))

        result_dfs = []

        for cid, grp in cleaned_df.groupby("chamber_id"):
            g = grp.copy().sort_values("frame").reset_index(drop=True)
            n = len(g)

            # 1. 提取时间戳、速度与归一化垂直高度
            if "timestamp_s" in g.columns:
                timestamps = g["timestamp_s"].to_numpy()
            elif "timestamp" in g.columns:
                timestamps = g["timestamp"].to_numpy()
            else:
                timestamps = g["frame"].to_numpy() / self.fps
                g["timestamp_s"] = timestamps

            speeds = g["speed"].to_numpy() if "speed" in g.columns else np.zeros(n)
            
            if "norm_height" in g.columns:
                heights = g["norm_height"].to_numpy()
            elif "norm_pos" in g.columns:
                heights = g["norm_pos"].to_numpy()
            else:
                heights = np.zeros(n)

            # 2. 划分给药前（基线期）与给药后阶段
            post_gas_mask = timestamps >= gas_onset
            pre_gas_mask = ~post_gas_mask

            # 3. 基线活跃度校准（仅取给药前移动段，剔除停歇影响）
            pre_gas_speeds = speeds[pre_gas_mask]
            if len(pre_gas_speeds) >= int(self.fps):
                moving_speeds = pre_gas_speeds[pre_gas_speeds > self.anesthesia_speed_thresh]
                if len(moving_speeds) >= int(self.fps):
                    baseline_speed = float(np.percentile(moving_speeds, 75))
                else:
                    baseline_speed = float(np.percentile(pre_gas_speeds, 85))
                baseline_speed = max(baseline_speed, 2.0)
            else:
                baseline_speed = 15.0

            # 4. 深度麻醉判定 (Anaesthesia)
            # 严格限定在 post_gas_mask 内滑动，完全禁止借用给药前静止
            is_anaesthesia = np.zeros(n, dtype=bool)
            post_indices = np.where(post_gas_mask)[0]

            if len(post_indices) >= still_win_frames:
                post_speeds = speeds[post_indices]
                post_still = post_speeds < self.anesthesia_speed_thresh
                post_rolling_still = (
                    pd.Series(post_still)
                    .rolling(window=still_win_frames, min_periods=still_win_frames)
                    .sum() == still_win_frames
                ).to_numpy()

                still_ends = np.where(post_rolling_still)[0]
                if len(still_ends) > 0:
                    # 获取首个达到深度麻醉的时刻，该时刻及后续保持麻醉锁定
                    first_still_end = still_ends[0]
                    first_ana_idx = post_indices[first_still_end - still_win_frames + 1]
                    is_anaesthesia[first_ana_idx:] = True

            # 5. 镇静判定 (Sedate / Knockdown)
            # 优先依据急剧跌落 + 底部滞留，兜底依据底部静息
            is_sedate = np.zeros(n, dtype=bool)
            is_drop_event = np.zeros(n, dtype=bool)
            first_sedate_idx = None

            # 路径 A: 扫描失控跌落事件（Knockdown）
            for i in range(drop_dt_frames, n):
                if not post_gas_mask[i]:
                    continue

                h_start = heights[i - drop_dt_frames]
                h_end = heights[i]
                dh = h_start - h_end

                # 急剧落体特征：短时间内从中高处坠至底部
                if (h_start > self.drop_top_thresh) and (h_end < self.drop_bottom_thresh) and (dh > self.drop_height_delta):
                    is_drop_event[i] = True

                    # 验证摔落后是否无法恢复攀爬
                    if first_sedate_idx is None:
                        eval_end = min(n, i + drop_sustain_frames)
                        post_drop_heights = heights[i:eval_end]
                        # 滞留于中低位 (不超过中位高度)，即确立为神经麻痹导致的镇静
                        if len(post_drop_heights) > 0 and np.all(post_drop_heights < (self.drop_top_thresh * 0.8)):
                            first_sedate_idx = i

            # 路径 B: 兜底逻辑（给药后未发生高位跌落、但持续停留在箱底低活动）
            if first_sedate_idx is None and len(post_indices) >= bottom_quiescent_frames:
                rolling_speed = (
                    pd.Series(speeds)
                    .rolling(smooth_frames, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                quiescent_at_bottom = (
                    post_gas_mask
                    & (heights < self.drop_bottom_thresh)
                    & (rolling_speed < max(1.0, baseline_speed * 0.25))
                )
                rolling_quiescent = (
                    pd.Series(quiescent_at_bottom[post_indices])
                    .rolling(window=bottom_quiescent_frames, min_periods=bottom_quiescent_frames)
                    .sum() == bottom_quiescent_frames
                ).to_numpy()

                q_ends = np.where(rolling_quiescent)[0]
                if len(q_ends) > 0:
                    first_sedate_idx = post_indices[q_ends[0] - bottom_quiescent_frames + 1]

            # 标记镇静区间
            if first_sedate_idx is not None:
                is_sedate[first_sedate_idx:] = True

            # 6. 状态分配与合并
            states = np.full(n, "Active", dtype=object)
            states[is_sedate & (~is_anaesthesia)] = "Sedate"
            states[is_anaesthesia] = "Anaesthesia"

            g["state"] = states
            g["is_drop_event"] = is_drop_event.astype(int)
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
        聚合输出通道统计指标与潜伏期
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

            # 潜伏期层级保护：若因特殊情况先进入麻醉，镇静时间不得晚于麻醉时间
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