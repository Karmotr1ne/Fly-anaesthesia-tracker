"""
Module 2: Kinematic Cleaning & Artifact Rejection
=================================================
Robust kinematic filtering algorithm for Drosophila chamber tracking:
1. Dynamic physical chamber boundaries (ROI percentile bounds 1% ~ 99%)
2. Occlusion Trap Clamping (detects jump to plug/mesh shadows and clamps to nearest physical edge)
3. Savitzky-Golay trajectory smoothing filter
4. Normalized vertical coordinate (0.0=bottom ground, 1.0=top ceiling) and speed metrics.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def fill_missing_with_midpoints(series: pd.Series) -> pd.Series:
    """
    若出现连续缺失(NaN)，取其前一个有效值与后一个有效值的中点进行恒定填充。
    若开头缺失，由首个有效值回填；若末尾缺失，由末尾有效值前推。
    """
    s = series.copy().reset_index(drop=True)
    is_na = s.isna().to_numpy()
    n = len(s)
    if not np.any(is_na):
        return s
    if np.all(is_na):
        return s.fillna(0.0)

    in_gap = False
    start_idx = 0

    for i in range(n):
        if is_na[i] and not in_gap:
            in_gap = True
            start_idx = i
        elif not is_na[i] and in_gap:
            in_gap = False
            end_idx = i - 1
            prev_val = s[start_idx - 1] if start_idx > 0 else s[i]
            next_val = s[i]
            midpoint = (prev_val + next_val) / 2.0
            s.iloc[start_idx : end_idx + 1] = midpoint

    if in_gap:
        prev_val = s[start_idx - 1] if start_idx > 0 else 0.0
        s.iloc[start_idx:] = prev_val

    return s


class KinematicCleaner:
    def __init__(
        self,
        fps: float = 30.0,
        max_speed_px: float = 45.0,
        body_len_thresh: float = 0.5,
        body_len_px: float = 12.0,
        occlusion_disp_thresh: float = 60.0,
        occlusion_var_thresh: float = 5.0,
        savgol_window: int = 7,
        savgol_poly: int = 2,
    ):
        self.fps = fps
        self.max_speed = max_speed_px
        self.dist_thresh = body_len_thresh * body_len_px
        self.occlusion_disp_thresh = occlusion_disp_thresh
        self.occlusion_var_thresh = occlusion_var_thresh
        self.savgol_window = savgol_window if savgol_window % 2 != 0 else savgol_window + 1
        self.savgol_poly = savgol_poly

    def clean_trajectory(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return pd.DataFrame()

        df = raw_df.copy().sort_values(["chamber_id", "frame"]).reset_index(drop=True)
        cleaned_dfs = []

        for ch_id, group in df.groupby("chamber_id"):
            grp = group.copy().sort_values("frame").reset_index(drop=True)

            if "timestamp_s" not in grp.columns:
                grp["timestamp_s"] = grp["frame"] / self.fps

            valid_x = grp["x_px"].dropna()
            valid_y = grp["y_px"].dropna()
            if len(valid_x) < 15:
                continue

            # 1. 物理管腔 1% ~ 99% 分位数基线
            min_x, max_x = np.percentile(valid_x, [1, 99])
            min_y, max_y = np.percentile(valid_y, [1, 99])
            span_x = max(1.0, max_x - min_x)

            # 2. 消除塞子边缘跳跃陷阱 (Occlusion Trap)
            dx = grp["x_px"].diff().abs()
            roll_std = grp["x_px"].rolling(5, center=True, min_periods=1).std()
            is_trap = (dx > self.occlusion_disp_thresh) & (roll_std < self.occlusion_var_thresh)
            grp.loc[is_trap, "x_px"] = np.where(
                grp.loc[is_trap, "x_px"] < (min_x + max_x) / 2, min_x, max_x
            )

            # 3. 速度突变跳点过滤（置为 NaN 后由前后中点填充修复）
            step_dist = np.hypot(grp["x_px"].diff().fillna(0), grp["y_px"].diff().fillna(0))
            grp.loc[step_dist > self.max_speed, ["x_px", "y_px"]] = np.nan

            # 4. 缺失数据锚定中点填充
            grp["x_px"] = fill_missing_with_midpoints(grp["x_px"])
            grp["y_px"] = fill_missing_with_midpoints(grp["y_px"])

            # 5. Savitzky-Golay 平滑
            if len(grp) > self.savgol_window:
                grp["x_clean"] = savgol_filter(grp["x_px"], self.savgol_window, self.savgol_poly)
                grp["y_clean"] = savgol_filter(grp["y_px"], self.savgol_window, self.savgol_poly)
            else:
                grp["x_clean"], grp["y_clean"] = grp["x_px"], grp["y_px"]

            # 6. 归一化高度 (0.0=底部, 1.0=顶部)
            grp["norm_height"] = np.clip((grp["x_clean"] - min_x) / span_x, 0.0, 1.0)
            grp["norm_pos"] = grp["norm_height"]  # 兼容字段

            # 7. 微位移死区过滤与瞬时速度
            dt = grp["timestamp_s"].diff().fillna(1.0 / self.fps).clip(lower=0.001)
            disp = np.hypot(grp["x_clean"].diff().fillna(0), grp["y_clean"].diff().fillna(0))
            disp = np.where(disp < self.dist_thresh, 0.0, disp)
            grp["speed"] = disp / dt
            grp["activity_bin"] = np.where(grp["speed"] > 0.1, 1.0, 0.0)

            cleaned_dfs.append(grp)

        return pd.concat(cleaned_dfs, ignore_index=True).sort_values(["frame", "chamber_id"])


# Alias for backward compatibility
class DrosophilaProcessor:
    @staticmethod
    def clean_data(
        df: pd.DataFrame,
        fps: float = 30.0,
        max_speed_px: float = 45.0,
        occlusion_disp_thresh: float = 80.0,
        occlusion_var_thresh: float = 8.0,
    ) -> pd.DataFrame:
        cleaner = KinematicCleaner(
            fps=fps,
            max_speed_px=max_speed_px,
            occlusion_disp_thresh=occlusion_disp_thresh,
            occlusion_var_thresh=occlusion_var_thresh
        )
        return cleaner.clean_trajectory(df)
