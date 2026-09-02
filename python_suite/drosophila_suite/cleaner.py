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


class KinematicCleaner:
    """
    Kinematic cleaning and artifact suppression engine.
    """

    def __init__(
        self,
        fps: float = 30.0,
        max_speed_px: float = 45.0,
        body_len_thresh: float = 0.5,
        body_len_px: float = 12.0,
        occlusion_disp_thresh: float = 80.0,
        occlusion_var_thresh: float = 8.0,
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
        """
        Cleans raw tracking DataFrame across all chamber groups.
        
        Input columns expected: ['frame', 'chamber_id', 'x_px', 'y_px'] (and optionally 'timestamp_s', 'area')
        Output columns: ['frame', 'chamber_id', 'timestamp_s', 'x_clean', 'y_clean', 'norm_pos', 'speed', 'activity_bin']
        """
        if raw_df.empty:
            return pd.DataFrame(columns=[
                "frame", "chamber_id", "timestamp_s", "x_clean", "y_clean", "norm_pos", "speed", "activity_bin"
            ])

        df = raw_df.copy().sort_values(["chamber_id", "frame"]).reset_index(drop=True)
        df.columns = df.columns.str.strip()

        cleaned_dfs = []
        for ch_id, group in df.groupby("chamber_id"):
            grp = group.copy().reset_index(drop=True)

            # Ensure x_px and y_px columns exist
            if "x_px" not in grp.columns or "y_px" not in grp.columns:
                continue

            # Ensure timestamp_s column exists
            if "timestamp_s" not in grp.columns:
                if "time_sec" in grp.columns:
                    grp["timestamp_s"] = grp["time_sec"]
                else:
                    grp["timestamp_s"] = grp["frame"] / self.fps

            valid_x = grp["x_px"].dropna()
            valid_y = grp["y_px"].dropna()

            if len(valid_x) < 15:
                grp["x_clean"] = grp["x_px"]
                grp["y_clean"] = grp["y_px"]
                grp["norm_pos"] = np.nan
                grp["speed"] = 0.0
                grp["activity_bin"] = 0.0
                cleaned_dfs.append(grp)
                continue

            # 1. Dynamic Chamber Physical Boundaries (1% to 99% quantiles)
            min_x, max_x = np.percentile(valid_x, 1), np.percentile(valid_x, 99)
            min_y, max_y = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
            center_x = (min_x + max_x) / 2.0
            span_x = max_x - min_x if max_x > min_x else 1.0

            # Initial hard out-of-bounds removal (with 15px margin)
            out_of_bounds = (
                (grp["x_px"] < min_x - 15)
                | (grp["x_px"] > max_x + 15)
                | (grp["y_px"] < min_y - 15)
                | (grp["y_px"] > max_y + 15)
            )
            grp.loc[out_of_bounds, ["x_px", "y_px"]] = np.nan

            # 2. Occlusion Trap Clamping (e.g. fly hops onto cotton plug/shadow with low variance)
            win = 5
            rolling_x_std = grp["x_px"].rolling(window=win, center=True, min_periods=1).std()
            dx = grp["x_px"].diff()
            dy = grp["y_px"].diff()
            step_dist = np.sqrt(dx**2 + dy**2)

            is_static_trap = (
                (step_dist > self.occlusion_disp_thresh)
                | (grp["x_px"].diff().abs() > self.occlusion_disp_thresh)
            ) & (rolling_x_std < self.occlusion_var_thresh)

            # Vectorized replacement for trap clamping without Python for-loop
            trap_mask = is_static_trap.fillna(False)
            if trap_mask.any():
                clamped_x = np.where(grp["x_px"] < center_x, min_x, max_x)
                grp.loc[trap_mask, "x_px"] = clamped_x[trap_mask]

            # 3. Speed Spike & Jump Filtering
            dx_new = grp["x_px"].diff()
            dy_new = grp["y_px"].diff()
            dist_new = np.sqrt(dx_new**2 + dy_new**2)
            jump_mask = dist_new > self.max_speed
            grp.loc[jump_mask, ["x_px", "y_px"]] = np.nan

            # 4. Savitzky-Golay Trajectory Smoothing Filter
            x_clean = grp["x_px"].copy()
            y_clean = grp["y_px"].copy()
            valid_mask = x_clean.notna()

            if valid_mask.sum() > self.savgol_window:
                try:
                    x_clean.loc[valid_mask] = savgol_filter(
                        x_clean.loc[valid_mask],
                        window_length=self.savgol_window,
                        polyorder=self.savgol_poly
                    )
                    y_clean.loc[valid_mask] = savgol_filter(
                        y_clean.loc[valid_mask],
                        window_length=self.savgol_window,
                        polyorder=self.savgol_poly
                    )
                except Exception:
                    pass

            grp["x_clean"] = x_clean
            grp["y_clean"] = y_clean

            # 5. Normalized Position (0.0 = Ground / Bottom, 1.0 = Ceiling / Top)
            norm_pos = (x_clean - min_x) / span_x
            grp["norm_pos"] = np.clip(norm_pos, 0.0, 1.0)

            # 6. Robust Speed & Inactivity Thresholding (with Variable Frame Rate VFR dt compensation)
            delta_x = grp["x_clean"].diff()
            delta_y = grp["y_clean"].diff()
            step_disp = np.sqrt(delta_x**2 + delta_y**2)
            step_disp = np.where(step_disp < self.dist_thresh, 0.0, step_disp)

            dt = grp["timestamp_s"].diff().fillna(1.0 / self.fps)
            # Guard against zero or negative timestamps
            dt = np.where(dt <= 0.001, 1.0 / self.fps, dt)
            grp["speed"] = step_disp / dt
            grp["activity_bin"] = np.where(grp["speed"] > 0.1, 1.0, 0.0)

            cleaned_dfs.append(grp)

        final_df = pd.concat(cleaned_dfs, ignore_index=True).sort_values(["frame", "chamber_id"])
        return final_df


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
