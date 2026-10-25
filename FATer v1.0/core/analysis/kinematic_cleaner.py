"""V0.2 cleaning with time-weighted short gaps and adaptive SG windows."""
from core import schema
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from .trajectory import interpolate_gaps, fill_midpoints_array


class KinematicCleaner:
    def __init__(self, fps=30.0, max_step=45.0, body_ratio=0.5,
                 body_length=12.0, occlusion_disp_thresh=60.0,
                 occlusion_var_thresh=5.0, savgol_window=7, savgol_poly=2):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and positive.")
        if savgol_poly < 0 or savgol_window <= savgol_poly:
            raise ValueError("SG window must exceed its non-negative polynomial order.")
        self.fps = float(fps)
        self.max_step = float(max_step)
        self.dist_thresh = body_ratio * body_length
        self.occlusion_disp_thresh = occlusion_disp_thresh
        self.occlusion_var_thresh = occlusion_var_thresh
        self.savgol_window = int(savgol_window) | 1
        self.savgol_poly = int(savgol_poly)

    def clean_trajectory(self, raw_df, interpolate=True, max_gap=30,
                         gap_speed=400.0, cancel_callback=None, *, _canonical=False,
                         _min_valid=15):
        df = raw_df if _canonical else schema.normalise(raw_df)
        if max_gap < 0 or not np.isfinite(gap_speed) or gap_speed <= 0:
            raise ValueError("Invalid gap interpolation settings.")
        if df.empty:
            result = df.copy()
            for name in ("x_clean", "y_clean", "height", "position", "speed", "activity"):
                result[name] = pd.Series(dtype=float)
            return result
        df = df.sort_values(["chamber_id", "frame"], kind="stable").reset_index(drop=True)
        if "time" not in df:
            df["time"] = df["frame"] / self.fps
        if "is_interpolated" not in df:
            df["is_interpolated"] = 0
        if "is_detected" not in df:
            df["is_detected"] = (df["x"].notna() & df["y"].notna() & df["is_interpolated"].eq(0)).astype(int)
        if 'tracking_status' in df and 'segment_id' in df:
            return self._clean_segments(df, cancel_callback)
        results = []
        excluded = []
        for chamber, group in df.groupby("chamber_id"):
            if cancel_callback and cancel_callback():
                raise InterruptedError("Trajectory cleaning cancelled.")
            group = group.reset_index(drop=True)
            x = group["x"].to_numpy(dtype=float, copy=True)
            y = group["y"].to_numpy(dtype=float, copy=True)
            if interpolate and max_gap:
                flags = group["is_interpolated"].to_numpy(copy=True)
                ratios = [group[name].to_numpy(dtype=float, copy=True) if name in group
                          else np.full(len(group), np.nan) for name in ("x_ratio", "y_ratio")]
                interpolate_gaps(x, y, group["time"].to_numpy(dtype=float),
                                 max_gap, gap_speed, flags, ratios, cancel_callback)
                group["is_interpolated"] = flags
                group["x_ratio"], group["y_ratio"] = ratios
            group["x"], group["y"] = x, y
            valid_x, valid_y = group["x"].dropna(), group["y"].dropna()
            if len(valid_x) < _min_valid or len(valid_y) < _min_valid:
                excluded.append(int(chamber))
                continue
            min_x, max_x = np.percentile(valid_x, [1, 99])
            span = max(1.0, max_x - min_x)
            deviation = group["x"].rolling(5, center=True, min_periods=1).std()
            trapped = (group["x"].diff().abs() > self.occlusion_disp_thresh) & (deviation < self.occlusion_var_thresh)
            group.loc[trapped, "x"] = np.where(
                group.loc[trapped, "x"] < (min_x + max_x) / 2, min_x, max_x)
            displacement = np.hypot(group["x"].diff().fillna(0), group["y"].diff().fillna(0))
            rejected = displacement > self.max_step
            group.loc[rejected, ["x", "y"]] = np.nan
            missing = group["x"].isna() | group["y"].isna()
            group["x"] = fill_midpoints_array(group["x"].to_numpy(dtype=float, copy=True))
            group["y"] = fill_midpoints_array(group["y"].to_numpy(dtype=float, copy=True))
            window = min(self.savgol_window, len(group) if len(group) % 2 else len(group) - 1)
            for axis in ("x", "y"):
                group[axis + "_clean"] = (savgol_filter(group[axis], window, self.savgol_poly)
                    if window > self.savgol_poly else group[axis])
            group["height"] = np.clip((group["x_clean"] - min_x) / span, 0, 1)
            group["position"] = group["height"]
            dt = group["time"].diff().fillna(1 / self.fps).clip(lower=.001)
            displacement = np.hypot(group["x_clean"].diff().fillna(0), group["y_clean"].diff().fillna(0))
            group["speed"] = np.where(displacement < self.dist_thresh, 0., displacement) / dt
            group["activity"] = np.where(group["speed"] > .1, 1., 0.)
            group["is_rejected"] = rejected.astype(int)
            group["is_filled"] = missing.astype(int)
            group["coordinate_status"] = np.where(missing, "filled", np.where(group["is_interpolated"].eq(1), "interpolated", "detected"))
            results.append(group)
        if not results:
            raise ValueError("No chamber has at least 15 valid x/y tracking samples.")
        result = pd.concat(results, ignore_index=True).sort_values(["frame", "chamber_id"]).reset_index(drop=True)
        result.attrs["cleaning_excluded_chamber_ids"] = excluded
        return result

    def _clean_segments(self, df, cancel_callback):
        """Keep rejected observations missing; never smooth or derive speed across them."""
        results, excluded = [], []
        derived = ('x_clean', 'y_clean', 'height', 'position', 'speed', 'activity')
        for chamber, group in df.groupby('chamber_id', sort=False):
            if cancel_callback and cancel_callback():
                raise InterruptedError('Trajectory cleaning cancelled.')
            group = group.reset_index(drop=True).copy()
            accepted = (group.tracking_status.eq('Accepted') &
                        np.isfinite(group.x) & np.isfinite(group.y))
            if accepted.sum() < 15:
                excluded.append(int(chamber))
                continue
            count = len(group)
            outputs = {name: np.full(count, np.nan) for name in derived}
            outputs['is_rejected'] = (~accepted).to_numpy(dtype=int)
            outputs['is_filled'] = np.zeros(count, dtype=int)
            outputs['coordinate_status'] = np.where(accepted, 'detected', 'rejected').astype(object)
            valid = accepted.to_numpy()
            x = group.x.to_numpy(dtype=float)
            y = group.y.to_numpy(dtype=float)
            times = group.time.to_numpy(dtype=float)
            segments = group.segment_id.to_numpy()
            frames = group.frame.to_numpy()
            boundaries = np.r_[True, (valid[1:] != valid[:-1]) |
                (segments[1:] != segments[:-1]) | (np.diff(frames) != 1), True]
            edges = np.flatnonzero(boundaries)
            flags = group.is_interpolated.to_numpy()
            for start, end in zip(edges[:-1], edges[1:]):
                if cancel_callback and cancel_callback():
                    raise InterruptedError('Trajectory cleaning cancelled.')
                if not valid[start]:
                    continue
                cleaned = self._clean_run_arrays(x[start:end], y[start:end],
                    times[start:end], flags[start:end])
                for name, values in cleaned.items():
                    outputs[name][start:end] = values
            low, high = np.percentile(x[valid], [1, 99])
            outputs['height'] = np.clip((outputs['x_clean'] - low) / max(1., high - low), 0, 1)
            outputs['position'] = outputs['height']
            group.loc[~accepted, ['x', 'y', 'is_detected']] = [np.nan, np.nan, 0]
            # One assignment per output column, rather than per segment/column.
            for name, values in outputs.items():
                group[name] = values
            results.append(group)
        if not results:
            result = df.iloc[:0].copy()
            for name in derived:
                result[name] = pd.Series(dtype=float)
            result.attrs['cleaning_excluded_chamber_ids'] = excluded
            return result
        result = pd.concat(results, ignore_index=True).sort_values(['frame', 'chamber_id']).reset_index(drop=True)
        result.attrs['cleaning_excluded_chamber_ids'] = excluded
        return result

    def _clean_run_arrays(self, raw_x, raw_y, times, flags):
        """Clean a contiguous accepted slice without constructing DataFrames."""
        x, y = raw_x.copy(), raw_y.copy()
        low, high = np.percentile(x, [1, 99])
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(x, (2, 2),
            constant_values=np.nan), 5)
        counts = np.isfinite(windows).sum(axis=1)
        means = np.nansum(windows, axis=1) / counts
        variance = np.nansum((windows - means[:, None]) ** 2, axis=1)
        deviation = np.sqrt(np.divide(variance, counts - 1,
            out=np.full(len(x), np.nan), where=counts > 1))
        differences = np.r_[np.nan, np.diff(x)]
        trapped = (np.abs(differences) > self.occlusion_disp_thresh) & (deviation < self.occlusion_var_thresh)
        x[trapped] = np.where(x[trapped] < (low + high) / 2, low, high)
        displacement = np.hypot(np.r_[0., np.diff(x)], np.r_[0., np.diff(y)])
        rejected = displacement > self.max_step
        x[rejected], y[rejected] = np.nan, np.nan
        missing = ~np.isfinite(x) | ~np.isfinite(y)
        x, y = fill_midpoints_array(x), fill_midpoints_array(y)
        window = min(self.savgol_window, len(x) if len(x) % 2 else len(x) - 1)
        if window > self.savgol_poly:
            x = savgol_filter(x, window, self.savgol_poly)
            y = savgol_filter(y, window, self.savgol_poly)
        dt = np.maximum(np.r_[1 / self.fps, np.diff(times)], .001)
        dt[np.isnan(dt)] = 1 / self.fps
        displacement = np.hypot(np.r_[0., np.diff(x)], np.r_[0., np.diff(y)])
        speed = np.where(displacement < self.dist_thresh, 0., displacement) / dt
        return {'x_clean': x, 'y_clean': y, 'speed': speed,
                'activity': (speed > .1).astype(float),
                'is_rejected': rejected.astype(int), 'is_filled': missing.astype(int),
                'coordinate_status': np.where(missing, 'filled',
                    np.where(flags == 1, 'interpolated', 'detected'))}
