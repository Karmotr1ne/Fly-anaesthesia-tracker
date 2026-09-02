"""
Module 3: Core Stationary Detection Engine
==========================================
Unified high-performance inactivity and sliding-window extreme filter operator.
Provides mathematical foundation for both anesthesia induction and sleep staging.
"""

from typing import Union, List, Tuple
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import maximum_filter1d
    HAS_SCIPY_NDIMAGE = True
except ImportError:
    HAS_SCIPY_NDIMAGE = False


class StationaryDetectionEngine:
    """
    Core engine for sliding-window inactivity and state-duration analysis.
    Evaluates condition: max(Activity[t : t + W]) < threshold
    """

    @staticmethod
    def sliding_window_max_filter(
        activity_series: Union[np.ndarray, List[float], pd.Series],
        window_size: int,
        threshold: float
    ) -> np.ndarray:
        """
        Sliding Window Maximum Filter:
        Determines whether the maximum activity across the forward window
        of length `window_size` strictly remains below `threshold`.

        Parameters
        ----------
        activity_series : array-like of shape (N,)
            Continuous or binned activity metric (e.g., speed, delta distance).
        window_size : int
            Number of points/bins in forward sliding window.
        threshold : float
            Activity cutoff below which animal is considered stationary.

        Returns
        -------
        np.ndarray of bool, shape (N,)
            Boolean mask where True indicates onset of a sustained stationary window.
        """
        # Ensure memory continuity and standard float64 dtype to safely utilize vectorization & as_strided
        arr = np.ascontiguousarray(activity_series, dtype=np.float64)
        n = len(arr)
        if n == 0 or window_size <= 0:
            return np.zeros(0, dtype=bool)

        if window_size > n:
            # Entire series is shorter than window: check whole array max
            all_below = (np.nanmax(arr) < threshold) if not np.all(np.isnan(arr)) else False
            res = np.zeros(n, dtype=bool)
            if all_below:
                res[0] = True
            return res

        is_stationary = np.zeros(n, dtype=bool)
        valid_len = n - window_size + 1

        # 1. Attempt O(N) 1D rolling maximum filter via SciPy if available
        if HAS_SCIPY_NDIMAGE and not np.any(np.isnan(arr)):
            try:
                # Forward window [t : t + window_size]: origin offset shifts center of kernel to start
                origin = -(window_size // 2)
                filtered = maximum_filter1d(arr, size=window_size, mode="nearest", origin=origin)
                is_stationary[:valid_len] = filtered[:valid_len] < threshold
                return is_stationary
            except Exception:
                pass

        # 2. Fast 2D strided view on guaranteed contiguous array
        try:
            shape = (valid_len, window_size)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            max_vals = np.nanmax(windows, axis=1)
            is_stationary[:valid_len] = max_vals < threshold
            return is_stationary
        except Exception:
            pass

        # 3. Robust O(N*W) iterative fallback
        for t in range(valid_len):
            w_max = np.nanmax(arr[t : t + window_size])
            if w_max < threshold:
                is_stationary[t] = True

        return is_stationary

    @staticmethod
    def extract_stationary_bouts(
        binary_mask: np.ndarray,
        fps_or_sampling_rate: float = 1.0
    ) -> List[Tuple[int, int, float]]:
        """
        Extract continuous stationary bouts [start_idx, end_idx, duration_sec].
        """
        bouts = []
        in_bout = False
        start_idx = 0
        n = len(binary_mask)

        for i in range(n):
            if binary_mask[i] and not in_bout:
                in_bout = True
                start_idx = i
            elif not binary_mask[i] and in_bout:
                in_bout = False
                duration_sec = (i - start_idx) / fps_or_sampling_rate
                bouts.append((start_idx, i - 1, duration_sec))

        if in_bout:
            duration_sec = (n - start_idx) / fps_or_sampling_rate
            bouts.append((start_idx, n - 1, duration_sec))

        return bouts
