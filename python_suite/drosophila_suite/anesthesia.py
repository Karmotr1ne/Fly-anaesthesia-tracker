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
    """
    Analyzes behavioral knockdown dynamics, induction time points, and sedation kinetics.
    """

    def __init__(
        self,
        bin_size_sec: float = 5.0,
        window_duration_sec: float = 120.0,
        window_bins: Optional[int] = None,
        activity_threshold: float = 0.01,
        fps: float = 30.0,
    ):
        self.bin_size_sec = bin_size_sec
        self.window_duration_sec = window_duration_sec
        if window_bins is not None:
            self.window_bins = window_bins
        else:
            self.window_bins = max(1, int(round(window_duration_sec / bin_size_sec)))
        self.activity_threshold = activity_threshold
        self.fps = fps

    def evaluate_induction(
        self,
        cleaned_df: pd.DataFrame,
        fps: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Calculates time to sedation per chamber.

        Parameters
        ----------
        cleaned_df : pd.DataFrame
            DataFrame containing columns: ['frame', 'chamber_id', 'speed'] (and optionally 'timestamp_s')
        fps : float, optional
            Video acquisition frame rate (defaults to instance fps).

        Returns
        -------
        pd.DataFrame
            Summary dataframe with columns:
            ['chamber_id', 'induction_time_sec', 'is_sedated', 'baseline_speed', 'pre_sedation_activity', 'stillness_bins_count']
        """
        if cleaned_df.empty:
            return pd.DataFrame(columns=[
                "chamber_id", "induction_time_sec", "is_sedated", "baseline_speed", "pre_sedation_activity", "stillness_bins_count"
            ])

        effective_fps = fps or self.fps
        frames_per_bin = max(1, int(round(self.bin_size_sec * effective_fps)))
        results = []

        for cid, group in cleaned_df.groupby("chamber_id"):
            grp = group.sort_values("frame").reset_index(drop=True)
            speeds = grp["speed"].fillna(0.0).to_numpy()
            total_frames = len(speeds)

            if total_frames == 0:
                continue

            # 1. Temporal Binning of Mean Speed/Activity (5s bins)
            num_bins = int(np.ceil(total_frames / frames_per_bin))
            binned_activity = np.zeros(num_bins, dtype=np.float64)
            time_axis_sec = np.arange(num_bins) * self.bin_size_sec

            for b in range(num_bins):
                start_f = b * frames_per_bin
                end_f = min(total_frames, (b + 1) * frames_per_bin)
                binned_activity[b] = np.mean(speeds[start_f:end_f])

            # Baseline speed (first 10% of recording or initial 60 seconds)
            init_bins = max(1, min(12, int(num_bins * 0.1)))
            baseline_speed = float(np.mean(binned_activity[:init_bins]))

            # Count bins below activity threshold
            stillness_bins_count = int(np.sum(binned_activity < self.activity_threshold))

            # 2. Sliding Window Max Inactivity Detection Engine (W = 120s, 24 bins)
            stationary_mask = StationaryDetectionEngine.sliding_window_max_filter(
                activity_series=binned_activity,
                window_size=self.window_bins,
                threshold=self.activity_threshold
            )

            # 3. Extraction of First True Inactivity Onset
            induction_time_sec = None
            is_sedated = False
            pre_sedation_activity = 0.0

            true_indices = np.where(stationary_mask)[0]
            if len(true_indices) > 0:
                first_onset_bin = true_indices[0]
                induction_time_sec = round(float(time_axis_sec[first_onset_bin]), 2)
                is_sedated = True

                # Compute average activity prior to sedation onset
                if first_onset_bin > 0:
                    pre_sedation_activity = float(np.mean(binned_activity[:first_onset_bin]))
                else:
                    pre_sedation_activity = float(binned_activity[0])
            else:
                # Animal did not reach full sustained sedation
                pre_sedation_activity = float(np.mean(binned_activity))

            results.append({
                "chamber_id": int(cid),
                "induction_time_sec": induction_time_sec,
                "is_sedated": is_sedated,
                "baseline_speed": round(baseline_speed, 2),
                "pre_sedation_activity": round(pre_sedation_activity, 2),
                "stillness_bins_count": stillness_bins_count,
            })

        summary_df = pd.DataFrame(results)
        return summary_df
