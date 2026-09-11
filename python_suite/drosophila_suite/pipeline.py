"""
Pipeline Orchestrator
=====================
Unifies Modules 1 through 4 (Tracking, Cleaning, Stationary Engine & Anesthesia, Visualizer)
into an end-to-end processing pipeline.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .models import PipelineConfig, AnesthesiaSummary
from .tracker import FlyVisionTracker, get_video_metadata
from .cleaner import KinematicCleaner
from .anesthesia import AnesthesiaAnalyzer
from .visualizer import ScientificVisualizer


class DrosophilaBehaviorPipeline:
    """
    End-to-end processing orchestrator for Drosophila Anesthesia experiments.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.cleaner = KinematicCleaner(
            fps=self.config.fps,
            max_speed_px=self.config.max_speed_px_per_frame,
            body_len_thresh=self.config.body_length_thresh,
            body_len_px=self.config.body_length_px,
            occlusion_disp_thresh=self.config.occlusion_disp_thresh,
            occlusion_var_thresh=self.config.occlusion_var_thresh,
            savgol_window=self.config.savgol_window,
            savgol_poly=self.config.savgol_poly,
        )
        self.anesthesia_analyzer = AnesthesiaAnalyzer(
            fps=self.config.fps,
            anesthesia_still_sec=getattr(self.config, 'anesthesia_still_sec', 120.0),
            anesthesia_speed_thresh=getattr(self.config, 'anesthesia_speed_thresh', 0.10),
            sedate_speed_ratio=getattr(self.config, 'sedate_speed_ratio', 0.35),
            drop_height_delta=getattr(self.config, 'sedate_drop_speed', 0.25),
            anesthesia_onset_time=getattr(self.config, 'anesthesia_onset_time', 0.0),
        )
        self.visualizer = ScientificVisualizer(fps=self.config.fps)

    def process_file_pair(
        self,
        csv_path: Optional[str] = None,
        video_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        base_name: Optional[str] = None,
        anesthesia_onset_time: Optional[float] = None,
        save_raw_csv: bool = True,
        save_cleaned_csv: bool = True,
        generate_plots: bool = True,
        render_video_overlay: bool = False,
        chamber_rois: Optional[List[Tuple[int, int, int, int]]] = None,
        progress_callback=None,
        render_progress_callback=None
    ) -> Dict[str, Any]:
        """
        Executes the behavioral analysis pipeline for a single session.
        """
        start_time = time.time()
        
        # 1. Resolve output directory and base name
        ref_path = csv_path or video_path
        if not ref_path:
            raise ValueError("Must provide at least csv_path or video_path.")

        target_dir = output_dir or os.path.dirname(os.path.abspath(ref_path))
        os.makedirs(target_dir, exist_ok=True)

        if not base_name:
            base_name = os.path.splitext(os.path.basename(ref_path))[0]
            for suffix in ["_tracked", "_raw", "_cleaned", "_v1", "_v2", "_v3"]:
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]

        out_prefix = os.path.join(target_dir, base_name)
        raw_csv_target_path = f"{out_prefix}_raw.csv"

        # Step 1: Obtain raw tracking DataFrame
        is_freshly_tracked = False
        if csv_path and os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)
        elif video_path and os.path.exists(video_path):
            if not chamber_rois:
                raise ValueError(f"Chamber ROIs required for video tracking on {base_name}.")
            tracker = FlyVisionTracker(chamber_rois=chamber_rois)
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
            is_freshly_tracked = True
            if save_raw_csv:
                raw_df.to_csv(raw_csv_target_path, index=False)
        else:
            raise FileNotFoundError(f"Input file not found: {csv_path or video_path}")

        # Step 2: Kinematic Cleaning & Artifact Clamping
        cleaned_df = self.cleaner.clean_trajectory(raw_df)
        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            cleaned_df.to_csv(cleaned_csv_path, index=False)

        # Step 3: Anesthesia Induction Kinetics & 3-State Evaluation
        gas_onset = float(
            anesthesia_onset_time
            if anesthesia_onset_time is not None
            else getattr(self.config, 'anesthesia_onset_time', 0.0)
        )
        cleaned_df = self.anesthesia_analyzer.evaluate_states(cleaned_df, anesthesia_onset_time=gas_onset)

        # Step 4: Summary Metrics Consolidation
        summary_df = self.anesthesia_analyzer.extract_summary(cleaned_df, anesthesia_onset_time=gas_onset)
        summary_csv_path = f"{out_prefix}_results_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        # Step 5: Scientific Plotting
        plot_paths = {}
        if generate_plots:
            act_pos_plot = f"{out_prefix}_activity_position.png"
            kymo_plot = f"{out_prefix}_kymograph_norm.png"
            survival_plot = f"{out_prefix}_survival_kinetics.png"

            self.visualizer.plot_activity_position_overview(
                cleaned_df, act_pos_plot, fps=self.config.fps
            )
            self.visualizer.plot_kymograph_hexbin(cleaned_df, kymo_plot, fps=self.config.fps)
            
            # 导出累计麻醉动力学生存图
            self.visualizer.plot_survival_kinetics(summary_df, survival_plot)

            plot_paths["activity_position"] = act_pos_plot
            plot_paths["kymograph"] = kymo_plot
            plot_paths["survival_kinetics"] = survival_plot

        # Step 6: Video Overlay Synthesis
        overlay_video_path = None
        if render_video_overlay and video_path and os.path.exists(video_path):
            overlay_video_path = f"{out_prefix}_cleaned_overlay.mp4"
            self.visualizer.render_overlay_video(
                cleaned_df,
                video_path,
                overlay_video_path,
                scale=None,
                progress_callback=render_progress_callback or progress_callback
            )

        elapsed = round(time.time() - start_time, 2)
        return {
            "base_name": base_name,
            "elapsed_sec": elapsed,
            "cleaned_df": cleaned_df,
            "summary_df": summary_df,
            "cleaned_csv_path": cleaned_csv_path if save_cleaned_csv else None,
            "summary_csv_path": summary_csv_path,
            "plot_paths": plot_paths,
            "overlay_video_path": overlay_video_path
        }