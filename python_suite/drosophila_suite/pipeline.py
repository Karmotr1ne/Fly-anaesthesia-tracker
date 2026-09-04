"""
Pipeline Orchestrator
=====================
Unifies Modules 1 through 5 into an end-to-end processing pipeline.
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .models import PipelineConfig
from .tracker import FlyVisionTracker, get_video_metadata
from .cleaner import KinematicCleaner
from .anesthesia import AnesthesiaAnalyzer
from .visualizer import ScientificVisualizer


class DrosophilaBehaviorPipeline:
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
        # 对齐三阶段状态机参数接口
        self.anesthesia_analyzer = AnesthesiaAnalyzer(
            fps=self.config.fps,
            anesthesia_still_sec=self.config.anesthesia_still_sec,
            anesthesia_speed_thresh=self.config.anesthesia_speed_thresh,
            sedate_speed_ratio=self.config.sedate_speed_ratio,
            sedate_drop_speed=self.config.sedate_drop_speed,
        )
        self.visualizer = ScientificVisualizer(fps=self.config.fps)

    def process_file_pair(
        self,
        csv_path: Optional[str] = None,
        video_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        base_name: Optional[str] = None,
        save_cleaned_csv: bool = True,
        generate_plots: bool = True,
        render_video_overlay: bool = False,
        chamber_rois: Optional[List[Tuple[int, int, int, int]]] = None,
        progress_callback=None,
        render_progress_callback=None
    ) -> Dict[str, Any]:
        start_time = time.time()
        
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

        # 阶段 1 & 3: 提取或读取 raw 坐标
        if csv_path and os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)
        elif video_path and os.path.exists(video_path):
            if not chamber_rois:
                raise ValueError("Chamber ROIs required for video tracking.")
            tracker = FlyVisionTracker(chamber_rois=chamber_rois)
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
            raw_df.to_csv(f"{out_prefix}_raw.csv", index=False)
        else:
            raise FileNotFoundError(f"Input file not found: {csv_path or video_path}")

        # 阶段 4: 运动学清洗与前后中点缺失填充
        cleaned_df = self.cleaner.clean_trajectory(raw_df)

        # 阶段 5: 三阶段非强制连续状态机判定 (Active -> Sedate -> Anaesthesia)
        cleaned_df = self.anesthesia_analyzer.evaluate_states(cleaned_df)
        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            cleaned_df.to_csv(cleaned_csv_path, index=False)

        # 阶段 6: 汇总表与科学图谱
        summary_df = self.anesthesia_analyzer.extract_summary(cleaned_df)
        summary_csv_path = f"{out_prefix}_results_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        plot_paths = {}
        if generate_plots:
            spectrogram_plot = f"{out_prefix}_activity_spectrogram.png"
            preference_plot = f"{out_prefix}_pos_preference.png"
            self.visualizer.plot_activity_spectrogram(cleaned_df, spectrogram_plot)
            self.visualizer.plot_position_preference(cleaned_df, preference_plot)
            plot_paths["spectrogram"] = spectrogram_plot
            plot_paths["preference"] = preference_plot

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
