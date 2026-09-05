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
        save_raw_csv: bool = True,
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
        raw_csv_target_path = f"{out_prefix}_raw.csv"

        # -----------------------------------------------------------------
        # 阶段 1: 判定数据源（优先直接加载已有 CSV，无 CSV 时才触发 CV 跟踪）
        # -----------------------------------------------------------------
        is_freshly_tracked = False

        if csv_path and os.path.exists(csv_path):
            # 直接从已有 CSV（如 *_raw.csv）加载数据，跳过耗时追踪
            raw_df = pd.read_csv(csv_path)
            raw_csv_path = csv_path
        elif video_path and os.path.exists(video_path):
            # 只有当缺少 CSV 且存在视频时，才调用视觉追踪
            if not chamber_rois:
                raise ValueError(f"Chamber ROIs required for video tracking on {base_name}.")
            tracker = FlyVisionTracker(chamber_rois=chamber_rois)
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
            is_freshly_tracked = True
            raw_csv_path = raw_csv_target_path
        else:
            raise FileNotFoundError(f"Input file not found: {csv_path or video_path}")

        # -----------------------------------------------------------------
        # 核心解突机制：只有“新追踪出的数据”且“用户勾选保存”时才写入磁盘；
        # 若本就是从 raw.csv 读进来的，绝不执行覆盖写入！
        # -----------------------------------------------------------------
        if is_freshly_tracked and save_raw_csv:
            raw_df.to_csv(raw_csv_target_path, index=False)

        # -----------------------------------------------------------------
        # 阶段 2: 运动学清洗与前后中点补全
        # -----------------------------------------------------------------
        cleaned_df = self.cleaner.clean_trajectory(raw_df)

        # -----------------------------------------------------------------
        # 阶段 3: 三阶段动态状态机判定 (Active -> Sedate -> Anaesthesia)
        # -----------------------------------------------------------------
        cleaned_df = self.anesthesia_analyzer.evaluate_states(cleaned_df)
        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            cleaned_df.to_csv(cleaned_csv_path, index=False)

        # -----------------------------------------------------------------
        # 阶段 4: 统计汇总与科学图表导出
        # -----------------------------------------------------------------
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
            "raw_csv_path": raw_csv_path if (is_freshly_tracked and save_raw_csv) or not is_freshly_tracked else None,
            "cleaned_df": cleaned_df,
            "summary_df": summary_df,
            "cleaned_csv_path": cleaned_csv_path if save_cleaned_csv else None,
            "summary_csv_path": summary_csv_path,
            "plot_paths": plot_paths,
            "overlay_video_path": overlay_video_path
        }