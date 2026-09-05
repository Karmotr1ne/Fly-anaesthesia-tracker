"""
Pipeline Orchestrator
=====================
Unifies Modules 1 through 5 into an end-to-end processing pipeline.
Execution Flow:
1. Vision Tracking / Load Raw Data
2. Kinematic Cleaning
3. Early Plotting (Dual Y-Axis Activity/Position & Normalized Kymograph Heatmap)
4. Anesthesia Kinetics Analysis (Baseline Calibration & 3-State Machine)
5. State Spectrograms & Video Overlay Rendering
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
        # 对齐麻醉时间参数接口
        self.anesthesia_analyzer = AnesthesiaAnalyzer(
            fps=self.config.fps,
            anesthesia_still_sec=self.config.anesthesia_still_sec,
            anesthesia_speed_thresh=self.config.anesthesia_speed_thresh,
            sedate_speed_ratio=self.config.sedate_speed_ratio,
            sedate_drop_speed=self.config.sedate_drop_speed,
            anesthesia_onset_time=getattr(self.config, "anesthesia_onset_time", 0.0),
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

        # 确定麻醉释放时间戳（优先使用传参，兜底使用 config）
        gas_onset = float(
            anesthesia_onset_time
            if anesthesia_onset_time is not None
            else getattr(self.config, "anesthesia_onset_time", 0.0)
        )

        # -----------------------------------------------------------------
        # 阶段 1: 判定数据源（优先加载已有 CSV，无 CSV 时才触发视觉跟踪）
        # -----------------------------------------------------------------
        is_freshly_tracked = False

        if csv_path and os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)
            raw_csv_path = csv_path
        elif video_path and os.path.exists(video_path):
            if not chamber_rois:
                raise ValueError(f"Chamber ROIs required for video tracking on {base_name}.")
            tracker = FlyVisionTracker(chamber_rois=chamber_rois)
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
            is_freshly_tracked = True
            raw_csv_path = raw_csv_target_path
        else:
            raise FileNotFoundError(f"Input file not found: {csv_path or video_path}")

        if is_freshly_tracked and save_raw_csv:
            raw_df.to_csv(raw_csv_target_path, index=False)

        # -----------------------------------------------------------------
        # 阶段 2: 运动学清洗
        # -----------------------------------------------------------------
        cleaned_df = self.cleaner.clean_trajectory(raw_df)

        # -----------------------------------------------------------------
        # 阶段 3: 【先绘图】在状态判定前，先绘制双 Y 轴图与 Kymograph 热图
        # -----------------------------------------------------------------
        plot_paths = {}
        if generate_plots:
            # 1. 双 Y 轴行为概览图 (Activity / Velocity & Position)
            act_pos_plot = f"{out_prefix}_activity_position.png"
            self.visualizer.plot_activity_position_overview(cleaned_df, act_pos_plot)
            plot_paths["activity_position"] = act_pos_plot

            # 2. 空间-时间归一化 Hexbin 热图
            kymo_plot = f"{out_prefix}_kymograph_norm.png"
            self.visualizer.plot_kymograph_hexbin(cleaned_df, kymo_plot)
            plot_paths["kymograph"] = kymo_plot

        # -----------------------------------------------------------------
        # 阶段 4: 【后判定】三阶段状态判定与动力学汇总（执行 Baseline 校正）
        # -----------------------------------------------------------------
        # 在给药前的数据严格用于个体 baseline 校正，且在此之前状态强制为 Active
        df_with_states = self.anesthesia_analyzer.evaluate_states(
            cleaned_df,
            anesthesia_onset_time=gas_onset
        )

        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            df_with_states.to_csv(cleaned_csv_path, index=False)

        # 提取关键动力学指标（含潜伏期）
        summary_df = self.anesthesia_analyzer.extract_summary(
            df_with_states,
            anesthesia_onset_time=gas_onset
        )
        summary_csv_path = f"{out_prefix}_results_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        # 状态机衍生图表（依赖 state 列）
        if generate_plots:
            spectrogram_plot = f"{out_prefix}_activity_spectrogram.png"
            preference_plot = f"{out_prefix}_pos_preference.png"
            self.visualizer.plot_activity_spectrogram(df_with_states, spectrogram_plot)
            self.visualizer.plot_position_preference(df_with_states, preference_plot)
            plot_paths["spectrogram"] = spectrogram_plot
            plot_paths["preference"] = preference_plot

        # -----------------------------------------------------------------
        # 阶段 5: 最终视频重合标注渲染（Overlay Video）
        # -----------------------------------------------------------------
        overlay_video_path = None
        if render_video_overlay and video_path and os.path.exists(video_path):
            overlay_video_path = f"{out_prefix}_cleaned_overlay.mp4"
            self.visualizer.render_overlay_video(
                df_with_states,
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
            "cleaned_df": df_with_states,
            "summary_df": summary_df,
            "cleaned_csv_path": cleaned_csv_path if save_cleaned_csv else None,
            "summary_csv_path": summary_csv_path,
            "plot_paths": plot_paths,
            "overlay_video_path": overlay_video_path
        }