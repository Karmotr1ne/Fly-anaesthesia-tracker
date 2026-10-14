"""
Pipeline Orchestrator (Refactored)
==================================
Unifies Modules 1 through 5 into an end-to-end processing pipeline:
1. Vision Tracking / Load Raw Tracking CSV[cite: 14]
2. Kinematic Cleaning & Outlier Rejection[cite: 14]
3. Behavioral State Machine (Confidence-Driven 3-State Evaluation)[cite: 11, 14]
4. Scientific Figures (Dual-Y Overview, Space-Time Kymograph, Survival Kinetics)[cite: 14]
5. Annotated Video Overlay Synthesis[cite: 14]
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from .models import PipelineConfig
from .tracker import FlyVisionTracker, get_video_metadata, SymmetricGridAligner
from .cleaner import KinematicCleaner
from .anesthesia import AnesthesiaAnalyzer
from .visualizer import ScientificVisualizer


class DrosophilaBehaviorPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # 1. 运动学物理清洗器[cite: 14]
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
        
        # 2. 状态机与动力学分析器 (参数严格对齐新版 AnesthesiaAnalyzer)[cite: 11, 14]
        self.anesthesia_analyzer = AnesthesiaAnalyzer(
            fps=self.config.fps,
            anesthesia_still_sec=self.config.anesthesia_still_sec,
            speed_still_thresh=self.config.anesthesia_speed_thresh,
            baseline_quantile=self.config.baseline_quantile,
            sedate_speed_ratio=self.config.sedate_speed_ratio,
            floor_height_thresh=self.config.floor_height_thresh,
            ceiling_height_thresh=self.config.ceiling_height_thresh,
            drop_dt_sec=self.config.drop_dt_sec,
            drop_delta_h=self.config.sedate_drop_speed,
            anesthesia_onset_time=self.config.anesthesia_onset_time,
            gas_mixing_window_sec=self.config.gas_mixing_window_sec,
            sedation_conf_thresh=self.config.sedation_conf_thresh,
            rebound_horizon_sec=self.config.rebound_horizon_sec,
        )
        
        # 3. 科学可视化组件[cite: 14]
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
        执行单组实验会话的端到端批处理分析[cite: 14]
        """
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
                    base_name = base_name[:-len(suffix)]

        out_prefix = os.path.join(target_dir, base_name)
        raw_csv_target_path = f"{out_prefix}_raw.csv"

        # 确定通气开启时刻 (优先使用入参，回退配置)[cite: 14]
        gas_onset = float(
            anesthesia_onset_time
            if anesthesia_onset_time is not None
            else getattr(self.config, "anesthesia_onset_time", 300.0)
        )

        # -----------------------------------------------------------------
        # 阶段 1: 判定数据源（优先加载已有 CSV，无 CSV 时才跑视觉跟踪）[cite: 14]
        # -----------------------------------------------------------------
        is_freshly_tracked = False
        if csv_path and os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)
            raw_csv_path = csv_path
        elif video_path and os.path.exists(video_path):
            if not chamber_rois:
                _, _, width, height = get_video_metadata(video_path)
                chambers = SymmetricGridAligner.generate_symmetric_chambers(
                    (height, width), rows=self.config.grid_rows,
                    cols=self.config.grid_cols, order=self.config.grid_order,
                )
                chamber_rois = [c["roi"] for c in chambers]
            tracker = FlyVisionTracker(chamber_rois=chamber_rois)
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
            is_freshly_tracked = True
            raw_csv_path = raw_csv_target_path
            if save_raw_csv:
                raw_df.to_csv(raw_csv_target_path, index=False)
        else:
            raise FileNotFoundError(f"Input file not found: {csv_path or video_path}")

        # -----------------------------------------------------------------
        # 阶段 2: 运动学清洗与物理边界规整[cite: 14]
        # -----------------------------------------------------------------
        # 若传入的本身已经是 _cleaned.csv，包含 x_clean 和 speed，则可跳过清洗阶段直接进入状态评估
        if "x_clean" in raw_df.columns and "speed" in raw_df.columns:
            cleaned_df = raw_df.copy()
        else:
            cleaned_df = self.cleaner.clean_trajectory(raw_df)

        if cleaned_df.empty:
            raise ValueError("No valid trajectory data available for state analysis")

        # -----------------------------------------------------------------
        # 阶段 3: 状态机判定 (三阶段状态机评估与潜伏期汇总)[cite: 11, 14]
        # -----------------------------------------------------------------
        df_with_states = self.anesthesia_analyzer.evaluate_states(
            cleaned_df,
            anesthesia_onset_time=gas_onset
        )

        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            df_with_states.to_csv(cleaned_csv_path, index=False)

        summary_df = self.anesthesia_analyzer.extract_summary(
            df_with_states,
            anesthesia_onset_time=gas_onset
        )
        summary_csv_path = f"{out_prefix}_results_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)

        # -----------------------------------------------------------------
        # 阶段 4: 科学图表导出[cite: 14]
        # -----------------------------------------------------------------
        plot_paths = {}
        if generate_plots:
            # 1. 双 Y 轴行为概览图[cite: 14]
            act_pos_plot = f"{out_prefix}_activity_position.png"
            self.visualizer.plot_activity_position_overview(df_with_states, act_pos_plot, fps=self.config.fps)
            plot_paths["activity_position"] = act_pos_plot

            # 2. 空间-时间归一化 Hexbin 热图[cite: 14]
            kymo_plot = f"{out_prefix}_kymograph_norm.png"
            self.visualizer.plot_kymograph_hexbin(df_with_states, kymo_plot, fps=self.config.fps)
            plot_paths["kymograph"] = kymo_plot

            # 3. 累计麻醉动力学生存曲线[cite: 14]
            survival_plot = f"{out_prefix}_survival_kinetics.png"
            self.visualizer.plot_survival_kinetics(summary_df, survival_plot)
            plot_paths["survival_kinetics"] = survival_plot

        # -----------------------------------------------------------------
        # 阶段 5: 最终视频重合标注渲染 (Overlay Video)[cite: 14]
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
            "cleaned_df": df_with_states,
            "summary_df": summary_df,
            "cleaned_csv_path": cleaned_csv_path if save_cleaned_csv else None,
            "summary_csv_path": summary_csv_path,
            "plot_paths": plot_paths,
            "overlay_video_path": overlay_video_path
        }