"""
Module 5: Scientific Visualization, Video Overlay & Report Engine
=================================================================
Generates publication-quality dual Y-axis dynamic plots, normalized space-time
Kymographs, annotated video overlays, and structured statistical reports.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering & PyInstaller safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ScientificVisualizer:
    """
    Visual reporting and scientific graphic generation suite for Drosophila Anesthesia.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Publication styling
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_activity_position_overview(
        self,
        cleaned_df: pd.DataFrame,
        output_filepath: str,
        fps: Optional[float] = None
    ) -> str:
        """
        Generates multi-chamber dual Y-axis behavioral dynamic panels:
        Left Y-axis: Activity / smoothed velocity (area fill #00A2E8).
        Right Y-axis: Vertical Position (0.0=Ground to 1.0=Top, red line #E53935).
        """
        effective_fps = fps or self.fps
        chambers = sorted(cleaned_df["chamber_id"].dropna().unique())
        if not chambers:
            return ""

        num_ch = len(chambers)
        fig, axes = plt.subplots(
            nrows=num_ch,
            ncols=1,
            figsize=(11, max(2.2 * num_ch, 4.0)),
            sharex=True
        )
        if num_ch == 1:
            axes = [axes]

        for i, ch in enumerate(chambers):
            ax_pos = axes[i]        # Position axis (Right)
            ax_act = ax_pos.twinx() # Activity axis (Left)

            sub = cleaned_df[cleaned_df["chamber_id"] == ch].sort_values("frame").reset_index(drop=True)
            
            if "timestamp_s" in sub.columns and sub["timestamp_s"].notna().sum() > 0:
                time_sec = sub["timestamp_s"].to_numpy()
            else:
                time_sec = (sub["frame"] / effective_fps).to_numpy()

            # Normalized Position
            if "norm_pos" in sub.columns and sub["norm_pos"].notna().sum() > 5:
                norm_pos = sub["norm_pos"]
            else:
                valid_x = sub["x_clean"].dropna() if "x_clean" in sub.columns else sub["x_px"].dropna()
                if len(valid_x) > 10:
                    min_x, max_x = np.percentile(valid_x, [1, 99])
                    span = max_x - min_x if max_x > min_x else 1.0
                    norm_pos = np.clip(((sub["x_clean"] if "x_clean" in sub.columns else sub["x_px"]) - min_x) / span, 0.0, 1.0)
                else:
                    norm_pos = pd.Series(0.5, index=sub.index)

            # Activity / Smoothed Speed
            speed_col = sub["speed"] if "speed" in sub.columns else pd.Series(0.0, index=sub.index)
            smooth_speed = speed_col.rolling(window=15, min_periods=1, center=True).mean().fillna(0.0)

            # 1. Plot Activity (Left axis - Cyan/Blue fill + line)
            ax_act.plot(time_sec, smooth_speed, color="#00A2E8", lw=1.2, label="Activity (px/s)", alpha=0.9)
            ax_act.fill_between(time_sec, 0, smooth_speed, color="#00A2E8", alpha=0.22)
            ax_act.set_ylabel("Activity (px/s)", color="#0083B8", fontweight="bold", fontsize=8.5)
            ax_act.tick_params(axis="y", labelcolor="#0083B8", labelsize=8)
            ax_act.set_ylim(0, max(float(smooth_speed.max()) * 1.15, 60.0))

            # 2. Plot Vertical Position (Right axis - Crimson red line)
            ax_pos.plot(time_sec, norm_pos, color="#E53935", lw=1.6, label="Vertical Pos (au)")
            ax_pos.set_ylabel("Position (au)", color="#C62828", fontweight="bold", fontsize=8.5)
            ax_pos.tick_params(axis="y", labelcolor="#C62828", labelsize=8)
            ax_pos.set_ylim(-0.05, 1.05)
            ax_pos.set_yticks([0.0, 0.5, 1.0])
            ax_pos.set_yticklabels(["0.0 (Bottom)", "0.5", "1.0 (Top)"], fontsize=7.5)

            # Layout & titles
            ax_pos.set_title(f"Chamber {int(ch)}", loc="left", fontsize=9.5, fontweight="bold", pad=2)
            ax_pos.grid(True, linestyle="--", alpha=0.35)

            # Reorder axis labels for standard dual presentation
            ax_pos.yaxis.set_label_position("right")
            ax_pos.yaxis.tick_right()
            ax_act.yaxis.set_label_position("left")
            ax_act.yaxis.tick_left()

        axes[-1].set_xlabel("Time (s)", fontweight="bold", fontsize=10)
        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plt.savefig(output_filepath, dpi=200)
        plt.close()
        return output_filepath

    def plot_kymograph_hexbin(
        self,
        cleaned_df: pd.DataFrame,
        output_filepath: str,
        fps: Optional[float] = None
    ) -> str:
        """
        Renders population-level space-time distribution Kymograph (Hexbin Heatmap).
        """
        effective_fps = fps or self.fps
        if cleaned_df.empty or "norm_pos" not in cleaned_df.columns:
            return ""

        valid_df = cleaned_df.dropna(subset=["norm_pos", "frame"]).copy()
        if valid_df.empty:
            return ""

        if "timestamp_s" in valid_df.columns and valid_df["timestamp_s"].notna().sum() > 0:
            time_sec = valid_df["timestamp_s"]
        else:
            time_sec = valid_df["frame"] / effective_fps

        plt.figure(figsize=(10, 5.2))
        
        hb = plt.hexbin(
            time_sec,
            valid_df["norm_pos"],
            gridsize=(80, 30),
            cmap="inferno",
            bins="log",
            mincnt=1
        )
        cb = plt.colorbar(hb)
        cb.set_label(r"$\log_{10}(\mathrm{Drosophila\ Occurrence\ Count})$", fontsize=9.5)

        plt.title("Population Space-Time Distribution (Normalized Kymograph)", fontweight="bold", fontsize=11, pad=10)
        plt.xlabel("Time (seconds)", fontweight="bold", fontsize=10)
        plt.ylabel("Vertical Position (0.0=Ground, 1.0=Top)", fontweight="bold", fontsize=10)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plt.savefig(output_filepath, dpi=200)
        plt.close()
        return output_filepath

    def plot_survival_kinetics(self, summary_df: pd.DataFrame, save_path: str):
        """
        根据 summary_df 绘制累积击倒/麻醉生存图 (Fraction reached vs. Time after gas onset)
        """
        if summary_df is None or summary_df.empty:
            return

        total_n = len(summary_df)
        if total_n == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

        # 提取有效潜伏期数据
        sed_latencies = summary_df["sedation_latency_sec"].dropna().to_numpy()
        ana_latencies = summary_df["anesthesia_latency_sec"].dropna().to_numpy()

        # 配色规范（对齐示例图）
        color_sed = "#E67E22"   # 亮橙色
        color_ana = "#B03A2E"   # 深红/红棕色

        max_time = 60.0

        # 1. 绘制 Sedation 曲线
        if len(sed_latencies) > 0:
            sorted_sed = np.sort(sed_latencies)
            max_time = max(max_time, sorted_sed[-1])
            # 阶梯曲线累积点：(0, 0) -> 各事件点
            x_sed = np.concatenate(([0.0], sorted_sed))
            y_sed = np.concatenate(([0.0], (np.arange(1, len(sorted_sed) + 1) / total_n) * 100.0))
            ax.step(x_sed, y_sed, where="post", color=color_sed, lw=2.5, label="Sedation")

            # 中位数标注
            med_sed = float(np.median(sorted_sed))
            ax.axvline(med_sed, color=color_sed, ls=":", lw=1.2, alpha=0.85)
            ax.text(
                med_sed + (max_time * 0.02),
                60.0,
                f"Median sed: {med_sed:.2f} s",
                color=color_sed,
                fontsize=9.5,
                fontweight="medium",
                va="center",
            )

        # 2. 绘制 Anesthesia 曲线
        if len(ana_latencies) > 0:
            sorted_ana = np.sort(ana_latencies)
            max_time = max(max_time, sorted_ana[-1])
            x_ana = np.concatenate(([0.0], sorted_ana))
            y_ana = np.concatenate(([0.0], (np.arange(1, len(sorted_ana) + 1) / total_n) * 100.0))
            ax.step(x_ana, y_ana, where="post", color=color_ana, lw=2.5, label="Anesthesia")

            # 中位数标注
            med_ana = float(np.median(sorted_ana))
            ax.axvline(med_ana, color=color_ana, ls=":", lw=1.2, alpha=0.85)
            ax.text(
                med_ana + (max_time * 0.02),
                52.0,
                f"Median anes: {med_ana:.2f} s",
                color=color_ana,
                fontsize=9.5,
                fontweight="medium",
                va="center",
            )

        # 3. 气体释放起始线 (Time = 0)
        ax.axvline(0.0, color="#94A3B8", ls="--", lw=1.2, alpha=0.85)
        ax.text(1.5, 2.0, "CO2 onset", color="#64748B", fontsize=9, va="bottom")

        # 4. 坐标轴与排版美化
        ax.set_xlabel("Time after gas onset (s)", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_ylabel("Fraction reached (%)", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_ylim(-2, 105)
        ax.set_xlim(-max_time * 0.03, max_time * 1.08)

        # 仅保留底部和左侧轴脊柱
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.grid(axis="x", linestyle="-", color="#F1F5F9", lw=0.8)
        ax.legend(frameon=False, loc="lower right", fontsize=10)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def render_overlay_video(
        self,
        cleaned_df: pd.DataFrame,
        input_video_path: str,
        output_video_path: str,
        scale: Optional[float] = None,
        target_width: int = 1080,
        progress_callback=None
    ) -> str:
        """
        High-performance annotated video overlay synthesis with adaptive downsizing
        and clear chamber indicators.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        # 1. Adaptive downscale calculation
        if scale is None:
            if w > target_width:
                scale = target_width / float(w)
            else:
                scale = 1.0

        out_w = int(w * scale)
        out_h = int(h * scale)

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

        # Fast group indexing by frame
        frame_grouped = {f: grp for f, grp in cleaned_df.groupby("frame")}

        f_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if scale != 1.0:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                if f_idx in frame_grouped:
                    grp = frame_grouped[f_idx]
                    for _, row in grp.iterrows():
                        x_col = "x_clean" if "x_clean" in row and pd.notna(row["x_clean"]) else "x_px"
                        y_col = "y_clean" if "y_clean" in row and pd.notna(row["y_clean"]) else "y_px"
                        
                        if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
                            cx = int(round(float(row[x_col]) * scale))
                            cy = int(round(float(row[y_col]) * scale))
                            ch_id = int(row.get("chamber_id", 0))

                            # Green core point
                            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
                            # Cyan crosshair ring
                            cv2.circle(frame, (cx, cy), 6, (0, 220, 255), 1, cv2.LINE_AA)
                            # Clear chamber badge
                            cv2.putText(
                                frame,
                                f"CH{ch_id}",
                                (cx + 6, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.38,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA
                            )

                writer.write(frame)
                f_idx += 1

                if progress_callback and f_idx % 100 == 0:
                    progress_callback(f_idx, total_frames)
        finally:
            cap.release()
            writer.release()

        return output_video_path
