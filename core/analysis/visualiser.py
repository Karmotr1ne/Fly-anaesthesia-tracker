"""
Module 5: Scientific Visualisation, Video Overlay & Report Engine
=================================================================
Generates publication-quality dual Y-axis dynamic plots, normalised space-time
Kymographs, annotated video overlays, and structured statistical reports.
"""
from core import schema

import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering & PyInstaller safety
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ScientificVisualiser:
    """
    Visual reporting and scientific graphic generation suite for Drosophila Anaesthesia.
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
        Generates multi-chamber dual Y-axis behavioural dynamic panels:
        Left Y-axis: Activity / smoothed velocity (area fill #00A2E8).
        Right Y-axis: Vertical Position (0.0=Ground to 1.0=Top, red line #E53935).
        """
        cleaned_df = schema.normalise(cleaned_df)
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

        # Compare physical speeds directly: every chamber uses the same limits.
        speeds = {}
        speed_max = 0.0
        for ch in chambers:
            sub = cleaned_df.loc[cleaned_df['chamber_id'] == ch].sort_values('frame')
            values = sub['speed'] if 'speed' in sub else pd.Series(0., index=sub.index)
            smooth = values.replace([np.inf, -np.inf], np.nan).rolling(
                window=15, min_periods=1, center=True).mean()
            speeds[ch] = smooth.to_numpy()
            finite = smooth[np.isfinite(smooth)]
            if not finite.empty:
                speed_max = max(speed_max, float(finite.max()))
        speed_limit = max(speed_max * 1.15, 1.0)

        for i, ch in enumerate(chambers):
            ax_pos = axes[i]        # Position axis (Right)
            ax_act = ax_pos.twinx() # Activity axis (Left)

            sub = cleaned_df[cleaned_df["chamber_id"] == ch].sort_values("frame").reset_index(drop=True)
            
            if 'time' in sub.columns and sub['time'].notna().sum() > 0:
                time_sec = sub['time'].to_numpy()
            else:
                time_sec = (sub["frame"] / effective_fps).to_numpy()

            # Normalised Position
            if 'position' in sub.columns and sub['position'].notna().sum() > 5:
                position = sub['position']
            else:
                valid_x = sub["x_clean"].dropna() if "x_clean" in sub.columns else sub['x'].dropna()
                if len(valid_x) > 10:
                    min_x, max_x = np.percentile(valid_x, [1, 99])
                    span = max_x - min_x if max_x > min_x else 1.0
                    position = np.clip(((sub["x_clean"] if "x_clean" in sub.columns else sub['x']) - min_x) / span, 0.0, 1.0)
                else:
                    position = pd.Series(0.5, index=sub.index)

            # Activity / Smoothed Speed
            smooth_speed = speeds[ch]

            # 1. Plot Activity (Left axis - Cyan/Blue fill + line)
            ax_act.plot(time_sec, smooth_speed, color="#00A2E8", lw=1.2, label=schema.label("speed"), alpha=0.9)
            ax_act.fill_between(time_sec, 0, smooth_speed, color="#00A2E8", alpha=0.22)
            ax_act.set_ylabel(schema.label("speed"), color="#0083B8", fontweight="bold", fontsize=8.5)
            ax_act.tick_params(axis="y", labelcolor="#0083B8", labelsize=8)
            ax_act.set_ylim(0, speed_limit)

            # 2. Plot Vertical Position (Right axis - Crimson red line)
            ax_pos.plot(time_sec, position, color="#E53935", lw=1.6, label=schema.label("position"))
            ax_pos.set_ylabel(schema.label("position"), color="#C62828", fontweight="bold", fontsize=8.5)
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

        axes[-1].set_xlabel(schema.label("time"), fontweight="bold", fontsize=10)
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
        cleaned_df = schema.normalise(cleaned_df)
        effective_fps = fps or self.fps
        if cleaned_df.empty or 'position' not in cleaned_df.columns:
            return ""

        valid_df = cleaned_df.dropna(subset=['position', "frame"]).copy()
        if valid_df.empty:
            return ""

        if 'time' in valid_df.columns and valid_df['time'].notna().sum() > 0:
            time_sec = valid_df['time']
        else:
            time_sec = valid_df["frame"] / effective_fps

        plt.figure(figsize=(10, 5.2))
        
        hb = plt.hexbin(
            time_sec,
            valid_df['position'],
            gridsize=(80, 30),
            cmap="inferno",
            bins="log",
            mincnt=1
        )
        cb = plt.colorbar(hb)
        cb.set_label(r"$\log_{10}(\mathrm{Drosophila\ Occurrence\ Count})$", fontsize=9.5)

        plt.title("Population Space-Time Distribution (Normalised Kymograph)", fontweight="bold", fontsize=11, pad=10)
        plt.xlabel(schema.label("time"), fontweight="bold", fontsize=10)
        plt.ylabel(schema.label("position"), fontweight="bold", fontsize=10)
        plt.ylim(-0.02, 1.02)
        plt.grid(True, linestyle=":", alpha=0.3)
        plt.tight_layout()

        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        plt.savefig(output_filepath, dpi=200)
        plt.close()
        return output_filepath

    def plot_survival_kinetics(self, summary_df: pd.DataFrame, save_path: str):
        'Plot cumulative sedation and anaesthesia fractions after gas onset.'
        summary_df = schema.normalise(summary_df) if summary_df is not None else None
        if summary_df is None or summary_df.empty:
            return

        total_n = len(summary_df)
        if total_n == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

        # Extract observed event latencies.
        sed_latencies = summary_df['sedation_latency'].dropna().to_numpy()
        ana_latencies = summary_df['anaesthesia_latency'].dropna().to_numpy()

        # Event colors.
        color_sed = "#E67E22"   # Orange.
        color_ana = "#B03A2E"   # Dark red.

        max_time = 60.0

        # 1. Plot sedation events.
        if len(sed_latencies) > 0:
            sorted_sed = np.sort(sed_latencies)
            max_time = max(max_time, sorted_sed[-1])
            # Build cumulative event steps from the origin.
            x_sed = np.concatenate(([0.0], sorted_sed))
            y_sed = np.concatenate(([0.0], (np.arange(1, len(sorted_sed) + 1) / total_n) * 100.0))
            ax.step(x_sed, y_sed, where="post", color=color_sed, lw=2.5, label="Sedation")

            # Annotate the observed median.
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

        # 2. Plot anaesthesia events.
        if len(ana_latencies) > 0:
            sorted_ana = np.sort(ana_latencies)
            max_time = max(max_time, sorted_ana[-1])
            x_ana = np.concatenate(([0.0], sorted_ana))
            y_ana = np.concatenate(([0.0], (np.arange(1, len(sorted_ana) + 1) / total_n) * 100.0))
            ax.step(x_ana, y_ana, where="post", color=color_ana, lw=2.5, label="Anaesthesia")

            # Annotate the observed median.
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

        # 3. Mark gas onset at time zero.
        ax.axvline(0.0, color="#94A3B8", ls="--", lw=1.2, alpha=0.85)
        ax.text(1.5, 2.0, "Anaesthetic gas onset", color="#64748B", fontsize=9, va="bottom")

        # 4. Format axes and layout.
        ax.set_xlabel("time after anaesthetic gas onset(s)", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_ylabel("fraction reached(%)", fontsize=10.5, fontweight="bold", labelpad=6)
        ax.set_ylim(-2, 105)
        ax.set_xlim(-max_time * 0.03, max_time * 1.08)

        # Show only the bottom and left spines.
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

    def plot_spatial_preferences(self, data, save_path, floor_height=.05, ceiling_height=.95,
                                 windows=None, statistics=None):
        """All protocol steps in one figure, with identical colour scales."""
        from .spatial_preference import spatial_statistics
        if windows is None:
            windows = [dict(label='Whole recording', start=None, end=None)]
        if statistics is None:
            statistics = spatial_statistics(data, windows, self.fps, floor_height, ceiling_height)
        if not statistics:
            raise ValueError("No chambers available for spatial preference plots")
        statistics = sorted(statistics, key=lambda item: item[0])
        chambers = [item[0] for item in statistics]
        occupancy = np.stack([item[1] for item in statistics], axis=1)
        preference = np.stack([item[2] for item in statistics], axis=1)
        finite = occupancy[np.isfinite(occupancy)]
        colour_max = max(1., float(finite.max())) if finite.size else 100.
        palette = plt.get_cmap('Blues').copy()
        palette.set_bad('#DDDDDD')
        fig, axes = plt.subplots(len(windows), 2, squeeze=False,
                                 figsize=(12, len(windows) * max(3, len(chambers) * .35 + 1)),
                                 gridspec_kw={"width_ratios": [4, 1]}, constrained_layout=True)
        try:
            for index, window in enumerate(windows):
                left, right = axes[index]
                title = window['label']
                if window['start'] is not None:
                    title += f" | {window['start']:g}-{window['end']:g} s (planned)"
                if 'flows' in window:
                    flows = window['flows']
                    title += (f"\nO2 {flows['o2']:g} | Anaesthetic {flows['anes']:g} | "
                              f"N2 {flows['n2']:g} sccm")
                height_image = left.imshow(occupancy[index], aspect="auto", origin="lower",
                    extent=(0, 1, -.5, len(chambers) - .5), cmap=palette, vmin=0, vmax=colour_max)
                left.set_xlabel("Normalised height (0 = bottom, 1 = top)")
                left.set_title(title, fontsize=10)
                preference_image = right.imshow(preference[index], aspect="auto", origin="lower",
                                                cmap=palette, vmin=0, vmax=100)
                right.set_xticks([0, 1], ["Floor", "Ceiling"])
                right.set_title("Spatial preference", fontsize=10)
                for axis in (left, right):
                    axis.set_yticks(range(len(chambers)))
                    axis.set_yticklabels([f"CH {item}" for item in chambers], fontsize=8)
            fig.colorbar(height_image, ax=list(axes[:, 0]), label="Time per height bin (%)", shrink=.8)
            fig.colorbar(preference_image, ax=list(axes[:, 1]), label="Time proportion (%)", shrink=.8)
            fig.suptitle('Spatial preference by protocol step | Grey = no valid observations', fontsize=11)
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200)
        finally:
            plt.close(fig)
        return save_path

    def render_overlay_video(self, cleaned_df, input_video_path, output_video_path,
                             scale=None, target_width=1080, progress_callback=None,
                             cancel_callback=None):
        """Render valid cleaned coordinates; missing clean points never fall back to raw detections."""
        from core.tracking.video_overlay import render_trajectory_overlay
        return render_trajectory_overlay(
            cleaned_df,input_video_path,output_video_path,fps_fallback=self.fps,
            scale=scale,target_width=target_width,progress_callback=progress_callback,
            cancel_callback=cancel_callback)
