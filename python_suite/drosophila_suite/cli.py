"""
Command-Line Interface (CLI)
============================
Headless execution script for high-throughput cluster and server batch jobs.
"""

import argparse
import sys
import os
import glob

try:
    from .pipeline import DrosophilaBehaviorPipeline
    from .models import PipelineConfig
except (ImportError, ValueError):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from drosophila_suite.pipeline import DrosophilaBehaviorPipeline
    from drosophila_suite.models import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Behavior & Anesthesia Phenotyping Pipeline (CLI)"
    )
    parser.add_argument("--csv", type=str, help="Path to input raw tracking CSV")
    parser.add_argument("--video", type=str, help="Path to experiment video file")
    parser.add_argument("--dir", type=str, help="Batch directory containing CSVs or videos")
    parser.add_argument("--out", type=str, help="Output directory for generated artifacts")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera frame rate (default: 30.0)")
    parser.add_argument("--window-sec", type=float, default=120.0, help="Sedation evaluation window duration in seconds (default: 120.0)")
    parser.add_argument("--no-plots", action="store_true", help="Skip scientific figure generation")
    parser.add_argument("--overlay", action="store_true", help="Render annotated overlay video")
    parser.add_argument("--gui", action="store_true", help="Launch PyQt6 Desktop GUI")

    args = parser.parse_args()

    if args.gui:
        from .gui_app import run_gui
        run_gui()
        return

    if not args.csv and not args.video and not args.dir:
        parser.print_help()
        sys.exit(1)

    config = PipelineConfig(
        fps=args.fps,
        anesthesia_window_duration_sec=args.window_sec,
        anesthesia_window_bins=max(1, int(round(args.window_sec / 5.0)))
    )
    pipeline = DrosophilaBehaviorPipeline(config=config)

    if args.dir:
        csv_files = glob.glob(os.path.join(args.dir, "*.csv"))
        print(f">> Found {len(csv_files)} CSV files in {args.dir}")
        for csv_f in csv_files:
            if "_cleaned" in csv_f or "_summary" in csv_f or "_anesthesia" in csv_f:
                continue
            print(f">> Processing: {os.path.basename(csv_f)}...")
            res = pipeline.process_file_pair(
                csv_path=csv_f,
                output_dir=args.out or args.dir,
                generate_plots=not args.no_plots,
                render_video_overlay=args.overlay
            )
            print(f"   Done in {res['elapsed_sec']}s -> {res['summary_csv_path']}")
    else:
        print(f">> Processing single session...")
        res = pipeline.process_file_pair(
            csv_path=args.csv,
            video_path=args.video,
            output_dir=args.out,
            generate_plots=not args.no_plots,
            render_video_overlay=args.overlay
        )
        print(f">> Completed in {res['elapsed_sec']}s")
        print(f">> Cleaned CSV: {res.get('cleaned_csv_path')}")
        print(f">> Phenotype Summary: {res['summary_csv_path']}")
        if res.get("plot_paths"):
            for name, path in res["plot_paths"].items():
                print(f">> Plot ({name}): {path}")


if __name__ == "__main__":
    main()
