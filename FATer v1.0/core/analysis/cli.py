"""Headless entry point using the same analysis engine as the desktop UI."""
import argparse
from pathlib import Path
from .anaesthesia_analyser import AnaesthesiaParams
from .evaluation_service import evaluate_dataset
from core.session.file_matcher import discover_and_match_sessions


def main():
    parser = argparse.ArgumentParser(description='FATer trajectory evaluation')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', type=Path)
    source.add_argument('--dir', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--anaesthetic-gas-onset', '--gas-onset', dest='anaesthetic_onset', type=float, default=300)
    parser.add_argument('--still-sec', type=float, default=120)
    parser.add_argument('--speed-thresh', type=float, default=0.1)
    parser.add_argument('--sedation-ratio', type=float, default=0.05)
    parser.add_argument('--drop-thresh', type=float, default=0.25)
    parser.add_argument('--video', type=Path, help='Source video for an annotated overlay')
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--summary-only', action='store_true', help='Export summary without frame CSV or plots')
    args = parser.parse_args()
    if args.overlay and (not args.csv or not args.video):
        parser.error('--overlay requires --csv and --video')
    inputs = [args.csv] if args.csv else [Path(d['csv']) for d in discover_and_match_sessions([args.dir]).values() if d.get('csv')]
    if not inputs:
        parser.error('No trajectory CSV files found')
    for raw in inputs:
        params = AnaesthesiaParams(fps=args.fps, anaesthetic_onset=args.anaesthetic_onset,
                                  still_window=args.still_sec, still_speed=args.speed_thresh,
                                  drop_height=args.drop_thresh, sedation_ratio=args.sedation_ratio)
        _, output, warnings = evaluate_dataset(raw, params, plots=not args.no_plots,
            output_dir=args.out, overlay_video=args.video if args.overlay else None,
            summary_only=args.summary_only)
        print(f'Exported: {output}')
        for warning in warnings:
            print(f'Plot warning: {warning}')


if __name__ == '__main__':
    main()
