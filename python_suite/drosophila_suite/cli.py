"""
Command-Line Interface (CLI)
============================
Headless execution script for high-throughput cluster and server batch jobs.
Supports deep recursive directory traversal, automatic CSV/Video pairing,
adaptive 95% boundary detection, and relative drop-ratio kinetics.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

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

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CSV_EXTS = {".csv"}


def clean_session_base_name(filename: str) -> str:
    """提取纯粹的实验会话唯一标识 Base Name，去除中间衍生后缀"""
    import re
    stem = Path(filename).stem
    pattern = r"(_tracked|_raw|_cleaned|_v\d+|_overlay|_filtered|_kinematics)$"
    while True:
        new_stem = re.sub(pattern, "", stem, flags=re.IGNORECASE)
        if new_stem == stem:
            break
        stem = new_stem
    return stem.strip()


def scan_and_pair_directory_recursive(root_dir: str) -> Dict[str, Dict[str, Any]]:
    root_path = os.path.abspath(root_dir)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Directory not found: {root_path}")

    artifact_keywords = [
        "_results_summary", "_frames_meta", "_summary", "_meta",
        "_activity_position", "_kymograph_norm", "_survival_kinetics",
        "_overlay", "_cleaned_overlay"
    ]

    sessions: Dict[str, Dict[str, Any]] = {}
    print(f">> [Scanner] Deep scanning directory: {root_path}")

    for dirpath, _, filenames in os.walk(root_path):
        # 从完整路径中提取 Session_YYYYMMDD_HHMMSS 标识
        session_tag = ""
        for part in Path(dirpath).parts:
            m = re.search(r"(Session_\d{8}_\d{6})", part, flags=re.IGNORECASE)
            if m:
                session_tag = m.group(1)
                break
            elif part.lower().startswith("session_"):
                session_tag = part
                break

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VIDEO_EXTS and ext not in CSV_EXTS:
                continue

            fname_lower = fname.lower()
            if any(kw in fname_lower for kw in artifact_keywords):
                continue

            full_path = os.path.join(dirpath, fname)
            base_stem = clean_session_base_name(fname)
            uid = f"{session_tag}_{base_stem}" if session_tag else base_stem

            if uid not in sessions:
                sessions[uid] = {
                    "session_id": uid,
                    "session_tag": session_tag,
                    "base_name": base_stem,
                    "folder_dir": dirpath,
                    "video_path": None,
                    "cleaned_csv": None,
                    "raw_csv": None,
                }

            if ext in VIDEO_EXTS:
                sessions[uid]["video_path"] = full_path
            elif ext in CSV_EXTS:
                if "_cleaned" in fname_lower:
                    sessions[uid]["cleaned_csv"] = full_path
                elif "_raw" in fname_lower:
                    sessions[uid]["raw_csv"] = full_path
                else:
                    if not sessions[uid]["cleaned_csv"] and not sessions[uid]["raw_csv"]:
                        sessions[uid]["raw_csv"] = full_path

    valid_sessions = {uid: s for uid, s in sessions.items() if s["cleaned_csv"] or s["raw_csv"] or s["video_path"]}
    print(f">> [Scanner] Successfully paired {len(valid_sessions)} distinct sessions.")
    return valid_sessions


def main():
    parser = argparse.ArgumentParser(
        description="Drosophila Behavior & Anesthesia Phenotyping Pipeline (High-Throughput CLI)"
    )
    defaults = PipelineConfig()
    # 输入与输出
    parser.add_argument("--csv", type=str, help="Path to single input tracking CSV")
    parser.add_argument("--video", type=str, help="Path to single input video file")
    parser.add_argument("--dir", type=str, help="Root directory for deep recursive batch scanning")
    parser.add_argument("--out", type=str, help="Output directory (defaults to each input session folder)")

    # 视频与基础参数
    parser.add_argument("--fps", type=float, default=30.0, help="Camera frame rate (default: 30.0)")

    # 最新动力学与气路参数 (严格统一)
    parser.add_argument("--gas-onset", type=float, default=300.0, help="Anesthesia gas delivery onset in seconds (default: 300.0s)")
    parser.add_argument("--mixing-sec", type=float, default=10.0, help="Gas mixing & dead-volume characteristic time (default: 10.0s)")
    parser.add_argument("--still-sec", type=float, default=120.0, help="Floor quiescent window for anaesthesia (default: 120.0s)")
    parser.add_argument("--speed-thresh", type=float, default=defaults.anesthesia_speed_thresh, help="Absolute speed threshold for immobility in px/s")
    parser.add_argument("--speed-ratio", type=float, default=defaults.sedate_speed_ratio, help="Deceleration ratio relative to baseline")
    parser.add_argument("--drop-height", "--drop-ratio", dest="drop_height", type=float, default=defaults.sedate_drop_speed, help="Absolute normalized height decrease over the drop window (legacy alias: --drop-ratio)")
    parser.add_argument("--conf-thresh", type=float, default=0.60, help="Sedation confidence threshold (default: 0.60)")

    # 自适应网格几何参数
    parser.add_argument("--rows", type=int, default=4, help="Grid chamber rows (default: 4)")
    parser.add_argument("--cols", type=int, default=2, help="Grid chamber columns (default: 2)")
    parser.add_argument("--order", type=str, default="column_first", choices=["column_first", "row_first"])

    # 控制开关
    parser.add_argument("--force-retrack", action="store_true", help="Force vision tracking even if CSV exists")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation")
    parser.add_argument("--overlay", action="store_true", help="Render annotated overlay video")
    parser.add_argument("--gui", action="store_true", help="Launch PyQt6 Desktop GUI")

    args = parser.parse_args()

    # GUI 启动入口
    if args.gui:
        try:
            from .gui_app import run_gui
        except (ImportError, ValueError):
            from drosophila_suite.gui_app import run_gui
        run_gui()
        return

    # 初始化全局配置
    config = PipelineConfig(
        fps=args.fps,
        grid_rows=args.rows,
        grid_cols=args.cols,
        grid_order=args.order,
        anesthesia_onset_time=args.gas_onset,
        gas_mixing_window_sec=args.mixing_sec,
        anesthesia_still_sec=args.still_sec,
        anesthesia_speed_thresh=args.speed_thresh,
        sedate_speed_ratio=args.speed_ratio,
        sedate_drop_speed=args.drop_height,
        sedation_conf_thresh=args.conf_thresh,
    )

    pipeline = DrosophilaBehaviorPipeline(config=config)

    # 批处理模式 (支持深层多级递归)
    if args.dir:
        sessions = scan_and_pair_directory_recursive(args.dir)
        total = len(sessions)
        if total == 0:
            print(">> No valid experiment sessions found.")
            return

        for idx, (session_id, paths) in enumerate(sessions.items(), start=1):
            print(f"\n=======================================================")
            print(f">> [{idx}/{total}] Processing Session: {session_id}")
            print(f"   Folder: {paths['folder_dir']}")

            # 决定首选 CSV（优先使用 cleaned_csv 实现极速跳步）
            target_csv = None
            if not args.force_retrack:
                target_csv = paths["cleaned_csv"] or paths["raw_csv"]

            target_vid = paths["video_path"]
            out_folder = args.out if args.out else paths["folder_dir"]

            try:
                res = pipeline.process_file_pair(
                    csv_path=target_csv,
                    video_path=target_vid,
                    base_name=session_id,
                    output_dir=out_folder,
                    anesthesia_onset_time=args.gas_onset,
                    save_raw_csv=True,
                    save_cleaned_csv=True,
                    generate_plots=not args.no_plots,
                    render_video_overlay=args.overlay
                )
                print(f"   Done in {res.get('elapsed_sec', 0):.1f}s -> {res.get('summary_csv_path')}")
            except Exception as e:
                print(f"   Error processing {session_id}: {e}")

    # 单会话模式
    elif args.csv or args.video:
        print(f">> Processing single session...")
        target_name = clean_session_base_name(args.csv or args.video)
        out_folder = args.out or os.path.dirname(os.path.abspath(args.csv or args.video))

        res = pipeline.process_file_pair(
            csv_path=args.csv,
            video_path=args.video,
            base_name=target_name,
            output_dir=out_folder,
            anesthesia_onset_time=args.gas_onset,
            save_raw_csv=True,
            save_cleaned_csv=True,
            generate_plots=not args.no_plots,
            render_video_overlay=args.overlay
        )
        print(f">> Completed in {res.get('elapsed_sec', 0):.1f}s")
        print(f">> Phenotype Summary: {res.get('summary_csv_path')}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()