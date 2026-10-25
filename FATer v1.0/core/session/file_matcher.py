"""Deterministic video discovery, trajectory matching, and collision-free outputs."""
from core import schema
from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import json
import re
from .layout import SessionPaths, session_root, temporary_path, tracking_config_candidates, dataset_name, dataset_identity, experiment_name

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def is_source_video(path):
    path = Path(path)
    name = path.stem.casefold()
    return (path.suffix.casefold() in VIDEO_SUFFIXES
            and not name.endswith(("_cleaned_overlay", "_overlay", ".partial"))
            and ".partial." not in path.name.casefold())


def discover_tracking_videos(paths, cancel_callback=None):
    """Scan folders once, excluding generated videos and duplicate paths."""
    videos = {}
    for value in paths:
        folder = Path(value).resolve()
        if not folder.is_dir():
            continue
        for candidate in folder.rglob("*"):
            if cancel_callback and cancel_callback():
                raise InterruptedError("Folder import cancelled.")
            if not temporary_path(candidate) and is_source_video(candidate) and candidate.is_file():
                video = candidate.resolve()
                videos[str(video).casefold()] = video
    counts = Counter(video.parent for video in videos.values())
    result = {}
    for key, video in sorted(videos.items()):
        candidates = [video.with_name(video.stem + "_telemetry.csv")]
        if counts[video.parent] == 1 or video.stem.casefold() == "raw_video":
            candidates.extend((SessionPaths(session_root(video)).logs / "sync_telemetry.csv",
                               video.parent / "sync_telemetry.csv"))
        telemetry = next((item for item in candidates if item.is_file()), None)
        result[key] = {"video": str(video), "session_dir": session_root(video),
                       "telemetry_csv": str(telemetry) if telemetry else None}
    return result


def tracking_output_path(video_path, session_dir):
    """Name outputs by experiment time; disambiguate multiple source videos."""
    video, session = Path(video_path).resolve(), Path(session_dir).resolve()
    prefix = experiment_name(video)
    peers = [path for path in session.rglob('*') if path.is_file()
             and not temporary_path(path) and is_source_video(path)
             and experiment_name(path) == prefix]
    if len(peers) > 1:
        identity = str(video.relative_to(session)) if video.is_relative_to(session) else str(video)
        prefix += '_' + sha256(identity.casefold().encode('utf-8')).hexdigest()[:6]
    return SessionPaths(session).raw / (prefix + '_position.csv')


def discover_and_match_sessions(paths, cancel_callback=None):
    """Prefer cleaned Position CSV while preserving every independent video dataset."""
    def check():
        if cancel_callback and cancel_callback():
            raise InterruptedError("Dataset import cancelled.")
    files = set()
    for value in paths:
        check()
        path = Path(value).resolve()
        if path.is_dir():
            for item in path.rglob("*.csv"):
                check()
                if item.is_file() and not temporary_path(item):
                    files.add(item)
        elif path.is_file() and path.suffix.casefold() == ".csv":
            if not temporary_path(path):
                files.add(path)
    result, ranks = {}, {}
    for path in sorted(files):
        check()
        stem = path.stem
        if re.search(r"_(telemetry|evaluated|summary|results_summary|frames_meta|detections)$", stem, re.I):
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                fields = {schema.field(name) for name in next(csv.reader(stream))}
                if not {"chamber_id", "frame"}.issubset(fields):
                    continue
        except (OSError, StopIteration, UnicodeError, csv.Error):
            continue
        prefix = dataset_name(path)
        session = session_root(path)
        key = dataset_identity(path)
        is_position = stem.casefold().endswith("_position")
        is_cleaned = stem.casefold().endswith("_cleaned")
        is_raw = stem.casefold().endswith("_raw")
        processed = schema.has_kinematics(fields)
        # A new detection Raw is evidence of a pending Position, never an analysis input.
        pending = is_raw and not processed
        kind = 0 if is_position else 1 if is_cleaned else 2 if processed else 3
        rank = kind * 2 + (0 if path.parent == SessionPaths(session).raw else 1)
        if rank >= ranks.get(key, 99):
            continue
        trajectory = path.with_name(prefix + "_position.csv") if pending else path
        data = {"session_dir": session, "video": None, "csv": str(trajectory),
                "position_csv": str(trajectory) if is_position or pending else None,
                "raw_csv": str(path) if is_raw else None,
                "cleaned_csv": str(path) if is_cleaned else None,
                "telemetry_csv": None, "chamber_rois": None}
        config_path = next((candidate for candidate in tracking_config_candidates(path)
                            if candidate.is_file()), None)
        if config_path is not None:
            try:
                config = schema.normalise_config(json.loads(config_path.read_text(encoding="utf-8")))
                data["video"] = config.get("video")
                telemetry_name = config.get("telemetry_csv")
                if telemetry_name:
                    telemetry_name = Path(telemetry_name).name
                    data["telemetry_csv"] = next((str(item) for item in (
                        SessionPaths(session).logs / telemetry_name, path.parent / telemetry_name,
                        session / "recordings" / telemetry_name) if item.is_file()), None)
                data["calibration"] = config.get("calibration")
            except (OSError, ValueError, TypeError):
                pass
        if data["telemetry_csv"] is None:
            candidates = [path.with_name(prefix + "_telemetry.csv")]
            if prefix.casefold() in ("raw_video", "video"):
                candidates.extend([SessionPaths(session).logs / "sync_telemetry.csv",
                                   session / "recordings" / "sync_telemetry.csv",
                                   path.parent / "sync_telemetry.csv"])
            candidates.append(SessionPaths(session).logs / "sync_telemetry.csv")
            data["telemetry_csv"] = next((str(item) for item in candidates if item.is_file()), None)
        data["raw_csv"] = str(detection_path(trajectory))
        result[key], ranks[key] = data, rank
    return result


def detection_path(source):
    """Locate unmodified detections without mislabelling a legacy processed Raw."""
    source = Path(source)
    session, prefix = session_root(source), dataset_name(source)
    candidates = []
    for config_path in tracking_config_candidates(source):
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
            name = (config.get("raw_csv") if config.get("tracking_schema_version", 0) >= 8
                    else config.get("detection_archive"))
            if name:
                # Stored absolute paths may refer to an experiment before it was copied.
                archive = Path(name)
                candidates.extend(folder / archive.name for folder in
                    (source.parent, SessionPaths(session).raw, SessionPaths(session).analysis))
        except (OSError, ValueError, TypeError):
            pass
    candidates.extend(folder / (prefix + suffix) for folder in
        (source.parent, SessionPaths(session).raw, SessionPaths(session).analysis)
        for suffix in ("_raw.csv", "_detections.csv"))
    processed_candidates = set()
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        try:
            with candidate.open(encoding="utf-8-sig", newline="") as stream:
                fields = next(csv.reader(stream))
            if not schema.has_kinematics(fields):
                return candidate
            processed_candidates.add(candidate.resolve())
        except (OSError, UnicodeError, StopIteration, csv.Error):
            continue
    expected = SessionPaths(session).raw / (prefix + "_raw.csv")
    return expected.with_name(prefix + "_detections.csv") if expected.resolve() in processed_candidates else expected
