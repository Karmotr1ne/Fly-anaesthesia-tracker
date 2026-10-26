"""Read-only session asset checks for the evaluation task table."""
from pathlib import Path
import csv
import json
import math
import re

from core import schema
from .file_matcher import (discover_and_match_sessions,
                           discover_tracking_videos, is_source_video, tracking_output_path, detection_path)
from .layout import SessionPaths, dataset_name, session_root, temporary_path, experiment_name


def _identity(path):
    return str(Path(path).resolve()).casefold()


def _video_for(source, context, videos):
    configured = context.get("video")
    if configured:
        candidate = Path(configured)
        present = next((video for video in videos if _identity(video) == _identity(candidate)), None)
        if present is not None:
            return present
        named = [video for video in videos if video.name.casefold() == candidate.name.casefold()]
        if len(named) == 1:
            return named[0]
    prefix = dataset_name(source)
    exact = [video for video in videos if prefix in (
        video.stem, dataset_name(tracking_output_path(video, session_root(source))))]
    if len(exact) == 1:
        return exact[0]
    timed = [video for video in videos if experiment_name(video) == experiment_name(source)]
    if len(timed) == 1:
        return timed[0]
    # Legacy CSV names may not contain the modern video identity hash.
    # A unique acquisition video is safe; multiple videos stay unmatched.
    if len(videos) == 1 and videos[0].stem.casefold() == 'raw_video':
        return videos[0]
    return None


def discover_evaluation_inputs(paths, cancel_callback=None):
    """Include source videos whose cleaned Position CSV has not yet been produced."""
    paths = tuple(Path(path) for path in paths)
    def check():
        if cancel_callback and cancel_callback():
            raise InterruptedError("Dataset import cancelled.")
    matched = discover_and_match_sessions(paths, cancel_callback)
    videos = discover_tracking_videos((path for path in paths if path.is_dir()), cancel_callback)
    linked = {_identity(data["video"]) for data in matched.values() if data.get("video")}
    for video_key, entry in videos.items():
        check()
        video = Path(entry["video"])
        if video_key in linked:
            continue
        expected = tracking_output_path(video, entry["session_dir"])
        session_videos = [Path(item['video']) for item in videos.values()
                          if item['session_dir'] == entry['session_dir']]
        existing = next((data for data in matched.values()
                         if session_root(Path(data["csv"])) == entry["session_dir"]
                         and _video_for(Path(data["csv"]), data, session_videos) == video), None)
        if existing is not None:
            existing["video"] = str(video)
            linked.add(video_key)
            continue
        matched[_identity(expected)] = {
            **entry, "csv": str(expected), "position_csv": str(expected),
            "raw_csv": str(expected.with_name(dataset_name(expected) + "_raw.csv")), "cleaned_csv": None}
    # A session with only a log or plan must also be visible as incomplete.
    represented = {_identity(data["session_dir"]) for data in matched.values()}
    sessions = set()
    for folder in paths:
        if not folder.is_dir():
            continue
        if folder.name.startswith("Session_") or (folder / "params").is_dir() or (folder / "logs").is_dir():
            sessions.add(folder)
        for path in folder.rglob("Session_*"):
            check()
            if path.is_dir() and not temporary_path(path):
                sessions.add(path)
    for session in sorted(sessions):
        check()
        if _identity(session) in represented:
            continue
        video = SessionPaths(session).raw / "raw_video.mp4"
        expected = tracking_output_path(video, session)
        matched[_identity(expected)] = {
            "session_dir": session, "video": str(video), "csv": str(expected),
            "position_csv": str(expected),
            "raw_csv": str(expected.with_name(dataset_name(expected) + "_raw.csv")),
            "cleaned_csv": None, "telemetry_csv": None}
    return matched


def _snapshot(session):
    logs = sorted(path for folder in ('logs', 'log') for path in (session / folder).rglob('*')
                  if path.is_file() and path.suffix.lower() in ('.txt', '.md', '.log'))
    logs.extend(path for stem in ("acquisition_runtime", "session_log", "experiment_log", "acquisition_log")
                for suffix in (".txt", ".md", ".log")
                if (path := session / (stem + suffix)).is_file())
    logs = [path for path in logs if path.stat().st_size > 0
            and not path.stem.lower().endswith("_tracking")]
    videos = sorted(path for path in session.rglob("*") if path.is_file() and not temporary_path(path) and is_source_video(path))
    expected, protocol_note = None, "Protocol duration unavailable"
    plans = [session / folder / name for folder in ("logs", "params")
             for name in ("protocol_steps.json", "runtime_config.json")]
    for path in plans:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if path.name == "protocol_steps.json":
                durations = [float(step["time"]) for step in value]
                if not durations or any(not math.isfinite(item) or item <= 0 for item in durations):
                    raise ValueError("Invalid protocol step duration")
                duration = sum(durations)
            else:
                duration = float(value["experiment_duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid protocol duration")
            expected, protocol_note = duration, str(path)
            break
        except (OSError, ValueError, KeyError, TypeError):
            protocol_note = f"Invalid duration in {path.name}"
    # Initial plans do not establish the duration after interactive schedule changes.
    changed = False
    for path in logs:
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            if any(re.search(r"Current phase extended|Skip requested|Step \d+ edited|Protocol schedule paused", line)
                   for line in stream):
                changed = True
                break
    if changed:
        expected, protocol_note = None, "Protocol was edited, extended, skipped or paused; verify its executed duration"
    return {"logs": logs, "videos": videos, "expected": expected, "protocol_note": protocol_note}


def _csv_state(path, processed=False):
    if not path.is_file():
        return "Missing"
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = {schema.field(name) for name in next(reader)}
            valid = {"chamber_id", "frame"}.issubset(fields)
            if processed:
                valid = valid and schema.has_kinematics(fields)
            first = next(reader, None)
            return "OK" if valid and first else "Empty" if valid else "Invalid"
    except (OSError, UnicodeError, StopIteration, csv.Error):
        return "Invalid"


def inspect_session_assets(source_path, context=None, cache=None):
    """Check existence, CSV usability and video duration without decoding all frames."""
    import cv2
    source, context = Path(source_path), context or {}
    session = Path(context.get("session_dir") or session_root(source))
    cache = {} if cache is None else cache
    key = _identity(session)
    if key not in cache:
        cache[key] = _snapshot(session)
    snapshot = cache[key]
    log_state = "OK" if snapshot["logs"] else "Missing"
    required_raw = detection_path(source)
    raw_state = _csv_state(required_raw)
    position, position_state = source, _csv_state(source, processed=True)
    video = _video_for(source, context, snapshot["videos"])
    actual = tolerance = None
    if video is None:
        video_state = "Missing" if not snapshot["videos"] else "Unmatched"
    else:
        video_key = ("video", _identity(video))
        if video_key not in cache:
            capture = cv2.VideoCapture(str(video))
            try:
                fps = capture.get(cv2.CAP_PROP_FPS)
                frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
                cache[video_key] = ((frames / fps, max(1.0, 1.0 / fps))
                                    if capture.isOpened() and math.isfinite(fps) and fps > 0
                                    and math.isfinite(frames) and frames > 0 else (None, None))
            finally:
                capture.release()
        actual, tolerance = cache[video_key]
        video_state = "Unreadable" if actual is None else "OK"
    analysis_id = experiment_name(source)
    analysis_dir = SessionPaths(session).analysis
    # Current concise names first; retain compatibility with old Position-prefixed outputs.
    evaluated = next((path for path in (
        analysis_dir / (analysis_id + '_evaluated.csv'),
        analysis_dir / (source.stem + '_evaluated.csv'),
        analysis_dir / ('Session_' + analysis_id + '_' + source.stem + '_evaluated.csv'))
        if path.is_file()), analysis_dir / (analysis_id + '_evaluated.csv'))
    summary = next((path for path in (
        analysis_dir / (analysis_id + '_summary.csv'),
        analysis_dir / (source.stem + '_summary.csv'),
        analysis_dir / ('Session_' + analysis_id + '_' + source.stem + '_summary.csv'))
        if path.is_file()), analysis_dir / (analysis_id + '_summary.csv'))
    summary_valid = False
    if summary.is_file():
        try:
            with summary.open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                fields = {schema.field(name) for name in next(reader)}
                summary_valid = 'chamber_id' in fields and bool(next(reader, None))
        except (OSError, UnicodeError, StopIteration, csv.Error):
            pass
    evaluated_state = _csv_state(evaluated, processed=True)
    analysis_state = ('Done' if evaluated_state == 'OK' and summary_valid else
                      'Summary only' if summary_valid else
                      'Incomplete' if evaluated.exists() or summary.exists() else 'Not analysed')
    complete = position_state == raw_state == video_state == "OK"
    notes = [f"Session: {session}", "Logs: " + (", ".join(str(path) for path in snapshot["logs"]) or "missing"),
             f"Position CSV (cleaned trajectory): {position}",
             f"Raw CSV (unmodified detections): {required_raw}", f"Video: {video or 'unmatched'}", snapshot["protocol_note"]]
    notes.extend((f'Analysis ID: {analysis_id}', f'Evaluated CSV: {evaluated}',
                  f'Summary CSV: {summary}'))
    if snapshot["expected"] is not None:
        notes.append(f"Protocol duration: {snapshot['expected']:.3f} s")
    if actual is not None:
        notes.append(f"Video duration: {actual:.3f} s")
    return {"text": f"Video: {video_state} | Position: {position_state} | Raw: {raw_state} | Analysis: {analysis_state}",
            "analysis_state": analysis_state, "analysis_id": analysis_id,
            "complete": complete, "tooltip": "\n".join(notes)}
