"""Worker-owned import, preview decoding, and sequential tracking exports."""
from __future__ import annotations
from core import schema

from dataclasses import asdict
from pathlib import Path
import json
import math
import threading
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.session.file_matcher import discover_tracking_videos, tracking_output_path
from core.session.layout import OutputTransaction, SessionPaths, dataset_name
from core.analysis.kinematic_cleaner import KinematicCleaner
from .models import VideoMetadata, VideoTask, TrackingResult
from .calibration_cache import load_calibration
from .trajectory_layer import TrajectoryLayer
from .streaming_csv import DetectionCSVWriter, merge_position_csvs
from .tracker_engine import FlyTrackerEngine
from .legacy_tracker import build_median_background, RobustFlyTracker
from .video_overlay import render_trajectory_overlay


def read_video_task(data):
    video = Path(data["video"]).resolve()
    stamp = video.stat()
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video.name}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or frames <= 0:
            raise ValueError(f"Invalid video metadata: {video.name}")
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
    finally:
        capture.release()
    task = VideoTask(str(video).casefold(), video,
                     Path(data.get("session_dir") or video.parent).resolve(),
                     Path(data["telemetry_csv"]) if data.get("telemetry_csv") else None,
                     VideoMetadata(width, height, fps, frames, stamp.st_size, stamp.st_mtime_ns))
    task.calibration = load_calibration(task)
    return task


class ImportWorker(QThread):
    result_ready = pyqtSignal(object)
    message = pyqtSignal(str)

    def __init__(self, folders=(), recordings=(), parent=None):
        super().__init__(parent)
        self.folders, self.recordings = tuple(folders), tuple(recordings)

    def run(self):
        tasks = []
        try:
            discovered = discover_tracking_videos(self.folders, self.isInterruptionRequested)
            for data in self.recordings:
                discovered[str(Path(data["video"]).resolve()).casefold()] = data
            for data in discovered.values():
                if self.isInterruptionRequested():
                    return
                try:
                    tasks.append(read_video_task(data))
                except (OSError, ValueError, cv2.error) as exc:
                    self.message.emit(f"Video skipped: {exc}")
            self.result_ready.emit(tasks)
        except InterruptedError:
            pass
        except Exception as exc:
            self.message.emit(f"Folder import failed: {exc}")


class PreviewWorker(QThread):
    result_ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending = None
        self._latest = 0

    def submit(self, request):
        with self._condition:
            self._latest = request.token
            self._pending = request
            self._condition.notify()

    def cancel_pending(self):
        with self._condition:
            self._latest = -1
            self._pending = None
            self._condition.notify()

    def stop(self):
        self.requestInterruption()
        with self._condition:
            self._condition.notify()

    def run(self):
        capture, current_video = None, None
        background, cache_key = None, None
        try:
            while not self.isInterruptionRequested():
                with self._condition:
                    while self._pending is None and not self.isInterruptionRequested():
                        self._condition.wait()
                    request, self._pending = self._pending, None
                if self.isInterruptionRequested():
                    break
                cancelled = lambda: self.isInterruptionRequested() or request.token != self._latest
                try:
                    video = str(request.video_path)
                    if current_video != video:
                        if capture is not None:
                            capture.release()
                        capture = cv2.VideoCapture(video)
                        current_video = video
                    if not capture.isOpened():
                        raise ValueError("Cannot open the selected video.")
                    capture.set(cv2.CAP_PROP_POS_FRAMES, request.frame)
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError("Could not decode the selected frame.")
                    if cache_key != request.cache_key:
                        background, cache_key = None, request.cache_key
                    if request.mode != "original" and background is None:
                        options = request.options
                        built = build_median_background(
                            video, options.background_samples, cancelled,
                            sample_start=options.background_start, sample_end=options.background_end)
                        if cancelled():
                            continue
                        background = built
                    if cancelled():
                        continue
                    centroids = []
                    if request.mode == "centroids" and request.rois:
                        tracker = RobustFlyTracker(request.rois, median_bg=background,
                                                   diff_thresh=14, temporal=False)
                        try:
                            records, _ = tracker.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), request.frame)
                            centroids = [(r['x'], r['y']) for r in records
                                         if np.isfinite(r['x']) and np.isfinite(r['y'])]
                        finally:
                            tracker.close()
                    if not cancelled():
                        self.result_ready.emit(request.token, {
                            "video": video, "original": frame,
                            "display": background if request.mode == "background" else frame,
                            "centroids": centroids, "background": background, "cache_key": cache_key})
                except InterruptedError:
                    pass
                except Exception as exc:
                    if not cancelled():
                        self.failed.emit(request.token, str(exc))
        finally:
            if capture is not None:
                capture.release()


class TrackingWorker(QThread):
    progress = pyqtSignal(str, int, int, str, int, int, float)
    result_ready = pyqtSignal(str, object)
    error_signal = pyqtSignal(str)

    def __init__(self, jobs, parent=None):
        super().__init__(parent)
        self.jobs = tuple(jobs)
        self._cancel = threading.Event()
        self._preview_lock = threading.Lock()
        self._latest_preview = None
        self._preview_enabled = False
        self._next_preview_time = 0.
        self._trajectory_key = None
        self._trajectory_layer = None

    def set_preview_enabled(self, enabled):
        """Toggle display publishing without pausing tracking or exporting."""
        with self._preview_lock:
            self._preview_enabled = bool(enabled)
            self._next_preview_time = 0.
            if not enabled:
                self._latest_preview = None

    def cancel(self):
        self._cancel.set()
        with self._preview_lock:
            self._latest_preview = None

    def take_preview(self):
        """Consume one latest frame without queuing images in the GUI thread."""
        with self._preview_lock:
            preview, self._latest_preview = self._latest_preview, None
        return preview

    def _publish_preview(self, job, frame_index, frame, records):
        if self._cancel.is_set():
            return
        if self._trajectory_key != job.key:
            self._trajectory_key = job.key
            self._trajectory_layer = TrajectoryLayer(frame.shape)
        self._trajectory_layer.update(frame_index, records)
        # VideoCapture creates a fresh frame; neither thread modifies its pixels.
        with self._preview_lock:
            if self._cancel.is_set() or not self._preview_enabled:
                return
            now = time.monotonic()
            if now < self._next_preview_time:
                return
            self._next_preview_time = now + 1. / 18.
            centroids = tuple((record['x'], record['y']) for record in records
                              if np.isfinite(record['x']) and np.isfinite(record['y']))
            self._latest_preview = (job.key, frame_index, frame,
                                    job.calibration.rois, centroids, self._trajectory_layer.snapshot())

    def _save_csv(self, data, path):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            exported = schema.output(data)
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                for start in range(0, max(1, len(exported)), 50000):
                    if self._cancel.is_set():
                        raise InterruptedError("Tracking export cancelled.")
                    exported.iloc[start:start + 50000].to_csv(
                        stream, index=False, header=start == 0)
            if self._cancel.is_set():
                raise InterruptedError("Tracking export cancelled.")
            temporary.replace(path)
        finally:
            if temporary.is_file():
                temporary.unlink()

    @staticmethod
    def _save_text(path, content):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)
        finally:
            if temporary.is_file():
                temporary.unlink()

    def run(self):
        total = len(self.jobs)
        for index, job in enumerate(self.jobs, 1):
            if self._cancel.is_set():
                for pending in self.jobs[index - 1:]:
                    self.result_ready.emit(pending.key,
                        TrackingResult(status="Cancelled", calibration=pending.calibration))
                break
            self._process_job(job, index, total)
        self.jobs = ()

    def _process_job(self, job, index, total):
        result = TrackingResult(calibration=job.calibration)
        output = None
        staging = None
        committed = False
        events = []
        started = time.monotonic()
        phase = "Preparing"

        def on_phase(value):
            nonlocal phase
            phase = value
            events.append(f"{time.monotonic()-started:.3f}s {value}")
            self.progress.emit(job.key, index, total, value, 0, job.metadata.frames, time.monotonic()-started)

        def on_progress(current, count):
            self.progress.emit(job.key, index, total, phase, current, count, time.monotonic()-started)

        try:
            if self._cancel.is_set():
                raise InterruptedError("Tracking cancelled.")
            job.options.validate()
            job.calibration.validate(job.metadata.shape)
            stamp = job.video_path.stat()
            if (stamp.st_size, stamp.st_mtime_ns) != (job.metadata.size, job.metadata.modified_ns):
                raise ValueError("Video changed after import. Import its folder again.")
            output = tracking_output_path(job.video_path, job.session_dir)
            paths = SessionPaths(job.session_dir)
            paths.ensure()
            staging = OutputTransaction(paths.root)
            prefix = dataset_name(output)
            engine = FlyTrackerEngine()
            # Explicit direct connections keep worker-local callbacks in this thread.
            from PyQt5.QtCore import Qt
            engine.phase_changed.connect(on_phase, Qt.DirectConnection)
            engine.progress_updated.connect(
                lambda current, percent: on_progress(current, job.metadata.frames), Qt.DirectConnection)
            engine.error_occurred.connect(events.append, Qt.DirectConnection)
            try:
                options = job.options
                detection_path = output.with_name(prefix + '_raw.csv')
                raw_stage = staging.stage(detection_path)
                with DetectionCSVWriter(raw_stage, staging.directory / 'chambers', self._cancel.is_set) as sink:
                    data = engine.run_tracking(
                        str(job.video_path), job.calibration.rois,
                        background_samples=options.background_samples,
                        background_start=options.background_start,
                        background_end=options.background_end,
                        interpolate=False, cancel_callback=self._cancel.is_set,
                        background=job.background,
                        selection_options=options.selection,
                        record_sink=sink.write_frame, collect_records=False,
                        frame_callback=lambda frame_index, frame, records:
                            self._publish_preview(job, frame_index, frame, records))
            finally:
                engine.phase_changed.disconnect()
                engine.progress_updated.disconnect()
                engine.error_occurred.disconnect()
            if self._cancel.is_set():
                raise InterruptedError("Tracking cancelled.")
            if data.empty and not data.attrs.get("processed_frames"):
                raise ValueError(events[-1] if events else "Tracking returned no decoded frames.")
            result.included_chamber_ids = tuple(data.attrs.get("included_chamber_ids", ()))
            result.excluded_chamber_ids = tuple(data.attrs.get("excluded_chamber_ids", ()))
            result.raw_csv = detection_path
            events.append(f'Detections written incrementally: {detection_path}')
            on_phase("Cleaning")
            cleaner = KinematicCleaner(fps=job.metadata.fps)
            cleaned_paths, clean_ids = [], []
            empty_header = [schema.label(name) for name in (*sink.fields,
                'x_clean', 'y_clean', 'height', 'position', 'speed', 'activity',
                'is_rejected', 'is_filled', 'coordinate_status', 'processing_method')]
            for chamber, source in sorted(sink.chamber_paths.items()):
                if self._cancel.is_set():
                    raise InterruptedError('Cleaning cancelled.')
                if sink.valid_counts[chamber] < 15:
                    source.unlink()
                    continue
                on_phase(f'Cleaning Chamber {chamber}')
                chamber_data = schema.read_csv(source)
                cleaned = cleaner.clean_trajectory(chamber_data,
                    interpolate=options.interpolate, max_gap=options.max_gap,
                    gap_speed=options.gap_speed, cancel_callback=self._cancel.is_set)
                del chamber_data
                source.unlink()
                if not cleaned.empty:
                    cleaned['processing_method'] = 'edge-reacquisition-segmented-sg'
                    chamber_output = staging.directory / 'chambers' / f'cleaned_{chamber}.csv'
                    self._save_csv(cleaned, chamber_output)
                    cleaned_paths.append(chamber_output)
                    clean_ids.append(chamber)
                del cleaned
            if self._cancel.is_set():
                raise InterruptedError("Cleaning cancelled.")
            on_phase("Saving Cleaned Position CSV")
            position_stage = staging.stage(output)
            cleaned_count = merge_position_csvs(cleaned_paths, position_stage, empty_header, self._cancel.is_set)
            for chamber_output in cleaned_paths:
                chamber_output.unlink()
            result.position_csv = output
            clean_ids = tuple(clean_ids)
            result.excluded_chamber_ids = tuple(sorted(set(result.excluded_chamber_ids)
                | (set(result.included_chamber_ids) - set(clean_ids))))
            result.included_chamber_ids = clean_ids
            events.append(f'Cleaning excluded chambers (<15 valid x/y samples): {result.excluded_chamber_ids}')
            result.status = "No Fly Detected" if not cleaned_count else "Done"
            if options.render_overlay and cleaned_count:
                on_phase("Rendering Overlay")
                try:
                    target = paths.analysis / (prefix + "_cleaned_overlay.mp4")
                    overlay_stage = staging.stage(target)
                    render_trajectory_overlay(
                        position_stage, job.video_path, overlay_stage,
                        fps_fallback=job.metadata.fps, scale=1.0,
                        progress_callback=on_progress, cancel_callback=self._cancel.is_set)
                    result.overlay_video = target
                    result.overlay_status = "Done"
                except InterruptedError:
                    result.overlay_status = "Cancelled"
                    raise
                except Exception as exc:
                    if overlay_stage.is_file():
                        overlay_stage.unlink()
                    result.overlay_status, result.overlay_error = "Failed", str(exc)
                    self.error_signal.emit(f"{job.video_path.name}: overlay failed: {exc}")
            elif options.render_overlay:
                result.overlay_status = "No Fly Detected"
        except InterruptedError as exc:
            result.status = "Cancelled"
            events.append(str(exc))
        except Exception as exc:
            result.status, result.error = "Failed", str(exc)
            self.error_signal.emit(f"{job.video_path.name}: {exc}")
        finally:
            if self._cancel.is_set():
                result.status = "Cancelled"
            if output is not None and staging is not None and result.status in ("Done", "No Fly Detected"):
                try:
                    result.config_path = paths.logs / (prefix + "_tracking_config.json")
                    config = {"video": str(job.video_path),
                              "telemetry_csv": str(job.telemetry_path) if job.telemetry_path else None,
                              "calibration": asdict(job.calibration),
                              "shared_calibration": job.shared,
                              "tracking_options": asdict(job.options),
                              "tracking_schema_version": 11, "detection_threshold": 14,
                              "storage": "incremental_csv; per_chamber_cleaning; streaming_merge",
                              "candidate_selection": asdict(job.options.selection),
                              "raw_csv": str(detection_path),
                              "raw_contains": "unmodified_detections",
                              "position_csv": str(output),
                              "position_contains": "interpolated_and_cleaned_trajectory",
                              "cleaning_method": "edge-reacquisition-segmented-sg",
                              "cleaning_options": {"max_step": 45, "body_ratio": 0.5,
                                  "body_length": 12, "occlusion_disp_thresh": 60,
                                  "occlusion_var_thresh": 5, "savgol_window": 7,
                                  "savgol_poly": 2, "min_valid": 15,
                                  "remaining_gaps": "rejected_observations_preserved; accepted_runs_only"},
                              "result": asdict(result)}
                    self._save_text(staging.stage(result.config_path),
                                    json.dumps(config, indent=2, default=str))
                    events.extend([f"Included chambers: {result.included_chamber_ids}",
                                   f"Excluded chambers: {result.excluded_chamber_ids}",
                                   f"Status: {result.status}", f"Error: {result.error}",
                                   f"Overlay: {result.overlay_status} {result.overlay_error}"])
                    self._save_text(staging.stage(paths.logs / (prefix + "_tracking.txt")), "\n".join(events) + "\n")
                    if self._cancel.is_set():
                        result.status = "Cancelled"
                    else:
                        on_phase('Saving complete video CSVs to Session')
                        staging.commit(self._cancel.is_set)
                        committed = True
                        on_phase(f'Saved CSVs: {detection_path.name}; {output.name}')
                except InterruptedError:
                    result.status = "Cancelled"
                except (OSError, ValueError, TypeError, RuntimeError) as exc:
                    result.status, result.error = "Failed", str(exc)
                    self.error_signal.emit(f"Tracking results could not be saved: {exc}")
            if not committed:
                result.position_csv = result.raw_csv = result.overlay_video = result.config_path = None
            if staging is not None:
                try:
                    staging.cleanup()
                except (OSError, ValueError) as exc:
                    result.error = f"Temporary tracking files could not be removed: {exc}"
                    self.error_signal.emit(result.error)
            with self._preview_lock:
                self._latest_preview = None
            self.result_ready.emit(job.key, result)
