'Industrial camera acquisition with queued PFS updates, dynamic writer dimensions, and cooperative shutdown.'

from __future__ import annotations
from core import schema
from pathlib import Path
import time
import csv
import cv2
import numpy as np

from PyQt5.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

try:
    from pypylon import pylon
    PYLON_INSTALLED = True
except ImportError:
    PYLON_INSTALLED = False



class RobustCameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str, str, str)  # State, title, details.
    io_status_signal = pyqtSignal(bool)
    recording_failed = pyqtSignal(str)

    def __init__(self, pfs_path: str = "", serial_worker=None):
        super().__init__()
        self.serial_worker = serial_worker
        self._telemetry_file = None
        self._telemetry_writer = None
        self._frame_count = 0
        self._rec_start_time = 0.0
        self.pfs_path = pfs_path
        self._running = False
        self._recording = False
        self.camera = None
        self.video_writer = None

        self._mutex = QMutex()
        self._pending_pfs: str | None = None
        self._rec_path: str | None = None
        self._rec_fps: float = 30.0
        self._writer_shape = None

        self._stop_requested = False
        self._last_preview = 0.0
        self.frame_width = 1920
        self.frame_height = 1080

    def set_pfs_path_safely(self, path: str):
        'Queue camera settings for application between grabs.'
        with QMutexLocker(self._mutex):
            if self._recording:
                self.status_changed.emit("OK", "PFS not applied", "Stop recording before changing PFS")
                return
            if not Path(path).is_file():
                self.status_changed.emit("PENDING", "PFS not applied", "Camera settings file does not exist")
                return
            self._pending_pfs = path

    @property
    def recording(self):
        return self._recording

    apply_pfs_safely = set_pfs_path_safely

    def start_recording(self, video_path: str, fps: float = 30.0, telemetry_csv_path=None, start_time=None):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Recording fps must be positive and finite")
        if start_time is not None and not np.isfinite(float(start_time)):
            raise ValueError("Recording start time must be finite")
        with QMutexLocker(self._mutex):
            if self._recording:
                raise RuntimeError("Recording already active")
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            if self._pending_pfs:
                raise RuntimeError("Wait for pending camera settings to be applied")
            self._writer_shape = None
            self._frame_count = 0
            self._rec_start_time = time.monotonic() if start_time is None else float(start_time)
            if telemetry_csv_path:
                Path(telemetry_csv_path).parent.mkdir(parents=True, exist_ok=True)
                self._telemetry_file = open(telemetry_csv_path, "w", newline="", encoding="utf-8")
                self._telemetry_writer = csv.writer(self._telemetry_file)
                self._telemetry_writer.writerow([schema.label(name) for name in ("frame", "elapsed", "video_time", "o2_flow", "anes_flow", "n2_flow", "o2_sp", "anes_sp", "n2_sp")])
            self._rec_path = str(video_path)
            self._rec_fps = float(fps)
            self._recording = True

    def stop_recording(self):
        with QMutexLocker(self._mutex):
            self._recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self._telemetry_file:
                self._telemetry_file.close()
                self._telemetry_file = None
                self._telemetry_writer = None
            return self._frame_count

    def run(self):
        if self._stop_requested:
            return
        self._running = True
        tl_factory = pylon.TlFactory.GetInstance() if PYLON_INSTALLED else None

        converter = pylon.ImageFormatConverter() if PYLON_INSTALLED else None
        if converter is not None:
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        while self._running:
            # 1. Connect to the camera.
            if self.camera is None or not self.camera.IsOpen():
                if not self._connect_camera(tl_factory):
                    time.sleep(1.5)
                    continue

            # 2. Apply queued PFS settings between grabs.
            with QMutexLocker(self._mutex):
                pfs_to_load = self._pending_pfs if not self._recording else None
                if pfs_to_load:
                    self._pending_pfs = None

            if pfs_to_load and Path(pfs_to_load).is_file():
                self._apply_pfs_internal(pfs_to_load)

            # 3. Retrieve frames.
            try:
                if not self.camera.IsGrabbing():
                    self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                    self.status_changed.emit(
                        "OK",
                        "Camera: Basler Online",
                        f"Hardware ready | Resolution: {self.frame_width}x{self.frame_height}",
                    )

                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                try:
                    if grab_result.GrabSucceeded():
                        frame = converter.Convert(grab_result).GetArray().copy()

                        # Poll the Line1 hardware input.
                        line_high = False
                        try:
                            self.camera.LineSelector.SetValue("Line1")
                            line_high = bool(self.camera.LineStatus.GetValue())
                        except Exception:
                            pass
                        captured_at = time.monotonic()
                        if captured_at - self._last_preview >= 1 / 30:
                            self.io_status_signal.emit(line_high)
                            self.frame_ready.emit(frame)
                            self._last_preview = captured_at

                        # Write recording frames under the mutex.
                        try:
                            with QMutexLocker(self._mutex):
                                if self._recording:
                                    self._write_frame_safely(frame, captured_at)
                        except Exception as exc:
                            # Release the lock before closing writers; keep preview alive.
                            self.stop_recording()
                            self.recording_failed.emit(str(exc))

                finally:
                    grab_result.Release()
            except Exception as e:
                # Report interruptions only while acquisition is active.
                if self._running:
                    self.status_changed.emit("PENDING", "Camera: Reconnecting", str(e))
                    if self.recording:
                        self.stop_recording()
                        self.recording_failed.emit(f"Camera acquisition interrupted: {e}")
                    self._close_camera_resources()
                    time.sleep(1.0)

        self.stop_recording()
        self._close_camera_resources()

    def _connect_camera(self, tl_factory) -> bool:
        if not PYLON_INSTALLED:
            self.status_changed.emit("OFFLINE", "pypylon Not Installed", "Install pypylon to enable Basler acquisition")
            return False

        try:
            devices = tl_factory.EnumerateDevices()
        except Exception as exc:
            self.status_changed.emit("PENDING", "Camera discovery failed", str(exc))
            return False
        if not devices:
            self.status_changed.emit("PENDING", "Waiting for Camera", "No Basler camera found; check power and cables")
            return False

        try:
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()
            nodemap = self.camera.GetNodeMap()

            if self.pfs_path and Path(self.pfs_path).is_file():
                pylon.FeaturePersistence.Load(self.pfs_path, nodemap)

            self.frame_width = int(nodemap.GetNode("Width").GetValue())
            self.frame_height = int(nodemap.GetNode("Height").GetValue())
            return True
        except Exception as e:
            self.status_changed.emit("PENDING", "Camera Busy or Open Failed", str(e))
            self._close_camera_resources()
            return False

    def _apply_pfs_internal(self, pfs_file: str):
        try:
            was_grabbing = self.camera.IsGrabbing()
            if was_grabbing:
                self.camera.StopGrabbing()

            nodemap = self.camera.GetNodeMap()
            pylon.FeaturePersistence.Load(pfs_file, nodemap)
            self.frame_width = int(nodemap.GetNode("Width").GetValue())
            self.frame_height = int(nodemap.GetNode("Height").GetValue())

            if was_grabbing:
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.pfs_path = pfs_file
            self.status_changed.emit(
                "OK", "PFS Loaded", f"Settings applied | Resolution: {self.frame_width}x{self.frame_height}"
            )
        except Exception as e:
            self.status_changed.emit("PENDING", "PFS Apply Failed", str(e))

    def _write_frame_safely(self, frame: np.ndarray, captured_at=None):
        """Write constant-rate output against the same monotonic experiment clock."""
        captured_at = time.monotonic() if captured_at is None else captured_at
        elapsed = max(0.0, captured_at - self._rec_start_time)
        target_count = int(elapsed * self._rec_fps) + 1
        frames_to_write = target_count - self._frame_count
        if frames_to_write <= 0:
            return
        if frames_to_write > max(1, int(self._rec_fps * 2)):
            raise RuntimeError("Recording stalled for over two seconds; acquisition stopped")
        if frame.dtype != np.uint8:
            raise ValueError("Recording requires an 8-bit frame")
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Recording requires a grayscale or BGR frame")
        fh, fw = frame.shape[:2]
        if self.video_writer is None and self._rec_path:
            self._writer_shape = (fh, fw)
            self.video_writer = cv2.VideoWriter(
                self._rec_path, cv2.VideoWriter_fourcc(*"mp4v"), self._rec_fps, (fw, fh), True)
        if self._writer_shape != (fh, fw):
            raise RuntimeError("Frame size changed during recording")
        if not self.video_writer or not self.video_writer.isOpened():
            raise RuntimeError("VideoWriter could not open the recording path")
        snap = self.serial_worker.get_telemetry_snapshot() if self.serial_worker else {}
        values = [snap.get(uid, {}).get(key, float("nan")) for key in ("flow", "sp") for uid in "ABC"]
        for _ in range(frames_to_write):
            self.video_writer.write(frame)
            if self._telemetry_writer:
                self._telemetry_writer.writerow([
                    self._frame_count, elapsed, self._frame_count / self._rec_fps, *values])
            self._frame_count += 1

    def _close_camera_resources(self):
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
        except Exception:
            pass
        self.camera = None

    def terminate_worker(self):
        'Request shutdown and wait for worker-owned resource cleanup.'
        self._stop_requested = True
        self._running = False
        self.stop_recording()
        self.wait()

SynchronisedCameraWorker = RobustCameraWorker
