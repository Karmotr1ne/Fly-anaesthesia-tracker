"""FATer desktop entry point and cross-tab coordination."""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCloseEvent, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from core.hardware.acquisition_controller import AcquisitionController
from core.hardware.camera_worker import SynchronisedCameraWorker
from core.hardware.protocol_scheduler import ProtocolScheduler
from core.hardware.serial_mfc_worker import SerialPollingWorker
from core.protocol.flowvision_model import FlowVisionScriptModel, HardwareConfig
from core.session.session_manager import SessionManager
from core.session.layout import SessionPaths
from core.protocol.library import protocol_root, camera_presets
from ui.styles.theme import SCIENTIFIC_QSS, apply_typography, set_device_status, flowvision_state
from ui.tabs.tab_acquisition import AcquisitionTab
from ui.tabs.tab_evaluation import UnifiedEvaluationTab
from ui.tabs.tab_protocol import ProtocolTab
from ui.tabs.tab_tracking import TrackingTab
from ui.widgets.layout_helpers import scroll_panel, install_window_centering

BASE_DIR = Path(__file__).resolve().parent


def application_icon():
    """Load the shared transparent UI asset at native Windows icon sizes."""
    pixmap = QPixmap(str(BASE_DIR / 'ui' / 'icon.png'))
    icon = QIcon()
    if not pixmap.isNull():
        for size in (16, 24, 32, 48, 64, 128, 256):
            icon.addPixmap(pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    return icon


@dataclass
class RecordingContext:
    mode: str
    session_dir: Path
    video: Path
    telemetry: Path
    schedule_attempted: bool = False


class FATerApp(QMainWindow):
    """Connect page-owned workflows without duplicating their execution state."""

    def __init__(self, start_hardware=True):
        super().__init__()
        self.setWindowIcon(application_icon())
        self._closing = False
        self._finishing = False
        self._recording: RecordingContext | None = None
        self.camera_ready = self.serial_ready = False
        self._camera_state = "Offline"
        self._camera_details = ""
        self.config = HardwareConfig()
        presets = camera_presets()
        self.config.pfs_path = str(presets[0] if presets else protocol_root() / 'camera' / '12 chamber.pfs')
        self.session_mgr = SessionManager(str(BASE_DIR / "workspace"))
        self.serial_worker = SerialPollingWorker(
            port_name=self.config.fv_com_port,
            baudrate=self.config.serial_baudrate,
        )
        self.camera_worker = SynchronisedCameraWorker(
            serial_worker=self.serial_worker, pfs_path=self.config.pfs_path,
        )
        self.acquisition_controller = AcquisitionController(self)
        self._build_ui()
        self.tab_acquire.bind_controller(self.acquisition_controller)
        self._bind_signals()
        self.tab_acquire.txt_pfs.setText(self.config.pfs_path)
        self.tab_acquire.txt_com_port.setText(self.config.fv_com_port)
        self._sync_geometry_to_tracking(
            self.tab_protocol.spin_rows.value(), self.tab_protocol.spin_cols.value(),
        )
        self.on_camera_status_changed(
            "PENDING" if start_hardware else "OFFLINE", "Camera",
            "Connecting." if start_hardware else "Hardware startup is disabled.",
        )
        self.on_serial_status_changed(
            False, "FlowVision: Connecting..." if start_hardware else "FlowVision: Disabled",
        )
        if start_hardware:
            self.serial_worker.start()
            self.camera_worker.start()

    def _build_ui(self):
        self.setWindowTitle("FATer - Acquisition & Analysis Workbench")
        self.setMinimumSize(820, 520)
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1080, 680)
        else:
            size = screen.availableGeometry()
            self.resize(min(1080, int(size.width() * .90)),
                        min(680, int(size.height() * .86)))
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 10, 12, 10)
        status = QHBoxLayout()
        self.lbl_cam_status = QLabel("Camera: Offline")
        self.lbl_serial_status = QLabel("FlowVision: Offline")
        set_device_status(self.lbl_cam_status, "Camera", "Offline")
        set_device_status(self.lbl_serial_status, "Air Valve", "Offline")
        status.setSpacing(8)
        status.addWidget(self.lbl_cam_status)
        status.addWidget(self.lbl_serial_status)
        status.addStretch()
        layout.addLayout(status)
        self.tab_widget = QTabWidget()
        self.tab_protocol = ProtocolTab(
            protocols_dir=str(protocol_root()), parent=self,
        )
        self.tab_protocol.hardware_config = self.config
        self.tab_acquire = AcquisitionTab(parent=self)
        self.tab_track = TrackingTab(parent=self)
        self.tab_eval = UnifiedEvaluationTab(parent=self)
        for widget, title in (
            (scroll_panel(self.tab_protocol), "Protocol"),
            (scroll_panel(self.tab_acquire), "Acquisition"),
            (self.tab_track, "Tracking"),
            (scroll_panel(self.tab_eval), "Evaluation"),
        ):
            self.tab_widget.addTab(widget, title)
        self.tab_widget.setCurrentIndex(1)
        layout.addWidget(self.tab_widget, 1)

    def _bind_signals(self):
        acquire = self.tab_acquire
        for signal, slot in (
            (self.camera_worker.frame_ready, self._on_camera_frame),
            (self.camera_worker.status_changed, self.on_camera_status_changed),
            (self.camera_worker.recording_failed, self._on_recording_failed),
            (self.serial_worker.port_status_signal, self.on_serial_status_changed),
            (self.serial_worker.data_updated_signal, acquire.update_mfc_telemetry),
            (self.serial_worker.command_failed_signal, self._on_command_failed),
            (acquire.pfs_selected, self.camera_worker.apply_pfs_safely),
            (acquire.port_changed, self._on_port_changed),
            (acquire.reconnect_camera_requested, self._reconnect_camera),
            (acquire.reconnect_serial_requested, self._reconnect_serial),
            (acquire.request_start_experiment, self.start_sync_experiment),
            (acquire.request_stop_experiment, self.stop_sync_experiment),
            (acquire.request_start_cam_recording, self.start_standalone_recording),
            (acquire.request_stop_cam_recording, self.stop_standalone_recording),
            (self.acquisition_controller.setpoint_requested,
             self.serial_worker.queue_setpoint),
            (self.acquisition_controller.finished, self._on_protocol_finished),
            (self.tab_protocol.protocol_applied, self.on_protocol_applied),
            (self.tab_protocol.plate_geometry_changed, self._sync_geometry_to_tracking),
        ):
            signal.connect(slot)

    def _refresh_acquisition_state(self):
        acquire = self.tab_acquire
        busy = self._recording is not None
        protocol = busy and self._recording.mode == "protocol"
        available = not self._closing and not self._finishing
        acquire.btn_sync_start.setEnabled(
            available and not busy and self.camera_ready and self.serial_ready,
        )
        acquire.btn_sync_stop.setEnabled(available and protocol)
        acquire.btn_toggle_cam_rec.setEnabled(
            available and not protocol and (busy or self.camera_ready),
        )
        # Protocol plans and connection settings stay fixed during recording.
        self.tab_protocol.setEnabled(available and not busy)
        for control in (acquire.txt_com_port, acquire.txt_pfs, acquire.combo_camera_presets,
                        acquire.btn_browse_pfs, acquire.btn_reconnect_cam,
                        acquire.btn_release_port, acquire.btn_reconnect_serial):
            control.setEnabled(available and not busy)
        state = "Recording" if self.camera_worker.recording else self._camera_state
        set_device_status(self.lbl_cam_status, "Camera", state, self._camera_details)
        set_device_status(acquire.lbl_stream_state, "Camera", state, self._camera_details)

    def _on_camera_frame(self, frame):
        if not self._closing:
            self.tab_acquire.update_viewport_frame(frame)
            height, width = frame.shape[:2]
            self.tab_acquire.lbl_resolution.setText(f"{width} x {height}")

    def on_camera_status_changed(self, state, title, details):
        if self._closing:
            return
        self.camera_ready = state == "OK"
        self._camera_state = "Ready" if self.camera_ready else "Connecting" if state == "PENDING" else "Offline"
        self._camera_details = f"{title}: {details}"
        self.tab_acquire.on_camera_status_changed(state, title, details)
        self._refresh_acquisition_state()

    def on_serial_status_changed(self, online, text):
        if self._closing:
            return
        self.serial_ready = online
        set_device_status(self.lbl_serial_status, "Air Valve", flowvision_state(online, text), text)
        self.tab_acquire.on_serial_status_changed(online, text)
        self._refresh_acquisition_state()
        if not online and self._recording and self._recording.mode == "protocol":
            self._on_recording_failed(f"FlowVision connection lost: {text}")

    def _on_port_changed(self, port):
        self.config.fv_com_port = port.strip().upper()
        self.serial_worker.set_port(self.config.fv_com_port)

    def _reconnect_camera(self):
        if not self._closing and not self.camera_worker.isRunning():
            self.camera_worker.start()

    def _reconnect_serial(self):
        if not self._closing:
            self.serial_worker.resume_polling()
            if not self.serial_worker.isRunning():
                self.serial_worker.start()

    def _sync_geometry_to_tracking(self, rows, cols):
        track = self.tab_track
        if track._tracking_busy or track.active_video_path is not None:
            return
        for control, value in ((track.spin_rows, rows), (track.spin_cols, cols)):
            previous = control.blockSignals(True)
            try:
                control.setValue(value)
            finally:
                control.blockSignals(previous)
        track._configure_canvas()

    @staticmethod
    def _validate_plan(steps, onset):
        validated = ProtocolScheduler.validate_steps(steps)
        onset = float(onset)
        duration = sum(step["time"] for step in validated)
        if not math.isfinite(onset) or not 0 <= onset <= duration:
            raise ValueError("Mix gas onset must be within the experiment duration.")
        return validated, onset, duration

    def on_protocol_applied(self, name, onset, steps):
        if self._recording or self.tab_acquire._session_log_active:
            QMessageBox.warning(self, "Protocol Not Applied",
                                "Stop acquisition and save its log first.")
            return
        try:
            steps, onset, _ = self._validate_plan(steps, onset)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid Protocol", str(exc))
            return
        self.tab_acquire.load_protocol_steps(steps, onset, name=name)
        QMessageBox.information(self, "Protocol Applied",
                                f"Applied {len(steps)} steps from '{name}'.")

    def _prepare_recording(self, mode):
        acquire = self.tab_acquire
        if mode == "protocol":
            steps, onset, duration = self._validate_plan(
                acquire.script_steps, acquire.mix_onset,
            )
            self.config.mix_onset = onset
            plan_steps = steps
            if onset > 0:
                steps = [{"time": onset, "o2": 0.0, "anes": 0.0, "n2": 0.0}, *steps]
            self.config.experiment_duration = sum(step["time"] for step in steps)
            self.config.pfs_path = self.camera_worker.pfs_path
            script = FlowVisionScriptModel.generate_script_text(
                port=self.config.fv_com_port, mix_onset=onset,
                duration=duration, steps=plan_steps,
            )
            directory, video, telemetry, _ = self.session_mgr.create_session(
                self.config, script, steps,
            )
        else:
            steps = None
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            directory = self.session_mgr.workspace_root / "standalone" / f"Session_{stamp}"
            paths = SessionPaths(directory)
            paths.ensure()
            video = paths.raw / "raw_video.mp4"
            telemetry = paths.logs / "sync_telemetry.csv"
        return RecordingContext(mode, directory, video, telemetry), steps

    def _start_recording(self, mode):
        if (self._closing or self._finishing or self._recording
                or self.camera_worker.recording):
            return
        acquire = self.tab_acquire
        if acquire._session_log_active:
            QMessageBox.warning(self, "Pending Session Log", "Save the previous session log first.")
            return
        if not self.camera_ready or (mode == "protocol" and not self.serial_ready):
            QMessageBox.warning(self, "Hardware Unavailable",
                                "Wait for the required devices to become ready.")
            return
        try:
            context, steps = self._prepare_recording(mode)
            self._recording = context
            acquire.attach_session_log(context.session_dir)
            acquire.log(f"Recording mode: {mode}")
            started = time.monotonic()
            self.camera_worker.start_recording(
                video_path=str(context.video), telemetry_csv_path=str(context.telemetry),
                fps=self.config.recording_fps, start_time=started,
            )
            if mode == "protocol":
                # Mark before starting so partial controller startup is rolled back.
                context.schedule_attempted = True
                acquire.start_schedule(steps)
            else:
                acquire.set_standalone_recording_state(True)
            acquire.recording_start_time = started
        except Exception as exc:
            acquire.log(f"Recording startup failed: {exc}")
            self._finish_recording(f"Startup failed: {exc}")
            QMessageBox.warning(self, "Recording Not Started", str(exc))
        finally:
            self._refresh_acquisition_state()

    def start_sync_experiment(self):
        self._start_recording("protocol")

    def start_standalone_recording(self):
        self._start_recording("standalone")

    def stop_sync_experiment(self):
        if self._recording and self._recording.mode == "protocol":
            self._finish_recording("Manual stop")

    def stop_standalone_recording(self):
        if self._recording and self._recording.mode == "standalone":
            self._finish_recording("Manual stop")

    def _on_protocol_finished(self):
        self._finish_recording("Protocol completed")

    def _finish_recording(self, reason):
        if self._recording is None or self._finishing:
            return
        context = self._recording
        acquire = self.tab_acquire
        self._finishing = True
        errors = []
        frames = 0
        try:
            # Stopping the protocol and closing the video must be independent.
            if context.schedule_attempted:
                try:
                    acquire.stop_schedule()
                except Exception as exc:
                    errors.append(f"Protocol stop failed: {exc}")
            try:
                frames = self.camera_worker.stop_recording()
            except Exception as exc:
                errors.append(f"Video close failed: {exc}")
            if errors:
                for error in errors:
                    acquire.log(error)
                if not self._closing:
                    QMessageBox.warning(self, "Recording Finalization Failed",
                                        "\n".join(errors) + "\nRetry Stop before starting another recording.")
                return
            if acquire.standalone_recording:
                acquire.set_standalone_recording_state(False)
            self._recording = None
            acquire.log(f"Recording ended: {reason}; reported frames: {frames}")
            try:
                if context.video.is_file() and context.video.stat().st_size > 0:
                    acquire.log(f"Recording saved: {context.video}")
                    self.tab_track.enqueue_recording(
                        session_dir=context.session_dir, video_path=context.video,
                        telemetry_csv=context.telemetry)
                    acquire.log("Recording submitted for Tracking import.")
                else:
                    acquire.log("No video was saved before recording stopped.")
            except Exception as exc:
                acquire.log(f"Tracking import failed; recording retained: {exc}")
            acquire.finish_session_log(reason)
        finally:
            self._finishing = False
            self._refresh_acquisition_state()

    def _on_command_failed(self, message):
        if self._closing:
            return
        self.tab_acquire.log(message)
        if self._recording and self._recording.mode == "protocol":
            self._on_recording_failed(message)

    def _on_recording_failed(self, message):
        if self._closing or self._recording is None:
            return
        self.tab_acquire.log(f"Acquisition failed: {message}")
        self._finish_recording(f"Acquisition failed: {message}")
        QMessageBox.warning(self, "Acquisition Failed", message)


    def closeEvent(self, event: QCloseEvent):
        self._closing = True
        self._finish_recording("Application closed")
        if self._recording is not None:
            self._closing = False
            self._refresh_acquisition_state()
            event.ignore()
            return
        if self.tab_acquire._session_log_active:
            if not self.tab_acquire.finish_session_log("Application closed"):
                self._closing = False
                self._refresh_acquisition_state()
                event.ignore()
                return
        self.tab_eval.cleanup()
        self.tab_track.cleanup_preview()
        for worker in (self.tab_track.worker, self.tab_eval.worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait()
        self.tab_acquire.cleanup()
        self.camera_worker.terminate_worker()
        self.serial_worker.terminate_worker()
        event.accept()


def main():
    # Scale fonts, inline styles, controls and fixed dimensions together.
    # Keep an explicit user-supplied Qt scale factor when present.
    os.environ.setdefault("QT_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('FATer.Workbench')
    app = QApplication(sys.argv)
    app.setWindowIcon(application_icon())
    install_window_centering(app)
    apply_typography(app)
    app.setStyleSheet(SCIENTIFIC_QSS)
    window = FATerApp(start_hardware="--no-hardware" not in sys.argv)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
