'Camera capture, gas-flow controls and synchronised session logging.'

from __future__ import annotations

import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QComboBox,
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTabWidget, QTabBar, QStylePainter, QStyleOptionTab, QStyle,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QLineEdit,
    QMessageBox,
    QFrame, QSizePolicy,
)

from ui.widgets.video_viewport import AspectRatioVideoWidget

from core.hardware.process_cleaner import clean_flowvision_processes
from core.protocol.library import protocol_root
from core.hardware.protocol_scheduler import DEVICE_MAP, ProtocolScheduler


from ui.styles.theme import style_page, set_device_status, flowvision_state

class _AirValveTabBar(QTabBar):
    """Draw labels in the tab rectangle without the native label inset."""

    def paintEvent(self, event):
        painter = QStylePainter(self)
        for index in range(self.count()):
            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.CE_TabBarTabShape, option)
            painter.setFont(self.font())
            selected = bool(option.state & QStyle.State_Selected)
            painter.setPen(QColor("#FFFFFF" if selected else "#475569"))
            painter.drawText(self.tabRect(index).translated(-2, 0),
                             Qt.AlignCenter | Qt.TextSingleLine, self.tabText(index))


def _section_line(layout, thickness=1):
    line = QFrame()
    line.setFixedHeight(thickness)
    line.setStyleSheet("background: #E2E8F0; border: none;")
    layout.addWidget(line)


class AcquisitionTab(QWidget):
    # Signals communicating with main application
    request_start_experiment = pyqtSignal()
    request_stop_experiment = pyqtSignal()
    request_start_cam_recording = pyqtSignal()
    request_stop_cam_recording = pyqtSignal()

    reconnect_camera_requested = pyqtSignal()
    reconnect_serial_requested = pyqtSignal()
    pfs_selected = pyqtSignal(str)
    port_changed = pyqtSignal(str)

    DEVICE_MAP = DEVICE_MAP

    def __init__(self, parent=None):
        super().__init__(parent)
        self.script_steps = [
            {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
            {"time": 240.0, "o2": 180.0, "anes": 20.0, "n2": 0.0},
            {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
        ]
        self.mix_onset = 0.0
        self.active_protocol_name = "protocol_custom.fv"
        self.protocol_running = False
        self.schedule_paused = False
        self.standalone_recording = False
        self.current_step_idx = 0
        self.current_step_remaining = 60.0

        # Video FPS and session logging state
        self.recording_start_time: float | None = None
        self.frame_timestamps = deque(maxlen=60)
        self.active_session_log_path: Path | None = None

        self._session_log_entries = []
        self._session_log_active = False
        self._session_log_started = None
        self.controller = None
        self.init_ui()
        style_page(self)
        for label in (self.lbl_real_fps, self.lbl_rec_timer, self.lbl_resolution):
            label.setFixedHeight(26)
        for label in (self.lbl_proto_name, self.lbl_step_info, self.lbl_step_ratio):
            label.setFixedHeight(20)
        for name in ("lbl_status_val", "lbl_physics_val"):
            for label in self.tab_mfc.findChildren(QLabel, name):
                label.setMinimumHeight(26)
        # Match control heights while retaining useful input and action lengths.
        hardware_controls = (self.txt_pfs, self.txt_com_port,
                             self.btn_browse_pfs, self.btn_release_port,
                             self.btn_reconnect_cam, self.btn_reconnect_serial)
        for control in hardware_controls:
            control.setMinimumWidth(72)
            control.setMaximumWidth(16777215)
            control.setFixedHeight(26)
            control.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            control.setStyleSheet("""
                background: #FFFFFF; color: #0F172A;
                border: 1px solid #CBD5E1; border-radius: 4px;
                padding: 0 10px; margin: 0px;
            """)
        for button in (self.btn_browse_pfs, self.btn_release_port,
                       self.btn_reconnect_cam, self.btn_reconnect_serial):
            button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            button.setMinimumWidth(button.sizeHint().width())
        self.txt_com_port.setFixedWidth(90)
        for label in (lbl for lbl in self.findChildren(QLabel)
                      if lbl.text() in ("Feature:", "Air Valve:", "Setpoint:")):
            label.setFixedHeight(26)
        for button in (self.btn_edit_steps, self.btn_sched_extend,
                       self.btn_sched_pause, self.btn_sched_skip):
            button.setMinimumWidth(button.sizeHint().width())
        set_device_status(self.lbl_stream_state, "Camera", "Offline")
        set_device_status(self.lbl_serial_status, "FlowVision", "Offline")

        # 1Hz timer for step countdown and recording duration
        self.step_timer = QTimer(self)
        self.step_timer.timeout.connect(self._timer_tick)
        self.step_timer.start(1000)

    def init_ui(self):
        # Use the shared theme instead of cascading border resets.
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(14)

        # ---------------- Middle Work Area (Split Left/Right) ----------------
        h_mid = QHBoxLayout()
        h_mid.setSpacing(14)

        # =====================================================================
        # LEFT COLUMN: Video viewport, status strip and hardware controls
        # =====================================================================
        v_left = QVBoxLayout()
        v_left.setSpacing(8)
        camera_heading = QLabel("1. Camera Preview")
        camera_heading.setProperty("role", "section")
        v_left.addWidget(camera_heading)
        _section_line(v_left)

        # 1. Main Viewport
        self.video_viewport = AspectRatioVideoWidget()
        v_left.addWidget(self.video_viewport, 1)

        # 2. Viewport Live Status Strip (Clean pills, zero borders)
        status_bar_frame = QWidget()
        h_vid_stat = QHBoxLayout(status_bar_frame)
        h_vid_stat.setContentsMargins(2, 2, 2, 2)
        h_vid_stat.setSpacing(8)

        pill_base = "border-radius: 4px; padding: 2px 7px;  font-weight: 600;"

        self.lbl_stream_state = QLabel("● Online")
        self.lbl_stream_state.setStyleSheet(f"background-color: #F0FDF4; color: #15803D; {pill_base}")

        self.lbl_real_fps = QLabel("FPS: 0.0")
        self.lbl_real_fps.setStyleSheet(f"background-color: #F0F9FF; color: #0284C7; font-family: 'Consolas'; {pill_base}")

        self.lbl_rec_timer = QLabel("REC: Standby")
        self.lbl_rec_timer.setStyleSheet(f"background-color: #F1F5F9; color: #64748B; font-family: 'Consolas'; {pill_base}")

        self.lbl_resolution = QLabel("4200×1300")
        self.lbl_resolution.setStyleSheet("color: #94A3B8;  padding-right: 2px;")

        self.lbl_stream_state.hide()
        h_vid_stat.addWidget(self.lbl_real_fps)
        h_vid_stat.addWidget(self.lbl_rec_timer)
        h_vid_stat.addStretch()
        h_vid_stat.addWidget(self.lbl_resolution)
        v_left.addWidget(status_bar_frame, 0)

        # 3. Camera Configuration & Air Valve Panel (Flat, no QGroupBox outline)
        pnl_hw_cfg = QWidget()
        pnl_hw_cfg.setObjectName("acquire_pnl_hw_cfg")
        pnl_hw_cfg.setStyleSheet("QWidget#acquire_pnl_hw_cfg { background-color: #FFFFFF; border-radius: 6px;  border: none; }")
        grid_hw = QGridLayout(pnl_hw_cfg)
        grid_hw.setContentsMargins(1, 1, 1, 1)
        grid_hw.setHorizontalSpacing(8)
        grid_hw.setColumnStretch(1, 1)
        grid_hw.setVerticalSpacing(8)

        lbl_pfs = QLabel("Feature:")
        lbl_pfs.setStyleSheet("font-weight: 400; color: #334155;")
        grid_hw.addWidget(lbl_pfs, 0, 0)

        self.txt_pfs = QLineEdit("12 chamber.pfs")
        self.txt_pfs.setMinimumWidth(0)
        grid_hw.addWidget(self.txt_pfs, 0, 1)

        self.btn_browse_pfs = QPushButton("Browse...")
        self.btn_browse_pfs.setStyleSheet("background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; font-weight: 400;")
        grid_hw.addWidget(self.btn_browse_pfs, 0, 2)

        self.btn_reconnect_cam = QPushButton("Reconnect Cam")
        self.btn_reconnect_cam.setStyleSheet("background-color: #FFF7ED; border: 1px solid #FFEDD5; border-radius: 4px; font-weight: 400;")
        grid_hw.addWidget(self.btn_reconnect_cam, 0, 3)

        lbl_valve = QLabel("Air Valve:")
        lbl_valve.setStyleSheet("font-weight: 400; color: #334155;")
        grid_hw.addWidget(lbl_valve, 1, 0)

        self.txt_com_port = QLineEdit("COM3")
        self.lbl_serial_status = QLabel("FlowVision: Offline")
        self.lbl_serial_status.hide()
        grid_hw.addWidget(self.txt_com_port, 1, 1)

        self.btn_release_port = QPushButton("Release")
        self.btn_release_port.setStyleSheet("background-color: #FEF2F2; color: #B91C1C; font-weight: 400; border: 1px solid #FECACA; border-radius: 4px;")
        grid_hw.addWidget(self.btn_release_port, 1, 2)

        self.btn_reconnect_serial = QPushButton("Reconnect Serial")
        self.btn_reconnect_serial.setStyleSheet("background-color: #FFF7ED; border: 1px solid #FFEDD5; border-radius: 4px; font-weight: 400;")
        grid_hw.addWidget(self.btn_reconnect_serial, 1, 3)

        hardware_heading = QLabel("2. Hardware Configuration")
        hardware_heading.setProperty("role", "section")
        v_left.addSpacing(6)
        v_left.addWidget(hardware_heading)
        _section_line(v_left)
        v_left.addWidget(pnl_hw_cfg, 0)
        h_mid.addLayout(v_left, 3)

        # =====================================================================
        # RIGHT COLUMN: Air valve, schedule and status log
        # =====================================================================
        right_content = QWidget()
        v_right = QVBoxLayout(right_content)
        v_right.setContentsMargins(0, 0, 0, 0)
        v_right.setSpacing(14)

        # Section 1: FlowVision Telemetry (Flat Compact Card)
        pnl_mfc = QWidget()
        pnl_mfc.setObjectName("acquire_pnl_mfc")
        pnl_mfc.setStyleSheet("QWidget#acquire_pnl_mfc { background-color: #FFFFFF; border-radius: 6px; border: none; }")
        v_mfc_layout = QVBoxLayout(pnl_mfc)
        v_mfc_layout.setContentsMargins(0, 0, 0, 0)
        v_mfc_layout.setSpacing(8)

        lbl_mfc_header = QLabel("3. Air Valve")
        lbl_mfc_header.setProperty("role", "section")
        v_mfc_layout.addWidget(lbl_mfc_header)
        _section_line(v_mfc_layout)

        self.tab_mfc = QTabWidget()
        self.tab_mfc.setTabBar(_AirValveTabBar(self.tab_mfc))
        # Use the rendered font for native tab text measurements.
        tab_font = QFont(self.font())
        tab_font.setFamily("IBM Plex Sans")
        tab_font.setPixelSize(12)
        tab_font.setWeight(QFont.DemiBold)
        self.tab_mfc.tabBar().setFont(tab_font)
        self.tab_mfc.tabBar().setElideMode(Qt.ElideNone)
        self.tab_mfc.tabBar().setUsesScrollButtons(True)
        self.tab_mfc.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                background: #E2E8F0;
                color: #475569;
                border-radius: 4px;
                padding: 4px 14px;
                margin-right: 4px;
                border: none;
                font-weight: 600;
                
            }
            QTabBar::tab:selected {
                border: none;
                font-weight: 600;
                background: #1D4ED8;
                color: #FFFFFF;
            }
        """)
        v_mfc_layout.addWidget(self.tab_mfc)
        self._init_mfc_tabs()
        v_right.addWidget(pnl_mfc, 0)

        # Section 2: Dynamic Schedule Controls (Flat Compact Card)
        pnl_sched = QWidget()
        pnl_sched.setObjectName("acquire_pnl_sched")
        pnl_sched.setStyleSheet("QWidget#acquire_pnl_sched { background-color: #FFFFFF; border-radius: 6px; border: none; }")
        v_sched = QVBoxLayout(pnl_sched)
        v_sched.setContentsMargins(0, 0, 0, 0)
        v_sched.setSpacing(6)

        lbl_sched_header = QLabel("4. Protocol Dynamic Schedule")
        lbl_sched_header.setProperty("role", "section")
        v_sched.addWidget(lbl_sched_header)
        _section_line(v_sched)

        # Compact protocol status row.
        h_proto_status = QHBoxLayout()
        h_proto_status.setSpacing(8)
        self.lbl_proto_name = QLabel(f"Protocol: {self.active_protocol_name}")
        self.lbl_proto_name.setStyleSheet("font-weight: 400; color: #1E293B; ")
        self.lbl_step_info = QLabel("Phase: Standby (1 / 3)")
        self.lbl_step_info.setStyleSheet("color: #64748B; font-weight: 400; ")
        h_proto_status.addWidget(self.lbl_proto_name)
        h_proto_status.addStretch()
        h_proto_status.addWidget(self.lbl_step_info)
        v_sched.addLayout(h_proto_status)

        self.progress_step = QProgressBar()
        self.progress_step.setTextVisible(True)
        self.progress_step.setValue(0)
        self.progress_step.setFixedHeight(22)
        self.progress_step.setStyleSheet("""
            QProgressBar {
                min-height: 22px;
                max-height: 22px;
                padding: 0px;
                margin: 0px;
                border: none;
                font-size: 12px;
                background-color: #E2E8F0;
                border-radius: 3px;
                text-align: center;
                
                font-weight: 600;
                color: #0F172A;
            }
            QProgressBar::chunk {
                background-color: #1D4ED8;
                border-radius: 3px;
            }
        """)
        v_sched.addWidget(self.progress_step)

        self.lbl_step_ratio = QLabel("Target: O2: 200.0 | Anes: 0.0 | N2: 0.0 sccm")
        self.lbl_step_ratio.setStyleSheet("color: #334155; font-weight: 400; ")
        v_sched.addWidget(self.lbl_step_ratio)

        h_sched_actions = QHBoxLayout()
        h_sched_actions.setSpacing(8)
        btn_sched_style = """
            QPushButton {
                background-color: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: 600;
                
                color: #334155;
            }
            QPushButton:hover {
                background-color: #F1F5F9;
                border-color: #94A3B8;
            }
        """

        self.btn_sched_extend = QPushButton("+30s")
        self.btn_sched_extend.setStyleSheet(btn_sched_style)
        self.btn_sched_pause = QPushButton("Stop / Run")
        self.btn_sched_pause.setStyleSheet(btn_sched_style)
        self.btn_sched_skip = QPushButton("Skip")
        self.btn_sched_skip.setStyleSheet(btn_sched_style)

        self.btn_edit_steps = QPushButton("Edit")
        self.btn_edit_steps.setStyleSheet(btn_sched_style)
        self.btn_edit_steps.clicked.connect(self._edit_steps)
        for button in (self.btn_sched_extend, self.btn_sched_pause,
                       self.btn_sched_skip, self.btn_edit_steps):
            h_sched_actions.addWidget(button, 1)
        v_sched.addLayout(h_sched_actions)

        v_right.addWidget(pnl_sched, 0)

        # Section 3: Diagnostics Log (Flat Card, Clean Text)
        pnl_log = QWidget()
        pnl_log.setObjectName("acquire_pnl_log")
        pnl_log.setStyleSheet("QWidget#acquire_pnl_log { background-color: #FFFFFF; border-radius: 6px;  border: none; }")
        v_log = QVBoxLayout(pnl_log)
        v_log.setContentsMargins(0, 0, 0, 0)
        v_log.setSpacing(8)

        lbl_log_header = QLabel("5. Status Log")
        lbl_log_header.setProperty("role", "section")
        lbl_log_header.setStyleSheet("font-weight: 600; color: #000000;")
        v_log.addWidget(lbl_log_header)
        _section_line(v_log)
        self.btn_save_log = QPushButton("Retry Save Log")
        self.btn_save_log.clicked.connect(self._retry_save_log)
        self.btn_save_log.hide()

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setPlaceholderText("Hardware telemetry events and system runtime logs...")
        self.txt_log.setStyleSheet("background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; font-family: 'Consolas';  padding: 4px;")
        v_log.addWidget(self.txt_log, 1)
        v_log.addWidget(self.btn_save_log, 0, Qt.AlignRight)
        v_right.addWidget(pnl_log, 1)

        # Natural minimum sizes protect buttons; equal stretches share spare space.
        for section in (pnl_mfc, pnl_sched, pnl_log):
            section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pnl_mfc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        pnl_sched.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.tab_mfc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.txt_log.setMinimumHeight(90)
        h_mid.addWidget(right_content, 2)
        main_layout.addLayout(h_mid, 1)

        # ---------------- Bottom Execution Strip (Flat Group) ----------------
        pnl_exec = QWidget()
        pnl_exec.setObjectName("acquire_pnl_exec")
        pnl_exec.setStyleSheet("QWidget#acquire_pnl_exec { background-color: #FFFFFF; border-radius: 6px;  border: none; }")
        h_btn = QHBoxLayout(pnl_exec)
        h_btn.setContentsMargins(10, 6, 10, 6)
        h_btn.setSpacing(12)

        self.btn_toggle_cam_rec = QPushButton("Record Video Only")
        self.btn_toggle_cam_rec.setFixedHeight(38)
        self.btn_toggle_cam_rec.setStyleSheet("background-color: #334155; color: #FFFFFF; font-weight: 600; border-radius: 4px;")

        self.btn_sync_start = QPushButton("Start Synchronized Experiment")
        self.btn_sync_start.setFixedHeight(38)
        self.btn_sync_start.setStyleSheet("background-color: #15803D; color: #FFFFFF; font-weight: 600; border-radius: 4px; ")

        self.btn_sync_stop = QPushButton("Stop Experiment")
        self.btn_sync_stop.setFixedHeight(38)
        self.btn_sync_stop.setEnabled(False)
        self.btn_sync_stop.setStyleSheet("background-color: #FEF2F2; color: #B91C1C; font-weight: 600; border: 1px solid #FECACA; border-radius: 4px; ")

        h_btn.addWidget(self.btn_toggle_cam_rec, 2)
        h_btn.addWidget(self.btn_sync_start, 4)
        h_btn.addWidget(self.btn_sync_stop, 4)
        main_layout.addWidget(pnl_exec, 0)

        # ================= Signal Connections =================
        self.btn_browse_pfs.clicked.connect(self._choose_pfs)
        self.txt_pfs.editingFinished.connect(self._apply_feature_path)
        self.btn_reconnect_cam.clicked.connect(self.reconnect_camera_requested.emit)
        self.btn_reconnect_serial.clicked.connect(self.reconnect_serial_requested.emit)
        self.btn_release_port.clicked.connect(self._handle_release_port)
        self.txt_com_port.textChanged.connect(self.port_changed.emit)

        self.btn_toggle_cam_rec.clicked.connect(self._handle_toggle_camera_rec)
        self.btn_sync_start.clicked.connect(self.request_start_experiment)
        self.btn_sync_stop.clicked.connect(self.request_stop_experiment)

        self.btn_sched_extend.clicked.connect(self._extend_time)
        self.btn_sched_pause.clicked.connect(self._toggle_pause)
        self.btn_sched_skip.clicked.connect(self._skip_step)

    def _set_manual_flow(self, unit_id, value):
        try:
            if self.controller is None:
                raise ValueError("Acquisition controller is not available")
            self.controller.set_manual_flow(unit_id, value)
            return True
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Change Not Applied", str(exc))
            return False

    def _zero_manual_flow(self, unit_id, spin):
        if self._set_manual_flow(unit_id, 0.0):
            spin.setValue(0.0)

    def edit_plan_step(self, index, replacement, onset):
        if self.protocol_running:
            self.controller.update_step(index, replacement)
            return
        if self._session_log_active:
            raise ValueError("Save the previous session log before editing another plan")
        replacement = ProtocolScheduler.validate_steps([replacement])[0]
        if not 0 <= index < len(self.script_steps):
            raise ValueError("Invalid step index")
        duration = sum(step["time"] for i, step in enumerate(self.script_steps) if i != index) + replacement["time"]
        if not 0 <= onset <= duration:
            raise ValueError("Planned mix gas onset must be within the experiment duration")
        before = dict(self.script_steps[index])
        self.script_steps[index] = replacement
        previous_onset = self.mix_onset
        self.mix_onset = onset
        self.log(f"Planned step {index + 1} edited: {before} -> {replacement}")
        if previous_onset != onset:
            self.log(f"Planned mix gas onset edited: {previous_onset:.1f}s -> {onset:.1f}s")

    def _edit_steps(self):
        if self.standalone_recording:
            QMessageBox.warning(self, "Recording Active", "Stop standalone recording before editing a plan.")
            return
        running = self.protocol_running
        steps = self.controller.scheduler.steps if running else self.script_steps
        if not steps:
            return
        first = self.controller.scheduler.index if running else 0
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Acquisition Steps")
        form = QFormLayout(dialog)
        selector = QComboBox()
        for index in range(first, len(steps)):
            selector.addItem(f"Step {index + 1}", index)
        form.addRow("Step", selector)
        fields = {}
        for key, label in (("time", "Phase duration (s)"), ("o2", "O2 (sccm)"),
                           ("anes", "Anaesthetics (sccm)"), ("n2", "N2 (sccm)")):
            spin = QDoubleSpinBox()
            spin.setDecimals(1)
            spin.setRange(0.1 if key == "time" else 0.0, 86400.0 if key == "time" else 5000.0)
            fields[key] = spin
            form.addRow(label, spin)
        def populate():
            for key, spin in fields.items():
                spin.setValue(steps[selector.currentData()][key])
        selector.currentIndexChanged.connect(populate)
        populate()
        onset = QDoubleSpinBox()
        onset.setDecimals(1)
        onset.setRange(0.0, 86400.0)
        onset.setValue(self.mix_onset)
        onset.setEnabled(not running)
        form.addRow("Planned mix gas onset (s)", onset)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        def apply():
            try:
                if running != self.protocol_running:
                    raise ValueError("Experiment state changed. Reopen the editor.")
                self.edit_plan_step(selector.currentData(),
                    {key: spin.value() for key, spin in fields.items()}, onset.value())
            except (ValueError, TypeError) as exc:
                QMessageBox.warning(dialog, "Change Not Applied", str(exc))
                return
            dialog.accept()
        buttons.accepted.connect(apply)
        dialog.exec_()

    def _init_mfc_tabs(self):
        """Construct non-overlapping, well-spaced MFC telemetry cards."""
        self.tab_mfc.clear()
        units = [("A", "O2", "O2"), ("B", "Anaesthetics", "CO2"), ("C", "N2", "N2")]

        for uid, name, gas_medium in units:
            w = QWidget()
            v_inner = QVBoxLayout(w)
            v_inner.setContentsMargins(0, 8, 0, 0)
            v_inner.setSpacing(0)

            # Device name and valve status.
            h_top = QHBoxLayout()
            lbl_dev = QLabel(f"Unit {uid}: {name} ({gas_medium})")
            lbl_dev.setStyleSheet("font-weight: 600; color: #0F172A; ")

            lbl_stat = QLabel("Valve: Standby")
            lbl_stat.setObjectName("lbl_status_val")
            lbl_stat.setStyleSheet("""
                color: #1D4ED8; font-weight: 600; 
                background-color: #EFF6FF; border: 1px solid #BFDBFE;
                border-radius: 3px; padding: 2px 6px;
            """)

            h_top.addWidget(lbl_dev)
            h_top.addStretch()
            h_top.addWidget(lbl_stat)
            v_inner.addLayout(h_top)
            v_inner.addSpacing(8)

            # Explicit gaps balance text and controls without stretching rows.

            # Measured flow.
            lbl_f = QLabel("0000.0 sccm")
            lbl_f.setObjectName("lbl_flow_val")
            lbl_f.setFixedHeight(40)
            lbl_f.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lbl_f.setStyleSheet("""
                font-size: 26px;
                font-family: "IBM Plex Sans";
                
                font-weight: 700;
                color: #000000;
            """)
            v_inner.addWidget(lbl_f)
            v_inner.addSpacing(12)

            # Setpoint input and controls.
            h_ctrl = QHBoxLayout()
            h_ctrl.setSpacing(8)
            lbl_sp_title = QLabel("Setpoint:")
            lbl_sp_title.setStyleSheet(" font-weight: 400; color: #64748B;")

            spin = QDoubleSpinBox()
            spin.setObjectName("spin_sp_input")
            spin.setDecimals(1)
            spin.setRange(0.0, 5000.0)
            spin.setValue(0.0)
            spin.setSuffix(" sccm")
            spin.setFixedWidth(105)

            btn_set = QPushButton("SET")
            btn_set.setFixedWidth(48)
            btn_set.setStyleSheet("""
                QPushButton {
                    background-color: #1D4ED8;
                    color: white;
                    font-weight: 400;
                    border-radius: 4px;
                    padding: 4px;
                }
                QPushButton:hover { background-color: #1D4ED8; }
            """)

            btn_zero = QPushButton("ZERO")
            btn_zero.setFixedWidth(48)
            btn_zero.setStyleSheet("""
                QPushButton {
                    background-color: #475569;
                    color: white;
                    font-weight: 400;
                    border-radius: 4px;
                    padding: 4px;
                }
                QPushButton:hover { background-color: #334155; }
            """)

            h_ctrl.addWidget(lbl_sp_title)
            h_ctrl.addWidget(spin)
            h_ctrl.addWidget(btn_set)
            h_ctrl.addWidget(btn_zero)
            h_ctrl.addStretch(1)
            v_inner.addLayout(h_ctrl)
            v_inner.addSpacing(8)

            # Pressure and temperature readings.
            lbl_physics = QLabel("P: +014.70 PSIA  |  T: +025.0 C")
            lbl_physics.setObjectName("lbl_physics_val")
            lbl_physics.setStyleSheet("color: #94A3B8;  font-family: 'Consolas';")
            v_inner.addWidget(lbl_physics)
            v_inner.addStretch(1)

            btn_set.clicked.connect(lambda _, u=uid, s=spin: self._set_manual_flow(u, s.value()))
            btn_zero.clicked.connect(lambda _, u=uid, s=spin: self._zero_manual_flow(u, s))

            self.tab_mfc.addTab(w, f"{uid}: {name}")
            
    def update_viewport_frame(self, frame):
        """Frame update hook computing rendering/grab FPS."""
        now = time.perf_counter()
        self.frame_timestamps.append(now)

        if len(self.frame_timestamps) >= 2:
            dt = self.frame_timestamps[-1] - self.frame_timestamps[0]
            if dt > 0:
                fps = (len(self.frame_timestamps) - 1) / dt
                self.lbl_real_fps.setText(f"FPS: {fps:04.1f}")

        self.video_viewport.update_frame(frame)

    def _handle_toggle_camera_rec(self):
        if not self.standalone_recording:
            self.request_start_cam_recording.emit()
        else:
            self.request_stop_cam_recording.emit()

    def set_standalone_recording_state(self, is_recording: bool):
        self.standalone_recording = is_recording
        if is_recording:
            self.recording_start_time = time.monotonic()
            self.btn_toggle_cam_rec.setText("Stop Camera Recording (REC)")
            self.btn_toggle_cam_rec.setStyleSheet("background-color: #B91C1C; color: white; font-weight: 600; border-radius: 4px;")
            self.lbl_rec_timer.setStyleSheet("background-color: #FEF2F2; color: #B91C1C; border-radius: 4px; padding: 2px 7px; font-family: 'Consolas'; font-weight: 600; ")
            self.btn_sync_start.setEnabled(False)
            self.log("Standalone Camera Recording Started.")
        else:
            self.recording_start_time = None
            self.btn_toggle_cam_rec.setText("Record Video Only")
            self.btn_toggle_cam_rec.setStyleSheet("background-color: #334155; color: white; font-weight: 600; border-radius: 4px;")
            self.lbl_rec_timer.setText("REC: Standby")
            self.lbl_rec_timer.setStyleSheet("background-color: #F1F5F9; color: #64748B; border-radius: 4px; padding: 2px 7px; font-family: 'Consolas'; font-weight: 600; ")
            self.btn_sync_start.setEnabled(True)
            self.log("Standalone Camera Recording Stopped.")

    def on_camera_status_changed(self, state: str, title: str, desc: str):
        status = "Ready" if state == "OK" else "Connecting" if state == "PENDING" else "Offline"
        set_device_status(self.lbl_stream_state, "Camera", status, f"{title}: {desc}")
        if state == "OK":
            if not self.standalone_recording and not self.protocol_running:
                self.btn_sync_start.setEnabled(True)
            self.btn_toggle_cam_rec.setEnabled(True)
        else:
            self.btn_sync_start.setEnabled(False)
            self.btn_toggle_cam_rec.setEnabled(False)
        self.log(f"Camera: {title} | {desc}")

    def on_serial_status_changed(self, online: bool, status_text: str):
        set_device_status(self.lbl_serial_status, "FlowVision", flowvision_state(online, status_text), status_text)

    def _handle_release_port(self):
        terminated = clean_flowvision_processes()
        if terminated:
            self.log("Terminated FlowVision background processes. COM port reclaimed.")
            QMessageBox.information(self, "Port Released", "Terminated background FlowVision processes. COM port handle is free.")
        else:
            self.log("No locking process found. Retriggering serial link...")
        self.reconnect_serial_requested.emit()

    def _choose_pfs(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Basler Camera Feature File", str(protocol_root() / 'camera'), "PFS Files (*.pfs);;All Files (*.*)")
        if path:
            self.txt_pfs.setText(path)
            self._apply_feature_path()

    def _apply_feature_path(self):
        if not self.txt_pfs.isEnabled():
            return
        value = self.txt_pfs.text().strip()
        path = Path(value).expanduser()
        if not value or not path.is_file() or path.suffix.lower() != ".pfs":
            self.log("Camera feature not applied: select an existing .pfs file.")
            return
        resolved = str(path.resolve())
        if resolved == getattr(self, "_submitted_feature", None):
            return
        self._submitted_feature = resolved
        self.txt_pfs.setText(resolved)
        self.pfs_selected.emit(resolved)
        self.log(f"Camera feature requested: {path.name}")

    def cleanup(self):
        self.step_timer.stop()
        if self.controller:
            self.controller.stop()
        self.protocol_running = False
        if self.standalone_recording:
            self.request_stop_cam_recording.emit()
        self.log("Safety Teardown: Dispatched 0.0 sccm to all channels.")

    def bind_controller(self, controller):
        self.controller = controller
        controller.step_changed.connect(self._show_step)
        controller.progress_changed.connect(self._show_progress)
        controller.message.connect(self.log)

    def _extend_time(self):
        if self.controller:
            self.controller.extend()

    def _toggle_pause(self):
        if self.controller:
            self.controller.toggle_pause()

    def _skip_step(self):
        if self.controller:
            self.controller.skip()

    def _show_step(self, idx, step):
        self.current_step_idx = idx
        self.lbl_step_ratio.setText(
            f"Target Flow: O2: {step['o2']:.1f} | Anes: {step['anes']:.1f} | N2: {step['n2']:.1f} sccm")
        self.log(f"Step {idx + 1} targets requested: {step}")

    def _show_progress(self, state):
        self.current_step_idx = state['index']
        self.current_step_remaining = state['remaining']
        self.schedule_paused = state['paused']
        self.progress_step.setValue(state['percent'])
        remaining = int(state['remaining'])
        self.lbl_step_info.setText(
            f"Phase {state['index'] + 1} / {state['count']}: "
            f"{remaining // 60:02d}:{remaining % 60:02d} left ({state['percent']}%)")

    def _retry_save_log(self):
        if self.protocol_running:
            try:
                self._write_session_log()
            except OSError as exc:
                QMessageBox.warning(self, "Log Save Failed", str(exc))
        elif self._session_log_active:
            self.finish_session_log("manual_log_save")

    def attach_session_log(self, session_dir: Path):
        if self._session_log_active:
            raise RuntimeError("Save the previous session log before starting another experiment")
        path = Path(session_dir) / "logs" / "acquisition_runtime.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.active_session_log_path = path
        self._session_log_started = time.monotonic()
        self._session_log_active = True
        self.log(f"Session log started: {path}")
        self.log("Starting plan: " + json.dumps({
            "protocol": self.active_protocol_name,
            'mix_onset': self.mix_onset,
            "steps": self.script_steps,
        }, ensure_ascii=False))

    def _write_session_log(self):
        path = self.active_session_log_path
        if path is None:
            return
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(self._session_log_entries) + "\n", encoding="utf-8")
        temporary.replace(path)

    def log(self, text: str):
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        elapsed = ""
        if self._session_log_active:
            elapsed = f" [+{time.monotonic() - self._session_log_started:.3f}s]"
        entry = f"[{timestamp}]{elapsed} {text}"
        self.txt_log.insertPlainText(entry + "\n")
        self._session_log_entries.append(entry)
        if self._session_log_active:
            try:
                self._write_session_log()
            except OSError as exc:
                self.txt_log.insertPlainText(f"[Log write failed; entries retained in memory] {exc}\n")

    def finish_session_log(self, reason: str):
        if not self._session_log_active:
            return True
        self.log(f"Experiment ended: {reason}")
        try:
            self._write_session_log()
        except OSError as exc:
            QMessageBox.warning(self, "Log Save Failed",
                f"Log entries remain in memory. Retry Save Log before closing.\n"
                f"Path: {self.active_session_log_path}\n{exc}")
            self.btn_save_log.show()
            return False
        self.btn_save_log.hide()
        saved_path = self.active_session_log_path
        self._session_log_active = False
        self._session_log_started = None
        self.active_session_log_path = None
        self._session_log_entries = []
        self.txt_log.insertPlainText(f"Log saved: {saved_path}\n")
        return True

    def update_mfc_telemetry(self, devices: list[dict]):
        for dev in devices:
            uid = dev.get("unit_id")
            idx = {"A": 0, "B": 1, "C": 2}.get(uid, -1)
            if idx >= 0 and idx < self.tab_mfc.count():
                tab_w = self.tab_mfc.widget(idx)
                f_lbl = tab_w.findChild(QLabel, "lbl_flow_val")
                s_lbl = tab_w.findChild(QLabel, "lbl_status_val")
                p_lbl = tab_w.findChild(QLabel, "lbl_physics_val")

                if f_lbl:
                    f_lbl.setText(f"{dev.get('flow_sccm', 0.0):06.1f} sccm")
                if s_lbl:
                    valve_status = dev.get("valve_status", "Standby")
                    s_lbl.setText(f"Valve: {valve_status}")
                    if valve_status == "Active":
                        s_lbl.setStyleSheet("color: #15803D; font-weight: 600;  background-color: #F0FDF4; border-radius: 3px; padding: 2px 6px;")
                    else:
                        s_lbl.setStyleSheet("color: #1D4ED8; font-weight: 600;  background-color: #EFF6FF; border-radius: 3px; padding: 2px 6px;")
                if p_lbl:
                    psia = dev.get("pressure_psia", 14.7)
                    temp_c = dev.get("temp_c", 25.0)
                    p_lbl.setText(f"P: {psia:+07.2f} PSIA  |  T: {temp_c:+05.1f} C")

    def load_protocol_steps(self, steps: list[dict], onset: float, name: str = "protocol_custom.fv"):
        self.script_steps = ProtocolScheduler.validate_steps(steps)
        self.mix_onset = onset
        self.active_protocol_name = name
        self.lbl_proto_name.setText(f"Active Protocol: {name}")
        self.log(f"Loaded protocol '{name}' ({len(steps)} steps).")

    def start_schedule(self, steps: list[dict]):
        if self.controller is None:
            raise RuntimeError("Acquisition controller is not bound")
        self.controller.start(steps)
        self.script_steps = [dict(step) for step in self.controller.scheduler.steps]
        self.schedule_paused = False
        self.protocol_running = True
        self.recording_start_time = time.monotonic()

        self.btn_sync_start.setEnabled(False)
        self.btn_sync_stop.setEnabled(True)
        self.btn_toggle_cam_rec.setEnabled(False)
        self.lbl_rec_timer.setStyleSheet("background-color: #FEF2F2; color: #B91C1C; border-radius: 4px; padding: 2px 7px; font-family: 'Consolas'; font-weight: 600; ")

    def stop_schedule(self):
        if self.controller:
            self.controller.stop()
        self.protocol_running = False
        self.recording_start_time = None
        self.lbl_step_info.setText("Protocol Stopped")
        self.lbl_rec_timer.setText("REC: Standby")
        self.lbl_rec_timer.setStyleSheet("background-color: #F1F5F9; color: #64748B; border-radius: 4px; padding: 2px 7px; font-family: 'Consolas'; font-weight: 600; ")

        self.btn_sync_start.setEnabled(True)
        self.btn_sync_stop.setEnabled(False)
        self.btn_toggle_cam_rec.setEnabled(True)

    def _timer_tick(self):
        """Refresh recording duration; protocol timing belongs to the controller."""
        if self.recording_start_time is not None:
            elapsed = int(time.monotonic() - self.recording_start_time)
            self.lbl_rec_timer.setText(
                f"REC: {elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}")
