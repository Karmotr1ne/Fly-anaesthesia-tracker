"""Tracking workbench using the shared application typography and control styles."""

from __future__ import annotations

from core import schema

import copy
import logging
import re
from dataclasses import replace
from pathlib import Path
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLayout, QSizePolicy,
    QSplitter, QLabel, QPushButton, QListWidget, QListWidgetItem, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider,
    QToolButton, QMessageBox, QFrame,
    QStyledItemDelegate, QStyleOptionViewItem, QStyle, QScrollArea
)

from core.tracking.models import Calibration, TrackingOptions, TrackingJob, PreviewRequest, TrackingResult
from core.tracking.tracker_engine import ChamberROI, FlyTrackerEngine
from core.tracking.grid_alignment import RobustGridAligner
from core.tracking.calibration_cache import save_calibration
from core.tracking.tracking_jobs import ImportWorker, PreviewWorker, TrackingWorker
from ui.widgets.chamber_canvas import InteractiveChamberCanvas
from ui.widgets.drag_drop_zone import FolderImportArea
from ui.widgets.layout_helpers import scroll_panel

from ui.styles.theme import style_page


class SessionItemDelegate(QStyledItemDelegate):
    """Keep session names and right-aligned ROI status on one line."""
    def paint(self, painter, option, index):
        styled = QStyleOptionViewItem(option)
        self.initStyleOption(styled, index)
        name = styled.text
        styled.text = ''
        style = styled.widget.style()
        style.drawControl(QStyle.CE_ItemViewItem, styled, painter, styled.widget)
        rect = style.subElementRect(QStyle.SE_ItemViewItemText, styled, styled.widget).adjusted(3, 0, -5, 0)
        status = index.data(Qt.UserRole + 1) or 'No ROI'
        width = styled.fontMetrics.horizontalAdvance(status)
        painter.save()
        painter.setFont(styled.font)
        from PyQt5.QtGui import QPalette
        painter.setPen(styled.palette.color(QPalette.Text))
        painter.drawText(rect, Qt.AlignRight | Qt.AlignVCenter, status)
        rect.setRight(rect.right() - width - 12)
        painter.drawText(rect, Qt.AlignLeft | Qt.AlignVCenter,
                         styled.fontMetrics.elidedText(name, Qt.ElideLeft, max(0, rect.width())))
        painter.restore()


class TrackingTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tasks = {}
        self._items = {}
        self._active_video_key = None
        self.active_video_path = None
        self._last_calibration = None
        self._tracking_busy = False
        self._shutting_down = False
        self._abort_requested = False
        self._pending_jobs = None
        self._pending_recordings = {}
        self._preview_serial = 0
        self._preview_background = None
        self._preview_cache_key = None
        self._tracking_percent = 0.0
        self._tracking_video_key = None
        self._preview_mode_before_tracking = None
        self.worker = self.preview_worker = self.import_worker = None

        self._build_ui()
        style_page(self)
        for control in (self.spin_rows, self.spin_cols):
            control.setMinimumWidth(54)
        self.drop_box.setFixedHeight(36)
        self.btn_run.setMinimumWidth(220)
        self.btn_run.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(80)
        self._tracking_timer = QTimer(self)
        self._tracking_timer.setInterval(56)
        self._tracking_timer.timeout.connect(self._render_tracking_frame)
        self._bind_signals()
        self._refresh_controls()

    @staticmethod
    def _heading(text: str, secondary: bool = False) -> QWidget:
        section = QWidget()
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(text)
        label.setProperty("role", "secondary" if secondary else "section")
        layout.addWidget(label)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #E2E8F0; border: none;")
        layout.addWidget(divider)
        return section

    def _card(self, title: str, content: QWidget, expanded: bool = False) -> QFrame:
        card = QFrame()
        card.setProperty("role", "panel")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        toggle = QToolButton()
        toggle.setAccessibleName(title)
        toggle.setCheckable(True)
        toggle.setChecked(expanded)
        toggle.setProperty("role", "sectionToggle")
        toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        header = QHBoxLayout(toggle)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        label = QLabel(title)
        label.setProperty("role", "section")
        label.setAttribute(Qt.WA_TransparentForMouseEvents)
        indicator = QLabel()
        indicator.setFixedWidth(16)
        indicator.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        indicator.setAttribute(Qt.WA_TransparentForMouseEvents)
        header.addWidget(label)
        header.addStretch(1)
        header.addWidget(indicator)

        def update(checked):
            indicator.setText("\u25be" if checked else "\u25c2")
            content.setVisible(checked)
            layout.invalidate()
            card.updateGeometry()
            if hasattr(self, 'sidebar_area'):
                self.sidebar_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        layout.addWidget(toggle)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #E2E8F0; border: none;")
        layout.addWidget(divider)
        layout.addWidget(content)
        toggle.toggled.connect(update)
        update(expanded)
        return card

    def _build_ui(self):
        self.setObjectName("trackingTab")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_viewport_and_pipeline())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([310, 1000])
        self.workspace_splitter = splitter
        layout.addWidget(splitter, 1)

    # =========================================================================
    # 1. Compact session queue and calibration controls
    # =========================================================================
    def _build_left_panel(self) -> QWidget:
        content = QWidget()
        content.setObjectName("trackingSidebarContent")
        content.setAutoFillBackground(True)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSpacing(4)
        layout.setSizeConstraint(QLayout.SetMinimumSize)

        # 1. Session Queue
        layout.addWidget(self._heading("1. Session Queue"))
        self.drop_box = FolderImportArea()
        layout.addWidget(self.drop_box)
        
        self.list_sessions = QListWidget()
        self.list_sessions.setItemDelegate(SessionItemDelegate(self.list_sessions))
        self.list_sessions.setMinimumHeight(48)
        self.list_sessions.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.list_sessions.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_sessions.setTextElideMode(Qt.ElideRight)
        layout.addWidget(self.list_sessions, 1)

        queue_actions = QHBoxLayout()
        queue_actions.setSpacing(6)
        self.btn_clear_list = QPushButton("Clear List")
        self.btn_remove = QPushButton("Remove")
        self.lbl_queue_info = QLabel("0 vids | 0 chk | 0 roi")
        self.lbl_queue_info.setProperty("role", "secondary")
        queue_actions.addWidget(self.lbl_queue_info, 1)
        queue_actions.addWidget(self.btn_clear_list)
        queue_actions.addWidget(self.btn_remove)
        layout.addLayout(queue_actions)

        # 2. Chamber Calibration
        layout.addWidget(self._heading("2. Chamber Calibration"))
        geometry = QFrame()
        geometry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        geometry.setProperty("role", "panel")
        grid = QGridLayout(geometry)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.spin_rows, self.spin_cols = QSpinBox(), QSpinBox()
        self.spin_rows.setRange(1, 32); self.spin_rows.setValue(4)
        self.spin_cols.setRange(1, 16); self.spin_cols.setValue(3)
        grid.addWidget(QLabel("Rows"), 0, 0)
        grid.addWidget(self.spin_rows, 0, 1)
        grid.addWidget(QLabel("Columns"), 0, 2)
        grid.addWidget(self.spin_cols, 0, 3)

        self.combo_order = QComboBox()
        self.combo_order.addItem("Column-first", "column_first")
        self.combo_order.addItem("Row-first", "row_first")
        grid.addWidget(QLabel("Order"), 1, 0)
        grid.addWidget(self.combo_order, 1, 1, 1, 3)

        self.check_equal_size = QCheckBox("Lock Equal Size")
        self.check_equal_size.setChecked(True)
        self.check_shared_calibration = QCheckBox("Shared Grid")
        self.check_shared_calibration.setChecked(False)
        grid.addWidget(self.check_equal_size, 2, 0, 1, 2)
        grid.addWidget(self.check_shared_calibration, 2, 2, 1, 2)
        layout.addWidget(geometry)

        self.btn_snap_grid = QPushButton("Auto-Snap Grid")
        self.btn_undo = QPushButton("Undo")
        self.btn_clear_roi = QPushButton("Reset")
        tools = QHBoxLayout()
        tools.setSpacing(6)
        tools.addWidget(self.btn_snap_grid, 2)
        tools.addWidget(self.btn_undo, 1)
        tools.addWidget(self.btn_clear_roi, 1)
        layout.addLayout(tools)
        self.btn_clear_all_roi = QPushButton("Clear All ROI")

        layout.addWidget(self.btn_clear_all_roi)

        # 3. Tracking Options
        options_panel = QWidget()
        options_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        options_layout = QVBoxLayout(options_panel)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        options_layout.addWidget(self._heading("3. Tracking Options"))
        self.check_render_overlay = QCheckBox("Export Overlay Video")
        options_layout.addWidget(self.check_render_overlay)
        layout.addWidget(options_panel)

        # 4. Advance
        self.advanced_panel = self._build_advanced()
        self.card_advance = self._card("4. Advance", self.advanced_panel, expanded=False)
        layout.addWidget(self.card_advance)

        area = scroll_panel(content)
        self.sidebar_area = area
        area.setObjectName("trackingSidebar")
        area.viewport().setObjectName("trackingSidebarViewport")
        area.viewport().setAutoFillBackground(True)
        area.setMinimumWidth(300)
        area.setMaximumWidth(360)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    def _build_advanced(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.addWidget(self._heading("Background Model", secondary=True), 0, 0, 1, 2)
        
        self.spin_background_samples = QSpinBox()
        self.spin_background_samples.setRange(1, 240); self.spin_background_samples.setValue(60)
        grid.addWidget(QLabel("Sample frames"), 1, 0)
        grid.addWidget(self.spin_background_samples, 1, 1)
        
        self.spin_background_start, self.spin_background_end = QDoubleSpinBox(), QDoubleSpinBox()
        for control, lower, upper, value in (
            (self.spin_background_start, 0, 99.9, 0),
            (self.spin_background_end, .1, 100, 30),
        ):
            control.setDecimals(1)
            control.setRange(lower, upper)
            control.setValue(value)
            control.setSuffix(" %")
        grid.addWidget(QLabel("Sampling start"), 2, 0)
        grid.addWidget(self.spin_background_start, 2, 1)
        grid.addWidget(QLabel("Sampling end"), 3, 0)
        grid.addWidget(self.spin_background_end, 3, 1)
        
        self.btn_rebuild_background = QPushButton("Rebuild Background")
        grid.addWidget(self.btn_rebuild_background, 4, 0, 1, 2)
        grid.addWidget(self._heading("Gap Interpolation", secondary=True), 5, 0, 1, 2)
        
        self.check_interpolate = QCheckBox("Interpolate Short Gaps")
        self.check_interpolate.setChecked(True)
        grid.addWidget(self.check_interpolate, 6, 0, 1, 2)
        
        self.spin_max_gap = QSpinBox()
        self.spin_max_gap.setRange(0, 300); self.spin_max_gap.setValue(30)
        self.spin_interp_speed = QDoubleSpinBox()
        self.spin_interp_speed.setRange(.1, 100000); self.spin_interp_speed.setDecimals(1); self.spin_interp_speed.setValue(400)
        grid.addWidget(QLabel(schema.label("max_gap")), 7, 0)
        grid.addWidget(self.spin_max_gap, 7, 1)
        grid.addWidget(QLabel(schema.label("gap_speed")), 8, 0)
        grid.addWidget(self.spin_interp_speed, 8, 1)
        return panel

    # =========================================================================
    # 2. Video viewport, tracking controls and progress
    # =========================================================================
    def _build_viewport_and_pipeline(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.lbl_src = QLabel("No video selected")
        self.lbl_src.setProperty("role", "section")
        self.lbl_src.setTextFormat(Qt.PlainText)
        self.lbl_src.setWordWrap(True)
        self.lbl_src.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        header.addWidget(self.lbl_src, 1)
        self.lbl_video_info = QLabel()
        self.lbl_video_info.setProperty("role", "secondary")
        self.lbl_video_info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        header.addWidget(self.lbl_video_info)
        header.addWidget(QLabel("View"))
        self.combo_view_mode = QComboBox()
        for text, value in (("Original", "original"), ("Background", "background"), ("Centroids", "centroids")):
            self.combo_view_mode.addItem(text, value)
        header.addWidget(self.combo_view_mode)
        self.canvas = InteractiveChamberCanvas()
        layout.addWidget(self.canvas, 1)
        layout.addLayout(header)
        self.status_panel = QWidget()
        status = QHBoxLayout(self.status_panel)
        status.setContentsMargins(0, 0, 0, 0)
        self.status_panel.hide()
        self.lbl_stat_state = QLabel("Ready")
        self.lbl_stat_state.setWordWrap(True)
        self.lbl_stat_percent = QLabel("0%")
        status.addWidget(self.lbl_stat_state, 1)
        status.addWidget(self.lbl_stat_percent)
        timeline = QHBoxLayout()
        timeline.setSpacing(8)
        self.btn_previous_frame = QPushButton("Previous")
        self.btn_next_frame = QPushButton("Next")
        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setPrefix("Frame ")
        self.lbl_frame_count = QLabel("/ 0")
        self.lbl_frame_count.setProperty("role", "secondary")
        self.slider_frame = QSlider(Qt.Horizontal)
        self.slider_frame.setRange(0, 0)
        for control in (self.btn_previous_frame, self.spin_frame, self.lbl_frame_count, self.btn_next_frame):
            timeline.addWidget(control)
        timeline.addWidget(self.slider_frame, 1)
        layout.addLayout(timeline)
        layout.addWidget(self.status_panel)
        actions = QHBoxLayout()
        self.btn_run = QPushButton("Track")
        self.btn_run.setProperty("role", "primary")
        self.check_live_tracking = QCheckBox("Live Tracking")
        self.check_live_tracking.setChecked(False)
        self.check_live_tracking.setToolTip('Show live frames and detections while tracking. Can be changed during tracking.')
        actions.addWidget(self.check_live_tracking)
        self.check_show_trail = QCheckBox('Show Trail')
        self.check_show_trail.setToolTip('Show accumulated accepted trajectories; gaps and reacquisition start new lines.')
        actions.addWidget(self.check_show_trail)
        actions.addStretch(1)
        actions.addWidget(self.btn_run)
        layout.addLayout(actions)
        return panel

    # =========================================================================
    # Session selection, calibration and tracking actions
    # =========================================================================
    def _bind_signals(self):
        for signal, slot in (
            (self.drop_box.folders_selected, self.import_folders),
            (self.drop_box.rejected, self._append_log),
            (self.btn_clear_list.clicked, self._clear_list),
            (self.btn_remove.clicked, self._remove_current),
            (self.list_sessions.currentItemChanged, self._select_session),
            (self.list_sessions.itemChanged, self._on_check_changed),
            (self.btn_run.clicked, self._toggle_tracking),
            (self.check_live_tracking.toggled, self._live_tracking_changed),
            (self.check_show_trail.toggled, self._show_trail_changed),
            (self.check_shared_calibration.toggled, self._shared_calibration_changed),
            (self.btn_snap_grid.clicked, self._snap_grid),
            (self.btn_undo.clicked, self.canvas.undo),
            (self.btn_clear_roi.clicked, self._reset_calibration),
            (self.btn_clear_all_roi.clicked, self._clear_all_calibration),
            (self.canvas.first_roi_drawn, self._infer_grid),
            (self.canvas.calibration_changed, self._save_calibration),
            (self.canvas.geometry_restored, self._restore_geometry),
            (self.canvas.selection_changed, self._update_selection),
            (self.spin_rows.valueChanged, self._grid_changed),
            (self.spin_cols.valueChanged, self._grid_changed),
            (self.combo_order.currentIndexChanged, self._order_changed),
            (self.check_equal_size.toggled, self._equal_size_changed),
            (self.combo_view_mode.currentIndexChanged, self._schedule_preview),
            (self.spin_frame.valueChanged, self._frame_changed),
            (self.slider_frame.valueChanged, self._frame_changed),
            (self._preview_timer.timeout, self._request_preview),
            (self.check_interpolate.toggled, self._refresh_controls),
            (self.btn_rebuild_background.clicked, self._rebuild_background),
        ):
            signal.connect(slot)

        for control in (self.spin_background_samples, self.spin_background_start, self.spin_background_end):
            control.valueChanged.connect(self._invalidate_background)
        self.btn_previous_frame.clicked.connect(lambda: self.spin_frame.setValue(self.spin_frame.value() - 1))
        self.btn_next_frame.clicked.connect(lambda: self.spin_frame.setValue(self.spin_frame.value() + 1))

    def _append_log(self, message: str):
        if not self._shutting_down:
            logging.getLogger(__name__).debug("%s", message)

    def _set_tracking_progress(self, state=None, percent=None):
        self.status_panel.show()
        if state is not None:
            self.lbl_stat_state.setText(state)
        if percent is not None:
            self._tracking_percent = max(0.0, min(100.0, percent))
            self.lbl_stat_percent.setText(f"{self._tracking_percent:.0f}%")

    def _tracking_error(self, message: str):
        if not self._shutting_down:
            self.lbl_stat_state.setText("Tracking error")
            self.lbl_stat_state.setToolTip(message)

    def _active_task(self):
        return self.tasks.get(self._active_video_key)

    def _refresh_controls(self, *args):
        editable = not self._tracking_busy and not self._shutting_down
        active = self._active_task()
        checked = [task for task in self.tasks.values() if task.checked]
        importing = self.import_worker is not None
        self.check_live_tracking.setEnabled(not self._shutting_down)

        for control in (self.drop_box, self.btn_clear_list, self.btn_remove, self.list_sessions):
            control.setEnabled(editable and not importing)
        self.btn_remove.setEnabled(editable and not importing and active is not None)

        for control in (self.spin_rows, self.spin_cols, self.combo_order, self.check_equal_size,
                        self.check_shared_calibration, self.check_render_overlay, self.advanced_panel):
            control.setEnabled(editable)

        for control in (self.spin_frame, self.slider_frame, self.btn_previous_frame,
                        self.btn_next_frame, self.combo_view_mode):
            control.setEnabled(editable and active is not None)

        self.canvas.setEnabled(editable and active is not None)
        have_frame = self.canvas.raw_frame is not None
        self.btn_snap_grid.setEnabled(editable and bool(self.canvas.rois) and have_frame)
        self.btn_clear_roi.setEnabled(editable and bool(self.canvas.rois))
        self.btn_clear_all_roi.setEnabled(editable and any(task.calibration for task in self.tasks.values()))
        self.btn_undo.setEnabled(editable and bool(self.canvas.undo_stack))

        self.btn_run.setEnabled(
            not self._shutting_down and not self._abort_requested if self._tracking_busy
            else editable and not importing and bool(checked))
        text = "Cancel" if self._tracking_busy else "Track"
        if self.btn_run.text() != text:
            self.btn_run.setText(text)
            self.btn_run.setProperty("role", "danger" if self._tracking_busy else "primary")
            colour = "#DC2626" if self._tracking_busy else "#1D4ED8"
            self.btn_run.setStyleSheet(
                "QPushButton { background: " + colour + "; color: #FFFFFF; "
                "border: 1px solid " + colour + "; padding: 0 10px; font-weight: 600; }"
                "QPushButton:disabled { background: #F1F5F9; color: #94A3B8; border-color: #E2E8F0; }")

        enabled = editable and self.check_interpolate.isChecked()
        self.spin_max_gap.setEnabled(enabled)
        self.spin_interp_speed.setEnabled(enabled)
        self.lbl_queue_info.setText(
            f"{len(self.tasks)} vids | {len(checked)} chk | {len(self.canvas.rois)} roi")

    def _update_item(self, task):
        self.list_sessions.blockSignals(True)
        try:
            item = self._items.get(task.key)
            if item is None:
                item = QListWidgetItem()
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setData(Qt.UserRole, task.key)
                self._items[task.key] = item
                self.list_sessions.addItem(item)
            item.setText(self._session_name(task))
            item.setData(Qt.UserRole + 1, 'ROI' if task.calibration else 'No ROI')
            item.setToolTip(f"{self._session_name(task)}\n{task.video_path}\n{task.status}")
            item.setCheckState(Qt.Checked if task.checked else Qt.Unchecked)
        finally:
            self.list_sessions.blockSignals(False)
        self._refresh_controls()

    @staticmethod
    def _session_name(task):
        for directory in (task.session_dir, *task.session_dir.parents):
            match = re.search(r'(?<!\d)(\d{8})_(\d{6})(?:_(\d{6}))?(?!\d)', directory.name)
            if match:
                date, clock, fraction = match.groups()
                stamp = f'{date[:4]}-{date[4:6]}-{date[6:]} {clock[:2]}:{clock[2:4]}:{clock[4:]}'
                if fraction:
                    stamp += f'.{fraction}'
                return stamp
        return f'{task.session_dir.name} / {task.video_path.name}'

    def _on_check_changed(self, item):
        task = self.tasks.get(item.data(Qt.UserRole))
        if task is not None:
            task.checked = item.checkState() == Qt.Checked
        self._refresh_controls()

    def import_folders(self, folders):
        if self._tracking_busy or self.import_worker is not None or self._shutting_down:
            return
        self._start_import(folders=folders)

    def _start_import(self, folders=(), recordings=()):
        worker = ImportWorker(folders=folders, recordings=recordings, parent=self)
        self.import_worker = worker
        worker.result_ready.connect(self._merge_import)
        worker.message.connect(self._append_log)
        worker.finished.connect(lambda worker=worker: self._import_finished(worker))
        self._append_log("Scanning session folders...")
        self._refresh_controls()
        worker.start()

    def _merge_import(self, incoming):
        if self._shutting_down:
            return
        added = 0
        for task in incoming:
            existing = self.tasks.get(task.key)
            if existing is not None:
                if existing.metadata == task.metadata:
                    if task.telemetry_path:
                        existing.telemetry_path = task.telemetry_path
                    continue
                if self._last_calibration and self._last_calibration.source_video == str(task.video_path):
                    self._last_calibration = None
                self._append_log(f"Reset calibration: {task.video_path.name}")
            else:
                added += 1
            self.tasks[task.key] = task
            self._update_item(task)
        if self.tasks:
            key = self._active_video_key if self._active_video_key in self.tasks else next(iter(self.tasks))
            self.list_sessions.blockSignals(True)
            self.list_sessions.setCurrentItem(self._items[key])
            self.list_sessions.blockSignals(False)
            self._activate_video(key)
            if self.check_shared_calibration.isChecked():
                self._shared_calibration_changed(True)
        self._append_log(f"Imported {added} new video(s).")

    def _import_finished(self, worker):
        if self._shutting_down:
            return
        if self.import_worker is worker:
            self.import_worker = None
        worker.deleteLater()
        if not self._shutting_down:
            self._refresh_controls()
            self._drain_pending_recordings()

    def enqueue_recording(self, session_dir, video_path, telemetry_csv=None):
        if self._shutting_down:
            return
        data = {"video": str(video_path), "session_dir": Path(session_dir),
                "telemetry_csv": str(telemetry_csv) if telemetry_csv else None}
        self._pending_recordings[str(Path(video_path).resolve()).casefold()] = data
        self._drain_pending_recordings()

    def _drain_pending_recordings(self):
        if self._shutting_down or self._tracking_busy or self.import_worker or not self._pending_recordings:
            return
        data = tuple(self._pending_recordings.values())
        self._pending_recordings.clear()
        self._start_import(recordings=data)

    def _select_session(self, item, previous=None):
        if item is not None and not self._tracking_busy:
            self._activate_video(item.data(Qt.UserRole))

    def _activate_video(self, key):
        old = self._active_task()
        if old is not None:
            old.undo_history = copy.deepcopy(self.canvas.undo_stack)
        task = self.tasks[key]
        self._active_video_key, self.active_video_path = key, task.video_path
        self._preview_serial += 1
        self._clear_canvas()

        calibration = task.calibration
        if calibration:
            self._restore_geometry({"grid_rows": calibration.rows, "grid_cols": calibration.cols,
                                    "grid_order": calibration.order, "equal_size": calibration.equal_size})
            self.canvas.rois = self._canvas_rois(calibration.rois)
            self.canvas.selected_indices = {0}

        self.canvas.undo_stack = copy.deepcopy(task.undo_history)
        self._configure_canvas()
        self.lbl_src.setText(self._session_name(task))
        self.lbl_src.setToolTip(str(task.video_path))
        metadata = task.metadata
        self.lbl_video_info.setText(f"{metadata.width}×{metadata.height} | {metadata.fps:.1f} fps")

        frame = min(task.preview_frame, metadata.frames - 1)
        for control in (self.spin_frame, self.slider_frame):
            control.blockSignals(True)
            control.setRange(0, metadata.frames - 1)
            control.setValue(frame)
            control.blockSignals(False)
        self.lbl_frame_count.setText(f"/ {metadata.frames - 1}")

        self._update_selection()
        self._refresh_controls()
        self._schedule_preview()

    @staticmethod
    def _canvas_rois(bounds):
        return [ChamberROI(a, b, c - a, d - b, index + 1) for index, (a, b, c, d) in enumerate(bounds)]

    def _clear_canvas(self):
        self.canvas.raw_frame = self.canvas.gray_frame = None
        self.canvas.rois, self.canvas.centroids = [], []
        self.canvas.set_trail(None)
        self.canvas.show_trail = False
        self.canvas.undo_stack.clear()
        self.canvas.selected_indices.clear()
        self.canvas.is_dragging = self.canvas.rubber_band_active = self.canvas.drawing_first_roi = False
        self.canvas.draw_first_requested = False
        self.canvas.active_handle = self.canvas.drag_start_pos = None
        self.canvas.set_display_frame(None)

    def _clear_list(self):
        if self._tracking_busy or self.import_worker:
            return
        self.tasks.clear()
        self._items.clear()
        self.list_sessions.clear()
        self._last_calibration = None
        self._clear_active()
        self._append_log("Queue cleared.")

    def _remove_current(self):
        if self._active_video_key is None or self._tracking_busy or self.import_worker:
            return
        key = self._active_video_key
        self.tasks.pop(key)
        item = self._items.pop(key)
        self.list_sessions.blockSignals(True)
        self.list_sessions.takeItem(self.list_sessions.row(item))
        self.list_sessions.blockSignals(False)
        self._clear_active()
        if self.tasks:
            key = next(iter(self.tasks))
            self.list_sessions.setCurrentItem(self._items[key])

    def _clear_active(self):
        self._preview_serial += 1
        self._preview_timer.stop()
        if self.preview_worker:
            self.preview_worker.cancel_pending()
        self._active_video_key = self.active_video_path = None
        self._preview_background = self._preview_cache_key = None
        self._clear_canvas()
        self.lbl_src.setText("No video selected")
        self.lbl_video_info.clear()
        for control in (self.spin_frame, self.slider_frame):
            previous = control.blockSignals(True)
            control.setRange(0, 0)
            control.setValue(0)
            control.blockSignals(previous)
        self.lbl_frame_count.setText("/ 0")
        self._update_selection()
        self._refresh_controls()

    def _configure_canvas(self):
        self.canvas.grid_rows, self.canvas.grid_cols = self.spin_rows.value(), self.spin_cols.value()
        self.canvas.grid_order = self.combo_order.currentData()
        self.canvas.equal_size = self.check_equal_size.isChecked()

    def _restore_geometry(self, state):
        for control, value in ((self.spin_rows, state["grid_rows"]),
                               (self.spin_cols, state["grid_cols"]),
                               (self.combo_order, 0 if state["grid_order"] == "column_first" else 1),
                               (self.check_equal_size, state["equal_size"])):
            previous = control.blockSignals(True)
            if isinstance(control, QComboBox):
                control.setCurrentIndex(value)
            elif isinstance(control, QCheckBox):
                control.setChecked(value)
            else:
                control.setValue(value)
            control.blockSignals(previous)
        self._configure_canvas()

    def _persist_calibration(self, task):
        try:
            save_calibration(task)
        except OSError as exc:
            self._append_log(f'ROI cache could not be saved: {task.video_path.name}: {exc}')

    def _save_calibration(self, *, share=True):
        task = self._active_task()
        if task is None or self._tracking_busy:
            return
        calibration = None
        if self.canvas.rois:
            calibration = Calibration(tuple(roi.as_xyxy() for roi in self.canvas.rois),
                                      task.metadata.shape, self.spin_rows.value(), self.spin_cols.value(),
                                      self.combo_order.currentData(), self.check_equal_size.isChecked(), str(task.video_path))
        previous = task.calibration
        if previous is not None and calibration is not None and previous == replace(
                calibration, source_video=previous.source_video):
            calibration = previous
        if previous != calibration:
            task.calibration, task.result = calibration, None
        else:
            # Selection/no-op edits must not turn rounded shared pixels into
            # a new calibration source.
            calibration = task.calibration
        if calibration:
            self._last_calibration = calibration
        elif self._last_calibration and self._last_calibration.source_video == str(task.video_path):
            self._last_calibration = None
        task.undo_history = copy.deepcopy(self.canvas.undo_stack)
        self._configure_canvas()
        self._update_item(task)
        self._persist_calibration(task)
        if share and calibration is not None and self.check_shared_calibration.isChecked():
            self._synchronise_calibration(calibration)
        self._update_selection()
        self._schedule_preview()

    def _update_selection(self):
        count = len(self.canvas.rois)
        self.canvas.setToolTip(f"{count} chambers ready" if count else "Draw CH1 (top-left)" if self._active_task() else "Import a folder to begin.")
        selected = [self.canvas.rois[i] for i in sorted(self.canvas.selected_indices) if i < count]
        text = ", ".join(f"CH{roi.chamber_id}" for roi in selected)
        self.canvas.setStatusTip(text or "No chamber selected")
        self._refresh_controls()

    def _shared_calibration_changed(self, checked):
        if not checked or self._tracking_busy:
            return
        task = self._active_task()
        calibration = task.calibration if task else None
        try:
            if calibration is None:
                raise ValueError('Calibrate the current video before sharing its ROI.')
            for other in self.tasks.values():
                calibration.for_shape(other.metadata.shape)
        except ValueError as exc:
            self.check_shared_calibration.blockSignals(True)
            self.check_shared_calibration.setChecked(False)
            self.check_shared_calibration.blockSignals(False)
            QMessageBox.warning(self, 'Calibration Not Shared', str(exc))
            return
        self._synchronise_calibration(calibration)

    def _synchronise_calibration(self, calibration):
        active = self._active_task()
        for other in self.tasks.values():
            if other is active:
                self._persist_calibration(other)
                continue
            mapped = calibration.for_shape(other.metadata.shape) if calibration else None
            if other.calibration != mapped:
                other.calibration, other.result = mapped, None
                other.undo_history.clear()
            else:
                # Keep the common source even when this video's rounded pixels
                # happen to be unchanged by an edit on another video.
                other.calibration = mapped
            self._persist_calibration(other)
            self._update_item(other)

    def _infer_grid(self, first_roi):
        task = self._active_task()
        if task is None or self.canvas.gray_frame is None or self._tracking_busy:
            return
        try:
            x, y, width, height = first_roi
            if self.check_equal_size.isChecked() and (width * self.spin_cols.value() > task.metadata.width or height * self.spin_rows.value() > task.metadata.height):
                raise ValueError("CH1 is too large for the grid.")
            rois = FlyTrackerEngine.generate_grid_rois_snapped(
                self.canvas.gray_frame, first_roi, self.spin_rows.value(), self.spin_cols.value(),
                self.combo_order.currentData(), self.check_equal_size.isChecked())
            self.canvas.set_rois(rois)
            self.canvas.draw_first_requested = False
            self._save_calibration()
            self._append_log(f"Inferred {len(rois)} chambers.")
        except Exception as exc:
            self._restore_geometry({"grid_rows": self.canvas.grid_rows, "grid_cols": self.canvas.grid_cols,
                                    "grid_order": self.canvas.grid_order, "equal_size": self.canvas.equal_size})
            QMessageBox.warning(self, "Calibration Error", str(exc))

    def _grid_changed(self, *args):
        if self._tracking_busy:
            return
        if self.canvas.rois and self.canvas.gray_frame is not None:
            roi = self.canvas.rois[0]
            self._infer_grid((roi.x, roi.y, roi.w, roi.h))
        else:
            self._configure_canvas()

    def _order_changed(self, *args):
        if self._tracking_busy or not self.canvas.rois:
            self._configure_canvas()
            return
        rows, cols = self.canvas.grid_rows, self.canvas.grid_cols
        order = self.combo_order.currentData()
        old = self.canvas.rois
        if len(old) != rows * cols or order == self.canvas.grid_order:
            return
        if order == "row_first":
            rois = [copy.deepcopy(old[c * rows + r]) for r in range(rows) for c in range(cols)]
        else:
            rois = [copy.deepcopy(old[r * cols + c]) for c in range(cols) for r in range(rows)]
        for index, roi in enumerate(rois, 1):
            roi.chamber_id = index
        self.canvas.set_rois(rois)
        self._save_calibration()

    def _equal_size_changed(self, checked):
        if self._tracking_busy:
            return
        if self.canvas.rois:
            self.canvas._push_undo()
        self.canvas.equal_size = checked
        if checked and self.canvas.rois:
            self.canvas.normalise_equal_size(push_undo=False)
        else:
            self._save_calibration()

    def _snap_grid(self):
        if not self.canvas.rois or self.canvas.gray_frame is None or self._tracking_busy:
            return
        rows, cols = self.spin_rows.value(), self.spin_cols.value()
        bounds = [roi.as_xyxy() for roi in self.canvas.rois]
        if len(bounds) != rows * cols:
            return
        try:
            order = self.combo_order.currentData()
            columns = bounds if order == "column_first" else [bounds[r * cols + c] for c in range(cols) for r in range(rows)]
            snapped = RobustGridAligner.snap_all_boxes(self.canvas.gray_frame, columns, rows, cols,
                                                      equal_size=self.check_equal_size.isChecked())
            if order == "row_first":
                snapped = [snapped[c * rows + r] for r in range(rows) for c in range(cols)]
            rois = self._canvas_rois(snapped)
            if self.check_equal_size.isChecked():
                width, height = self.canvas.rois[0].w, self.canvas.rois[0].h
                image_h, image_w = self.canvas.gray_frame.shape
                for roi in rois:
                    roi.x = int(np.clip(roi.x + roi.w / 2 - width / 2, 0, image_w - width))
                    roi.y = int(np.clip(roi.y + roi.h / 2 - height / 2, 0, image_h - height))
                    roi.w, roi.h = width, height
            self.canvas.set_rois(rois)
            self._save_calibration()
        except Exception as exc:
            QMessageBox.warning(self, "Grid Alignment Failed", str(exc))

    def _reset_calibration(self):
        if not self._tracking_busy:
            self.canvas.set_rois([])
            self.canvas.draw_first_requested = False
            self._save_calibration(share=False)

    def _clear_all_calibration(self):
        if self._tracking_busy or self._shutting_down:
            return
        for task in self.tasks.values():
            task.calibration = task.result = None
            task.undo_history.clear()
            self._persist_calibration(task)
            self._update_item(task)
        self._last_calibration = None
        self._clear_canvas()
        self._update_selection()
        self._schedule_preview()

    def _read_options(self):
        options = TrackingOptions(
            background_samples=self.spin_background_samples.value(),
            background_start=self.spin_background_start.value() / 100,
            background_end=self.spin_background_end.value() / 100,
            interpolate=self.check_interpolate.isChecked(), max_gap=self.spin_max_gap.value(),
            gap_speed=self.spin_interp_speed.value(), render_overlay=self.check_render_overlay.isChecked())
        options.validate()
        return options

    def _frame_changed(self, value):
        task = self._active_task()
        if task:
            task.preview_frame = value
        for control in (self.spin_frame, self.slider_frame):
            control.blockSignals(True)
            control.setValue(value)
            control.blockSignals(False)
        self._schedule_preview()

    def _schedule_preview(self, *args):
        if not self._tracking_busy and not self._shutting_down and self._active_task():
            self._preview_serial += 1
            if self.preview_worker:
                self.preview_worker.cancel_pending()
            self._preview_timer.start()

    def _request_preview(self):
        task = self._active_task()
        if task is None or self._tracking_busy or self._shutting_down:
            return
        try:
            options = self._read_options()
        except ValueError as exc:
            self.canvas.setToolTip(str(exc))
            return
        if self.preview_worker is None:
            worker = PreviewWorker(self)
            self.preview_worker = worker
            worker.result_ready.connect(self._on_preview_result)
            worker.failed.connect(self._on_preview_error)
            worker.finished.connect(lambda worker=worker: self._preview_finished(worker))
            worker.start()
        request = PreviewRequest(self._preview_serial, task.video_path, task.preview_frame,
                                 self.combo_view_mode.currentData(), tuple(roi.as_xyxy() for roi in self.canvas.rois),
                                 options, options.background_key(task.video_path, task.metadata))
        self.preview_worker.submit(request)

    def _on_preview_result(self, token, result):
        if self._shutting_down or self._tracking_busy or token != self._preview_serial:
            return
        if str(self.active_video_path) != result["video"]:
            return
        self.canvas.set_frame(result["original"])
        self.canvas.set_display_frame(result["display"])
        self.canvas.centroids = result["centroids"]
        self._preview_background = result["background"]
        self._preview_cache_key = result["cache_key"]
        self.canvas.update()
        self._refresh_controls()

    def _on_preview_error(self, token, message):
        if token == self._preview_serial:
            self._append_log(f"Preview failed: {message}")

    def _preview_finished(self, worker):
        if self._shutting_down:
            return
        if self.preview_worker is worker:
            self.preview_worker = None
        worker.deleteLater()
        if not self._shutting_down:
            if self._pending_jobs is not None:
                self._launch_tracking()
            elif not self._tracking_busy:
                self._schedule_preview()

    def _invalidate_background(self, *args):
        self._preview_background = self._preview_cache_key = None
        self._schedule_preview()

    def _rebuild_background(self):
        if self._tracking_busy:
            return
        self._invalidate_background()
        if self.preview_worker:
            self.preview_worker.stop()
        self.combo_view_mode.setCurrentIndex(self.combo_view_mode.findData("background"))
        self._schedule_preview()

    def _toggle_tracking(self):
        if self._tracking_busy:
            self._abort_tracking()
        else:
            self._run_all()

    def _run_all(self):
        tasks = [task for task in self.tasks.values() if task.checked]
        if not tasks:
            QMessageBox.warning(self, 'No Videos Selected', 'Check at least one video to track.')
            return
        self._start_tasks(tasks)

    def _start_tasks(self, tasks):
        if self._tracking_busy or self.import_worker or not tasks or self._shutting_down:
            return
        try:
            options = self._read_options()
            shared = self.check_shared_calibration.isChecked()
            jobs = []
            for task in tasks:
                calibration = task.calibration
                if calibration is None:
                    raise ValueError(f"{task.video_path.name}: calibrate a video before tracking.")
                calibration.validate(task.metadata.shape)
                background = self._preview_background if self._preview_cache_key == options.background_key(task.video_path, task.metadata) else None
                jobs.append(TrackingJob(task.key, task.video_path, task.session_dir, task.telemetry_path,
                                        task.metadata, calibration, options, shared, background))
        except ValueError as exc:
            QMessageBox.warning(self, "Tracking Not Started", str(exc))
            return
        for task in tasks:
            task.result = TrackingResult()
            self._update_item(task)
        self._tracking_busy = True
        self._tracking_video_key = None
        self._preview_mode_before_tracking = self.combo_view_mode.currentData()
        self.combo_view_mode.blockSignals(True)
        self.combo_view_mode.setCurrentIndex(self.combo_view_mode.findData("centroids"))
        self.combo_view_mode.blockSignals(False)
        self.canvas.centroids = []
        self.canvas.draw_first_requested = False
        self._abort_requested = False
        self._set_tracking_progress("Preparing", 0)
        self._pending_jobs = tuple(jobs)
        self._preview_serial += 1
        self._preview_timer.stop()
        self._preview_background = self._preview_cache_key = None
        self._refresh_controls()
        self._append_log(f"Tracking {len(jobs)} video(s).")
        if self.preview_worker:
            self.preview_worker.stop()
        else:
            self._launch_tracking()

    def _launch_tracking(self):
        if self._pending_jobs is None or self._shutting_down:
            return
        jobs, self._pending_jobs = self._pending_jobs, None
        worker = TrackingWorker(jobs, self)
        self.worker = worker
        worker.set_preview_enabled(self.check_live_tracking.isChecked())
        worker.progress.connect(self._on_job_progress)
        worker.result_ready.connect(self._on_job_result)
        worker.error_signal.connect(self._tracking_error)
        worker.finished.connect(lambda worker=worker: self._jobs_finished(worker))
        if self.check_live_tracking.isChecked():
            self._tracking_timer.start()
        worker.start()

    def _live_tracking_changed(self, enabled):
        self.canvas.show_trail = enabled and self.check_show_trail.isChecked()
        if self.worker is not None:
            self.worker.set_preview_enabled(enabled)
        if (enabled and self._tracking_busy and self.worker is not None
                and not self._shutting_down and not self._abort_requested):
            self._tracking_timer.start()
        else:
            self._tracking_timer.stop()
            if self._tracking_busy:
                self.canvas.centroids = []
                self.canvas.set_trail(None)
                self.canvas.update()

    def _show_trail_changed(self, enabled):
        self.canvas.show_trail = enabled and self.check_live_tracking.isChecked() and self._tracking_busy
        self.canvas.update()

    def _render_tracking_frame(self):
        if (self._shutting_down or self._abort_requested or not self.check_live_tracking.isChecked()
                or not self._tracking_busy or self.worker is None):
            return
        preview = self.worker.take_preview()
        if preview is None:
            return
        key, frame_index, frame, rois, centroids, trail = preview
        task = self.tasks.get(key)
        if task is None:
            return
        if self._tracking_video_key != key:
            self._tracking_video_key = key
            self.canvas.rois = self._canvas_rois(rois)
            self.canvas.selected_indices.clear()
            self.lbl_src.setText(self._session_name(task))
            self.lbl_src.setToolTip(str(task.video_path))
            metadata = task.metadata
            self.lbl_video_info.setText(f"{metadata.width}×{metadata.height} | {metadata.fps:.1f} fps")
            self.lbl_frame_count.setText(f"/ {metadata.frames - 1}")
        self.canvas.set_frame(frame)
        self.canvas.centroids = centroids
        self.canvas.show_trail = self.check_show_trail.isChecked()
        self.canvas.set_trail(trail if self.check_show_trail.isChecked() else None)
        for control in (self.spin_frame, self.slider_frame):
            control.blockSignals(True)
            control.setRange(0, max(0, task.metadata.frames - 1))
            control.setValue(frame_index)
            control.blockSignals(False)
        self.canvas.update()

    def _restore_tracking_preview(self):
        self._tracking_timer.stop()
        self.canvas.set_trail(None)
        self.canvas.show_trail = False
        self._tracking_video_key = None
        if self._preview_mode_before_tracking is not None:
            self.combo_view_mode.blockSignals(True)
            self.combo_view_mode.setCurrentIndex(
                self.combo_view_mode.findData(self._preview_mode_before_tracking))
            self.combo_view_mode.blockSignals(False)
            self._preview_mode_before_tracking = None
        if self._active_video_key in self.tasks:
            self._activate_video(self._active_video_key)

    def _on_job_progress(self, key, index, total, phase, current, count, elapsed):
        if self._shutting_down or self._abort_requested or key not in self.tasks:
            return
        task = self.tasks[key]
        if task.result.status != "Running":
            task.result.status = "Running"
            self._update_item(task)
        if current > 0 and count > 0:
            fraction = min(1, current / count)
            stage = 0.8 * fraction if phase == "Tracking" else 0.9 + 0.1 * fraction
            percent = ((index - 1) + stage) / max(1, total) * 100
        else:
            percent = (index - 1) / max(1, total) * 100
        self._set_tracking_progress(
            f"Video {index}/{total}: {task.video_path.name} | {phase}",
            max(self._tracking_percent, percent))

    def _on_job_result(self, key, result):
        if self._shutting_down or key not in self.tasks:
            return
        task = self.tasks[key]
        task.result = result
        self._update_item(task)
        self._append_log(f"{task.video_path.name}: {result.status}")
        if result.position_csv:
            text = f'Saved video CSVs: {result.position_csv.parent}'
            self._append_log(text)

    def _jobs_finished(self, worker):
        if self._shutting_down:
            return
        if self.worker is worker:
            self._render_tracking_frame()
            self._tracking_timer.stop()
            self.worker = None
        worker.deleteLater()
        self._tracking_busy = False
        if self._shutting_down:
            return
        if not self._abort_requested:
            self._set_tracking_progress(percent=100)
        failed = any(task.result and task.result.status == "Failed" for task in self.tasks.values())
        self.lbl_stat_state.setText("Cancelled" if self._abort_requested else
                                    "Finished with errors" if failed else "Finished")
        self._restore_tracking_preview()
        self._refresh_controls()
        self._drain_pending_recordings()

    def _abort_tracking(self):
        if not self._tracking_busy or self._abort_requested:
            return
        self._abort_requested = True
        self._preview_background = self._preview_cache_key = None
        self._set_tracking_progress("Cancelling")
        self._tracking_timer.stop()
        if self._pending_jobs is not None:
            for job in self._pending_jobs:
                self.tasks[job.key].result.status = "Cancelled"
                self._update_item(self.tasks[job.key])
            self._pending_jobs = None
            self._tracking_busy = False
            self._set_tracking_progress("Cancelled", 0)
            self._restore_tracking_preview()
            self._drain_pending_recordings()
        if self.worker:
            self.worker.cancel()
        self.canvas.centroids = []
        self.canvas.set_trail(None)
        self.canvas.show_trail = False
        self.canvas.update()
        self._append_log("Cancellation requested.")
        self._refresh_controls()

    def cleanup_preview(self):
        self._shutting_down = True
        self._preview_timer.stop()
        self._tracking_timer.stop()
        self._pending_jobs = None
        self._pending_recordings.clear()
        if self.import_worker:
            self.import_worker.requestInterruption()
        if self.preview_worker:
            self.preview_worker.stop()
        for worker in (self.import_worker, self.preview_worker):
            if worker:
                worker.wait()
