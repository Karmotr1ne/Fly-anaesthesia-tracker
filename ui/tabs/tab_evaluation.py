"""Evaluation workbench with dataset/archive tools left and analysis controls right."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import List, Tuple, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QSizePolicy,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QDoubleSpinBox,
    QCheckBox,
    QFrame,
    QLineEdit,
    QFileDialog,
    QSplitter,
)

from core.analysis.anaesthesia_analyser import AnaesthesiaParams
from core.session.layout import session_root, experiment_name
from ui.widgets.drag_drop_zone import DragDropUploadArea
from core.analysis.evaluation_jobs import EvaluationWorker, ArchiveWorker, SessionCheckWorker, EvaluationImportWorker


from ui.styles.theme import style_page

class UnifiedEvaluationTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.archive_worker = None
        self.import_worker = None
        self._pending_import_paths = []
        self._file_check_worker = None
        self._pending_file_checks = None
        self._file_check_serial = 0
        self._closing = False
        self.task_context = {}
        self.worker: Optional[QThread] = None
        self.matched_tasks: List[Tuple[Path, Optional[Path]]] = []
        self.anaesthetic_onset = AnaesthesiaParams().anaesthetic_onset
        self.evaluated_df = None
        self._evaluation_completed = 0
        self._evaluation_failed = 0
        self._archive_failed = False
        self._archive_result_received = False
        self._init_ui()
        style_page(self)

    @staticmethod
    def _heading(text: str) -> QWidget:
        section = QWidget()
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(text)
        label.setProperty("role", "section")
        layout.addWidget(label)
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: #E2E8F0; border: none;")
        layout.addWidget(divider)
        return section

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        # =====================================================================
        # =====================================================================
        left_panel = QWidget()
        v_left = QVBoxLayout(left_panel)
        v_left.setContentsMargins(0, 0, 6, 0)
        v_left.setSpacing(8)

        v_left.addWidget(self._heading("1. Datasets"))
        self.drop_area = DragDropUploadArea(
            title="Import Location Data",
            hint="Auto-matches Raw CSVs & Telemetry. Click to browse folders.",
            parent=self,
            folder_selection=True,
        )
        self.drop_area.lbl_hint.hide()
        self.drop_area.setFixedHeight(75)
        self.drop_area.lbl_title.setWordWrap(True)
        self.drop_area.layout().setContentsMargins(4, 4, 4, 4)
        self.drop_area.paths_dropped.connect(self.handle_incoming_paths)
        v_left.addWidget(self.drop_area)

        h_table_bar = QHBoxLayout()
        h_table_bar.setContentsMargins(0, 0, 0, 0)
        h_table_bar.setSpacing(8)
        h_table_bar.addStretch(1)
        btn_t_style = "padding: 2px 8px; font-size: 12px; min-height: 20px;"
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.setStyleSheet(btn_t_style)
        btn_sel_all.clicked.connect(lambda: self._set_all_checks(True))
        btn_sel_none = QPushButton("Deselect All")
        btn_sel_none.setStyleSheet(btn_t_style)
        btn_sel_none.clicked.connect(lambda: self._set_all_checks(False))
        self.btn_clear = QPushButton("Clear Tasks")
        self.btn_clear.setStyleSheet(btn_t_style)
        self.btn_clear.clicked.connect(self.clear_tasks)

        h_table_bar.addWidget(btn_sel_all)
        h_table_bar.addWidget(btn_sel_none)
        h_table_bar.addWidget(self.btn_clear)

        self._running_rows = []
        self.table_results = QTableWidget(0, 5)
        self.table_results.setHorizontalHeaderLabels([
            "Select", "No.", "Source", "Session Files", "Evaluation"
        ])
        self.table_results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.setColumnWidth(0, 48)
        self.table_results.setStyleSheet("""
            QTableWidget {
                background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; font-size: 12px;
            }
            QHeaderView::section {
                background: #F8FAFC; border: none; border-bottom: 1px solid #E2E8F0;
                font-weight: 600; color: #475569; padding: 4px; font-size: 12px;
            }
        """)
        v_left.addWidget(self.table_results, 1)
        v_left.addLayout(h_table_bar)

        card_summary = QFrame()
        card_summary.setObjectName("evaluationSummaryCard")
        card_summary.setStyleSheet("""
            QFrame#evaluationSummaryCard {
                background-color: #F0F4F8;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
            }
        """)
        v_summary = QVBoxLayout(card_summary)
        v_summary.setContentsMargins(10, 8, 10, 8)
        v_summary.setSpacing(6)

        v_summary.addWidget(self._heading("Summary Tool"))

        h_dest = QHBoxLayout()
        h_dest.setSpacing(4)
        self.txt_summary_dest = QLineEdit()
        self.txt_summary_dest.setPlaceholderText("Select destination directory for compiled reports...")
        self.txt_summary_dest.setFixedHeight(24)
        self.txt_summary_dest.setStyleSheet("background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; font-size: 12px;")
        btn_dest = QPushButton("Browse...")
        btn_dest.setFixedHeight(24)
        btn_dest.setStyleSheet("padding: 0 8px; font-size: 12px;")
        btn_dest.clicked.connect(self._choose_summary_destination)
        h_dest.addWidget(self.txt_summary_dest, 1)
        h_dest.addWidget(btn_dest)
        v_summary.addLayout(h_dest)

        h_summary_btns = QHBoxLayout()
        h_summary_btns.setSpacing(6)

        self.btn_export_summary = QPushButton("Sort Summary Metrics")
        self.btn_export_summary.setToolTip(
            'Export metric tables grouped by session start time: 08:00–10:59, '
            '11:00–13:59, 14:00–16:59, then other times; blank columns separate groups.')
        self.btn_export_summary.setFixedHeight(28)
        self.btn_export_summary.setStyleSheet("""
            QPushButton {
                background-color: #15803D; color: white; font-weight: 600;
                border-radius: 4px; border: none; font-size: 12px;
            }
            QPushButton:hover { background-color: #166534; }
        """)
        self.btn_export_summary.clicked.connect(self._run_cross_session_summary)

        self.btn_copy_raw = QPushButton("Extract Raw CSVs")
        self.btn_copy_raw.setFixedHeight(28)
        self.btn_copy_raw.setStyleSheet(self.btn_export_summary.styleSheet())
        self.btn_copy_raw.clicked.connect(self._extract_raw_csvs)

        h_summary_btns.addWidget(self.btn_export_summary, 1)
        h_summary_btns.addWidget(self.btn_copy_raw, 1)
        v_summary.addLayout(h_summary_btns)

        v_left.addWidget(card_summary, 0)
        splitter.addWidget(left_panel)

        # =====================================================================
        # =====================================================================
        right_panel = QWidget()
        v_right = QVBoxLayout(right_panel)
        v_right.setContentsMargins(6, 0, 0, 0)
        v_right.setSpacing(10)

        v_right.addWidget(self._heading("2. Analysis Parameters"))

        def create_spin(val, min_v, max_v, decimals=2, step=1.0, suffix=""):
            sp = QDoubleSpinBox()
            sp.setRange(min_v, max_v)
            sp.setValue(val)
            sp.setDecimals(decimals)
            sp.setSingleStep(step)
            if suffix:
                sp.setSuffix(suffix)
            sp.setFixedHeight(24)
            sp.setStyleSheet("background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 4px; font-size: 12px;")
            return sp

        def make_param_group(title_text: str):
            box = QFrame()
            box.setObjectName("evaluationParameterGroup")
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            box.setStyleSheet("QFrame#evaluationParameterGroup { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; }")
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(10, 8, 10, 8)
            box_layout.setSpacing(6)
            title = QLabel(title_text)
            title.setProperty("role", "secondary")
            box_layout.addWidget(title)
            form_grid = QGridLayout()
            form_grid.setHorizontalSpacing(10)
            form_grid.setVerticalSpacing(8)
            box_layout.addLayout(form_grid)
            return box, form_grid

        box_onset, grid_onset = make_param_group("Stimulation Timing")
        lbl_onset = QLabel("Anaesthetic Gas Onset:")
        lbl_onset.setStyleSheet("font-size: 12px; color: #475569; font-weight: 500;")
        self.spin_onset = create_spin(AnaesthesiaParams().anaesthetic_onset, 0, 86400, 1, 10.0, " s")
        grid_onset.addWidget(lbl_onset, 0, 0)
        grid_onset.addWidget(self.spin_onset, 0, 1)
        grid_onset.setColumnStretch(1, 1)
        v_right.addWidget(box_onset)

        box_sedate, grid_sedate = make_param_group("Stage 1: Sedation Criteria")
        lbl_ratio = QLabel("Sedate Speed Ratio:")
        lbl_ratio.setStyleSheet("font-size: 12px; color: #475569; font-weight: 500;")
        self.spin_ratio = create_spin(0.050, 0.001, 0.499, 3, 0.01)

        lbl_drop = QLabel("Sedate Drop Height (/1s):")
        lbl_drop.setStyleSheet("font-size: 12px; color: #475569; font-weight: 500;")
        self.spin_drop = create_spin(0.25, 0.01, 1.0, 2, 0.05)

        grid_sedate.addWidget(lbl_ratio, 0, 0)
        grid_sedate.addWidget(self.spin_ratio, 0, 1)
        grid_sedate.addWidget(lbl_drop, 1, 0)
        grid_sedate.addWidget(self.spin_drop, 1, 1)
        grid_sedate.setColumnStretch(1, 1)
        v_right.addWidget(box_sedate)

        box_anes, grid_anes = make_param_group("Stage 2: Anaesthesia Criteria")
        lbl_still = QLabel("Still Window (Knockdown):")
        lbl_still.setStyleSheet("font-size: 12px; color: #475569; font-weight: 500;")
        self.spin_still = create_spin(120.0, 0.1, 1200.0, 1, 10.0, " s")

        lbl_speed = QLabel("Still Speed Thresh:")
        lbl_speed.setStyleSheet("font-size: 12px; color: #475569; font-weight: 500;")
        self.spin_speed = create_spin(0.10, 0.01, 20.0, 2, 0.05, " px/s")

        grid_anes.addWidget(lbl_still, 0, 0)
        grid_anes.addWidget(self.spin_still, 0, 1)
        grid_anes.addWidget(lbl_speed, 1, 0)
        grid_anes.addWidget(self.spin_speed, 1, 1)
        grid_anes.setColumnStretch(1, 1)
        v_right.addWidget(box_anes)

        v_right.addSpacing(4)
        v_right.addStretch(1)

        card_exec = QFrame()
        card_exec.setObjectName("evaluationExecution")
        card_exec.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card_exec.setStyleSheet("QFrame#evaluationExecution { background: #FFFFFF; border: none; }")
        v_exec = QVBoxLayout(card_exec)
        v_exec.setContentsMargins(10, 8, 10, 8)
        v_exec.setSpacing(6)

        v_exec.addWidget(self._heading("3. In-Place Evaluation & Plotting"))
        self.check_summary_only = QCheckBox("Summary Only (skip frame CSV and plots)")
        self.check_summary_only.setChecked(False)
        self.check_summary_only.toggled.connect(self._summary_mode_changed)
        v_exec.addWidget(self.check_summary_only)

        self.check_plots = QCheckBox("Behaviour Overview (Activity, Kymograph, Survival)")
        self.check_plots.setChecked(True)
        self.check_plots.setStyleSheet("color: #334155; font-size: 12px; font-weight: 500;")
        v_exec.addWidget(self.check_plots)

        self.check_heatmaps = QCheckBox("Spatial Preference by Protocol Step (Floor/Ceiling)")
        self.check_heatmaps.setToolTip(
            "Export all saved protocol steps in one figure, using planned step times from recording start. "
            "Without a saved protocol, show the whole recording and report a warning.")
        self.check_heatmaps.setChecked(True)
        self.check_heatmaps.setStyleSheet("color: #334155; font-size: 12px; font-weight: 500;")
        v_exec.addWidget(self.check_heatmaps)

        self.check_median_plot = QCheckBox("Summary Plot")
        self.check_median_plot.setToolTip("Time-binned chamber distributions: median, quartiles, 1.5 IQR whiskers and outliers.")
        self.check_median_plot.setStyleSheet(
            "QCheckBox { background: #EFF6FF; border: 1px solid #BFDBFE; "
            "border-radius: 4px; padding: 5px 8px; color: #334155; }")
        self.status_panel = QWidget()
        status_layout = QHBoxLayout(self.status_panel)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_task_count = QLabel("Ready")
        self.lbl_task_count.setStyleSheet("font-size: 12px; color: #475569;")
        self.lbl_percent = QLabel("0%")
        self.lbl_percent.setStyleSheet("font-size: 12px; font-weight: bold; color: #1D4ED8;")
        status_layout.addWidget(self.lbl_task_count, 1)
        status_layout.addWidget(self.lbl_percent)
        self.status_panel.hide()
        v_exec.addWidget(self.status_panel)

        self.btn_run_all = QPushButton("Run Evaluation on Selected")
        self.btn_run_all.setEnabled(False)
        self.btn_run_all.setFixedHeight(30)
        self.btn_run_all.setStyleSheet("""
            QPushButton {
                background-color: #1D4ED8; color: white; font-weight: 600;
                border-radius: 4px; border: none; font-size: 12px;
            }
            QPushButton:hover { background-color: #1E40AF; }
            QPushButton:disabled { background-color: #93C5FD; }
        """)
        self.btn_run_all.clicked.connect(self.run_evaluation)
        execution_row = QHBoxLayout()
        execution_row.addWidget(self.check_median_plot)
        execution_row.addWidget(self.btn_run_all, 1)
        v_exec.addLayout(execution_row)

        v_right.addWidget(card_exec)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 46)
        splitter.setStretchFactor(1, 54)
        splitter.setSizes([460, 540])


    # =========================================================================
    # Dataset intake and selection
    # =========================================================================
    @pyqtSlot(list)
    def handle_incoming_paths(self, paths: List[Path]):
        if self._closing or self.archive_worker or (self.worker and self.worker.isRunning()):
            return
        self._pending_import_paths = list(dict.fromkeys(
            self._pending_import_paths + [Path(path).resolve() for path in paths]))
        if self.import_worker is None:
            self._launch_import()

    def _launch_import(self):
        if self._closing or not self._pending_import_paths:
            return
        paths, self._pending_import_paths = self._pending_import_paths, []
        worker = EvaluationImportWorker(paths, self)
        self.import_worker = worker
        for button in (self.btn_run_all, self.btn_clear, self.btn_export_summary, self.btn_copy_raw):
            button.setEnabled(False)
        worker.result_ready.connect(self._on_import_result)
        worker.failed.connect(self._append_log)
        worker.finished.connect(lambda worker=worker: self._import_finished(worker))
        worker.start()

    def _on_import_result(self, matched):
        if self._closing:
            return
        if not matched:
            QMessageBox.warning(self, "Import Notice", "No valid trajectory CSV files were found.")
            return
        self.task_context.update(matched)
        incoming = [(Path(data["csv"]), Path(data["telemetry_csv"]) if data.get("telemetry_csv") else None)
                    for data in matched.values() if data.get("csv")]
        self.matched_tasks = list(dict.fromkeys(self.matched_tasks + incoming))
        self._refresh_table()

    def _import_finished(self, worker):
        if self.import_worker is worker:
            self.import_worker = None
        worker.deleteLater()
        if self._closing:
            return
        if self._pending_import_paths:
            self._launch_import()
        else:
            self.btn_run_all.setEnabled(bool(self.matched_tasks))
            for button in (self.btn_clear, self.btn_export_summary, self.btn_copy_raw):
                button.setEnabled(True)

    def _refresh_table(self):
        self.lbl_task_count.setText(f"Ready ({len(self.matched_tasks)} tasks queued)")
        self.table_results.setRowCount(len(self.matched_tasks))

        for idx, (source_path, telem_path) in enumerate(self.matched_tasks):
            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked)
            self.table_results.setItem(idx, 0, chk_item)

            self.table_results.setItem(idx, 1, QTableWidgetItem(str(idx + 1)))
            source_item = QTableWidgetItem(experiment_name(source_path))
            source_item.setToolTip(str(source_path))
            self.table_results.setItem(idx, 2, source_item)
            self.table_results.setItem(idx, 3, QTableWidgetItem("Checking files..."))
            self.table_results.setItem(idx, 4, QTableWidgetItem("Ready"))

        self._queue_file_checks()
        if self.matched_tasks and not self.txt_summary_dest.text():
            first_source = self.matched_tasks[0][0]
            s_dir = session_root(first_source)
            suggested_out = s_dir.parent / "Cross_Session_Reports"
            self.txt_summary_dest.setText(str(suggested_out))

    def showEvent(self, event):
        super().showEvent(event)
        if self.matched_tasks and not self._closing:
            self._queue_file_checks()

    def _queue_file_checks(self):
        self._file_check_serial += 1
        contexts = {}
        for data in self.task_context.values():
            path = data.get("csv") or data.get("position_csv") or data.get("cleaned_csv") or data.get("raw_csv")
            if path:
                contexts[str(Path(path)).casefold()] = data
        requests = tuple((source, copy.deepcopy(contexts.get(str(source).casefold(), {})))
                         for source, _ in self.matched_tasks)
        self._pending_file_checks = (self._file_check_serial, requests)
        if self._file_check_worker is not None:
            self._file_check_worker.cancel()
        else:
            self._launch_file_checks()

    def _launch_file_checks(self):
        if self._closing or self._pending_file_checks is None:
            return
        serial, requests = self._pending_file_checks
        self._pending_file_checks = None
        if not requests:
            return
        worker = SessionCheckWorker(serial, requests, self)
        self._file_check_worker = worker
        worker.result_ready.connect(self._on_file_check_result)
        worker.finished.connect(lambda worker=worker: self._file_checks_finished(worker))
        worker.start()

    def _on_file_check_result(self, serial, source, result):
        if self._closing or serial != self._file_check_serial:
            return
        for row, (path, _) in enumerate(self.matched_tasks):
            if str(path).casefold() == source.casefold():
                item = QTableWidgetItem(result["text"].replace(" | ", "\n"))
                item.setToolTip(result["tooltip"])
                item.setForeground(QColor("#15803D" if result["complete"] else "#C2410C"))
                self.table_results.setItem(row, 3, item)
                if not (self.worker and self.worker.isRunning()):
                    state_item = QTableWidgetItem(result.get('analysis_state', 'Not analysed'))
                    state_item.setToolTip(result['tooltip'])
                    self.table_results.setItem(row, 4, state_item)
                self.table_results.resizeRowToContents(row)
                break

    def _file_checks_finished(self, worker):
        if self._file_check_worker is worker:
            self._file_check_worker = None
        worker.deleteLater()
        self._launch_file_checks()

    def _set_all_checks(self, state: bool):
        for r in range(self.table_results.rowCount()):
            item = self.table_results.item(r, 0)
            if item:
                item.setCheckState(Qt.Checked if state else Qt.Unchecked)

    def _get_checked_tasks(self) -> List[Tuple[int, Tuple[Path, Optional[Path]]]]:
        checked = []
        for r in range(self.table_results.rowCount()):
            item = self.table_results.item(r, 0)
            if item and item.checkState() == Qt.Checked and r < len(self.matched_tasks):
                checked.append((r, self.matched_tasks[r]))
        return checked

    def clear_tasks(self):
        if self.import_worker or self.archive_worker or (self.worker and self.worker.isRunning()):
            return
        self.matched_tasks.clear()
        self.task_context.clear()
        self.evaluated_df = None
        self._refresh_table()
        self.btn_run_all.setEnabled(False)

    # =========================================================================
    # Evaluation execution
    # =========================================================================
    def run_evaluation(self):
        if self.archive_worker or self.import_worker:
            return
        checked_pairs = self._get_checked_tasks()
        if not checked_pairs or (self.worker and self.worker.isRunning()):
            QMessageBox.warning(self, "No Tasks Selected", "Please select at least one task to evaluate.")
            return

        missing = [source.name for _, (source, _) in checked_pairs if not source.is_file()]
        if missing:
            QMessageBox.warning(self, "Missing Position CSV",
                "Run tracking before evaluating these tasks:\n" + "\n".join(missing))
            return
        self._running_rows = [row for row, _ in checked_pairs]
        tasks_to_run = [pair for _, pair in checked_pairs]
        self._evaluation_completed = self._evaluation_failed = 0
        self._set_status(f"Evaluating 0/{len(tasks_to_run)}", 0)
        self.btn_run_all.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.drop_area.setEnabled(False)

        params = AnaesthesiaParams(
            still_window=self.spin_still.value(),
            still_speed=self.spin_speed.value(),
            drop_height=self.spin_drop.value(),
            sedation_ratio=self.spin_ratio.value(),
        )
        self.worker = EvaluationWorker(
            tasks_to_run,
            self.spin_onset.value(),
            params,
            self.check_plots.isChecked(),
            self.check_heatmaps.isChecked(),
            self.check_summary_only.isChecked(),
            self.check_median_plot.isChecked()
        )
        self.check_median_plot.setEnabled(False)
        self.check_summary_only.setEnabled(False)
        self.check_plots.setEnabled(False)
        self.check_heatmaps.setEnabled(False)
        self.btn_export_summary.setEnabled(False)
        self.btn_copy_raw.setEnabled(False)
        self.worker.result_ready.connect(self._on_result)
        self.worker.progress.connect(self._on_analysis_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _summary_mode_changed(self, checked):
        self.check_median_plot.setEnabled(not checked)
        self.check_plots.setEnabled(not checked)
        self.check_heatmaps.setEnabled(not checked)

    def _on_analysis_progress(self, index, message, fraction):
        total = max(1, len(self._running_rows))
        text = f'Video {index + 1}/{total}: {message}'
        self._set_status(text, (index + fraction) / total * 100)

    def _on_result(self, sub_idx, evaluated, message):
        # Results are already committed; keep no per-frame data in the UI.
        self.evaluated_df = None
        if sub_idx < len(self._running_rows):
            table_row = self._running_rows[sub_idx]
            status_txt = ("Saved: " if evaluated is not None else "Failed: ") + message
            status_item = QTableWidgetItem(status_txt)
            status_item.setToolTip(status_txt)
            self.table_results.setItem(table_row, 4, status_item)
            self.table_results.resizeRowToContents(table_row)
        self._evaluation_completed += 1
        self._evaluation_failed += int(evaluated is None)
        total = max(1, len(self._running_rows))
        self._set_status(f"Evaluating {self._evaluation_completed}/{total}",
                         self._evaluation_completed / total * 100)
        if evaluated is None:
            self.lbl_task_count.setToolTip(message)

    def _on_finished(self):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.deleteLater()
        self._set_status("Finished with errors" if self._evaluation_failed else "Finished",
                         self._evaluation_completed / max(1, len(self._running_rows)) * 100)
        self.btn_run_all.setEnabled(bool(self.matched_tasks))
        self.btn_clear.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.btn_export_summary.setEnabled(True)
        self.btn_copy_raw.setEnabled(True)
        self.check_summary_only.setEnabled(True)
        self._summary_mode_changed(self.check_summary_only.isChecked())

    # =========================================================================
    # Cross-session summary and detection extraction
    # =========================================================================
    def _choose_summary_destination(self):
        d = QFileDialog.getExistingDirectory(self, "Select Cross-Session Export Destination")
        if d:
            self.txt_summary_dest.setText(d)

    def _get_checked_session_dirs(self) -> List[Path]:
        checked_dirs = set()
        for _, (source, _) in self._get_checked_tasks():
            s_dir = session_root(source)
            checked_dirs.add(s_dir)
        return sorted(checked_dirs, key=lambda path: str(path).casefold())

    def _run_cross_session_summary(self):
        dest_dir = self.txt_summary_dest.text().strip()
        if not dest_dir:
            QMessageBox.warning(self, "Destination Required", "Please choose a destination directory for the summary report.")
            return

        checked_dirs = self._get_checked_session_dirs()
        if not checked_dirs:
            QMessageBox.warning(self, "No Sessions Selected", "Check at least one session in the table.")
            return

        self._start_archive("summary", checked_dirs, dest_dir)

    def _extract_raw_csvs(self):
        dest_dir = self.txt_summary_dest.text().strip()
        if not dest_dir:
            QMessageBox.warning(self, "Destination Required", "Please choose a destination directory to extract Raw CSVs.")
            return

        checked_tasks = self._get_checked_tasks()
        if not checked_tasks:
            QMessageBox.warning(self, "No Tasks Selected", "Check at least one task in the table.")
            return

        self._start_archive("raw", [source for _, (source, _) in checked_tasks], dest_dir)

    def _start_archive(self, mode, paths, destination):
        if self.archive_worker or self.import_worker or (self.worker and self.worker.isRunning()):
            self._append_log("Wait for the current task to finish.")
            return
        worker = ArchiveWorker(mode, paths, destination, self)
        self.archive_worker = worker
        self._archive_failed = self._archive_result_received = False
        self._set_status("Collecting summary" if mode == "summary" else "Extracting Raw CSVs", 0)
        worker.message.connect(self._append_log)
        worker.result_ready.connect(self._on_archive_result)
        worker.failed.connect(self._on_archive_failed)
        worker.finished.connect(self._on_archive_finished)
        self.btn_export_summary.setEnabled(False)
        self.btn_copy_raw.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        worker.start()

    def _on_archive_result(self, result):
        if self._closing:
            return
        self._archive_result_received = True
        count, target = result["count"], result["path"]
        self._set_status("Finished" if count else "No results found", 100)
        if result["mode"] == "summary":
            if target is None:
                QMessageBox.information(self, "No Summary Files Found",
                    "None of the selected sessions have *_summary.csv yet.\nRun evaluation first.")
                return
            files = result.get('paths', [])
            self.lbl_task_count.setToolTip('\n'.join(str(path) for path in files))
            QMessageBox.information(self, "Summary Report Exported",
                f"Exported {len(files)} sorted metric table(s) from {count} chamber records "
                f"across {result['sessions']} selected folder(s):\n" +
                '\n'.join(str(path) for path in files))
        elif count:
            self.lbl_task_count.setToolTip(f"Extracted {count} Raw CSVs: {target}")
            QMessageBox.information(self, "Raw CSVs Extracted",
                f"Successfully extracted {count} Raw CSVs to:\n{target}")
        else:
            QMessageBox.information(self, "No Raw CSVs Found",
                "No *_raw.csv files were found for the selected tasks.")

    def _on_archive_failed(self, message):
        self._archive_failed = True
        self._set_status("Failed")
        self.lbl_task_count.setToolTip(message)

    def _on_archive_finished(self):
        worker, self.archive_worker = self.archive_worker, None
        if worker:
            worker.deleteLater()
        if not self._closing:
            if not self._archive_failed and not self._archive_result_received:
                self._set_status("Cancelled")
            self.btn_export_summary.setEnabled(True)
            self.btn_copy_raw.setEnabled(True)
            self.btn_run_all.setEnabled(bool(self.matched_tasks))

    def cleanup(self):
        self._closing = True
        self._pending_file_checks = None
        self._pending_import_paths = []
        for worker in (self.worker, self.archive_worker, self.import_worker, self._file_check_worker):
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait()

    def _set_status(self, state: str, percent=None):
        if self._closing:
            return
        self.status_panel.show()
        self.lbl_task_count.setText(state)
        if percent is not None:
            self.lbl_percent.setText(f"{max(0, min(100, percent)):.0f}%")

    def _append_log(self, text: str):
        """Display only the current operation, without a log history."""
        self._set_status(text)
