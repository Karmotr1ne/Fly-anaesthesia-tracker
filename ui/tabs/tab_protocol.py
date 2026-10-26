"""
ui/tabs/tab_protocol.py
Protocol designer and plate geometry configuration tab for FATer.

Features:
- Single-page compact layout (zero page overflow, properly scaled).
- Plate Design at Top-Left (global geometry input).
- Gas Saturation & Behaviour Safety: Relaxed max safe flow (< 18 cm/s) to prevent
  fly freezing/shear-stress fear state while allowing high flow experiments.
- Left-bottom: Compact Saved Protocol Library and File Storage (New, Browse, Save).
- Right side: Row action buttons integrated tightly above the step table.
- Full signal integration with main.py (protocol_applied, plate_geometry_changed).
- Fully documented in English.
"""

from pathlib import Path
from typing import List, Dict, Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHeaderView, QDoubleSpinBox, QSpinBox, QComboBox, QListWidget, QMessageBox, QFileDialog, QFrame, QSizePolicy

from core.hardware.protocol_scheduler import ProtocolScheduler
from core.protocol.flowvision_model import FlowVisionScriptModel

from core.protocol.saturation_calculator import SaturationCalculator
from core.protocol.library import protocol_root


from ui.styles.theme import style_page

class _FlowHint(QLabel):
    """Keep flow direction and column timings on one line at any panel width."""
    def __init__(self, text):
        super().__init__()
        self._full_text = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setText(text)

    def setText(self, text):
        self._full_text = text
        self.setToolTip(text)
        super().setText(self.fontMetrics().elidedText(text, Qt.ElideRight, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self._full_text)


class ProtocolTab(QWidget):
    # Signals communicating with FATer main window
    protocol_applied = pyqtSignal(str, float, list)     # (protocol_name, mix_onset_sec, steps)
    plate_geometry_changed = pyqtSignal(int, int)       # (rows, cols)

    def __init__(self, protocols_dir: Optional[str] = None, parent: Optional[QWidget] = None, *args, **kwargs):
        super().__init__(parent)

        self.protocols_dir = protocol_root(protocols_dir) / 'air valve'

        self.active_protocol_name = "Custom Protocol"

        self.init_ui()
        style_page(self)
        self.populate_protocol_library()
        self._apply_calculated_flows()
        self.update_saturation_estimate()

    @staticmethod
    def _add_section_header(section, title):
        layout = section.layout()
        heading = QLabel(title)
        heading.setProperty("role", "section")
        divider = QFrame()
        divider.setObjectName("protocolDivider")
        divider.setFixedHeight(1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.insertWidget(0, divider)
        layout.insertWidget(0, heading)

    def init_ui(self):
        self.setObjectName("protocolTab")
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 8, 12, 8)
        main_layout.setSpacing(14)

        # =====================================================================
        # LEFT PANEL: Geometry, Saturation Safety, and Compact File Storage
        # =====================================================================
        left_column = QVBoxLayout()
        left_column.setSpacing(14)

        # 1. Plate Design Box
        grp_plate = QFrame()
        v_plate = QVBoxLayout(grp_plate)
        v_plate.setContentsMargins(0, 6, 0, 0)
        v_plate.setSpacing(8)

        h_preset = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_label.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        preset_label.setFixedHeight(26)
        h_preset.addWidget(preset_label)
        self.combo_preset = QComboBox()
        self.combo_preset.addItems([
            "4 Rows × 3 Cols (Standard 12-Well)",
            "4 Rows × 2 Cols (8-Well)",
            "6 Rows × 4 Cols (24-Well)",
            "Custom Configuration..."
        ])
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.combo_preset.setMinimumContentsLength(18)
        self.combo_preset.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_preset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_preset.addWidget(self.combo_preset, 1)
        v_plate.addLayout(h_preset)

        grid_dims = QGridLayout()
        grid_dims.addWidget(QLabel("Parallel channels (Rows):"), 0, 0)
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 16)
        self.spin_rows.setValue(4)
        self.spin_rows.valueChanged.connect(self._on_geometry_changed)
        grid_dims.addWidget(self.spin_rows, 0, 1)

        grid_dims.addWidget(QLabel("Serial chambers (Cols):"), 1, 0)
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 12)
        self.spin_cols.setValue(3)
        self.spin_cols.valueChanged.connect(self._on_geometry_changed)
        grid_dims.addWidget(self.spin_cols, 1, 1)
        grid_dims.setHorizontalSpacing(8)
        grid_dims.setVerticalSpacing(8)
        self.spin_rows.setMaximumWidth(120)
        self.spin_cols.setMaximumWidth(120)
        v_plate.addLayout(grid_dims)

        grp_plate.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left_column.addWidget(grp_plate)

        # 2. Gas Saturation & Behaviour Disturbance Limit
        grp_dynamics = QFrame()
        v_dyn = QVBoxLayout(grp_dynamics)
        v_dyn.setContentsMargins(0, 6, 0, 0)
        v_dyn.setSpacing(8)

        self.lbl_dead_volume = QLabel("Dead Volume Latency: 4.98 s")
        self.lbl_saturation = QLabel("99% Saturation Time: 10.9 s (0.18 min)")
        self.lbl_saturation.setStyleSheet("color: #0F172A; background: #EFF6FF; border-radius: 4px; padding: 4px 6px; font-weight: 600;")
        self.lbl_linear_speed = QLabel("Chamber Linear Speed: 5.77 cm/s")

        # Relaxed safe flow limit: 18 cm/s ensures no freezing or fear response
        self.lbl_max_safe_flow = QLabel("Calculated Flow Limit: 623.5 sccm")
        self.lbl_max_safe_flow.setStyleSheet("color: #0F172A; background: #EFF6FF; border-radius: 4px; padding: 4px 6px; font-weight: 600;")

        self.lbl_row_progress = _FlowHint("Flow: top/right -> CH1 | Top -> Bottom: calculating...")
        self.lbl_row_progress.setWordWrap(False)
        self.lbl_row_progress.setStyleSheet("color: #64748B; font-size: 10px;")

        self.lbl_max_flow_tip = QLabel("*Speed < 18 cm/s avoids shear-induced freezing fear behaviour.")
        self.lbl_max_flow_tip.setStyleSheet("color: #64748B; ")

        v_dyn.addWidget(self.lbl_dead_volume)
        v_dyn.addWidget(self.lbl_linear_speed)
        highlighted = QFrame()
        highlighted.setObjectName("protocolMetrics")
        highlighted_layout = QVBoxLayout(highlighted)
        highlighted_layout.setContentsMargins(0, 0, 0, 0)
        highlighted_layout.setSpacing(8)
        highlighted_layout.addWidget(self.lbl_saturation)
        highlighted_layout.addWidget(self.lbl_max_safe_flow)
        v_dyn.addWidget(highlighted)
        v_dyn.addWidget(self.lbl_row_progress)
        for output in (self.lbl_dead_volume, self.lbl_linear_speed, self.lbl_saturation,
                       self.lbl_max_safe_flow, self.lbl_row_progress):
            output.setFixedHeight(26)
            output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.lbl_max_flow_tip.hide()
        grp_dynamics.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        left_column.addWidget(grp_dynamics)

        # 3. Compact Saved Protocol Library & Storage
        grp_lib = QFrame()
        v_lib = QVBoxLayout(grp_lib)
        v_lib.setContentsMargins(0, 6, 0, 0)
        v_lib.setSpacing(6)

        self.list_protocols = QListWidget()
        self.list_protocols.setMinimumHeight(96)
        self.list_protocols.itemClicked.connect(self._on_protocol_item_clicked)
        v_lib.addWidget(self.list_protocols, 1)

        # Compact File Operations
        h_file_btns = QHBoxLayout()
        h_file_btns.setSpacing(6)

        self.btn_new = QPushButton("New")
        self.btn_new.setStyleSheet("padding: 4px 8px; background-color: #F8FAFC; border: 1px solid #E2E8F0;")

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setStyleSheet("padding: 4px 8px; background-color: #F8FAFC; border: 1px solid #E2E8F0;")

        self.btn_save = QPushButton("Save Protocol")
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #EFF6FF;
                color: #1D4ED8;
                font-weight: 400;
                border: 1px solid #C7D2FE;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #E0E7FF;
            }
        """)

        h_file_btns.addWidget(self.btn_new)
        h_file_btns.addWidget(self.btn_browse)
        h_file_btns.addWidget(self.btn_save)
        v_lib.addLayout(h_file_btns)

        left_column.addWidget(grp_lib, 1)
        main_layout.addLayout(left_column, 4)

        # =====================================================================
        # RIGHT PANEL: Integrated Sequence Editor & Dispatch (One-Page Locked)
        # =====================================================================
        right_column = QVBoxLayout()
        right_column.setSpacing(14)

        # Step Sequence Editor Group
        grp_seq = QFrame()
        v_seq = QVBoxLayout(grp_seq)
        v_seq.setContentsMargins(0, 6, 0, 0)
        v_seq.setSpacing(6)

        # Row actions remain below the table with neutral visual weight.
        h_row_actions = QHBoxLayout()
        h_row_actions.setSpacing(6)

        self.btn_add = QPushButton("Add Step")
        self.btn_add.setStyleSheet("")

        self.btn_del = QPushButton("Delete Step")
        self.btn_del.setStyleSheet("")

        self.btn_up = QPushButton("Move Up")
        self.btn_up.setStyleSheet("")

        self.btn_down = QPushButton("Move Down")
        self.btn_down.setStyleSheet("")

        for action in (self.btn_add, self.btn_del, self.btn_up, self.btn_down):
            action.setProperty("role", "rowAction")
            action.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        h_row_actions.addWidget(self.btn_add, 1)
        h_row_actions.addWidget(self.btn_del, 1)
        h_row_actions.addWidget(self.btn_up, 1)
        h_row_actions.addWidget(self.btn_down, 1)

        # Step Sequence Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Phase",
            "Duration (s)",
            "O2 (sccm)",
            "Anes (sccm)",
            "N2 (sccm)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(26)  # Compact row height
        self.table.itemChanged.connect(lambda: self.update_saturation_estimate())
        v_seq.addWidget(self.table, 1)
        v_seq.addLayout(h_row_actions)

        self.table.setMinimumHeight(100)
        grp_seq.setMinimumHeight(180)
        grp_seq.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_column.addWidget(grp_seq, 2)

        # Bottom Bar: Execution Workbench Alignment
        grp_bottom = QFrame()
        execution_layout = QVBoxLayout(grp_bottom)
        execution_layout.setContentsMargins(0, 0, 0, 0)
        execution_layout.setSpacing(8)
        h_bottom = QGridLayout()
        execution_layout.addLayout(h_bottom)
        execution_layout.addStretch(1)
        h_bottom.setContentsMargins(0, 0, 0, 0)
        h_bottom.setHorizontalSpacing(8)
        h_bottom.setVerticalSpacing(12)
        h_bottom.setColumnStretch(0, 1)

        h_bottom.addWidget(QLabel("Mix Gas Onset(s):"), 0, 0)
        self.spin_onset = QDoubleSpinBox()
        self.spin_onset.setRange(0.0, 7200.0)
        self.spin_onset.setValue(0.0)
        self.spin_onset.setSuffix(" s")
        self.spin_onset.setMinimumWidth(220)
        self.spin_onset.setAlignment(Qt.AlignRight)
        h_bottom.addWidget(self.spin_onset, 0, 1, Qt.AlignRight)

        self.lbl_total_duration = QLabel("Total Duration: 360.0 s (06m 00s)")
        self.lbl_total_duration.setFixedHeight(26)
        self.lbl_total_duration.setStyleSheet("font-weight: 600; color: #1D4ED8;")
        h_bottom.addWidget(self.lbl_total_duration, 1, 0)

        self.btn_apply = QPushButton("Apply to Main Acquisition")
        self.btn_apply.setFixedHeight(34)
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #1D4ED8;
                color: #FFFFFF;
                font-weight: 600;
                
                border-radius: 5px;
                padding: 4px 16px;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        h_bottom.addWidget(self.btn_apply, 1, 1, Qt.AlignRight)

        grp_bottom.setMinimumHeight(180)
        grp_bottom.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        right_column.addWidget(grp_bottom, 1)
        main_layout.addLayout(right_column, 7)

        for section, title in (
            (grp_plate, "1. Plate Design"), (grp_dynamics, "2. Gas Dynamics"),
            (grp_lib, "3. Protocol Library"), (grp_seq, "4. Protocol Steps"),
            (grp_bottom, "5. Execution"),
        ):
            self._add_section_header(section, title)

        # =====================================================================
        # Signal Connections & Defaults
        # =====================================================================
        self.btn_new.clicked.connect(self._on_new_protocol)
        self.btn_browse.clicked.connect(self._on_browse_directory)
        self.btn_save.clicked.connect(self._on_save_protocol)

        self.btn_add.clicked.connect(lambda checked=False: self._add_row())
        self.spin_onset.valueChanged.connect(self._update_total_duration)
        self.btn_del.clicked.connect(self._del_row)
        self.btn_up.clicked.connect(self._move_row_up)
        self.btn_down.clicked.connect(self._move_row_down)

        self.btn_apply.clicked.connect(self._apply_protocol)

        # Standard initial rows
        self._add_row(60.0, 200.0, 0.0, 0.0)
        self._add_row(240.0, 180.0, 20.0, 0.0)
        self._add_row(60.0, 200.0, 0.0, 0.0)

    # -------------------------------------------------------------------------
    # Geometry & Dynamics Callbacks
    # -------------------------------------------------------------------------
    def _on_preset_changed(self, idx: int):
        presets = {0: (4, 3), 1: (4, 2), 2: (6, 4)}
        if idx not in presets:
            return
        rows, cols = presets[idx]
        if (self.spin_rows.value(), self.spin_cols.value()) == (rows, cols):
            return
        controls = (self.spin_rows, self.spin_cols)
        previous = [control.blockSignals(True) for control in controls]
        try:
            self.spin_rows.setValue(rows)
            self.spin_cols.setValue(cols)
        finally:
            for control, blocked in zip(controls, previous):
                control.blockSignals(blocked)
        self._on_geometry_changed()

    def _on_geometry_changed(self):
        r = self.spin_rows.value()
        c = self.spin_cols.value()
        self.plate_geometry_changed.emit(r, c)
        if hasattr(self, "table"):
            self._apply_calculated_flows()
            self.update_saturation_estimate()

    def _apply_calculated_flows(self):
        calc = SaturationCalculator(rows=self.spin_rows.value(), cols=self.spin_cols.value())
        updates = []
        for row in range(self.table.rowCount()):
            try:
                anaesthetic = float(self.table.item(row, 3).text())
                updates.append((row, calc.carrier_flows(anaesthetic)))
            except (AttributeError, ValueError) as exc:
                QMessageBox.warning(self, "Flow Not Applied", str(exc))
                return False
        previous = self.table.blockSignals(True)
        try:
            for row, flows in updates:
                for col, field in ((2, "o2"), (3, "anes"), (4, "n2")):
                    self.table.setItem(row, col, QTableWidgetItem(f"{flows[field]:.1f}"))
        finally:
            self.table.blockSignals(previous)
        return True

    def update_saturation_estimate(self):
        """Recalculate dead volume latency, saturation, and relaxed safety flow."""
        rows = self.spin_rows.value()
        cols = self.spin_cols.value()

        try:
            steps = self._extract_steps()
        except (ValueError, TypeError):
            return
        total_flow = 200.0
        max_step_flow = 0.0
        if steps:
            total_flow = max(10.0, steps[0]["o2"] + steps[0]["anes"] + steps[0]["n2"])
            max_step_flow = max(s["o2"] + s["anes"] + s["n2"] for s in steps)

        calc = SaturationCalculator(rows=rows, cols=cols, total_flow_sccm=total_flow)
        mixture = steps[0] if steps else {"o2": total_flow, "anes": 0.0, "n2": 0.0}
        metrics = calc.compute_metrics(
            q_o2_sccm=mixture["o2"], q_anes_sccm=mixture["anes"], q_n2_sccm=mixture["n2"]
        )

        # Relaxed safe flow limit: 18.0 cm/s prevents fear/freezing defense response
        area_cm2 = metrics.geometry_summary.get("single_cross_area_cm2", 0.1924)
        max_safe_flow_sccm = calc.maximum_flow()

        self.lbl_dead_volume.setText(f"Dead Volume Latency: {metrics.dead_volume_latency_s:.2f} s")
        self.lbl_saturation.setText(
            f"99% Full Saturation Time: {metrics.full_saturation_s:.1f} s ({metrics.full_saturation_s/60:.2f} min)"
        )
        self.lbl_linear_speed.setText(f"Chamber Linear Speed: {metrics.linear_velocity_cm_s:.2f} cm/s")

        # Behavioural freeze warning check
        if max_step_flow > max_safe_flow_sccm:
            self.lbl_max_safe_flow.setText(
                f"Calculated Flow Limit: {max_safe_flow_sccm:.1f} sccm (Warning: High Wind > {max_step_flow:.1f} sccm!)"
            )
            self.lbl_max_safe_flow.setStyleSheet("color: #0F172A; background: #EFF6FF; border-radius: 4px; padding: 4px 6px; font-weight: 600;")
        else:
            self.lbl_max_safe_flow.setText(f"Calculated Flow Limit: {max_safe_flow_sccm:.1f} sccm")
            self.lbl_max_safe_flow.setStyleSheet("color: #0F172A; background: #EFF6FF; border-radius: 4px; padding: 4px 6px; font-weight: 600;")

        prog_items = []
        for c in range(cols - 1, -1, -1):
            first_ch = c * rows + 1
            last_ch = (c + 1) * rows
            prog_items.append(f"CH{first_ch}-{last_ch}:{metrics.row_progress_s[c]:.1f}s")
        self.lbl_row_progress.setText(f"Flow: top/right -> CH1 | Top -> Bottom: {' / '.join(prog_items)}")
        self._update_total_duration()

    # -------------------------------------------------------------------------
    # Table Sequence Operations
    # -------------------------------------------------------------------------
    def _add_row(self, duration: float = 60.0, o2: float = 200.0, anes: float = 0.0, n2: float = 0.0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(f"Step {r+1}"))
        self.table.setItem(r, 1, QTableWidgetItem(str(duration)))
        self.table.setItem(r, 2, QTableWidgetItem(str(o2)))
        self.table.setItem(r, 3, QTableWidgetItem(str(anes)))
        self.table.setItem(r, 4, QTableWidgetItem(str(n2)))
        self._update_total_duration()

    def _del_row(self):
        curr = self.table.currentRow()
        if curr >= 0:
            self.table.removeRow(curr)
            self._relabel_phases()
            self.update_saturation_estimate()

    def _move_row_up(self):
        curr = self.table.currentRow()
        if curr > 0:
            self._swap_rows(curr, curr - 1)
            self.table.setCurrentCell(curr - 1, 0)

    def _move_row_down(self):
        curr = self.table.currentRow()
        if 0 <= curr < self.table.rowCount() - 1:
            self._swap_rows(curr, curr + 1)
            self.table.setCurrentCell(curr + 1, 0)

    def _swap_rows(self, r1: int, r2: int):
        for c in range(self.table.columnCount()):
            item1 = self.table.takeItem(r1, c)
            item2 = self.table.takeItem(r2, c)
            self.table.setItem(r1, c, item2)
            self.table.setItem(r2, c, item1)
        self._relabel_phases()

    def _relabel_phases(self):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item:
                item.setText(f"Step {r+1}")

    def _update_total_duration(self):
        try:
            total_sec = sum(step["time"] for step in self._extract_steps()) + self.spin_onset.value()
        except (ValueError, TypeError):
            self.lbl_total_duration.setText("Total Duration: invalid protocol")
            return
        mins = int(total_sec // 60)
        secs = int(total_sec % 60)
        self.lbl_total_duration.setText(f"Total Duration: {total_sec:.1f} s ({mins:02d}m {secs:02d}s)")

    def _extract_steps(self) -> List[Dict[str, float]]:
        steps = []
        for r in range(self.table.rowCount()):
            try:
                steps.append({
                    "time": float(self.table.item(r, 1).text()),
                    "o2": float(self.table.item(r, 2).text()),
                    "anes": float(self.table.item(r, 3).text()),
                    "n2": float(self.table.item(r, 4).text()),
                })
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Invalid values in protocol row {r + 1}") from exc
        return ProtocolScheduler.validate_steps(steps)

    # -------------------------------------------------------------------------
    # Protocol File I/O Management
    # -------------------------------------------------------------------------
    def populate_protocol_library(self):
        self.list_protocols.clear()
        if not self.protocols_dir.exists():
            return
        files = sorted((path for path in self.protocols_dir.rglob('*')
                        if path.is_file() and path.suffix.casefold() == '.fv'),
                       key=lambda path: str(path.relative_to(self.protocols_dir)).casefold())
        for file in files:
            self.list_protocols.addItem(str(file.relative_to(self.protocols_dir)))
            item = self.list_protocols.item(self.list_protocols.count() - 1)
            item.setData(Qt.UserRole, str(file.resolve()))
            item.setToolTip(str(file.resolve()))

    def _on_protocol_item_clicked(self, item):
        file_path = Path(item.data(Qt.UserRole) or self.protocols_dir / item.text())
        if file_path.is_file():
            try:
                self._load_protocol_from_file(file_path)
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(self, 'Protocol Not Loaded', str(exc))

    def _load_protocol_from_file(self, file_path: Path):
        self.active_protocol_name = file_path.stem
        data = FlowVisionScriptModel.parse_script(file_path)
        self.table.setRowCount(0)
        for s in data.get("steps", []):
            self._add_row(
                duration=s.get("time", 60.0),
                o2=s.get("o2", 200.0),
                anes=s.get("anes", 0.0),
                n2=s.get("n2", 0.0),
            )
        self.spin_onset.setValue(data.get('mix_onset', 0.0))
        self.update_saturation_estimate()

    def _on_new_protocol(self):
        self.spin_onset.setValue(0.0)
        self.active_protocol_name = "New Protocol"
        self.table.setRowCount(0)
        self._add_row(60.0, 200.0, 0.0, 0.0)
        self.update_saturation_estimate()

    def _on_browse_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Protocol Directory", str(self.protocols_dir))
        if dir_path:
            self.protocols_dir = Path(dir_path)
            self.populate_protocol_library()

    def _on_save_protocol(self):
        try:
            steps = self._extract_steps()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid Protocol", str(exc))
            return
        if not steps:
            QMessageBox.warning(self, "Warning", "Protocol table is empty. Add steps before saving.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Protocol File", str(self.protocols_dir / "protocol_custom.fv"), "FlowVision Script (*.fv)"
        )
        if file_path:
            p = Path(file_path)
            self.active_protocol_name = p.stem
            model = FlowVisionScriptModel()
            try:
                model.export_script(
                    file_path=p,
                    steps=steps,
                    mix_onset=self.spin_onset.value(),
                    total_duration=sum(s["time"] for s in steps),
                    port=getattr(getattr(self, "hardware_config", None), "fv_com_port", "COM3"),
                )
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(self, "Protocol Not Saved", str(exc))
                return
            self.populate_protocol_library()
            QMessageBox.information(self, "Saved", f"Protocol successfully saved to:\n{p.name}")

    def _apply_protocol(self):
        self.update_saturation_estimate()
        try:
            steps = self._extract_steps()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid Protocol", str(exc))
            return
        onset_sec = self.spin_onset.value()

        self.protocol_applied.emit(self.active_protocol_name, onset_sec, steps)

