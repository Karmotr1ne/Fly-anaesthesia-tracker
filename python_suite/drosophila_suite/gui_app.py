"""
Desktop Application (PyQt6)
===========================
Workstation Desktop GUI: Supports drag-and-drop file pairing, interactive multi-chamber grid calibration,
fly vision tracking, kinematic cleaning, dual-Y & kymograph plotting, anesthesia induction kinetics analysis,
and annotated video synthesis.
"""

import os
import re
import sys
import time
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import pandas as pd
import numpy as np

try:
    from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QPointF, QRectF
    from PyQt6.QtGui import QImage, QColor, QPen, QPainter, QFont, QBrush, QCursor, QKeySequence, QShortcut
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QDialog,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QCheckBox,
        QGroupBox,
        QListWidget,
        QMessageBox,
        QFrame,
        QFileDialog,
        QSpinBox,
        QDoubleSpinBox,
        QRadioButton,
        QComboBox,
        QScrollArea,
    )
except ImportError:
    pass

import cv2

try:
    from .pipeline import DrosophilaBehaviorPipeline
    from .models import PipelineConfig
    from .cleaner import KinematicCleaner
    from .anesthesia import AnesthesiaAnalyzer
    from .visualizer import ScientificVisualizer
    from .tracker import (
        RobustGridAligner,
        SymmetricGridAligner,
        Interactive8ChamberCalibrator,
        RobustFlyTracker,
        FlyVisionTracker,
        get_video_metadata,
    )
except (ImportError, ValueError):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from drosophila_suite.pipeline import DrosophilaBehaviorPipeline
        from drosophila_suite.models import PipelineConfig
        from drosophila_suite.cleaner import KinematicCleaner
        from drosophila_suite.anesthesia import AnesthesiaAnalyzer
        from drosophila_suite.visualizer import ScientificVisualizer
        from drosophila_suite.tracker import (
            RobustGridAligner,
            SymmetricGridAligner,
            Interactive8ChamberCalibrator,
            RobustFlyTracker,
            FlyVisionTracker,
            get_video_metadata,
        )
    except ImportError:
        pass


class SessionPhase(str, Enum):
    IDLE = "Idle"
    READING_CSV = "Loading CSV"
    TRACKING = "Vision Tracking"
    CLEANING = "Kinematic Cleaning"
    PLOTTING_EARLY = "Generating Plots (Dual Y & Kymo)"
    ANALYZING = "Anesthesia Kinetics (State Machine)"
    RENDERING = "Rendering Overlay Video"
    COMPLETED = "Completed"
    FAILED = "Failed"


# =====================================================================
# Drag & Drop File Import Area
# =====================================================================
class DragDropArea(QFrame):
    filesChanged = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.setLineWidth(2)
        self.setMinimumHeight(140)
        self.default_style = "QFrame { border: 2px dashed #90A4AE; border-radius: 8px; background-color: #FAFAFA; }"
        self.active_style = "QFrame { border: 2px dashed #0284C7; border-radius: 8px; background-color: #F0F9FF; }"
        self.setStyleSheet(self.default_style)

        layout = QVBoxLayout()
        self.label = QLabel("Drag & Drop CSV or Video files here\n(or click to browse)")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #455A64; font-size: 14px; font-weight: 500;")
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.all_files = []

    def mousePressEvent(self, event):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Experiment Files",
            "",
            "Experiment Data (*.csv *.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if files:
            for f in files:
                if f not in self.all_files:
                    self.all_files.append(f)
            self.filesChanged.emit(self.all_files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet(self.active_style)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet(self.default_style)

    def dropEvent(self, event):
        self.setStyleSheet(self.default_style)
        urls = event.mimeData().urls()
        new_files = [u.toLocalFile() for u in urls if u.toLocalFile()]
        for f in new_files:
            if f not in self.all_files:
                self.all_files.append(f)
        self.filesChanged.emit(self.all_files)


# =====================================================================
# Background Asynchronous Workers & Signals
# =====================================================================
class WorkerSignals(QObject):
    progress = pyqtSignal(int, int, str)
    session_finished = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)


class TrackingOnlyWorker(QRunnable):
    def __init__(self, matched_pairs: Dict[str, dict], config: PipelineConfig, save_raw_csv: bool = True):
        super().__init__()
        self.matched_pairs = matched_pairs
        self.config = config
        self.save_raw_csv = save_raw_csv
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @pyqtSlot()
    def run(self):
        video_sessions = {k: v for k, v in self.matched_pairs.items() if v.get("video")}
        total_sessions = len(video_sessions)
        processed = 0
        all_results = {}

        for idx, (base, paths) in enumerate(video_sessions.items(), start=1):
            if self._is_cancelled:
                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"Tracking cancelled ({processed}/{total_sessions})."
                )
                break

            vid_path = paths["video"]
            ch_rois = paths.get("chamber_rois")

            if not ch_rois:
                cap = cv2.VideoCapture(vid_path)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    configs = SymmetricGridAligner.generate_symmetric_chambers(
                        frame_shape=frame.shape,
                        rows=getattr(self.config, "grid_rows", 4),
                        cols=getattr(self.config, "grid_cols", 2),
                        order=getattr(self.config, "grid_order", "column_first"),
                    )
                    ch_rois = [c["roi"] for c in configs]
                    paths["chamber_rois"] = ch_rois

            out_dir = os.path.dirname(os.path.abspath(vid_path))
            raw_csv_path = os.path.join(out_dir, f"{base}_raw.csv")

            last_t = [time.time()]
            last_f = [0]
            current_fps = [0.0]

            def on_frame_progress(f_cur: int, f_tot: int):
                if self._is_cancelled:
                    return

                if f_tot > 0 and (f_cur % 15 == 0 or f_cur == f_tot):
                    curr_t = time.time()
                    dt = curr_t - last_t[0]
                    df = f_cur - last_f[0]

                    if dt >= 0.05 and df > 0:
                        current_fps[0] = df / dt
                        last_t[0] = curr_t
                        last_f[0] = f_cur

                    pct = min(100, max(0, int((f_cur / f_tot) * 100)))
                    msg = (
                        f"Tracking [{base}] ({idx}/{total_sessions}): "
                        f"{pct}% ({f_cur}/{f_tot} frames) | {current_fps[0]:.1f} fps"
                    )
                    self.signals.progress.emit(processed, total_sessions, msg)

            try:
                tracker = FlyVisionTracker(chamber_rois=ch_rois)
                raw_df = tracker.track_video(vid_path, progress_callback=on_frame_progress)

                if self.save_raw_csv:
                    raw_df.to_csv(raw_csv_path, index=False)
                    paths["csv"] = raw_csv_path
                else:
                    paths["csv"] = None

                all_results[base] = {
                    "raw_csv": raw_csv_path if self.save_raw_csv else None,
                    "frames": len(raw_df)
                }
                processed += 1
                self.signals.session_finished.emit(base, all_results[base])
                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"Finished video tracking: {base} ({processed}/{total_sessions})"
                )
            except Exception as e:
                self.signals.error.emit(base, str(e))

        self.signals.finished.emit(all_results)


class PipelineBatchWorker(QRunnable):
    def __init__(
        self,
        matched_pairs: Dict[str, dict],
        config: PipelineConfig,
        anesthesia_onset_time: float = 0.0,
        save_raw_csv: bool = True,
        save_cleaned_csv: bool = True,
        plot_act_pos: bool = True,
        plot_kymo: bool = True,
        render_video_overlay: bool = False,
    ):
        super().__init__()
        self.matched_pairs = matched_pairs
        self.config = config
        self.anesthesia_onset_time = anesthesia_onset_time
        self.save_raw_csv = save_raw_csv
        self.save_cleaned_csv = save_cleaned_csv
        self.plot_act_pos = plot_act_pos
        self.plot_kymo = plot_kymo
        self.render_video_overlay = render_video_overlay
        self.signals = WorkerSignals()
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    @pyqtSlot()
    def run(self):
        total_sessions = len(self.matched_pairs)
        processed = 0
        all_results = {}

        cleaner = KinematicCleaner(fps=getattr(self.config, "fps", 30.0))
        analyzer = AnesthesiaAnalyzer(
            fps=getattr(self.config, "fps", 30.0),
            anesthesia_still_sec=getattr(self.config, "anesthesia_still_sec", 120.0),
            anesthesia_speed_thresh=getattr(self.config, "anesthesia_speed_thresh", 0.10),
            sedate_speed_ratio=getattr(self.config, "sedate_speed_ratio", 0.35),
            sedate_drop_speed=getattr(self.config, "sedate_drop_speed", 0.25),
            anesthesia_onset_time=self.anesthesia_onset_time,
        )
        visualizer = ScientificVisualizer(fps=getattr(self.config, "fps", 30.0))

        for idx, (base, paths) in enumerate(self.matched_pairs.items(), start=1):
            if self._is_cancelled:
                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"Task cancelled ({processed}/{total_sessions}).",
                )
                break

            out_dir = os.path.dirname(os.path.abspath(paths.get("csv") or paths.get("video") or "."))
            raw_csv = paths.get("csv")
            vid_path = paths.get("video")

            try:
                if not raw_csv or not os.path.exists(raw_csv):
                    if not vid_path or not os.path.exists(vid_path):
                        raise FileNotFoundError(f"No valid video or CSV found for session '{base}'.")

                    ch_rois = paths.get("chamber_rois")
                    if not ch_rois:
                        cap = cv2.VideoCapture(vid_path)
                        ret, frame = cap.read()
                        cap.release()
                        if ret and frame is not None:
                            configs = SymmetricGridAligner.generate_symmetric_chambers(
                                frame_shape=frame.shape,
                                rows=getattr(self.config, "grid_rows", 4),
                                cols=getattr(self.config, "grid_cols", 2),
                                order=getattr(self.config, "grid_order", "column_first"),
                            )
                            ch_rois = [c["roi"] for c in configs]
                            paths["chamber_rois"] = ch_rois

                    last_t = [time.time()]
                    last_f = [0]
                    current_fps = [0.0]

                    def on_tracking_progress(f_cur: int, f_tot: int):
                        if self._is_cancelled:
                            return
                        if f_tot > 0 and (f_cur % 15 == 0 or f_cur == f_tot):
                            curr_t = time.time()
                            dt = curr_t - last_t[0]
                            df = f_cur - last_f[0]
                            if dt >= 0.05 and df > 0:
                                current_fps[0] = df / dt
                                last_t[0] = curr_t
                                last_f[0] = f_cur
                            pct = min(100, max(0, int((f_cur / f_tot) * 100)))
                            msg = (
                                f"Tracking [{base}] ({idx}/{total_sessions}): "
                                f"{pct}% ({f_cur}/{f_tot} frames) | {current_fps[0]:.1f} fps"
                            )
                            self.signals.progress.emit(processed, total_sessions, msg)

                    tracker = FlyVisionTracker(chamber_rois=ch_rois)
                    raw_df = tracker.track_video(vid_path, progress_callback=on_tracking_progress)

                    if self.save_raw_csv:
                        raw_csv = os.path.join(out_dir, f"{base}_raw.csv")
                        raw_df.to_csv(raw_csv, index=False)
                        paths["csv"] = raw_csv
                else:
                    self.signals.progress.emit(
                        processed,
                        total_sessions,
                        f"[{SessionPhase.READING_CSV.value}] {base}...",
                    )
                    raw_df = pd.read_csv(raw_csv)

                if self._is_cancelled:
                    break

                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"[{SessionPhase.CLEANING.value}] {base}...",
                )
                cleaned_df = cleaner.clean_trajectory(raw_df)
                cleaned_csv_path = os.path.join(out_dir, f"{base}_cleaned.csv")
                if self.save_cleaned_csv:
                    cleaned_df.to_csv(cleaned_csv_path, index=False)

                if self._is_cancelled:
                    break

                plot_files = {}
                if self.plot_act_pos or self.plot_kymo:
                    self.signals.progress.emit(
                        processed,
                        total_sessions,
                        f"[{SessionPhase.PLOTTING_EARLY.value}] {base}...",
                    )
                    if self.plot_act_pos:
                        act_pos_path = os.path.join(out_dir, f"{base}_activity_position.png")
                        visualizer.plot_activity_position_overview(cleaned_df, act_pos_path)
                        plot_files["act_pos"] = act_pos_path

                    if self.plot_kymo:
                        kymo_path = os.path.join(out_dir, f"{base}_kymograph_norm.png")
                        visualizer.plot_kymograph_hexbin(cleaned_df, kymo_path)
                        plot_files["kymo"] = kymo_path

                if self._is_cancelled:
                    break

                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"[{SessionPhase.ANALYZING.value}] {base}...",
                )
                try:
                    df_with_states = analyzer.evaluate_states(
                        cleaned_df,
                        anesthesia_onset_time=self.anesthesia_onset_time,
                    )
                except TypeError:
                    df_with_states = analyzer.evaluate_states(cleaned_df)

                try:
                    summary_df = analyzer.extract_summary(
                        df_with_states,
                        anesthesia_onset_time=self.anesthesia_onset_time,
                    )
                except TypeError:
                    summary_df = analyzer.extract_summary(df_with_states)

                summary_csv_path = os.path.join(out_dir, f"{base}_summary.csv")
                summary_df.to_csv(summary_csv_path, index=False)

                if self._is_cancelled:
                    break

                overlay_path = None
                if self.render_video_overlay and vid_path and os.path.exists(vid_path):
                    self.signals.progress.emit(
                        processed,
                        total_sessions,
                        f"[{SessionPhase.RENDERING.value}] {base}...",
                    )
                    overlay_path = os.path.join(out_dir, f"{base}_overlay.mp4")
                    visualizer.render_overlay_video(df_with_states, vid_path, overlay_path)

                session_res = {
                    "cleaned_csv": cleaned_csv_path if self.save_cleaned_csv else None,
                    "summary_csv": summary_csv_path,
                    "plots": plot_files,
                    "overlay_video": overlay_path,
                    "summary_data": summary_df.to_dict(orient="records") if hasattr(summary_df, "to_dict") else None,
                }
                all_results[base] = session_res
                processed += 1
                self.signals.session_finished.emit(base, session_res)
                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"[{SessionPhase.COMPLETED.value}] {base} ({processed}/{total_sessions})",
                )

            except Exception as e:
                self.signals.error.emit(base, str(e))
                self.signals.progress.emit(
                    processed,
                    total_sessions,
                    f"[{SessionPhase.FAILED.value}] Error on {base}",
                )

        self.signals.finished.emit(all_results)


# =====================================================================
# Interactive Multi-Chamber Calibration Canvas
# =====================================================================
class InteractiveChamberCanvas(QWidget):
    chamberSelected = pyqtSignal(int)
    firstRoiDrawn = pyqtSignal(tuple)
    boxChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)

    MIN_WIDTH = 25
    MIN_HEIGHT = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.sample_frame = None
        self.cached_gray = None
        self.cached_qimage_base = None
        self.boxes: List[List[int]] = []

        self.undo_stack: List[Tuple[List[List[int]], set, int]] = []
        self.max_undo = 50

        self.selected_idx = -1
        self.selected_indices: set = set()

        self.link_mode = "single"
        self.rows = 4
        self.cols = 2
        self.diff_thresh = 14
        self.fly_centroids: Dict[int, Optional[Tuple[float, float]]] = {}

        self.is_drawing_first = False
        self.draw_start_point = None
        self.current_drawing_rect = None

        self.is_box_selecting = False
        self.select_start_img = None
        self.current_select_rect_img = None
        self.press_pos = None

        self.drag_mode = None
        self.drag_start_pos = None
        self.drag_initial_boxes: List[List[int]] = []

        self.setStyleSheet("background-color: #0F172A; border-radius: 8px;")
        self.setMinimumSize(720, 480)

    def _push_undo(self):
        state = ([list(b) for b in self.boxes], set(self.selected_indices), self.selected_idx)
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        boxes_prev, indices_prev, idx_prev = self.undo_stack.pop()
        self.boxes = [list(b) for b in boxes_prev]
        self.selected_indices = set(indices_prev)
        self.selected_idx = idx_prev if (0 <= idx_prev < len(self.boxes)) else (-1 if not self.boxes else 0)
        self._update_fly_detections()
        self.selectionChanged.emit(len(self.selected_indices))
        self.boxChanged.emit()
        self.update()

    def set_data(self, frame: np.ndarray, boxes: List[List[int]], rows: int = 4, cols: int = 2):
        self.sample_frame = frame
        self.cached_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        self.cached_qimage_base = None

        if self.sample_frame is not None:
            rgb_base = np.ascontiguousarray(cv2.cvtColor(self.sample_frame, cv2.COLOR_BGR2RGB))
            h, w, ch = rgb_base.shape
            self.cached_qimage_base = QImage(rgb_base.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        self.boxes = [list(b) for b in boxes]
        self.rows = rows
        self.cols = cols
        self.is_drawing_first = (len(self.boxes) == 0)
        self.undo_stack.clear()

        if self.boxes:
            self.selected_idx = 0
            self.selected_indices = {0}
        else:
            self.selected_idx = -1
            self.selected_indices = set()

        self._update_fly_detections()
        self.selectionChanged.emit(len(self.selected_indices))
        self.update()

    def start_redraw_first_roi(self):
        self._push_undo()
        self.is_drawing_first = True
        self.boxes = []
        self.selected_idx = -1
        self.selected_indices.clear()
        self.selectionChanged.emit(0)
        self.update()

    def _update_fly_detections(self):
        if self.sample_frame is None or not self.boxes:
            return
        gray = self.cached_gray if self.cached_gray is not None else cv2.cvtColor(self.sample_frame, cv2.COLOR_BGR2GRAY)
        tracker = RobustFlyTracker(
            chamber_rois=[tuple(b) for b in self.boxes],
            diff_thresh=self.diff_thresh,
        )
        _, detections = tracker.process_frame(gray, 0, 0.0)
        tracker.close()
        self.fly_centroids = {}
        for idx, b in enumerate(self.boxes):
            cid = idx + 1
            cand = detections.get(cid)
            if cand and "local_pos" in cand:
                abs_x = b[0] + cand["local_pos"][0]
                abs_y = b[1] + cand["local_pos"][1]
                self.fly_centroids[cid] = (abs_x, abs_y)
            else:
                self.fly_centroids[cid] = None

    def get_scale_and_offsets(self):
        if self.sample_frame is None:
            return 1.0, 0, 0
        img_h, img_w = self.sample_frame.shape[:2]
        canvas_w, canvas_h = self.width(), self.height()
        scale = min(canvas_w / img_w, canvas_h / img_h)
        offset_x = (canvas_w - img_w * scale) / 2.0
        offset_y = (canvas_h - img_h * scale) / 2.0
        return scale, offset_x, offset_y

    def img_to_canvas(self, x, y):
        s, ox, oy = self.get_scale_and_offsets()
        return ox + x * s, oy + y * s

    def canvas_to_img(self, cx, cy):
        s, ox, oy = self.get_scale_and_offsets()
        return (cx - ox) / s, (cy - oy) / s

    def _hit_test(self, cx, cy):
        if self.is_drawing_first or not self.boxes:
            return None, -1

        handle_hit_margin = 10.0

        if 0 <= self.selected_idx < len(self.boxes):
            bx1, by1, bx2, by2 = self.boxes[self.selected_idx]
            rx1, ry1 = self.img_to_canvas(bx1, by1)
            rx2, ry2 = self.img_to_canvas(bx2, by2)

            near_l = abs(cx - rx1) <= handle_hit_margin
            near_r = abs(cx - rx2) <= handle_hit_margin
            near_t = abs(cy - ry1) <= handle_hit_margin
            near_b = abs(cy - ry2) <= handle_hit_margin

            in_y_range = (ry1 - handle_hit_margin <= cy <= ry2 + handle_hit_margin)
            in_x_range = (rx1 - handle_hit_margin <= cx <= rx2 + handle_hit_margin)

            if near_l and near_t: return "resize_tl", self.selected_idx
            if near_r and near_t: return "resize_tr", self.selected_idx
            if near_l and near_b: return "resize_bl", self.selected_idx
            if near_r and near_b: return "resize_br", self.selected_idx
            if near_l and in_y_range: return "resize_l", self.selected_idx
            if near_r and in_y_range: return "resize_r", self.selected_idx
            if near_t and in_x_range: return "resize_t", self.selected_idx
            if near_b and in_x_range: return "resize_b", self.selected_idx

        for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
            kx1, ky1 = self.img_to_canvas(x1, y1)
            kx2, ky2 = self.img_to_canvas(x2, y2)
            if kx1 <= cx <= kx2 and ky1 <= cy <= ky2:
                return "move", idx

        return None, -1

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            cx, cy = event.position().x(), event.position().y()
            self.press_pos = (cx, cy)
            modifiers = event.modifiers()
            has_shift = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))

            if self.is_drawing_first:
                ix, iy = self.canvas_to_img(cx, cy)
                self.draw_start_point = (ix, iy)
                self.current_drawing_rect = (ix, iy, ix, iy)
                self.update()
                return

            action, idx = self._hit_test(cx, cy)

            if action is None:
                if not has_shift:
                    self.selected_indices.clear()
                    self.selected_idx = -1
                    self.selectionChanged.emit(0)
                ix, iy = self.canvas_to_img(cx, cy)
                self.is_box_selecting = True
                self.select_start_img = (ix, iy)
                self.current_select_rect_img = (ix, iy, ix, iy)
                self.update()
                return

            if action == "move":
                if has_shift:
                    if idx in self.selected_indices:
                        self.selected_indices.remove(idx)
                        self.selected_idx = next(iter(self.selected_indices)) if self.selected_indices else -1
                    else:
                        self.selected_indices.add(idx)
                        self.selected_idx = idx
                else:
                    if idx not in self.selected_indices:
                        self.selected_indices = {idx}
                        self.selected_idx = idx
                    else:
                        self.selected_idx = idx
            else:
                self.selected_idx = idx
                self.selected_indices.add(idx)

            if self.selected_idx >= 0:
                self.chamberSelected.emit(self.selected_idx + 1)
            self.selectionChanged.emit(len(self.selected_indices))

            self.drag_mode = action
            self.drag_start_pos = (cx, cy)
            self.drag_initial_boxes = [list(b) for b in self.boxes]
            self.update()

    def mouseMoveEvent(self, event):
        cx, cy = event.position().x(), event.position().y()

        if self.is_drawing_first and self.draw_start_point:
            ix, iy = self.canvas_to_img(cx, cy)
            sx, sy = self.draw_start_point
            self.current_drawing_rect = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.update()
            return

        if self.is_box_selecting and self.select_start_img:
            ix, iy = self.canvas_to_img(cx, cy)
            sx, sy = self.select_start_img
            self.current_select_rect_img = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update()
            return

        if self.drag_mode and self.drag_start_pos:
            s, _, _ = self.get_scale_and_offsets()
            raw_dx = (cx - self.drag_start_pos[0]) / s
            raw_dy = (cy - self.drag_start_pos[1]) / s
            img_h, img_w = self.sample_frame.shape[:2] if self.sample_frame is not None else (1000, 1000)

            active_set = set(self.selected_indices)
            if self.selected_idx >= 0:
                active_set.add(self.selected_idx)
            if not active_set:
                active_set = {0}

            sel_row = self.selected_idx % self.rows if self.selected_idx >= 0 else 0
            sel_col = self.selected_idx // self.rows if self.selected_idx >= 0 else 0

            if len(self.selected_indices) <= 1:
                if self.link_mode == "col":
                    active_set = {i for i in range(len(self.boxes)) if (i // self.rows) == sel_col}
                elif self.link_mode == "row":
                    active_set = {i for i in range(len(self.boxes)) if (i % self.rows) == sel_row}
                elif self.link_mode == "all":
                    active_set = set(range(len(self.boxes)))

            tentative_boxes = [list(b) for b in self.boxes]

            if self.drag_mode == "move":
                min_allow_dx = max([-self.drag_initial_boxes[idx][0] for idx in active_set])
                max_allow_dx = min([img_w - self.drag_initial_boxes[idx][2] for idx in active_set])
                min_allow_dy = max([-self.drag_initial_boxes[idx][1] for idx in active_set])
                max_allow_dy = min([img_h - self.drag_initial_boxes[idx][3] for idx in active_set])

                if min_allow_dx > max_allow_dx:
                    min_allow_dx, max_allow_dx = max_allow_dx, min_allow_dx
                if min_allow_dy > max_allow_dy:
                    min_allow_dy, max_allow_dy = max_allow_dy, min_allow_dy

                rigid_dx = np.clip(raw_dx, min_allow_dx, max_allow_dx)
                rigid_dy = np.clip(raw_dy, min_allow_dy, max_allow_dy)

                for idx in active_set:
                    ox1, oy1, ox2, oy2 = self.drag_initial_boxes[idx]
                    tentative_boxes[idx] = [
                        int(round(ox1 + rigid_dx)),
                        int(round(oy1 + rigid_dy)),
                        int(round(ox2 + rigid_dx)),
                        int(round(oy2 + rigid_dy))
                    ]
            else:
                target_resize_set = active_set if (self.link_mode != "single" or len(self.selected_indices) > 1) else {self.selected_idx}
                for idx in target_resize_set:
                    ox1, oy1, ox2, oy2 = self.drag_initial_boxes[idx]
                    nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2

                    if "resize_l" in self.drag_mode or self.drag_mode in ("resize_tl", "resize_bl"):
                        nx1 = min(ox2 - self.MIN_WIDTH, ox1 + raw_dx)
                    if "resize_r" in self.drag_mode or self.drag_mode in ("resize_tr", "resize_br"):
                        nx2 = max(ox1 + self.MIN_WIDTH, ox2 + raw_dx)
                    if "resize_t" in self.drag_mode or self.drag_mode in ("resize_tl", "resize_tr"):
                        ny1 = min(oy2 - self.MIN_HEIGHT, oy1 + raw_dy)
                    if "resize_b" in self.drag_mode or self.drag_mode in ("resize_bl", "resize_br"):
                        ny2 = max(oy1 + self.MIN_HEIGHT, oy2 + raw_dy)

                    nx1 = int(np.clip(nx1, 0, img_w - self.MIN_WIDTH))
                    nx2 = int(np.clip(nx2, self.MIN_WIDTH, img_w))
                    if nx2 - nx1 < self.MIN_WIDTH:
                        if "resize_l" in self.drag_mode:
                            nx1 = nx2 - self.MIN_WIDTH
                        else:
                            nx2 = nx1 + self.MIN_WIDTH

                    ny1 = int(np.clip(ny1, 0, img_h - self.MIN_HEIGHT))
                    ny2 = int(np.clip(ny2, self.MIN_HEIGHT, img_h))
                    if ny2 - ny1 < self.MIN_HEIGHT:
                        if "resize_t" in self.drag_mode:
                            ny1 = ny2 - self.MIN_HEIGHT
                        else:
                            ny2 = ny1 + self.MIN_HEIGHT

                    tentative_boxes[idx] = [nx1, ny1, nx2, ny2]

            self.boxes = tentative_boxes
            self.boxChanged.emit()
            self.update()
        else:
            action, _ = self._hit_test(cx, cy)
            cursor_map = {
                "resize_l": Qt.CursorShape.SizeHorCursor,
                "resize_r": Qt.CursorShape.SizeHorCursor,
                "resize_t": Qt.CursorShape.SizeVerCursor,
                "resize_b": Qt.CursorShape.SizeVerCursor,
                "resize_tl": Qt.CursorShape.SizeFDiagCursor,
                "resize_br": Qt.CursorShape.SizeFDiagCursor,
                "resize_tr": Qt.CursorShape.SizeBDiagCursor,
                "resize_bl": Qt.CursorShape.SizeBDiagCursor,
                "move": Qt.CursorShape.SizeAllCursor,
            }
            self.setCursor(QCursor(cursor_map.get(action, Qt.CursorShape.ArrowCursor)))

    def mouseReleaseEvent(self, event):
        if self.is_drawing_first and self.current_drawing_rect:
            x1, y1, x2, y2 = self.current_drawing_rect
            if (x2 - x1) >= self.MIN_WIDTH and (y2 - y1) >= self.MIN_HEIGHT:
                self.is_drawing_first = False
                self.draw_start_point = None
                self.current_drawing_rect = None
                self._push_undo()
                self.firstRoiDrawn.emit((int(x1), int(y1), int(x2), int(y2)))
            else:
                self.draw_start_point = None
                self.current_drawing_rect = None
                self.update()
            return

        if self.is_box_selecting:
            if self.current_select_rect_img:
                rx1, ry1, rx2, ry2 = self.current_select_rect_img
                if (rx2 - rx1) > 6 and (ry2 - ry1) > 6:
                    hit_set = set()
                    for idx, (bx1, by1, bx2, by2) in enumerate(self.boxes):
                        if not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                            hit_set.add(idx)

                    modifiers = event.modifiers()
                    has_shift = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
                    if has_shift:
                        self.selected_indices.update(hit_set)
                    else:
                        self.selected_indices = hit_set

                    if self.selected_indices:
                        self.selected_idx = min(self.selected_indices)
                        self.chamberSelected.emit(self.selected_idx + 1)
                    else:
                        self.selected_idx = -1
                    self.selectionChanged.emit(len(self.selected_indices))

            self.is_box_selecting = False
            self.select_start_img = None
            self.current_select_rect_img = None
            self.update()
            return

        if self.drag_mode:
            if self.drag_initial_boxes != self.boxes:
                state = ([list(b) for b in self.drag_initial_boxes], set(self.selected_indices), self.selected_idx)
                self.undo_stack.append(state)
                if len(self.undo_stack) > self.max_undo:
                    self.undo_stack.pop(0)

            self.drag_mode = None
            self.drag_start_pos = None
            self._update_fly_detections()
            self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Z and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.undo()
            return

        if not self.boxes:
            super().keyPressEvent(event)
            return

        step = 5 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1
        dx, dy = 0, 0

        key = event.key()
        if key == Qt.Key.Key_Left: dx = -step
        elif key == Qt.Key.Key_Right: dx = step
        elif key == Qt.Key.Key_Up: dy = -step
        elif key == Qt.Key.Key_Down: dy = step
        else:
            super().keyPressEvent(event)
            return

        self._push_undo()
        active_set = self.selected_indices if self.selected_indices else ({self.selected_idx} if self.selected_idx >= 0 else set(range(len(self.boxes))))
        img_h, img_w = self.sample_frame.shape[:2] if self.sample_frame is not None else (1000, 1000)

        for idx in active_set:
            x1, y1, x2, y2 = self.boxes[idx]
            w, h = x2 - x1, y2 - y1
            nx1 = int(np.clip(x1 + dx, 0, img_w - w))
            ny1 = int(np.clip(y1 + dy, 0, img_h - h))
            self.boxes[idx] = [nx1, ny1, nx1 + w, ny1 + h]

        self.boxChanged.emit()
        self._update_fly_detections()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.sample_frame is None or self.cached_qimage_base is None:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video frame available")
            return

        s, ox, oy = self.get_scale_and_offsets()
        img_h, img_w = self.sample_frame.shape[:2]

        painter.drawImage(QRectF(ox, oy, img_w * s, img_h * s), self.cached_qimage_base)

        mid_x = ox + (img_w * 0.5) * s
        painter.setPen(QPen(QColor("#94A3B8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(mid_x), int(oy), int(mid_x), int(oy + img_h * s))

        if self.is_drawing_first and self.current_drawing_rect:
            x1, y1, x2, y2 = self.current_drawing_rect
            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(245, 158, 11, 40)))
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))
            painter.setPen(QColor("#FDE68A"))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(int(rx1 + 6), int(ry1 + 18), "CH 1 (Release to infer grid)")
            return

        font_label = QFont("Arial", 9, QFont.Weight.Bold)
        font_dim = QFont("Arial", 8)

        for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
            cid = idx + 1
            is_active = (idx == self.selected_idx)
            is_in_group = (idx in self.selected_indices)

            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            rw, rh = rx2 - rx1, ry2 - ry1

            if is_active:
                border_color = QColor("#F59E0B")
                fill_color = QColor(245, 158, 11, 40)
                line_w = 2.5
            elif is_in_group:
                border_color = QColor("#38BDF8")
                fill_color = QColor(56, 189, 248, 30)
                line_w = 2.0
            else:
                border_color = QColor("#10B981")
                fill_color = QColor(16, 185, 129, 15)
                line_w = 1.2

            painter.setPen(QPen(border_color, line_w))
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(QRectF(rx1, ry1, rw, rh), 4, 4)

            painter.setPen(QPen(QColor("#F97316"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(rx1 + rw * 0.08), int(ry1), int(rx1 + rw * 0.08), int(ry2))
            painter.setPen(QPen(QColor("#38BDF8"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(rx2 - rw * 0.08), int(ry1), int(rx2 - rw * 0.08), int(ry2))

            tag_rect = QRectF(rx1 + 4, ry1 + 4, 48, 18)
            painter.setPen(Qt.PenStyle.NoPen)
            badge_bg = QColor("#D97706") if is_active else (QColor("#0284C7") if is_in_group else QColor("#0F172A"))
            painter.setBrush(QBrush(badge_bg))
            painter.drawRoundedRect(tag_rect, 3, 3)

            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(font_label)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, f"CH {cid}")

            if is_active:
                painter.setFont(font_dim)
                painter.setPen(QColor("#FDE68A"))
                painter.drawText(int(rx1 + 56), int(ry1 + 17), f"{int(x2 - x1)}x{int(y2 - y1)} px")

            if cid in self.fly_centroids and self.fly_centroids[cid] is not None:
                fx, fy = self.fly_centroids[cid]
                cfx, cfy = self.img_to_canvas(fx, fy)
                painter.setPen(QPen(QColor("#22C55E"), 1))
                painter.drawLine(int(cfx - 7), int(cfy), int(cfx + 7), int(cfy))
                painter.drawLine(int(cfx), int(cfy - 7), int(cfx), int(cfy + 7))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#EF4444")))
                painter.drawEllipse(QPointF(cfx, cfy), 3.5, 3.5)

            if is_active:
                h_size = 7
                painter.setBrush(QBrush(QColor("#FFFFFF")))
                painter.setPen(QPen(QColor("#F59E0B"), 1.5))
                handle_points = [
                    (rx1, ry1), ((rx1 + rx2) / 2, ry1), (rx2, ry1),
                    (rx1, (ry1 + ry2) / 2), (rx2, (ry1 + ry2) / 2),
                    (rx1, ry2), ((rx1 + rx2) / 2, ry2), (rx2, ry2)
                ]
                for px, py in handle_points:
                    painter.drawRect(QRectF(px - h_size / 2, py - h_size / 2, h_size, h_size))

        if self.is_box_selecting and self.current_select_rect_img:
            x1, y1, x2, y2 = self.current_select_rect_img
            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            painter.setPen(QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(56, 189, 248, 45)))
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))


# =====================================================================
# Chamber Calibration Dialog
# =====================================================================
class ChamberCalibrationDialog(QDialog):
    def __init__(self, video_path: str, parent=None, initial_chambers=None, rows: int = 4, cols: int = 2, order: str = "column_first"):
        super().__init__(parent)
        self.setWindowTitle(f"Multi-Chamber Grid Calibrator - {os.path.basename(video_path)}")
        self.resize(1140, 740)
        self.video_path = video_path
        self.sample_frame = None
        self.order = order
        self.last_first_box = None

        if initial_chambers and len(initial_chambers) > 0:
            self.boxes = [list(b) for b in initial_chambers]
            self.last_first_box = tuple(self.boxes[0])
            if len(self.boxes) != (rows * cols):
                self.rows = rows
                self.cols = max(1, len(self.boxes) // rows)
            else:
                self.rows = rows
                self.cols = cols
        else:
            self.boxes = []
            self.rows = max(1, rows)
            self.cols = max(1, cols)

        self._load_video_sample()
        self._setup_ui()

        self.undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.activated.connect(self.canvas.undo)

        if self.boxes:
            self.canvas.set_data(self.sample_frame, self.boxes, self.rows, self.cols)
            self._rebuild_chamber_buttons()
            self.tip_label.setText(f"<b>Loaded {len(self.boxes)} calibrated chambers</b>.")
        else:
            self.canvas.set_data(self.sample_frame, [], self.rows, self.cols)
            self.canvas.start_redraw_first_roi()

    def _load_video_sample(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sample_idx = min(30, max(0, total_frames // 10))
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.sample_frame = frame
                cap.release()
        except Exception:
            pass
        if self.sample_frame is None:
            self.sample_frame = np.full((500, 880, 3), 40, dtype=np.uint8)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(14)

        left_layout = QVBoxLayout()
        self.canvas = InteractiveChamberCanvas()
        self.canvas.firstRoiDrawn.connect(self._on_first_roi_drawn)
        self.canvas.boxChanged.connect(self._on_canvas_box_changed)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.canvas, 1)

        self.tip_label = QLabel(f"<b>Draw the first tube (CH 1) in top-left</b>: System infers {self.rows}x{self.cols} grid. Press Ctrl+Z to undo.")
        self.tip_label.setStyleSheet("color: #E2E8F0; font-size: 13px; background-color: #1E293B; padding: 8px; border-radius: 6px;")
        left_layout.addWidget(self.tip_label)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        grp_grid = QGroupBox("1. Grid Geometry")
        v_grid = QVBoxLayout()

        h_dim = QHBoxLayout()
        h_dim.addWidget(QLabel("Rows:"))
        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(1, 32)
        self.spin_rows.setValue(self.rows)
        self.spin_rows.valueChanged.connect(self._on_grid_dims_changed)
        h_dim.addWidget(self.spin_rows)

        h_dim.addWidget(QLabel("Cols:"))
        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(1, 16)
        self.spin_cols.setValue(self.cols)
        self.spin_cols.valueChanged.connect(self._on_grid_dims_changed)
        h_dim.addWidget(self.spin_cols)
        v_grid.addLayout(h_dim)

        h_ord = QHBoxLayout()
        h_ord.addWidget(QLabel("Chamber Order:"))
        self.combo_ord = QComboBox()
        self.combo_ord.addItems(["Column-first (1..N)", "Row-first (1..N)"])
        self.combo_ord.setCurrentIndex(0 if self.order == "column_first" else 1)
        self.combo_ord.currentIndexChanged.connect(self._on_grid_dims_changed)
        h_ord.addWidget(self.combo_ord)
        v_grid.addLayout(h_ord)

        btn_redraw = QPushButton("Redraw First Tube ROI (CH 1)")
        btn_redraw.setStyleSheet("background-color: #0284C7; color: white; font-weight: bold; padding: 7px; border-radius: 4px;")
        btn_redraw.clicked.connect(self._on_click_redraw)
        v_grid.addWidget(btn_redraw)

        grp_grid.setLayout(v_grid)
        right_layout.addWidget(grp_grid)

        grp_mode = QGroupBox("2. Drag & Resize Link Mode")
        v_mode = QVBoxLayout()
        self.rb_single = QRadioButton("Single Active Chamber (Default)")
        self.rb_all = QRadioButton("Move All Selected Chambers")
        self.rb_col = QRadioButton("Active Column Linked (Column)")
        self.rb_row = QRadioButton("Active Row Linked (Row)")
        self.rb_single.setChecked(True)

        self.rb_single.toggled.connect(lambda: self._set_mode("single"))
        self.rb_all.toggled.connect(lambda: self._set_mode("all"))
        self.rb_col.toggled.connect(lambda: self._set_mode("col"))
        self.rb_row.toggled.connect(lambda: self._set_mode("row"))

        v_mode.addWidget(self.rb_single)
        v_mode.addWidget(self.rb_all)
        v_mode.addWidget(self.rb_col)
        v_mode.addWidget(self.rb_row)
        grp_mode.setLayout(v_mode)
        right_layout.addWidget(grp_mode)

        grp_tools = QGroupBox("3. Visual Tools & History")
        v_tools = QVBoxLayout()

        btn_undo = QPushButton("Undo Last Action (Ctrl+Z)")
        btn_undo.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        btn_undo.clicked.connect(self.canvas.undo)
        v_tools.addWidget(btn_undo)

        btn_snap = QPushButton("Auto-Snap Tube Boundaries")
        btn_snap.setStyleSheet("background-color: #334155; color: white; padding: 6px; border-radius: 5px;")
        btn_snap.clicked.connect(self._on_auto_snap)
        v_tools.addWidget(btn_snap)

        grp_tools.setLayout(v_tools)
        right_layout.addWidget(grp_tools)

        grp_sel = QGroupBox("4. Active Chamber (1..N)")
        v_sel = QVBoxLayout()
        self.scroll_ch = QScrollArea()
        self.scroll_ch.setFixedHeight(70)
        self.scroll_ch.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.grid_ch = QHBoxLayout(self.scroll_widget)
        self.grid_ch.setContentsMargins(4, 4, 4, 4)
        self.grid_ch.setSpacing(4)
        self.scroll_ch.setWidget(self.scroll_widget)
        v_sel.addWidget(self.scroll_ch)
        grp_sel.setLayout(v_sel)
        right_layout.addWidget(grp_sel)

        right_layout.addStretch()

        h_btn = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("Save & Apply Calibration")
        btn_ok.setStyleSheet("background-color: #16A34A; color: white; font-weight: bold; padding: 10px 18px; border-radius: 6px;")
        btn_ok.clicked.connect(self.accept)
        h_btn.addWidget(btn_cancel)
        h_btn.addWidget(btn_ok)
        right_layout.addLayout(h_btn)

        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)

    def _on_selection_changed(self, count: int):
        multi = (count > 1)
        self.rb_col.setEnabled(not multi)
        self.rb_row.setEnabled(not multi)
        if multi and (self.rb_col.isChecked() or self.rb_row.isChecked()):
            self.rb_all.setChecked(True)
            self.canvas.link_mode = "all"

    def _on_auto_snap(self):
        if self.sample_frame is None or not self.canvas.boxes:
            return
        base_boxes = [list(b) for b in self.canvas.boxes]
        refined = RobustGridAligner.snap_all_boxes(
            frame_bgr=self.sample_frame,
            boxes=base_boxes,
            rows=self.rows,
            cols=self.cols,
        )
        self.canvas._push_undo()
        self.canvas.boxes = refined
        self.canvas._update_fly_detections()
        self.canvas.boxChanged.emit()
        self.canvas.update()

    def _on_first_roi_drawn(self, first_box: Tuple[int, int, int, int]):
        self.last_first_box = first_box
        self._recompute_inference()

    def _on_grid_dims_changed(self):
        self.rows = self.spin_rows.value()
        self.cols = self.spin_cols.value()
        self.order = "column_first" if self.combo_ord.currentIndex() == 0 else "row_first"
        if self.last_first_box is not None:
            self._recompute_inference()

    def _recompute_inference(self):
        if self.sample_frame is not None and self.last_first_box is not None:
            estimated_boxes = RobustGridAligner.estimate_chambers_from_first_roi(
                self.sample_frame,
                self.last_first_box,
                rows=self.rows,
                cols=self.cols,
                order=self.order,
            )
            self.boxes = [list(b) for b in estimated_boxes]
            self.canvas.set_data(self.sample_frame, self.boxes, self.rows, self.cols)
            self._rebuild_chamber_buttons()
            self.tip_label.setText(f"<b>Generated {len(self.boxes)} Chambers ({self.rows}x{self.cols})</b>. Drag to fine-tune.")

    def _on_click_redraw(self):
        self.canvas.start_redraw_first_roi()
        self.tip_label.setText("<b>Draw the first tube (CH 1) in top-left</b>...")

    def _rebuild_chamber_buttons(self):
        while self.grid_ch.count():
            item = self.grid_ch.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.ch_buttons = []
        for i in range(len(self.canvas.boxes)):
            cid = i + 1
            b = QPushButton(f"{cid}")
            b.setFixedWidth(34)
            b.setStyleSheet("font-weight: bold; background-color: #1E293B; color: #F8FAFC;")
            b.clicked.connect(lambda checked, idx=i: self._select_chamber(idx))
            self.ch_buttons.append(b)
            self.grid_ch.addWidget(b)

    def _select_chamber(self, idx: int):
        self.canvas.selected_idx = max(0, min(len(self.canvas.boxes) - 1, idx))
        self.canvas.selected_indices = {self.canvas.selected_idx}
        self.canvas.selectionChanged.emit(1)
        self.canvas.update()

    def _on_canvas_box_changed(self):
        self.boxes = self.canvas.boxes
        if len(self.boxes) > 0:
            self.last_first_box = tuple(self.boxes[0])

    def _set_mode(self, mode: str):
        self.canvas.link_mode = mode

    def get_chambers(self) -> List[Tuple[int, int, int, int]]:
        return [tuple(b) for b in self.canvas.boxes]


# =====================================================================
# Main Application Window
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drosophila Anesthesia Tracker & Multi-Chamber Workstation")
        self.resize(1180, 840)
        self.matched_pairs = {}
        self.config = PipelineConfig()
        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None
        self.setup_ui()
        self.sync_config_to_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        left_layout.addWidget(QLabel("<b>1. Batch File Import</b>"))

        self.drop_area = DragDropArea()
        self.drop_area.filesChanged.connect(self.on_files_updated)
        left_layout.addWidget(self.drop_area)

        left_layout.addWidget(QLabel("<b>2. Experiment Sessions & Pairing Status</b>"))

        self.pair_list = QListWidget()
        self.pair_list.itemDoubleClicked.connect(self.calibrate_selected_session)
        left_layout.addWidget(self.pair_list)

        btn_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate Chamber ROI")
        self.btn_calibrate.clicked.connect(self.calibrate_selected_session)
        self.btn_calibrate.setStyleSheet("background-color: #0284C7; color: white; padding: 7px; font-weight: bold; border-radius: 4px;")
        btn_row.addWidget(self.btn_calibrate)

        btn_clear = QPushButton("Clear List")
        btn_clear.clicked.connect(self.clear_all)
        btn_clear.setStyleSheet("background-color: #ECEFF1; padding: 7px; font-weight: 500; border-radius: 4px;")
        btn_row.addWidget(btn_clear)
        left_layout.addLayout(btn_row)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        right_layout.addWidget(QLabel("<b>3. Modules & Analysis Settings</b>"))

        grp_prep = QGroupBox("Module 1 & 2: Tracking & Kinematic Cleaning")
        vbox_prep = QVBoxLayout()
        self.cb_save_raw = QCheckBox("Export Raw Location Data (*_raw.csv)")
        self.cb_save_raw.setChecked(True)
        self.cb_save_clean = QCheckBox("Export Cleaned Location Data (*_cleaned.csv)")
        self.cb_save_clean.setChecked(True)
        vbox_prep.addWidget(self.cb_save_raw)
        vbox_prep.addWidget(self.cb_save_clean)
        grp_prep.setLayout(vbox_prep)
        right_layout.addWidget(grp_prep)

        grp_anesthesia = QGroupBox("Module 4: Kinetics & Gas Delivery Onset")
        vbox_anes = QVBoxLayout()
        vbox_anes.setSpacing(8)

        h_gas = QHBoxLayout()
        lbl_gas = QLabel("<b>Anesthesia Gas Onset Time:</b>")
        lbl_gas.setStyleSheet("color: #D97706; font-size: 13px;")
        h_gas.addWidget(lbl_gas)
        self.spin_gas_onset = QDoubleSpinBox()
        self.spin_gas_onset.setRange(0.0, 7200.0)
        self.spin_gas_onset.setSingleStep(5.0)
        self.spin_gas_onset.setValue(0.0)
        self.spin_gas_onset.setSuffix(" s")
        self.spin_gas_onset.setToolTip("Time when anesthetic gas was introduced. Prior interval serves as active baseline.")
        h_gas.addWidget(self.spin_gas_onset)
        vbox_anes.addLayout(h_gas)

        h_spd = QHBoxLayout()
        h_spd.addWidget(QLabel("Sedate Speed Ratio:"))
        self.spin_speed_ratio = QDoubleSpinBox()
        self.spin_speed_ratio.setRange(0.05, 0.90)
        self.spin_speed_ratio.setSingleStep(0.05)
        self.spin_speed_ratio.setValue(0.35)
        h_spd.addWidget(self.spin_speed_ratio)
        vbox_anes.addLayout(h_spd)

        h_drop = QHBoxLayout()
        h_drop.addWidget(QLabel("Sedate Drop Height (/1s):"))
        self.spin_drop_thresh = QDoubleSpinBox()
        self.spin_drop_thresh.setRange(0.05, 0.80)
        self.spin_drop_thresh.setSingleStep(0.05)
        self.spin_drop_thresh.setValue(0.25)
        h_drop.addWidget(self.spin_drop_thresh)
        vbox_anes.addLayout(h_drop)

        h_still = QHBoxLayout()
        h_still.addWidget(QLabel("Anaesthesia Still Window:"))
        self.spin_still_sec = QSpinBox()
        self.spin_still_sec.setRange(10, 600)
        self.spin_still_sec.setValue(120)
        self.spin_still_sec.setSuffix(" s")
        h_still.addWidget(self.spin_still_sec)
        vbox_anes.addLayout(h_still)

        h_thresh = QHBoxLayout()
        h_thresh.addWidget(QLabel("Anaesthesia Speed Thresh:"))
        self.spin_speed_thresh = QDoubleSpinBox()
        self.spin_speed_thresh.setRange(0.01, 2.0)
        self.spin_speed_thresh.setValue(0.10)
        self.spin_speed_thresh.setSuffix(" px/s")
        h_thresh.addWidget(self.spin_speed_thresh)
        vbox_anes.addLayout(h_thresh)

        grp_anesthesia.setLayout(vbox_anes)
        right_layout.addWidget(grp_anesthesia)

        grp_vis = QGroupBox("Module 5: Graphics & Annotated Video Synthesis")
        vbox_vis = QVBoxLayout()
        self.cb_plot_act_pos = QCheckBox("Dual Y-Axis Behavioral Overview (*_activity_position.png)")
        self.cb_plot_act_pos.setChecked(True)
        self.cb_plot_kymo = QCheckBox("Space-Time Kymograph Heatmap (*_kymograph_norm.png)")
        self.cb_plot_kymo.setChecked(True)
        self.cb_video_overlay = QCheckBox("Render Annotated Video Overlay (*_overlay.mp4)")
        self.cb_video_overlay.setChecked(False)
        vbox_vis.addWidget(self.cb_plot_act_pos)
        vbox_vis.addWidget(self.cb_plot_kymo)
        vbox_vis.addWidget(self.cb_video_overlay)
        grp_vis.setLayout(vbox_vis)
        right_layout.addWidget(grp_vis)

        right_layout.addStretch()

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #546E7A; font-size: 13px;")
        right_layout.addWidget(self.lbl_status)

        h_exec_row = QHBoxLayout()
        self.btn_track_only = QPushButton("Track Fly Only")
        self.btn_track_only.setFixedHeight(46)
        self.btn_track_only.setStyleSheet("background-color: #0284C7; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_track_only.clicked.connect(self.execute_tracking_only)
        h_exec_row.addWidget(self.btn_track_only, 2)

        self.btn_run = QPushButton("Run Selected Modules")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_run.clicked.connect(self.execute_tasks)
        h_exec_row.addWidget(self.btn_run, 3)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(46)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; border-radius: 6px;")
        self.btn_cancel.clicked.connect(self.cancel_execution)
        h_exec_row.addWidget(self.btn_cancel, 1)

        right_layout.addLayout(h_exec_row)

        main_layout.addLayout(left_layout, 5)
        main_layout.addLayout(right_layout, 4)

    def sync_config_to_ui(self):
        self.spin_gas_onset.setValue(float(getattr(self.config, "anesthesia_onset_time", 0.0)))
        self.spin_speed_ratio.setValue(getattr(self.config, "sedate_speed_ratio", 0.35))
        self.spin_drop_thresh.setValue(getattr(self.config, "sedate_drop_speed", 0.25))
        self.spin_still_sec.setValue(int(getattr(self.config, "anesthesia_still_sec", 120.0)))
        self.spin_speed_thresh.setValue(float(getattr(self.config, "anesthesia_speed_thresh", 0.10)))

    def sync_ui_to_config(self):
        self.config.anesthesia_onset_time = float(self.spin_gas_onset.value())
        self.config.sedate_speed_ratio = float(self.spin_speed_ratio.value())
        self.config.sedate_drop_speed = float(self.spin_drop_thresh.value())
        self.config.anesthesia_still_sec = float(self.spin_still_sec.value())
        self.config.anesthesia_speed_thresh = float(self.spin_speed_thresh.value())

    def on_files_updated(self, files):
        self.pair_list.clear()
        self.matched_pairs = {}
        csvs = [f for f in files if f.lower().endswith(".csv")]
        vids = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        def extract_base_name(file_path: str) -> str:
            name = os.path.splitext(os.path.basename(file_path))[0]
            pattern = r"(_tracked|_raw|_cleaned|_v\d+|_overlay|_filtered)$"
            while True:
                new_name = re.sub(pattern, "", name, flags=re.IGNORECASE)
                if new_name == name:
                    break
                name = new_name
            return name.strip()

        csv_map = {extract_base_name(f): f for f in csvs}
        vid_map = {extract_base_name(f): f for f in vids}

        all_bases = sorted(set(csv_map.keys()) | set(vid_map.keys()))
        for base in all_bases:
            has_csv = base in csv_map
            has_vid = base in vid_map
            if has_csv and has_vid:
                status = "[Paired] CSV + Video"
                self.matched_pairs[base] = {"csv": csv_map[base], "video": vid_map[base]}
            elif has_csv and not has_vid:
                status = "[CSV Data Only]"
                self.matched_pairs[base] = {"csv": csv_map[base], "video": None}
            elif not has_csv and has_vid:
                status = "[Video Awaiting Tracking]"
                self.matched_pairs[base] = {"csv": None, "video": vid_map[base]}
            else:
                continue
            self.pair_list.addItem(f"{base}  ->  {status}")

    def calibrate_selected_session(self):
        row = self.pair_list.currentRow()
        if row < 0 and self.matched_pairs:
            row = 0
        if row < 0 or not self.matched_pairs:
            QMessageBox.information(self, "Select Session", "Please select a video session to calibrate.")
            return
        base = list(self.matched_pairs.keys())[row]
        session = self.matched_pairs[base]
        vid_path = session.get("video")
        if not vid_path:
            QMessageBox.information(self, "No Video", f"Session '{base}' does not contain a video file.")
            return

        dlg = ChamberCalibrationDialog(
            vid_path,
            parent=self,
            initial_chambers=session.get("chamber_rois"),
            rows=session.get("grid_rows", getattr(self.config, "grid_rows", 4)),
            cols=session.get("grid_cols", getattr(self.config, "grid_cols", 2)),
            order=session.get("grid_order", getattr(self.config, "grid_order", "column_first")),
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            session["chamber_rois"] = dlg.get_chambers()
            session["grid_rows"] = dlg.rows
            session["grid_cols"] = dlg.cols
            session["grid_order"] = dlg.order
            QMessageBox.information(
                self,
                "Saved",
                f"Calibrated {len(session['chamber_rois'])} chambers ({dlg.rows}x{dlg.cols}) for {base}",
            )

    def cancel_execution(self):
        if self.current_worker:
            self.lbl_status.setText("Cancelling task execution...")
            self.current_worker.cancel()
            self.btn_cancel.setEnabled(False)

    def update_status_progress(self, processed: int, total: int, status_text: str):
        if total > 1:
            display_text = f"[{processed}/{total} Sessions] {status_text}"
        else:
            display_text = status_text
        self.lbl_status.setText(display_text)

    def clear_all(self):
        self.drop_area.all_files = []
        self.drop_area.label.setText("Drag & Drop CSV or Video files here\n(or click to browse)")
        self.pair_list.clear()
        self.matched_pairs = {}
        self.lbl_status.setText("Ready, awaiting task execution.")

    def on_session_finished(self, base: str, res: dict):
        for idx in range(self.pair_list.count()):
            item = self.pair_list.item(idx)
            if item.text().startswith(base):
                item.setText(f"{base}  ->  [Completed]")
                break

    def on_worker_finished(self, results: dict):
        self.btn_run.setEnabled(True)
        self.btn_track_only.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.current_worker = None
        total = len(self.matched_pairs)
        self.lbl_status.setText(f"Batch processing complete: {len(results)}/{total} sessions processed.")
        QMessageBox.information(self, "Execution Complete", f"All tasks finished!\nProcessed {len(results)}/{total} experiments.")

    def on_tracking_session_finished(self, base: str, res: dict):
        for idx in range(self.pair_list.count()):
            item = self.pair_list.item(idx)
            if item.text().startswith(base):
                item.setText(f"{base}  ->  [Video Tracked: Raw CSV Generated]")
                break

    def on_tracking_finished(self, results: dict):
        self.btn_track_only.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.current_worker = None
        count = len(results)
        self.lbl_status.setText(f"Vision tracking complete: {count} raw trajectory CSV files generated.")
        QMessageBox.information(
            self,
            "Tracking Complete",
            f"Vision tracking complete!\nSuccessfully extracted coordinates for {count} videos into *_raw.csv.",
        )

    @pyqtSlot(str, str)
    def on_worker_error(self, session: str, error_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_track_only.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(f"Error on {session}: {error_msg}")
        QMessageBox.critical(self, "Processing Error", f"Failed on session '{session}':\n\n{error_msg}")

    def execute_tracking_only(self):
        video_sessions = {k: v for k, v in self.matched_pairs.items() if v.get("video")}
        if not video_sessions:
            QMessageBox.warning(self, "Warning", "No video files detected in experiment list! Please import .mp4 / .avi files.")
            return

        self.sync_ui_to_config()
        self.btn_track_only.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("Initializing video vision tracking...")

        worker = TrackingOnlyWorker(
            matched_pairs=self.matched_pairs,
            config=self.config,
            save_raw_csv=self.cb_save_raw.isChecked(),
        )
        worker.signals.progress.connect(self.update_status_progress)
        worker.signals.session_finished.connect(self.on_tracking_session_finished)
        worker.signals.finished.connect(self.on_tracking_finished)
        worker.signals.error.connect(self.on_worker_error)

        self.current_worker = worker
        self.thread_pool.start(worker)

    def execute_tasks(self):
        if not self.matched_pairs:
            QMessageBox.warning(self, "Warning", "No sessions loaded for execution!")
            return

        self.sync_ui_to_config()
        self.btn_run.setEnabled(False)
        self.btn_track_only.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("Starting asynchronous processing thread pool...")

        worker = PipelineBatchWorker(
            matched_pairs=self.matched_pairs,
            config=self.config,
            anesthesia_onset_time=float(self.spin_gas_onset.value()),
            save_raw_csv=self.cb_save_raw.isChecked(),
            save_cleaned_csv=self.cb_save_clean.isChecked(),
            plot_act_pos=self.cb_plot_act_pos.isChecked(),
            plot_kymo=self.cb_plot_kymo.isChecked(),
            render_video_overlay=self.cb_video_overlay.isChecked(),
        )
        worker.signals.progress.connect(self.update_status_progress)
        worker.signals.session_finished.connect(self.on_session_finished)
        worker.signals.finished.connect(self.on_worker_finished)
        worker.signals.error.connect(self.on_worker_error)

        self.current_worker = worker
        self.thread_pool.start(worker)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()