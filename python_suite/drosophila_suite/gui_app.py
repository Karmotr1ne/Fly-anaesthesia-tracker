"""
Desktop Application (PyQt6)
===========================
Workstation Desktop GUI: Supports drag-and-drop file pairing, interactive multi-chamber grid calibration,
fly vision tracking, kinematic cleaning, and anesthesia induction kinetics analysis.
"""

import os
import re
import sys
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

try:
    from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject, QRunnable, QThreadPool, QPointF, QRectF
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QPainter, QFont, QBrush, QCursor
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
        QProgressBar,
        QListWidget,
        QMessageBox,
        QFrame,
        QFileDialog,
        QSpinBox,
        QDoubleSpinBox,
        QTabWidget,
        QTextEdit,
        QSlider,
        QButtonGroup,
        QRadioButton,
        QComboBox,
        QScrollArea,
        QGridLayout,
    )
except ImportError:
    pass

import cv2

try:
    from .pipeline import DrosophilaBehaviorPipeline
    from .models import PipelineConfig
    from .tracker import (
        RobustGridAligner,
        SymmetricGridAligner,
        Interactive8ChamberCalibrator,
        RobustFlyTracker,
        FlyVisionTracker,
        get_video_metadata,
    )
except (ImportError, ValueError):
    # Fallback when executed directly as a standalone script
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(cur_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from drosophila_suite.pipeline import DrosophilaBehaviorPipeline
    from drosophila_suite.models import PipelineConfig
    from drosophila_suite.tracker import (
        RobustGridAligner,
        SymmetricGridAligner,
        Interactive8ChamberCalibrator,
        RobustFlyTracker,
        FlyVisionTracker,
        get_video_metadata,
    )

from enum import Enum

class SessionPhase(str, Enum):
    IDLE = "Idle"
    READING_CSV = "Loading CSV"
    TRACKING = "Vision Tracking"
    CLEANING = "Kinematic Cleaning"
    ANALYZING = "Anesthesia Kinetics"
    PLOTTING = "Generating Plots"
    RENDERING = "Rendering Overlay Video"
    COMPLETED = "Completed"
    FAILED = "Failed"


# =====================================================================
# Interactive Multi-Chamber Calibration Canvas (Direct Drag & 8-Way Handles)
# =====================================================================
class InteractiveChamberCanvas(QWidget):
    chamberSelected = pyqtSignal(int)
    firstRoiDrawn = pyqtSignal(tuple)
    boxChanged = pyqtSignal()
    selectionChanged = pyqtSignal(int)  # 传递当前选中的数量

    MIN_WIDTH = 25
    MIN_HEIGHT = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.sample_frame = None
        self.boxes: List[List[int]] = []

        # 撤销历史栈
        self.undo_stack: List[Tuple[List[List[int]], set, int]] = []
        self.max_undo = 30

        # 选中状态管理：默认单选
        self.selected_idx = -1
        self.selected_indices = set()

        # 默认模式改为 "single"
        self.link_mode = "single"  # "single", "col", "row", "all" (表示所有选中)
        self.rows = 4
        self.cols = 2
        self.show_mask = False
        self.diff_thresh = 14
        self.fly_centroids: Dict[int, Optional[Tuple[float, float]]] = {}

        # 绘制与框选状态
        self.is_drawing_first = False
        self.draw_start_point = None
        self.current_drawing_rect = None

        self.is_box_selecting = False
        self.select_start_img = None
        self.current_select_rect_img = None
        self.press_pos = None

        # 拖拽与缩放状态
        self.drag_mode = None
        self.drag_start_pos = None
        self.drag_initial_boxes = []

        self.setStyleSheet("background-color: #0F172A; border-radius: 8px;")
        self.setMinimumSize(720, 480)

    def _push_undo(self):
        """保存当前 boxes 副本与选择状态到撤销栈"""
        state = ([list(b) for b in self.boxes], set(self.selected_indices), self.selected_idx)
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def undo(self):
        """撤销到上一步"""
        if not self.undo_stack:
            return
        boxes_prev, indices_prev, idx_prev = self.undo_stack.pop()
        self.boxes = [list(b) for b in boxes_prev]
        self.selected_indices = set(indices_prev)
        self.selected_idx = idx_prev
        self._update_fly_detections()
        self.selectionChanged.emit(len(self.selected_indices))
        self.boxChanged.emit()
        self.update()

    def set_data(self, frame: np.ndarray, boxes: List[List[int]], rows: int = 4, cols: int = 2):
        self.sample_frame = frame
        self.boxes = [list(b) for b in boxes]
        self.rows = rows
        self.cols = cols
        self.is_drawing_first = (len(self.boxes) == 0)
        self.undo_stack.clear()
        
        # 默认选中第 1 个 chamber
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
        gray = cv2.cvtColor(self.sample_frame, cv2.COLOR_BGR2GRAY)
        tracker = RobustFlyTracker(
            chamber_rois=[tuple(b) for b in self.boxes],
            diff_thresh=self.diff_thresh
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

        # 适当扩大手柄触控范围至 10px，避免鼠标难以精准对准
        handle_hit_margin = 10.0

        # 1. 优先判定当前主选中 Chamber 的 8 个调节手柄
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

            # 四角手柄
            if near_l and near_t: return "resize_tl", self.selected_idx
            if near_r and near_t: return "resize_tr", self.selected_idx
            if near_l and near_b: return "resize_bl", self.selected_idx
            if near_r and near_b: return "resize_br", self.selected_idx
            
            # 四边手柄
            if near_l and in_y_range: return "resize_l", self.selected_idx
            if near_r and in_y_range: return "resize_r", self.selected_idx
            if near_t and in_x_range: return "resize_t", self.selected_idx
            if near_b and in_x_range: return "resize_b", self.selected_idx

        # 2. 检查是否点击在任何 Chamber 内部（用于移动）
        for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
            kx1, ky1 = self.img_to_canvas(x1, y1)
            kx2, ky2 = self.img_to_canvas(x2, y2)
            if kx1 <= cx <= kx2 and ky1 <= cy <= ky2:
                return "move", idx

        return None, -1

    def mouseMoveEvent(self, event):
        cx, cy = event.position().x(), event.position().y()

        # 1. 首次绘制模式
        if self.is_drawing_first and self.draw_start_point:
            ix, iy = self.canvas_to_img(cx, cy)
            sx, sy = self.draw_start_point
            self.current_drawing_rect = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.update()
            return

        # 2. 空白处框选移动
        if self.is_box_selecting and self.select_start_img:
            ix, iy = self.canvas_to_img(cx, cy)
            sx, sy = self.select_start_img
            self.current_select_rect_img = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.update()
            return

        # 3. 拖拽移动与边缘拉伸
        if self.drag_mode and self.drag_start_pos:
            s, _, _ = self.get_scale_and_offsets()
            dx = (cx - self.drag_start_pos[0]) / s
            dy = (cy - self.drag_start_pos[1]) / s
            img_h, img_w = self.sample_frame.shape[:2] if self.sample_frame is not None else (1000, 1000)

            if not self.selected_indices:
                self.selected_indices = {self.selected_idx} if self.selected_idx >= 0 else {0}
                if self.selected_idx < 0:
                    self.selected_idx = 0
                self.selectionChanged.emit(len(self.selected_indices))

            sel_row = self.selected_idx % self.rows
            sel_col = self.selected_idx // self.rows

            # 先克隆一份临时候选数据预演移动/拉伸
            tentative_boxes = [list(b) for b in self.boxes]

            for idx in range(len(self.boxes)):
                apply = False
                cur_row = idx % self.rows
                cur_col = idx // self.rows

                if len(self.selected_indices) > 1:
                    apply = (idx in self.selected_indices)
                else:
                    if self.link_mode == "all":
                        apply = (idx in self.selected_indices)
                    elif self.link_mode == "col" and cur_col == sel_col:
                        apply = True
                    elif self.link_mode == "row" and cur_row == sel_row:
                        apply = True
                    elif self.link_mode == "single":
                        apply = (idx in self.selected_indices or idx == self.selected_idx)

                if apply:
                    ox1, oy1, ox2, oy2 = self.drag_initial_boxes[idx]
                    nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2

                    if self.drag_mode == "move":
                        nx1, ny1 = ox1 + dx, oy1 + dy
                        nx2, ny2 = ox2 + dx, oy2 + dy
                    else:
                        # 边角手柄拉伸计算
                        if "resize_l" in self.drag_mode or self.drag_mode in ("resize_tl", "resize_bl"):
                            nx1 = min(ox2 - self.MIN_WIDTH, ox1 + dx)
                        if "resize_r" in self.drag_mode or self.drag_mode in ("resize_tr", "resize_br"):
                            nx2 = max(ox1 + self.MIN_WIDTH, ox2 + dx)
                        if "resize_t" in self.drag_mode or self.drag_mode in ("resize_tl", "resize_tr"):
                            ny1 = min(oy2 - self.MIN_HEIGHT, oy1 + dy)
                        if "resize_b" in self.drag_mode or self.drag_mode in ("resize_bl", "resize_br"):
                            ny2 = max(oy1 + self.MIN_HEIGHT, oy2 + dy)

                    tentative_boxes[idx][0] = int(np.clip(nx1, 0, img_w - self.MIN_WIDTH))
                    tentative_boxes[idx][1] = int(np.clip(ny1, 0, img_h - self.MIN_HEIGHT))
                    tentative_boxes[idx][2] = int(np.clip(nx2, self.MIN_WIDTH, img_w))
                    tentative_boxes[idx][3] = int(np.clip(ny2, self.MIN_HEIGHT, img_h))

            # 优化碰撞检查：允许贴合，仅当被修改的 box 发生实质性侵入穿透时才阻断
            can_apply = True
            for idx in range(len(self.boxes)):
                # 如果这个 box 根本没参与本次拖动变形，跳过检测
                if tentative_boxes[idx] == self.drag_initial_boxes[idx]:
                    continue
                # 与其他 box 进行碰撞检测
                for other_idx in range(len(self.boxes)):
                    if idx == other_idx:
                        continue
                    # 容许 2px 的边界重叠误差，防止相邻边缘直接把拉伸卡死
                    if self._is_intersecting(tentative_boxes[idx], tentative_boxes[other_idx], margin=2):
                        can_apply = False
                        break
                if not can_apply:
                    break

            if can_apply:
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

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.MouseButton.LeftButton:
            cx, cy = event.position().x(), event.position().y()
            self.press_pos = (cx, cy)
            modifiers = event.modifiers()
            has_modifier = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))

            if self.is_drawing_first:
                ix, iy = self.canvas_to_img(cx, cy)
                self.draw_start_point = (ix, iy)
                self.current_drawing_rect = (ix, iy, ix, iy)
                self.update()
                return

            action, idx = self._hit_test(cx, cy)

            if action is None:
                # 点击在 Chamber 外部：准备橡皮筋框选
                ix, iy = self.canvas_to_img(cx, cy)
                self.is_box_selecting = True
                self.select_start_img = (ix, iy)
                self.current_select_rect_img = (ix, iy, ix, iy)
                self.update()
                return

            # 点击了手柄或内部
            if action == "move":
                if has_modifier:
                    # Shift/Ctrl 加选 / 减选
                    if idx in self.selected_indices:
                        self.selected_indices.remove(idx)
                        self.selected_idx = next(iter(self.selected_indices)) if self.selected_indices else -1
                    else:
                        self.selected_indices.add(idx)
                        self.selected_idx = idx
                else:
                    # 若点击的项未在当前多选组内，则重置为仅单选该项
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

    def mouseReleaseEvent(self, event):
        # 1. 首次手工拉框
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

        # 2. 外部框选或单击 Chamber 外取消选择
        if self.is_box_selecting:
            cx, cy = event.position().x(), event.position().y()
            is_click = self.press_pos and (abs(cx - self.press_pos[0]) < 4 and abs(cy - self.press_pos[1]) < 4)

            if is_click:
                # 选定 chamber 时，单击 chamber 外直接取消选择
                modifiers = event.modifiers()
                has_modifier = bool(modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier))
                if not has_modifier:
                    self.selected_indices.clear()
                    self.selected_idx = -1
                    self.selectionChanged.emit(0)
            elif self.current_select_rect_img:
                rx1, ry1, rx2, ry2 = self.current_select_rect_img
                if (rx2 - rx1) > 5 and (ry2 - ry1) > 5:
                    hit_set = set()
                    for idx, (bx1, by1, bx2, by2) in enumerate(self.boxes):
                        if not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                            hit_set.add(idx)

                    modifiers = event.modifiers()
                    if modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
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

        # 3. 拖拽变形结束：推入撤销栈
        if self.drag_mode:
            if self.drag_initial_boxes != self.boxes:
                # 将初始状态作为撤销点推入栈
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

        if not self.selected_indices:
            self.selected_indices = set(range(len(self.boxes)))
            self.selected_idx = 0
            self.selectionChanged.emit(len(self.selected_indices))

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

        img_h, img_w = self.sample_frame.shape[:2] if self.sample_frame is not None else (1000, 1000)
        tentative_boxes = [list(b) for b in self.boxes]

        for idx in list(self.selected_indices):
            x1, y1, x2, y2 = tentative_boxes[idx]
            w, h = x2 - x1, y2 - y1
            nx1 = np.clip(x1 + dx, 0, img_w - w)
            ny1 = np.clip(y1 + dy, 0, img_h - h)
            tentative_boxes[idx] = [int(nx1), int(ny1), int(nx1 + w), int(ny1 + h)]

        # 碰撞校验：无重合才真正保存并生效
        if not self._has_any_overlap(tentative_boxes):
            self._push_undo()
            self.boxes = tentative_boxes
            self.boxChanged.emit()
            self._update_fly_detections()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.sample_frame is None:
            painter.setPen(QColor("#64748B"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video frame available")
            return

        s, ox, oy = self.get_scale_and_offsets()
        img_h, img_w = self.sample_frame.shape[:2]

        if self.show_mask:
            gray = cv2.cvtColor(self.sample_frame, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            bh_s = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_small)
            bh_l = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_large)
            energy = cv2.addWeighted(bh_s, 0.6, bh_l, 0.4, 0)
            energy = cv2.addWeighted(energy, 0.7, inv, 0.3, 0)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(energy)
            norm_energy = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            _, thresh_mask = cv2.threshold(norm_energy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            boosted = cv2.addWeighted(norm_energy, 0.7, thresh_mask, 0.3, 0)
            vis = cv2.applyColorMap(boosted, cv2.COLORMAP_JET)
        else:
            vis = self.sample_frame.copy()

        rgb_frame = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        painter.drawImage(QRectF(ox, oy, w * s, h * s), qimg)

        # 中轴参考线
        mid_x = ox + (img_w * 0.5) * s
        painter.setPen(QPen(QColor("#94A3B8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(mid_x), int(oy), int(mid_x), int(oy + img_h * s))

        # 首次绘制引导
        if self.is_drawing_first and self.current_drawing_rect:
            x1, y1, x2, y2 = self.current_drawing_rect
            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(245, 158, 11, 40)))
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))
            painter.setPen(QColor("#FDE68A"))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(int(rx1 + 6), int(ry1 + 18), "CH 1 (Release mouse to infer grid)")
            return

        font_label = QFont("Arial", 9, QFont.Weight.Bold)
        font_dim = QFont("Arial", 8)

        # 绘制所有 Chamber
        for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
            cid = idx + 1
            is_active = (idx == self.selected_idx)
            is_in_group = (idx in self.selected_indices)

            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            rw, rh = rx2 - rx1, ry2 - ry1

            if is_active:
                border_color = QColor("#F59E0B")
                fill_color = QColor(245, 158, 11, 35)
                line_w = 2.5
            elif is_in_group:
                border_color = QColor("#38BDF8")
                fill_color = QColor(56, 189, 248, 25)
                line_w = 2.0
            else:
                border_color = QColor("#10B981")
                fill_color = QColor(16, 185, 129, 15)
                line_w = 1.2

            painter.setPen(QPen(border_color, line_w))
            painter.setBrush(QBrush(fill_color))
            painter.drawRoundedRect(QRectF(rx1, ry1, rw, rh), 4, 4)

            # 内边缘标线
            painter.setPen(QPen(QColor("#F97316"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(rx1 + rw * 0.08), int(ry1), int(rx1 + rw * 0.08), int(ry2))
            painter.setPen(QPen(QColor("#38BDF8"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(int(rx2 - rw * 0.08), int(ry1), int(rx2 - rw * 0.08), int(ry2))

            # ID Badge
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

            # 十字质心
            if cid in self.fly_centroids and self.fly_centroids[cid] is not None:
                fx, fy = self.fly_centroids[cid]
                cfx, cfy = self.img_to_canvas(fx, fy)
                painter.setPen(QPen(QColor("#22C55E"), 1))
                painter.drawLine(int(cfx - 7), int(cfy), int(cfx + 7), int(cfy))
                painter.drawLine(int(cfx), int(cfy - 7), int(cfx), int(cfy + 7))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor("#EF4444")))
                painter.drawEllipse(QPointF(cfx, cfy), 3.5, 3.5)

            # 仅在当前主活跃项上绘制拉伸手柄
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

        # 框选矩形
        if self.is_box_selecting and self.current_select_rect_img:
            x1, y1, x2, y2 = self.current_select_rect_img
            rx1, ry1 = self.img_to_canvas(x1, y1)
            rx2, ry2 = self.img_to_canvas(x2, y2)
            painter.setPen(QPen(QColor("#38BDF8"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(56, 189, 248, 45)))
            painter.drawRect(QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1))

    @staticmethod
    def _is_intersecting(box_a: List[int], box_b: List[int], margin: int = 1) -> bool:
        """矩形 AABB 相交判断（支持容差 margin，避免相邻试管贴合时误判碰撞）"""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        # 若存在任意一条轴分离（计入 margin 容差），则不相交
        if (ax2 - margin) <= (bx1 + margin) or (ax1 + margin) >= (bx2 - margin) or \
           (ay2 - margin) <= (by1 + margin) or (ay1 + margin) >= (by2 - margin):
            return False
        return True

    def _has_any_overlap(self, candidate_boxes: List[List[int]], margin: int = 1) -> bool:
        """检查整组 boxes 内部是否存在任意两两实质性重合"""
        n = len(candidate_boxes)
        for i in range(n):
            for j in range(i + 1, n):
                if self._is_intersecting(candidate_boxes[i], candidate_boxes[j], margin=margin):
                    return True
        return False


class ChamberCalibrationDialog(QDialog):
    def __init__(self, video_path: str, parent=None, initial_chambers=None, rows: int = 4, cols: int = 2, order: str = "column_first"):
        super().__init__(parent)
        self.setWindowTitle(f"Multi-Chamber Grid Calibrator - {os.path.basename(video_path)}")
        self.resize(1120, 720)
        self.video_path = video_path
        self.sample_frame = None
        self.order = order
        self.last_first_box = None

        # 核心防护：若已有选框，以已有选框数量为主
        if initial_chambers and len(initial_chambers) > 0:
            self.boxes = [list(b) for b in initial_chambers]
            self.last_first_box = tuple(self.boxes[0])
            # 如果已有数量与传入行列不符，维持传入 rows，动态适配 cols
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

        # Left: Interactive Canvas
        left_layout = QVBoxLayout()
        self.canvas = InteractiveChamberCanvas()
        self.canvas.firstRoiDrawn.connect(self._on_first_roi_drawn)
        self.canvas.boxChanged.connect(self._on_canvas_box_changed)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.canvas, 1)

        self.tip_label = QLabel(f"<b>Draw the first tube (CH 1) in top-left</b>: System infers {self.rows}x{self.cols} grid. Press Ctrl+Z to undo.")
        self.tip_label.setStyleSheet("color: #E2E8F0; font-size: 13px; background-color: #1E293B; padding: 8px; border-radius: 6px;")
        left_layout.addWidget(self.tip_label)

        # Right: Control Panel
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 1. Grid Geometry
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

        # 2. Link Mode (重命名 "All" -> "移动所有选中", 默认 "Single")
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

        # 3. Vision Tools, Auto-Snap & Undo
        grp_tools = QGroupBox("3. Visual Tools & History")
        v_tools = QVBoxLayout()
        
        btn_undo = QPushButton("Undo Last Action")
        btn_undo.setStyleSheet("background-color: #475569; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        btn_undo.clicked.connect(self.canvas.undo)
        v_tools.addWidget(btn_undo)

        btn_snap = QPushButton("Auto-Snap Tube Boundaries")
        btn_snap.setStyleSheet("background-color:  #334155; color: white; padding: 6px; border-radius: 5px;")
        btn_snap.clicked.connect(self._on_auto_snap)
        v_tools.addWidget(btn_snap)

        self.btn_mask = QPushButton("Toggle Darkness Energy Mask")
        self.btn_mask.setStyleSheet("background-color: #334155; color: white; padding: 6px; border-radius: 5px;")
        self.btn_mask.clicked.connect(self._toggle_mask)
        v_tools.addWidget(self.btn_mask)
        grp_tools.setLayout(v_tools)
        right_layout.addWidget(grp_tools)

        # 4. Quick Chamber Selector
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

        # Dialog Buttons
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
        """当多选时，禁用按行/按列模式"""
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

        # 1. 调用刚性 1D 吸附策略
        refined = RobustGridAligner.snap_all_boxes(
            frame_bgr=self.sample_frame,
            boxes=base_boxes,
            rows=self.rows,
            cols=self.cols
        )

        # 2. 幂等性检查：若所有单边变化 <= 1px 则认定已完全贴合，不再重复更新
        has_effective_change = False
        for old, new in zip(base_boxes, refined):
            if any(abs(o - n) > 1 for o, n in zip(old, new)):
                has_effective_change = True
                break

        if not has_effective_change:
            return

        # 3. 提交生效
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
        else:
            self.tip_label.setText(f"<b>Draw the first tube (CH 1) in top-left</b>: System will infer {self.rows} Rows × {self.cols} Cols.")

    def _recompute_inference(self):
        if self.sample_frame is not None and self.last_first_box is not None:
            estimated_boxes = RobustGridAligner.estimate_chambers_from_first_roi(
                self.sample_frame,
                self.last_first_box,
                rows=self.rows,
                cols=self.cols,
                order=self.order
            )
            self.boxes = [list(b) for b in estimated_boxes]
            self.canvas.set_data(self.sample_frame, self.boxes, self.rows, self.cols)
            self._rebuild_chamber_buttons()
            self.tip_label.setText(f"<b>Generated {len(self.boxes)} Chambers ({self.rows}x{self.cols})</b>. You can directly drag on the canvas.")

    def _on_click_redraw(self):
        self.canvas.start_redraw_first_roi()
        self.tip_label.setText(f"<b>Draw the first tube (CH 1) in top-left</b>...")

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
        self.canvas.update()

    def _on_canvas_box_changed(self):
        self.boxes = self.canvas.boxes
        if len(self.boxes) > 0:
            self.last_first_box = tuple(self.boxes[0])

    def _set_mode(self, mode: str):
        self.canvas.link_mode = mode

    def _toggle_mask(self):
        self.canvas.show_mask = not self.canvas.show_mask
        self.canvas.update()

    def get_chambers(self) -> List[Tuple[int, int, int, int]]:
        return [tuple(b) for b in self.canvas.boxes]


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
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #90A4AE;
                border-radius: 8px;
                background-color: #FAFAFA;
            }
        """)
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
            "Experiment Data & Videos (*.csv *.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if files:
            for f in files:
                if f not in self.all_files:
                    self.all_files.append(f)
            self.filesChanged.emit(self.all_files)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #1976D2;
                    border-radius: 8px;
                    background-color: #E3F2FD;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #90A4AE;
                border-radius: 8px;
                background-color: #FAFAFA;
            }
        """)

    def dropEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #90A4AE;
                border-radius: 8px;
                background-color: #FAFAFA;
            }
        """)
        urls = event.mimeData().urls()
        new_files = [u.toLocalFile() for u in urls if u.toLocalFile()]
        for f in new_files:
            if f not in self.all_files:
                self.all_files.append(f)
        self.filesChanged.emit(self.all_files)


# =====================================================================
# Background Asynchronous Workers
# =====================================================================
class WorkerSignals(QObject):
    progress = pyqtSignal(int, int, str)
    session_finished = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)


class TrackingOnlyWorker(QRunnable):
    """
    Dedicated worker for batch video vision tracking, producing *_raw.csv.
    """
    def __init__(self, matched_pairs: Dict[str, dict], config: PipelineConfig):
        super().__init__()
        self.matched_pairs = matched_pairs
        self.config = config
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
                self.signals.progress.emit(processed, total_sessions, f"Task cancelled ({idx}/{total_sessions}).")
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
                        rows=self.config.grid_rows,
                        cols=self.config.grid_cols,
                        order=self.config.grid_order,
                    )
                    ch_rois = [c["roi"] for c in configs]
                    paths["chamber_rois"] = ch_rois

            out_dir = os.path.dirname(os.path.abspath(vid_path))
            raw_csv_path = os.path.join(out_dir, f"{base}_raw.csv")

            start_t = time.time()
            last_t = [start_t]

            def on_frame_progress(f_cur, f_tot):
                if f_tot > 0:
                    curr_t = time.time()
                    dt = max(1e-5, curr_t - last_t[0])
                    fps_val = 100.0 / dt
                    last_t[0] = curr_t
                    pct = int((f_cur / f_tot) * 100)
                    self.signals.progress.emit(
                        processed,
                        total_sessions,
                        f"Tracking [{base}]: {pct}% ({f_cur}/{f_tot} frames) | {fps_val:.1f} fps"
                    )

            try:
                tracker = FlyVisionTracker(chamber_rois=ch_rois)
                raw_df = tracker.track_video(vid_path, progress_callback=on_frame_progress)
                raw_df.to_csv(raw_csv_path, index=False)
                
                paths["csv"] = raw_csv_path
                all_results[base] = {"raw_csv": raw_csv_path, "frames": len(raw_df)}
                processed += 1
                self.signals.session_finished.emit(base, all_results[base])
                self.signals.progress.emit(processed, total_sessions, f"Finished video tracking: {base} ({processed}/{total_sessions})")
            except Exception as e:
                self.signals.error.emit(base, str(e))

        self.signals.finished.emit(all_results)


class PipelineBatchWorker(QRunnable):
    """
    Worker for end-to-end full pipeline execution (Modules 1 to 5).
    """
    def __init__(
        self,
        matched_pairs: Dict[str, dict],
        config: PipelineConfig,
        save_cleaned_csv: bool = True,
        generate_plots: bool = True,
        render_video_overlay: bool = False,
    ):
        super().__init__()
        self.matched_pairs = matched_pairs
        self.config = config
        self.save_cleaned_csv = save_cleaned_csv
        self.generate_plots = generate_plots
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
        pipeline = DrosophilaBehaviorPipeline(self.config)

        for idx, (base, paths) in enumerate(self.matched_pairs.items(), start=1):
            if self._is_cancelled:
                self.signals.progress.emit(
                    processed, total_sessions, f"[{SessionPhase.IDLE.value}] Batch job cancelled ({idx}/{total_sessions})."
                )
                break

            has_csv = bool(paths.get("csv") and os.path.exists(paths.get("csv")))
            has_vid = bool(paths.get("video") and os.path.exists(paths.get("video")))

            phase_label = (
                "Cleaning & State Analysis" if has_csv else SessionPhase.TRACKING.value
            )
            self.signals.progress.emit(
                processed,
                total_sessions,
                f"[{phase_label}] {base} ({idx}/{total_sessions})..."
            )

            try:
                ch_rois = paths.get("chamber_rois")
                if not has_csv and has_vid and not ch_rois:
                    cap = cv2.VideoCapture(paths["video"])
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        chamber_configs = SymmetricGridAligner.generate_symmetric_chambers(
                            frame_shape=frame.shape,
                            rows=self.config.grid_rows,
                            cols=self.config.grid_cols,
                            order=self.config.grid_order,
                        )
                        ch_rois = [c["roi"] for c in chamber_configs]

                def on_tracking_progress(f_cur, f_tot):
                    if f_tot > 0:
                        pct = int((f_cur / f_tot) * 100)
                        self.signals.progress.emit(
                            processed,
                            total_sessions,
                            f"[{SessionPhase.TRACKING.value}] {base}: {pct}% ({f_cur}/{f_tot} frames)"
                        )

                def on_rendering_progress(f_cur, f_tot):
                    if f_tot > 0:
                        pct = int((f_cur / f_tot) * 100)
                        self.signals.progress.emit(
                            processed,
                            total_sessions,
                            f"[{SessionPhase.RENDERING.value}] {base}: {pct}% ({f_cur}/{f_tot} frames)"
                        )

                res = pipeline.process_file_pair(
                    csv_path=paths.get("csv"),
                    video_path=paths.get("video"),
                    base_name=base,
                    chamber_rois=ch_rois,
                    save_cleaned_csv=self.save_cleaned_csv,
                    generate_plots=self.generate_plots,
                    render_video_overlay=self.render_video_overlay,
                    progress_callback=on_tracking_progress,
                    render_progress_callback=on_rendering_progress
                )
                
                all_results[base] = res
                processed += 1
                self.signals.session_finished.emit(base, res)
                self.signals.progress.emit(
                    processed, total_sessions, f"[{SessionPhase.COMPLETED.value}] {base} ({processed}/{total_sessions})"
                )
            except Exception as e:
                self.signals.error.emit(base, str(e))
                self.signals.progress.emit(
                    processed, total_sessions, f"[{SessionPhase.FAILED.value}] Error on {base}"
                )

        self.signals.finished.emit(all_results)


# =====================================================================
# Main Application Window
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drosophila Anesthesia Tracker & Multi-Chamber Workstation (v0.1)")
        self.resize(1180, 820)
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

        # Left Column: File Management & Pairing
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        header_import = QLabel("<b>1. Batch File Import</b>")
        left_layout.addWidget(header_import)

        self.drop_area = DragDropArea()
        self.drop_area.filesChanged.connect(self.on_files_updated)
        left_layout.addWidget(self.drop_area)

        header_pairs = QLabel("<b>2. Experiment Sessions & Pairing Status</b>")
        left_layout.addWidget(header_pairs)

        self.pair_list = QListWidget()
        self.pair_list.itemDoubleClicked.connect(self.calibrate_selected_session)
        left_layout.addWidget(self.pair_list)

        btn_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("Calibrate Chamber ROI")
        self.btn_calibrate.clicked.connect(self.calibrate_selected_session)
        self.btn_calibrate.setStyleSheet("""
            QPushButton {
                background-color: #0284C7;
                color: white;
                border: 1px solid #0369A1;
                padding: 7px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #0369A1; }
        """)
        btn_row.addWidget(self.btn_calibrate)

        btn_clear = QPushButton("Clear List")
        btn_clear.clicked.connect(self.clear_all)
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #ECEFF1;
                border: 1px solid #B0BEC5;
                padding: 7px 12px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #CFD8DC; }
        """)
        btn_row.addWidget(btn_clear)
        left_layout.addLayout(btn_row)

        # Right Column: Modules & Execution Parameters
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        header_tasks = QLabel("<b>3. Modules & Analysis Settings</b>")
        right_layout.addWidget(header_tasks)

        # Module 1: Vision Tracking
        grp_track = QGroupBox("Module 1: Fly Tracking")
        vbox_track = QVBoxLayout()
        self.cb_save_raw = QCheckBox("Export Raw Location Data (*_raw.csv)")
        self.cb_save_raw.setChecked(True)
        vbox_track.addWidget(self.cb_save_raw)
        grp_track.setLayout(vbox_track)
        right_layout.addWidget(grp_track)

        # Module 2: Kinematic Cleaning
        grp_a = QGroupBox("Module 2: Kinematic Cleaning & Artifact Clamping")
        vbox_a = QVBoxLayout()
        self.cb_save_clean = QCheckBox("Export Cleaned Location Data (*_cleaned.csv)")
        self.cb_save_clean.setChecked(True)
        vbox_a.addWidget(self.cb_save_clean)
        grp_a.setLayout(vbox_a)
        right_layout.addWidget(grp_a)

# Module 4: 3-State Kinetics & Anesthesia
        grp_b = QGroupBox("Module 4: 3-State Machine & Kinetics")
        vbox_b = QVBoxLayout()
        vbox_b.setSpacing(8)

        # 1. 速度衰减阈值比例
        h_spd = QHBoxLayout()
        h_spd.addWidget(QLabel("Sedate Speed Dropping Ratio:"))
        self.spin_speed_ratio = QDoubleSpinBox()
        self.spin_speed_ratio.setRange(0.05, 0.90)
        self.spin_speed_ratio.setSingleStep(0.05)
        self.spin_speed_ratio.setValue(0.35)
        h_spd.addWidget(self.spin_speed_ratio)
        vbox_b.addLayout(h_spd)

        # 2. 1秒单步跌落高度差阈值
        h_drop = QHBoxLayout()
        h_drop.addWidget(QLabel("Sedate Drop Height (percentage/1s):"))
        self.spin_drop_thresh = QDoubleSpinBox()
        self.spin_drop_thresh.setRange(0.05, 0.80)
        self.spin_drop_thresh.setSingleStep(0.05)
        self.spin_drop_thresh.setValue(0.25)
        h_drop.addWidget(self.spin_drop_thresh)
        vbox_b.addLayout(h_drop)

        # 3. 麻醉持续极值静止时长
        h_still = QHBoxLayout()
        h_still.addWidget(QLabel("Anaesthesia Time Window:"))
        self.spin_still_sec = QSpinBox()
        self.spin_still_sec.setRange(30, 600)
        self.spin_still_sec.setSingleStep(10)
        self.spin_still_sec.setValue(120)
        self.spin_still_sec.setSuffix(" s")
        h_still.addWidget(self.spin_still_sec)
        vbox_b.addLayout(h_still)

        grp_b.setLayout(vbox_b)
        right_layout.addWidget(grp_b)

        # Module 5: Scientific Visualizer
        grp_c = QGroupBox("Module 5: Scientific Graphics & Video Synthesis")
        vbox_c = QVBoxLayout()
        self.cb_plot_act_pos = QCheckBox("Dual Y-Axis Behavioral Overview (*_activity_position.png)")
        self.cb_plot_act_pos.setChecked(True)
        self.cb_plot_kymo = QCheckBox("Normalized Space-Time Kymograph (*_kymograph_norm.png)")
        self.cb_plot_kymo.setChecked(True)
        self.cb_video_overlay = QCheckBox("Render Annotated Video Overlay (*_overlay.mp4)")
        vbox_c.addWidget(self.cb_plot_act_pos)
        vbox_c.addWidget(self.cb_plot_kymo)
        vbox_c.addWidget(self.cb_video_overlay)
        grp_c.setLayout(vbox_c)
        right_layout.addWidget(grp_c)

        right_layout.addStretch()

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #546E7A; font-size: 13px;")
        right_layout.addWidget(self.lbl_status)

        # Bottom Execution Buttons
        h_exec_row = QHBoxLayout()
        
        self.btn_track_only = QPushButton("Track Fly Only")
        self.btn_track_only.setFixedHeight(46)
        self.btn_track_only.setStyleSheet("""
            QPushButton {
                background-color: #0284C7;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #0369A1; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.btn_track_only.clicked.connect(self.execute_tracking_only)
        h_exec_row.addWidget(self.btn_track_only, 2)

        self.btn_run = QPushButton("Run Selected Module")
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #1B5E20; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.btn_run.clicked.connect(self.execute_tasks)
        h_exec_row.addWidget(self.btn_run, 3)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setFixedHeight(46)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #D32F2F;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 6px;
            }
            QPushButton:hover { background-color: #B71C1C; }
            QPushButton:disabled { background-color: #EF9A9A; color: #ECEFF1; }
        """)
        self.btn_cancel.clicked.connect(self.cancel_execution)
        h_exec_row.addWidget(self.btn_cancel, 1)
        
        right_layout.addLayout(h_exec_row)

        main_layout.addLayout(left_layout, 5)
        main_layout.addLayout(right_layout, 4)

    def sync_config_to_ui(self):
        self.spin_speed_ratio.setValue(getattr(self.config, "sedate_speed_ratio", 0.35))
        self.spin_drop_thresh.setValue(getattr(self.config, "sedate_drop_speed", 0.25))
        self.spin_still_sec.setValue(int(getattr(self.config, "anesthesia_still_sec", 120.0)))

    def sync_ui_to_config(self):
        self.config.sedate_speed_ratio = float(self.spin_speed_ratio.value())
        self.config.sedate_drop_speed = float(self.spin_drop_thresh.value())
        self.config.anesthesia_still_sec = float(self.spin_still_sec.value())

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

        # 从当前会话读取几何参数（若无则兜底使用全局默认 4x2）
        cur_rois = session.get("chamber_rois")
        cur_rows = session.get("grid_rows", self.config.grid_rows)
        cur_cols = session.get("grid_cols", self.config.grid_cols)
        cur_order = session.get("grid_order", self.config.grid_order)

        dlg = ChamberCalibrationDialog(
            vid_path,
            parent=self,
            initial_chambers=cur_rois,
            rows=cur_rows,
            cols=cur_cols,
            order=cur_order
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # 将弹窗中最终确认的 ROI 坐标与几何参数一并固化到该 session
            rois = dlg.get_chambers()
            session["chamber_rois"] = rois
            session["grid_rows"] = dlg.rows
            session["grid_cols"] = dlg.cols
            session["grid_order"] = dlg.order
            
            QMessageBox.information(
                self,
                "Calibration Saved",
                f"Successfully saved {len(rois)} chambers ({dlg.rows}x{dlg.cols}) for:\n{base}"
            )

    def clear_all(self):
        self.drop_area.all_files = []
        self.pair_list.clear()
        self.matched_pairs = {}
        self.drop_area.label.setText("Drag & Drop CSV or Video files here\n(or click to browse)")
        self.lbl_status.setText("Ready, awaiting task execution.")

    def _on_anesthesia_duration_changed(self, val: int):
        bins = max(1, int(round(val / self.config.anesthesia_bin_size_sec)))
        minutes = val / 60.0
        self.lbl_window_info.setText(f"({minutes:.1f} min, {bins} bins)")

    def cancel_execution(self):
        if self.current_worker:
            self.lbl_status.setText("Cancelling task execution...")
            self.current_worker.cancel()
            self.btn_cancel.setEnabled(False)

    def on_worker_progress(self, processed: int, total: int, status_text: str):
        self.lbl_status.setText(status_text)

    def on_session_finished(self, base: str, res: dict):
        pass

    def on_worker_finished(self, results: dict):
        self.btn_run.setEnabled(True)
        self.btn_track_only.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.current_worker = None
        total = len(self.matched_pairs)
        completed = len(results)
        self.lbl_status.setText(f"Batch processing complete: {completed}/{total} sessions processed.")
        QMessageBox.information(
            self,
            "Execution Complete",
            f"All sessions processed successfully!\nCompleted: {completed}/{total} experiments."
        )

    def on_worker_error(self, base: str, error_msg: str):
        QMessageBox.critical(self, "Processing Error", f"Error encountered while analyzing {base}:\n{error_msg}")

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
            save_cleaned_csv=self.cb_save_clean.isChecked(),
            generate_plots=(self.cb_plot_act_pos.isChecked() or self.cb_plot_kymo.isChecked()),
            render_video_overlay=self.cb_video_overlay.isChecked(),
        )
        worker.signals.progress.connect(self.on_worker_progress)
        worker.signals.session_finished.connect(self.on_session_finished)
        worker.signals.finished.connect(self.on_worker_finished)
        worker.signals.error.connect(self.on_worker_error)
        self.current_worker = worker
        self.thread_pool.start(worker)

    def execute_tracking_only(self):
        video_sessions = {k: v for k, v in self.matched_pairs.items() if v.get("video")}
        if not video_sessions:
            QMessageBox.warning(self, "Warning", "No video files detected in experiment list! Please import .mp4 / .avi files.")
            return

        self.sync_ui_to_config()
        self.btn_track_only.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("Starting vision tracking worker thread pool...")

        worker = TrackingOnlyWorker(
            matched_pairs=self.matched_pairs,
            config=self.config
        )
        worker.signals.progress.connect(self.on_worker_progress)
        worker.signals.session_finished.connect(self.on_tracking_session_finished)
        worker.signals.finished.connect(self.on_tracking_finished)
        worker.signals.error.connect(self.on_worker_error)
        self.current_worker = worker
        self.thread_pool.start(worker)
        

    def on_tracking_session_finished(self, base: str, res: dict):
        self.pair_list.clear()
        for b, paths in self.matched_pairs.items():
            status = "[Paired] CSV + Video" if (paths.get("csv") and paths.get("video")) else "[Video Tracked]"
            self.pair_list.addItem(f"{b}  ->  {status}")

    def on_tracking_finished(self, results: dict):
        self.btn_track_only.setEnabled(True)
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.current_worker = None
        count = len(results)
        self.lbl_status.setText(f"Vision tracking complete: Generated {count} raw trajectory CSV files.")
        QMessageBox.information(
            self,
            "Tracking Complete",
            f"Vision tracking complete!\nSuccessfully extracted coordinates for {count} videos into *_raw.csv."
        )


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_gui()
