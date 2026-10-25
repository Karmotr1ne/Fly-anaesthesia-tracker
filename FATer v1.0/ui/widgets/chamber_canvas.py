"""Letterboxed chamber editing with a cached image and lightweight vector overlays."""
import copy
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import QWidget, QShortcut, QApplication
from core.tracking.tracker_engine import ChamberROI

class InteractiveChamberCanvas(QWidget):
    'Draw, snap, resize, translate, and undo chamber ROI edits.'
    first_roi_drawn = pyqtSignal(tuple)
    calibration_changed = pyqtSignal()
    selection_changed = pyqtSignal()
    geometry_restored = pyqtSignal(dict)

    HANDLE_SIZE = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 120)
        self.setMouseTracking(True)
        self.equal_size = True
        self.grid_rows, self.grid_cols = 4, 3
        self.grid_order = "column_first"
        self.draw_first_requested = False
        self.centroids = []
        self._trail_pixmap = None
        self.show_trail = False
        self._background_pixmap = None
        self._scaled_background = None
        self._scaled_size = None
        self._drag_anchor_index = 0
        self.setStyleSheet("background-color: #0F172A; border-radius: 4px;")
        self.setFocusPolicy(Qt.StrongFocus)

        self.raw_frame: np.ndarray | None = None
        self.gray_frame: np.ndarray | None = None
        self.rois: list[ChamberROI] = []

        self.selected_indices: set[int] = set()
        self.active_handle: str | None = None
        self.is_dragging = False
        self.drag_start_pos: QPointF | None = None
        self.drag_snapshot: list[ChamberROI] = []

        self.undo_stack: list[dict] = []

        self.drawing_first_roi = False
        self.rubber_band_active = False
        self.band_start_pos: QPointF | None = None
        self.band_cur_pos: QPointF | None = None

        self.shortcut_undo = QShortcut(QKeySequence.Undo, self)
        self.shortcut_undo.activated.connect(self.undo)

    def set_frame(self, frame: np.ndarray):
        self.raw_frame = frame
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        self.set_display_frame(frame)

    def set_trail(self, image):
        self._trail_pixmap = None
        if image is not None:
            image = np.ascontiguousarray(image)
            self._trail_pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0],
                image.strides[0], QImage.Format_RGBA8888).copy())
        self.update()

    def set_rois(self, rois: list[ChamberROI]):
        self._push_undo()
        self.rois = copy.deepcopy(rois)
        self.selected_indices = {0} if self.rois else set()
        self.selection_changed.emit()
        self.update()

    def _push_undo(self):
        if len(self.undo_stack) >= 30:
            self.undo_stack.pop(0)
        self.undo_stack.append({
            "rois":copy.deepcopy(self.rois), "grid_rows":self.grid_rows,
            "grid_cols":self.grid_cols, "grid_order":self.grid_order,
            "equal_size":self.equal_size,
        })

    def undo(self):
        if self.undo_stack:
            state = self.undo_stack.pop()
            self.rois = state["rois"]
            self.grid_rows,self.grid_cols = state["grid_rows"],state["grid_cols"]
            self.grid_order,self.equal_size = state["grid_order"],state["equal_size"]
            self.draw_first_requested = False
            self.geometry_restored.emit(state)
            self.selected_indices = {i for i in self.selected_indices if i < len(self.rois)}
            if not self.selected_indices and self.rois:
                self.selected_indices = {0}
            self.calibration_changed.emit()
            self.selection_changed.emit()
            self.update()

    def _calc_transform(self) -> tuple[float, int, int]:
        if self.raw_frame is None:
            return 1.0, 0, 0
        cw, ch = self.width(), self.height()
        fw, fh = self.raw_frame.shape[1], self.raw_frame.shape[0]
        scale = min(cw / max(1, fw), ch / max(1, fh))
        dw, dh = int(fw * scale), int(fh * scale)
        return scale, (cw - dw) // 2, (ch - dh) // 2

    def _get_handles(self, rect: QRectF) -> dict[str, QRectF]:
        s = self.HANDLE_SIZE
        hs = s / 2.0
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        return {
            "tl": QRectF(x - hs, y - hs, s, s),
            "t":  QRectF(x + w / 2 - hs, y - hs, s, s),
            "tr": QRectF(x + w - hs, y - hs, s, s),
            "r":  QRectF(x + w - hs, y + h / 2 - hs, s, s),
            "br": QRectF(x + w - hs, y + h - hs, s, s),
            "b":  QRectF(x + w / 2 - hs, y + h - hs, s, s),
            "bl": QRectF(x - hs, y + h - hs, s, s),
            "l":  QRectF(x - hs, y + h / 2 - hs, s, s),
        }

    def mousePressEvent(self, event):
        if self.raw_frame is None or event.button() != Qt.LeftButton:
            return

        scale, ox, oy = self._calc_transform()
        m_pos = event.pos()

        # Draw the first chamber before calibration.
        if not self.rois or self.draw_first_requested:
            self.drawing_first_roi = True
            self.band_start_pos = m_pos
            self.band_cur_pos = m_pos
            self.update()
            return

        # Shift-click changes selection without starting a resize operation.
        for idx in (() if event.modifiers() & Qt.ShiftModifier else sorted(self.selected_indices)):
            if idx < len(self.rois):
                roi = self.rois[idx]
                r = QRectF(ox + roi.x * scale, oy + roi.y * scale, roi.w * scale, roi.h * scale)
                handles = self._get_handles(r)
                for h_name, h_rect in handles.items():
                    if h_rect.contains(m_pos):
                        self._drag_anchor_index = idx
                        self._push_undo()
                        self.active_handle = h_name
                        self.is_dragging = True
                        self.drag_start_pos = m_pos
                        self.drag_snapshot = copy.deepcopy(self.rois)
                        return

        # Check whether the pointer is inside an ROI.
        hit_idx = None
        for idx, roi in enumerate(self.rois):
            r = QRectF(ox + roi.x * scale, oy + roi.y * scale, roi.w * scale, roi.h * scale)
            if r.contains(m_pos):
                hit_idx = idx
                break

        if hit_idx is not None:
            if event.modifiers() & Qt.ShiftModifier:
                if hit_idx in self.selected_indices:
                    self.selected_indices.remove(hit_idx)
                else:
                    self.selected_indices.add(hit_idx)
                self.selection_changed.emit()
                self.update()
                return
            elif hit_idx not in self.selected_indices:
                self.selected_indices = {hit_idx}
            # Clicking an already-selected chamber preserves the group for dragging.
            self.selection_changed.emit()
            if not self.selected_indices:
                self.update()
                return
            self._push_undo()
            self._drag_anchor_index = hit_idx
            self.active_handle = "move"
            self.is_dragging = True
            self.drag_start_pos = m_pos
            self.drag_snapshot = copy.deepcopy(self.rois)
            self.update()
            return

        # Drag empty space to select multiple ROIs.
        if not (event.modifiers() & Qt.ShiftModifier):
            self.selected_indices.clear()
        self.rubber_band_active = True
        self.band_start_pos = m_pos
        self.band_cur_pos = m_pos
        self.update()

    def mouseMoveEvent(self, event):
        scale, ox, oy = self._calc_transform()
        fw = self.raw_frame.shape[1] if self.raw_frame is not None else 1
        fh = self.raw_frame.shape[0] if self.raw_frame is not None else 1

        if not self.is_dragging and not self.rubber_band_active and not self.drawing_first_roi:
            cursor_set = False
            for idx in self.selected_indices:
                if idx < len(self.rois):
                    roi = self.rois[idx]
                    r = QRectF(ox + roi.x * scale, oy + roi.y * scale, roi.w * scale, roi.h * scale)
                    for h_name, h_rect in self._get_handles(r).items():
                        if h_rect.contains(event.pos()):
                            if h_name in ("tl", "br"): self.setCursor(Qt.SizeFDiagCursor)
                            elif h_name in ("tr", "bl"): self.setCursor(Qt.SizeBDiagCursor)
                            elif h_name in ("t", "b"): self.setCursor(Qt.SizeVerCursor)
                            elif h_name in ("l", "r"): self.setCursor(Qt.SizeHorCursor)
                            cursor_set = True
                            break
                    if cursor_set:
                        break
            if not cursor_set:
                self.setCursor(Qt.ArrowCursor)

        if self.drawing_first_roi or self.rubber_band_active:
            self.band_cur_pos = event.pos()
            self.update()
            return

        if self.is_dragging and self.drag_start_pos is not None:
            dx = (event.pos().x() - self.drag_start_pos.x()) / scale
            dy = (event.pos().y() - self.drag_start_pos.y()) / scale

            if self.active_handle == "move":
                min_dx = min(-self.drag_snapshot[i].x for i in self.selected_indices)
                max_dx = min(fw - (self.drag_snapshot[i].x + self.drag_snapshot[i].w) for i in self.selected_indices)
                min_dy = min(-self.drag_snapshot[i].y for i in self.selected_indices)
                max_dy = min(fh - (self.drag_snapshot[i].y + self.drag_snapshot[i].h) for i in self.selected_indices)

                clamped_dx = int(np.clip(dx, min_dx, max_dx))
                clamped_dy = int(np.clip(dy, min_dy, max_dy))

                for i in self.selected_indices:
                    self.rois[i].x = self.drag_snapshot[i].x + clamped_dx
                    self.rois[i].y = self.drag_snapshot[i].y + clamped_dy
            else:
                indices = set(range(len(self.rois))) if self.equal_size else set(self.selected_indices)
                base_ref = self.drag_snapshot[self._drag_anchor_index]
                target_w, target_h = base_ref.w, base_ref.h
                if "l" in self.active_handle:
                    target_w = int(np.clip(base_ref.w-dx,10,fw))
                elif "r" in self.active_handle:
                    target_w = int(np.clip(base_ref.w+dx,10,fw))
                if "t" in self.active_handle:
                    target_h = int(np.clip(base_ref.h-dy,10,fh))
                elif "b" in self.active_handle:
                    target_h = int(np.clip(base_ref.h+dy,10,fh))
                for index in indices:
                    base = self.drag_snapshot[index]
                    self.rois[index].x = int(np.clip(base.x+base.w/2-target_w/2,0,fw-target_w))
                    self.rois[index].y = int(np.clip(base.y+base.h/2-target_h/2,0,fh-target_h))
                    self.rois[index].w, self.rois[index].h = target_w,target_h

            self.update()

    def mouseReleaseEvent(self, event):
        scale, ox, oy = self._calc_transform()
        fw = self.raw_frame.shape[1] if self.raw_frame is not None else 1
        fh = self.raw_frame.shape[0] if self.raw_frame is not None else 1

        if self.drawing_first_roi and self.band_start_pos and self.band_cur_pos:
            self.drawing_first_roi = False
            p1, p2 = self.band_start_pos, self.band_cur_pos
            ix1 = int(np.clip((min(p1.x(), p2.x()) - ox) / scale, 0, fw))
            iy1 = int(np.clip((min(p1.y(), p2.y()) - oy) / scale, 0, fh))
            ix2 = int(np.clip((max(p1.x(), p2.x()) - ox) / scale, 0, fw))
            iy2 = int(np.clip((max(p1.y(), p2.y()) - oy) / scale, 0, fh))
            w, h = ix2 - ix1, iy2 - iy1
            if w >= 10 and h >= 10:
                self.draw_first_requested = False
                self.first_roi_drawn.emit((ix1, iy1, w, h))
            self.update()

        if self.rubber_band_active and self.band_start_pos and self.band_cur_pos:
            self.rubber_band_active = False
            band_rect = QRectF(self.band_start_pos, self.band_cur_pos).normalized()
            for idx, roi in enumerate(self.rois):
                r = QRectF(ox + roi.x * scale, oy + roi.y * scale, roi.w * scale, roi.h * scale)
                if band_rect.intersects(r):
                    self.selected_indices.add(idx)
            # An empty box selection remains empty instead of selecting the whole plate.
            self.update()

        self.selection_changed.emit()
        if self.is_dragging:
            self.is_dragging = False
            self.active_handle = None
            self.drag_start_pos = None
            self.calibration_changed.emit()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        painter.setRenderHint(QPainter.Antialiasing)

        if self.raw_frame is None:
            painter.setPen(QColor("#64748B"))
            font = QFont(QApplication.font())
            font.setPixelSize(13)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, "No Video Loaded\n(Please load video to calibrate)")
            return

        scale, ox, oy = self._calc_transform()
        fw, fh = self.raw_frame.shape[1], self.raw_frame.shape[0]
        dw, dh = int(fw * scale), int(fh * scale)

        if self._background_pixmap is not None:
            if self._scaled_background is None or self._scaled_size != (dw,dh):
                self._scaled_background = self._background_pixmap.scaled(dw,dh,Qt.KeepAspectRatio,Qt.SmoothTransformation)
                self._scaled_size = (dw,dh)
            painter.drawPixmap(ox,oy,self._scaled_background)

        pen_default = QPen(QColor("#00E5FF"), 1.2, Qt.SolidLine)
        pen_selected = QPen(QColor("#F59E0B"), 2.0, Qt.SolidLine)
        pen_txt = QPen(QColor("#FFFFFF"))
        font_cid = QFont(QApplication.font())
        font_cid.setPixelSize(12)
        font_cid.setWeight(QFont.Normal)

        for idx, roi in enumerate(self.rois):
            rx = ox + int(roi.x * scale)
            ry = oy + int(roi.y * scale)
            rw = int(roi.w * scale)
            rh = int(roi.h * scale)
            rect = QRectF(rx, ry, rw, rh)

            is_sel = idx in self.selected_indices
            painter.setPen(pen_selected if is_sel else pen_default)
            if is_sel:
                painter.setBrush(QColor(245, 158, 11, 45))
            else:
                painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            painter.setPen(pen_txt)
            painter.setFont(font_cid)
            painter.drawText(rx + 4, ry + 13, f"CH{roi.chamber_id}")

            if is_sel:
                handles = self._get_handles(rect)
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
                painter.setBrush(QColor("#F59E0B"))
                for h_rect in handles.values():
                    painter.drawRect(h_rect)

        if self.show_trail and self._trail_pixmap is not None:
            painter.drawPixmap(QRectF(ox, oy, dw, dh), self._trail_pixmap,
                               QRectF(self._trail_pixmap.rect()))
        painter.setPen(QPen(QColor("#22C55E"),2))
        painter.setBrush(QColor("#22C55E"))
        for x,y in self.centroids:
            painter.drawEllipse(QPointF(ox+x*scale,oy+y*scale),4,4)

        if self.drawing_first_roi and self.band_start_pos and self.band_cur_pos:
            painter.setPen(QPen(QColor("#10B981"), 1.5, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(self.band_start_pos, self.band_cur_pos).normalized())

        if self.rubber_band_active and self.band_start_pos and self.band_cur_pos:
            painter.setPen(QPen(QColor("#38BDF8"), 1.2, Qt.DashLine))
            painter.setBrush(QColor(56, 189, 248, 30))
            painter.drawRect(QRectF(self.band_start_pos, self.band_cur_pos).normalized())


    def set_display_frame(self, frame):
        if frame is None:
            self._background_pixmap = None
        else:
            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) if frame.ndim==3 else frame
            image = np.ascontiguousarray(image)
            fmt = QImage.Format_RGB888 if image.ndim==3 else QImage.Format_Grayscale8
            self._background_pixmap = QPixmap.fromImage(QImage(image.data,image.shape[1],image.shape[0],image.strides[0],fmt).copy())
        self._scaled_background = None
        self._scaled_size = None
        self.update()


    def normalise_equal_size(self, push_undo=True):
        if not self.rois or self.raw_frame is None:
            return
        if push_undo:
            self._push_undo()
        height,width = self.raw_frame.shape[:2]
        w,h = min(self.rois[0].w,width),min(self.rois[0].h,height)
        for roi in self.rois:
            roi.x = int(np.clip(roi.x+roi.w/2-w/2,0,width-w))
            roi.y = int(np.clip(roi.y+roi.h/2-h/2,0,height-h))
            roi.w,roi.h = w,h
        self.calibration_changed.emit()
        self.selection_changed.emit()
        self.update()
