"""ui/widgets/video_viewport.py"""
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QWidget

class AspectRatioVideoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setStyleSheet("background-color: #0F172A; border-radius: 9px; border: 2px solid #CBD5E1;")
        self._pixmap = None

    def update_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#94A3B8"))
            f = painter.font()
            f.setFamily("Segoe UI")
            f.setPointSize(14)
            painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignCenter, "Waiting for camera...")
