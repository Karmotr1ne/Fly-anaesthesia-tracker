'File and folder selection through drag and drop or a file dialog.'
from __future__ import annotations

from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QFileDialog, QSizePolicy


class DragDropUploadArea(QFrame):
    'Select files and folders through drag and drop or clicking.'

    paths_dropped = pyqtSignal(list)  # Emit parsed paths as list[Path].

    def __init__(self, title: str = "Drop files or folders here", hint: str = "Supports trajectory CSVs, telemetry CSVs, and session folders", parent=None, folder_selection=False):
        super().__init__(parent)
        self.folder_selection = folder_selection
        self.setObjectName("dropUploadArea")
        self.setAcceptDrops(True)
        self._init_ui(title, hint)

    def _init_ui(self, title: str, hint: str):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame#dropUploadArea {
                border: 1px dashed #9CA3AF;
                border-radius: 4px;
                background-color: #F9FAFB;
                padding: 8px;
            }
            QFrame#dropUploadArea:hover {
                border-color: #1D4ED8;
                background-color: #EFF6FF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet(" font-weight: 600; color: #1F2937;")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_hint = QLabel(hint)
        self.lbl_hint.setStyleSheet(" color: #6B7280;")
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.lbl_title)
        self.lbl_hint.hide()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QFrame#dropUploadArea {
                    border: 1px dashed #1D4ED8;
                    border-radius: 4px;
                    background-color: #DBEAFE;
                    padding: 8px;
                }
            """)

    def dragLeaveEvent(self, event):
        self._reset_style()
        event.accept()

    def dropEvent(self, event):
        self._reset_style()
        urls = event.mimeData().urls()
        paths: list[Path] = [Path(url.toLocalFile()).resolve() for url in urls if url.isLocalFile()]
        if paths:
            self.paths_dropped.emit(paths)
        event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.folder_selection:
                folder = QFileDialog.getExistingDirectory(self, 'Import Experiment Session Folder', '')
                if folder:
                    self.paths_dropped.emit([Path(folder).resolve()])
                super().mousePressEvent(event)
                return
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Trajectory or Telemetry CSV Files",
                "",
                "CSV Files (*.csv);;All Files (*.*)"
            )
            if files:
                self.paths_dropped.emit([Path(f).resolve() for f in files])
        super().mousePressEvent(event)

    def _reset_style(self):
        self.setStyleSheet("""
            QFrame#dropUploadArea {
                border: 1px dashed #9CA3AF;
                border-radius: 4px;
                background-color: #F9FAFB;
                padding: 8px;
            }
            QFrame#dropUploadArea:hover {
                border-color: #1D4ED8;
                background-color: #EFF6FF;
            }
        """)



class FolderImportArea(DragDropUploadArea):
    """One folder-only entry for clicking and dropping."""
    folders_selected = pyqtSignal(list)
    rejected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Import Video Folder", "Click or drop a video folder", parent)
        self.lbl_title.setProperty("role", "section")
        self.lbl_hint.setProperty("role", "secondary")
        self.lbl_title.setWordWrap(True)
        self.lbl_hint.setWordWrap(True)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout().setContentsMargins(10, 8, 10, 8)
        self.layout().setSpacing(4)
        self.lbl_title.setStyleSheet("")
        self.lbl_hint.setStyleSheet("")
        self._reset_style()

    def _reset_style(self):
        self.setStyleSheet("""
            QFrame#dropUploadArea {
                background: #F8FAFC;
                border: 1px dashed #E2E8F0;
                border-radius: 6px;
                padding: 0;
            }
            QFrame#dropUploadArea:hover {
                background: #EFF6FF;
                border-color: #1D4ED8;
            }
            QFrame#dropUploadArea:disabled {
                background: #F1F5F9;
                border-color: #E2E8F0;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            folder = QFileDialog.getExistingDirectory(self, "Import Video Folder", "")
            if folder:
                self.folders_selected.emit([Path(folder)])
        QFrame.mousePressEvent(self, event)

    def dropEvent(self, event):
        self._reset_style()
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        folders = [path for path in paths if path.is_dir()]
        if len(folders) != len(paths) or not folders:
            self.rejected.emit("Drop a folder to import videos.")
        if folders:
            self.folders_selected.emit(folders)
            event.acceptProposedAction()
        else:
            event.ignore()
