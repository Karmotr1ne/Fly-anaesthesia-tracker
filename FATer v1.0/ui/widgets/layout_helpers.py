from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QSizePolicy

def scroll_panel(widget, minimum_width=0):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setMinimumWidth(minimum_width)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    area.setWidget(widget)
    return area


def install_window_centering(app):
    """Centre initial main windows and Qt dialogs on their owner's screen."""
    import weakref
    from PyQt5 import sip
    from PyQt5.QtCore import QObject, QEvent, QTimer
    from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

    class WindowPositionFilter(QObject):
        def eventFilter(self, widget, event):
            if event.type() == QEvent.Show and isinstance(widget, (QDialog, QMainWindow)) and widget.isWindow():
                first_main_show = isinstance(widget, QMainWindow) and not widget.property("initiallyCentred")
                if isinstance(widget, QDialog) or first_main_show:
                    if first_main_show:
                        widget.setProperty("initiallyCentred", True)
                    reference = weakref.ref(widget)
                    QTimer.singleShot(0, lambda ref=reference: self.centre(ref))
            return False

        @staticmethod
        def centre(reference):
            widget = reference()
            if widget is None or sip.isdeleted(widget) or not widget.isVisible():
                return
            parent = widget.parentWidget()
            parent = parent.window() if parent else None
            screen = (parent.windowHandle().screen() if parent and parent.windowHandle()
                      else widget.windowHandle().screen() if widget.windowHandle()
                      else QApplication.primaryScreen())
            if screen is None:
                return
            available = screen.availableGeometry()
            centre = parent.frameGeometry().center() if parent and parent.isVisible() else available.center()
            frame = widget.frameGeometry()
            frame.moveCenter(centre)
            x = max(available.left(), min(frame.left(), available.right() - frame.width() + 1))
            y = max(available.top(), min(frame.top(), available.bottom() - frame.height() + 1))
            widget.move(x, y)

    app._window_position_filter = WindowPositionFilter(app)
    app.installEventFilter(app._window_position_filter)
