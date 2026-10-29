"""Shared compact workbench typography, surfaces and control sizing."""
SCIENTIFIC_QSS = """
QWidget { font-family: "IBM Plex Sans", "Segoe UI"; font-size: 12px; font-weight: 400; color: #0F172A; }
QMainWindow, QDialog { background: #F8FAFC; }
QWidget[workbenchPage="true"] { background: #FFFFFF; }
QWidget#evaluationSummary { background: #F0F6F5; }
QLabel { border: none; background: transparent; }
QLabel[role="section"], QToolButton[role="sectionToggle"] { font-size: 13px; font-weight: 600; color: #000000; }
QLabel[role="secondary"] { font-size: 12px; font-weight: 400; color: #64748B; }
QGroupBox { background: #FFFFFF; border: none; margin-top: 18px; padding: 12px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #475569; font-size: 13px; }
QFrame[role="panel"] { background: #FFFFFF; border: none; }
QTabWidget::pane { background: #FFFFFF; border: none; }
QTabBar::tab { background: #F8FAFC; border: none; padding: 8px 14px; color: #64748B; font-size: 12px; font-weight: 400; }
QTabBar::tab:selected { background: #FFFFFF; color: #1D4ED8; border-bottom: 2px solid #1D4ED8; font-weight: 400; }
QPushButton, QToolButton { background: #FFFFFF; color: #0F172A; border: 1px solid #E2E8F0; border-radius: 4px; padding: 0 10px; min-height: 24px; font-size: 12px; font-weight: 400; }
QPushButton:hover, QToolButton:hover { background: #EFF6FF; border-color: #BFDBFE; }
QPushButton:disabled, QToolButton:disabled { background: #F1F5F9; color: #94A3B8; border-color: #E2E8F0; }
QPushButton[role="primary"] { background: #1D4ED8; color: #FFFFFF; border-color: #1D4ED8; font-weight: 600; }
QPushButton[role="danger"] { background: #FEF2F2; color: #B91C1C; border-color: #FECACA; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; padding: 0 6px; min-height: 24px; font-size: 12px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #1D4ED8; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { background: #F1F5F9; color: #94A3B8; }
QSpinBox, QDoubleSpinBox, QTableWidget, QTableView { font-family: "Consolas", "Courier New"; font-size: 12px; font-weight: 400; }
QPlainTextEdit, QTextEdit { font-family: "Consolas", "Courier New"; font-size: 11px; font-weight: 400; background: #FAFCFE; border: 1px solid #E2E8F0; border-radius: 4px; }
QListWidget, QTableWidget, QTableView { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px; }
QHeaderView::section { font-family: "IBM Plex Sans", "Segoe UI"; font-size: 12px; font-weight: 600; color: #64748B; background: #F8FAFC; padding: 5px; border: none; border-bottom: 1px solid #E2E8F0; }
QListWidget::item { padding: 5px 6px; }
QTableWidget::item:selected, QListWidget::item:selected { background: #EFF6FF; color: #1D4ED8; }
QCheckBox, QRadioButton { font-size: 12px; font-weight: 400; spacing: 6px; }
QProgressBar { background: #E2E8F0; border: none; border-radius: 3px; min-height: 6px; text-align: center; font-family: "Consolas"; font-size: 11px; }
QProgressBar::chunk { background: #1D4ED8; border-radius: 3px; }
QScrollArea { border: none; background: transparent; }
QSplitter::handle { background: #E2E8F0; width: 4px; }
QWidget#trackingSidebarContent, QWidget#trackingSidebarViewport { background: #FFFFFF; }
QWidget#trackingTab QToolButton[role="sectionToggle"] { border: none; background: transparent; color: #000000; text-align: left; font-weight: 600; padding: 0; }
/* Protocol uses compact titled sections with a single horizontal divider. */
QWidget#protocolTab QGroupBox { margin-top: 0; padding: 0; border: none; background: #FFFFFF; }
QWidget#protocolTab QGroupBox::title { subcontrol-origin: margin; left: 0; padding: 0 8px 0 0; background: #FFFFFF; color: #475569; }
QFrame#protocolDivider { background: #E2E8F0; border: none; }
QFrame#protocolMetrics { background: transparent; border: none; border-radius: 4px; }
QFrame#protocolMetrics QLabel { color: #0F172A; font-weight: 600; background: transparent; border: none; }

QPushButton[role="rowAction"] { background: #FFFFFF; color: #0F172A; border: 1px solid #E2E8F0; font-weight: 400; }
QPushButton[role="rowAction"]:hover { background: #F8FAFC; color: #0F172A; border-color: #CBD5E1; }

"""
# Historical import retained; there is no manual 150% scaling.
SCALED_150_SCIENTIFIC_QSS = SCIENTIFIC_QSS


def apply_typography(app):
    """Use bundled IBM Plex Sans directly without enumerating system fonts."""
    from pathlib import Path
    from PyQt5.QtGui import QFont, QFontDatabase

    folder = Path(__file__).resolve().parent / "fonts"
    for name in ("IBMPlexSans-Regular.ttf", "IBMPlexSans-SemiBold.ttf"):
        path = folder / name
        if path.is_file():
            QFontDatabase.addApplicationFont(str(path))
    font = QFont("IBM Plex Sans")
    font.setPixelSize(12)
    font.setWeight(QFont.Normal)
    app.setFont(font)


def style_page(page):
    """Apply shared sizing once after constructing a page; never add controls."""
    from PyQt5.QtWidgets import (QPushButton, QToolButton, QLineEdit, QSpinBox,
                                QDoubleSpinBox, QComboBox, QTableWidget)

    import re
    page.setProperty("workbenchPage", True)
    for kind in (QPushButton, QToolButton, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox):
        for control in page.findChildren(kind):
            if isinstance(control, QLineEdit) and (control.objectName() == "qt_spinbox_lineedit"
                    or isinstance(control.parent(), (QSpinBox, QDoubleSpinBox, QComboBox))):
                continue
            css = control.styleSheet()
            # Old inline dimensions must not override the common control size.
            css = re.sub(r"(?<![\w-])(?:min-height|max-height|height)\s*:\s*[^;{}]+;", "", css)
            if control.property("role") == "sectionToggle":
                padding = "padding: 0;"
            else:
                padding = "padding: 0 10px;" if isinstance(control, (QPushButton, QToolButton)) else "padding: 0 6px;"
            css = re.sub(r"padding\s*:\s*[^;{}]+;", padding, css)
            if control.property("role") == "sectionToggle" and "padding:" not in css:
                css += padding
            control.setStyleSheet(css)
            control.setFixedHeight(26)
            if isinstance(control, (QPushButton, QToolButton)) and control.text().strip():
                control.setMinimumWidth(72)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setMinimumWidth(96)
    for table in page.findChildren(QTableWidget):
        table.setStyleSheet("")
        table.verticalHeader().setDefaultSectionSize(28)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
    page.style().unpolish(page)
    page.style().polish(page)


def flowvision_state(online, details):
    """Map connection events to the same three-colour status vocabulary."""
    if online:
        return "Ready"
    text = details.casefold()
    if any(word in text for word in ("connecting", "handshaking", "reconnecting")):
        return "Connecting"
    if "disabled" in text:
        return "Disabled"
    return "Offline"


def set_device_status(label, device, state, details=""):
    """Paint one compact, bordered device status badge consistently everywhere."""
    if state in ("Ready", "Recording"):
        background, border, foreground = "#F0FDF4", "#86EFAC", "#166534"
    elif state == "Connecting":
        background, border, foreground = "#FFFBEB", "#FCD34D", "#92400E"
    else:
        background, border, foreground = "#FEF2F2", "#FCA5A5", "#991B1B"
    from PyQt5.QtCore import Qt
    label.setAlignment(Qt.AlignCenter)
    label.setText(f"{device}: {state}")
    label.setToolTip(details)
    label.setMinimumHeight(26)
    label.setMinimumWidth(152)
    label.setStyleSheet(
        f"QLabel {{ background-color: {background}; color: {foreground}; "
        f"border: 1px solid {border}; border-radius: 4px; "
        "padding: 2px 9px; font-weight: 600; font-size: 12px; }"
    )
