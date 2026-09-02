# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification File
==============================
Builds standalone release binary for Drosophila Behavior & Anesthesia Tracking Suite.
Compatible with Windows, Linux, and macOS execution.
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Dynamically resolve directory where this spec file resides
spec_dir = os.path.abspath(SPECPATH) if 'SPECPATH' in globals() else os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(spec_dir)

# Ensure both python_suite and parent are discoverable in pathex
search_paths = [
    spec_dir,
    os.path.join(spec_dir, "drosophila_suite"),
    parent_dir,
]

entry_script = os.path.join(spec_dir, "main.py")
if not os.path.exists(entry_script):
    entry_script = "main.py"

block_cipher = None

# Comprehensive hidden imports for scientific stack and UI
hidden_imports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'cv2',
    'numpy',
    'pandas',
    'scipy',
    'scipy.signal',
    'scipy.ndimage',
    'scipy.ndimage._filters_core',
    'matplotlib',
    'matplotlib.backends.backend_agg',
    'matplotlib.pyplot',
    'yaml',
    'drosophila_suite',
    'drosophila_suite.models',
    'drosophila_suite.tracker',
    'drosophila_suite.cleaner',
    'drosophila_suite.stationary_engine',
    'drosophila_suite.anesthesia',
    'drosophila_suite.visualizer',
    'drosophila_suite.pipeline',
    'drosophila_suite.gui_app',
    'drosophila_suite.cli',
    'drosophila_suite.main',
]

try:
    hidden_imports += collect_submodules('scipy.signal')
    hidden_imports += collect_submodules('scipy.ndimage')
    hidden_imports += collect_submodules('matplotlib')
except Exception:
    pass

datas = []

a = Analysis(
    [entry_script],
    pathex=search_paths,
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'IPython', 'notebook', 'pytest', 'sphinx'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DrosophilaAnesthesiaTracker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
