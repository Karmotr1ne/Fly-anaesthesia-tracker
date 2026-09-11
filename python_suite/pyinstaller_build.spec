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

# 1. 动态解析当前 SPEC 所在根路径与父目录
spec_dir = os.path.abspath(SPECPATH) if 'SPECPATH' in globals() else os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(spec_dir)

# 确保 python_suite、drosophila_suite 及上级目录均在搜索路径中
search_paths = [
    spec_dir,
    os.path.join(spec_dir, "drosophila_suite"),
    parent_dir,
]

entry_script = os.path.join(spec_dir, "main.py")
if not os.path.exists(entry_script):
    entry_script = "main.py"

# 2. 隐式依赖与科学计算子模块收集
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

# 3. 收集必要的静态数据资源（Matplotlib 默认字体/配置等）
datas = []
try:
    datas += collect_data_files('matplotlib')
except Exception:
    pass

# 若有本地图标或模板文件需要打入包内，取消注释并在此添加：
# if os.path.exists(os.path.join(spec_dir, "roi_template.json")):
#     datas.append((os.path.join(spec_dir, "roi_template.json"), "."))

# 4. 分析代码依赖拓扑
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
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 5. 构建单文件可执行对象
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
    upx=False,  # 避免 UPX 压缩破坏 PyQt6/OpenCV 动态库导致闪退
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # 若排查闪退错误可临时改为 True
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon=os.path.join(spec_dir, 'icon.ico') if sys.platform.startswith('win') else None,
)

# 6. macOS 专属 App Bundle 封装（确保在 Mac 上生成 .app）
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='DrosophilaAnesthesiaTracker.app',
        icon=None,  # 若有 Mac 图标可指定: os.path.join(spec_dir, 'icon.icns')
        bundle_identifier='com.drosophila.tracker',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSBackgroundOnly': 'False',
        },
    )