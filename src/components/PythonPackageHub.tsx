import React, { useState } from 'react';
import {
  FileCode,
  Package,
  Terminal,
  Download,
  Copy,
  Check,
  ExternalLink,
  Cpu,
  Layers,
  FolderArchive,
  Sparkles,
  Zap
} from 'lucide-react';

export const PythonPackageHub: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<string>('anesthesia.py');
  const [copied, setCopied] = useState<boolean>(false);

  const fileContents: Record<string, { desc: string; code: string; module: string }> = {
    'anesthesia.py': {
      desc: 'Module 4: Anesthesia & Sedation Kinetics Analyzer (Knockdown sliding window max filter W=120s, induction latency, and baseline locomotion speed).',
      module: 'Module 4: Anesthesia Kinetics',
      code: `"""
Module 4: Anesthesia & Sedation Kinetics Analyzer (v0.1)
=========================================================
Anesthesia induction / sedation knockdown kinetics (sliding window W=120s, theta=0.01).
Quantifies latency to sustained sedation and baseline locomotion dynamics.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from .models import AnesthesiaSummary
from .stationary_engine import StationaryDetectionEngine


class AnesthesiaAnalyzer:
    def __init__(
        self,
        bin_size_sec: float = 5.0,
        window_duration_sec: float = 120.0,
        window_bins: Optional[int] = None,
        activity_threshold: float = 0.01,
        fps: float = 30.0,
    ):
        self.bin_size_sec = bin_size_sec
        self.window_duration_sec = window_duration_sec
        self.window_bins = window_bins or max(1, int(round(window_duration_sec / bin_size_sec)))
        self.activity_threshold = activity_threshold
        self.fps = fps

    def evaluate_induction(self, cleaned_df: pd.DataFrame, fps: Optional[float] = None) -> pd.DataFrame:
        effective_fps = fps or self.fps
        frames_per_bin = max(1, int(round(self.bin_size_sec * effective_fps)))
        results = []

        for cid, group in cleaned_df.groupby("chamber_id"):
            grp = group.sort_values("frame").reset_index(drop=True)
            speeds = grp["speed"].fillna(0.0).to_numpy()
            total_frames = len(speeds)
            if total_frames == 0:
                continue

            num_bins = int(np.ceil(total_frames / frames_per_bin))
            binned_activity = np.zeros(num_bins, dtype=np.float64)
            time_axis_sec = np.arange(num_bins) * self.bin_size_sec

            for b in range(num_bins):
                start_f = b * frames_per_bin
                end_f = min(total_frames, (b + 1) * frames_per_bin)
                binned_activity[b] = np.mean(speeds[start_f:end_f])

            baseline_speed = float(np.mean(binned_activity[:max(1, min(12, int(num_bins * 0.1)))]))
            stillness_bins_count = int(np.sum(binned_activity < self.activity_threshold))

            stationary_mask = StationaryDetectionEngine.sliding_window_max_filter(
                binned_activity, self.window_bins, self.activity_threshold
            )

            true_indices = np.where(stationary_mask)[0]
            if len(true_indices) > 0:
                first_bin = true_indices[0]
                induction_time = round(float(time_axis_sec[first_bin]), 2)
                is_sedated = True
                pre_activity = float(np.mean(binned_activity[:max(1, first_bin)])) if first_bin > 0 else float(binned_activity[0])
            else:
                induction_time = None
                is_sedated = False
                pre_activity = float(np.mean(binned_activity))

            results.append({
                "chamber_id": int(cid),
                "induction_time_sec": induction_time,
                "is_sedated": is_sedated,
                "baseline_speed": round(baseline_speed, 2),
                "pre_sedation_activity": round(pre_activity, 2),
                "stillness_bins_count": stillness_bins_count,
            })

        return pd.DataFrame(results)`
    },
    'pipeline.py': {
      desc: 'Master Pipeline Orchestrator connecting Modules 1 through 4 with batch processing, video overlay, and CSV reporting.',
      module: 'Core Pipeline',
      code: `"""
Drosophila Anesthesia Tracker Pipeline (v0.1)
==============================================
Unified orchestrator for vision tracking, kinematic cleaning,
anesthesia kinetics, and scientific visualization.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from .models import PipelineConfig
from .tracker import FlyVisionTracker
from .cleaner import KinematicCleaner
from .anesthesia import AnesthesiaAnalyzer
from .visualizer import ScientificVisualizer


class DrosophilaBehaviorPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.cleaner = KinematicCleaner(
            fps=self.config.fps,
            max_speed_px=self.config.max_speed_px_per_frame,
            body_len_thresh=self.config.body_length_thresh,
            body_len_px=self.config.body_length_px,
            occlusion_disp_thresh=self.config.occlusion_disp_thresh,
            occlusion_var_thresh=self.config.occlusion_var_thresh,
            savgol_window=self.config.savgol_window,
            savgol_poly=self.config.savgol_poly,
        )
        self.anesthesia_analyzer = AnesthesiaAnalyzer(
            bin_size_sec=self.config.anesthesia_bin_size_sec,
            window_bins=self.config.anesthesia_window_bins,
            activity_threshold=self.config.anesthesia_activity_threshold,
            fps=self.config.fps,
        )
        self.visualizer = ScientificVisualizer(fps=self.config.fps)

    def process_file_pair(
        self,
        csv_path: Optional[str] = None,
        video_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        base_name: Optional[str] = None,
        save_cleaned_csv: bool = True,
        generate_plots: bool = True,
        render_video_overlay: bool = False,
        chamber_rois: Optional[List[Tuple[int, int, int, int]]] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Processes a single experimental session from CSV and/or Video.
        """
        ref_path = csv_path or video_path
        if not ref_path:
            raise ValueError("Must provide csv_path or video_path.")

        target_dir = output_dir or os.path.dirname(os.path.abspath(ref_path))
        os.makedirs(target_dir, exist_ok=True)

        if not base_name:
            base_name = os.path.splitext(os.path.basename(ref_path))[0]
            for suffix in ["_tracked", "_raw", "_cleaned"]:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]

        out_prefix = os.path.join(target_dir, base_name)

        # 1. Obtain raw trajectory
        if csv_path and os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)
        elif video_path and os.path.exists(video_path):
            tracker = FlyVisionTracker(chamber_rois=chamber_rois or [])
            raw_df = tracker.track_video(video_path, progress_callback=progress_callback)
        else:
            raise FileNotFoundError(f"Input file not found: {ref_path}")

        # 2. Kinematic Cleaning & Occlusion Clamping
        cleaned_df = self.cleaner.clean_trajectory(raw_df)
        cleaned_csv_path = f"{out_prefix}_cleaned.csv"
        if save_cleaned_csv:
            cleaned_df.to_csv(cleaned_csv_path, index=False)

        # 3. Anesthesia Induction Kinetics
        anesthesia_df = self.anesthesia_analyzer.evaluate_induction(cleaned_df, fps=self.config.fps)

        # 4. Consolidated Summary CSV
        summary_csv_path = f"{out_prefix}_results_summary.csv"
        anesthesia_df.to_csv(summary_csv_path, index=False)

        # 5. Scientific Plotting
        plot_paths = {}
        if generate_plots:
            act_pos_plot = f"{out_prefix}_activity_position.png"
            kymo_plot = f"{out_prefix}_kymograph_norm.png"
            self.visualizer.plot_activity_position_overview(cleaned_df, act_pos_plot, fps=self.config.fps)
            self.visualizer.plot_kymograph_hexbin(cleaned_df, kymo_plot, fps=self.config.fps)
            plot_paths["activity_position"] = act_pos_plot
            plot_paths["kymograph"] = kymo_plot

        return {
            "base_name": base_name,
            "cleaned_df": cleaned_df,
            "anesthesia_df": anesthesia_df,
            "summary_df": anesthesia_df,
            "cleaned_csv_path": cleaned_csv_path,
            "summary_csv_path": summary_csv_path,
            "plot_paths": plot_paths,
        }`
    },
    'stationary_engine.py': {
      desc: 'Module 3: Core Stationary Detection Engine (Vectorized sliding window maximum filter max(Activity[t:t+W]) < theta).',
      module: 'Module 3: Core Engine',
      code: `"""
Module 3: Core Stationary Detection Engine (v0.1)
=================================================
Vectorized sliding window extreme filter operator.
"""

from typing import Union, List
import numpy as np
import pandas as pd


class StationaryDetectionEngine:
    @staticmethod
    def sliding_window_max_filter(
        activity_series: Union[np.ndarray, List[float], pd.Series],
        window_size: int,
        threshold: float
    ) -> np.ndarray:
        arr = np.asarray(activity_series, dtype=np.float64)
        n = len(arr)
        if n == 0 or window_size <= 0:
            return np.zeros(0, dtype=bool)

        if window_size > n:
            all_below = (np.nanmax(arr) < threshold) if not np.all(np.isnan(arr)) else False
            res = np.zeros(n, dtype=bool)
            if all_below:
                res[0] = True
            return res

        is_stationary = np.zeros(n, dtype=bool)
        try:
            arr_c = np.ascontiguousarray(arr, dtype=np.float64)
            shape = (n - window_size + 1, window_size)
            strides = (arr_c.strides[0], arr_c.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr_c, shape=shape, strides=strides)
            max_vals = np.nanmax(windows, axis=1)
            is_stationary[: n - window_size + 1] = max_vals < threshold
        except Exception:
            for t in range(n - window_size + 1):
                if np.nanmax(arr[t : t + window_size]) < threshold:
                    is_stationary[t] = True

        return is_stationary`
    },
    'gui_app.py': {
      desc: 'Desktop GUI Workstation (PyQt6) with async QThreadPool, multi-chamber grid calibration, and batch execution.',
      module: 'Desktop GUI',
      code: `"""
Desktop Application (PyQt6) — v0.1 Release
==========================================
Asynchronous QThreadPool-backed desktop workstation for Drosophila Anesthesia Analysis.
"""

import os
import sys
from PyQt6.QtCore import Qt, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QGroupBox, QProgressBar,
    QListWidget, QMessageBox, QFrame, QFileDialog, QSpinBox
)
from drosophila_suite.pipeline import DrosophilaBehaviorPipeline


class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)


def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())`
    },
    'setup.py': {
      desc: 'Python Setuptools package specification for v0.1 (drosophila-anesthesia-tracker).',
      module: 'Setup Manifest',
      code: `from setuptools import setup, find_packages

setup(
    name="drosophila-anesthesia-tracker",
    version="0.1.0",
    description="Integrated Drosophila vision tracking, kinematic cleaning, and anesthesia kinetics testing workbench.",
    author="Drosophila Behavioral Phenotyping Lab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.6.0",
        "PyQt6>=6.4.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "anesthesia-gui = drosophila_suite.gui_app:run_gui",
            "anesthesia-cli = drosophila_suite.cli:main",
        ]
    },
)`
    },
    'pyinstaller_build.spec': {
      desc: 'PyInstaller Specification File for v0.1 release binary packaging.',
      module: 'PyInstaller Spec',
      code: `# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden_imports = [
    'PyQt6', 'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'cv2', 'numpy', 'pandas', 'scipy', 'scipy.signal', 'scipy.ndimage',
    'matplotlib', 'matplotlib.backends.backend_agg', 'matplotlib.pyplot',
    'yaml', 'drosophila_suite', 'drosophila_suite.models',
    'drosophila_suite.tracker', 'drosophila_suite.cleaner',
    'drosophila_suite.stationary_engine', 'drosophila_suite.anesthesia',
    'drosophila_suite.visualizer', 'drosophila_suite.pipeline',
    'drosophila_suite.gui_app', 'drosophila_suite.cli',
]
hidden_imports += collect_submodules('scipy.signal')
hidden_imports += collect_submodules('matplotlib')

a = Analysis(
    ['drosophila_suite/gui_app.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    excludes=['tkinter', 'IPython', 'notebook', 'pytest'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DrosophilaAnesthesiaTracker',
    debug=False,
    console=False,
    upx=True,
)`
    }
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(fileContents[selectedFile]?.code || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadFile = () => {
    const content = fileContents[selectedFile]?.code || '';
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = selectedFile;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Release Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 p-6 rounded-xl text-white border border-slate-700 shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="p-2 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
              <Package className="w-5 h-5" />
            </span>
            <h2 className="text-xl font-bold tracking-tight">
              Drosophila Anesthesia Tracker — Python Package v0.1
            </h2>
          </div>
          <p className="text-xs text-slate-300 mt-2 max-w-2xl leading-relaxed">
            Modular, strictly-typed Python package with dedicated stationary detection engine and volatile anesthetic kinetics.
            Packaged by functional modules with full PyInstaller release support.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={handleDownloadFile}
            className="flex items-center space-x-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-bold shadow-sm transition cursor-pointer"
          >
            <Download className="w-4 h-4" />
            <span>Download {selectedFile}</span>
          </button>
        </div>
      </div>

      {/* PyInstaller Quick Packaging Instructions */}
      <div className="bg-slate-900 border border-slate-800 p-5 rounded-xl text-slate-100 space-y-3">
        <h3 className="text-sm font-bold text-slate-200 flex items-center space-x-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span>Quick PyInstaller Build &amp; Release Commands (v0.1)</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] mb-1 font-sans font-semibold">1. Install Dependencies:</span>
            <span className="text-emerald-400">pip install -r requirements.txt</span>
          </div>

          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
            <span className="text-slate-400 block text-[11px] mb-1 font-sans font-semibold">2. Build Standalone Executable:</span>
            <span className="text-amber-300">python build_release.py</span>
            <span className="text-slate-500 block text-[10px] font-sans mt-1">(or: pyinstaller --clean -y pyinstaller_build.spec)</span>
          </div>
        </div>
      </div>

      {/* Code Explorer Interface */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* File Tree Sidebar */}
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2 flex items-center space-x-1.5">
            <FolderArchive className="w-4 h-4 text-slate-600" />
            <span>Package Files (v0.1)</span>
          </h4>

          <div className="space-y-1">
            {Object.keys(fileContents).map((fileName) => {
              const info = fileContents[fileName];
              const isSelected = selectedFile === fileName;
              return (
                <button
                  key={fileName}
                  onClick={() => setSelectedFile(fileName)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition flex items-center justify-between cursor-pointer ${
                    isSelected
                      ? 'bg-slate-900 text-white font-semibold shadow-sm'
                      : 'text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <span className="flex items-center space-x-2 truncate">
                    <FileCode className={`w-3.5 h-3.5 ${isSelected ? 'text-emerald-400' : 'text-slate-400'}`} />
                    <span className="truncate">{fileName}</span>
                  </span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${isSelected ? 'bg-slate-800 text-slate-300' : 'bg-slate-100 text-slate-500'}`}>
                    {info.module.split(':')[0]}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Code Content Viewer */}
        <div className="lg:col-span-3 bg-slate-950 rounded-xl border border-slate-800 shadow-md flex flex-col overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 bg-slate-900 border-b border-slate-800 text-xs">
            <div className="flex items-center space-x-2">
              <FileCode className="w-4 h-4 text-emerald-400" />
              <span className="font-mono font-bold text-slate-100">{selectedFile}</span>
              <span className="text-[11px] text-slate-400">({fileContents[selectedFile]?.module})</span>
            </div>

            <div className="flex items-center space-x-2">
              <button
                onClick={handleCopyCode}
                className="flex items-center space-x-1 px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 text-xs font-medium transition cursor-pointer"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Code'}</span>
              </button>

              <button
                onClick={handleDownloadFile}
                className="flex items-center space-x-1 px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-xs font-bold transition shadow-sm cursor-pointer"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Save</span>
              </button>
            </div>
          </div>

          <div className="p-3 bg-slate-900/60 border-b border-slate-800 text-xs text-slate-300">
            {fileContents[selectedFile]?.desc}
          </div>

          {/* Syntax Highlighted Box */}
          <pre className="p-4 text-xs font-mono text-slate-200 overflow-x-auto max-h-[550px] leading-relaxed scrollbar-thin scrollbar-thumb-slate-800">
            <code>{fileContents[selectedFile]?.code}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
