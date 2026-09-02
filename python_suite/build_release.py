"""
Release Build Script for PyInstaller
====================================
Automated release builder for Drosophila Behavior & Anesthesia Tracking Suite.
Can be executed from repository root or from within python_suite/.

Usage:
    python build_release.py
    python python_suite/build_release.py
"""

import sys
import os
import subprocess
import shutil

# Ensure directory context is normalized
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def check_prerequisites():
    print(">> Checking build prerequisites...")
    required = ["PyQt6", "cv2", "numpy", "pandas", "scipy", "matplotlib", "PyInstaller"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[Warning] Optional/missing packages: {missing}")
        print("For building release executables, run: pip install -r requirements.txt pyinstaller")
    else:
        print(">> All core dependencies verified.")
    return True


def run_unit_tests():
    print(">> Running module smoke tests before release build...")
    try:
        import numpy as np
        import pandas as pd
        from drosophila_suite.cleaner import KinematicCleaner
        from drosophila_suite.stationary_engine import StationaryDetectionEngine
        from drosophila_suite.anesthesia import AnesthesiaAnalyzer

        # 1. Test Stationary Engine
        test_act = np.array([5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0])
        res_mask = StationaryDetectionEngine.sliding_window_max_filter(test_act, window_size=3, threshold=0.1)
        assert res_mask[2] == True, "Stationary engine test failed"

        # 2. Test Cleaner & Artifact Clamping
        mock_raw = pd.DataFrame({
            "frame": list(range(100)),
            "chamber_id": [0] * 100,
            "x_px": [50.0 + (i % 20) for i in range(100)],
            "y_px": [20.0 + (i % 5) for i in range(100)],
        })
        mock_raw.loc[40:45, "x_px"] = 200.0  # simulate occlusion jump
        cleaner = KinematicCleaner(fps=30.0)
        cleaned = cleaner.clean_trajectory(mock_raw)
        assert "norm_pos" in cleaned.columns and "speed" in cleaned.columns, "Cleaner failed"

        # 3. Test Anesthesia Induction
        an_analyzer = AnesthesiaAnalyzer(bin_size_sec=1.0, window_bins=5, activity_threshold=0.5, fps=30.0)
        an_res = an_analyzer.evaluate_induction(cleaned)
        assert not an_res.empty, "Anesthesia analyzer test failed"

        print(">> All unit test verifications passed successfully!\n")
        return True
    except Exception as e:
        print(f"[Error] Unit test failed: {e}")
        return False


def build_executable():
    spec_file = os.path.join(current_dir, "pyinstaller_build.spec")
    
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "-y",
        spec_file
    ]

    print(f">> Executing PyInstaller command in {current_dir}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=current_dir)
    if result.returncode == 0:
        print("\n" + "="*60)
        print("  BUILD SUCCESSFUL!")
        print("  Release executable located in 'dist/DrosophilaAnesthesiaTracker'")
        print("="*60 + "\n")
    else:
        print("\n[Error] PyInstaller build failed with return code:", result.returncode)


if __name__ == "__main__":
    if check_prerequisites():
        if run_unit_tests():
            build_executable()
