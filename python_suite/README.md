# Drosophila Anesthesia Tracker Suite (v0.1)

Integrated vision tracking, kinematic cleaning, anesthesia kinetics, and scientific visualization workbench with PyInstaller desktop packaging.

---

## 🏗️ Functional Architecture Overview (v0.1)

1. **Module 1: Vision Tracking & Multi-Chamber Calibration (`drosophila_suite.tracker`)**
   - Multi-Chamber adaptive grid alignment with **Auto-Snap** (intensity-adaptive tube bounding box detection).
   - Multi-frame temporal median background subtraction (`build_median_background`).
   - **Darkness Mass Score** integral centroid extraction (robust against mesh reflections and shadow artifacts).
   - Millisecond timestamp handling (`cv2.CAP_PROP_POS_MSEC`) for Variable Frame Rate (VFR) & dropped frames.

2. **Module 2: Kinematic Cleaning & Artifact Clamping (`drosophila_suite.cleaner`)**
   - 1%~99% quantile physical ROI boundary constraints.
   - **Occlusion Trap Clamping**: Detects jumps into plug gaps and relocates centroids to physical boundaries.
   - Velocity spike outlier filtering with jump rejection.
   - **Savitzky-Golay trajectory smoothing** (window = 7, order = 2) with 0.5 body-length micro-movement thresholding.

3. **Module 3: Core Stationary Detection Engine (`drosophila_suite.stationary_engine`)**
   - Vectorized sliding-window extreme filter operator:
     $$\max(A[t : t + W]) < \theta$$
   - Universal inactivity engine powering anesthesia induction and stillness duration calculations.

4. **Module 4: Anesthesia Kinetics Analyzer (`drosophila_suite.anesthesia`)**
   - Temporal 5-second binning of locomotion velocity.
   - Induction knockdown time determination ($W = 120\text{ s}$, 24 bins, $\theta = 0.01\text{ px/s}$).
   - Baseline speed and pre-sedation activity quantification.

5. **Module 5: Scientific Visualization & Reporting (`drosophila_suite.visualizer`)**
   - Multi-chamber dual Y-axis dynamic overview (`*_activity_position.png`).
   - Normalized Space-Time Kymograph (`*_kymograph_norm.png`) with hexbin density map.
   - Annotated video overlay synthesis (`*_cleaned_overlay.mp4`).
   - Consolidated quantitative CSV export (`*_results_summary.csv`).

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Launch Desktop GUI (v0.1)
```bash
python -m drosophila_suite.gui_app
# or
anesthesia-gui
```

### 3. Headless Batch CLI Processing
```bash
# Process a single session
python -m drosophila_suite.cli --csv data/trial_01_tracked.csv --out results/

# Process batch directory with video overlay
python -m drosophila_suite.cli --dir data/experiment_batch/ --overlay
```

---

## 📦 PyInstaller Packaging & Standalone Release

Build a self-contained, standalone executable release:

```bash
# Automated release builder with pre-flight smoke tests
python build_release.py

# Or direct PyInstaller spec compilation:
pyinstaller --clean -y pyinstaller_build.spec
```

The compiled release will be generated in `dist/DrosophilaAnesthesiaTracker` (or `dist/DrosophilaAnesthesiaTracker.exe` on Windows).
