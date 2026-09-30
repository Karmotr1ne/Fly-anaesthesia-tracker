# Fly-Anaesthesia-Tracker (FATer)

An Automated Computer-Vision Pipeline for Kinematic Analysis and Behavioural State Transition Profiling in *Drosophila melanogaster* under Chemical and Physical Anaesthesia.

---

## Overview

Quantitative phenotyping of loss-of-righting reflex, sedation depth, and recovery dynamics in *Drosophila melanogaster* requires high-resolution spatial tracking coupled with objective kinematic classification. **Fly-Anaesthesia-Tracker** provides an integrated computational environment to quantify locomotor trajectory dynamics across single- and multi-arena behavioral configurations.

The software addresses common experimental challenges in behavioral pharmacology and neurogenetics, such as non-uniform illumination profiles, minor background shifts, and boundary occlusion artifacts, while delivering reproducible, millisecond-scale kinetic readouts. Standalone binary builds are distributed via PyInstaller to enable zero-dependency execution in standard laboratory environments.

---

## Computation Methods and Pipeline

The processing architecture follows a sequential, modular pipeline:

1. **Spatial Normalisation & ROI Segmentation**
   - User-defined or automated detection of multi-well chambers (circular wells or rectangular arenas).
   - Pixel-to-metric spatial calibration factor ($\text{px} \to \text{mm}$) estimation.
   - Spatial masking to eliminate inter-arena border cross-talk and perimeter optical aberrations.

2. **Foreground Extraction & Motion Segmentation**
   - Adaptive background modeling with morphological opening/closing operations to attenuate camera noise and dust artifacts.
   - Connected component analysis and contour detection for centroid $(x_t, y_t)$ and bounding box determination.
   - Dynamic thresholding tolerant to non-homogeneous arena illumination.

3. **Trajectory Reconstruction & Filtering**
   - Nearest-neighbor temporal association between consecutive frames with trajectory linking.
   - Coordinate smoothing via Savitzky-Golay or moving-average digital filtering to attenuate sub-pixel jitter.
   - Handling of temporary loss-of-signal during deep immobility or near-boundary contacts.

4. **Kinematic Feature Derivation & State Classification**
   - Instantaneous velocity computation:
     $$v_t = \frac{\sqrt{(x_t - x_{t-\Delta t})^2 + (y_t - y_{t-\Delta t})^2}}{\Delta t}$$
   - Angular displacement and turning velocity derivation.
   - Quantitative thresholding into discrete behavioral states:
     - Active locomotion
     - Micromovement / slow grooming
     - Quiescence / complete immobility (loss of voluntary displacement)
   - Extraction of latency to sedation (induction time) and time to recovery (emergence kinetics).

---

## Key Features

- **High-Throughput Multi-Subject Tracking:** Synchronous analysis of multi-arena assays from single continuous video streams.
- **Robustness to Illumination Variations:** Resilient foreground-background separation designed for ambient and transmitted-light chamber setups.
- **Customizable Behavioral Thresholds:** User-definable velocity, spatial perimeter, and duration thresholds for defining immobilisation bouts.
- **Standalone Distribution:** PyInstaller-compiled binaries eliminate the requirement for local Python runtimes, C++ compilers, or external package managers.
- **Downstream Analytics Compatibility:** Exports structured, tidy CSV tables formatted for immediate ingestion by R, Python (Pandas/Polars), and GraphPad Prism.

---

## Supported Input Specifications

- **Video Formats:** Standard uncompressed or compressed containers (`.mp4`, `.avi`, `.mov`, `.mkv`).
- **Codec Support:** H.264, MPEG-4, MJPEG, and raw uncompressed frames.
- **Frame Rates:** 10–120 fps (calibrated internally via metadata or explicit manual parameter entry).

---

## Deployment and Execution

### 1. Standalone Executable (Recommended for Experimentalists)

Pre-compiled standalone packages are provided for Windows and macOS. These archives contain all bundled scientific dependencies (OpenCV, NumPy, SciPy, PyQt).

1. Download the latest pre-compiled archive from the **Releases** page:
   - Windows: `Fly-Anaesthesia-Tracker-Windows-x64.zip`
   - macOS: `Fly-Anaesthesia-Tracker-macOS.zip`
2. Extract the archive into a preferred directory.
3. Run the executable:
   - **Windows:** Launch `Fly-Anaesthesia-Tracker.exe`.
   - **macOS:** Open a terminal window, grant execution privileges if required (`chmod +x Fly-Anaesthesia-Tracker.app`), and launch.

*Note: Administrative privileges are not required. All dependencies run entirely within the isolated application runtime directory.*

### 2. Execution from Source (Development Environment)

To modify source algorithms or integrate into automated cluster workflows:

```bash
# Clone repository
git clone [https://github.com/Karmotr1ne/Fly-anaesthesia-tracker.git](https://github.com/Karmotr1ne/Fly-anaesthesia-tracker.git)
cd Fly-anaesthesia-tracker

# Initialize virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run main application
python main.py
3. Compilation via PyInstallerTo build the standalone bundle directly from local modifications:Bashpip install pyinstaller
pyinstaller FlyTracker.spec --clean --noconfirm
The compiled binary and runtime resources will populate within the dist/ directory.Data Structure and Output FormatProcessing an assay generates two structured tabular outputs:1. Time-Series Trajectory Log (*_trajectories.csv)Columns contain continuous kinematics sampled across each frame:Column NameUnitsDescriptionframe_idxintegerFrame sequence indextimestampsecondsRelative time elapsed from assay onsetarena_idstring / intArena or well identifierpos_x_mmmillimetersCalibrated horizontal coordinatepos_y_mmmillimetersCalibrated vertical coordinatevelocitymm/sInstantaneous Euclidean velocitystatecategoricalClassified state (active, micromovement, immobilised)2. Summary Phenotype Metrics (*_summary.csv)Columns summarize cumulative phenotypic variables per subject:Metric NameUnitsDescriptionarena_idstring / intArena or well identifiertotal_distancemillimetersTotal cumulative path length traversedmean_speed_activemm/sMean speed during non-quiescent phasesinduction_latencysecondsElapsed time to sustained immobilityrecovery_latencysecondsElapsed time to resumption of coordinated movementimmobility_ratiofraction (0–1)Proportion of assay duration spent in quiescent stateCitation and AttributionIf this software contributes to research resulting in academic publication, please cite the framework as follows:
@software{FlyAnaesthesiaTracker,
  author    = {Songlin Yang (Karmotr1ne)},
  title     = {Fly-Anaesthesia-Tracker: Quantitative Trajectory Tracking and Behavioural State Transition Profiling in Drosophila},
  year      = {2026},
  publisher = {GitHub},
  url       = {[https://github.com/Karmotr1ne/Fly-anaesthesia-tracker](https://github.com/Karmotr1ne/Fly-anaesthesia-tracker)}
}
LicenseThis software is released under the MIT License. Refer to the LICENSE file for terms governing distribution, modification, and academic reuse.