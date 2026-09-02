"""
Drosophila Anesthesia Tracker Suite
===================================
Modular pipeline for high-throughput Drosophila vision tracking,
kinematic cleaning, and anesthesia kinetics testing.
"""

__version__ = "0.1.0"
__author__ = "Drosophila Behavioral Phenotyping Lab"

from .models import (
    TrackingRecord,
    KinematicRecord,
    AnesthesiaSummary,
    PipelineConfig,
)
from .tracker import (
    FlyVisionTracker,
    RobustFlyTracker,
    Interactive8ChamberCalibrator,
    RobustGridAligner,
    SymmetricGridAligner,
    build_median_background,
    get_video_metadata,
)
from .cleaner import KinematicCleaner, DrosophilaProcessor
from .stationary_engine import StationaryDetectionEngine
from .anesthesia import AnesthesiaAnalyzer
from .visualizer import ScientificVisualizer
from .pipeline import DrosophilaBehaviorPipeline

__all__ = [
    "TrackingRecord",
    "KinematicRecord",
    "AnesthesiaSummary",
    "PipelineConfig",
    "FlyVisionTracker",
    "RobustFlyTracker",
    "Interactive8ChamberCalibrator",
    "RobustGridAligner",
    "SymmetricGridAligner",
    "build_median_background",
    "get_video_metadata",
    "KinematicCleaner",
    "DrosophilaProcessor",
    "StationaryDetectionEngine",
    "AnesthesiaAnalyzer",
    "ScientificVisualizer",
    "DrosophilaBehaviorPipeline",
]
