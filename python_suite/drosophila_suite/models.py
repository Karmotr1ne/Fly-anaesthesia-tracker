"""
Data Models and Configuration Types
===================================
Typed structures for Drosophila Anesthesia Tracker Pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Union
import json
import os
from pathlib import Path
import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class PipelineConfig:
    """Master pipeline configuration parameters for Drosophila Anesthesia Tracker."""
    fps: float = 30.0
    body_length_px: float = 12.0
    body_length_thresh: float = 0.5    # 0.5 body lengths for micro-movement threshold
    max_speed_px_per_sec: float = 400.0
    max_speed_px_per_frame: float = 45.0
    occlusion_disp_thresh: float = 80.0
    occlusion_var_thresh: float = 8.0
    savgol_window: int = 7
    savgol_poly: int = 2
    
    # Multi-Chamber Grid Defaults
    grid_rows: int = 4
    grid_cols: int = 2
    grid_order: str = "column_first"   # "column_first" or "row_first"
    
    # Anesthesia Kinetics Parameters
    anesthesia_bin_size_sec: float = 5.0
    anesthesia_window_duration_sec: float = 120.0  # 判断窗口持续时间（秒），默认 120s
    anesthesia_window_bins: int = 24                # 24 * 5s = 120s
    anesthesia_activity_threshold: float = 0.01

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration dataclass to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Constructs PipelineConfig instance from dictionary, filtering unknown keys."""
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serializes configuration to JSON string or writes to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "PipelineConfig":
        """Loads configuration from JSON string or file path."""
        if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.loads(source)
        return cls.from_dict(data)

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Serializes configuration to YAML string or writes to file."""
        d = self.to_dict()
        if HAS_YAML:
            yaml_str = yaml.safe_dump(d, default_flow_style=False, sort_keys=False)
        else:
            lines = [f"{k}: {v}" for k, v in d.items()]
            yaml_str = "\n".join(lines) + "\n"

        if filepath is not None:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_yaml(cls, source: Union[str, Path]) -> "PipelineConfig":
        """Loads configuration from YAML string or file path."""
        if HAS_YAML:
            if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
                with open(source, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            else:
                data = yaml.safe_load(source)
        else:
            content = source
            if isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source)):
                with open(source, "r", encoding="utf-8") as f:
                    content = f.read()
            data = {}
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        if "." in v:
                            v = float(v)
                        else:
                            v = int(v)
                    except ValueError:
                        pass
                    data[k] = v
        return cls.from_dict(data or {})

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves configuration to file based on extension (.json, .yaml, .yml)."""
        fp = str(filepath).lower()
        if fp.endswith(".yaml") or fp.endswith(".yml"):
            self.to_yaml(filepath)
        else:
            self.to_json(filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PipelineConfig":
        """Loads configuration from file based on extension (.json, .yaml, .yml)."""
        fp = str(filepath).lower()
        if fp.endswith(".yaml") or fp.endswith(".yml"):
            return cls.from_yaml(filepath)
        return cls.from_json(filepath)


@dataclass
class TrackingRecord:
    """Single frame tracking record with millisecond timestamp support for VFR/dropped frames."""
    frame: int
    chamber_id: int
    fly_id: int = 0
    timestamp_s: float = 0.0
    x_px: float = np.nan
    y_px: float = np.nan
    norm_x: float = np.nan
    norm_y: float = np.nan
    area: float = np.nan
    roi_x1: Optional[int] = None
    roi_y1: Optional[int] = None
    roi_x2: Optional[int] = None
    roi_y2: Optional[int] = None


@dataclass
class KinematicRecord:
    """Cleaned kinematic record with explicit time coordinate."""
    frame: int
    chamber_id: int
    timestamp_s: float
    x_clean: float
    y_clean: float
    norm_pos: float
    speed: float
    activity_bin: float = 0.0


@dataclass
class AnesthesiaSummary:
    """Summary metrics for anesthesia induction kinetics."""
    chamber_id: int
    induction_time_sec: Optional[float]
    is_sedated: bool
    baseline_speed: float
    pre_sedation_activity: float
    stillness_bins_count: int = 0
    notes: str = ""
