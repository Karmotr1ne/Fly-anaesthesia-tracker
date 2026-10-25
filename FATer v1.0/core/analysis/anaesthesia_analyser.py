"""Schema adapter: legacy kinematic cleaning and three-state analysis (speed in px/s)."""
from core import schema
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .legacy_anaesthesia import AnaesthesiaAnalyser as LegacyAnalyser
from .kinematic_cleaner import KinematicCleaner

@dataclass
class AnaesthesiaParams:
    anaesthetic_onset: float = 300.0
    sedation_ratio: float = 0.05
    drop_height: float = 0.25
    still_window: float = 120.0
    still_speed: float = 0.1
    fps: float = 30.0
    baseline_quantile: float = 0.75
    floor_height: float = 0.05
    ceiling_height: float = 0.95
    drop_window: float = 1.0
    mixing_window: float = 10.0
    confidence: float = 0.60
    rebound_window: float = 50.0

    def validate(self):
        values = vars(self)
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("Analysis parameters must be finite")
        if self.fps <= 0 or self.still_window <= 0 or self.drop_window <= 0:
            raise ValueError("Frame rate and analysis windows must be positive")
        if any(values[name] < 0 for name in ("anaesthetic_onset", "still_speed", "mixing_window", "rebound_window")):
            raise ValueError("Onset, speed and time windows must be nonnegative")
        if any(not 0 <= values[name] <= 1 for name in
               ("sedation_ratio", "drop_height", "baseline_quantile", "floor_height", "ceiling_height", "confidence")):
            raise ValueError("Analysis ratios and normalised heights must be within 0-1")
        if self.floor_height >= self.ceiling_height:
            raise ValueError("Floor height must be below ceiling height")


class AnaesthesiaAnalyser:
    def __init__(self, params=None, **kwargs):
        self.params = replace(params) if params is not None else AnaesthesiaParams()
        for key, value in schema.normalise_config(kwargs).items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise TypeError(f"Unknown analysis parameter: {key}")
        self.params.validate()
    def _prepare(self, df):
        self.params.validate()
        out = schema.normalise(df)
        if out.empty:
            return out
        if "chamber_id" not in out:
            raise ValueError("Missing chamber id.")
        if not np.isfinite(self.params.fps) or self.params.fps <= 0:
            raise ValueError("Frame rate must be positive and finite.")
        rates = pd.to_numeric(out["fps"], errors="coerce").dropna() if "fps" in out else pd.Series(dtype=float)
        if not rates.empty and rates.nunique() != 1:
            raise ValueError("A dataset must have one consistent frame rate.")
        self.fps = float(rates.iloc[0]) if not rates.empty else self.params.fps
        if rates.empty and {"time", "frame"}.issubset(out):
            ordered = out.sort_values(["chamber_id", "frame"], kind="stable")
            delta = ordered.groupby("chamber_id")[["frame", "time"]].diff()
            valid = (delta["time"] > 0) & (delta["frame"] > 0)
            estimates = (delta.loc[valid, "frame"] / delta.loc[valid, "time"]).replace([np.inf, -np.inf], np.nan).dropna()
            if not estimates.empty:
                self.fps = float(estimates.median())
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Frame rate must be positive and finite.")
        if "time" not in out:
            if "frame" not in out:
                raise ValueError("Missing frame or time field.")
            out["time"] = out["frame"] / self.fps
        if "frame" not in out:
            out["frame"] = np.rint(out["time"] * self.fps).astype(int)
        if not schema.has_kinematics(out.columns):
            if not {"x", "y"}.issubset(out):
                raise ValueError("Need coordinates or speed(px/s); missing data is not stillness.")
            out = KinematicCleaner(fps=self.fps).clean_trajectory(out, _canonical=True)
        else:
            out = out.sort_values(["chamber_id", "frame"], kind="stable").reset_index(drop=True)
        if not out.empty and "height" not in out:
            if "position" not in out:
                raise ValueError("Missing normalised height.")
            out["height"] = out["position"]
        if not out.empty and "position" not in out:
            out["position"] = out["height"]
        return out
    def _engine(self):
        p = self.params
        return LegacyAnalyser(
            fps=getattr(self, "fps", p.fps), anaesthetic_onset=p.anaesthetic_onset,
            still_window=p.still_window, still_speed=p.still_speed,
            drop_height=p.drop_height, baseline_quantile=p.baseline_quantile,
            sedation_ratio=p.sedation_ratio, floor_height=p.floor_height,
            ceiling_height=p.ceiling_height, drop_window=p.drop_window,
            mixing_window=p.mixing_window, confidence=p.confidence,
            rebound_window=p.rebound_window,
        )
    def process_dataframe(self, df):
        prepared = self._prepare(df)
        if prepared.empty: return prepared
        out = self._engine().evaluate_states(prepared, _prepared=True)
        out['behavioural_state'] = out["state"].map({"Active":"Active","Sedate":"Sedated","Anaesthesia":"Anaesthetised", "Unknown":"Unknown"})
        from .directional_velocity import add_directional_velocity
        return add_directional_velocity(out)
    def evaluate_states(self, df, anaesthetic_onset=None):
        if anaesthetic_onset is not None: self.params.anaesthetic_onset = float(anaesthetic_onset)
        return self.process_dataframe(df)
    def extract_summary(self, df, anaesthetic_onset=None, *, _prepared=False):
        if anaesthetic_onset is not None: self.params.anaesthetic_onset = float(anaesthetic_onset)
        if df.empty: return pd.DataFrame()
        if "state" not in df:
            df = self.process_dataframe(df)
            _prepared = True
        return self._engine().extract_summary(df, _prepared=_prepared)
