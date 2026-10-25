"""Canonical internal fields and human-readable, unit-labelled CSV output."""
from __future__ import annotations

import pandas as pd

# Names are intentionally short; units belong to exported labels.
LABELS = {
    "appearance_confidence": "appearance confidence(0–1)",
    "edge_fraction": "edge fraction(0–1)",
    "tracking_status": "tracking status", "segment_id": "trajectory segment",
    "search_mode": "search mode", "reacquisition_reason": "reacquisition reason",
    "frame": "frame", "chamber_id": "chamber id", "fly_id": "fly id",
    "time": "time(s)", "elapsed": "elapsed time(s)", "video_time": "video time(s)",
    "x": "x(px)", "y": "y(px)",
    "x_clean": "cleaned x(px)", "y_clean": "cleaned y(px)",
    "x_ratio": "normalised x(0–1)", "y_ratio": "normalised y(0–1)",
    "height": "normalised height(0–1)", "position": "normalised position(0–1)",
    "speed": "speed(px/s)", "baseline_speed": "baseline speed(px/s)",
    "mean_speed": "mean speed(px/s)",
    "still_window": "still window(s)", "still_speed": "still speed(px/s)",
    "drop_height": "height drop(0–1)", "drop_window": "drop window(s)",
    "drop_sustain": "drop sustain(s)", "bottom_window": "bottom window(s)",
    "max_gap": "max gap(frames)", "gap_speed": "gap speed(px/s)",
    "activity": "activity(0–1)", "fps": "frame rate(fps)", "area": "area(px²)",
    "roi_x1": "roi x1(px)", "roi_y1": "roi y1(px)",
    "roi_x2": "roi x2(px)", "roi_y2": "roi y2(px)",
    "anaesthetic_onset": "anaesthetic gas onset(s)", "mix_onset": "mix gas onset(s)", "sedation_onset": "sedation onset(s)",
    "anaesthesia_onset": "anaesthesia onset(s)",
    "sedation_latency": "sedation latency(s)", "anaesthesia_latency": "anaesthesia latency(s)",
    "sedation_confidence": "sedation confidence(0–1)",
    "is_detected": "detected", "is_interpolated": "interpolated", "is_rejected": "rejected",
    "coordinate_status": "coordinate status", "behavioural_state": "behavioural state",
    "is_drop_event": "drop candidate", "is_sedated": "sedated", "is_anaesthetised": "anaesthetised",
    "drops": "drop candidate count",
    "o2_flow": "o2 flow(sccm)", "anes_flow": "anaesthetic flow(sccm)", "n2_flow": "n2 flow(sccm)",
    "o2_sp": "o2 target(sccm)", "anes_sp": "anaesthetic target(sccm)", "n2_sp": "n2 target(sccm)",
}

FIELD_ALIASES = {
    "anaesthetic_gas_onset": "anaesthetic_onset", "mix_gas_onset": "mix_onset",
    "gas_onset": "anaesthetic_onset", "gas onset(s)": "anaesthetic_onset",
    "timestamp_s": "time", "time_s": "time", "timestamp": "time",
    "rel_time_s": "elapsed", "video_time_s": "video_time",
    "x_px": "x", "y_px": "y", "speed_px_s": "speed",
    "norm_x": "x_ratio", "norm_y": "y_ratio", "norm_height": "height", "norm_pos": "position",
    "activity_bin": "activity", "behavioral_state": "behavioural_state",
    "anesthesia_gas_onset_sec": "anaesthetic_onset", "anaesthesia_gas_onset_sec": "anaesthetic_onset",
    "gas_onset_sec": "anaesthetic_onset", "sedation_onset_sec": "sedation_onset",
    "anesthesia_onset_sec": "anaesthesia_onset", "anaesthesia_onset_sec": "anaesthesia_onset",
    "sedation_latency_sec": "sedation_latency", "anesthesia_latency_sec": "anaesthesia_latency",
    "anaesthesia_latency_sec": "anaesthesia_latency", "is_anesthetized": "is_anaesthetised",
    "is_anaesthetized": "is_anaesthetised", "total_drop_events": "drops",
}

CONFIG_ALIASES = {
    **FIELD_ALIASES,
    "anesthesia_onset_time": "anaesthetic_onset", "anaesthesia_onset_time": "anaesthetic_onset",
    "anesthesia_still_sec": "still_window", "anaesthesia_still_sec": "still_window",
    "anesthesia_speed_thresh": "still_speed", "anaesthesia_speed_thresh": "still_speed",
    "gas_mixing_window_sec": "mixing_window", "sedation_conf_thresh": "confidence",
    "rebound_horizon_sec": "rebound_window", "floor_height_thresh": "floor_height",
    "ceiling_height_thresh": "ceiling_height", "speed_still_thresh": "still_speed",
    "drop_delta_h": "drop_height",
    "sedate_speed_ratio": "sedation_ratio", "drop_height_delta": "drop_height",
    "sedate_drop_speed": "drop_height", "drop_dt_sec": "drop_window",
    "drop_sustain_sec": "drop_sustain", "bottom_quiescent_sec": "bottom_window",
    "max_gap_frames": "max_gap", "max_interp_gap": "max_gap",
    "max_interp_speed_px_s": "gap_speed", "max_speed_px_per_sec": "gap_speed",
    "interpolate_gaps": "interpolate", "body_len_px": "body_length",
    "body_len_thresh": "body_ratio", "max_speed_px": "max_step",
}

LABELS.update({
    "sedation_ratio": "sedation speed ratio(0–1)", "is_filled": "midpoint filled",
    "mixing_window": "gas mixing window(s)", "rebound_window": "rebound window(s)",
    "sample": "sample", "source": "source", "dataset": "dataset", "session": "session",
    "condition": "condition", "time_window": "time window", "batch": "batch",
    "custom": "custom group", "include": "include", "status": "status",
    "sedation_latency_state": "sedation event status",
    "anaesthesia_latency_state": "anaesthesia event status",
    "version_conflict": "version conflict", "quality": "quality status",
    "observed_ratio": "observed proportion(0–1)",
})

OUTPUT_FIELDS = {label: key for key, label in LABELS.items()}

STATES = {"Anesthetized": "Anaesthetised", "Anaesthetized": "Anaesthetised",
          "Anesthetised": "Anaesthetised", "Anesthesia": "Anaesthesia"}


def field(name):
    """Resolve only known aliases; unknown units are never silently converted."""
    name = str(name).strip()
    return FIELD_ALIASES.get(name, OUTPUT_FIELDS.get(name, name))


def label(name):
    return LABELS.get(name, name)


def normalise(data):
    """Accept old or displayed headers and reject conflicting duplicate fields."""
    if not data.columns.is_unique:
        raise ValueError("Duplicate input column names must be resolved before import.")
    result = data.copy()
    for source in list(result.columns):
        target = field(source)
        if source == target:
            continue
        if target in result:
            left, right = result[target], result[source]
            present = left.notna() & right.notna()
            if not left[present].eq(right[present]).all():
                raise ValueError(f"Conflicting input fields: {target!r} and {source!r}")
            result[target] = left.combine_first(right)
            result = result.drop(columns=[source])
        else:
            result = result.rename(columns={source: target})
    for name in ("state", "behavioural_state"):
        if name in result:
            result[name] = result[name].replace(STATES)
    return result


def output(data):
    result = normalise(data)
    return result.rename(columns={name: label(name) for name in result.columns})


def read_csv(path, **kwargs):
    return normalise(pd.read_csv(path, **kwargs))


def write_csv(data, path, **kwargs):
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8-sig")
    output(data).to_csv(path, **kwargs)


def normalise_config(value):
    """Read legacy JSON names without changing their numerical meanings."""
    if isinstance(value, list):
        return [normalise_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for source, item in value.items():
        target = CONFIG_ALIASES.get(source, source)
        item = normalise_config(item)
        if target in result and result[target] != item:
            raise ValueError(f"Conflicting configuration fields: {source!r} and {target!r}")
        result[target] = item
    return result


def has_kinematics(fields):
    """Recognise cleaned trajectories, including legacy speed/height-only exports."""
    fields = {field(name) for name in fields}
    return ({"x_clean", "y_clean", "speed"}.issubset(fields) or
            ("speed" in fields and bool({"height", "position"} & fields)))
