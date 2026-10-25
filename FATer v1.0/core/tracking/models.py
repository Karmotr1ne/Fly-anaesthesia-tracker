"""Explicit contracts shared by the tracking page and its background jobs."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from .candidate_selection import SelectionOptions


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frames: int
    size: int
    modified_ns: int

    @property
    def shape(self):
        return self.height, self.width


@dataclass(frozen=True)
class Calibration:
    rois: tuple[tuple[int, int, int, int], ...]
    shape: tuple[int, int]
    rows: int
    cols: int
    order: str = "column_first"
    equal_size: bool = True
    source_video: str = ""
    reference_rois: tuple[tuple[int, int, int, int], ...] = field(default=(), repr=False, compare=False)
    reference_shape: tuple[int, int] = field(default=(), repr=False, compare=False)

    def validate(self, shape):
        if tuple(shape) != self.shape:
            raise ValueError("Video dimensions do not match the calibration.")
        if not self.rois or len(self.rois) != self.rows * self.cols:
            raise ValueError("Chamber count does not match the grid dimensions.")
        height, width = shape
        if any(not (0 <= a < c <= width and 0 <= b < d <= height)
               for a, b, c, d in self.rois):
            raise ValueError("Calibration extends beyond the video frame.")

    def for_shape(self, shape):
        """Map shared geometry in normalised coordinates to a video's pixels."""
        from dataclasses import replace
        self.validate(self.shape)
        height, width = shape
        reference_rois = self.reference_rois or self.rois
        reference_shape = self.reference_shape or self.shape
        source_height, source_width = reference_shape
        if min(height, width) <= 0:
            raise ValueError('Video dimensions must be positive.')
        boxes = []
        if tuple(shape) == reference_shape:
            mapped = replace(self, rois=reference_rois, shape=tuple(shape),
                             reference_rois=reference_rois, reference_shape=reference_shape)
            mapped.validate(shape)
            return mapped
        shared_width = max(1, min(width, round(
            (reference_rois[0][2] - reference_rois[0][0]) * width / source_width)))
        shared_height = max(1, min(height, round(
            (reference_rois[0][3] - reference_rois[0][1]) * height / source_height)))
        for left, top, right, bottom in reference_rois:
            x1 = min(width - 1, max(0, round(left * width / source_width)))
            y1 = min(height - 1, max(0, round(top * height / source_height)))
            x2 = min(width, max(x1 + 1, round(right * width / source_width)))
            y2 = min(height, max(y1 + 1, round(bottom * height / source_height)))
            if self.equal_size:
                x1, y1 = min(x1, width - shared_width), min(y1, height - shared_height)
                x2, y2 = x1 + shared_width, y1 + shared_height
            boxes.append((x1, y1, x2, y2))
        mapped = replace(self, rois=tuple(boxes), shape=tuple(shape),
                         reference_rois=reference_rois, reference_shape=reference_shape)
        mapped.validate(shape)
        return mapped


@dataclass(frozen=True)
class TrackingOptions:
    background_samples: int = 60
    background_start: float = 0.
    background_end: float = .30
    interpolate: bool = True
    max_gap: int = 30
    gap_speed: float = 400.0
    render_overlay: bool = False
    selection: SelectionOptions = field(default_factory=SelectionOptions)

    def validate(self):
        self.selection.validate()
        if self.background_samples < 1 or not 0 <= self.background_start < self.background_end <= 1:
            raise ValueError("Invalid background sampling settings.")
        if self.max_gap < 0 or not math.isfinite(self.gap_speed) or self.gap_speed <= 0:
            raise ValueError("Invalid interpolation settings.")

    def background_key(self, video, metadata):
        return (str(video).casefold(), metadata.size, metadata.modified_ns,
                self.background_samples, self.background_start, self.background_end)


@dataclass
class TrackingResult:
    status: str = "Queued"
    position_csv: Path | None = None
    raw_csv: Path | None = None
    overlay_video: Path | None = None
    overlay_status: str = "Not requested"
    error: str = ""
    overlay_error: str = ""
    included_chamber_ids: tuple[int, ...] = ()
    excluded_chamber_ids: tuple[int, ...] = ()
    config_path: Path | None = None
    calibration: Calibration | None = None


@dataclass
class VideoTask:
    key: str
    video_path: Path
    session_dir: Path
    telemetry_path: Path | None
    metadata: VideoMetadata
    calibration: Calibration | None = None
    checked: bool = True
    preview_frame: int = 0
    undo_history: list = field(default_factory=list)
    result: TrackingResult | None = None

    @property
    def status(self):
        if self.result is not None:
            return self.result.status
        return "Ready" if self.calibration is not None else "No ROI"



@dataclass(frozen=True)
class TrackingJob:
    key: str
    video_path: Path
    session_dir: Path
    telemetry_path: Path | None
    metadata: VideoMetadata
    calibration: Calibration
    options: TrackingOptions
    shared: bool
    background: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PreviewRequest:
    token: int
    video_path: Path
    frame: int
    mode: str
    rois: tuple[tuple[int, int, int, int], ...]
    options: TrackingOptions
    cache_key: tuple
