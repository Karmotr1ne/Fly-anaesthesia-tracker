"""Qt adapter for calibrated chamber tracking with explicit execution options."""
from dataclasses import dataclass
import threading
import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from .legacy_tracker import FlyVisionTracker, get_video_metadata, build_median_background
from .grid_alignment import RobustGridAligner
from .trajectory_data import interpolate_tracking_dataframe


@dataclass
class ChamberROI:
    x: int
    y: int
    w: int
    h: int
    chamber_id: int = 1

    def as_xyxy(self):
        return (int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h))


class FlyTrackerEngine(QObject):
    progress_updated = pyqtSignal(int, float)
    phase_changed = pyqtSignal(str)
    tracking_completed = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cancel = threading.Event()

    def abort(self):
        self._cancel.set()

    @staticmethod
    def generate_grid_rois_snapped(gray_frame, first_roi, rows=4, cols=2,
                                   order="column_first", equal_size=True):
        x, y, width, height = first_roi
        boxes = RobustGridAligner.estimate_chambers_from_first_roi(
            gray_frame, (x, y, x + width, y + height), rows, cols, order="column_first")
        boxes = RobustGridAligner.snap_all_boxes(gray_frame, boxes, rows, cols, equal_size=equal_size)
        image_h, image_w = gray_frame.shape[:2]
        if equal_size:
            width, height = min(int(width), image_w), min(int(height), image_h)
            boxes = [(int(np.clip((a+c-width)/2, 0, image_w-width)),
                      int(np.clip((b+d-height)/2, 0, image_h-height))) for a,b,c,d in boxes]
            boxes = [(a,b,a+width,b+height) for a,b in boxes]
        if order == "row_first":
            boxes = [boxes[c*rows+r] for r in range(rows) for c in range(cols)]
        return [ChamberROI(a,b,c-a,d-b,index+1) for index,(a,b,c,d) in enumerate(boxes)]

    compute_temporal_median_background = staticmethod(build_median_background)

    @pyqtSlot(str, object)
    def run_tracking(self, video_path, rois=None, diff_threshold=14, max_gap=30,
                     background_samples=60, interpolate=True,
                     gap_speed=400.0, cancel_callback=None,
                     background_start=0.0, background_end=0.30, background=None,
                     frame_callback=None, selection_options=None,
                     record_sink=None, collect_records=True):
        self._cancel.clear()
        cancelled = lambda: self._cancel.is_set() or bool(cancel_callback and cancel_callback())
        try:
            total, fps, width, height = get_video_metadata(str(video_path))
            bounds = [roi.as_xyxy() if isinstance(roi, ChamberROI) else tuple(roi)
                      for roi in (rois or [(0,0,width,height)])]
            if any(not (0 <= a < c <= width and 0 <= b < d <= height) for a,b,c,d in bounds):
                raise ValueError("Chamber calibration extends beyond the video frame")
            if background_samples < 1 or max_gap < 0 or gap_speed <= 0:
                raise ValueError("Invalid tracking options")
            if not collect_records and record_sink is None:
                raise ValueError('Streaming tracking requires a CSV sink.')
            if cancelled():
                return pd.DataFrame()
            tracker = FlyVisionTracker(bounds, diff_thresh=int(diff_threshold), selection_options=selection_options)
            self.phase_changed.emit("Building Background")
            if background is not None:
                if background.shape != (height, width):
                    raise ValueError("Cached background dimensions do not match the video")
                tracker.median_bg = background
            else:
                tracker.median_bg = build_median_background(
                    str(video_path), int(background_samples), cancel_callback=cancelled,
                    sample_start=background_start, sample_end=background_end)
            if cancelled():
                return pd.DataFrame()
            self.phase_changed.emit("Tracking")
            data = tracker.track_video(
                str(video_path), interpolate=False,
                progress_callback=lambda current,count: self.progress_updated.emit(
                    current, current/max(1,count)*100.0),
                cancel_callback=cancelled, max_gap=int(max_gap),
                gap_speed=float(gap_speed), frame_callback=frame_callback,
                record_sink=record_sink, collect_records=collect_records)
            if cancelled():
                return pd.DataFrame()
            if not collect_records:
                if not data.attrs.get('processed_frames'):
                    raise ValueError('Tracking produced no decoded frames.')
                self.tracking_completed.emit(data)
                return data
            if data.empty:
                raise ValueError("Tracking produced no trajectory records")
            processed_frames = int(data.frame.nunique())
            observed = np.isfinite(data['x']) & np.isfinite(data['y'])
            if "is_interpolated" in data:
                observed &= data["is_interpolated"].fillna(0).eq(0)
            included = sorted(int(value) for value in data.loc[observed,"chamber_id"].unique())
            all_chambers = sorted(int(value) for value in data["chamber_id"].unique())
            data["is_detected"] = observed.astype(int)
            data = data.loc[data["chamber_id"].isin(included)].copy()
            if interpolate and not data.empty:
                data = interpolate_tracking_dataframe(data, fps, int(max_gap), float(gap_speed))
            data["fps"] = fps
            data.attrs["processed_frames"] = processed_frames
            data.attrs["included_chamber_ids"] = included
            data.attrs["excluded_chamber_ids"] = [value for value in all_chambers if value not in included]
            self.progress_updated.emit(processed_frames, 100.0)
            self.tracking_completed.emit(data)
            return data
        except InterruptedError:
            return pd.DataFrame()
        except Exception as exc:
            self.error_occurred.emit(str(exc))
            return pd.DataFrame()
