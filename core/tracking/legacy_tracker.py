"""Background modelling and local tracking with global chamber reacquisition."""

import cv2
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Optional, Any
from .candidate_selection import CandidateSelector
# Preserve the historical import surface for callers outside this package.
from .grid_alignment import RobustGridAligner
from .trajectory_data import (DetectionBlocks, interpolate_tracking_dataframe,
                              post_process_dynamic_interpolate)


def get_video_metadata(video_path: str) -> Tuple[int, float, int, int]:
    """Reads basic video metadata (total_frames, fps, width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total_frames, fps, w, h


def build_median_background(video_path: str, num_samples: int = 60, cancel_callback=None,
                            sample_start: float = 0.0, sample_end: float = 0.30) -> np.ndarray:
    """Sample the video uniformly, releasing the decoder on cancellation or failure."""
    total_frames, _, _, _ = get_video_metadata(video_path)
    if num_samples < 1:
        raise ValueError("Background sample count must be positive")
    if not 0 <= sample_start < sample_end <= 1:
        raise ValueError("Background sampling start must be before sampling end")
    cap = cv2.VideoCapture(video_path)
    samples, count = None, 0
    start = min(max(0,total_frames-1),int(total_frames*sample_start))
    end = max(start+1,int(total_frames*sample_end))
    indices = np.linspace(start, end - 1, min(num_samples, end - start), dtype=int)
    try:
        for index in indices:
            if cancel_callback and cancel_callback():
                raise InterruptedError("Background modelling cancelled")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if samples is None:
                    samples = np.empty((len(indices), *gray.shape), dtype=np.uint8)
                if gray.shape != samples.shape[1:]:
                    raise ValueError("Background sample dimensions changed")
                samples[count] = gray
                count += 1
    finally:
        cap.release()
    if not count:
        raise ValueError("Could not extract frames for the background model")
    background = np.empty(samples.shape[1:], dtype=np.uint8)
    for top in range(0, background.shape[0], 128):
        if cancel_callback and cancel_callback():
            raise InterruptedError("Background modelling cancelled")
        bottom = min(top + 128, background.shape[0])
        block = samples[:count, top:bottom].copy()
        background[top:bottom] = np.median(block, axis=0, overwrite_input=True).astype(np.uint8)
    return background


class RobustFlyTracker:
    'Track dark targets using background subtraction, preallocated masks, and sequential chamber processing.'
    def __init__(
        self,
        chamber_rois: List[Tuple[int, int, int, int]],
        chamber_ids: Optional[List[int]] = None,
        median_bg: Optional[np.ndarray] = None,
        diff_thresh: int = 14,
        min_fly_area: Optional[int] = None,
        max_fly_area: Optional[int] = None,
        num_workers: Optional[int] = None,
        temporal: bool = True,
        selection_options=None
    ):
        self.chambers = chamber_rois
        self.chamber_ids = chamber_ids or [i + 1 for i in range(len(chamber_rois))]
        self.median_bg = median_bg
        self.diff_thresh = diff_thresh

        heights = [max(20, b[3] - b[1]) for b in chamber_rois]
        avg_h = float(np.median(heights)) if heights else 80.0

        self.target_fly_area = max(60.0, avg_h * avg_h * 0.08)
        self.min_area = min_fly_area if min_fly_area is not None else max(15.0, self.target_fly_area * 0.15)
        self.max_area = max_fly_area if max_fly_area is not None else max(400.0, self.target_fly_area * 6.0)

        # 1. Preallocate morphology kernels.
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 2. Precompute chamber edge masks to suppress reflections.
        self.border_masks: Dict[int, np.ndarray] = {}
        for idx, (x1, y1, x2, y2) in enumerate(self.chambers):
            h, w = max(1, y2 - y1), max(1, x2 - x1)
            b_mask = np.zeros((h, w), dtype=np.uint8)
            pad_y = max(1, int(h * 0.04))
            pad_x = max(1, int(w * 0.02))
            # Image-left is the physical floor: keep it available for detection.
            b_mask[pad_y : h - pad_y, : w - pad_x] = 255
            self.border_masks[self.chamber_ids[idx]] = b_mask

        # Tracking state.
        self.last_known_pos: Dict[int, Optional[Tuple[float, float]]] = {cid: None for cid in self.chamber_ids}
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {cid: [] for cid in self.chamber_ids}
        self.last_debug_masks: Dict[int, np.ndarray] = {}
        self.selector = CandidateSelector(self.chamber_ids, temporal=temporal, options=selection_options)
        self.last_candidates = {}
        self._chamber_cache = None

    def _prepare_chambers(self, shape):
        """Build immutable geometry/background views once per tracker run."""
        image_height, image_width = shape
        self._chamber_cache = []
        for cid, (x1, y1, x2, y2) in zip(self.chamber_ids, self.chambers):
            left, top = max(0, x1), max(0, y1)
            right, bottom = min(image_width, x2), min(image_height, y2)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                raise ValueError('Chamber lies outside the video frame.')
            mask = self.border_masks[cid][top-y1:bottom-y1, left-x1:right-x1]
            background = self.median_bg[top:bottom, left:right] if self.median_bg is not None else None
            self._chamber_cache.append(dict(cid=cid, bounds=(left, top, right, bottom),
                width=width, height=height, inv_width=1./width, inv_height=1./height,
                inv_area=1./(width*height), mask=mask, background=background,
                pad_x=max(1, int(width*.02)), pad_y=max(1, int(height*.04))))

    def _search_bounds(self, chamber, timestamp):
        state = self.selector.states[chamber['cid']]
        full = (0, 0, chamber['width'], chamber['height'])
        if (not self.selector.temporal or state.mode != 'Tracking' or state.trusted_area is None
                or timestamp - state.last_global_time >= self.selector.options.global_check_seconds):
            return full, 'Global'
        radius = max(self.selector.options.minimum_search_radius,
                     self.selector.options.search_radius_factor * math.sqrt(state.trusted_area))
        x, y = state.trusted_pos
        bounds = (max(0, int(x-radius)), max(0, int(y-radius)),
                  min(chamber['width'], int(math.ceil(x+radius+1))),
                  min(chamber['height'], int(math.ceil(y+radius+1))))
        return bounds, 'Local' if bounds != full else 'Global'

    def close(self):
        'Provide the compatibility cleanup interface.'
        pass

    def __del__(self):
        self.close()

    def _extract_fly_candidates(
        self,
        chamber_crop: np.ndarray,
        bg_crop: Optional[np.ndarray],
        cid: int,
        roi_box: Tuple[int, int, int, int],
        chamber,
        search_bounds
    ):
        crop_h, crop_w = chamber_crop.shape[:2]
        h, w = chamber['height'], chamber['width']
        offset_x, offset_y, _, _ = search_bounds
        if crop_h < 5 or crop_w < 10:
            return [], np.zeros((crop_h, crop_w), dtype=np.uint8)

        # 1. Subtract the median background to isolate dark targets.
        if bg_crop is not None and bg_crop.shape == chamber_crop.shape:
            diff = cv2.subtract(bg_crop, chamber_crop)
        else:
            diff = cv2.bitwise_not(chamber_crop)

        # 2. Apply Gaussian filtering and a fixed threshold.
        blurred = cv2.GaussianBlur(diff, (3, 3), 0)
        _, mask = cv2.threshold(blurred, self.diff_thresh, 255, cv2.THRESH_BINARY)

        # 3. Reuse precomputed edge masks.
        sx1, sy1, sx2, sy2 = search_bounds
        b_mask = chamber['mask'][sy1:sy2, sx1:sx2]
        if b_mask is not None and b_mask.shape == mask.shape:
            mask = cv2.bitwise_and(mask, b_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_close)

        # 4. Rank contour centroids using temporal distance weighting.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"] + offset_x
                cy = M["m01"] / M["m00"] + offset_y

                _, (side_a, side_b), _ = cv2.minAreaRect(cnt)
                aspect_ratio = max(side_a, side_b) / max(1., min(side_a, side_b))
                left, top, width, height = cv2.boundingRect(cnt)
                touches_search_boundary = ((offset_x > 0 and left <= 1) or
                    (offset_y > 0 and top <= 1) or
                    (sx2 < w and left+width >= crop_w-1) or
                    (sy2 < h and top+height >= crop_h-1))
                local_left, local_top = left, top
                left, top = left + offset_x, top + offset_y
                pad_x, pad_y = chamber['pad_x'], chamber['pad_y']
                state = self.selector.states[cid]
                band = max(2., np.sqrt(state.trusted_area) * .5) if state.trusted_area else max(2., h * .08)
                edge_fraction = 0.
                # Bounds cover every pixel of this contour: interior candidates
                # need no raster mask or coordinate arrays. Image-left is exempt.
                if (left + width - 1 >= w - pad_x - band or
                        top < pad_y + band or top + height - 1 >= h - pad_y - band):
                    component = np.zeros((height, width), dtype=np.uint8)
                    shifted = cnt - np.array([[[local_left, local_top]]], dtype=cnt.dtype)
                    cv2.drawContours(component, [shifted], -1, 255, -1)
                    ys, xs = np.nonzero(component)
                    xs, ys = xs + left, ys + top
                    edge = ((xs >= w - pad_x - band) |
                            (ys < pad_y + band) | (ys >= h - pad_y - band))
                    edge_fraction = float(edge.mean()) if len(edge) else 0.

                candidates.append({
                    "local_pos": (cx, cy),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "roi_fraction": area * chamber['inv_area'],
                    "edge_fraction": edge_fraction,
                    "touches_search_boundary": touches_search_boundary
                })

        return candidates, mask

    def process_frame(
        self,
        frame_gray: np.ndarray,
        frame_idx: int,
        time: float = 0.0
    ) -> Tuple[List[dict], Dict[int, Optional[dict]]]:
        'Process chambers sequentially with preallocated tracking resources.'
        records = []
        frame_detections = {}
        if self._chamber_cache is None:
            self._prepare_chambers(frame_gray.shape[:2])

        for idx, chamber in enumerate(self._chamber_cache):
            cid = chamber['cid']
            x1, y1, x2, y2 = self.chambers[idx]
            gx1, gy1, gx2, gy2 = chamber['bounds']
            full_crop = frame_gray[gy1:gy2, gx1:gx2]
            bounds, search_mode = self._search_bounds(chamber, time)
            state = self.selector.states[cid]
            reason = 'Periodic' if search_mode == 'Global' and state.mode == 'Tracking' else 'Initial'

            def detect(bounds):
                a, b, c, d = bounds
                background = chamber['background']
                candidates, mask = self._extract_fly_candidates(full_crop[b:d, a:c],
                    background[b:d, a:c] if background is not None else None,
                    cid, chamber['bounds'], chamber, bounds)
                return [self.selector.score(value, cid) for value in candidates], mask

            candidates, mask = detect(bounds)
            if search_mode == 'Local':
                trusted = [value for value in candidates if self.selector.trustworthy(value)]
                best = max(trusted, key=lambda value: value['appearance_confidence'], default=None)
                reason = ('Missing' if best is None else 'WindowBoundary' if best['touches_search_boundary']
                          else 'Trajectory' if self.selector.trajectory_abnormal(state, best, time) else '')
                if reason:
                    bounds = (0, 0, chamber['width'], chamber['height'])
                    candidates, mask = detect(bounds)
                    search_mode = 'GlobalFallback'
            if search_mode != 'Local':
                state.last_global_time = time
            candidate = self.selector.select(cid, candidates, time, scored=True)
            self.last_candidates[cid] = candidates
            self.last_debug_masks[cid] = mask
            frame_detections[cid] = candidate

            cw = max(1.0, float(gx2 - gx1))
            ch = max(1.0, float(gy2 - gy1))

            record = {
                'chamber_id': cid, 'fly_id': 0, 'frame': frame_idx,
                'time': round(time, 3),
                'x': np.nan, 'y': np.nan, 'x_ratio': np.nan, 'y_ratio': np.nan,
                'roi_x1': x1, 'roi_y1': y1, 'roi_x2': x2, 'roi_y2': y2,
                'area': np.nan, 'is_interpolated': 0,
            }
            if candidate is not None:
                local_x, local_y = candidate['local_pos']
                abs_x, abs_y = gx1 + local_x, gy1 + local_y
                self.last_known_pos[cid] = (abs_x, abs_y)
                self.trajectory_history[cid].append((abs_x, abs_y))
                if len(self.trajectory_history[cid]) > 35:
                    self.trajectory_history[cid].pop(0)
                record.update(
                    x=round(abs_x, 2), y=round(abs_y, 2),
                    x_ratio=round(np.clip((abs_x - gx1) / cw, 0.0, 1.0), 4),
                    y_ratio=round(np.clip((abs_y - gy1) / ch, 0.0, 1.0), 4),
                    area=round(candidate['area'], 1),
                )
            records.append(record)

            state = self.selector.states[cid]
            diagnostic = candidate or max(self.last_candidates.get(cid, []),
                key=lambda value: value['appearance_confidence'], default={})
            records[-1].update({
                'appearance_confidence': diagnostic.get('appearance_confidence', 0.),
                'edge_fraction': diagnostic.get('edge_fraction', 0.),
                'tracking_status': ('Accepted' if candidate else state.mode
                                    if state.mode == 'Reacquiring' else 'Rejected'),
                'segment_id': state.segment_id,
                'search_mode': search_mode,
                'reacquisition_reason': reason if candidate is None or search_mode != 'Local' else '',
            })
            if candidate is None and state.mode == 'Reacquiring':
                self.last_known_pos[cid] = None

        return records, frame_detections


class FlyVisionTracker:
    'Track within a bounding envelope to reduce unnecessary pixel conversion.'
    def __init__(
        self,
        chamber_rois: List[Any],
        chamber_ids: Optional[List[int]] = None,
        diff_thresh: int = 14,
        selection_options=None
    ):
        parsed_rois = []
        parsed_ids = []
        for i, item in enumerate(chamber_rois):
            if isinstance(item, dict) and "roi" in item:
                parsed_rois.append(item["roi"])
                parsed_ids.append(item.get("chamber_id", i + 1))
            else:
                parsed_rois.append(item)
                parsed_ids.append(i + 1)

        self.chambers = parsed_rois
        self.chamber_ids = chamber_ids or parsed_ids
        self.diff_thresh = diff_thresh
        self.selection_options = selection_options
        self.median_bg = None

        # Compute a bounding envelope around all chambers.
        if self.chambers:
            self.env_x1 = max(0, min(b[0] for b in self.chambers) - 5)
            self.env_y1 = max(0, min(b[1] for b in self.chambers) - 5)
            self.env_x2 = max(b[2] for b in self.chambers) + 5
            self.env_y2 = max(b[3] for b in self.chambers) + 5
        else:
            self.env_x1, self.env_y1, self.env_x2, self.env_y2 = 0, 0, 0, 0

    def build_background(self, video_path: str, num_samples: int = 60) -> np.ndarray:
        self.median_bg = build_median_background(video_path, num_samples=num_samples)
        return self.median_bg

    def track_video(
        self,
        video_path: str,
        interpolate: bool = True,
        progress_callback=None,
        cancel_callback=None,
        max_gap=30,
        gap_speed=400.0,
        frame_callback=None,
        record_sink=None,
        collect_records=True,
    ) -> pd.DataFrame:
        total_frames, fps, img_w, img_h = get_video_metadata(video_path)
        if self.median_bg is None:
            self.build_background(video_path)

        tracker = RobustFlyTracker(
            self.chambers,
            chamber_ids=self.chamber_ids,
            median_bg=self.median_bg,
            diff_thresh=self.diff_thresh,
            selection_options=self.selection_options
        )

        cap = cv2.VideoCapture(video_path)
        all_records = DetectionBlocks() if collect_records else None
        observed_chambers = set()
        f_idx = 0

        # Clamp the bounding envelope to the frame.
        e_x1 = max(0, self.env_x1)
        e_y1 = max(0, self.env_y1)
        e_x2 = min(img_w, self.env_x2) if self.env_x2 > 0 else img_w
        e_y2 = min(img_h, self.env_y2) if self.env_y2 > 0 else img_h
        use_env = (e_x2 > e_x1 and e_y2 > e_y1 and (e_x2 - e_x1) < img_w and (e_y2 - e_y1) < img_h)
        frame_gray = np.zeros((img_h, img_w), dtype=np.uint8)
        # OpenCV can write directly into a contiguous destination buffer.
        envelope_gray = np.empty((e_y2 - e_y1, e_x2 - e_x1), dtype=np.uint8) if use_env else None

        try:
            while True:
                if cancel_callback and cancel_callback():
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                time = (timestamp_ms / 1000.0) if (timestamp_ms is not None and timestamp_ms > 0) else (f_idx / fps)

                # Convert only the active envelope, or the full frame, to grayscale.
                if use_env:
                    cv2.cvtColor(frame[e_y1:e_y2, e_x1:e_x2], cv2.COLOR_BGR2GRAY, dst=envelope_gray)
                    frame_gray[e_y1:e_y2, e_x1:e_x2] = envelope_gray
                else:
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=frame_gray)

                frame_records, _ = tracker.process_frame(frame_gray, f_idx, time=time)
                if record_sink is not None:
                    record_sink(frame_records, fps)
                if all_records is not None:
                    all_records.extend(frame_records)
                for record in frame_records:
                    if np.isfinite(record['x']) and np.isfinite(record['y']):
                        observed_chambers.add(int(record['chamber_id']))
                if frame_callback is not None:
                    frame_callback(f_idx, frame, frame_records)
                f_idx += 1

                if progress_callback and f_idx % 100 == 0:
                    progress_callback(f_idx, total_frames)
        finally:
            cap.release()
            tracker.close()

        if cancel_callback and cancel_callback():
            return pd.DataFrame()
        if all_records is None:
            data = pd.DataFrame()
            data.attrs.update(processed_frames=f_idx, included_chamber_ids=sorted(observed_chambers),
                excluded_chamber_ids=sorted(set(self.chamber_ids)-observed_chambers), fps=fps)
            return data
        data = all_records.dataframe()
        if interpolate and not data.empty:
            data = interpolate_tracking_dataframe(data, fps, max_gap, gap_speed)
        return data
