"""
Module 1: Vision Tracking & Multi-Chamber Calibration
=====================================================
Multi-Chamber computer vision tracking engine:
1. Interactive Multi-Chamber Calibrator with Auto-Snap (intensity adaptive alignment)
2. Robust Grid Aligner for arbitrary (Rows x Cols) pitch and center divider detection
3. Multi-frame temporal median background modeling
4. Darkness Mass Score centroid extraction (robust to wire mesh & reflection artifacts)
5. Temporal kinematic interpolation recovery.
"""

import os
import csv
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import pandas as pd


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


def build_median_background(video_path: str, num_samples: int = 60) -> np.ndarray:
    """
    Uniformly samples frames across the video and builds a temporal median background.
    Removes moving flies while preserving chamber walls and static lighting.
    """
    total_frames, _, w, h = get_video_metadata(video_path)
    cap = cv2.VideoCapture(video_path)

    start_f = int(total_frames * 0.05)
    end_f = int(total_frames * 0.95)
    step = max(1, (end_f - start_f) // max(1, num_samples))
    sample_indices = list(range(start_f, end_f, step))[:num_samples]
    frames_gray = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_gray.append(gray)

    cap.release()
    if not frames_gray:
        raise ValueError("Could not extract sample frames for median background.")

    stacked = np.stack(frames_gray, axis=0)
    median_bg = np.median(stacked, axis=0).astype(np.uint8)
    return median_bg


class SymmetricGridAligner:
    """
    High-precision grid generator for multi-row, multi-column arenas (e.g. 8-Chamber non-symmetric tubes):
    Identifies center dividing grooves and spans outer column boundaries.
    """
    @staticmethod
    def generate_symmetric_chambers(
        frame_shape: Tuple[int, ...],
        first_roi: Optional[Tuple[int, int, int, int]] = None,
        rows: int = 4,
        cols: int = 2,
        order: str = "column_first"
    ) -> List[Dict[str, Any]]:
        img_h, img_w = frame_shape[:2]
        
        # Default margins: 2% horizontal, 6% vertical
        margin_x = int(img_w * 0.02)
        margin_y = int(img_h * 0.06)
        avail_w = img_w - 2 * margin_x
        avail_h = img_h - 2 * margin_y
        
        # Center divider gap (approx 4% of width)
        center_divider_gap = int(img_w * 0.04) if cols > 1 else 0
        
        col_w = (avail_w - (cols - 1) * center_divider_gap) // cols
        row_h = int(avail_h / rows * 0.72)  # Individual tube vertical height
        row_step = avail_h / rows
        
        chambers = []
        ch_idx = 1
        
        if order == "column_first":
            for c in range(cols):
                cx1 = margin_x + c * (col_w + center_divider_gap)
                cx2 = min(img_w - margin_x, cx1 + col_w)
                for r in range(rows):
                    cy1 = int(round(margin_y + r * row_step + (row_step - row_h) * 0.5))
                    cy2 = min(img_h - 2, cy1 + row_h)
                    chambers.append({
                        "chamber_id": ch_idx,
                        "roi": (int(cx1), int(cy1), int(cx2), int(cy2)),
                        "row": r,
                        "col": c
                    })
                    ch_idx += 1
        else:
            for r in range(rows):
                cy1 = int(round(margin_y + r * row_step + (row_step - row_h) * 0.5))
                cy2 = min(img_h - 2, cy1 + row_h)
                for c in range(cols):
                    cx1 = margin_x + c * (col_w + center_divider_gap)
                    cx2 = min(img_w - margin_x, cx1 + col_w)
                    chambers.append({
                        "chamber_id": ch_idx,
                        "roi": (int(cx1), int(cy1), int(cx2), int(cy2)),
                        "row": r,
                        "col": c
                    })
                    ch_idx += 1
                    
        return chambers


class RobustGridAligner:
    """
    Estimates all chamber bounding boxes from a single user-drawn first ROI (CH 1, top-left),
    adapting to arbitrary row counts, column counts, center dividers, and mechanical tilt.
    """
class RobustGridAligner:

    @staticmethod
    def estimate_chambers_from_first_roi(
        frame_bgr: np.ndarray,
        first_box: Tuple[int, int, int, int],
        rows: int = 4,
        cols: int = 2,
        order: str = "column_first"
    ) -> List[Tuple[int, int, int, int]]:
        img_h, img_w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        x1, y1, x2, y2 = first_box
        box_w = x2 - x1
        box_h = y2 - y1

        # -----------------------------------------------------------
        # 1. 垂直方向（Y轴）：利用 CH1 模板锁定 4 行试管的绝对行中心
        # -----------------------------------------------------------
        # 取 CH1 内部作为模板（避开边缘反光）
        tpl_margin_y = int(box_h * 0.15)
        tpl_y = gray[y1 + tpl_margin_y : y2 - tpl_margin_y, x1:x2]

        # 仅在左侧列所在的垂直带内搜索
        res_y = cv2.matchTemplate(gray[:, x1:x2], tpl_y, cv2.TM_CCOEFF_NORMED)
        y_corr = np.max(res_y, axis=1)

        # 逐行寻找峰值（锁定每一行的顶部 y 坐标）
        row_y = [y1]
        step_min = int(box_h * 1.05)  # 最小行跨度（管高 + 间隙）
        curr_y = y1

        for _ in range(1, rows):
            search_start = curr_y + step_min
            search_end = min(len(y_corr), search_start + int(box_h * 0.5))
            if search_end > search_start:
                peak_offset = int(np.argmax(y_corr[search_start:search_end]))
                best_y = search_start + peak_offset
                row_y.append(best_y)
                curr_y = best_y
            else:
                # 保底：若超出画面则使用固定步长
                pitch_y = (row_y[-1] - row_y[0]) / (len(row_y) - 1)
                row_y.append(int(curr_y + pitch_y))

        # -----------------------------------------------------------
        # 2. 水平方向（X轴）：利用中央暗槽的精确几何边界定位右列
        # -----------------------------------------------------------
        # 在 CH1 右侧到画面 65% 区域内探测中央暗槽的“谷底”和“宽度”
        seam_search_x1 = x2 - 10
        seam_search_x2 = min(img_w, x2 + int(box_w * 0.4))
        
        # 截取中部横带计算垂直投影（使用所有已探测试管的高度范围，提升抗噪能力）
        sample_y1, sample_y2 = row_y[0], row_y[-1] + box_h
        vertical_strip = gray[sample_y1:sample_y2, seam_search_x1:seam_search_x2]
        col_prof = np.mean(vertical_strip, axis=0)  # 暗槽表现为一个深谷

        # 寻找暗槽两侧壁边缘：对投影求导找“最暗谷底”及其右侧的“上升跳变沿”
        smooth_prof = cv2.GaussianBlur(col_prof.reshape(1, -1), (1, 9), 0).ravel()
        valley_rel_x = int(np.argmin(smooth_prof))
        
        # 寻找暗槽右边界（谷底右侧梯度最大的上升沿，即右列试管左边缘）
        grad_prof = np.gradient(smooth_prof)
        search_right_edge = grad_prof[valley_rel_x:]
        if len(search_right_edge) > 0 and np.max(search_right_edge) > 0:
            right_edge_rel = valley_rel_x + int(np.argmax(search_right_edge))
            col2_x1 = seam_search_x1 + right_edge_rel
        else:
            # 保底回退：若未测到上升沿，使用暗槽中心对称
            seam_center = seam_search_x1 + valley_rel_x
            gap = max(6, (seam_center - x2) * 2)
            col2_x1 = x2 + gap

        # -----------------------------------------------------------
        # 3. 刚性装配网格（严格锁定尺寸，消灭边缘漂移）
        # -----------------------------------------------------------
        # 左右两列的 X 起始点
        cols_x = [x1, col2_x1]

        grid = []
        for r in range(rows):
            r_boxes = []
            for c in range(cols):
                bx1 = cols_x[c]
                by1 = row_y[r]
                bx2 = bx1 + box_w
                by2 = by1 + box_h
                r_boxes.append((int(bx1), int(by1), int(bx2), int(by2)))
            grid.append(r_boxes)

        # -----------------------------------------------------------
        # 4. 按指定排布顺序输出
        # -----------------------------------------------------------
        chambers = []
        if order == "column_first":
            for c in range(cols):
                for r in range(rows):
                    chambers.append(grid[r][c])
        else:
            for r in range(rows):
                for c in range(cols):
                    chambers.append(grid[r][c])

        return chambers


class RobustFlyTracker:
    """
    Centroid tracking with Darkness Mass Score, boundary rejection, and 1-based chamber indexing.
    """

    def __init__(
        self,
        chamber_rois: List[Tuple[int, int, int, int]],
        chamber_ids: Optional[List[int]] = None,
        median_bg: Optional[np.ndarray] = None,
        diff_thresh: int = 14,
        min_fly_area: Optional[int] = None,
        max_fly_area: Optional[int] = None,
        num_workers: Optional[int] = None
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

        k_size = int(np.clip(avg_h * 0.22, 7, 31))
        if k_size % 2 == 0:
            k_size += 1
        self.kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

        self.last_known_pos: Dict[int, Optional[Tuple[float, float]]] = {cid: None for cid in self.chamber_ids}
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {cid: [] for cid in self.chamber_ids}
        self.last_debug_masks: Dict[int, np.ndarray] = {}

        workers = num_workers or min(16, max(1, len(self.chambers)))
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def close(self):
        """Releases thread pool resources."""
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=False)

    def __del__(self):
        self.close()

    def _extract_fly_candidate(
        self,
        chamber_crop: np.ndarray,
        bg_crop: Optional[np.ndarray],
        cid: int,
        roi_box: Tuple[int, int, int, int]
    ) -> Tuple[Optional[dict], np.ndarray]:
        if chamber_crop.size == 0:
            return None, np.zeros((1, 1), dtype=np.uint8)
        h, w = chamber_crop.shape[:2]
        if h < 5 or w < 10:
            return None, np.zeros((1, 1), dtype=np.uint8)

        # Border protection mask (reject outer boundary seam artifacts)
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[1:h - 1, 1:w - 1] = 255

        # Background subtraction for dark moving object extraction
        if bg_crop is not None and bg_crop.shape == chamber_crop.shape:
            diff = cv2.subtract(bg_crop, chamber_crop)
        else:
            diff = cv2.bitwise_not(chamber_crop)

        blurred = cv2.GaussianBlur(diff, (3, 3), 0)
        _, mask = cv2.threshold(blurred, self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, border_mask)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        last_pos = self.last_known_pos.get(cid, None)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]

                # Continuity weighting based on distance to previous frame
                s_dist = 1.0
                if last_pos is not None:
                    abs_cx = roi_box[0] + cx
                    abs_cy = roi_box[1] + cy
                    d = np.hypot(abs_cx - last_pos[0], abs_cy - last_pos[1])
                    s_dist = np.exp(-d / 70.0)

                candidates.append({
                    "local_pos": (cx, cy),
                    "area": area,
                    "score": area * s_dist
                })

        best = max(candidates, key=lambda c: c["score"]) if candidates else None
        return best, mask

    def _extract_single_chamber_task(
        self,
        args: Tuple[int, int, Tuple[int, int, int, int], np.ndarray, Optional[np.ndarray]]
    ) -> Tuple[int, int, Tuple[int, int, int, int], Optional[dict], np.ndarray]:
        idx, cid, (x1, y1, x2, y2), frame_gray, bg_img = args
        img_h, img_w = frame_gray.shape[:2]
        gx1, gy1 = max(0, x1), max(0, y1)
        gx2, gy2 = min(img_w, x2), min(img_h, y2)

        chamber_crop = frame_gray[gy1:gy2, gx1:gx2]
        bg_crop = bg_img[gy1:gy2, gx1:gx2] if bg_img is not None else None

        candidate, mask = self._extract_fly_candidate(chamber_crop, bg_crop, cid, (gx1, gy1, gx2, gy2))
        return idx, cid, (gx1, gy1, gx2, gy2), candidate, mask

    def process_frame(
        self,
        frame_gray: np.ndarray,
        frame_idx: int,
        timestamp_s: float = 0.0
    ) -> Tuple[List[dict], Dict[int, Optional[dict]]]:
        tasks = []
        for idx, (x1, y1, x2, y2) in enumerate(self.chambers):
            cid = self.chamber_ids[idx]
            tasks.append((idx, cid, (x1, y1, x2, y2), frame_gray, self.median_bg))

        if len(tasks) > 1 and hasattr(self, "executor") and self.executor:
            results = list(self.executor.map(self._extract_single_chamber_task, tasks))
        else:
            results = [self._extract_single_chamber_task(t) for t in tasks]

        records = []
        frame_detections = {}

        for idx, cid, (gx1, gy1, gx2, gy2), candidate, mask in results:
            orig_x1, orig_y1, orig_x2, orig_y2 = self.chambers[idx]
            self.last_debug_masks[cid] = mask
            frame_detections[cid] = candidate

            cw = max(1.0, float(gx2 - gx1))
            ch = max(1.0, float(gy2 - gy1))

            if candidate is not None:
                local_x, local_y = candidate["local_pos"]
                abs_x = gx1 + local_x
                abs_y = gy1 + local_y
                self.last_known_pos[cid] = (abs_x, abs_y)

                self.trajectory_history[cid].append((abs_x, abs_y))
                if len(self.trajectory_history[cid]) > 35:
                    self.trajectory_history[cid].pop(0)

                records.append({
                    "chamber_id": cid,
                    "fly_id": 0,
                    "frame": frame_idx,
                    "timestamp_s": round(timestamp_s, 3),
                    "x_px": round(abs_x, 2),
                    "y_px": round(abs_y, 2),
                    "norm_x": round(np.clip((abs_x - gx1) / cw, 0.0, 1.0), 4),
                    "norm_y": round(np.clip((abs_y - gy1) / ch, 0.0, 1.0), 4),
                    "roi_x1": orig_x1,
                    "roi_y1": orig_y1,
                    "roi_x2": orig_x2,
                    "roi_y2": orig_y2,
                    "area": round(candidate["area"], 1),
                    "is_interpolated": 0
                })
            else:
                records.append({
                    "chamber_id": cid,
                    "fly_id": 0,
                    "frame": frame_idx,
                    "timestamp_s": round(timestamp_s, 3),
                    "x_px": np.nan,
                    "y_px": np.nan,
                    "norm_x": np.nan,
                    "norm_y": np.nan,
                    "roi_x1": orig_x1,
                    "roi_y1": orig_y1,
                    "roi_x2": orig_x2,
                    "roi_y2": orig_y2,
                    "area": np.nan,
                    "is_interpolated": 0
                })

        return records, frame_detections


def post_process_dynamic_interpolate(
    records: List[dict],
    fps: float = 30.0,
    max_gap_frames: int = 30,
    max_speed_px_per_sec: float = 400.0
) -> List[dict]:
    """
    Recovers missing tracking coordinates with linear kinematics interpolation.
    """
    by_chamber = defaultdict(list)
    for r in records:
        by_chamber[r["chamber_id"]].append(dict(r))

    processed_all = []
    for cid, ch_records in by_chamber.items():
        n = len(ch_records)
        i = 0
        while i < n:
            if np.isnan(ch_records[i]["x_px"]):
                j = i
                while j < n and np.isnan(ch_records[j]["x_px"]):
                    j += 1
                gap_len = j - i

                if 0 < i and j < n and gap_len <= max_gap_frames:
                    prev_r = ch_records[i - 1]
                    next_r = ch_records[j]
                    
                    t_prev = prev_r.get("timestamp_s", prev_r["frame"] / fps)
                    t_next = next_r.get("timestamp_s", next_r["frame"] / fps)
                    dt = max(0.001, t_next - t_prev)

                    dx = next_r["x_px"] - prev_r["x_px"]
                    dy = next_r["y_px"] - prev_r["y_px"]
                    dist = np.hypot(dx, dy)
                    speed = dist / dt

                    if speed <= max_speed_px_per_sec:
                        for step, k in enumerate(range(i, j), start=1):
                            alpha = step / (gap_len + 1)
                            ch_records[k]["x_px"] = round(prev_r["x_px"] + alpha * dx, 2)
                            ch_records[k]["y_px"] = round(prev_r["y_px"] + alpha * dy, 2)
                            ch_records[k]["norm_x"] = round(np.clip(prev_r["norm_x"] + alpha * (next_r["norm_x"] - prev_r["norm_x"]), 0.0, 1.0), 4)
                            ch_records[k]["norm_y"] = round(np.clip(prev_r["norm_y"] + alpha * (next_r["norm_y"] - prev_r["norm_y"]), 0.0, 1.0), 4)
                            if "timestamp_s" not in ch_records[k] or ch_records[k]["timestamp_s"] == 0.0:
                                ch_records[k]["timestamp_s"] = round(t_prev + alpha * dt, 3)
                            ch_records[k]["is_interpolated"] = 1
                i = j
            else:
                if "is_interpolated" not in ch_records[i]:
                    ch_records[i]["is_interpolated"] = 0
                i += 1

        processed_all.extend(ch_records)

    processed_all.sort(key=lambda r: (r["frame"], r["chamber_id"]))
    return processed_all


class FlyVisionTracker:
    """
    Facade class for video tracking with support for arbitrary 1-based chamber counts
    and multi-core parallel ROI extraction.
    """
    def __init__(
        self,
        chamber_rois: List[Any],
        chamber_ids: Optional[List[int]] = None,
        diff_thresh: int = 14
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
        self.median_bg = None

    def build_background(self, video_path: str, num_samples: int = 60) -> np.ndarray:
        self.median_bg = build_median_background(video_path, num_samples=num_samples)
        return self.median_bg

    def track_video(
        self,
        video_path: str,
        interpolate_gaps: bool = True,
        progress_callback=None
    ) -> pd.DataFrame:
        total_frames, fps, _, _ = get_video_metadata(video_path)
        if self.median_bg is None:
            self.build_background(video_path)

        tracker = RobustFlyTracker(
            self.chambers,
            chamber_ids=self.chamber_ids,
            median_bg=self.median_bg,
            diff_thresh=self.diff_thresh
        )
        cap = cv2.VideoCapture(video_path)
        all_records = []
        f_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if timestamp_ms is not None and timestamp_ms > 0:
                    timestamp_s = timestamp_ms / 1000.0
                else:
                    timestamp_s = f_idx / fps

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_records, _ = tracker.process_frame(gray, f_idx, timestamp_s=timestamp_s)
                all_records.extend(frame_records)
                f_idx += 1

                if progress_callback and f_idx % 100 == 0:
                    progress_callback(f_idx, total_frames)
        finally:
            cap.release()
            tracker.close()

        if interpolate_gaps:
            all_records = post_process_dynamic_interpolate(all_records, fps=fps)

        df = pd.DataFrame(all_records)
        return df


class Interactive8ChamberCalibrator:
    """
    Adaptive Auto-Snap: Centers vertically based on illumination boundaries and
    extends horizontally across circular inlet holes and wire mesh.
    """
    def __init__(self, sample_frame: np.ndarray, initial_boxes: List[Tuple[int, int, int, int]]):
        self.orig_img = sample_frame.copy()
        self.img_h, self.img_w = sample_frame.shape[:2]
        self.boxes = [list(b) for b in initial_boxes]

    def auto_snap_chambers(self):
        gray = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
        for idx in range(len(self.boxes)):
            bx1, by1, bx2, by2 = self.boxes[idx]
            bw = bx2 - bx1
            bh = by2 - by1

            # 1. Vertical height (Y-axis) snap to bright tube lumen
            sy1 = max(0, by1 - int(bh * 0.4))
            sy2 = min(self.img_h, by2 + int(bh * 0.4))
            v_crop = gray[sy1:sy2, bx1:bx2]
            if v_crop.size > 0:
                v_prof = np.mean(v_crop, axis=1)
                v_smooth = cv2.GaussianBlur(v_prof.reshape(-1, 1), (15, 1), 0).ravel()
                bright_thresh = np.percentile(v_smooth, 35)
                valid_y = np.where(v_smooth >= bright_thresh)[0]
                if len(valid_y) > 10:
                    new_y1 = sy1 + valid_y[0] - 2
                    new_y2 = sy1 + valid_y[-1] + 2
                    if new_y2 - new_y1 >= 25:
                        self.boxes[idx][1] = max(0, int(new_y1))
                        self.boxes[idx][3] = min(self.img_h, int(new_y2))

            # 2. Horizontal span (X-axis) extension to cover circular holes and wire mesh
            cur_y1, cur_y2 = self.boxes[idx][1], self.boxes[idx][3]
            h_crop = gray[cur_y1:cur_y2, max(0, bx1 - 20):min(self.img_w, bx2 + 30)]
            if h_crop.size > 0:
                self.boxes[idx][2] = min(self.img_w - 5, self.boxes[idx][2] + 8)
                self.boxes[idx][0] = max(5, self.boxes[idx][0] - 5)

    def run(self) -> List[Tuple[int, int, int, int]]:
        self.auto_snap_chambers()
        return [tuple(b) for b in self.boxes]
