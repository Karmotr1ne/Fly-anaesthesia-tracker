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
    @classmethod
    def snap_all_boxes(
        cls,
        frame_bgr: np.ndarray,
        boxes: List[List[int]],
        rows: int = 4,
        cols: int = 2
    ) -> List[List[int]]:
        """
        基于 1D 亮度投影与隔缝极小值对已有 boxes 进行精准微调吸附：
        1. 纵向（Y轴）：利用管腔亮带与上下暗缝的阶跃精确定位管顶与管底；
        2. 横向（X轴）：列内中位数严格共线，杜绝单行漂移与宽度发散；
        3. 严格禁止向外膨胀超出原始框。
        """
        if frame_bgr is None or not boxes:
            return boxes

        img_h, img_w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        n_boxes = len(boxes)

        refined_boxes = []

        # -----------------------------------------------------------
        # 1. 逐个 Box 纵向（Y轴）内壁吸附
        # -----------------------------------------------------------
        for i in range(n_boxes):
            bx1, by1, bx2, by2 = [int(v) for v in boxes[i]]
            bw = bx2 - bx1
            bh = by2 - by1

            # 取纯净中段（避开端头加药孔和左侧刻度），提取纵向亮度剖面
            core_x1 = max(0, bx1 + int(bw * 0.25))
            core_x2 = min(img_w, bx1 + int(bw * 0.65))

            if core_x2 > core_x1 and bh > 10:
                # 垂直方向在当前框上下适度延展一点（±15%）观察完整的黑-白-黑分布
                pad_y = int(bh * 0.15)
                roi_y1 = max(0, by1 - pad_y)
                roi_y2 = min(img_h, by2 + pad_y)

                strip = gray[roi_y1:roi_y2, core_x1:core_x2]
                vert_prof = np.mean(strip, axis=1)
                # 平滑曲线
                smooth_prof = cv2.GaussianBlur(vert_prof.reshape(-1, 1), (1, 9), 0).ravel()

                # 求一阶梯度寻找上下黑白交界边缘
                grad_y = np.gradient(smooth_prof)

                # 上边界：寻找进入管腔的上升沿（梯度最大正值）
                # 搜索范围锁定在当前 by1 对应局部区域的 ±12px 内
                local_by1 = by1 - roi_y1
                s_top_1 = max(0, local_by1 - 12)
                s_top_2 = min(len(grad_y), local_by1 + 15)
                if s_top_2 > s_top_1:
                    ny1 = roi_y1 + s_top_1 + int(np.argmax(grad_y[s_top_1:s_top_2]))
                else:
                    ny1 = by1

                # 下边界：寻找离开管腔的下降沿（梯度最大负值，即 -grad_y 最大）
                local_by2 = by2 - roi_y1
                s_bot_1 = max(0, local_by2 - 15)
                s_bot_2 = min(len(grad_y), local_by2 + 12)
                if s_bot_2 > s_bot_1:
                    ny2 = roi_y1 + s_bot_1 + int(np.argmin(grad_y[s_bot_1:s_bot_2]))
                else:
                    ny2 = by2

                # 保护：严禁向外膨胀超过原始框 2px，防止越吸越大
                final_y1 = max(by1 - 1, ny1)
                final_y2 = min(by2 + 1, ny2)

                # 高度健康度校验
                if (final_y2 - final_y1) < 15:
                    final_y1, final_y2 = by1, by2
            else:
                final_y1, final_y2 = by1, by2

            refined_boxes.append([bx1, int(final_y1), bx2, int(final_y2)])

        # -----------------------------------------------------------
        # 2. 横向（X轴）网格刚性约束：整列严格共线对齐
        # -----------------------------------------------------------
        for c in range(cols):
            # 获取当前列的所有框索引
            col_indices = [c * rows + r for r in range(rows) if (c * rows + r) < n_boxes]
            if not col_indices:
                continue

            # 使用中位数对齐整列的 X1 和 X2，彻底消除单行（如第2行）偏离
            med_x1 = int(np.median([refined_boxes[idx][0] for idx in col_indices]))
            med_x2 = int(np.median([refined_boxes[idx][2] for idx in col_indices]))

            for idx in col_indices:
                refined_boxes[idx][0] = med_x1
                refined_boxes[idx][2] = med_x2

        return refined_boxes
    
    @staticmethod
    def estimate_chambers_from_first_roi(
        frame_bgr: np.ndarray,
        first_box: Tuple[int, int, int, int],
        rows: int = 4,
        cols: int = 2,
        order: str = "column_first"
    ) -> List[Tuple[int, int, int, int]]:
        """
        基于第一个腔室(CH1)推断全阵列腔室。
        采用纯净带垂直/水平投影定位，杜绝模板匹配平台效应与累积漂移。
        具备百分之百返回非空列表的保底保障。
        """
        if frame_bgr is None or len(first_box) != 4:
            return []

        img_h, img_w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr

        x1, y1, x2, y2 = [int(v) for v in first_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        box_w = max(20, x2 - x1)
        box_h = max(15, y2 - y1)

        # -----------------------------------------------------------
        # 0. 几何理论默认值（保底安全基线）
        # -----------------------------------------------------------
        # 图像中管间缝隙约为 20~30px，行跨度一般约为管高的 1.10 ~ 1.15 倍
        fallback_pitch_y = int(box_h * 1.12)
        fallback_col2_gap = int(box_w * 0.05)  # 两列中间中缝估算

        try:
            # -------------------------------------------------------
            # 1. 垂直方向（Y轴）：利用管身亮带中心投影锁定全局周期行距
            # -------------------------------------------------------
            # 仅截取 CH1 纯净中段（20%~60% 宽度），避开右端圆孔与左端阴影
            strip_x1 = int(x1 + box_w * 0.20)
            strip_x2 = int(x1 + box_w * 0.60)
            strip_x1, strip_x2 = max(0, strip_x1), min(img_w, strip_x2)

            if strip_x2 > strip_x1:
                # 沿 X 轴求均值，压成一条从上至下的亮度纵向曲线 I(y)
                vert_profile = np.mean(gray[:, strip_x1:strip_x2], axis=1)
                # 使用大核高斯滤波消除果蝇活动黑点与反光高频噪波
                smooth_prof = cv2.GaussianBlur(vert_profile.reshape(-1, 1), (1, 21), 0).ravel()

                ch1_center_y = (y1 + y2) // 2
                detected_centers = [ch1_center_y]
                curr_c = ch1_center_y
                est_pitch = fallback_pitch_y

                # 逐行向下在预期窗口内寻找下一个高亮管身中心
                for r in range(1, rows):
                    expected_c = curr_c + est_pitch
                    # 搜索窗口限制在预期位置的上下 20% 管高内
                    win_start = max(0, expected_c - int(box_h * 0.20))
                    win_end = min(img_h, expected_c + int(box_h * 0.20))

                    if win_end > win_start:
                        peak_offset = int(np.argmax(smooth_prof[win_start:win_end]))
                        best_c = win_start + peak_offset
                        detected_centers.append(best_c)
                        # 动态微调后续预测步长
                        if best_c > curr_c:
                            est_pitch = best_c - curr_c
                        curr_c = best_c
                    else:
                        detected_centers.append(expected_c)
                        curr_c = expected_c

                # 利用模具刚性先验：使用中位数步长统一所有行，杜绝第 2 行等单行漂移
                pitches = [detected_centers[i] - detected_centers[i - 1] for i in range(1, len(detected_centers))]
                valid_pitches = [p for p in pitches if int(box_h * 0.95) <= p <= int(box_h * 1.4)]
                median_pitch = int(np.median(valid_pitches)) if valid_pitches else fallback_pitch_y

                # 重新以刚性公式精确生成各行 Y 起始点
                row_y = [int(ch1_center_y - box_h // 2 + r * median_pitch) for r in range(rows)]
            else:
                row_y = [int(y1 + r * fallback_pitch_y) for r in range(rows)]

            # -------------------------------------------------------
            # 2. 水平方向（X轴）：利用中央黑色竖隔梁定位右列
            # -------------------------------------------------------
            # 探测中央黑色立梁（在 CH1 右边界附近往右一段区域内）
            seam_search_x1 = max(0, x2 - 10)
            seam_search_x2 = min(img_w, x2 + int(box_w * 0.25))

            sample_y1 = max(0, row_y[0])
            sample_y2 = min(img_h, row_y[-1] + box_h)

            col2_x1 = x2 + fallback_col2_gap
            if (seam_search_x2 > seam_search_x1) and (sample_y2 > sample_y1):
                vert_seam_strip = gray[sample_y1:sample_y2, seam_search_x1:seam_search_x2]
                col_prof = np.mean(vert_seam_strip, axis=0)
                smooth_col = cv2.GaussianBlur(col_prof.reshape(1, -1), (1, 11), 0).ravel()

                # 找到中央隔梁最暗处（深谷）
                valley_rel_x = int(np.argmin(smooth_col))

                # 隔梁右侧通常是由暗转亮的陡峭正跳变（右列试管左内沿）
                grad_col = np.gradient(smooth_col)
                search_right = grad_col[valley_rel_x:]
                if len(search_right) > 0 and np.max(search_right) > 0:
                    edge_offset = valley_rel_x + int(np.argmax(search_right))
                    col2_x1 = seam_search_x1 + edge_offset

            cols_x = [x1, col2_x1]

        except Exception:
            # 发生任何未预料异常，平稳回退到等距几何网格
            row_y = [int(y1 + r * fallback_pitch_y) for r in range(rows)]
            cols_x = [x1, x2 + fallback_col2_gap]

        # -----------------------------------------------------------
        # 3. 刚性装配网格并做图像边界裁剪
        # -----------------------------------------------------------
        grid = []
        for r in range(rows):
            r_boxes = []
            for c in range(cols):
                bx1 = cols_x[c] if c < len(cols_x) else (x1 + c * (box_w + fallback_col2_gap))
                by1 = row_y[r]
                bx2 = bx1 + box_w
                by2 = by1 + box_h

                # 边界保护
                cbx1 = int(np.clip(bx1, 0, img_w - 10))
                cby1 = int(np.clip(by1, 0, img_h - 10))
                cbx2 = int(np.clip(bx2, cbx1 + 10, img_w))
                cby2 = int(np.clip(by2, cby1 + 10, img_h))

                r_boxes.append((cbx1, cby1, cbx2, cby2))
            grid.append(r_boxes)

        # -----------------------------------------------------------
        # 4. 按序输出，保证 100% 返回 List[Tuple]
        # -----------------------------------------------------------
        chambers: List[Tuple[int, int, int, int]] = []
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
    集成了 CLAHE 局部增强、双尺度黑帽变换与 OBB 几何过滤的果蝇高精度追踪器
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

        # 基准面积动态估算
        self.target_fly_area = max(60.0, avg_h * avg_h * 0.08)
        self.min_area = min_fly_area if min_fly_area is not None else max(15.0, self.target_fly_area * 0.15)
        self.max_area = max_fly_area if max_fly_area is not None else max(400.0, self.target_fly_area * 6.0)

        # 1. 初始化 CLAHE 算子（限制对比度局部自适应直方图均衡）
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        # 2. 初始化双尺度黑帽核：小核适配微弱肢体/幼小体态，大核覆盖躯干
        k_small = max(5, int(avg_h * 0.10) | 1)
        k_large = max(11, int(avg_h * 0.25) | 1)
        self.kernel_bh_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))
        self.kernel_bh_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_large, k_large))

        # 状态记录
        self.last_known_pos: Dict[int, Optional[Tuple[float, float]]] = {cid: None for cid in self.chamber_ids}
        self.trajectory_history: Dict[int, List[Tuple[float, float]]] = {cid: [] for cid in self.chamber_ids}
        self.last_debug_masks: Dict[int, np.ndarray] = {}

        workers = num_workers or min(16, max(1, len(self.chambers)))
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def close(self):
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

        # 管腔内壁边缘保护掩码（内缩 1~2px，屏蔽管壁死角反射条纹）
        border_mask = np.zeros((h, w), dtype=np.uint8)
        pad_y = max(1, int(h * 0.04))
        pad_x = max(1, int(w * 0.02))
        border_mask[pad_y : h - pad_y, pad_x : w - pad_x] = 255

        # -----------------------------------------------------------------
        # 步骤 1: CLAHE 局部对比度增强（消除暗区低动态范围缺陷）
        # -----------------------------------------------------------------
        chamber_enhanced = self.clahe.apply(chamber_crop)

        # -----------------------------------------------------------------
        # 步骤 2: 双尺度黑帽变换（滤除平缓暗阴影，高亮细小暗目标）
        # -----------------------------------------------------------------
        bh_s = cv2.morphologyEx(chamber_enhanced, cv2.MORPH_BLACKHAT, self.kernel_bh_small)
        bh_l = cv2.morphologyEx(chamber_enhanced, cv2.MORPH_BLACKHAT, self.kernel_bh_large)
        # 加权融合：小核保证轮廓边缘锐利，大核保证虫体中心饱满
        bh_energy = cv2.addWeighted(bh_s, 0.6, bh_l, 0.4, 0.0)

        # -----------------------------------------------------------------
        # 步骤 3: 能量图加权融合（黑帽暗斑能量 + 背景差分流）
        # -----------------------------------------------------------------
        if bg_crop is not None and bg_crop.shape == chamber_crop.shape:
            bg_enhanced = self.clahe.apply(bg_crop)
            diff_bg = cv2.subtract(bg_enhanced, chamber_enhanced)
        else:
            diff_bg = cv2.bitwise_not(chamber_enhanced)

        # 50% 差分流 + 50% 黑帽流
        fused_diff = cv2.addWeighted(diff_bg, 0.5, bh_energy, 0.5, 0.0)

        # -----------------------------------------------------------------
        # 步骤 4: 滤波二值化与形态学闭合
        # -----------------------------------------------------------------
        blurred = cv2.GaussianBlur(fused_diff, (3, 3), 0)
        _, mask = cv2.threshold(blurred, self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, border_mask)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # -----------------------------------------------------------------
        # 步骤 5: 提取连通域，利用 OBB 几何约束过滤假边缘并提取参数
        # -----------------------------------------------------------------
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        last_pos = self.last_known_pos.get(cid, None)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area <= area <= self.max_area:
                # 纯计算几何求解有向最小外接矩形 (OBB)
                # rect 返回: ((center_x, center_y), (width, height), angle_deg)
                rect = cv2.minAreaRect(cnt)
                (cx, cy), (rw, rh), angle = rect

                if rw <= 0 or rh <= 0:
                    continue

                major_axis = max(rw, rh)  # 果蝇体长 (长轴)
                minor_axis = min(rw, rh)  # 果蝇体宽 (短轴)
                aspect_ratio = major_axis / max(1e-3, minor_axis)

                # OBB 几何准则过滤：
                # 1. 长宽比过滤：果蝇正常爬行长宽比在 1.4 ~ 4.2 之间；剔除接缝条纹 (> 4.5) 或极扁噪斑
                if aspect_ratio < 1.3 or aspect_ratio > 4.5:
                    continue

                # 2. 矩形填充率 (Extent) 过滤：果蝇椭圆躯干占 OBB 面积比例通常在 0.5 ~ 0.9 之间
                obb_area = major_axis * minor_axis
                extent = area / obb_area if obb_area > 0 else 0
                if extent < 0.40:
                    continue

                # 计算时序连续性衰减打分
                s_dist = 1.0
                if last_pos is not None:
                    abs_cx = roi_box[0] + cx
                    abs_cy = roi_box[1] + cy
                    d = np.hypot(abs_cx - last_pos[0], abs_cy - last_pos[1])
                    s_dist = np.exp(-d / 70.0)

                # 提取 OBB 4 个顶点坐标备用 (顺时针/逆时针矩形角点)
                box_pts = cv2.boxPoints(rect)

                candidates.append({
                    "local_pos": (cx, cy),
                    "area": area,
                    "major_len": major_axis,
                    "minor_len": minor_axis,
                    "aspect_ratio": aspect_ratio,
                    "angle": angle,
                    "obb_box": box_pts,
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
                    # 新增由 OBB 提取出的真实物理几何字段
                    "body_len_px": round(candidate["major_len"], 2),
                    "body_width_px": round(candidate["minor_len"], 2),
                    "aspect_ratio": round(candidate["aspect_ratio"], 2),
                    "angle_deg": round(candidate["angle"], 2),
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
                    "body_len_px": np.nan,
                    "body_width_px": np.nan,
                    "aspect_ratio": np.nan,
                    "angle_deg": np.nan,
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
