"""Grid pitch estimation and confidence-gated chamber alignment."""
import math

import cv2
import numpy as np


class RobustGridAligner:
    """Anchor one grid and fit shared pitch before confidence-gated local snapping."""

    @staticmethod
    def _gradient(profile):
        profile = np.asarray(profile, dtype=np.float32)
        smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (9, 1), 0).ravel()
        return np.gradient(smoothed)

    @staticmethod
    def _edge_position(gradient, expected, radius, direction):
        left = max(0, int(math.floor(expected-radius)))
        right = min(len(gradient), int(math.ceil(expected+radius))+1)
        if right <= left:
            return None
        signed = direction*gradient[left:right]
        noise = np.median(np.abs(gradient-np.median(gradient)))
        threshold = max(.8, 4.*noise)
        if float(signed.max()) < threshold:
            return None
        indices = np.arange(left, right)
        # Prefer the nearest point on a flat edge response; repeated snaps
        # therefore do not walk along equal-strength pixels.
        merit = signed - threshold*.1*np.abs(indices-expected)/max(1., radius)
        position = int(indices[int(np.argmax(merit))])
        # Locate the peak between pixels; otherwise a half-pixel edge bias
        # becomes a systematic pitch error when fitted across distant rows.
        if 0 < position < len(gradient)-1:
            a, b, c = direction*gradient[position-1:position+2]
            curvature = a-2*b+c
            if curvature < -1e-6:
                return position + float(np.clip(.5*(a-c)/curvature, -.5, .5))
        return float(position)

    @classmethod
    def _estimate_pitch(cls, profile, origin, extent, count, fallback_ratio, maximum_ratio):
        if count <= 1:
            return float(extent)
        available = (len(profile)-origin-extent)/(count-1)
        if available < extent:
            raise ValueError('The requested grid does not fit the frame from CH1.')
        fallback = min(extent*fallback_ratio, available)
        gradient = cls._gradient(profile)
        radius = max(3., extent*.12)
        first_top = cls._edge_position(gradient, origin, radius, 1)
        first_bottom = cls._edge_position(gradient, origin+extent, radius, -1)
        if first_top is None or first_bottom is None:
            return fallback
        maximum = min(extent*maximum_ratio, available)
        if maximum < extent:
            return fallback
        pitches = np.linspace(extent, maximum, max(2, int((maximum-extent)*4)+1))
        indices = np.arange(1, count)
        top = first_top + pitches[:, None]*indices
        bottom = first_bottom + pitches[:, None]*indices
        up = np.maximum(0., np.interp(top, np.arange(len(gradient)), gradient))
        down = np.maximum(0., -np.interp(bottom, np.arange(len(gradient)), gradient))
        evidence = np.minimum(up, down)
        threshold = max(.8, 4.*float(np.median(np.abs(gradient-np.median(gradient)))))
        supported = (evidence >= threshold).sum(axis=1)
        required = max(1, int(math.ceil((count-1)*.6)))
        scores = np.minimum(evidence, max(threshold*2, min(
            gradient[int(first_top)], -gradient[int(first_bottom)])*2)).sum(axis=1)
        scores[supported < required] = -np.inf
        if not np.isfinite(scores).any():
            return fallback
        # Shared global search: no row/column prediction depends on its neighbour.
        best = np.flatnonzero(scores >= np.max(scores)-1e-6)
        pitch = float(pitches[best[np.argmin(np.abs(pitches[best]-fallback))]])
        measured = []
        for index in indices:
            a = cls._edge_position(gradient, first_top+index*pitch, max(2., extent*.04), 1)
            b = cls._edge_position(gradient, first_bottom+index*pitch, max(2., extent*.04), -1)
            if a is not None and b is not None:
                distance = ((a-first_top)+(b-first_bottom))*.5
                if abs(distance-index*pitch) <= max(2., extent*.04):
                    measured.append((index, distance))
        if len(measured) >= required:
            pitch = sum(index*distance for index,distance in measured)/sum(index*index for index,_ in measured)
        return float(np.clip(pitch, extent, maximum))

    @classmethod
    def snap_all_boxes(cls, frame_bgr, boxes, rows=4, cols=2, equal_size=True):
        """Keep CH1 fixed; move other boxes only with paired edge evidence."""
        if frame_bgr is None or not boxes:
            return boxes
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim==3 else frame_bgr
        image_height, image_width = gray.shape[:2]
        refined = []
        for index, box in enumerate(boxes):
            left, top, right, bottom = map(int, box)
            width, height = right-left, bottom-top
            if index == 0 or width < 10 or height < 15:
                refined.append([left, top, right, bottom])
                continue
            # Robust central projection suppresses isolated flies and holes.
            strip = gray[:, max(0,left+width//5):min(image_width,left+width*3//5)]
            if strip.size == 0:
                refined.append([left, top, right, bottom])
                continue
            gradient = cls._gradient(np.median(strip, axis=1))
            limit = max(2, min(16, round(height*.05)))
            a = cls._edge_position(gradient, top, limit, 1)
            b = cls._edge_position(gradient, bottom, limit, -1)
            if a is None or b is None or abs((a-top)-(b-bottom)) > limit:
                refined.append([left, top, right, bottom])
                continue
            if equal_size:
                shift = int(round(((a-top)+(b-bottom))*.5))
                shift = int(np.clip(shift, max(-limit,-top), min(limit,image_height-bottom)))
                top, bottom = top+shift, bottom+shift
            elif b-a >= 15:
                top, bottom = int(a), int(b)
            refined.append([left, top, right, bottom])
        return refined

    @classmethod
    def estimate_chambers_from_first_roi(cls, frame_bgr, first_box, rows=4, cols=2,
                                         order='column_first'):
        if frame_bgr is None or len(first_box)!=4:
            return []
        if rows < 1 or cols < 1:
            raise ValueError('Grid dimensions must be positive.')
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim==3 else frame_bgr
        image_height, image_width = gray.shape[:2]
        left, top, right, bottom = map(int, first_box)
        if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
            raise ValueError('CH1 must lie inside the video frame.')
        width, height = right-left, bottom-top
        if width < 10 or height < 15:
            raise ValueError('CH1 is too small for grid calibration.')
        row_profile = np.median(gray[:, left+width//5:left+width*3//5], axis=1)
        pitch_y = cls._estimate_pitch(row_profile, top, height, rows, 1.12, 1.4)
        # Use the interior of each predicted row, excluding chamber borders.
        bands = [gray[max(0,round(top+r*pitch_y+height*.25)):
                      min(image_height,round(top+r*pitch_y+height*.75)), :] for r in range(rows)]
        column_profile = np.median(np.concatenate(bands, axis=0), axis=0)
        pitch_x = cls._estimate_pitch(column_profile, left, width, cols, 1.05, 1.35)
        # Round absolute positions once, rather than accumulating rounded steps.
        columns = [round(left+c*pitch_x) for c in range(cols)]
        row_positions = [round(top+r*pitch_y) for r in range(rows)]
        indices = ((r,c) for c in range(cols) for r in range(rows)) if order=='column_first' else (
                   (r,c) for r in range(rows) for c in range(cols))
        return [(columns[c],row_positions[r],columns[c]+width,row_positions[r]+height)
                for r,c in indices]


