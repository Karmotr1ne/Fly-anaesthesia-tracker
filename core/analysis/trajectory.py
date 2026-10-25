"""Shared array operations for trajectory post-processing."""
from collections import deque
import numpy as np


def interpolate_gaps(x, y, times, max_gap, max_speed, flags, ratios=(), cancel=None,
                     segments=None, protected=None):
    """Interpolate bounded x gaps using time weights and existing rounding rules."""
    index, count = 0, len(x)
    while index < count:
        if cancel and cancel():
            raise InterruptedError("Interpolation cancelled.")
        if not np.isnan(x[index]):
            index += 1
            continue
        end = index
        while end < count and np.isnan(x[end]):
            end += 1
        length = end - index
        permitted = (protected is None or not np.any(protected[index:end]))
        same_segment = (segments is None or (index > 0 and end < count and
                        np.all(segments[index - 1:end + 1] == segments[index - 1])))
        if index and end < count and length <= max_gap and permitted and same_segment:
            duration = times[end] - times[index - 1]
            if np.isfinite(duration) and duration > 0:
                dx, dy = x[end] - x[index - 1], y[end] - y[index - 1]
                speed = np.hypot(dx, dy) / max(.001, duration)
                weights = (times[index:end] - times[index - 1]) / duration
                if speed <= max_speed and np.all(np.isfinite(weights) & (weights > 0) & (weights < 1)):
                    for offset, alpha in enumerate(weights, index):
                        x[offset] = round(x[index - 1] + alpha * dx, 2)
                        y[offset] = round(y[index - 1] + alpha * dy, 2)
                        for values in ratios:
                            values[offset] = round(np.clip(values[index - 1] + alpha *
                                (values[end] - values[index - 1]), 0., 1.), 4)
                        flags[offset] = 1
        index = end


def fill_midpoints_array(values):
    """Fill missing runs in place without iterating over every valid sample."""
    missing = np.isnan(values)
    if not missing.any():
        return values
    if missing.all():
        values[:] = 0.
        return values
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0:
            value = values[end]
        elif end == len(values):
            value = values[start - 1]
        else:
            value = (values[start - 1] + values[end]) / 2.
        values[start:end] = value
    return values


def forward_max(values, window):
    """Return exact truncated forward maxima, propagating NaNs like np.max."""
    result = np.empty(len(values), dtype=float)
    queue, missing = deque(), deque()
    for index in range(len(values) - 1, -1, -1):
        limit = index + window
        while queue and queue[0] >= limit:
            queue.popleft()
        while missing and missing[0] >= limit:
            missing.popleft()
        if np.isnan(values[index]):
            missing.append(index)
        else:
            while queue and values[queue[-1]] <= values[index]:
                queue.pop()
            queue.append(index)
        result[index] = np.nan if missing else values[queue[0]]
    return result
