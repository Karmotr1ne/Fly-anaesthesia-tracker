"""In-memory tracking result storage and compatibility interpolation adapters."""
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd


def post_process_dynamic_interpolate(records: List[dict], fps: float = 30.0,
                                     max_gap: int = 30, gap_speed: float = 400.0) -> List[dict]:
    """Compatibility record adapter for the shared time-weighted interpolator."""
    from core.analysis.trajectory import interpolate_gaps
    by_chamber = defaultdict(list)
    for record in records:
        by_chamber[record["chamber_id"]].append(dict(record))
    processed = []
    for group in by_chamber.values():
        x = np.asarray([r['x'] for r in group], dtype=float)
        y = np.asarray([r['y'] for r in group], dtype=float)
        times = np.asarray([r.get('time', r['frame'] / fps) for r in group], dtype=float)
        ratios = [np.asarray([r.get(name, np.nan) for r in group], dtype=float)
                  for name in ('x_ratio', 'y_ratio')]
        flags = np.asarray([r.get('is_interpolated', 0) for r in group])
        missing = np.isnan(x).copy()
        interpolate_gaps(x, y, times, max_gap, gap_speed, flags, ratios,
            segments=np.asarray([r.get('segment_id', 0) for r in group]),
            protected=np.asarray([r.get('tracking_status', 'Accepted') != 'Accepted' for r in group]))
        for index, record in enumerate(group):
            if missing[index] and not np.isnan(x[index]):
                record.update(x=x[index], y=y[index], x_ratio=ratios[0][index], y_ratio=ratios[1][index])
                if 'time' not in record or record['time'] == 0.:
                    record['time'] = round(times[index], 3)
            if not np.isnan(x[index]):
                record['is_interpolated'] = flags[index]
        processed.extend(group)
    processed.sort(key=lambda record: (record['frame'], record['chamber_id']))
    return processed


class DetectionBlocks:
    """Bound temporary record dictionaries to one frame; retain typed columns."""
    def __init__(self, block_size=8192):
        self.block_size = block_size
        self.columns = {}
        self.current = {}
        self.used = 0

    def extend(self, records):
        if not records:
            return
        if not self.columns:
            for name, value in records[0].items():
                dtype = 'U16' if isinstance(value, str) else np.int64 if isinstance(value, (int, np.integer)) else np.float64
                self.columns[name] = []
                self.current[name] = np.empty(self.block_size, dtype=dtype)
        offset = 0
        while offset < len(records):
            count = min(len(records) - offset, self.block_size - self.used)
            for name, values in self.current.items():
                values[self.used:self.used + count] = [record[name] for record in records[offset:offset + count]]
            self.used += count
            offset += count
            if self.used == self.block_size:
                for name, values in self.current.items():
                    self.columns[name].append(values)
                    self.current[name] = np.empty(self.block_size, dtype=values.dtype)
                self.used = 0

    def dataframe(self):
        columns = {}
        for name, blocks in self.columns.items():
            parts = blocks + ([self.current[name][:self.used]] if self.used else [])
            columns[name] = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self.columns.clear()
        self.current.clear()
        return pd.DataFrame(columns)


def interpolate_tracking_dataframe(data, fps=30., max_gap=30, gap_speed=400.):
    """Interpolate columns directly without rebuilding per-row dictionaries."""
    from core.analysis.trajectory import interpolate_gaps
    data = data.copy()
    for _, indices in data.groupby('chamber_id', sort=False).indices.items():
        group = data.iloc[indices]
        x, y = group.x.to_numpy(copy=True), group.y.to_numpy(copy=True)
        times = group.time.to_numpy() if 'time' in group else group.frame.to_numpy() / fps
        flags = group.is_interpolated.to_numpy(copy=True)
        ratios = [group[name].to_numpy(copy=True) for name in ('x_ratio', 'y_ratio')]
        interpolate_gaps(x, y, times, max_gap, gap_speed, flags, ratios,
            segments=group.segment_id.to_numpy() if 'segment_id' in group else None,
            protected=group.tracking_status.ne('Accepted').to_numpy() if 'tracking_status' in group else None)
        for name, values in zip(('x', 'y', 'is_interpolated', 'x_ratio', 'y_ratio'), (x, y, flags, *ratios)):
            data.iloc[indices, data.columns.get_loc(name)] = values
    return data


