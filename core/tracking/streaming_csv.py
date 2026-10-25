"""Incremental detection archive and disk-backed, per-chamber post-processing."""
import csv
import heapq
import math
import time
from contextlib import ExitStack
from pathlib import Path
from core import schema


class DetectionCSVWriter:
    """Write each frame immediately; retain only file handles and small counters."""
    def __init__(self, archive, directory, cancelled=None):
        self.archive = Path(archive)
        self.directory = Path(directory)
        self.cancelled = cancelled
        self._stack = ExitStack()
        self._archive_stream = None
        self._archive_writer = None
        self._chambers = {}
        self.chamber_paths = {}
        self.valid_counts = {}
        self.fields = None
        self._last_flush = float('-inf')

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        self._archive_stream = self._stack.enter_context(self.archive.open('w', encoding='utf-8-sig', newline=''))
        return self

    def _check(self):
        if self.cancelled and self.cancelled():
            raise InterruptedError('Tracking CSV writing cancelled.')

    def write_frame(self, records, fps):
        self._check()
        for record in records:
            row = dict(record)
            detected = math.isfinite(row['x']) and math.isfinite(row['y']) and not row.get('is_interpolated', 0)
            row.update(fps=fps, is_detected=int(detected))
            if self.fields is None:
                self.fields = tuple(row)
                self._archive_writer = csv.writer(self._archive_stream)
                self._archive_writer.writerow([schema.label(name) for name in self.fields])
            chamber = int(row['chamber_id'])
            if chamber not in self._chambers:
                path = self.directory / f'chamber_{chamber}.csv'
                stream = self._stack.enter_context(path.open('w', encoding='utf-8-sig', newline=''))
                writer = csv.writer(stream)
                writer.writerow([schema.label(name) for name in self.fields])
                self._chambers[chamber] = (stream, writer)
                self.chamber_paths[chamber] = path
                self.valid_counts[chamber] = 0
            values = [row[name] for name in self.fields]
            values = ['' if isinstance(value, float) and not math.isfinite(value) else value for value in values]
            self._archive_writer.writerow(values)
            self._chambers[chamber][1].writerow(values)
            self.valid_counts[chamber] += int(detected)
        now = time.monotonic()
        if now-self._last_flush >= .5:
            self.flush()
            self._last_flush = now

    def flush(self):
        if self._archive_stream:
            self._archive_stream.flush()
        for stream, _ in self._chambers.values():
            stream.flush()

    def __exit__(self, *exc):
        self._stack.close()


def merge_position_csvs(sources, target, empty_header, cancelled=None):
    """Merge ordered chamber files with one buffered row per chamber."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ExitStack() as stack:
        readers = [csv.reader(stack.enter_context(Path(path).open(encoding='utf-8-sig', newline='')))
                   for path in sources]
        headers = [next(reader) for reader in readers]
        header = headers[0] if headers else list(empty_header)
        if any(value != header for value in headers):
            raise ValueError('Chamber CSV schemas differ.')
        frame_column = next(index for index,name in enumerate(header) if schema.field(name)=='frame')
        chamber_column = next(index for index,name in enumerate(header) if schema.field(name)=='chamber_id')
        stream = stack.enter_context(target.open('w', encoding='utf-8-sig', newline=''))
        writer = csv.writer(stream)
        writer.writerow(header)
        ordered = heapq.merge(*readers, key=lambda row:(int(float(row[frame_column])), int(float(row[chamber_column]))))
        for row in ordered:
            if count % 4096 == 0:
                if cancelled and cancelled():
                    raise InterruptedError('Position CSV merge cancelled.')
                stream.flush()
            writer.writerow(row)
            count += 1
    return count
