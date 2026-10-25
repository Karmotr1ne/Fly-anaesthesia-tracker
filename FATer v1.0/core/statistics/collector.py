"""Summary collection adapted from get summary result.py."""
import os
import re
from pathlib import Path
from core import schema
from core.session.layout import experiment_name
from core.session.file_matcher import detection_path

class SummaryCollector:
    TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")

    def __init__(self, log, cancel):
        self.log, self.cancel = log, cancel

    def _check(self):
        if self.cancel():
            raise InterruptedError("Collection cancelled. Completed outputs are retained.")

    def _log(self, message):
        self._check()
        self.log(message)

    @classmethod
    def _time_group(cls, session):
        """Group by the session start hour; unknown times follow all windows."""
        match = cls.TIMESTAMP_PATTERN.search(session)
        if match:
            clock = match.group(1).split('_')[1]
            hour, minute, second = int(clock[:2]), int(clock[2:4]), int(clock[4:])
            if hour < 24 and minute < 60 and second < 60:
                for group, (start, end) in enumerate(((8, 11), (11, 14), (14, 17))):
                    if start <= hour < end:
                        return group
        return 3

    @classmethod
    def _group_metric_columns(cls, table):
        """Return vertically stacked time-window tables with date-only headers."""
        groups = [[] for _ in range(4)]
        for session in sorted(table.columns):
            groups[cls._time_group(session)].append(session)
        parts = []
        for columns in groups:
            if not columns:
                continue
            block = table.loc[:, columns].copy()
            headers = []
            for session in columns:
                match = cls.TIMESTAMP_PATTERN.search(session)
                date = match.group(1)[:8] if match else session
                headers.append(f'{date[:4]}-{date[4:6]}-{date[6:]}' if match else date)
            block.columns = headers
            block.index.name = 'chamber_id'
            parts.append(block)
        return parts

    def scan_csvs(self, folders):
        """Yield each resolved CSV once, including overlapping source folders."""
        seen = set()
        for folder in folders:
            for current, directories, files in os.walk(folder):
                self._check()
                directories[:] = sorted(name for name in directories
                                        if not name.startswith((".work-", ".tracking-")))
                for name in sorted(files):
                    self._check()
                    if not name.lower().endswith(".csv"):
                        continue
                    path = Path(current, name)
                    identity = os.path.normcase(os.path.realpath(path))
                    if identity not in seen:
                        seen.add(identity)
                        yield path

    def compile_report(self, folders, output):
        """Export original-script metric matrices: chambers x sorted sessions."""
        import pandas as pd
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        metrics = {
            'sedation_latency': ('sedation_latency',),
            'anaesthesia_latency': ('anaesthesia_latency', 'anesthesia_latency'),
            'speed': ('baseline_speed', 'speed', 'velocity'),
        }
        collected = {name: {} for name in metrics}
        count = 0
        # Materialise the input list before writing, including when output is
        # inside a selected session. Generated matrices have no target metrics.
        sources = sorted(self.scan_csvs(folders), key=lambda path: str(path).casefold())
        for path in sources:
            self._check()
            if 'summary' not in path.name.lower():
                continue
            try:
                data = schema.read_csv(path)
            except Exception as exc:
                self._log(f'Warning: Failed to read {path.name}: {exc}')
                continue
            chamber = next((name for name in ('chamber_id', 'chamber', 'id') if name in data), None)
            if chamber is None:
                continue
            data = data.drop_duplicates(subset=[chamber], keep='first').set_index(chamber)
            identifier = self._extract_column_header_implicit(str(path))
            used = False
            for metric, candidates in metrics.items():
                column = next((name for name in candidates if name in data), None)
                if column is None:
                    continue
                unique, suffix = identifier, 1
                while unique in collected[metric]:
                    unique = f'{identifier}_{suffix}'
                    suffix += 1
                collected[metric][unique] = data[column]
                used = True
            if used:
                count += len(data)
        paths = []
        for metric, columns in collected.items():
            self._check()
            if not columns:
                continue
            table = self._group_metric_columns(pd.DataFrame(columns).sort_index())
            target = Path(self._available_output_path(str(output), f'{metric}_summary.csv'))
            temporary = target.with_suffix('.csv.tmp')
            try:
                with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
                    for index, block in enumerate(table):
                        self._check()
                        if index:
                            stream.write('\n')
                        block.to_csv(stream)
                self._check()
                temporary.replace(target)
                paths.append(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return {'mode': 'summary', 'count': count, 'sessions': len(folders),
                'path': output if paths else None, 'paths': paths}

    def copy_raw(self, paths, output):
        destination = Path(output) / "Extracted_Raw_CSVs"
        destination.mkdir(parents=True, exist_ok=True)
        count, seen = 0, set()
        for path in paths:
            self._check()
            source = detection_path(path)
            identity = os.path.normcase(os.path.realpath(source))
            if identity in seen or not source.is_file():
                continue
            seen.add(identity)
            identifier = self._extract_column_header_implicit(str(source))
            target = Path(self._available_output_path(str(destination), experiment_name(source) + "_raw.csv"))
            temporary = target.with_suffix(".csv.tmp")
            try:
                with source.open("rb") as src, temporary.open("wb") as dst:
                    while True:
                        self._check()
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        dst.write(block)
                self._check()
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            count += 1
        return {"mode": "raw", "count": count, "path": destination}

    def _extract_column_header_implicit(self, file_path):
        """
        Read the nearest ancestor folder containing YYYYMMDD_HHMMSS.
        Preserve the original script session prefix and folder fallback.
        """
        from datetime import datetime
        match = self.TIMESTAMP_PATTERN.search(Path(file_path).name)
        if match:
            return 'Session_' + match.group(1)
        folder_path = os.path.dirname(os.path.abspath(file_path))
        fallback_name = os.path.basename(folder_path) or "Unknown"

        while folder_path:
            folder_name = os.path.basename(folder_path)
            match = self.TIMESTAMP_PATTERN.search(folder_name)

            if match:
                return 'Session_' + match.group(1)

            parent_path = os.path.dirname(folder_path)
            if parent_path == folder_path:
                break

            folder_path = parent_path

        return 'Session_' + datetime.fromtimestamp(Path(file_path).stat().st_mtime).strftime('%Y%m%d_%H%M%S') if Path(file_path).is_file() else fallback_name

    @staticmethod
    def _available_output_path(out_dir, filename):
        """Append a numeric suffix instead of overwriting an existing file."""
        destination = os.path.join(out_dir, filename)
        name_root, extension = os.path.splitext(destination)
        counter = 1

        while os.path.exists(destination):
            destination = f"{name_root}_{counter}{extension}"
            counter += 1

        return destination

