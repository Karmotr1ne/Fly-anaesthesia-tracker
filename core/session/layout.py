"""Shared three-folder session layout and rollback-safe output commits."""
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


DATA_FOLDERS = {"raw", "analysis", "logs", "log", "recordings", "params"}


def session_root(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    for candidate in (directory, *directory.parents):
        if candidate.name.casefold() in DATA_FOLDERS:
            return candidate.parent
        if candidate.name.casefold().startswith("session_"):
            return candidate
        if (candidate / 'raw').is_dir() and any(
                (candidate / name).is_dir() for name in ('logs', 'log', 'analysis')):
            return candidate
    return directory.parent if directory.name in DATA_FOLDERS else directory


@dataclass(frozen=True)
class SessionPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def raw(self):
        return self.root / "raw"

    @property
    def analysis(self):
        return self.root / "analysis"

    @property
    def logs(self):
        return self.root / "logs"

    def ensure(self):
        for directory in (self.raw, self.analysis, self.logs):
            directory.mkdir(parents=True, exist_ok=True)


def experiment_name(path):
    """Use acquisition time rather than accumulating processing suffixes."""
    import re
    from datetime import datetime
    path = Path(path).resolve()
    pattern = re.compile(r'(?<!\d)(\d{8}_\d{6})(?!\d)')
    for value in (path.stem, *(parent.name for parent in path.parents)):
        match = pattern.search(value)
        if match:
            base = match.group(1)
            if value == path.stem:
                suffix = re.match(r'_([0-9a-f]{6})(?:_(?:raw|position|evaluated|summary))?$', value[match.end():])
                if suffix:
                    base += '_' + suffix.group(1)
            return base
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
    return dataset_name(path)


def dataset_name(path):
    stem = Path(path).stem
    for suffix in ("_position", "_raw", "_cleaned", "_detections"):
        if stem.lower().endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def dataset_identity(path):
    path = Path(path).resolve()
    session = session_root(path)
    owner = session if path.parent in (session / "raw", session / "analysis", session / "recordings") else path.parent
    return str(owner / dataset_name(path)).casefold()


def tracking_config_candidates(raw):
    raw = Path(raw)
    name = dataset_name(raw) + "_tracking_config.json"
    return (SessionPaths(session_root(raw)).logs / name, raw.with_name(name))


def temporary_path(path):
    return any(part.startswith((".work-", ".tracking-")) for part in Path(path).parts)


class OutputTransaction:
    """Stage one dataset's files and restore previous outputs if commit fails."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=".work-", dir=self.root)).resolve()
        self.new = self.directory / "new"
        self.new.mkdir()
        self.keep = False

    def stage(self, destination):
        destination = Path(destination).resolve()
        relative = destination.relative_to(self.root)
        staged = self.new / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        return staged

    def commit(self, cancelled=None):
        journal = []
        try:
            for staged in sorted(self.new.rglob("*")):
                if not staged.is_file():
                    continue
                if cancelled and cancelled():
                    raise InterruptedError("Output commit cancelled.")
                relative = staged.relative_to(self.new)
                destination = (self.root / relative).resolve()
                destination.relative_to(self.root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                backup = self.directory / "previous" / relative
                if destination.exists():
                    if not destination.is_file():
                        raise ValueError(f"Output is not a file: {destination}")
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(backup)
                journal.append((destination, backup))
                staged.replace(destination)
            if cancelled and cancelled():
                raise InterruptedError("Output commit cancelled.")
        except Exception:
            failures = []
            for destination, backup in reversed(journal):
                try:
                    if destination.is_file():
                        destination.unlink()
                    if backup.is_file():
                        backup.replace(destination)
                except OSError as exc:
                    failures.append(str(exc))
            if failures:
                self.keep = True
                raise RuntimeError(f"Output rollback incomplete; recovery files retained in {self.directory}: "
                                   + "; ".join(failures))
            raise

    def cleanup(self):
        if self.keep:
            return
        # Only remove the temporary directory created and owned by this instance.
        directory = self.directory.resolve()
        if directory.parent != self.root or not directory.name.startswith(".work-"):
            raise ValueError("Refusing to remove an unowned output directory")
        if directory.is_dir():
            shutil.rmtree(directory)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.cleanup()
