"""Shared locations for valve protocols and camera feature presets."""
from pathlib import Path
import sys


def protocol_root(directory=None):
    if directory is not None:
        root = Path(directory).resolve()
    elif getattr(sys, 'frozen', False):
        root = Path(sys.executable).resolve().parent / 'protocols'
    else:
        root = Path(__file__).resolve().parents[2] / 'protocols'
    for name in ('camera', 'air valve'):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def camera_presets():
    directory = protocol_root() / 'camera'
    return sorted((path for path in directory.rglob('*')
                   if path.is_file() and path.suffix.casefold() == '.pfs'),
                  key=lambda path: str(path).casefold())
