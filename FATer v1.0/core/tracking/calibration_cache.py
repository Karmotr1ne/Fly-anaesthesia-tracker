"""Persist each video's editable ROI independently of tracking outputs."""
from dataclasses import asdict
import json
import logging

from core.session.layout import SessionPaths
from .models import Calibration


def calibration_path(task):
    return SessionPaths(task.session_dir).logs / (task.video_path.name + '_roi.json')


def save_calibration(task):
    path = calibration_path(task)
    if task.calibration is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    try:
        payload = dict(video=str(task.video_path), size=task.metadata.size,
                       modified_ns=task.metadata.modified_ns,
                       calibration=asdict(task.calibration))
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_calibration(task):
    path = calibration_path(task)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if (payload['video'], payload['size'], payload['modified_ns']) != (
                str(task.video_path), task.metadata.size, task.metadata.modified_ns):
            return None
        values = payload['calibration']
        for name in ('rois', 'reference_rois'):
            if name in values:
                values[name] = tuple(tuple(box) for box in values[name])
        for name in ('shape', 'reference_shape'):
            if name in values:
                values[name] = tuple(values[name])
        calibration = Calibration(**values)
        calibration.validate(task.metadata.shape)
        if calibration.reference_rois:
            Calibration(calibration.reference_rois, calibration.reference_shape,
                        calibration.rows, calibration.cols).validate(calibration.reference_shape)
        return calibration
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        logging.getLogger(__name__).warning('Ignoring invalid ROI cache: %s', path)
        return None
