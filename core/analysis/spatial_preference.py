"""Protocol time windows and compact, full-resolution spatial statistics."""
import json
import re
from pathlib import Path

import numpy as np

from core import schema
from core.session.layout import experiment_name, session_root
from core.hardware.protocol_scheduler import ProtocolScheduler


def protocol_windows(source):
    """Read this experiment's saved plan; never substitute a library default.

    Step durations are cumulative from recording start. The analysis gas-onset
    setting is an event-analysis parameter, not an offset for the whole plan.
    """
    source = Path(source).resolve()
    root = session_root(source)
    folders = list(dict.fromkeys([source.parent, root / 'logs', root / 'log',
                                 root / 'params', root / 'recordings', root]))
    name = experiment_name(source)
    candidates = [folder / filename for filename in
                  (name + '_protocol_steps.json', 'protocol_steps.json', 'active_control.fv')
                  for folder in folders]
    warnings = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == '.json':
                steps = json.loads(path.read_text(encoding='utf-8-sig'))
            else:
                from core.protocol.flowvision_model import FlowVisionScriptModel
                # The interactive parser creates a starter plan for an empty FV.
                # That default must never become experimental evidence.
                content = path.read_text(encoding='utf-8-sig')
                sections = re.findall(r'^\s*\[Step[^\]]*\][^\S\n]*\n(.*?)(?=^\s*\[|\Z)',
                                      content, re.M | re.S)
                if not sections or not all(re.search(r'^\s*Time\s*=', section, re.M | re.I)
                                           for section in sections):
                    raise ValueError('No saved protocol steps')
                steps = FlowVisionScriptModel.parse_script(path)['steps']
                if len(steps) != len(sections):
                    raise ValueError('Incomplete saved protocol steps')
            steps = ProtocolScheduler.validate_steps(steps)
            if not np.isfinite(sum(step['time'] for step in steps)):
                raise ValueError('Protocol duration exceeds the supported time range')
        except (OSError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            warnings.append(f'Spatial preference: cannot read {path.name}: {exc}')
            continue
        windows, start = [], 0.0
        for index, step in enumerate(steps, 1):
            end = start + step['time']
            windows.append(dict(label=f'Step {index}', start=start, end=end,
                                flows={key: step[key] for key in ('o2', 'anes', 'n2')}))
            start = end
        return windows, str(path), warnings
    warnings.append('Spatial preference: no valid saved protocol; showing the whole recording.')
    return [dict(label='Whole recording (protocol unavailable)', start=None, end=None)], None, warnings


def spatial_statistics(data, windows, fps, floor_height=.05, ceiling_height=.95):
    """Keep only 40 height bins and two preferences per chamber and step.

    Integrate frame intervals clipped to step boundaries, including short steps
    between timestamps. Missing frames never extend occupancy through a gap.
    Empty/unobserved steps stay NaN, rather than appearing as 0% preference.
    """
    data = schema.normalise(data)
    height = 'height' if 'height' in data else 'position'
    bins = np.linspace(0., 1., 41)
    results = []
    for chamber, group in data.groupby('chamber_id', sort=True):
        group = group.sort_values('frame', kind='stable')
        times = (group['time'] if 'time' in group else group['frame'] / fps).to_numpy(float)
        positions = group[height].to_numpy(float)
        durations = np.minimum(np.diff(times, append=times[-1] + 1 / fps), 1 / fps)
        valid = np.isfinite(times) & np.isfinite(positions) & np.isfinite(durations) & (durations > 0)
        if 'is_rejected' in group:
            valid &= group['is_rejected'].fillna(1).to_numpy() == 0
        times, positions, durations = times[valid], np.clip(positions[valid], 0, 1), durations[valid]
        occupancy = np.full((len(windows), 40), np.nan)
        preference = np.full((len(windows), 2), np.nan)
        for index, window in enumerate(windows):
            if window['start'] is None:
                weights = durations
            else:
                weights = np.maximum(0., np.minimum(times + durations, window['end'])
                                     - np.maximum(times, window['start']))
            total = weights.sum()
            if total > 0:
                occupancy[index] = np.histogram(positions, bins=bins, weights=weights)[0] * 100 / total
                preference[index] = [weights[positions <= floor_height].sum() * 100 / total,
                                     weights[positions >= ceiling_height].sum() * 100 / total]
        results.append((int(chamber), occupancy, preference))
    return results
