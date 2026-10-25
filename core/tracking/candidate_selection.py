"""Appearance-first selection and independent recovery for each chamber."""
from dataclasses import dataclass, field
import numpy as np
import math
from collections import deque


@dataclass(frozen=True)
class SelectionOptions:
    edge_penalty: float = .5
    minimum_confidence: float = .25
    edge_lock_seconds: float = .5
    lost_seconds: float = .5
    confirmation_frames: int = 3
    global_check_seconds: float = 5.
    search_radius_factor: float = 3.
    minimum_search_radius: float = 24.

    def validate(self):
        if (not all(math.isfinite(value) for value in vars(self).values()) or
                self.global_check_seconds <= 0 or self.search_radius_factor <= 0 or
                self.minimum_search_radius <= 0 or self.edge_lock_seconds <= 0 or
                self.lost_seconds <= 0 or self.confirmation_frames < 1 or
                int(self.confirmation_frames) != self.confirmation_frames or
                not 0 <= self.edge_penalty <= 1 or not 0 <= self.minimum_confidence <= 1):
            raise ValueError('Invalid candidate selection or search settings.')


@dataclass
class ChamberTrackingState:
    mode: str = 'Reacquiring'
    trusted_pos: object = None
    trusted_area: object = None
    area_samples: list = field(default_factory=list)
    suspicious_elapsed: float = 0.
    missing_elapsed: float = 0.
    pending_pos: object = None
    confirmation_count: int = 0
    segment_id: int = 0
    last_time: object = None
    last_global_time: float = float('-inf')
    history: object = field(default_factory=lambda: deque(maxlen=60))
    frame_interval: object = None


class CandidateSelector:
    def __init__(self, chamber_ids, temporal=True, options=None):
        self.options = options or SelectionOptions()
        self.options.validate()
        self.temporal = temporal
        self.states = {cid: ChamberTrackingState() for cid in chamber_ids}

    def score(self, candidate, cid):
        state = self.states[cid]
        reference = state.trusted_area
        ratio = candidate['area'] / reference if reference else 1.
        candidate['area_abnormal'] = bool(reference and not .25 <= ratio <= 4.)
        area_quality = np.exp(-abs(np.log(max(ratio, 1e-6)))) if reference else 1.
        candidate['shape_abnormal'] = candidate['aspect_ratio'] > 6.
        shape_quality = min(1., 3. / max(1., candidate['aspect_ratio']))
        # Before learning body size, a large component must not gain a bonus.
        if candidate['roi_fraction'] > .08:
            area_quality *= .08 / candidate['roi_fraction']
            candidate['area_abnormal'] = True
        candidate['near_edge'] = candidate['edge_fraction'] >= .25
        candidate['appearance_confidence'] = float(area_quality * shape_quality *
            (1. - self.options.edge_penalty * candidate['edge_fraction']))
        return candidate

    def trustworthy(self, candidate):
        return candidate is not None and (
            candidate['appearance_confidence'] >= self.options.minimum_confidence
            and not candidate['area_abnormal'] and not candidate['shape_abnormal'])

    def reset(self, state):
        state.mode = 'Reacquiring'
        state.trusted_pos = None
        state.pending_pos = None
        state.confirmation_count = 0
        state.suspicious_elapsed = state.missing_elapsed = 0.
        state.segment_id += 1
        state.history.clear()
        state.trusted_area = None
        state.area_samples.clear()
        state.frame_interval = None

    def accept(self, state, candidate):
        state.trusted_pos = candidate['local_pos']
        if state.history:
            interval = state.last_time - state.history[-1][0]
            if interval > 0:
                state.frame_interval = (interval if state.frame_interval is None
                                        else .8*state.frame_interval + .2*interval)
        state.history.append((state.last_time, *candidate['local_pos']))
        state.missing_elapsed = state.suspicious_elapsed = 0.
        if not candidate['near_edge']:
            state.area_samples.append(candidate['area'])
            state.area_samples = state.area_samples[-31:]
            if len(state.area_samples) >= 5:
                state.trusted_area = float(np.median(state.area_samples))
        return candidate

    def select(self, cid, candidates, timestamp, scored=False):
        state = self.states[cid]
        elapsed = max(0., timestamp - state.last_time) if state.last_time is not None else 0.
        state.last_time = timestamp
        if not scored:
            candidates = [self.score(candidate, cid) for candidate in candidates]
        trusted = [candidate for candidate in candidates if self.trustworthy(candidate)]
        appearance_best = max(trusted, key=lambda c: c['appearance_confidence'], default=None)
        if not self.temporal:
            return appearance_best if self.trustworthy(appearance_best) else None

        if state.mode == 'Tracking':
            def ranking(candidate):
                distance = np.linalg.norm(np.subtract(candidate['local_pos'], state.trusted_pos))
                scale = max(10., np.sqrt(state.trusted_area or candidate['area']) * 3.)
                return candidate['appearance_confidence'] * (.5 + .5 * np.exp(-distance / scale))
            selected = max(trusted, key=ranking, default=None)
            if selected is not None and self.trajectory_abnormal(state, selected, timestamp):
                self.reset(state)
                selected = None
                state.pending_pos = appearance_best['local_pos'] if appearance_best else None
                state.confirmation_count = 1 if appearance_best else 0
                return None
            if selected is not None:
                return self.accept(state, selected)
            suspicious = any(c['near_edge'] and (c['area_abnormal'] or c['shape_abnormal']) for c in candidates)
            state.suspicious_elapsed = state.suspicious_elapsed + elapsed if suspicious else 0.
            state.missing_elapsed += elapsed
            if (state.suspicious_elapsed >= self.options.edge_lock_seconds or
                    state.missing_elapsed >= self.options.lost_seconds):
                self.reset(state)
            return None

        if not self.trustworthy(appearance_best):
            state.pending_pos, state.confirmation_count = None, 0
            return None
        point = appearance_best['local_pos']
        radius = max(10., np.sqrt(appearance_best['area']) * 3.)
        consistent = state.pending_pos is not None and np.linalg.norm(
            np.subtract(point, state.pending_pos)) <= radius
        state.confirmation_count = state.confirmation_count + 1 if consistent else 1
        state.pending_pos = point
        if state.confirmation_count < self.options.confirmation_frames:
            return None
        state.mode = 'Tracking'
        return self.accept(state, appearance_best)

    def trajectory_abnormal(self, state, candidate, timestamp):
        if state.trusted_pos is None:
            return False
        point = candidate['local_pos']
        distance = math.hypot(point[0] - state.trusted_pos[0], point[1] - state.trusted_pos[1])
        radius = max(self.options.minimum_search_radius,
                     self.options.search_radius_factor * math.sqrt(state.trusted_area or candidate['area']))
        history = state.history
        if history and state.frame_interval:
            radius *= max(1., min(3., (timestamp-history[-1][0]) / state.frame_interval))
        if distance > radius:
            return True
        if len(history) >= 3:
            a, b = history[-2], history[-1]
            alternating = (math.hypot(point[0] - a[1], point[1] - a[2]) < radius * .15
                           and math.hypot(b[1] - a[1], b[2] - a[2]) > radius * .7)
            if alternating:
                return True
        return False
