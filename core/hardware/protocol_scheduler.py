"""Protocol timing model independent of Qt widgets and physical hardware."""
from __future__ import annotations

import math
import time

DEVICE_MAP = {
    "A": {"name": "O2", "gas_default": "O2"},
    "B": {"name": "Anaesthetics", "gas_default": "CO2"},
    "C": {"name": "N2", "gas_default": "N2"},
}


class ProtocolScheduler:
    """Monotonic countdown. Pause freezes timing and preserves current flow."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.steps = []
        self.running = False
        self.paused = False
        self.index = 0
        self.remaining = 0.0
        self.duration = 0.0
        self._last_tick = None

    @staticmethod
    def validate_steps(steps):
        if not steps:
            raise ValueError("Protocol requires at least one step")
        validated = []
        for step in steps:
            values = {key: float(step.get(key, default)) for key, default in
                      (("time", 60.0), ("o2", 200.0), ("anes", 0.0), ("n2", 0.0))}
            if any(not math.isfinite(v) for v in values.values()):
                raise ValueError("Protocol values must be finite")
            if values["time"] <= 0 or any(values[k] < 0 for k in ("o2", "anes", "n2")):
                raise ValueError("Duration must be positive and flows nonnegative")
            if any(not math.isclose(values[k], round(values[k], 1), abs_tol=1e-9, rel_tol=0)
                   for k in ("o2", "anes", "n2")):
                raise ValueError("MFC flow setpoints must use 0.1 sccm precision")
            validated.append(values)
        return validated

    def start(self, steps):
        if self.running:
            raise RuntimeError("Protocol already running")
        self.steps = self.validate_steps(steps)
        self.running = True
        self.paused = False
        self.index = 0
        self.remaining = self.duration = self.steps[0]["time"]
        self._last_tick = self.clock()

    def tick(self):
        """Return (step_changed, completed); catch up after delayed callbacks."""
        if not self.running:
            return False, False
        now = self.clock()
        elapsed = max(0.0, now - self._last_tick)
        self._last_tick = now
        if self.paused:
            return False, False
        self.remaining -= elapsed
        old_index = self.index
        while self.remaining <= 0:
            if self.index == len(self.steps) - 1:
                self.remaining = 0.0
                self.running = False
                return False, True
            self.index += 1
            self.duration = self.steps[self.index]["time"]
            self.remaining += self.duration
        return self.index != old_index, False

    def stop(self):
        self.running = False
        self.paused = False
        self._last_tick = None

    def toggle_pause(self):
        self.paused = not self.paused
        self._last_tick = self.clock()

    def extend(self, seconds=30.0):
        seconds = float(seconds)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Extension must be positive and finite")
        self.remaining += seconds
        self.duration += seconds

    def skip(self):
        self.remaining = 0.0
        # An explicit skip must not charge paused time to the next phase.
        self._last_tick = self.clock()

    def snapshot(self):
        percent = int(max(0, min(100, 100 * (1 - self.remaining / self.duration)))) if self.duration else 0
        return {"index": self.index, "remaining": max(0.0, self.remaining),
                "percent": percent, "count": len(self.steps), "paused": self.paused}
