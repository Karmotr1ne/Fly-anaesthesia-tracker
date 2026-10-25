"""Hardware-facing protocol control; no dependency on acquisition widgets."""
from __future__ import annotations

from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from core.hardware.protocol_scheduler import ProtocolScheduler


class AcquisitionController(QObject):
    setpoint_requested = pyqtSignal(str, float)
    step_changed = pyqtSignal(int, dict)
    progress_changed = pyqtSignal(dict)
    finished = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, parent=None, scheduler=None):
        super().__init__(parent)
        self.scheduler = scheduler if scheduler is not None else ProtocolScheduler()
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.tick)

    def start(self, steps):
        self.scheduler.start(steps)
        self._apply_step()
        self.progress_changed.emit(self.scheduler.snapshot())
        self.timer.start()

    def update_step(self, index, replacement):
        validated = ProtocolScheduler.validate_steps([replacement])[0]
        self.tick()
        scheduler = self.scheduler
        if not scheduler.running:
            raise ValueError("The experiment is no longer running")
        if not scheduler.index <= index < len(scheduler.steps):
            raise ValueError("Only the current or a future step can be edited")
        before = dict(scheduler.steps[index])
        if index == scheduler.index:
            elapsed = scheduler.duration - scheduler.remaining
            if validated["time"] <= elapsed:
                raise ValueError(f"This phase has run for {elapsed:.1f}s. Use a longer duration or Skip Step.")
            scheduler.duration = validated["time"]
            scheduler.remaining = validated["time"] - elapsed
        scheduler.steps[index] = dict(validated)
        self.message.emit(f"Step {index + 1} edited: {before} -> {validated}")
        if index == scheduler.index:
            self._apply_step()
        self.progress_changed.emit(scheduler.snapshot())

    def set_manual_flow(self, unit_id, value):
        key = {"A": "o2", "B": "anes", "C": "n2"}.get(unit_id)
        if key is None:
            raise ValueError("Unknown MFC unit")
        value = ProtocolScheduler.validate_steps([{"time": 1.0, key: value}])[0][key]
        if self.scheduler.running:
            self.tick()
            if not self.scheduler.running:
                raise ValueError("The experiment ended before this change was applied")
            index = self.scheduler.index
            step = dict(self.scheduler.steps[index])
            step[key] = value
            self.update_step(index, step)
        else:
            self.message.emit(f"Manual command requested: Unit {unit_id} = {value:.1f} sccm")
            self.setpoint_requested.emit(unit_id, value)

    def zero_all(self):
        for uid in "ABC":
            self.setpoint_requested.emit(uid, 0.0)

    def stop(self):
        self.timer.stop()
        self.scheduler.stop()
        self.zero_all()

    def _apply_step(self):
        step = self.scheduler.steps[self.scheduler.index]
        for uid, key in (("A", "o2"), ("B", "anes"), ("C", "n2")):
            self.setpoint_requested.emit(uid, step[key])
        self.step_changed.emit(self.scheduler.index, dict(step))

    def tick(self):
        changed, completed = self.scheduler.tick()
        if completed:
            self.timer.stop()
            self.zero_all()
            self.progress_changed.emit(self.scheduler.snapshot())
            self.finished.emit()
        elif self.scheduler.running:
            if changed:
                self._apply_step()
            self.progress_changed.emit(self.scheduler.snapshot())

    def extend(self):
        self.tick()
        if self.scheduler.running:
            self.scheduler.extend()
            self.scheduler.steps[self.scheduler.index]["time"] = self.scheduler.duration
            self.progress_changed.emit(self.scheduler.snapshot())
            self.message.emit("Current phase extended by 30 seconds.")

    def toggle_pause(self):
        self.tick()
        if self.scheduler.running:
            self.scheduler.toggle_pause()
            state = "paused" if self.scheduler.paused else "resumed"
            self.progress_changed.emit(self.scheduler.snapshot())
            self.message.emit(f"Protocol schedule {state}.")

    def skip(self):
        if self.scheduler.running:
            self.message.emit(f"Skip requested for step {self.scheduler.index + 1}")
            self.scheduler.skip()
            # Explicit skip advances even while the schedule is paused.
            paused = self.scheduler.paused
            self.scheduler.paused = False
            self.tick()
            self.scheduler.paused = paused if self.scheduler.running else False
            self.progress_changed.emit(self.scheduler.snapshot())
