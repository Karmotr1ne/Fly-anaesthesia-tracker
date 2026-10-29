"""Qt background adapters for evaluation and existing archive actions."""
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from core.analysis.anaesthesia_analyser import AnaesthesiaParams
from core.analysis.evaluation_service import evaluate_dataset
from dataclasses import replace
from core.statistics.collector import SummaryCollector

class EvaluationWorker(QThread):
    result_ready = pyqtSignal(int, object, str)
    progress = pyqtSignal(int, str, float)

    def __init__(self, tasks, onset, params=None, plots=False, heatmaps=False, summary_only=False, median_plot=False):
        super().__init__()
        self.tasks = list(tasks)
        self.onset = onset
        self.params = replace(params or AnaesthesiaParams())
        self.plots = plots
        self.heatmaps = heatmaps
        self.median_plot = median_plot
        self.summary_only = summary_only

    def cancel(self):
        self.requestInterruption()

    def run(self):
        for index, (source, _telemetry) in enumerate(self.tasks):
            if self.isInterruptionRequested():
                break
            try:
                # The explicit analysis selection always wins over acquisition snapshots.
                params = replace(self.params, anaesthetic_onset=self.onset)
                summary, output, warnings = evaluate_dataset(
                    source, params, plots=self.plots, heatmaps=self.heatmaps,
                    cancelled=self.isInterruptionRequested, summary_only=self.summary_only,
                    median_plot=self.median_plot,
                    progress=lambda text, fraction: self.progress.emit(index, text, fraction))
                message = str(output) + ("; Plot warning: " + "; ".join(warnings) if warnings else "")
                self.result_ready.emit(index, summary, message)
            except InterruptedError:
                break
            except Exception as exc:
                self.result_ready.emit(index, None, str(exc))


class ArchiveWorker(QThread):
    message = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, mode, paths, destination, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.paths = tuple(Path(p) for p in paths)
        self.destination = Path(destination)

    def cancel(self):
        self.requestInterruption()

    def run(self):
        try:
            collector = SummaryCollector(self.message.emit, self.isInterruptionRequested)
            if self.mode == "summary":
                result = collector.compile_report(self.paths, self.destination)
            else:
                result = collector.copy_raw(self.paths, self.destination)
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class EvaluationImportWorker(QThread):
    """Discover and match session assets without blocking the event loop."""
    result_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = tuple(Path(path) for path in paths)

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from core.session.asset_check import discover_evaluation_inputs
        try:
            result = discover_evaluation_inputs(self.paths, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.result_ready.emit(result)
        except InterruptedError:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class SessionCheckWorker(QThread):
    """Keep session metadata and video-header reads outside the GUI thread."""
    result_ready = pyqtSignal(int, str, object)

    def __init__(self, serial, requests, parent=None):
        super().__init__(parent)
        self.serial = serial
        self.requests = tuple(requests)

    def cancel(self):
        self.requestInterruption()

    def run(self):
        from core.session.asset_check import inspect_session_assets
        cache = {}
        for source, context in self.requests:
            if self.isInterruptionRequested():
                return
            try:
                result = inspect_session_assets(source, context, cache)
            except Exception as exc:
                result = {"text": "Unable to verify", "complete": False, "tooltip": str(exc)}
            if self.isInterruptionRequested():
                return
            self.result_ready.emit(self.serial, str(source), result)
