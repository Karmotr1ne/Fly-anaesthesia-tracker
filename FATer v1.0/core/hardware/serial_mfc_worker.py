"""core/hardware/serial_mfc_worker.py"""
from __future__ import annotations
import time
import math
from core.hardware.protocol_scheduler import DEVICE_MAP
from PyQt5.QtCore import QMutex, QMutexLocker, QThread, pyqtSignal

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class SerialPollingWorker(QThread):
    port_status_signal = pyqtSignal(bool, str)
    data_updated_signal = pyqtSignal(list)
    command_failed_signal = pyqtSignal(str)

    DEVICE_MAP = DEVICE_MAP

    def __init__(self, port_name: str = "COM3", baudrate: int = 19200):
        super().__init__()
        self.port_name = port_name
        self.running = True
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None
        self._paused = False
        self.mutex = QMutex()
        self._write_queue: list[tuple[str, float]] = []

        self._snapshot_time = 0.0
        self._latest_snapshot = {
            "A": {"flow": float("nan"), "sp": float("nan"), "pressure": float("nan"), "temp": float("nan")},
            "B": {"flow": float("nan"), "sp": float("nan"), "pressure": float("nan"), "temp": float("nan")},
            "C": {"flow": float("nan"), "sp": float("nan"), "pressure": float("nan"), "temp": float("nan")},
        }

    def set_port(self, port_name: str):
        with QMutexLocker(self.mutex):
            normalised = port_name.strip().upper()
            if normalised != self.port_name:
                self._write_queue.clear()
                for values in self._latest_snapshot.values():
                    values.update({key: float("nan") for key in values})
            self.port_name = normalised

    def queue_setpoint(self, unit_id: str, setpoint: float):
        setpoint = float(setpoint)
        if unit_id not in self.DEVICE_MAP or not math.isfinite(setpoint) or setpoint < 0:
            raise ValueError("Unknown MFC unit or invalid setpoint")
        if not math.isclose(setpoint, round(setpoint, 1), abs_tol=1e-9, rel_tol=0):
            raise ValueError("MFC flow setpoints must use 0.1 sccm precision")
        with QMutexLocker(self.mutex):
            # New targets supersede stale unsent targets, especially stop/zero.
            self._write_queue = [(uid, sp) for uid, sp in self._write_queue if uid != unit_id]
            self._write_queue.append((unit_id, setpoint))

    def get_telemetry_snapshot(self) -> dict:
        with QMutexLocker(self.mutex):
            if time.monotonic() - self._snapshot_time > 2.0 or self._paused:
                return {uid: {key: float("nan") for key in values} for uid, values in self._latest_snapshot.items()}
            return {k: dict(v) for k,v in self._latest_snapshot.items()}

    def pause_polling(self):
        self._paused = True
    def resume_polling(self):
        self._paused = False

    def run(self):
        while self.running:
            if self._paused:
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
                time.sleep(0.4)
                continue

            target_port = self.port_name.strip().upper()
            if not target_port or not SERIAL_AVAILABLE:
                self.port_status_signal.emit(False, "FlowVision: Offline")
                time.sleep(1.0)
                continue

            if self.ser is not None and self.ser.is_open and self.ser.port.upper() == target_port:
                available_ports = [target_port]
            else:
                try:
                    available_ports = [p.device.upper() for p in serial.tools.list_ports.comports()]
                except Exception:
                    available_ports = []

            if target_port not in available_ports:
                if self.ser and self.ser.is_open:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                self.port_status_signal.emit(False, f"FlowVision: Device Missing ({target_port})")
                time.sleep(1.0)
                continue

            if self.ser is not None and self.ser.is_open and self.ser.port.upper() != target_port:
                self.ser.close()
                self.ser = None
            if self.ser is None or not self.ser.is_open:
                try:
                    self.ser = serial.Serial(
                        port=target_port,
                        baudrate=self.baudrate,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=0.1,
                        write_timeout=0.2,
                    )
                    self.ser.dtr = True
                    self.ser.rts = True
                except Exception:
                    self.port_status_signal.emit(False, f"FlowVision: Connecting ({target_port})")
                    time.sleep(0.8)
                    continue

            with QMutexLocker(self.mutex):
                cmds_to_run = list(self._write_queue)
                self._write_queue.clear()

            for uid, sp in cmds_to_run:
                try:
                    self.ser.reset_input_buffer()
                    write_cmd = f"{uid}S{sp:.1f}\r".encode("ascii")
                    self.ser.write(write_cmd)
                    time.sleep(0.05)
                    reply = self.ser.read_until(b"\r")
                    if not reply.strip().startswith(uid.encode("ascii")):
                        raise RuntimeError("No matching device acknowledgement")
                except Exception as exc:
                    self.command_failed_signal.emit(f"MFC {uid} setpoint {sp:.1f} failed: {exc}")

            fresh_snapshot = {}
            device_results = []
            communicated = 0

            for unit_id, meta in self.DEVICE_MAP.items():
                dev_data = {
                    "id": f"Unit_{unit_id}",
                    "unit_id": unit_id,
                    "name": meta["name"],
                    "gas_type": meta["gas_default"],
                    "flow_sccm": float("nan"),
                    "setpoint_sccm": float("nan"),
                    "pressure_psia": float("nan"),
                    "temp_c": float("nan"),
                    "valve_status": "Standby",
                }

                if self.ser and self.ser.is_open:
                    try:
                        self.ser.reset_input_buffer()
                        self.ser.write(f"{unit_id}\r".encode("ascii"))
                        time.sleep(0.04)

                        raw_bytes = self.ser.read_until(b"\r")
                        raw_line = raw_bytes.decode("ascii", errors="ignore").strip()

                        if raw_line:
                            parts = raw_line.split()
                            if len(parts) >= 6 and parts[0] == unit_id:
                                dev_data["pressure_psia"] = float(parts[1])
                                dev_data["temp_c"] = float(parts[2])
                                dev_data["flow_sccm"] = float(parts[4])
                                dev_data["setpoint_sccm"] = float(parts[5])
                                if len(parts) >= 7:
                                    dev_data["gas_type"] = parts[6]
                                dev_data["valve_status"] = (
                                    "Active" if dev_data["setpoint_sccm"] > 0 else "Zero / Closed"
                                )
                                if not all(math.isfinite(dev_data[key]) for key in ("flow_sccm", "setpoint_sccm", "pressure_psia", "temp_c")):
                                    raise ValueError("Nonfinite MFC telemetry")
                                communicated += 1

                                fresh_snapshot[unit_id] = {
                                        "flow": dev_data["flow_sccm"],
                                        "sp": dev_data["setpoint_sccm"],
                                        "pressure": dev_data["pressure_psia"],
                                        "temp": dev_data["temp_c"],
                                    }
                    except Exception:
                        try:
                            self.ser.close()
                        except Exception:
                            pass
                        self.ser = None
                        break

                fresh_snapshot.setdefault(unit_id, {"flow": float("nan"), "sp": float("nan"), "pressure": float("nan"), "temp": float("nan")})
                device_results.append(dev_data)

            with QMutexLocker(self.mutex):
                self._latest_snapshot = {uid: fresh_snapshot.get(uid, {key: float("nan") for key in ("flow", "sp", "pressure", "temp")}) for uid in self.DEVICE_MAP}
                self._snapshot_time = time.monotonic()
            self.data_updated_signal.emit(device_results)
            if communicated == len(self.DEVICE_MAP):
                self.port_status_signal.emit(True, f"FlowVision: {target_port} Online")
            else:
                self.port_status_signal.emit(False, f"FlowVision: {target_port} Handshaking...")

            time.sleep(0.3)

        if self.ser and self.ser.is_open:
            with QMutexLocker(self.mutex):
                pending = list(self._write_queue)
                self._write_queue.clear()
            for uid, sp in pending:
                try:
                    self.ser.write(f"{uid}S{sp:.1f}\r".encode("ascii"))
                    self.ser.flush()
                except Exception:
                    break
            self.ser.close()

    def terminate_worker(self):
        self.running = False
        self.wait()
