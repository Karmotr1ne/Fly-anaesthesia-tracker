"""
Basler Pylon Camera & FlowVision Gas Control Integrated Synchronization Workbench
- Window dimensions scaled up 150% (Default: 1920x1140, Minimum: 1620x1020)
- Typography scaled up 150% across all elements (Body: 20px, Titles: 21px, Readout: 33px)
- Strictly enforced single universal font family across the entire application
- Clean, scientific UI styled after Fly-anaesthesia-tracker (neutral surfaces, zero emoji)
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import queue
import csv
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

# Serial communication
try:
    import serial
    import serial.tools.list_ports

    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from PyQt5.QtCore import QMutex, QMutexLocker, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Basler Pylon SDK detection
try:
    from pypylon import pylon

    PYLON_INSTALLED = True
except ImportError:
    PYLON_INSTALLED = False


# =============================================================================
# Global Hardware Handle Teardown & Process Killer
# =============================================================================
def kill_processes_by_name(target_names: list[str]) -> bool:
    terminated = False
    for proc in psutil.process_iter(["name"]):
        try:
            p_name = proc.info["name"]
            if p_name and any(tn in p_name.lower() for tn in target_names):
                proc.kill()
                terminated = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return terminated


def clean_all_camera_resources():
    kill_processes_by_name(
        ["pylonviewer.exe", "pylonviewerapp.exe", "pylonviewer"]
    )
    if PYLON_INSTALLED:
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            tl_factory.ReleaseTl()
        except Exception:
            pass


atexit.register(clean_all_camera_resources)


# =============================================================================
# Single Font Family Definition & 150% Scaled QSS
# =============================================================================
UNIFIED_FONT_FAMILY = (
    '"Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue",'
    " Arial, sans-serif"
)

SCALED_150_SCIENTIFIC_QSS = f"""
QWidget {{
    font-family: {UNIFIED_FONT_FAMILY};
    font-size: 20px;
    color: #1E293B;
}}

QMainWindow, QDialog {{
    background-color: #F8FAFC;
}}

QGroupBox {{
    background-color: #FFFFFF;
    border: 2px solid #CBD5E1;
    border-radius: 9px;
    margin-top: 21px;
    padding: 18px 15px 15px 15px;
    font-weight: 700;
    color: #0F172A;
    font-size: 21px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 18px;
    padding: 0 8px;
    background-color: #F8FAFC;
    color: #0F172A;
}}

QTabWidget::pane {{
    border: 2px solid #CBD5E1;
    background: #FFFFFF;
    border-radius: 6px;
}}

QTabBar::tab {{
    background: #F1F5F9;
    color: #475569;
    border: 2px solid #CBD5E1;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-height: 42px;
    min-width: 140px;
    padding: 5px 18px;
    font-weight: 600;
    font-size: 20px;
    margin-right: 6px;
}}

QTabBar::tab:selected {{
    background: #FFFFFF;
    color: #1D4ED8;
    border-bottom: 3px solid #2563EB;
}}

QLineEdit, QDoubleSpinBox {{
    background-color: #FFFFFF;
    border: 2px solid #CBD5E1;
    border-radius: 6px;
    padding: 5px 12px;
    min-height: 39px;
    color: #0F172A;
    font-size: 20px;
    font-weight: 500;
}}

QLineEdit:focus, QDoubleSpinBox:focus {{
    border: 2px solid #2563EB;
}}

QPushButton {{
    background-color: #FFFFFF;
    border: 2px solid #CBD5E1;
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 20px;
    font-weight: 600;
    color: #334155;
    min-height: 39px;
}}

QPushButton:hover {{
    background-color: #F1F5F9;
    border-color: #94A3B8;
    color: #0F172A;
}}

QPushButton:pressed {{
    background-color: #E2E8F0;
}}

QPushButton:disabled {{
    background-color: #F1F5F9;
    color: #94A3B8;
    border-color: #E2E8F0;
}}

QProgressBar {{
    border: 2px solid #CBD5E1;
    border-radius: 6px;
    background-color: #F1F5F9;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    color: #0F172A;
    min-height: 30px;
}}

QProgressBar::chunk {{
    background-color: #2563EB;
    border-radius: 5px;
}}

QTextEdit {{
    background-color: #FFFFFF;
    border: 2px solid #CBD5E1;
    border-radius: 6px;
    font-family: {UNIFIED_FONT_FAMILY};
    font-size: 18px;
    color: #334155;
    padding: 9px;
}}
"""


# =============================================================================
# 1. FlowVision Script Model
# =============================================================================
class FlowVisionScriptModel:

    @staticmethod
    def parse_script(file_path: str | Path) -> dict:
        data = {
            "port": "COM3",
            "gas_onset_sec": 60.0,
            "total_duration": 300.0,
            "steps": [],
        }
        p = Path(file_path)
        if not p.is_file():
            return data

        lines = p.read_text(encoding="utf-8").splitlines()
        current_section = ""
        current_step = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                if current_step and "time" in current_step:
                    data["steps"].append(current_step)
                    current_step = {}
                current_section = line[1:-1].strip()
                continue

            if "=" in line:
                k, v = [x.strip() for x in line.split("=", 1)]
                if current_section == "GasDeliveryControl":
                    if k.lower() == "port":
                        data["port"] = v
                    elif k.lower() in ("gasonsettime", "gas_onset_sec"):
                        data["gas_onset_sec"] = float(v)
                    elif k.lower() in ("duration", "totalduration"):
                        data["total_duration"] = float(v)
                elif current_section.startswith("Step"):
                    if k.lower() == "time":
                        current_step["time"] = float(v)
                    elif "o2" in k.lower():
                        current_step["o2"] = float(v)
                    elif "anesthe" in k.lower() or "co2" in k.lower():
                        current_step["anes"] = float(v)
                    elif "n2" in k.lower():
                        current_step["n2"] = float(v)

        if current_step and "time" in current_step:
            data["steps"].append(current_step)

        if not data["steps"]:
            data["steps"] = [
                {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
                {"time": 240.0, "o2": 180.0, "anes": 20.0, "n2": 0.0},
            ]
        return data

    @staticmethod
    def generate_script_text(
        port: str, gas_onset: float, duration: float, steps: list[dict]
    ) -> str:
        lines = [
            "# FlowVision2 Protocol Control Script",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "[GasDeliveryControl]",
            f"Port = {port}",
            f"GasOnsetTime = {gas_onset:.1f}",
            f"Duration = {duration:.1f}",
            f"TotalSteps = {len(steps)}",
            "",
        ]
        for idx, s in enumerate(steps, start=1):
            lines.append(f"[Step_{idx}]")
            lines.append(f"Time = {s.get('time', 60.0):.1f}")
            lines.append(f"Unit_A_O2 = {s.get('o2', 200.0):.1f}")
            lines.append(f"Unit_B_Anesthetics = {s.get('anes', 0.0):.1f}")
            lines.append(f"Unit_C_N2 = {s.get('n2', 0.0):.1f}")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# 2. Session Manager
# =============================================================================
@dataclass
class HardwareConfig:
    fv_exe_path: str = "C:/Program Files/FlowVision2/bin/flowvision.exe"
    fv_script_path: str = "active_control.fv"
    fv_com_port: str = "COM3"
    gas_onset_sec: float = 60.0
    experiment_duration: float = 300.0
    pfs_path: str = "Fly Presetting_4 column 30fps.pfs"


class SessionManager:

    def __init__(self, workspace_root: str = "./workspace"):
        self.workspace_root = Path(workspace_root)

    def create_session(
        self, config: HardwareConfig, script_content: str
    ) -> tuple[Path, Path, Path, Path]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.workspace_root / "sessions" / f"Session_{ts}"
        params_dir = session_dir / "params"
        rec_dir = session_dir / "recordings"
        logs_dir = session_dir / "logs"

        for p in (params_dir, rec_dir, logs_dir):
            p.mkdir(parents=True, exist_ok=True)

        with open(
            params_dir / "runtime_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(config), f, indent=4)

        saved_pfs_path = params_dir / "active_camera.pfs"
        if config.pfs_path and Path(config.pfs_path).is_file():
            shutil.copyfile(config.pfs_path, saved_pfs_path)

        active_script_path = params_dir / "active_control.fv"
        active_script_path.write_text(script_content, encoding="utf-8")

        active_video_path = rec_dir / "raw_video.mp4"
        return session_dir, active_script_path, active_video_path, saved_pfs_path


# =============================================================================
# 3. Dedicated Asynchronous Serial Worker (Alicat Multidrop Protocol)
# =============================================================================
class SerialPollingWorker(QThread):
    port_status_signal = pyqtSignal(bool, str)
    data_updated_signal = pyqtSignal(list)

    DEVICE_MAP = {
        "A": {"name": "O2", "gas_default": "O2"},
        "B": {"name": "Anesthetics", "gas_default": "CO2"},
        "C": {"name": "N2", "gas_default": "N2"},
    }

    def __init__(self, port_name: str = "COM3"):
        super().__init__()
        self.port_name = port_name
        self.running = True
        self.baudrate = 19200
        self.ser: serial.Serial | None = None
        self._paused = False
        self.mutex = QMutex()
        self._write_queue: list[tuple[str, float]] = []

    def set_port(self, port_name: str):
        with QMutexLocker(self.mutex):
            self.port_name = port_name

    def queue_setpoint(self, unit_id: str, setpoint: float):
        with QMutexLocker(self.mutex):
            self._write_queue.append((unit_id, float(setpoint)))

    def pause_polling(self):
        self._paused = True
        with QMutexLocker(self.mutex):
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def resume_polling(self):
        self._paused = False

    def run(self):
        while self.running:
            if self._paused:
                time.sleep(0.4)
                continue

            target_port = self.port_name.strip().upper()
            if not target_port or not SERIAL_AVAILABLE:
                self.port_status_signal.emit(False, "Serial Module Offline")
                time.sleep(1.0)
                continue

            try:
                available_ports = [
                    p.device.upper()
                    for p in serial.tools.list_ports.comports()
                ]
            except Exception:
                available_ports = []

            if target_port not in available_ports:
                if self.ser and self.ser.is_open:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                self.ser = None
                self.port_status_signal.emit(
                    False, f"Device Missing ({target_port})"
                )
                time.sleep(1.0)
                continue

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
                except serial.SerialException as se:
                    err_str = str(se)
                    if (
                        "PermissionError" in err_str
                        or "Access is denied" in err_str
                    ):
                        self.port_status_signal.emit(
                            False, "Port Locked (Click Release)"
                        )
                    else:
                        self.port_status_signal.emit(
                            False, f"Connecting ({target_port})"
                        )
                    time.sleep(0.8)
                    continue
                except Exception:
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
                    time.sleep(0.06)
                    self.ser.read_until(b"\r")
                except Exception:
                    pass

            device_results = []
            communicated = False

            for unit_id, meta in self.DEVICE_MAP.items():
                dev_data = {
                    "id": f"Unit_{unit_id}",
                    "unit_id": unit_id,
                    "name": meta["name"],
                    "gas_type": meta["gas_default"],
                    "flow_sccm": 0.0,
                    "setpoint_sccm": 0.0,
                    "pressure_psia": 14.7,
                    "temp_c": 25.0,
                    "valve_status": "Standby",
                }

                if self.ser and self.ser.is_open:
                    try:
                        self.ser.reset_input_buffer()
                        self.ser.write(f"{unit_id}\r".encode("ascii"))
                        time.sleep(0.04)

                        raw_bytes = self.ser.read_until(b"\r")
                        raw_line = raw_bytes.decode(
                            "ascii", errors="ignore"
                        )
                        raw_line = raw_line.lstrip("\x00\r\n ").strip()

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
                                    "Active"
                                    if dev_data["setpoint_sccm"] > 0
                                    else "Zero / Closed"
                                )
                                communicated = True
                    except Exception:
                        try:
                            self.ser.close()
                        except Exception:
                            pass
                        self.ser = None
                        break

                device_results.append(dev_data)

            if communicated:
                self.port_status_signal.emit(
                    True, f"FV Hardware: {target_port} Online"
                )
                self.data_updated_signal.emit(device_results)
            else:
                self.port_status_signal.emit(
                    False, f"FV Hardware: {target_port} Handshaking..."
                )

            time.sleep(0.3)

        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

    def terminate_worker(self):
        self.running = False
        self.wait(1000)


# =============================================================================
# 4. Display Enhancement Pipeline & Aspect-Ratio Preserving Viewport
# =============================================================================
class ImagePipelineEnhancer:

    def __init__(self, gamma: float = 2.0, contrast: float = 0.35):
        self.gamma = gamma
        self.contrast = contrast
        self.lut = self._build_lut()

    def _build_lut(self) -> np.ndarray:
        x = np.linspace(0, 1, 256)
        k = 1.0 + self.contrast * 8.0
        s_curve = 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
        s_curve = (s_curve - s_curve.min()) / (s_curve.max() - s_curve.min())

        inv_gamma = 1.0 / self.gamma
        y = np.power(s_curve, inv_gamma)
        return (y * 255.0).clip(0, 255).astype(np.uint8)

    def process(self, raw_frame: np.ndarray) -> np.ndarray:
        p_low = np.percentile(raw_frame, 1.0)
        p_high = np.percentile(raw_frame, 99.5)

        if p_high > p_low + 10:
            stretched = np.clip(
                (raw_frame.astype(np.float32) - p_low)
                * (255.0 / (p_high - p_low)),
                0,
                255,
            ).astype(np.uint8)
        else:
            stretched = raw_frame

        return cv2.LUT(stretched, self.lut)


class AspectRatioVideoWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 390)  # 150% scaled minimum
        self.setStyleSheet(
            "background-color: #0F172A; border-radius: 9px; border: 2px solid"
            " #CBD5E1;"
        )
        self._pixmap = None

    def update_frame(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        rgb = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            if len(frame.shape) == 2
            else frame
        )
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0F172A"))
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor("#94A3B8"))
            f = painter.font()
            f.setFamily("Segoe UI")
            f.setPointSize(21)
            painter.setFont(f)
            painter.drawText(
                self.rect(), Qt.AlignCenter, "Waiting for Camera Video Stream..."
            )


# =============================================================================
# 5. Robust Camera Acquisition Worker
# =============================================================================
class VideoWriterThread(QThread):
    def __init__(self, output_path: str, fps: float, meta_path: str):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.meta_path = meta_path

        # 缓冲队列：最大暂存 120 帧（根据超大分辨率 4200x1300 的内存占用约 2GB，可按需微调）
        self.queue: queue.Queue[tuple[np.ndarray, float, int] | None] = queue.Queue(maxsize=120)
        self.running = True
        self.total_frames_written = 0

        # 尺寸信息在 run 接收到第一帧时动态确定，避免配置加载竞争
        self.target_width = None
        self.target_height = None

    def enqueue_frame(self, frame: np.ndarray, timestamp: float, hw_frame_id: int):
        if not self.running:
            return
        try:
            # 队列满时不阻塞相机采集线程，以防相机驱动死锁
            self.queue.put_nowait((frame, timestamp, hw_frame_id))
        except queue.Full:
            print("[Recorder Buffer Warning] Write queue full! Frame dropped from write stream.")

    def run(self):
        writer: cv2.VideoWriter | None = None
        meta_file = open(self.meta_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(meta_file)
        csv_writer.writerow(["VideoFrameIndex", "WallClockTimestamp", "HWFrameId", "IsPaddedFrame"])

        start_time = None
        last_valid_frame = None

        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:  # 接收到退出信号
                break

            frame, frame_time, hw_id = item

            # ---------------- 1. 数据合法性检查与格式归一化 ----------------
            if frame is None or frame.size == 0:
                self.queue.task_done()
                continue

            # 确保通道数正确 (H, W, 3)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 确保数据类型为 uint8
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            # ---------------- 2. 延迟自适应初始化 VideoWriter ----------------
            h, w = frame.shape[:2]
            if writer is None:
                # 强制宽度和高度为偶数（避免某些 FFmpeg 宏块无法整除而报错）
                self.target_width = w if w % 2 == 0 else w - 1
                self.target_height = h if h % 2 == 0 else h - 1

                # 优先尝试 mp4v 或 avc1，在 4200 宽下兼容性更稳
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self.output_path,
                    fourcc,
                    self.fps,
                    (self.target_width, self.target_height),
                    True
                )
                print(f"[Recorder] VideoWriter initialized dynamically: {self.target_width}x{self.target_height} @ {self.fps} FPS")

            # ---------------- 3. 动态尺寸防爆校验 ----------------
            # 如果中间偶发相机分辨率切换，按目标尺寸强行 resize，避免抛出 Failed to write frame
            if w != self.target_width or h != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)

            last_valid_frame = frame

            if start_time is None:
                start_time = frame_time

            # ---------------- 4. 掉帧平滑保护与时间轴对齐 ----------------
            elapsed_real_time = frame_time - start_time
            expected_frames = int(elapsed_real_time * self.fps)
            frames_to_add = expected_frames - self.total_frames_written

            # 如果中途丢帧导致落后，补充重复帧，并限制单次突发写入上限防止编码缓冲区溢出
            if frames_to_add > 1:
                safe_padding = min(frames_to_add - 1, 10)
                for _ in range(safe_padding):
                    writer.write(frame)
                    self.total_frames_written += 1
                    csv_writer.writerow([self.total_frames_written, frame_time, hw_id, 1])

            # 写入当前原生抓取帧
            writer.write(frame)
            self.total_frames_written += 1
            csv_writer.writerow([self.total_frames_written, frame_time, hw_id, 0])

            self.queue.task_done()

        # ---------------- 5. 资源安全释放 ----------------
        if writer is not None:
            writer.release()
            print(f"[Recorder] Video file closed: {self.output_path}")

        meta_file.flush()
        meta_file.close()

    def stop_and_wait(self):
        self.running = False
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.wait(3000)


class RobustCameraWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_changed = pyqtSignal(str, str, str)
    pfs_applied_signal = pyqtSignal(bool, str)

    HARDCODED_OUTPUT_FPS = 30.0

    def __init__(self, pfs_path: str = ""):
        super().__init__()
        self.pfs_path = pfs_path
        self.running = False
        self.recording = False
        self.camera = None

        self.enhancer = ImagePipelineEnhancer(gamma=2.0, contrast=0.35)
        self.mutex = QMutex()
        self._pending_pfs_path = None
        self.frame_width = 4200
        self.frame_height = 1300
        self._retry_interval = 2.0

        self.writer_thread: VideoWriterThread | None = None
        self.output_video_path = ""

    def _safe_close_camera(self):
        try:
            if self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                if self.camera.IsOpen():
                    self.camera.Close()
        except Exception:
            pass
        self.camera = None

    def terminate_worker(self):
        self.running = False
        self.stop_recording()
        self.wait(1500)

    def set_pfs_path(self, path: str):
        with QMutexLocker(self.mutex):
            self._pending_pfs_path = path

    def request_apply_pfs(self, path: str):
        self.set_pfs_path(path)

    def _execute_pfs_load_internal(self, path: str):
        try:
            if self.camera and self.camera.IsOpen():
                was_grabbing = self.camera.IsGrabbing()
                if was_grabbing:
                    self.camera.StopGrabbing()

                pylon.FeaturePersistence.Load(path, self.camera.GetNodeMap(), True)
                print(f"[PFS Loader] Successfully loaded feature set: {path}")
                self.pfs_applied_signal.emit(True, f"Loaded: {path}")

                if was_grabbing and self.running:
                    # 录制与实时场景，抓取策略选 OneByOne 保证队列不被驱动抛弃
                    self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            else:
                self.pfs_applied_signal.emit(False, "Camera not open for feature loading")
        except Exception as e:
            print(f"[PFS Loader Error] Failed to load {path}: {e}")
            self.pfs_applied_signal.emit(False, str(e))

    def start_recording(self, video_path: str):
        with QMutexLocker(self.mutex):
            save_path = Path(video_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_video_path = str(save_path)
            meta_path = str(save_path.with_name(save_path.stem + "_frames_meta.csv"))

            # 移除了固定的 width=self.frame_width, height=self.frame_height 参数
            self.writer_thread = VideoWriterThread(
                output_path=self.output_video_path,
                fps=self.HARDCODED_OUTPUT_FPS,
                meta_path=meta_path,
            )
            self.writer_thread.start()
            self.recording = True
            print(f"[Recorder] Recording armed -> {self.output_video_path}")

    def stop_recording(self):
        with QMutexLocker(self.mutex):
            self.recording = False
            if self.writer_thread is not None:
                self.writer_thread.stop_and_wait()
                self.writer_thread = None
                print(f"[Recorder] File finalized and released.")

    def _attempt_open_device(self, tl_factory) -> bool:
        try:
            if not PYLON_INSTALLED or tl_factory is None:
                return False

            devices = tl_factory.EnumerateDevices()
            if not devices:
                return False

            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()
            return True
        except Exception as e:
            print(f"[Camera Error] Failed to open device: {e}")
            self._safe_close_camera()
            return False

    def run(self):
        self.running = True
        tl_factory = pylon.TlFactory.GetInstance() if PYLON_INSTALLED else None

        while self.running:
            if self.camera is None or not self.camera.IsOpen():
                success = self._attempt_open_device(tl_factory)
                if not success:
                    time.sleep(self._retry_interval)
                    continue

            # 热加载 PFS 处理
            reload_path = None
            with QMutexLocker(self.mutex):
                if self._pending_pfs_path:
                    reload_path = self._pending_pfs_path
                    self._pending_pfs_path = None

            if reload_path:
                self._execute_pfs_load_internal(reload_path)

            try:
                if not self.camera.IsGrabbing():
                    # 重点修改 1：改用 OneByOne，防止相机驱动在短时繁忙时抛弃帧
                    self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                    self.frame_width = self.camera.Width.GetValue() if hasattr(self.camera, "Width") else self.frame_width
                    self.frame_height = self.camera.Height.GetValue() if hasattr(self.camera, "Height") else self.frame_height
                    self.status_changed.emit("OK", "Camera: Online Ready", f"Resolution: {self.frame_width}x{self.frame_height}")

                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    raw_frame = grab_result.Array
                    hw_block_id = grab_result.GetBlockID()
                    now_timestamp = time.perf_counter()

                    # 界面增强显示
                    display_frame = self.enhancer.process(raw_frame)
                    self.frame_ready.emit(display_frame)

                    # 录制分支
                    with QMutexLocker(self.mutex):
                        if self.recording and self.writer_thread is not None:
                            # 格式化颜色以送入编码器
                            if len(display_frame.shape) == 2:
                                write_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
                            else:
                                write_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

                            # 重点修改 2：非阻塞塞入写入队列，毫秒级返回
                            self.writer_thread.enqueue_frame(write_frame, now_timestamp, hw_block_id)

                grab_result.Release()
            except Exception as e:
                with QMutexLocker(self.mutex):
                    is_reloading = self._pending_pfs_path is not None

                if not is_reloading and self.running:
                    self.status_changed.emit("PENDING", "Camera: Signal Interrupted...", str(e))
                    self._safe_close_camera()
                    time.sleep(1.0)

        self.stop_recording()
        self._safe_close_camera()

# =============================================================================
# 6. Independent Dialog: Protocol Library & Step Editor
# =============================================================================
class ProtocolLibraryDialog(QDialog):
    protocol_applied = pyqtSignal(str, float, list)

    def __init__(
        self,
        current_dir: str = "./protocols",
        active_steps: list[dict] | None = None,
        active_onset: float = 60.0,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Protocol Library & Step Designer")
        self.resize(1290, 720)  # 150% scaled dialog size
        self.protocols_dir = Path(current_dir)
        self.protocols_dir.mkdir(parents=True, exist_ok=True)

        self.active_steps = (
            list(active_steps)
            if active_steps
            else [
                {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
                {"time": 240.0, "o2": 180.0, "anes": 20.0, "n2": 0.0},
            ]
        )
        self.gas_onset = active_onset

        self.setup_ui()
        self.refresh_file_list()
        self.populate_table(self.active_steps)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(21, 21, 21, 21)
        main_layout.setSpacing(21)

        # Left Column: Protocol Browser
        v_left = QVBoxLayout()
        v_left.addWidget(QLabel("<b>Saved Protocol Library:</b>"))
        self.list_files = QListWidget()
        self.list_files.setStyleSheet(f"font-family: {UNIFIED_FONT_FAMILY}; font-size: 20px; padding: 6px;")
        v_left.addWidget(self.list_files)

        h_file_btns = QHBoxLayout()
        self.btn_new = QPushButton("New Blank")
        self.btn_browse_dir = QPushButton("Browse Dir...")
        h_file_btns.addWidget(self.btn_new)
        h_file_btns.addWidget(self.btn_browse_dir)
        v_left.addLayout(h_file_btns)
        main_layout.addLayout(v_left, 3)

        # Right Column: Editor
        v_right = QVBoxLayout()
        v_right.addWidget(QLabel("<b>Step Sequence Editor:</b>"))

        h_table_tools = QHBoxLayout()
        self.btn_add_step = QPushButton("Add Step")
        self.btn_del_step = QPushButton("Delete Step")
        self.btn_move_up = QPushButton("Move Up")
        self.btn_move_down = QPushButton("Move Down")
        h_table_tools.addWidget(self.btn_add_step)
        h_table_tools.addWidget(self.btn_del_step)
        h_table_tools.addWidget(self.btn_move_up)
        h_table_tools.addWidget(self.btn_move_down)
        v_right.addLayout(h_table_tools)

        self.table_steps = QTableWidget()
        self.table_steps.setColumnCount(5)
        self.table_steps.setHorizontalHeaderLabels(
            ["Phase", "Duration (s)", "Unit A: O2", "Unit B: Anes", "Unit C: N2"]
        )
        self.table_steps.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.table_steps.setStyleSheet(f"font-family: {UNIFIED_FONT_FAMILY}; font-size: 20px;")
        v_right.addWidget(self.table_steps)

        h_params = QHBoxLayout()
        lbl_onset = QLabel("Gas Onset Time:")
        lbl_onset.setStyleSheet("font-size: 20px; font-weight: 600;")
        h_params.addWidget(lbl_onset)

        self.spin_onset = QDoubleSpinBox()
        self.spin_onset.setRange(0, 3600)
        self.spin_onset.setValue(self.gas_onset)
        self.spin_onset.setSuffix(" s")
        self.spin_onset.setMinimumHeight(42)
        h_params.addWidget(self.spin_onset)

        self.lbl_total_duration = QLabel("Total Duration: 0.0 s")
        self.lbl_total_duration.setStyleSheet("font-size: 21px; font-weight: 700; color: #2563EB;")
        h_params.addWidget(self.lbl_total_duration)
        v_right.addLayout(h_params)

        h_bottom = QHBoxLayout()
        self.btn_save = QPushButton("Save to File")
        self.btn_apply = QPushButton("Apply to Main Session")
        self.btn_apply.setStyleSheet(
            "background-color: #2563EB; color: white; font-weight: 700; padding: 9px 24px; min-height: 45px;"
        )
        self.btn_close = QPushButton("Cancel")
        self.btn_close.setMinimumHeight(45)
        self.btn_save.setMinimumHeight(45)

        h_bottom.addWidget(self.btn_save)
        h_bottom.addStretch()
        h_bottom.addWidget(self.btn_apply)
        h_bottom.addWidget(self.btn_close)
        v_right.addLayout(h_bottom)

        main_layout.addLayout(v_right, 6)

        self.list_files.itemClicked.connect(self.on_file_selected)
        self.btn_browse_dir.clicked.connect(self.choose_directory)
        self.btn_new.clicked.connect(self.new_blank_protocol)
        self.btn_add_step.clicked.connect(self.add_step_row)
        self.btn_del_step.clicked.connect(self.del_step_row)
        self.btn_move_up.clicked.connect(self.move_step_up)
        self.btn_move_down.clicked.connect(self.move_step_down)
        self.btn_save.clicked.connect(self.save_to_file)
        self.btn_apply.clicked.connect(self.apply_to_main)
        self.btn_close.clicked.connect(self.reject)
        self.table_steps.cellChanged.connect(self.update_total_duration)

    def refresh_file_list(self):
        self.list_files.clear()
        if self.protocols_dir.is_dir():
            for f in sorted(self.protocols_dir.glob("*.fv")):
                self.list_files.addItem(f.name)
            for f in sorted(self.protocols_dir.glob("*.txt")):
                if not f.name.endswith(".fv.txt"):
                    self.list_files.addItem(f.name)

    def choose_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Protocols Directory", str(self.protocols_dir)
        )
        if d:
            self.protocols_dir = Path(d)
            self.refresh_file_list()

    def on_file_selected(self, item):
        fpath = self.protocols_dir / item.text()
        parsed = FlowVisionScriptModel.parse_script(fpath)
        self.spin_onset.setValue(parsed.get("gas_onset_sec", 60.0))
        self.populate_table(parsed.get("steps", []))

    def populate_table(self, steps: list[dict]):
        self.table_steps.blockSignals(True)
        self.table_steps.setRowCount(len(steps))
        for row, s in enumerate(steps):
            item_idx = QTableWidgetItem(f"Step {row + 1}")
            item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)

            item_time = QTableWidgetItem(str(s.get("time", 60.0)))
            item_o2 = QTableWidgetItem(str(s.get("o2", 200.0)))
            item_anes = QTableWidgetItem(str(s.get("anes", 0.0)))
            item_n2 = QTableWidgetItem(str(s.get("n2", 0.0)))

            self.table_steps.setItem(row, 0, item_idx)
            self.table_steps.setItem(row, 1, item_time)
            self.table_steps.setItem(row, 2, item_o2)
            self.table_steps.setItem(row, 3, item_anes)
            self.table_steps.setItem(row, 4, item_n2)

        self.table_steps.blockSignals(False)
        self.update_total_duration()

    def update_total_duration(self):
        steps = self.extract_steps()
        total = sum(s.get("time", 0.0) for s in steps)
        mins = int(total // 60)
        secs = int(total % 60)
        self.lbl_total_duration.setText(
            f"Total Duration: {total:.1f} s ({mins:02d}m {secs:02d}s)"
        )

    def extract_steps(self) -> list[dict]:
        steps = []
        rows = self.table_steps.rowCount()
        for r in range(rows):
            try:
                t = float(self.table_steps.item(r, 1).text())
                o2 = float(self.table_steps.item(r, 2).text())
                anes = float(self.table_steps.item(r, 3).text())
                n2 = float(self.table_steps.item(r, 4).text())
                steps.append({"time": t, "o2": o2, "anes": anes, "n2": n2})
            except Exception:
                continue
        return steps

    def new_blank_protocol(self):
        default_steps = [
            {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
            {"time": 240.0, "o2": 180.0, "anes": 20.0, "n2": 0.0},
        ]
        self.spin_onset.setValue(60.0)
        self.populate_table(default_steps)

    def add_step_row(self):
        r = self.table_steps.rowCount()
        self.table_steps.insertRow(r)
        item_idx = QTableWidgetItem(f"Step {r + 1}")
        item_idx.setFlags(item_idx.flags() & ~Qt.ItemIsEditable)
        self.table_steps.setItem(r, 0, item_idx)
        self.table_steps.setItem(r, 1, QTableWidgetItem("60.0"))
        self.table_steps.setItem(r, 2, QTableWidgetItem("200.0"))
        self.table_steps.setItem(r, 3, QTableWidgetItem("0.0"))
        self.table_steps.setItem(r, 4, QTableWidgetItem("0.0"))
        self.update_total_duration()

    def del_step_row(self):
        curr = self.table_steps.currentRow()
        if curr >= 0:
            self.table_steps.removeRow(curr)
        elif self.table_steps.rowCount() > 0:
            self.table_steps.removeRow(self.table_steps.rowCount() - 1)
        self.update_total_duration()

    def move_step_up(self):
        curr = self.table_steps.currentRow()
        if curr > 0:
            steps = self.extract_steps()
            steps[curr - 1], steps[curr] = steps[curr], steps[curr - 1]
            self.populate_table(steps)
            self.table_steps.setCurrentCell(curr - 1, 1)

    def move_step_down(self):
        curr = self.table_steps.currentRow()
        if 0 <= curr < self.table_steps.rowCount() - 1:
            steps = self.extract_steps()
            steps[curr], steps[curr + 1] = steps[curr + 1], steps[curr]
            self.populate_table(steps)
            self.table_steps.setCurrentCell(curr + 1, 1)

    def save_to_file(self):
        fpath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Protocol File",
            str(self.protocols_dir / "protocol_experiment.fv"),
            "FlowVision Files (*.fv);;Text Files (*.txt)",
        )
        if fpath:
            steps = self.extract_steps()
            total_dur = sum(s.get("time", 0.0) for s in steps)
            text = FlowVisionScriptModel.generate_script_text(
                port="COM3",
                gas_onset=self.spin_onset.value(),
                duration=total_dur,
                steps=steps,
            )
            Path(fpath).write_text(text, encoding="utf-8")
            self.refresh_file_list()
            QMessageBox.information(
                self, "Saved", f"Protocol successfully saved to:\n{Path(fpath).name}"
            )

    def apply_to_main(self):
        steps = self.extract_steps()
        if not steps:
            QMessageBox.warning(self, "Warning", "Protocol contains no steps!")
            return
        selected_item = self.list_files.currentItem()
        name = selected_item.text() if selected_item else "Custom_In_Memory.fv"
        self.protocol_applied.emit(name, self.spin_onset.value(), steps)
        self.accept()


# =============================================================================
# 7. Main Application Window (150% Scaled Geometry & Balanced Ratios)
# =============================================================================
class HardwareIntegrationApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Basler Pylon Camera & FlowVision Gas Control Integrated Synchronization Workbench"
        )
        # Scaled up exactly 150%: 1280x760 -> 1920x1140 (Minimum: 1620x1020)
        self.resize(1920, 1140)
        self.setMinimumSize(1620, 1020)

        clean_all_camera_resources()
        self.kill_flowvision_process()

        self.config = HardwareConfig()
        self.session_mgr = SessionManager()
        self.camera_worker = RobustCameraWorker(self.config.pfs_path)
        self.serial_worker = SerialPollingWorker(self.config.fv_com_port)

        self.active_session_dir: Path | None = None
        self.active_video_path: Path | None = None
        self.camera_is_ready = False
        self.cam_raw_state = "PENDING"

        # Protocol schedule state machine
        self.active_protocol_name = "default_biphasic_anesthesia.fv"
        self.script_steps = [
            {"time": 60.0, "o2": 200.0, "anes": 0.0, "n2": 0.0},
            {"time": 240.0, "o2": 180.0, "anes": 20.0, "n2": 0.0},
        ]
        self.protocol_running = False
        self.protocol_paused = False
        self.current_step_idx = 0
        self.current_step_remaining = 60.0

        self.setup_ui()
        self.bind_signals()

        self.camera_worker.start()
        self.serial_worker.start()

        self.step_timer = QTimer()
        self.step_timer.timeout.connect(self.on_step_timer_tick)
        self.step_timer.start(1000)

    def kill_flowvision_process(self) -> bool:
        names = ["flowvision.exe", "flowvision2.exe", "flowvision"]
        return kill_processes_by_name(names)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 15, 18, 15)
        main_layout.setSpacing(15)

        # ---------------- 1. Top Physical Status Bar ----------------
        top_bar = QWidget()
        h_status = QHBoxLayout(top_bar)
        h_status.setContentsMargins(0, 0, 0, 0)
        h_status.setSpacing(15)

        self.lbl_cam_status = QLabel("Camera: Detecting...")
        self.lbl_cam_status.setStyleSheet(
            "background-color: #D97706; color: white; padding: 8px 18px;"
            " border-radius: 6px; font-weight: 700; font-size: 20px;"
        )

        self.lbl_fv_hw_status = QLabel("FlowVision Hardware: Detecting...")
        self.lbl_fv_hw_status.setStyleSheet(
            "background-color: #D97706; color: white; padding: 8px 18px;"
            " border-radius: 6px; font-weight: 700; font-size: 20px;"
        )

        self.btn_kill_fv = QPushButton("Release COM Port")
        self.btn_kill_fv.setStyleSheet(
            "background-color: #DC2626; color: white; font-weight: 700;"
            " padding: 8px 21px; border-radius: 6px; border: none; font-size: 20px;"
        )

        h_status.addWidget(self.lbl_cam_status)
        h_status.addWidget(self.lbl_fv_hw_status)
        h_status.addWidget(self.btn_kill_fv)
        h_status.addStretch()
        main_layout.addWidget(top_bar)

        # ---------------- 2. Middle Row: Viewport (Left) & Controls (Right) ----------------
        h_mid = QHBoxLayout()
        h_mid.setSpacing(18)

        # Left Column (58%): Video Viewport & Bottom Config/Diagnostics
        v_left = QVBoxLayout()
        v_left.setSpacing(12)

        self.video_viewport = AspectRatioVideoWidget()
        v_left.addWidget(self.video_viewport, 5)

        # Config Strip
        grp_left_config = QGroupBox("Camera Configuration & Serial Interface")
        grid_left_cfg = QGridLayout(grp_left_config)
        grid_left_cfg.setContentsMargins(15, 9, 15, 9)
        grid_left_cfg.setHorizontalSpacing(12)
        grid_left_cfg.setVerticalSpacing(6)

        lbl_pfs_tag = QLabel("Camera PFS:")
        lbl_pfs_tag.setStyleSheet("font-size: 20px;")
        grid_left_cfg.addWidget(lbl_pfs_tag, 0, 0)

        h_pfs = QHBoxLayout()
        h_pfs.setSpacing(9)
        self.txt_pfs = QLineEdit(self.config.pfs_path)
        self.btn_browse_pfs = QPushButton("Browse...")
        self.btn_browse_pfs.setFixedWidth(120)
        h_pfs.addWidget(self.txt_pfs)
        h_pfs.addWidget(self.btn_browse_pfs)
        grid_left_cfg.addLayout(h_pfs, 0, 1)

        lbl_com_tag = QLabel("Serial Port:")
        lbl_com_tag.setStyleSheet("font-size: 20px;")
        grid_left_cfg.addWidget(lbl_com_tag, 1, 0)

        self.txt_com_port = QLineEdit(self.config.fv_com_port)
        grid_left_cfg.addWidget(self.txt_com_port, 1, 1)
        v_left.addWidget(grp_left_config, 0)

        # Diagnostics Log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(135)  # 150% scaled height
        self.txt_log.setPlaceholderText("System runtime events and diagnostics...")
        v_left.addWidget(self.txt_log, 2)
        h_mid.addLayout(v_left, 12)

        # Right Column (42%): Telemetry & Protocol Control (Compact, Auto-wrapping)
        v_right = QVBoxLayout()
        v_right.setSpacing(15)

        # Block 1: Multi-Drop Telemetry Card
        grp_mfc_tabs = QGroupBox("FlowVision Hardware Multi-Drop Telemetry")
        v_tab_layout = QVBoxLayout(grp_mfc_tabs)
        v_tab_layout.setContentsMargins(9, 9, 9, 9)

        self.tab_mfc_devices = QTabWidget()
        v_tab_layout.addWidget(self.tab_mfc_devices)
        self._init_native_mfc_tabs()
        v_right.addWidget(grp_mfc_tabs, 0)

        # Block 2: Protocol Scheduler Card
        grp_scheduler = QGroupBox("Protocol Dynamic Schedule & Controls")
        v_sched_layout = QVBoxLayout(grp_scheduler)
        v_sched_layout.setContentsMargins(18, 15, 18, 15)
        v_sched_layout.setSpacing(12)

        h_proto_header = QHBoxLayout()
        self.lbl_proto_name = QLabel(f"Active Protocol: {self.active_protocol_name}")
        self.lbl_proto_name.setStyleSheet("font-weight: 700; color: #0F172A; font-size: 21px;")
        h_proto_header.addWidget(self.lbl_proto_name)
        h_proto_header.addStretch()
        v_sched_layout.addLayout(h_proto_header)

        self.lbl_step_info = QLabel("Current Phase: Standby (Step 1 / 2)")
        self.lbl_step_info.setStyleSheet("color: #475569; font-weight: 600; font-size: 20px;")
        v_sched_layout.addWidget(self.lbl_step_info)

        self.progress_step = QProgressBar()
        self.progress_step.setTextVisible(True)
        self.progress_step.setValue(0)
        v_sched_layout.addWidget(self.progress_step)

        self.lbl_step_ratio = QLabel("Target Flow: O2: 200.0 | Anes: 0.0 | N2: 0.0 sccm")
        self.lbl_step_ratio.setStyleSheet(
            f"font-family: {UNIFIED_FONT_FAMILY}; color: #334155; font-size: 20px; font-weight: 600;"
        )
        v_sched_layout.addWidget(self.lbl_step_ratio)

        # Scheduler Action Strip
        h_sched_actions = QHBoxLayout()
        h_sched_actions.setSpacing(9)
        self.btn_sched_run = QPushButton("Run / Resume")
        self.btn_sched_extend = QPushButton("+30s Extend")
        self.btn_sched_pause = QPushButton("Pause")
        self.btn_sched_skip = QPushButton("Skip Step")
        h_sched_actions.addWidget(self.btn_sched_run)
        h_sched_actions.addWidget(self.btn_sched_extend)
        h_sched_actions.addWidget(self.btn_sched_pause)
        h_sched_actions.addWidget(self.btn_sched_skip)
        v_sched_layout.addLayout(h_sched_actions)

        self.btn_open_library = QPushButton("Open Protocol Library & Editor...")
        self.btn_open_library.setStyleSheet("font-weight: 700; padding: 9px; font-size: 20px;")
        v_sched_layout.addWidget(self.btn_open_library)

        v_right.addWidget(grp_scheduler, 0)
        v_right.addStretch(1)  # Anchors controls tightly to prevent vacant gaps

        h_mid.addLayout(v_right, 9)
        main_layout.addLayout(h_mid, 1)

        # ---------------- 3. Bottom Execution Strip ----------------
        grp_controls = QGroupBox("Integrated Synchronization Execution")
        h_sync = QHBoxLayout(grp_controls)
        h_sync.setContentsMargins(15, 9, 15, 9)
        h_sync.setSpacing(12)

        self.btn_toggle_cam = QPushButton("Toggle Camera Recording")
        self.btn_toggle_cam.setFixedHeight(51)  # 150% scaled height
        self.btn_toggle_cam.setStyleSheet("font-size: 20px; font-weight: 700;")

        self.btn_sync_start = QPushButton("Start Synchronized Experiment")
        self.btn_sync_start.setFixedHeight(51)
        self.btn_sync_start.setEnabled(False)
        self.btn_sync_start.setStyleSheet(
            "background-color: #059669; color: white; font-weight: 700; border: none; font-size: 20px;"
        )

        self.btn_sync_stop = QPushButton("Stop Synchronized Experiment")
        self.btn_sync_stop.setFixedHeight(51)
        self.btn_sync_stop.setEnabled(False)
        self.btn_sync_stop.setStyleSheet(
            "background-color: #DC2626; color: white; font-weight: 700; border: none; font-size: 20px;"
        )

        h_sync.addWidget(self.btn_toggle_cam, 2)
        h_sync.addWidget(self.btn_sync_start, 4)
        h_sync.addWidget(self.btn_sync_stop, 4)
        main_layout.addWidget(grp_controls, 0)

    def _init_native_mfc_tabs(self):
        """Construct cohesive card surfaces strictly bound to unified font system"""
        self.tab_mfc_devices.clear()
        for uid, meta in SerialPollingWorker.DEVICE_MAP.items():
            tab_w = QWidget()
            v_tab_inner = QVBoxLayout(tab_w)
            v_tab_inner.setContentsMargins(21, 15, 21, 15)
            v_tab_inner.setSpacing(12)

            # Row 1: Header (Device & Gas Medium)
            h_header = QHBoxLayout()
            lbl_name = QLabel(f"Device: Unit {uid} ({meta['name']})")
            lbl_name.setStyleSheet("font-weight: 700; color: #0F172A; font-size: 21px;")

            lbl_gas = QLabel(f"Medium: {meta['gas_default']}")
            lbl_gas.setObjectName("lbl_gas_val")
            lbl_gas.setStyleSheet("color: #64748B; font-size: 20px; font-weight: 600;")

            h_header.addWidget(lbl_name)
            h_header.addStretch()
            h_header.addWidget(lbl_gas)
            v_tab_inner.addLayout(h_header)

            # Row 2: Mass Flow Readout (Clean 33px) + Valve Status
            h_readout = QHBoxLayout()
            lbl_title_flow = QLabel("Mass Flow:")
            lbl_title_flow.setStyleSheet("font-size: 21px; font-weight: 700; color: #334155;")

            lbl_flow = QLabel("0000.0 sccm")
            lbl_flow.setObjectName("lbl_flow_val")
            lbl_flow.setStyleSheet(
                f"font-family: {UNIFIED_FONT_FAMILY}; font-size: 33px; font-weight: 700; color: #0F172A;"
            )

            lbl_status = QLabel("Valve: Standby")
            lbl_status.setObjectName("lbl_status_val")
            lbl_status.setStyleSheet("color: #334155; font-weight: 600; font-size: 20px;")

            h_readout.addWidget(lbl_title_flow)
            h_readout.addWidget(lbl_flow)
            h_readout.addStretch()
            h_readout.addWidget(lbl_status)
            v_tab_inner.addLayout(h_readout)

            # Row 3: Target Setpoint & Action Buttons
            h_controls = QHBoxLayout()
            lbl_title_sp = QLabel("Setpoint:")
            lbl_title_sp.setStyleSheet("font-size: 21px; font-weight: 700; color: #334155;")

            spin_sp = QDoubleSpinBox()
            spin_sp.setObjectName("spin_sp_input")
            spin_sp.setRange(0.0, 5000.0)
            spin_sp.setValue(0.0)
            spin_sp.setSuffix(" sccm")
            spin_sp.setFixedHeight(42)
            spin_sp.setStyleSheet("font-size: 20px; font-weight: 600;")

            btn_set = QPushButton("SET")
            btn_set.setFixedWidth(98)
            btn_set.setFixedHeight(42)
            btn_set.setStyleSheet(
                "font-weight: 700; background-color: #2563EB; color: white; border: none; font-size: 20px;"
            )

            btn_zero = QPushButton("ZERO")
            btn_zero.setFixedWidth(98)
            btn_zero.setFixedHeight(42)
            btn_zero.setStyleSheet(
                "font-weight: 700; background-color: #475569; color: white; border: none; font-size: 20px;"
            )

            h_controls.addWidget(lbl_title_sp)
            h_controls.addWidget(spin_sp, 1)
            h_controls.addWidget(btn_set)
            h_controls.addWidget(btn_zero)

            btn_set.clicked.connect(
                lambda _, u=uid, s=spin_sp: self.serial_worker.queue_setpoint(
                    u, s.value()
                )
            )
            btn_zero.clicked.connect(
                lambda _, u=uid, s=spin_sp: self._zero_and_update(u, s)
            )
            v_tab_inner.addLayout(h_controls)

            # Row 4: Auxiliary Physics Readout
            h_physics = QHBoxLayout()
            lbl_physics = QLabel("P: +014.70 PSIA | T: +025.0 C")
            lbl_physics.setObjectName("lbl_physics_val")
            lbl_physics.setStyleSheet(
                f"font-family: {UNIFIED_FONT_FAMILY}; color: #64748B; font-size: 18px; font-weight: 500;"
            )
            h_physics.addWidget(lbl_physics)
            h_physics.addStretch()
            v_tab_inner.addLayout(h_physics)

            self.tab_mfc_devices.addTab(tab_w, f"{uid}: {meta['name']}")

    def _zero_and_update(self, unit_id: str, spin_box: QDoubleSpinBox):
        spin_box.setValue(0.0)
        self.serial_worker.queue_setpoint(unit_id, 0.0)
        self.log(f"Manual zero command sent to Unit {unit_id}")

    def bind_signals(self):
        # Camera signals
        self.camera_worker.frame_ready.connect(self.video_viewport.update_frame)
        self.camera_worker.status_changed.connect(self.on_camera_status_changed)
        self.camera_worker.pfs_applied_signal.connect(self.on_pfs_applied)

        # Serial worker signals
        self.serial_worker.port_status_signal.connect(
            self.on_serial_status_changed
        )
        self.serial_worker.data_updated_signal.connect(
            self.update_mfc_device_tabs
        )

        # Hardware setup
        self.btn_kill_fv.clicked.connect(self.action_force_kill_flowvision)
        self.btn_browse_pfs.clicked.connect(self.choose_pfs_file)
        self.txt_com_port.textChanged.connect(self.serial_worker.set_port)

        # Protocol schedule actions
        self.btn_open_library.clicked.connect(self.open_protocol_library)
        self.btn_sched_run.clicked.connect(self.action_sched_run_resume)
        self.btn_sched_extend.clicked.connect(self.action_sched_extend)
        self.btn_sched_pause.clicked.connect(self.action_sched_pause)
        self.btn_sched_skip.clicked.connect(self.action_sched_skip)

        # Bottom control actions
        self.btn_toggle_cam.clicked.connect(self.action_toggle_cam)
        self.btn_sync_start.clicked.connect(self.action_sync_start)
        self.btn_sync_stop.clicked.connect(self.action_sync_stop)

    # ================== Protocol Scheduler State Machine ==================
    def on_step_timer_tick(self):
        if not self.protocol_running or self.protocol_paused:
            return

        if self.current_step_remaining > 0:
            self.current_step_remaining -= 1.0

        current_step = self.script_steps[self.current_step_idx]
        step_duration = max(1.0, current_step.get("time", 60.0))
        elapsed = step_duration - self.current_step_remaining
        pct = int(min(100, (elapsed / step_duration) * 100))

        rem_m = int(self.current_step_remaining // 60)
        rem_s = int(self.current_step_remaining % 60)
        self.progress_step.setValue(pct)
        self.lbl_step_info.setText(
            f"Phase {self.current_step_idx + 1} / {len(self.script_steps)}: "
            f"{rem_m:02d}:{rem_s:02d} remaining ({pct}%)"
        )

        if self.current_step_remaining <= 0:
            if self.current_step_idx < len(self.script_steps) - 1:
                self.current_step_idx += 1
                self._apply_step_hardware_flow(self.current_step_idx)
            else:
                self.log("All protocol steps finished successfully.")
                self.protocol_running = False
                self.lbl_step_info.setText("Protocol Completed")

    def _apply_step_hardware_flow(self, step_idx: int):
        if 0 <= step_idx < len(self.script_steps):
            step = self.script_steps[step_idx]
            self.current_step_remaining = step.get("time", 60.0)
            o2_flow = step.get("o2", 200.0)
            anes_flow = step.get("anes", 0.0)
            n2_flow = step.get("n2", 0.0)

            self.serial_worker.queue_setpoint("A", o2_flow)
            self.serial_worker.queue_setpoint("B", anes_flow)
            self.serial_worker.queue_setpoint("C", n2_flow)

            self.lbl_step_ratio.setText(
                f"Target Flow: O2: {o2_flow:.1f} | Anes: {anes_flow:.1f} | N2: {n2_flow:.1f} sccm"
            )
            self.log(
                f"Transitioned to Step {step_idx + 1} -> O2: {o2_flow} sccm, "
                f"Anesthetics: {anes_flow} sccm, N2: {n2_flow} sccm"
            )

    def action_sched_run_resume(self):
        if not self.protocol_running:
            self.protocol_running = True
            self.protocol_paused = False
            self.current_step_idx = 0
            self._apply_step_hardware_flow(0)
            self.log("Protocol schedule started.")
        elif self.protocol_paused:
            self.protocol_paused = False
            self.log("Protocol schedule resumed.")

    def action_sched_extend(self):
        if self.protocol_running:
            self.current_step_remaining += 30.0
            self.log("Extended current phase duration by 30 seconds.")

    def action_sched_pause(self):
        if self.protocol_running:
            self.protocol_paused = not self.protocol_paused
            state_str = "paused" if self.protocol_paused else "resumed"
            self.log(f"Protocol schedule {state_str}.")

    def action_sched_skip(self):
        if self.protocol_running:
            if self.current_step_idx < len(self.script_steps) - 1:
                self.current_step_idx += 1
                self._apply_step_hardware_flow(self.current_step_idx)
                self.log("Skipped to next step.")
            else:
                self.protocol_running = False
                self.log("Skipped final step. Protocol terminated.")
                self.lbl_step_info.setText("Protocol Completed")

    # ================== Dialog & Logging ==================
    def open_protocol_library(self):
        dlg = ProtocolLibraryDialog(
            current_dir="./protocols",
            active_steps=self.script_steps,
            active_onset=self.config.gas_onset_sec,
            parent=self,
        )
        dlg.protocol_applied.connect(self.on_protocol_applied_from_dialog)
        dlg.exec_()

    def on_protocol_applied_from_dialog(
        self, name: str, onset: float, steps: list[dict]
    ):
        self.active_protocol_name = name
        self.config.gas_onset_sec = onset
        self.script_steps = steps
        self.lbl_proto_name.setText(f"Active Protocol: {name}")
        self.lbl_step_info.setText(f"Loaded: 1 / {len(steps)} Phases Configured")

        total_dur = sum(s.get("time", 0.0) for s in steps)
        first_step = steps[0]
        self.lbl_step_ratio.setText(
            f"Initial: O2: {first_step.get('o2', 0.0)} | "
            f"Anes: {first_step.get('anes', 0.0)} | "
            f"N2: {first_step.get('n2', 0.0)} sccm"
        )
        self.log(
            f"Applied protocol '{name}' ({len(steps)} steps, total {total_dur:.1f}s)"
        )

    def log(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{now}] {text}")

    # ================== Status Callbacks ==================
    def on_camera_status_changed(self, state: str, title: str, desc: str):
        self.cam_raw_state = state
        self._refresh_camera_ui_badge()

    def _refresh_camera_ui_badge(self):
        if self.camera_worker.recording:
            self.lbl_cam_status.setText("Camera: Recording (REC)")
            self.lbl_cam_status.setStyleSheet(
                "background-color: #DC2626; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )
            return

        if self.cam_raw_state == "OK":
            self.lbl_cam_status.setText("Camera: Online Ready")
            self.lbl_cam_status.setStyleSheet(
                "background-color: #059669; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )
            self.camera_is_ready = True
            self.btn_sync_start.setEnabled(True)
        elif self.cam_raw_state == "PENDING":
            self.lbl_cam_status.setText("Camera: Connecting...")
            self.lbl_cam_status.setStyleSheet(
                "background-color: #D97706; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )
            self.camera_is_ready = False
            self.btn_sync_start.setEnabled(False)
        else:
            self.lbl_cam_status.setText("Camera: Offline")
            self.lbl_cam_status.setStyleSheet(
                "background-color: #DC2626; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )
            self.camera_is_ready = False
            self.btn_sync_start.setEnabled(False)

    def on_serial_status_changed(self, online: bool, status_text: str):
        self.lbl_fv_hw_status.setText(status_text)
        if online:
            self.lbl_fv_hw_status.setStyleSheet(
                "background-color: #059669; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )
        else:
            self.lbl_fv_hw_status.setStyleSheet(
                "background-color: #D97706; color: white; padding: 8px 18px;"
                " border-radius: 6px; font-weight: 700; font-size: 20px;"
            )

    def update_mfc_device_tabs(self, devices: list[dict]):
        for dev in devices:
            uid = dev.get("unit_id")
            tab_index = {"A": 0, "B": 1, "C": 2}.get(uid, -1)

            if 0 <= tab_index < self.tab_mfc_devices.count():
                tab_w = self.tab_mfc_devices.widget(tab_index)
                if not tab_w:
                    continue

                lbl_flow = tab_w.findChild(QLabel, "lbl_flow_val")
                lbl_status = tab_w.findChild(QLabel, "lbl_status_val")
                lbl_physics = tab_w.findChild(QLabel, "lbl_physics_val")
                lbl_gas = tab_w.findChild(QLabel, "lbl_gas_val")

                if lbl_flow:
                    lbl_flow.setText(f"{dev['flow_sccm']:06.1f} sccm")
                if lbl_status:
                    lbl_status.setText(f"Valve: {dev['valve_status']}")
                if lbl_gas:
                    lbl_gas.setText(f"Medium: {dev['gas_type']}")
                if lbl_physics:
                    lbl_physics.setText(
                        f"P: {dev['pressure_psia']:+07.2f} PSIA | T: {dev['temp_c']:+05.1f} C"
                    )

    def choose_pfs_file(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Basler Camera Feature File",
            "",
            "Pylon Feature Set (*.pfs);;All Files (*.*)",
        )
        if fpath:
            self.txt_pfs.setText(fpath)
            self.camera_worker.set_pfs_path(fpath)
            self.log(f"Selected PFS feature: {Path(fpath).name}")

    def on_pfs_applied(self, success: bool, msg: str):
        self.log(msg)
        if not success:
            QMessageBox.warning(self, "PFS Error", msg)

    def action_force_kill_flowvision(self):
        killed = self.kill_flowvision_process()
        if killed:
            self.log(
                "Terminated background FlowVision process. COM port handle reclaimed."
            )
        else:
            self.log(
                "No background process found. Resuming serial polling thread."
            )
        self.serial_worker.resume_polling()

    # ================== Execution Strip Actions ==================
    def action_toggle_cam(self):
        if not self.camera_worker.recording:
            temp_vid = Path("./debug_camera_recording.mp4").resolve()
            self.camera_worker.start_recording(str(temp_vid))
            self._refresh_camera_ui_badge()
            self.log(f"Independent recording started: {temp_vid.name}")
            self.btn_toggle_cam.setText("Stop Camera Recording")
        else:
            self.camera_worker.stop_recording()
            self._refresh_camera_ui_badge()
            self.log("Independent recording stopped.")
            self.btn_toggle_cam.setText("Toggle Camera Recording")

    def action_sync_start(self):
        if not self.camera_is_ready:
            QMessageBox.warning(
                self, "Warning", "Camera is not ready. Aborting start."
            )
            return

        total_dur = sum(s.get("time", 0.0) for s in self.script_steps)
        self.config.experiment_duration = total_dur
        self.config.fv_com_port = self.txt_com_port.text().strip()
        self.config.pfs_path = self.txt_pfs.text().strip()

        script_text = FlowVisionScriptModel.generate_script_text(
            port=self.config.fv_com_port,
            gas_onset=self.config.gas_onset_sec,
            duration=total_dur,
            steps=self.script_steps,
        )

        (
            self.active_session_dir,
            script_path,
            self.active_video_path,
            _,
        ) = self.session_mgr.create_session(self.config, script_text)
        self.log(f"Created Session Directory: {self.active_session_dir.name}")

        self.camera_worker.start_recording(str(self.active_video_path))
        self._refresh_camera_ui_badge()
        self.log(f"Camera recording started: {self.active_video_path.name}")

        self.protocol_running = True
        self.protocol_paused = False
        self.current_step_idx = 0
        self._apply_step_hardware_flow(0)

        self.btn_sync_start.setEnabled(False)
        self.btn_sync_stop.setEnabled(True)

    def action_sync_stop(self):
        self.camera_worker.stop_recording()
        self._refresh_camera_ui_badge()
        self.log("Video saved to session directory.")

        self.protocol_running = False
        self.lbl_step_info.setText("Protocol Stopped")

        self.serial_worker.queue_setpoint("A", 0.0)
        self.serial_worker.queue_setpoint("B", 0.0)
        self.serial_worker.queue_setpoint("C", 0.0)
        self.log("Safety mechanism: Sent zero commands to all gas channels.")

        self.btn_sync_start.setEnabled(True)
        self.btn_sync_stop.setEnabled(False)

        QMessageBox.information(
            self,
            "Experiment Finished",
            f"Session data archived:\n{self.active_session_dir}\n"
            f"Video file:\n{self.active_video_path}\n"
            "All gas valves set to 0.0 sccm.",
        )

    def closeEvent(self, event):
        self.step_timer.stop()
        self.serial_worker.terminate_worker()
        self.kill_flowvision_process()
        self.camera_worker.terminate_worker()
        clean_all_camera_resources()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(SCALED_150_SCIENTIFIC_QSS)
    window = HardwareIntegrationApp()
    window.show()
    sys.exit(app.exec_())