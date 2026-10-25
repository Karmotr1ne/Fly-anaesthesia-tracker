"""core/protocol/flowvision_model.py"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import math
from core.hardware.protocol_scheduler import ProtocolScheduler

@dataclass
class HardwareConfig:
    fv_com_port: str = "COM3"
    mix_onset: float = 0.0
    experiment_duration: float = 300.0
    pfs_path: str = "Fly Presetting_4 column 30fps.pfs"
    recording_fps: float = 30.0
    serial_baudrate: int = 19200

class FlowVisionScriptModel:
    @staticmethod
    def parse_script(file_path: str | Path) -> dict:
        data = {
            "port": "COM3",
            'mix_onset': 0.0,
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
                    elif k.lower() in ("mixgasonsettime", "mix_onset", "mix_onset_sec"):
                        data['mix_onset'] = float(v)
                    elif k.lower() in ("duration", "totalduration"):
                        data["total_duration"] = float(v)
                elif current_section.startswith("Step"):
                    if k.lower() == "time":
                        current_step["time"] = float(v)
                    elif k.lower() in ("o2", "unit_a_o2"):
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
        port: str, mix_onset: float, duration: float, steps: list[dict]
    ) -> str:
        steps = ProtocolScheduler.validate_steps(steps)
        mix_onset = float(mix_onset)
        duration = sum(s['time'] for s in steps)
        if not math.isfinite(mix_onset) or mix_onset < 0:
            raise ValueError("Mix gas onset must be finite and nonnegative")
        lines = [
            "# FlowVision2 Protocol Control Script",
            f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "[GasDeliveryControl]",
            f"Port = {port}",
            f"MixGasOnsetTime = {mix_onset:.12g}",
            f"Duration = {duration:.12g}",
            f"TotalSteps = {len(steps)}",
            "",
        ]
        for idx, s in enumerate(steps, start=1):
            lines.append(f"[Step_{idx}]")
            lines.append(f"Time = {s.get('time', 60.0):.12g}")
            lines.append(f"Unit_A_O2 = {s.get('o2', 200.0):.12g}")
            lines.append(f"Unit_B_Anesthetics = {s.get('anes', 0.0):.12g}")
            lines.append(f"Unit_C_N2 = {s.get('n2', 0.0):.12g}")
            lines.append("")
        return "\n".join(lines)
    @staticmethod
    def export_script(file_path, steps, mix_onset, total_duration, port="COM3"):
        text = FlowVisionScriptModel.generate_script_text(port, mix_onset, total_duration, steps)
        Path(file_path).write_text(text, encoding="utf-8")
