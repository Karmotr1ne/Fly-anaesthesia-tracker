'Create session folders and save acquisition settings before recording.'
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from core.protocol.flowvision_model import HardwareConfig
from core.hardware.protocol_scheduler import ProtocolScheduler
from .layout import SessionPaths


class SessionManager:

    def __init__(self, workspace_root: str = "./workspace"):
        self.workspace_root = Path(workspace_root)

    def create_session(
        self,
        config: HardwareConfig,
        script_content: str,
        steps: list[dict] | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        if steps is not None:
            steps = ProtocolScheduler.validate_steps(steps)
            config = replace(config, experiment_duration=sum(step['time'] for step in steps))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        total_dur = int(config.experiment_duration)
        anes_val = 0.0

        if steps:
            anes_rates = [float(s.get("anes", 0.0)) for s in steps]
            anes_val = max(anes_rates) if anes_rates else 0.0
            total_dur = int(sum(float(s.get("time", 60.0)) for s in steps))
        elif script_content:
            matches = re.findall(
                r"Unit_B_Anesthetics\s*=\s*([\d\.]+)",
                script_content,
                re.IGNORECASE,
            )
            if matches:
                anes_val = max(float(m) for m in matches)

        anes_tag = (
            f"{int(anes_val)}Anes"
            if anes_val.is_integer()
            else f"{anes_val:.1f}Anes"
        )
        session_folder_name = f"Session_{ts}_{anes_tag}_{total_dur}s"

        session_dir = self.workspace_root / "sessions" / session_folder_name
        paths = SessionPaths(session_dir)
        paths.ensure()

        with open(
            paths.logs / "runtime_config.json", "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(config), f, indent=4)

        if steps is not None:
            (paths.logs / "protocol_steps.json").write_text(
                json.dumps(steps, indent=4), encoding="utf-8")

        if config.pfs_path and Path(config.pfs_path).is_file():
            shutil.copyfile(config.pfs_path, paths.logs / "active_camera.pfs")

        (paths.logs / "active_control.fv").write_text(
            script_content, encoding="utf-8"
        )

        active_video_path = paths.raw / "raw_video.mp4"
        active_telemetry_csv = paths.logs / "sync_telemetry.csv"

        return (
            session_dir,
            active_video_path,
            active_telemetry_csv,
            paths.analysis,
        )
