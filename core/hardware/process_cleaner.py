'Explicit release actions for external camera and FlowVision processes.'
from __future__ import annotations
from pathlib import Path
import psutil

try:
    from pypylon import pylon
    PYLON_INSTALLED = True
except ImportError:
    PYLON_INSTALLED = False

def kill_processes_by_name(target_names: list[str]) -> bool:
    targets = {Path(name).stem.casefold() for name in target_names}
    terminated = False
    for proc in psutil.process_iter(["name"]):
        try:
            p_name = proc.info["name"]
            if p_name and Path(p_name).stem.casefold() in targets:
                proc.kill()
                terminated = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return terminated

def clean_all_camera_resources():
    kill_processes_by_name(["pylonviewer.exe", "pylonviewerapp.exe", "pylonviewer"])
    if PYLON_INSTALLED:
        try:
            tl_factory = pylon.TlFactory.GetInstance()
            tl_factory.ReleaseTl()
        except Exception:
            pass

def clean_flowvision_processes() -> bool:
    return kill_processes_by_name(["flowvision.exe", "flowvision2.exe", "flowvision"])
