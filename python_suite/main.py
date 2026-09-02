#!/usr/bin/env python3
"""
Drosophila Behavior & Anesthesia Tracking Suite
================================================
Main Application Entry Point for standalone Python executions and PyInstaller binaries.

Usage:
    python main.py                  # Launch Desktop GUI (Default)
    python main.py --gui            # Launch Desktop GUI explicitly
    python main.py --csv input.csv  # Run Headless CLI pipeline
    python main.py --help           # Show CLI arguments
"""

import sys
import os

# Robustly ensure python_suite directory and its parent are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    # If no arguments or explicitly requesting GUI, launch PyQt6 Desktop GUI
    gui_requested = ("--gui" in sys.argv) or (len(sys.argv) == 1)

    if gui_requested:
        # Filter out --gui flag if present before passing to QApplication
        if "--gui" in sys.argv:
            sys.argv.remove("--gui")
        try:
            from drosophila_suite.gui_app import run_gui
            run_gui()
        except ImportError as e:
            print(f"[Warning] GUI dependencies not available ({e}). Falling back to CLI mode.\n")
            from drosophila_suite.cli import main as cli_main
            cli_main()
    else:
        from drosophila_suite.cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
