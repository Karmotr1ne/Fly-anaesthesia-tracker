"""
Top-level Package Entry Point
=============================
Allows executing the suite via:
    python -m drosophila_suite
    python -m drosophila_suite --gui
    python -m drosophila_suite --csv path/to/file.csv
"""

import sys
import os

# Ensure package root is in sys.path
package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from drosophila_suite.cli import main as cli_main
from drosophila_suite.gui_app import run_gui


def main():
    if len(sys.argv) == 1 or "--gui" in sys.argv:
        run_gui()
    else:
        cli_main()


if __name__ == "__main__":
    main()
