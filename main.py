#!/usr/bin/env python3
"""
Root Application Entry Point
============================
Launches the Drosophila Behavior & Anesthesia Tracking Suite from repository root.

Usage:
    python main.py                  # Launch Desktop GUI (Default)
    python main.py --gui            # Launch Desktop GUI
    python main.py --csv file.csv   # Run batch CLI pipeline
"""

import sys
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
python_suite_dir = os.path.join(root_dir, "python_suite")

if python_suite_dir not in sys.path:
    sys.path.insert(0, python_suite_dir)

if __name__ == "__main__":
    from main import main
    main()
