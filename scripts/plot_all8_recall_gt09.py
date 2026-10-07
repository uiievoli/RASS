#!/usr/bin/env python3
"""Canonical entry point for the active eight-dataset recall plots."""

from pathlib import Path
import runpy

runpy.run_path(
    str(Path(__file__).with_name("plot_all10_recall_gt09.py")),
    run_name="__main__",
)
