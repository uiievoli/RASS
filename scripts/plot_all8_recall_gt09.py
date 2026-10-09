#!/usr/bin/env python3
"""Canonical entry point for the active eight-dataset recall plots."""

from pathlib import Path
import runpy
import sys

if not any(arg == "--datasets" or arg.startswith("--datasets=") for arg in sys.argv[1:]):
    sys.argv.extend(["--datasets", "nuswide,fasion_mnist_784,msong_holdout,sift1m,glove25,StarLightCurves,dbpedia1536m_holdout,sift1b"])

runpy.run_path(
    str(Path(__file__).with_name("plot_all10_recall_gt09.py")),
    run_name="__main__",
)
