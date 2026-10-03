"""Directories the plots and tables read results from.

Each can be overridden with the environment variable next to it, e.g.
LESS_RESULTS=logs/results_other.
"""
import os

MAIN = os.environ.get("LESS_RESULTS", "logs/results")
VARIANTS = os.environ.get("LESS_VARIANTS", "logs/variants")
ABLATION = os.environ.get("LESS_ABLATION", "logs/ablation")
INCREMENTAL = os.environ.get("LESS_INCREMENTAL", "logs/incremental")
TIMING = os.environ.get("LESS_TIMING", "logs/timing")
