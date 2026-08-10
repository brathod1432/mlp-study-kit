"""
conftest.py -- pytest configuration for mlp-study-kit.

Adds src/ to sys.path so that both ``import nn_core`` and
``import modules`` work in every test without requiring
``pip install -e .`` first.

src/ contains two packages:
  nn_core/   — MLP building blocks (activations, losses, network, logger)
  modules/   — helper utilities (general_utils, plot_utils, data_utils)
"""
import os
import sys

# Force headless matplotlib for all tests (no display window opens)
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure src/ is on path regardless of where pytest is invoked from
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
