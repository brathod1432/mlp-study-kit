"""
conftest.py -- pytest configuration for mlp-study-kit.

Adds the project root to sys.path so that both ``import nn_core`` and
``import modules`` work in every test without requiring
``pip install -e .`` first.

Both packages now live at the project root (flat layout):
  nn_core/   — MLP building blocks (activations, losses, network, logger)
  modules/   — helper utilities (general_utils, plot_utils, data_utils, metrics)
"""
import os
import sys

# Force headless matplotlib for all tests (no display window opens)
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure the project root is on sys.path so nn_core and modules are importable
# without requiring pip install -e . first.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
