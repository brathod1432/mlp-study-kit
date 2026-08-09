"""
conftest.py -- pytest configuration for mlp-study-kit.

Adds src/ to sys.path so that `import nn_core` works in every test
without requiring `pip install -e .` first.
"""
import sys
import os

# Ensure src/ is on path regardless of where pytest is invoked from
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))
