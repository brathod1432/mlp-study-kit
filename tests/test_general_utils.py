"""Tests for modules.general_utils."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modules.general_utils import (
    as_float_array,
    describe_array,
    ensure_directory,
    ensure_matrix,
    ensure_vector,
    print_matrices,
)


class TestEnsureDirectory:
    def test_creates_new_directory(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = ensure_directory(target)
        assert result.is_dir()

    def test_returns_path_object(self, tmp_path):
        result = ensure_directory(tmp_path / "new_dir")
        assert isinstance(result, Path)

    def test_existing_directory_no_error(self, tmp_path):
        ensure_directory(tmp_path)   # already exists — should not raise
        assert tmp_path.is_dir()

    def test_accepts_string_path(self, tmp_path):
        result = ensure_directory(str(tmp_path / "string_dir"))
        assert result.is_dir()


class TestAsFloatArray:
    def test_converts_list(self):
        arr = as_float_array([1, 2, 3])
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_ndim_check_passes(self):
        arr = as_float_array([[1, 2], [3, 4]], ndim=2)
        assert arr.ndim == 2

    def test_ndim_check_fails(self):
        with pytest.raises(ValueError, match="dimension"):
            as_float_array([1, 2, 3], name="w", ndim=2)

    def test_shape_check_passes(self):
        arr = as_float_array([[1, 2], [3, 4]], shape=(2, 2))
        assert arr.shape == (2, 2)

    def test_shape_check_fails(self):
        with pytest.raises(ValueError, match="shape"):
            as_float_array([[1, 2], [3, 4]], shape=(3, 2))

    def test_scalar_input(self):
        arr = as_float_array(5.0)
        assert arr.dtype == np.float64


class TestEnsureVector:
    def test_valid_1d(self):
        v = ensure_vector([1.0, 2.0, 3.0])
        assert v.ndim == 1
        assert v.dtype == np.float64

    def test_length_check_passes(self):
        v = ensure_vector([1, 2, 3], length=3)
        assert v.size == 3

    def test_length_check_fails(self):
        with pytest.raises(ValueError, match="elements"):
            ensure_vector([1, 2, 3], length=4)

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="dimension"):
            ensure_vector([[1, 2], [3, 4]])


class TestEnsureMatrix:
    def test_valid_2d(self):
        m = ensure_matrix([[1, 2], [3, 4]])
        assert m.ndim == 2

    def test_shape_check_passes(self):
        m = ensure_matrix([[1, 2], [3, 4]], shape=(2, 2))
        assert m.shape == (2, 2)

    def test_shape_check_fails(self):
        with pytest.raises(ValueError, match="shape"):
            ensure_matrix([[1, 2]], shape=(2, 2))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="dimension"):
            ensure_matrix([1, 2, 3])


class TestDescribeArray:
    def test_scalar(self):
        assert describe_array(5.0) == "scalar"

    def test_1d(self):
        assert describe_array([1, 2, 3]) == "3"

    def test_2d(self):
        assert describe_array([[1, 2], [3, 4]]) == "2×2"

    def test_3d(self):
        arr = np.zeros((2, 3, 4))
        assert describe_array(arr) == "2×3×4"


class TestPrintMatrices:
    def test_runs_without_error(self, capsys):
        print_matrices([np.array([1.0, 2.0]), np.eye(2)])
        out = capsys.readouterr().out
        assert out != ""

    def test_uses_names(self, capsys):
        print_matrices([np.eye(2)], names=["MyMatrix"])
        out = capsys.readouterr().out
        assert "MyMatrix" in out
