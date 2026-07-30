"""Tier 1 tests for the VASP embedding grid physics.

These require no VASP binary: the physics lives in pure-numpy functions
so it can be exercised directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp import vasp_utils


class TestMMChargeHandoff:

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        positions = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]])
        charges = np.array([-0.834, 0.417])
        vasp_utils.write_mm_charges(
            path, positions, charges, step=7, sigma=0.45,
        )
        got_pos, got_q, got_step, got_sigma = vasp_plugin.read_mm_charges(path)
        assert got_step == 7
        assert got_sigma == pytest.approx(0.45, abs=1e-12)
        assert got_pos == pytest.approx(positions, abs=1e-12)
        assert got_q == pytest.approx(charges, abs=1e-12)

    def test_empty_selection_round_trips(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.zeros((0, 3)), np.zeros(0), step=0, sigma=0.3,
        )
        got_pos, got_q, _, _ = vasp_plugin.read_mm_charges(path)
        assert got_pos.shape == (0, 3)
        assert got_q.shape == (0,)

    def test_stale_step_is_rejected(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.array([[0.0, 0.0, 0.0]]), np.array([1.0]),
            step=3, sigma=0.3,
        )
        with pytest.raises(ValueError, match="step"):
            vasp_plugin.read_mm_charges(path, expect_step=4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vasp_plugin.read_mm_charges(str(tmp_path / "nope"))

    def test_mismatched_count_is_rejected(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        with open(path, "w") as fh:
            fh.write("3 0 3.0e-01\n")
            fh.write("0.0 0.0 0.0 1.0\n")
        with pytest.raises(ValueError, match="declares"):
            vasp_plugin.read_mm_charges(path)

    def test_length_mismatch_is_rejected_on_write(self, tmp_path):
        with pytest.raises(ValueError, match="positions"):
            vasp_utils.write_mm_charges(
                str(tmp_path / "MM_CHARGES"),
                np.zeros((2, 3)), np.zeros(3), step=0, sigma=0.3,
            )
