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


CELL = np.diag([10.0, 10.0, 10.0])
SHAPE = (48, 48, 48)


class TestSpreadGaussian:

    def test_conserves_charge(self):
        positions = np.array([[5.0, 5.0, 5.0], [2.0, 3.0, 4.0]])
        charges = np.array([1.0, -0.5])
        rho = vasp_plugin.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        d_volume = abs(np.linalg.det(CELL)) / np.prod(SHAPE)
        assert rho.sum() * d_volume == pytest.approx(0.5, abs=1e-6)

    def test_peak_is_at_the_charge(self):
        positions = np.array([[5.0, 5.0, 5.0]])
        rho = vasp_plugin.spread_gaussian(
            positions, np.array([1.0]), SHAPE, CELL, sigma=0.4,
        )
        peak = np.unravel_index(np.argmax(rho), SHAPE)
        # 5.0 Angstrom on a 10 Angstrom / 48 point axis -> index 24.
        assert peak == (24, 24, 24)

    def test_wraps_across_the_periodic_boundary(self):
        # A charge on the origin must not pile up at the far face;
        # minimum-image wrapping makes the density symmetric about it.
        rho = vasp_plugin.spread_gaussian(
            np.array([[0.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert rho[1, 24, 24] == pytest.approx(rho[-1, 24, 24], rel=1e-9)

    def test_empty_selection_gives_zero_density(self):
        rho = vasp_plugin.spread_gaussian(
            np.zeros((0, 3)), np.zeros(0), SHAPE, CELL, sigma=0.4,
        )
        assert rho.shape == SHAPE
        assert np.all(rho == 0.0)
