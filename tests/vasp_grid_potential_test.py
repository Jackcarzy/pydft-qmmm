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


# e / (4 * pi * eps0) in eV*Angstrom/e -- the Coulomb constant.
COULOMB = 14.399645


class TestPoisson:

    def test_satisfies_poissons_equation(self):
        # Apply the Laplacian back to phi in reciprocal space and
        # recover rho.  Exact up to the G=0 term, so cell-size
        # independent -- the strongest available check.
        rho = vasp_plugin.spread_gaussian(
            np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.5,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        phi_g = np.fft.fftn(phi)
        g_squared = vasp_plugin._g_squared(SHAPE, CELL)
        recovered = np.fft.ifftn(phi_g * g_squared * vasp_plugin.EPS0).real
        assert recovered == pytest.approx(rho - rho.mean(), abs=1e-9)

    def test_positive_charge_lowers_electron_potential_energy(self):
        # The single likeliest bug, asserted alone.  Electrons are
        # attracted to a positive charge, so V_ext must be negative.
        v_ext = vasp_plugin.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert v_ext[24, 24, 24] < 0.0

    def test_electron_potential_is_the_negated_electrostatic_potential(self):
        # Units are asserted on phi (volts) separately from the sign, so
        # a unit error cannot cancel against a sign error.
        positions = np.array([[5.0, 5.0, 5.0]])
        charges = np.array([1.0])
        rho = vasp_plugin.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        v_ext = vasp_plugin.build_external_potential(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        assert phi[24, 24, 24] > 0.0
        assert v_ext == pytest.approx(-phi, abs=1e-12)

    def test_neutral_system_is_insensitive_to_the_g0_convention(self):
        rho = vasp_plugin.spread_gaussian(
            np.array([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.4,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        assert phi.mean() == pytest.approx(0.0, abs=1e-10)

    def test_charged_system_has_zero_mean_potential(self):
        # The G=0 choice fixes the average potential at zero, which is
        # the uniform neutralizing background convention.
        rho = vasp_plugin.spread_gaussian(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        assert phi.mean() == pytest.approx(0.0, abs=1e-10)


class TestInterpolation:

    def test_reproduces_grid_values_at_grid_points(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        # Grid point (3, 5, 7) sits at 10 Angstrom * index / 48.
        point = np.array([[10.0 * 3 / 48, 10.0 * 5 / 48, 10.0 * 7 / 48]])
        got = vasp_plugin.interpolate_at(field, CELL, point)
        assert got[0] == pytest.approx(field[3, 5, 7], rel=1e-9)

    def test_interpolation_is_periodic(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        inside = np.array([[1.0, 2.0, 3.0]])
        shifted = inside + np.array([10.0, -10.0, 20.0])
        assert vasp_plugin.interpolate_at(field, CELL, inside) == pytest.approx(
            vasp_plugin.interpolate_at(field, CELL, shifted), rel=1e-9,
        )

    def test_gradient_is_exact_for_a_fourier_mode(self):
        # The reciprocal-space derivative is exact for a band-limited
        # field, so a single mode pins it down to machine precision.
        # This is the rigorous test of gradient_at; comparing against
        # finite differences of the TRILINEAR interpolant cannot be,
        # because that interpolant is only C0 -- see the test below.
        # Probe ON grid nodes, so trilinear interpolation is exact and
        # only the derivative is under test.  Off-node probes carry an
        # additional O(h**2) interpolation error of about 0.2% here,
        # which would mask the thing being measured.
        length = 10.0
        spacing = length / SHAPE[0]
        axis_values = np.arange(SHAPE[0]) * spacing
        field = np.sin(2 * np.pi * axis_values / length)[:, None, None]
        field = np.broadcast_to(field, SHAPE).copy()
        nodes = np.array([6, 29])
        points = np.stack(
            [nodes * spacing, np.full(2, 4 * spacing), np.full(2, 7 * spacing)],
            axis=-1,
        )
        got = vasp_plugin.gradient_at(field, CELL, points)
        expected_x = (
            2 * np.pi / length * np.cos(2 * np.pi * points[:, 0] / length)
        )
        assert got[:, 0] == pytest.approx(expected_x, rel=1e-9)
        assert got[:, 1] == pytest.approx(np.zeros(2), abs=1e-9)
        assert got[:, 2] == pytest.approx(np.zeros(2), abs=1e-9)

    def test_gradient_agrees_with_interpolated_finite_differences(self):
        # Consistency check against interpolate_at.  The step must be
        # comparable to the grid spacing: trilinear interpolation is
        # piecewise linear, so a much smaller step just returns the
        # constant slope inside one grid cell, which differs from the
        # smooth spectral derivative by O(h).
        v_ext = vasp_plugin.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.8,
        )
        point = np.array([[6.5, 5.3, 4.7]])
        analytic = vasp_plugin.gradient_at(v_ext, CELL, point)[0]
        step = 10.0 / SHAPE[0]
        numerical = np.zeros(3)
        for axis in range(3):
            shift = np.zeros((1, 3))
            shift[0, axis] = step
            plus = vasp_plugin.interpolate_at(v_ext, CELL, point + shift)[0]
            minus = vasp_plugin.interpolate_at(v_ext, CELL, point - shift)[0]
            numerical[axis] = (plus - minus) / (2 * step)
        assert analytic == pytest.approx(numerical, rel=0.05)


class TestCoulombLimit:
    """Absolute-magnitude checks, which need interpolation."""

    def test_coulomb_prefactor_from_a_potential_difference(self):
        # Absolute phi carries a constant offset from the G=0 choice, so
        # compare a DIFFERENCE, which cancels it.  The residual is
        # periodic-image error.
        cell = np.diag([30.0, 30.0, 30.0])
        shape = (150, 150, 150)
        phi = vasp_plugin.poisson_fft(
            vasp_plugin.spread_gaussian(
                np.array([[15.0, 15.0, 15.0]]), np.array([1.0]),
                shape, cell, sigma=0.3,
            ),
            cell,
        )
        near = vasp_plugin.interpolate_at(
            phi, cell, np.array([[17.0, 15.0, 15.0]]),
        )[0]
        far = vasp_plugin.interpolate_at(
            phi, cell, np.array([[19.0, 15.0, 15.0]]),
        )[0]
        expected = COULOMB * (1.0 / 2.0 - 1.0 / 4.0)
        assert near - far == pytest.approx(expected, rel=0.03)

    def test_converges_to_the_point_charge_limit_as_sigma_shrinks(self):
        # A neutral pair, so the G=0 convention plays no part.  Probe
        # much closer to +q than to -q, where the point-charge value is
        # known.
        #
        # The residual falls monotonically while smearing error
        # dominates and then PLATEAUS at a sigma-independent floor set
        # by periodic images -- measured at 4.68% for this cell, and
        # identical to 5 decimals for every sigma <= 0.4.  So the test
        # asserts monotone decrease only over the sigma-dominated
        # range, and separately pins the floor.
        cell = np.diag([24.0, 24.0, 24.0])
        shape = (120, 120, 120)
        plus = np.array([8.0, 12.0, 12.0])
        minus = np.array([20.0, 12.0, 12.0])
        probe = np.array([[10.0, 12.0, 12.0]])
        expected = COULOMB * (1.0 / 2.0 - 1.0 / 10.0)

        def residual(sigma):
            phi = vasp_plugin.poisson_fft(
                vasp_plugin.spread_gaussian(
                    np.array([plus, minus]), np.array([1.0, -1.0]),
                    shape, cell, sigma,
                ),
                cell,
            )
            return abs(
                vasp_plugin.interpolate_at(phi, cell, probe)[0] - expected,
            )

        sigma_dominated = [residual(s) for s in (1.2, 0.8, 0.6)]
        assert sigma_dominated[1] < sigma_dominated[0]
        assert sigma_dominated[2] < sigma_dominated[1]
        # Below the knee the residual is periodic-image error only.
        assert residual(0.4) == pytest.approx(residual(0.25), rel=1e-3)
        assert residual(0.25) < 0.05 * expected
