"""Tier 1 tests for the VASP embedding grid physics.

These require no VASP binary: the physics lives in pure-numpy functions
so it can be exercised directly.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import grid_potential
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
        got_pos, got_q, got_step, got_sigma = grid_potential.read_mm_charges(path)
        assert got_step == 7
        assert got_sigma == pytest.approx(0.45, abs=1e-12)
        assert got_pos == pytest.approx(positions, abs=1e-12)
        assert got_q == pytest.approx(charges, abs=1e-12)

    def test_empty_selection_round_trips(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.zeros((0, 3)), np.zeros(0), step=0, sigma=0.3,
        )
        got_pos, got_q, _, _ = grid_potential.read_mm_charges(path)
        assert got_pos.shape == (0, 3)
        assert got_q.shape == (0,)

    def test_stale_step_is_rejected(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.array([[0.0, 0.0, 0.0]]), np.array([1.0]),
            step=3, sigma=0.3,
        )
        with pytest.raises(ValueError, match="step"):
            grid_potential.read_mm_charges(path, expect_step=4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            grid_potential.read_mm_charges(str(tmp_path / "nope"))

    def test_mismatched_count_is_rejected(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        with open(path, "w") as fh:
            fh.write("3 0 3.0e-01\n")
            fh.write("0.0 0.0 0.0 1.0\n")
        with pytest.raises(ValueError, match="declares"):
            grid_potential.read_mm_charges(path)

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
        rho = grid_potential.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        d_volume = abs(np.linalg.det(CELL)) / np.prod(SHAPE)
        assert rho.sum() * d_volume == pytest.approx(0.5, abs=1e-6)

    def test_peak_is_at_the_charge(self):
        positions = np.array([[5.0, 5.0, 5.0]])
        rho = grid_potential.spread_gaussian(
            positions, np.array([1.0]), SHAPE, CELL, sigma=0.4,
        )
        peak = np.unravel_index(np.argmax(rho), SHAPE)
        # 5.0 Angstrom on a 10 Angstrom / 48 point axis -> index 24.
        assert peak == (24, 24, 24)

    def test_wraps_across_the_periodic_boundary(self):
        # A charge on the origin must not pile up at the far face;
        # minimum-image wrapping makes the density symmetric about it.
        rho = grid_potential.spread_gaussian(
            np.array([[0.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert rho[1, 24, 24] == pytest.approx(rho[-1, 24, 24], rel=1e-9)

    def test_cutoff_matches_full_evaluation(self):
        # A large cutoff trips the fallback that evaluates every grid
        # point, so this compares the fast path against the exact one.
        positions = np.array([[5.0, 5.0, 5.0], [2.0, 3.0, 4.0]])
        charges = np.array([1.0, -0.5])
        fast = grid_potential.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        exact = grid_potential.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4, cutoff=1000.0,
        )
        assert fast == pytest.approx(exact, abs=1e-9)

    def test_conserves_charge_at_production_scale(self):
        # 1149 charges on a 96**3 grid is the SPC/E test system.  The
        # O(N_charges * N_grid) version took 191 s here; the cutoff
        # version must stay both fast and charge conserving.
        rng = np.random.RandomState(0)
        cell = np.diag([29.899, 29.899, 29.899])
        shape = (96, 96, 96)
        positions = rng.uniform(0.0, 29.899, (1149, 3))
        charges = rng.uniform(-1.0, 1.0, 1149)
        rho = grid_potential.spread_gaussian(
            positions, charges, shape, cell, sigma=0.3,
        )
        d_volume = abs(np.linalg.det(cell)) / np.prod(shape)
        assert rho.sum() * d_volume == pytest.approx(charges.sum(), abs=1e-5)

    def test_empty_selection_gives_zero_density(self):
        rho = grid_potential.spread_gaussian(
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
        rho = grid_potential.spread_gaussian(
            np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.5,
        )
        phi = grid_potential.poisson_fft(rho, CELL)
        phi_g = np.fft.fftn(phi)
        g_squared = grid_potential._g_squared(SHAPE, CELL)
        recovered = np.fft.ifftn(phi_g * g_squared * grid_potential.EPS0).real
        assert recovered == pytest.approx(rho - rho.mean(), abs=1e-9)

    def test_positive_charge_lowers_electron_potential_energy(self):
        # The single likeliest bug, asserted alone.  Electrons are
        # attracted to a positive charge, so V_ext must be negative.
        v_ext = grid_potential.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert v_ext[24, 24, 24] < 0.0

    def test_electron_potential_is_the_negated_electrostatic_potential(self):
        # Units are asserted on phi (volts) separately from the sign, so
        # a unit error cannot cancel against a sign error.
        positions = np.array([[5.0, 5.0, 5.0]])
        charges = np.array([1.0])
        rho = grid_potential.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        phi = grid_potential.poisson_fft(rho, CELL)
        v_ext = grid_potential.build_external_potential(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        assert phi[24, 24, 24] > 0.0
        assert v_ext == pytest.approx(-phi, abs=1e-12)

    def test_neutral_system_is_insensitive_to_the_g0_convention(self):
        rho = grid_potential.spread_gaussian(
            np.array([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.4,
        )
        phi = grid_potential.poisson_fft(rho, CELL)
        assert phi.mean() == pytest.approx(0.0, abs=1e-10)

    def test_charged_system_has_zero_mean_potential(self):
        # The G=0 choice fixes the average potential at zero, which is
        # the uniform neutralizing background convention.
        rho = grid_potential.spread_gaussian(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        phi = grid_potential.poisson_fft(rho, CELL)
        assert phi.mean() == pytest.approx(0.0, abs=1e-10)


class TestInterpolation:

    def test_reproduces_grid_values_at_grid_points(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        # Grid point (3, 5, 7) sits at 10 Angstrom * index / 48.
        point = np.array([[10.0 * 3 / 48, 10.0 * 5 / 48, 10.0 * 7 / 48]])
        got = grid_potential.interpolate_at(field, CELL, point)
        assert got[0] == pytest.approx(field[3, 5, 7], rel=1e-9)

    def test_interpolation_is_periodic(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        inside = np.array([[1.0, 2.0, 3.0]])
        shifted = inside + np.array([10.0, -10.0, 20.0])
        assert grid_potential.interpolate_at(field, CELL, inside) == pytest.approx(
            grid_potential.interpolate_at(field, CELL, shifted), rel=1e-9,
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
        got = grid_potential.gradient_at(field, CELL, points)
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
        v_ext = grid_potential.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.8,
        )
        point = np.array([[6.5, 5.3, 4.7]])
        analytic = grid_potential.gradient_at(v_ext, CELL, point)[0]
        step = 10.0 / SHAPE[0]
        numerical = np.zeros(3)
        for axis in range(3):
            shift = np.zeros((1, 3))
            shift[0, axis] = step
            plus = grid_potential.interpolate_at(v_ext, CELL, point + shift)[0]
            minus = grid_potential.interpolate_at(v_ext, CELL, point - shift)[0]
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
        phi = grid_potential.poisson_fft(
            grid_potential.spread_gaussian(
                np.array([[15.0, 15.0, 15.0]]), np.array([1.0]),
                shape, cell, sigma=0.3,
            ),
            cell,
        )
        near = grid_potential.interpolate_at(
            phi, cell, np.array([[17.0, 15.0, 15.0]]),
        )[0]
        far = grid_potential.interpolate_at(
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
            phi = grid_potential.poisson_fft(
                grid_potential.spread_gaussian(
                    np.array([plus, minus]), np.array([1.0, -1.0]),
                    shape, cell, sigma,
                ),
                cell,
            )
            return abs(
                grid_potential.interpolate_at(phi, cell, probe)[0] - expected,
            )

        sigma_dominated = [residual(s) for s in (1.2, 0.8, 0.6)]
        assert sigma_dominated[1] < sigma_dominated[0]
        assert sigma_dominated[2] < sigma_dominated[1]
        # Below the knee the residual is periodic-image error only.
        assert residual(0.4) == pytest.approx(residual(0.25), rel=1e-3)
        assert residual(0.25) < 0.05 * expected


def _constants(shape, cell, positions=None, zval=11.0):
    """Mimic VASP's Constants dataclasses.

    Field names verified against
    vasp.6.6.1/src/plugins/src/vasp/_local_potential.py and
    _force_and_stress.py.  positions are FRACTIONAL, and ion_types
    arrives already 0-indexed because _adjust_dataclass.adjust_indexing
    subtracts one from every IndexArray before the plugin sees it.
    """
    if positions is None:
        positions = np.zeros((0, 3))
    return types.SimpleNamespace(
        shape_grid=np.array(shape),
        lattice_vectors=np.asarray(cell, dtype=np.float64),
        positions=np.asarray(positions) @ np.linalg.inv(cell),
        ion_types=np.zeros(len(positions), dtype=int),
        ZVAL=np.array([zval]),
        # VASP normalizes sum(rho) = NELECT * N_grid; this stands in for
        # 8 electrons spread uniformly.
        charge_density=np.full(shape, 8.0),
        hartree_potential=None,
        ion_potential=None,
    )


def _additions(shape):
    return types.SimpleNamespace(
        total_potential=np.zeros(shape),
        total_energy=0.0,
        forces=np.zeros((1, 3)),
    )


class TestCallbacks:

    def test_local_potential_adds_a_negative_well(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(_constants(SHAPE, CELL), additions)
        assert additions.total_potential[24, 24, 24] < 0.0
        assert (tmp_path / vasp_plugin.SENTINEL).exists()

    def test_local_potential_builds_the_field_once(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        calls = []
        original = vasp_plugin.build_external_potential

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(vasp_plugin, "build_external_potential", counted)
        constants = _constants(SHAPE, CELL)
        for _ in range(3):
            vasp_plugin.local_potential(constants, _additions(SHAPE))
        assert len(calls) == 1

    def test_missing_charges_file_raises_and_records(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_plugin.reset_cache()
        with pytest.raises(Exception):
            vasp_plugin.local_potential(_constants(SHAPE, CELL), _additions(SHAPE))
        assert (tmp_path / vasp_plugin.ERROR_FILE).exists()

    def test_empty_selection_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.zeros((0, 3)), np.zeros(0), step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(_constants(SHAPE, CELL), additions)
        assert np.all(additions.total_potential == 0.0)

    def test_net_charge_warns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        with pytest.warns(RuntimeWarning, match="net charge"):
            vasp_plugin.local_potential(_constants(SHAPE, CELL), _additions(SHAPE))

    def test_grid_change_rebuilds_the_cache(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        vasp_plugin.local_potential(_constants(SHAPE, CELL), _additions(SHAPE))
        other = (32, 32, 32)
        additions = _additions(other)
        vasp_plugin.local_potential(_constants(other, CELL), additions)
        assert additions.total_potential.shape == other
        assert additions.total_potential.min() < 0.0

    def test_force_and_stress_applies_the_nuclear_terms(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # One MM charge at the cell centre, one QM ion offset from it
        # along x.  A positive MM charge attracts the positive pseudo-ion
        # core... no: like charges repel, so the ion is pushed AWAY,
        # towards +x.
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        ion = np.array([[7.0, 5.0, 5.0]])
        constants = _constants(SHAPE, CELL, positions=ion, zval=11.0)
        # Real VASP always calls local_potential before force_and_stress
        # every SCF step; the QM->MM back-reaction now added to
        # force_and_stress needs the Hartree/ion potentials that call
        # caches, so mimic that sequence here too.
        constants.hartree_potential = np.zeros(SHAPE)
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        # dE_I = -Z_I V_ext(R_I); V_ext < 0 near a positive charge, so
        # the energy correction is positive (repulsion).
        assert additions.total_energy > 0.0
        assert additions.forces[0, 0] > 0.0
        assert additions.forces[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_force_and_stress_with_no_ions_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL)
        constants.hartree_potential = np.zeros(SHAPE)
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        assert additions.total_energy == 0.0
        assert np.all(additions.forces == 0.0)

    def test_energy_and_force_corrections_are_consistent(self, tmp_path, monkeypatch):
        # The force must be the negative gradient of the energy it
        # accompanies.  Writing dF = -Z grad V_ext alongside
        # dE = -Z V_ext (as the spec originally did) flips every nuclear
        # force; this finite-difference check pins them together.
        monkeypatch.chdir(tmp_path)
        # sigma kept just under 0.8: contract_gaussian_gradient's cutoff
        # box (6*sigma) must fit inside this 10 A test cell, or the new
        # MM back-reaction this test now also exercises raises "cutoff
        # box wraps the cell" (build_external_potential's own cutoff
        # path tolerates the wrap by falling back to a full-grid sum;
        # contract_gaussian_gradient deliberately does not, to avoid
        # silently double-counting periodic images).
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.75,
        )

        def energy_at(x):
            vasp_plugin.reset_cache()
            constants = _constants(
                SHAPE, CELL, positions=np.array([[x, 5.3, 4.7]]),
            )
            constants.hartree_potential = np.zeros(SHAPE)
            constants.ion_potential = np.zeros(SHAPE)
            additions = _additions(SHAPE)
            vasp_plugin.local_potential(constants, additions)
            vasp_plugin.force_and_stress(constants, additions)
            return additions.total_energy, additions.forces[0, 0]

        step = 10.0 / SHAPE[0]
        _, force_x = energy_at(7.0)
        plus, _ = energy_at(7.0 + step)
        minus, _ = energy_at(7.0 - step)
        numerical = -(plus - minus) / (2 * step)
        assert force_x == pytest.approx(numerical, rel=0.05)

    def test_like_charges_repel(self, tmp_path, monkeypatch):
        # A +1 e MM charge and a +11 e pseudo-ion must push apart.
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL, positions=np.array([[7.0, 5.0, 5.0]]))
        constants.hartree_potential = np.zeros(SHAPE)
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        assert additions.forces[0, 0] > 0.0

    def test_opposite_charges_attract(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([-1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL, positions=np.array([[7.0, 5.0, 5.0]]))
        constants.hartree_potential = np.zeros(SHAPE)
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        assert additions.forces[0, 0] < 0.0


class TestElectronInteractionEnergy:

    def test_normalization_matches_vasp(self):
        # A uniform density of NELECT over the grid in a constant V_ext
        # must give exactly NELECT * V_ext.
        shape = (16, 16, 16)
        nelect = 8.0
        density = np.full(shape, nelect)
        v_ext = np.full(shape, -2.5)
        got = grid_potential.electron_interaction_energy(density, v_ext)
        assert got == pytest.approx(nelect * -2.5, rel=1e-12)

    def test_rejects_missing_density(self):
        with pytest.raises(RuntimeError, match="LVHAR"):
            grid_potential.electron_interaction_energy(None, np.zeros((4, 4, 4)))

    def test_rejects_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="shape"):
            grid_potential.electron_interaction_energy(
                np.zeros((4, 4, 4)), np.zeros((8, 8, 8)),
            )

    def test_local_potential_does_not_report_an_energy(self, tmp_path, monkeypatch):
        # MEASURED (jobs 11566990 / 11567275): VASP's TOTEN already
        # contains int(rho V_ext), so setting additions.total_energy
        # double-counts and shifts the forces by the whole nuclear
        # correction.  electron_interaction_energy is kept because it
        # documents the normalization, but it must not be wired in.
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(_constants(SHAPE, CELL), additions)
        assert additions.total_energy == 0.0

    def test_uniform_density_gives_zero_energy(self):
        # V_ext has zero mean (the G=0 term is dropped), so a uniform
        # electron gas feels no net interaction.  Worth pinning: it is
        # the reason the test above must localize the density.
        v_ext = grid_potential.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        got = grid_potential.electron_interaction_energy(
            np.full(SHAPE, 8.0), v_ext,
        )
        assert got == pytest.approx(0.0, abs=1e-9)


class TestInterpolantGradient:
    """The gradient that must match interpolate_at exactly."""

    def test_matches_finite_differences_to_machine_precision(self):
        v_ext = grid_potential.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.8,
        )
        point = np.array([[6.5, 5.3, 4.7]])
        analytic = grid_potential.interpolant_gradient_at(v_ext, CELL, point)[0]
        # A step well inside one grid cell, where the trilinear
        # interpolant is exactly linear, so the difference quotient is
        # the analytic slope.
        step = (10.0 / SHAPE[0]) * 0.2
        numerical = np.zeros(3)
        for axis in range(3):
            shift = np.zeros((1, 3))
            shift[0, axis] = step
            plus = grid_potential.interpolate_at(v_ext, CELL, point + shift)[0]
            minus = grid_potential.interpolate_at(v_ext, CELL, point - shift)[0]
            numerical[axis] = (plus - minus) / (2 * step)
        assert analytic == pytest.approx(numerical, rel=1e-9)

    def test_differs_from_the_spectral_gradient(self):
        # Documents WHY both exist: they are genuinely different, and
        # using the spectral one for the force while the energy uses
        # interpolate_at cost a reproducible 14 kJ/mol/A.
        v_ext = grid_potential.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.8,
        )
        point = np.array([[6.5, 5.3, 4.7]])
        spectral = grid_potential.gradient_at(v_ext, CELL, point)[0]
        interpolant = grid_potential.interpolant_gradient_at(v_ext, CELL, point)[0]
        assert not np.allclose(spectral, interpolant, rtol=1e-6)

    def test_energy_and_force_corrections_converge_quadratically(
            self, tmp_path, monkeypatch,
    ):
        # force_and_stress now takes the value and the gradient from one
        # Fourier series, so the force IS the analytic derivative of the
        # energy.  A central difference of a smooth function carries
        # O(h**2) truncation error, so the right assertion is that the
        # error falls ~4x when the step halves -- not that it reaches
        # machine precision.
        monkeypatch.chdir(tmp_path)
        # See test_energy_and_force_corrections_are_consistent: sigma
        # must stay under ~0.8 so contract_gaussian_gradient's cutoff
        # box fits inside this 10 A test cell.
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.75,
        )

        def probe(x):
            vasp_plugin.reset_cache()
            constants = _constants(
                SHAPE, CELL, positions=np.array([[x, 5.3, 4.7]]),
            )
            constants.hartree_potential = np.zeros(SHAPE)
            constants.ion_potential = np.zeros(SHAPE)
            additions = _additions(SHAPE)
            vasp_plugin.local_potential(constants, additions)
            vasp_plugin.force_and_stress(constants, additions)
            return additions.total_energy, additions.forces[0, 0]

        _, force_x = probe(6.5)
        errors = []
        for step in (0.02, 0.01):
            plus, _ = probe(6.5 + step)
            minus, _ = probe(6.5 - step)
            errors.append(abs(force_x + (plus - minus) / (2 * step)))
        assert errors[1] < errors[0] / 3.0
        assert errors[1] < 1e-3 * abs(force_x)

    def test_transverse_force_vanishes_by_symmetry(
            self, tmp_path, monkeypatch,
    ):
        # A charge at (5,5,5) and an ion at (7,5,5) share y and z, so
        # those force components must vanish.  The trilinear-interpolant
        # gradient gets this wrong by 2.0 kJ/mol/A because at a grid node
        # its slope is one-sided; the Fourier series is symmetric.
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL, positions=np.array([[7.0, 5.0, 5.0]]))
        constants.hartree_potential = np.zeros(SHAPE)
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        assert additions.forces[0, 0] > 0.0
        assert additions.forces[0, 1] == pytest.approx(0.0, abs=1e-9)
        assert additions.forces[0, 2] == pytest.approx(0.0, abs=1e-9)


class TestConstantFieldFigure2b:
    """The manuscript's Figure 2b validation, Tier 1.

    A neutral hydrogen atom in a constant field E_ext.  The atom is
    charge neutral and barely polarizable, so the TOTAL force must be
    ~0.  VASP alone gives 1e*E_ext, because it omits the core-field
    interaction; Eq. 4 supplies exactly that.

    This is a far better probe of the correction than a QM/MM system:
    the correction IS the entire signal, rather than a 1.7% residual on
    a near-total cancellation between two ~950 kJ/mol/A terms.
    """

    # 10 x 10 x 30 Angstrom, as in VASP-Python/H_in_constant_field.
    FIELD_CELL = np.diag([10.0, 10.0, 30.0])
    FIELD_SHAPE = (40, 40, 120)

    def _sheets(self, surface_charge):
        """Two oppositely charged sheets, +q at z=20 and -q at z=0."""
        grid = np.linspace(0.0, 10.0, 20, endpoint=False)
        xy = np.array([(x, y) for x in grid for y in grid])
        per_site = surface_charge * 100.0 / len(xy)
        top = np.column_stack([xy, np.full(len(xy), 20.0)])
        bottom = np.column_stack([xy, np.full(len(xy), 0.0)])
        positions = np.vstack([top, bottom])
        charges = np.concatenate([
            np.full(len(xy), per_site), np.full(len(xy), -per_site),
        ])
        return positions, charges

    def test_sheets_make_a_uniform_field_between_them(self):
        # Between two sheets of surface charge sigma the field is
        # sigma/eps0, and V_ext = -phi ramps linearly.
        surface_charge = 0.001          # e / Angstrom**2
        positions, charges = self._sheets(surface_charge)
        v_ext = grid_potential.build_external_potential(
            positions, charges, self.FIELD_SHAPE, self.FIELD_CELL, 0.4,
        )
        # Probe along z on the axis, well away from both sheets.
        probes = np.array([[5.0, 5.0, z] for z in (8.0, 10.0, 12.0)])
        _, gradients = grid_potential.spectral_value_and_gradient(
            v_ext, self.FIELD_CELL, probes,
        )
        # NOT sigma/eps0: that is the infinite-parallel-plate result and
        # does not hold under periodic boundary conditions.  Periodicity
        # forces the potential to return, so with an inside region of
        # d = 20 A in a cell of L = 30 A,
        #     E_in * d + E_out * (L - d) = 0
        #     E_out - E_in = sigma / eps0
        # giving |E_in| = (sigma/eps0) * (L - d) / L, i.e. a third here.
        inside, outside = 20.0, 10.0
        expected = (surface_charge / grid_potential.EPS0) * outside / (
            inside + outside
        )
        # dV_ext/dz is uniform between the sheets ...
        assert gradients[:, 2] == pytest.approx(
            np.full(3, gradients[0, 2]), rel=0.05,
        )
        # ... and matches the periodic result.
        assert abs(gradients[0, 2]) == pytest.approx(expected, rel=0.05)
        # ... with no transverse component on the axis.
        assert gradients[:, 0] == pytest.approx(np.zeros(3), abs=1e-3)

    def test_return_field_satisfies_periodicity(self):
        # A structural check on the periodic solve: the field outside
        # the plates must reverse and scale as -d/(L-d), so that the
        # potential closes on itself around the cell.
        positions, charges = self._sheets(0.001)
        v_ext = grid_potential.build_external_potential(
            positions, charges, self.FIELD_SHAPE, self.FIELD_CELL, 0.4,
        )
        probes = np.array([[5.0, 5.0, 10.0], [5.0, 5.0, 25.0]])
        _, gradients = grid_potential.spectral_value_and_gradient(
            v_ext, self.FIELD_CELL, probes,
        )
        assert gradients[1, 2] == pytest.approx(
            -2.0 * gradients[0, 2], rel=0.05,
        )

    def test_core_correction_equals_Z_times_the_field(self):
        # Eq. 4: dF = +Z grad V_ext.  For hydrogen Z = 1, so the
        # correction must equal 1e * E_ext -- the black dashed line in
        # Figure 2b, and exactly what cancels VASP's uncorrected force.
        surface_charge = 0.001
        positions, charges = self._sheets(surface_charge)
        v_ext = grid_potential.build_external_potential(
            positions, charges, self.FIELD_SHAPE, self.FIELD_CELL, 0.4,
        )
        hydrogen = np.array([[5.0, 5.0, 10.0]])
        _, gradient = grid_potential.spectral_value_and_gradient(
            v_ext, self.FIELD_CELL, hydrogen,
        )
        correction = 1.0 * gradient[0]              # Z_H = 1
        field = (surface_charge / grid_potential.EPS0) * 10.0 / 30.0
        assert abs(correction[2]) == pytest.approx(field, rel=0.05)
        assert correction[0] == pytest.approx(0.0, abs=1e-3)
        assert correction[1] == pytest.approx(0.0, abs=1e-3)


class TestPMEResolution:
    """helPME interpolates onto VASP's grid; that needs alpha*h small."""

    def _potential(self, alpha, gridnumber, tmp_path):
        import numpy as np
        from pydft_qmmm.interfaces.vasp import pme_external, vasp_utils
        rng = np.random.RandomState(0)
        length = 29.899
        cell = np.diag([length] * 3)
        positions = rng.uniform(0.0, length, (200, 3))
        charges = rng.uniform(-1.0, 1.0, 200)
        charges -= charges.mean()
        path = str(tmp_path / f"PME_{alpha}_{gridnumber}")
        vasp_utils.write_pme_data(
            path, positions, charges, [], alpha,
            (gridnumber,) * 3, 6, 0,
        )
        return pme_external.build_pme_potential(path, (48, 48, 48), cell)

    def test_converges_with_a_sane_ewald_parameter(self, tmp_path):
        coarse = self._potential(0.35, 30, tmp_path)
        fine = self._potential(0.35, 60, tmp_path)
        assert coarse.min() == pytest.approx(fine.min(), rel=1e-3)

    def test_diverges_when_alpha_outruns_the_grid(self, tmp_path):
        # Guard against silently reusing MM fixture parameters: at
        # alpha = 5.0 the reciprocal function varies on ~0.2 A while the
        # PME grid is ~1 A, so refining makes it WORSE, not better.
        coarse = self._potential(5.0, 30, tmp_path)
        fine = self._potential(5.0, 60, tmp_path)
        assert abs(fine.min()) > 1.5 * abs(coarse.min())


class TestGaussianContraction:
    """The transpose of spread_gaussian: forces from a grid potential."""

    def test_recovers_the_force_in_a_linear_field(self):
        # For phi = c.r the convolution with a normalized Gaussian is
        # exact, so contracting must return exactly the uniform-field
        # force F = qE = -q*grad(phi) = -q*c per charge, independent of
        # sigma.  The MINUS is the whole point: contracting returns a
        # force, not an energy gradient.
        gradient = np.array([0.3, -0.7, 0.2])
        axes = [np.arange(n) * 10.0 / n for n in SHAPE]
        mesh = np.meshgrid(*axes, indexing="ij")
        field = sum(g * m for g, m in zip(gradient, mesh))
        positions = np.array([[5.0, 5.0, 5.0], [3.0, 7.0, 4.0]])
        charges = np.array([1.0, -0.5])
        got = grid_potential.contract_gaussian_gradient(
            field, positions, charges, SHAPE, CELL, sigma=0.4,
        )
        assert got == pytest.approx(-np.outer(charges, gradient), rel=1e-6)

    def test_scales_linearly_with_charge(self):
        field = np.random.RandomState(0).normal(size=SHAPE)
        position = np.array([[5.0, 5.0, 5.0]])
        one = grid_potential.contract_gaussian_gradient(
            field, position, np.array([1.0]), SHAPE, CELL, sigma=0.5,
        )
        two = grid_potential.contract_gaussian_gradient(
            field, position, np.array([2.0]), SHAPE, CELL, sigma=0.5,
        )
        assert two == pytest.approx(2.0 * one, rel=1e-12)

    def test_uniform_field_exerts_no_force(self):
        field = np.full(SHAPE, 3.7)
        got = grid_potential.contract_gaussian_gradient(
            field, np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert got[0] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_empty_selection_returns_an_empty_array(self):
        got = grid_potential.contract_gaussian_gradient(
            np.zeros(SHAPE), np.zeros((0, 3)), np.zeros(0),
            SHAPE, CELL, sigma=0.4,
        )
        assert got.shape == (0, 3)


def poisson_of(positions, charges, sigma):
    """Electrostatic potential (volts) of Gaussian charges on the grid."""
    rho = grid_potential.spread_gaussian(
        positions, charges, SHAPE, CELL, sigma,
    )
    return grid_potential.poisson_fft(rho, CELL)


class TestSyntheticThirdLaw:

    def test_two_gaussians_push_each_other_apart(self):
        # Charge A makes a potential; charge B sits in it.  Compute the
        # force on B by contraction, and the force on A by contracting
        # B's potential.  They must be equal and opposite.
        a_pos = np.array([[4.0, 5.0, 5.0]])
        b_pos = np.array([[6.0, 5.0, 5.0]])
        a_q = np.array([1.0])
        b_q = np.array([1.0])
        sigma = 0.5
        phi_a = poisson_of(a_pos, a_q, sigma)
        phi_b = poisson_of(b_pos, b_q, sigma)
        force_on_b = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, b_q, SHAPE, CELL, sigma,
        )[0]
        force_on_a = grid_potential.contract_gaussian_gradient(
            phi_b, a_pos, a_q, SHAPE, CELL, sigma,
        )[0]
        assert force_on_a == pytest.approx(-force_on_b, rel=1e-6)
        # Like charges: B (at larger x) is pushed to +x.
        assert force_on_b[0] > 0.0

    def test_mismatched_sigma_breaks_the_third_law(self):
        # The guard that matters: contraction and spreading must agree
        # on sigma.  If they drift apart the forces stop balancing, and
        # this test is what would catch it.
        a_pos = np.array([[4.0, 5.0, 5.0]])
        b_pos = np.array([[6.0, 5.0, 5.0]])
        charge = np.array([1.0])
        phi_a = poisson_of(a_pos, charge, 0.5)
        phi_b = poisson_of(b_pos, charge, 0.5)
        good = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, charge, SHAPE, CELL, 0.5,
        )[0]
        bad = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, charge, SHAPE, CELL, 0.6,
        )[0]
        reference = grid_potential.contract_gaussian_gradient(
            phi_b, a_pos, charge, SHAPE, CELL, 0.5,
        )[0]
        assert good == pytest.approx(-reference, rel=1e-6)
        assert not np.allclose(bad, -reference, rtol=1e-3)


class TestVaspPotentialConversion:

    def test_negates_the_electron_referenced_sum(self):
        hartree = np.full((4, 4, 4), -2.0)
        ion = np.full((4, 4, 4), -3.0)
        got = grid_potential.electrostatic_potential_from_vasp(hartree, ion)
        # VASP reports what an ELECTRON feels, in eV.  The electrostatic
        # potential is the negative of that.
        assert got == pytest.approx(np.full((4, 4, 4), 5.0))

    def test_rejects_missing_fields(self):
        with pytest.raises(RuntimeError, match="PLUGINS/LOCAL_POTENTIAL"):
            grid_potential.electrostatic_potential_from_vasp(
                None, np.zeros((4, 4, 4)),
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="shape"):
            grid_potential.electrostatic_potential_from_vasp(
                np.zeros((4, 4, 4)), np.zeros((8, 8, 8)),
            )


class TestMMForceHandoff:

    def test_force_and_stress_writes_mm_forces(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[7.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.5,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL)
        # A positive Gaussian at the origin side: its potential pushes a
        # positive MM charge at +x further along +x.
        phi = poisson_of(np.array([[5.0, 5.0, 5.0]]), np.array([1.0]), 0.5)
        constants.hartree_potential = -phi
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        forces, step = vasp_utils.read_mm_forces("MM_FORCES")
        assert step == 0
        assert forces.shape == (1, 3)
        assert forces[0, 0] > 0.0
        assert forces[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_missing_potentials_raise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[7.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.5,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        with pytest.raises(RuntimeError, match="hartree_potential"):
            vasp_plugin.force_and_stress(constants, additions)
        assert (tmp_path / vasp_plugin.ERROR_FILE).exists()
