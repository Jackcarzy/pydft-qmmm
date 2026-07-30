"""Interface-level tests for VASP electrostatic embedding.

The wiring tests here need no VASP binary.  The Tier 0 smoke and Tier 2
gradient tests do, and are skipped unless PYDFT_QMMM_VASP_COMMAND is set.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory


class TestInterfaceWiring:

    def test_incar_gains_plugin_tags_when_embedding(self, vasp_embedded):
        vasp_embedded._write_input()
        with open(os.path.join(vasp_embedded.directory, "INCAR")) as fh:
            incar = fh.read()
        assert "PLUGINS/LOCAL_POTENTIAL = T" in incar
        assert "PLUGINS/FORCE_AND_STRESS = T" in incar

    def test_incar_has_no_plugin_tags_without_embedding(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
        )
        potential._write_input()
        with open(os.path.join(potential.directory, "INCAR")) as fh:
            incar = fh.read()
        assert "PLUGINS/" not in incar

    def test_plugin_is_copied_into_the_run_directory(self, vasp_embedded):
        vasp_embedded._write_input()
        assert os.path.isfile(
            os.path.join(vasp_embedded.directory, "vasp_plugin.py"),
        )

    def test_mm_charges_are_written(self, vasp_embedded):
        count = vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        positions, charges, _, sigma = vasp_plugin.read_mm_charges(path)
        assert count == len(charges) > 0
        assert sigma == pytest.approx(vasp_embedded.embedding_sigma)

    def test_written_charges_match_subsystem_ii(self, vasp_embedded):
        vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        positions, charges, _, _ = vasp_plugin.read_mm_charges(path)
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        assert len(charges) == len(indices)
        # A system whose charges are all zero builds an identically zero
        # V_ext, so every embedding test would pass while testing
        # nothing.  Job 11566828 did exactly that.
        assert np.count_nonzero(charges) > 0, "MM charges are all zero"
        assert positions == pytest.approx(
            np.asarray(vasp_embedded.system.positions)[indices], abs=1e-9,
        )
        assert charges == pytest.approx(
            np.asarray(vasp_embedded.system.charges)[indices], abs=1e-9,
        )

    def test_configured_sigma_reaches_the_plugin(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library,
    ):
        # Guards the process boundary: a module-level SIGMA in the
        # plugin would silently ignore the user's setting.
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            embedding_sigma=0.55,
        )
        potential._write_mm_charges()
        _, _, _, sigma = vasp_plugin.read_mm_charges(
            os.path.join(potential.directory, "MM_CHARGES"),
        )
        assert sigma == pytest.approx(0.55)

    def test_rewrite_replaces_rather_than_appends(self, vasp_embedded):
        vasp_embedded._write_mm_charges()
        first = vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        _, charges, _, _ = vasp_plugin.read_mm_charges(path)
        assert len(charges) == first

    def test_missing_sentinel_raises(self, vasp_embedded):
        with pytest.raises(Exception, match="sentinel|PLUGINS"):
            vasp_embedded._check_plugin_fired()

    def test_present_sentinel_passes(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        with open(
            os.path.join(vasp_embedded.directory, vasp_plugin.SENTINEL), "w",
        ) as fh:
            fh.write("local_potential callback executed\n")
        vasp_embedded._check_plugin_fired()


class TestFactory:

    def test_embedding_defaults_off(self, vasp_qmmm_system, tmp_path, vasp_pp_library):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
        )
        assert potential.embedding is False
        assert potential.embedding_sigma == 0.3

    def test_sigma_is_configurable(self, vasp_qmmm_system, tmp_path, vasp_pp_library):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            embedding_sigma=0.5,
        )
        assert potential.embedding_sigma == 0.5

    @pytest.mark.parametrize("bad", [-1.0, 0.0])
    def test_non_positive_sigma_is_rejected(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library, bad,
    ):
        with pytest.raises(ValueError, match="sigma"):
            vasp_interface_factory(
                vasp_qmmm_system,
                directory=str(tmp_path / "vasp"),
                pp_path=vasp_pp_library,
                embedding=True,
                embedding_sigma=bad,
            )


requires_vasp = pytest.mark.skipif(
    not os.environ.get("PYDFT_QMMM_VASP_COMMAND"),
    reason="set PYDFT_QMMM_VASP_COMMAND to run VASP-backed tests",
)


@requires_vasp
class TestSmoke:
    """Tier 0: does the plugin actually fire inside a real VASP run?"""

    def test_plugin_fires_and_perturbs_the_energy(self, vasp_embedded):
        energy = vasp_embedded.compute_energy()
        sentinel = os.path.join(vasp_embedded.directory, vasp_plugin.SENTINEL)
        assert os.path.isfile(sentinel), (
            "no sentinel: the plugin never ran, so the energy is the "
            "unembedded one"
        )
        with open(sentinel) as fh:
            report = fh.read()
        assert "mm_charges" in report
        assert np.isfinite(energy)
        print("\n" + report)
        # The plugin firing is not enough: it must build a potential
        # that is actually non-zero.  Parsed from the sentinel because
        # the array itself lives in the VASP process.
        minimum = float(
            [ln for ln in report.splitlines() if "min V_ext" in ln][0]
            .split("=")[1],
        )
        assert minimum < -1e-3, (
            f"V_ext minimum is {minimum} eV -- the plugin ran but built "
            "an essentially zero potential, so nothing was embedded"
        )

    def test_zero_charges_reproduce_the_unembedded_energy(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        # The control: machinery fully active, physics inert.  Separates
        # "the plumbing is a no-op when it should be" from "the physics
        # is right".
        assert np.count_nonzero(vasp_qmmm_system.charges) > 0, (
            "control is vacuous if the charges were already zero"
        )
        vasp_qmmm_system.charges[:] = 0.0
        # Same cutoff as the vasp_embedded fixture: without it this ran
        # at the default ENCUT=400, a 392**3 grid, for no benefit.
        light = {"ENCUT": 250}
        plain = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "plain"),
            pp_path=vasp_pp_library,
            incar=light,
        ).compute_energy()
        embedded = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "embed"),
            pp_path=vasp_pp_library,
            embedding=True,
            incar=light,
        ).compute_energy()
        assert embedded == pytest.approx(plain, abs=1e-5)


@requires_vasp
class TestGradients:
    """Tier 2: are the returned forces the gradient of the energy?

    This also settles the spec's open question about whether VASP's TOTEN
    already contains the electron-V_ext term: if it does not, energy and
    forces disagree by exactly that term.
    """

    def test_embedded_forces_match_numerical_gradient(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        from pydft_qmmm.calculators import PotentialCalculator
        from pydft_qmmm.utils import numerical_gradient
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
# ISTART=0 on EVERY evaluation.  The interface normally
            # restarts from the previous WAVECAR, which is right for MD
            # but poisons a finite difference: run #1 initializes from
            # atomic superposition, run #2 (x+) restarts from THAT, and
            # run #3 (x-) restarts from x+.  Only the x pair has
            # mismatched restart histories, and the unembedded control
            # showed exactly that signature -- x off by -3.43 kJ/mol/A
            # while y and z agreed to 0.03.
            incar={"ENCUT": 300, "EDIFF": 1e-7, "ISTART": 0},
        )
        calculator = PotentialCalculator(vasp_qmmm_system, potential)
        analytical = calculator.calculate().forces[0]
        numerical = -numerical_gradient(calculator, frozenset({0}), dist=1e-3)[0]
        assert analytical == pytest.approx(numerical, abs=1.0)

    def test_unembedded_forces_match_numerical_gradient(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        """Control: does the interface pass this test WITHOUT embedding?

        Every embedded run so far has shown a residual of order
        10 kJ/mol/A between the returned forces and the numerical
        gradient.  This isolates whether that is introduced by the
        embedding corrections at all, or is a pre-existing property of
        the VASP interface and this test setup -- restart reuse across
        displacements, Pulay/egg-box effects on a 320**3 grid, and so
        on.  Without it, any further tuning of the embedding code is
        guesswork.
        """
        from pydft_qmmm.calculators import PotentialCalculator
        from pydft_qmmm.utils import numerical_gradient
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=False,
            # See the note in the embedded test: independent evaluations.
            incar={"ENCUT": 300, "EDIFF": 1e-7, "ISTART": 0},
        )
        calculator = PotentialCalculator(vasp_qmmm_system, potential)
        analytical = calculator.calculate().forces[0]
        numerical = -numerical_gradient(calculator, frozenset({0}), dist=1e-3)[0]
        print(f"\n  unembedded analytical: {analytical}")
        print(f"  unembedded numerical : {numerical}")
        print(f"  unembedded residual  : {analytical - numerical}")
        assert analytical == pytest.approx(numerical, abs=1.0)


@requires_vasp
class TestFigure2b:
    """The manuscript's Figure 2b, against real VASP.

    A neutral hydrogen atom in a constant field.  VASP alone reports a
    force of 1e * E_ext because it omits the core-field interaction;
    Eq. 4 supplies exactly that, so the corrected force must be ~0.

    This is the clean probe of the correction.  The QM/MM
    finite-difference test measures the same physics through a
    near-total cancellation (a ~950 kJ/mol/A nuclear term against a ~27
    net force), where a 1.7% error looks like 50%.  Here the correction
    IS the signal.
    """

    # Periodic result, NOT sigma/eps0: the potential must close on
    # itself, giving (sigma/eps0) * (L-d)/L with d = 20, L = 30.
    FIELD = (0.001 / 0.005526349358057108) * 10.0 / 30.0    # V/Angstrom
    KJMOL_PER_EV = 96.48533212331

    def test_corrected_force_on_neutral_atom_is_near_zero(
            self, h_constant_field_system, vasp_workdir, vasp_pp_library,
    ):
        potential = vasp_interface_factory(
            h_constant_field_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            incar={"ENCUT": 400, "EDIFF": 1e-7, "ISMEAR": 0, "SIGMA": 0.05},
        )
        forces = potential.compute_forces()
        uncorrected = 1.0 * self.FIELD * self.KJMOL_PER_EV   # Z_H = 1
        print(f"\n  E_ext (periodic)   : {self.FIELD:.4f} V/A")
        print(f"  1e*E_ext           : {uncorrected:.3f} kJ/mol/A")
        print(f"  corrected force    : {forces[0]}")
        # The whole claim of Figure 2b: what would have been a force of
        # 1e*E_ext is cancelled to near zero by the core correction.
        assert abs(forces[0, 2]) < 0.2 * abs(uncorrected)
        assert forces[0, 0] == pytest.approx(0.0, abs=1.0)
        assert forces[0, 1] == pytest.approx(0.0, abs=1.0)


@requires_vasp
class TestPME:
    """The long-range (PME) embedding path against real VASP.

    Unlike the cutoff scheme, the potential is built by helPME inside
    VASP's interpreter, evaluated on VASP's own FFT grid.  This checks
    that the path runs end to end and produces a non-trivial potential;
    it is NOT yet a physics validation of the PME result itself.
    """

    def test_pme_path_runs_and_builds_a_potential(
            self, vasp_pme_system, vasp_workdir, vasp_pp_library,
    ):
        from pydft_qmmm.potentials.pme_potential import (
            PMEElectronicPotential,
        )
        potential = vasp_interface_factory(
            vasp_pme_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            incar={"ENCUT": 250, "EDIFF": 1e-6},
        )
        # alpha = 0.35, not the 5.0 the MM fixtures use.  helPME
        # B-spline interpolates from ITS grid onto VASP's points, which
        # is only valid while the PME grid resolves the reciprocal-space
        # function -- scale 1/alpha.  At alpha = 5.0 that scale is 0.2 A
        # against a 1 A PME spacing, and min V_ext DIVERGES with
        # refinement (-34.3, -56.7, -67.1 eV at 30/60/120**3) instead of
        # converging.  At alpha = 0.35 it is converged to four decimals
        # already at 30**3.
        # 100**3 over 29.899 A is 0.30 A spacing, inside the 0.24-0.34 A
        # range Pederson & McDaniel recommend (JCP 156, 174105 (2022),
        # Fig. 4).  Their point is that PME grid size sets the accuracy
        # of interpolating the potential onto the DFT grid, and that
        # such grids are far finer than MD practice yet essentially free
        # because the QM calculation dominates the cost.
        #
        # 30**3 (1.0 A) is coarser than anything they recommend.  A
        # single-probe convergence check here looked converged by
        # 0.5-0.66 A, but that is much weaker evidence than their
        # interaction-energy study, so follow the paper.
        pme = PMEElectronicPotential(vasp_pme_system, 0.35, (100, 100, 100), 6)
        potential.add_electronic_potential(pme)
        energy = potential.compute_energy()
        sentinel = os.path.join(potential.directory, vasp_plugin.SENTINEL)
        assert os.path.isfile(sentinel), "plugin never fired"
        report = open(sentinel).read()
        print("\n" + report)
        assert "PME" in report, "took the cutoff path, not PME"
        # Approach 2 needs BOTH halves present: PME supplies the
        # long-range tail, the analytic term reinstates the near field
        # that compute_P_adj removed.  Asserting on both stops the
        # near-field term going missing again.
        assert "V_pme" in report and "V_near" in report
        near = float(
            [ln for ln in report.splitlines() if "V_near" in ln][0]
            .split("=")[1],
        )
        assert near < -1.0, f"analytic near field is missing ({near})"
        minimum = float(
            [ln for ln in report.splitlines() if "min V_ext" in ln][0]
            .split("=")[1],
        )
        assert minimum < -1e-3, f"V_ext is essentially zero ({minimum})"
        assert np.isfinite(energy)
        print(f"  embedded energy: {energy:.6f} kJ/mol")
