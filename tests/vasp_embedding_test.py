"""Interface-level tests for VASP electrostatic embedding.

The wiring tests here need no VASP binary.  The Tier 0 smoke and Tier 2
gradient tests do, and are skipped unless PYDFT_QMMM_VASP_COMMAND is set.
"""
from __future__ import annotations

import os
from unittest.mock import Mock

import numpy as np
import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.calculators import PotentialCalculator
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp import vasp_utils
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

    def test_stale_mm_forces_is_removed_before_launch(self, vasp_embedded):
        # If a previous run wrote MM_FORCES and this one dies before the
        # plugin fires, the driver must not silently reread the old
        # step's forces.
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        path = os.path.join(vasp_embedded.directory, "MM_FORCES")
        with open(path, "w") as fh:
            fh.write("1 0\n0.0 0.0 0.0\n")
        vasp_embedded._write_input()
        assert not os.path.isfile(path)

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


class TestCouplingConfiguration:

    @staticmethod
    def _configure(coupling, potential, system, monkeypatch):
        mm_interface = Mock(spec=MMInterface)
        calculator = Mock()
        calculator.calculators = [
            PotentialCalculator(system, potential),
            PotentialCalculator(system, mm_interface),
        ]
        monkeypatch.setattr(coupling, "apply_exclusions", Mock())
        coupling.modify_calculator(calculator, system)

    def test_electrostatic_coupling_enables_vasp_embedding(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library, monkeypatch,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
        )
        assert potential.embedding is False
        coupling = QMMMHamiltonian(
            "electrostatic", "cutoff", partition=None,
        )
        self._configure(coupling, potential, vasp_qmmm_system, monkeypatch)
        assert potential.embedding is True

    def test_manual_embedding_with_mechanical_coupling_is_rejected(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library, monkeypatch,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
        )
        coupling = QMMMHamiltonian(
            "mechanical", "mechanical", partition=None,
        )
        with pytest.raises(ValueError, match="double-counting"):
            self._configure(
                coupling, potential, vasp_qmmm_system, monkeypatch,
            )


class TestMMForceReturn:
    """The driver side of the QM->MM back-reaction."""

    @staticmethod
    def _write(potential, rows, step=0):
        os.makedirs(potential.directory, exist_ok=True)
        path = os.path.join(potential.directory, "MM_FORCES")
        with open(path, "w") as fh:
            fh.write(f"{len(rows)} {step}\n")
            for fx, fy, fz in rows:
                fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
        # _run increments frame[0] AFTER the plugin has stamped
        # MM_FORCES, so a file written for step N is read back while the
        # counter already reads N+1.  Mirror that here rather than
        # letting the reader see a counter no real run could produce.
        potential.frame[0] = step + 1
        return path

    def test_reads_forces_into_subsystem_ii_rows(self, vasp_embedded):
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        rows = np.arange(3 * len(indices), dtype=float).reshape(-1, 3)
        self._write(vasp_embedded, rows)
        assert vasp_embedded._read_mm_forces() == pytest.approx(rows)

    def test_missing_file_raises(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        path = os.path.join(vasp_embedded.directory, "MM_FORCES")
        if os.path.isfile(path):
            os.remove(path)
        with pytest.raises(vasp_utils.VaspExecutionError, match="MM_FORCES"):
            vasp_embedded._read_mm_forces()

    def test_all_zero_forces_raise(self, vasp_embedded):
        # Identically zero is what the unimplemented back-reaction looked
        # like in Milestone 2, so it must not pass silently -- otherwise
        # this milestone is indistinguishable from not having run.
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        self._write(vasp_embedded, np.zeros((len(indices), 3)))
        with pytest.raises(vasp_utils.VaspExecutionError, match="zero"):
            vasp_embedded._read_mm_forces()

    def test_row_count_mismatch_raises(self, vasp_embedded):
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        self._write(vasp_embedded, np.ones((len(indices) - 1, 3)))
        with pytest.raises(vasp_utils.VaspExecutionError, match="rows"):
            vasp_embedded._read_mm_forces()

    def test_stale_step_raises(self, vasp_embedded):
        # The second of the two stale-file guards.  Deleting MM_FORCES
        # before a launch covers a run that dies early; this covers a
        # file that exists but belongs to a different step.
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        self._write(vasp_embedded, np.ones((len(indices), 3)), step=3)
        vasp_embedded.frame[0] = 9
        with pytest.raises(ValueError, match="step"):
            vasp_embedded._read_mm_forces()


class TestNetForceRestoration:
    """VASP removes the net force on the QM ions; we put it back.

    VASP subtracts the mean force from every ion, which is right for an
    isolated periodic cell and wrong under embedding: V_ext breaks
    translational invariance, and the net force is precisely the
    momentum the MM subsystem transfers to the QM one.  Measured on job
    11580378, VASP reported sum(F_QM) = 0 to machine precision while the
    true net was (9.90, -37.70, 29.28) kJ/mol/A.
    """

    def test_reads_the_net_force_and_converts_units(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        path = os.path.join(vasp_embedded.directory, "QM_NET_FORCE")
        with open(path, "w") as fh:
            # The plugin writes eV/Angstrom, VASP's own unit.
            fh.write("3\n")
            fh.write("1.025624600000e-01 -3.907679200000e-01 "
                     "3.034367400000e-01\n")
            fh.write("# vasp_own  0 0 0\n")
        got = vasp_embedded._read_qm_net_force()
        # The real values from job 11580378, in kJ/mol/A.
        assert got == pytest.approx([9.8958, -37.7034, 29.2772], abs=1e-3)

    def test_missing_file_raises(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        path = os.path.join(vasp_embedded.directory, "QM_NET_FORCE")
        if os.path.isfile(path):
            os.remove(path)
        with pytest.raises(
                vasp_utils.VaspExecutionError, match="QM_NET_FORCE",
        ):
            vasp_embedded._read_qm_net_force()

    def test_net_is_spread_evenly_over_the_qm_atoms(
            self, vasp_embedded, monkeypatch,
    ):
        qm = sorted(vasp_embedded.system.select("subsystem I"))
        near = sorted(vasp_embedded.system.select("subsystem II"))
        qm_rows = np.zeros((len(qm), 3))
        net = np.array([9.0, -36.0, 27.0])
        monkeypatch.setattr(vasp_embedded, "_run", lambda: (0.0, qm_rows))
        monkeypatch.setattr(
            vasp_embedded, "_read_mm_forces",
            lambda: np.full((len(near), 3), 1.0),
        )
        monkeypatch.setattr(
            vasp_embedded, "_read_qm_net_force", lambda: net,
        )

        forces = vasp_embedded.compute_forces()

        # VASP returned zero for every QM atom, so whatever appears on
        # the QM rows is the restoration and nothing else.
        assert forces[qm].sum(axis=0) == pytest.approx(net)
        assert forces[qm] == pytest.approx(
            np.tile(net / len(qm), (len(qm), 1)),
        )


class TestSubsystemIIIIsLeftToOpenMM:
    """Direct QM/MM/PME assigns subsystem III's force to the MM level.

    VASP supplies X = QM -- subsystem III polarizes the QM density and
    VASP differentiates that with respect to subsystem I.  It must NOT
    also supply Y: OpenMM already computes the force on subsystem III
    from the static forcefield charges on subsystem I, so a
    QM-density-derived contribution here would be counted twice.  See
    John et al., JCP 161, 034103 (2024), Table I.
    """

    def test_vasp_leaves_subsystem_iii_at_zero(
            self, vasp_three_subsystem_system, vasp_workdir, monkeypatch,
    ):
        potential = vasp_interface_factory(
            vasp_three_subsystem_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path="unused",
            embedding=True,
        )
        system = vasp_three_subsystem_system
        qm = sorted(system.select("subsystem I"))
        near = sorted(system.select("subsystem II"))
        far = sorted(system.select("subsystem III"))
        assert far, "fixture must populate subsystem III or this is vacuous"
        qm_rows = np.ones((len(qm), 3))
        near_rows = np.full((len(near), 3), 2.0)
        monkeypatch.setattr(potential, "_run", lambda: (0.0, qm_rows))
        monkeypatch.setattr(
            potential, "_read_mm_forces", lambda: near_rows,
        )
        # Zero net, so the restored force does not perturb the QM rows
        # this test is checking.  The restoration itself is covered by
        # TestNetForceRestoration.
        monkeypatch.setattr(
            potential, "_read_qm_net_force", lambda: np.zeros(3),
        )

        forces = potential.compute_forces()

        assert forces[qm] == pytest.approx(qm_rows)
        assert forces[near] == pytest.approx(near_rows)
        # Zero from VASP, not zero overall -- the composite calculator
        # adds OpenMM's nonzero MM-level rows on top of this.
        assert forces[far] == pytest.approx(np.zeros((len(far), 3)))


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


@requires_vasp
class TestThirdLaw:
    """Momentum conservation: the reason Milestone 3 exists."""

    def test_qm_and_mm_forces_cancel(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            incar={"ENCUT": 250, "EDIFF": 1e-7},
        )
        forces = potential.compute_forces()
        qm = sorted(vasp_qmmm_system.select("subsystem I"))
        mm = sorted(vasp_qmmm_system.select("subsystem II"))
        # Only over I + II.  Subsystem III feels nothing through the
        # near-field channel -- under direct QM/MM/PME its reaction is
        # the Y = MM term OpenMM supplies -- so including it would show
        # a violation that is expected rather than a bug.
        total = forces[qm].sum(axis=0) + forces[mm].sum(axis=0)
        scale = np.abs(forces[qm + mm]).max()
        print(f"\n  QM atoms          : {len(qm)}")
        print(f"  MM atoms (II)     : {len(mm)}")
        print(f"  sum F over I+II   : {total}")
        print(f"  largest |F|       : {scale:.3f} kJ/mol/A")
        print(f"  relative violation: {np.abs(total).max() / scale:.3e}")
        # A few percent rather than <1% means the contraction and the
        # spreading disagree on sigma or cutoff -- check that sigma is
        # read from MM_CHARGES in both, not hardcoded.
        assert np.abs(total).max() < 0.01 * scale


@requires_vasp
class TestMMFiniteDifference:
    """The strong per-atom check: is the MM force the energy's gradient?

    Newton's third law can hold while both sides are wrong by the same
    amount.  This pins the magnitude against a numerical derivative of
    the energy, one MM atom at a time.  Seven VASP launches.
    """

    def test_mm_force_matches_the_numerical_gradient(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        from pydft_qmmm.calculators import PotentialCalculator
        from pydft_qmmm.utils import numerical_gradient
        target = sorted(vasp_qmmm_system.select("subsystem II"))[0]
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            # ISTART=0 on every evaluation.  The interface normally
            # restarts from the previous WAVECAR, which is right for MD
            # but poisons a finite difference: only the x pair ends up
            # with mismatched restart histories, and that showed up in
            # Milestone 2 as x off by -3.43 kJ/mol/A while y and z
            # agreed to 0.03.  It cost three GPU jobs to diagnose.
            incar={"ENCUT": 250, "EDIFF": 1e-7, "ISTART": 0},
        )
        calculator = PotentialCalculator(vasp_qmmm_system, potential)
        analytical = calculator.calculate().forces[target]
        numerical = -numerical_gradient(
            calculator, frozenset({target}), dist=1e-3,
        )[0]
        print(f"\n  MM atom index: {target}")
        print(f"  analytical   : {analytical}")
        print(f"  numerical    : {numerical}")
        print(f"  residual     : {analytical - numerical}")
        # Do NOT tune this tolerance if it fails.  Compare the residual
        # against Milestone 2's unembedded control, which agreed to
        # 0.03 kJ/mol/A: a residual of that order is the interface's own
        # noise floor, while one comparable to the force itself is a
        # real defect in the contraction.
        assert analytical == pytest.approx(numerical, abs=1.0)
