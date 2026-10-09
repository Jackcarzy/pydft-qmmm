"""KSCED A–C electrostatics; run on compute nodes with KSCED_BACKEND=cpu/gpu.

A is subsystem I. Frozen B retains its independent MM classification, but
must never source the A–C field. These tests cover only the QM calculator;
B–C interactions belong to the MM calculator.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm import Atom, QMHamiltonian, System
from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH, Subsystem

pytestmark = pytest.mark.slow

ACTIVE = (1, 3)
FROZEN = 0
STEPS_BOHR = (2e-3, 1e-3, 5e-4)
FORCE_SCALE = KJMOL_PER_EH * BOHR_PER_ANGSTROM


def _host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _calculator(mode, **changes):
    # Coordinates and box below are Bohr; System expects Angstrom.
    # Interleave all three partitions to expose local/global index mistakes.
    entries = [
        ("He", (5.6, 4.5, 4.8), Subsystem.II, .43),
        ("H", (2., 3., 4.), Subsystem.I, 0.),
        ("He", (6.8, 2.1, 3.2), Subsystem.II,
         .25 if mode in ("near", "combined") else 0.),
        ("H", (3.4, 3.1, 4.), Subsystem.I, 0.),
        ("He", (8., 7.2, 6.4), Subsystem.III,
         -.17 if mode in ("pme", "combined") else 0.),
    ]
    system = System([
        Atom(element=element, position=np.array(position)/BOHR_PER_ANGSTROM,
             subsystem=subsystem, charge=charge, mass=1.)
        for element, position, subsystem, charge in entries
    ], np.eye(3)*10/BOHR_PER_ANGSTROM)
    options = dict(
        basis="gth-szv", pseudo="gth-pbe", charge=0, multiplicity=1,
        functional="PBE", mesh=(21, 21, 21), embedding_sigma=.3,
        conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=100,
        device=os.environ.get("KSCED_BACKEND", "cpu"),
        ksced=dict(active_atoms=[3, 1], frozen_atoms=[FROZEN],
                   basis_mode="M", t_nad="LDA_K_TF"),
    )
    options.update(changes)
    calculator = QMHamiltonian(interface="pyscf-pbc", **options)[
        list(ACTIVE)
    ].build_calculator(system)
    interface = calculator.potential
    interface.configure_electrostatic_embedding(True)
    if mode in ("pme", "combined"):
        # helPME uses Angstrom: alpha is inverse Angstrom, not inverse Bohr.
        interface.add_electronic_potential(
            PMEElectronicPotential(system, .6, (24, 24, 24), 6),
        )
    return calculator


def _fd_force(interface, atom, axis, step_bohr):
    """Central force in kJ/mol/Angstrom from energies in kJ/mol."""
    original = np.asarray(interface.system.positions).copy()
    step = step_bohr/BOHR_PER_ANGSTROM
    try:
        plus_positions = original.copy()
        plus_positions[atom, axis] += step
        interface.system.positions = plus_positions
        plus = interface.compute_energy()
        minus_positions = original.copy()
        minus_positions[atom, axis] -= step
        interface.system.positions = minus_positions
        minus = interface.compute_energy()
    finally:
        interface.system.positions = original
    return -(plus-minus)/(2*step)


def _assert_fd(interface, forces, atoms, axes=range(3)):
    for atom in atoms:
        for axis in axes:
            for step in STEPS_BOHR:
                numeric = _fd_force(interface, atom, axis, step)
                error = abs(forces[atom, axis]-numeric)/FORCE_SCALE
                print(f"KSCED embedding FD backend={interface.device} atom={atom} "
                      f"axis={axis} h_bohr={step} error_Ha_Bohr={error:.8g}",
                      flush=True)
                # Fixed FFT quadrature has no moving atom-centered grid term.
                assert error < (1e-5 if atom in ACTIVE else 1e-6)


def _freeze_guard(interface, monkeypatch):
    frozen = interface.frozen_method[0]
    environment = interface.frozen_environment[0]
    dm = _host(frozen.make_rdm1()).copy()

    def forbidden_kernel(*args, **kwargs):
        pytest.fail("Changing A or C reconverged frozen subsystem B")

    monkeypatch.setattr(frozen, "kernel", forbidden_kernel)

    def verify():
        assert interface.frozen_method[0] is frozen
        assert interface.frozen_environment[0] is environment
        np.testing.assert_array_equal(_host(frozen.make_rdm1()), dm)

    return verify


@pytest.mark.parametrize("mode", ["near", "pme", "combined"])
def test_embedding_forces_sources_and_frozen_reuse(mode, monkeypatch):
    calculator = _calculator(mode)
    interface = calculator.potential
    system = interface.system
    result = calculator.calculate()
    state = interface._scf_state()
    # GPU reductions may round differently when recomputing the gradient;
    # cache reuse is checked by object identity below.
    np.testing.assert_allclose(interface.compute_forces(), result.forces,
                               atol=1e-9, rtol=0)
    assert interface._scf_state() is state
    assert sum(result.components.values()) == pytest.approx(result.energy, abs=1e-9)
    assert system.select("subsystem I") == frozenset(ACTIVE)
    np.testing.assert_array_equal(result.forces[FROZEN], np.zeros(3))
    coupled_c = (2, 4) if mode == "combined" else ((2,) if mode == "near" else (4,))
    for atom in coupled_c:
        assert np.linalg.norm(result.forces[atom]) > 1e-3
    inactive_c = set((2, 4)) - set(coupled_c)
    for atom in inactive_c:
        np.testing.assert_array_equal(result.forces[atom], np.zeros(3))
    verify_frozen = _freeze_guard(interface, monkeypatch)
    _assert_fd(interface, result.forces, ACTIVE + coupled_c)
    verify_frozen()

    # Test B's charge while in each MM region, then its II/III assignment.
    # Neither the near field nor the reciprocal field may see this charge.
    for subsystem, charge in ((Subsystem.II, -.71), (Subsystem.III, -.71),
                              (Subsystem.III, .91), (Subsystem.II, .91)):
        system.subsystems[FROZEN] = subsystem
        system.charges[FROZEN] = charge
        assert interface.compute_energy() == pytest.approx(result.energy, abs=2e-6)
        np.testing.assert_allclose(interface.compute_forces(), result.forces,
                                   atol=5e-4, rtol=0)
        np.testing.assert_array_equal(interface.compute_forces()[FROZEN], np.zeros(3))
        verify_frozen()

    # Every C update must invalidate A's cached SCF, without changing B.
    for atom in coupled_c:
        before = interface._scf_state()
        old_energy = interface.compute_energy()
        positions = np.asarray(system.positions).copy()
        positions[atom, 0] += .04  # Angstrom
        system.positions = positions
        moved_energy = interface.compute_energy()
        assert interface._scf_state() is not before
        assert abs(moved_energy-old_energy) > 1e-7
        verify_frozen()
        before = interface._scf_state()
        system.charges[atom] *= 1.15
        changed_energy = interface.compute_energy()
        assert interface._scf_state() is not before
        assert abs(changed_energy-moved_energy) > 1e-7
        changed_forces = interface.compute_forces()
        _assert_fd(interface, changed_forces, (atom,), axes=(0,))
        verify_frozen()
        assert sum(interface.compute_components().values()) == pytest.approx(
            interface.compute_energy(), abs=1e-9,
        )


@pytest.mark.parametrize("case", ["uks", "smearing"])
def test_embedding_spin_and_free_energy(case, monkeypatch):
    changes = (dict(charge=1, multiplicity=2) if case == "uks"
               else dict(sigma=.15, smearing_method="fermi"))
    calculator = _calculator("combined", **changes)
    result = calculator.calculate()
    interface = calculator.potential
    state = interface._scf_state()
    verify_frozen = _freeze_guard(interface, monkeypatch)
    assert sum(result.components.values()) == pytest.approx(result.energy, abs=1e-9)
    np.testing.assert_array_equal(result.forces[FROZEN], np.zeros(3))
    assert np.linalg.norm(result.forces[[2, 4]]) > 1e-3
    if case == "uks":
        assert _host(state.dm).shape[0] == 2
    else:
        assert state.method.entropy > 1e-5
        assert result.energy == pytest.approx(
            (state.method.energy_potential()+state.nuclear_energy)*KJMOL_PER_EH,
            abs=2e-6,
        )
        assert abs(state.method.energy_potential()-state.method.e_tot) > 1e-5
    # Exercise spin-summed density reactions and the free-energy derivative.
    _assert_fd(interface, result.forces, (1, 2, 4), axes=(0,))
    verify_frozen()
