"""Periodic KSCED configuration and conservative A-force integration."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm import Atom, QMHamiltonian, System
from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import pyscf_pbc_interface_factory
from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH, Subsystem


def build_system(frozen_element="He"):
    # Interleave A, B and MM to expose incorrect local/global force mappings.
    entries = [(frozen_element, (5.6, 4.5, 4.8), Subsystem.II),
               ("H", (2., 3., 4.), Subsystem.I),
               ("He", (8., 8., 8.), Subsystem.III),
               ("H", (3.4, 3.1, 4.), Subsystem.I)]
    atoms = [Atom(element=e, position=np.array(r)/BOHR_PER_ANGSTROM,
                  subsystem=s, mass=1.) for e, r, s in entries]
    return System(atoms, np.eye(3)*10/BOHR_PER_ANGSTROM)


def options(**changes):
    out = dict(basis="gth-szv", pseudo="gth-pbe", charge=0,
               multiplicity=1, functional="PBE", mesh=(21, 21, 21),
               conv_tol=1e-12, conv_tol_grad=1e-9, max_cycle=100,
               ksced=dict(active_atoms=[3, 1], frozen_atoms=[0],
                          basis_mode="M", t_nad="LDA_K_TF"))
    out.update(changes)
    return out


def make_interface(system=None, **changes):
    return pyscf_pbc_interface_factory(
        build_system() if system is None else system, **options(**changes),
    )


def test_factory_exposes_independent_partition():
    interface = make_interface()
    assert interface.active_indices == (1, 3)
    assert interface.frozen_indices == (0,)
    assert interface.system.select("subsystem I") == frozenset([1, 3])
    assert interface.system.subsystems[0] == Subsystem.II
    assert interface.frozen_method[0] is None
    assert interface.frozen_environment[0] is None


def test_selection_strings_and_configuration_copy():
    config = dict(active_atoms="index 1 3", frozen_atoms="index 0")
    interface = make_interface(ksced=config)
    config["active_atoms"] = [0]
    assert interface.active_indices == (1, 3)
    assert interface.frozen_indices == (0,)


def test_nuclear_charges_cover_only_active_qm_atoms():
    interface = make_interface()
    np.testing.assert_array_equal(interface.nuclear_charges(), [1., 1.])


@pytest.mark.parametrize("change", [
    dict(active_atoms=[]), dict(frozen_atoms=[]), dict(active_atoms=[0, 1, 3]),
    dict(active_atoms=[1]), dict(active_atoms=[1, 2, 3]),
    dict(active_atoms=[-1, 1]), dict(active_atoms=[1, 99]),
    dict(active_atoms=[1, 1, 3]), dict(active_atoms=[True, 3]),
    dict(basis_mode="S"), dict(frozen_multiplicity=0), dict(unknown_option=1),
])
def test_invalid_partition_or_config_is_rejected(change):
    config = options()["ksced"] | change
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        make_interface(ksced=config)


@pytest.mark.parametrize("change", [
    dict(functional=None), dict(functional="PBE0"), dict(functional="SCAN"),
    dict(kpts=(2, 1, 1)), dict(mu0=0.1), dict(density_fit=True),
])
def test_unsupported_solver_configuration_is_rejected(change):
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        make_interface(**change)


def test_embedding_enabled_and_arbitrary_fields_rejected_before_scf():
    interface = make_interface()
    interface.configure_electrostatic_embedding(False)
    interface.configure_electrostatic_embedding(True)
    assert interface.embedding
    with pytest.raises((ValueError, NotImplementedError), match="KSCED|ksced"):
        interface.add_electronic_potential(object())
    assert interface.frozen_method[0] is None


def test_plain_periodic_factory_is_unchanged():
    interface = make_interface(ksced=None)
    assert type(interface).__name__ == "PySCFPBCPotential"


def _direct_reference(system, device):
    from pyscf import ksced
    from pyscf.pbc import dft, gto
    if device == "gpu":
        from gpu4pyscf.pbc import dft
    def cell(indices):
        return gto.M(atom=[(str(system.elements[i]), np.asarray(system.positions[i]))
                           for i in indices], a=np.asarray(system.box), unit="Angstrom",
                     basis="gth-szv", pseudo="gth-pbe", mesh=[21]*3, verbose=0)
    mf_b = dft.RKS(cell([0]), xc="PBE")
    mf_b.conv_tol = 1e-12
    mf_b.conv_tol_grad = 1e-9
    mf_b.kernel()
    assert mf_b.converged
    mf_ainb = ksced.embed(dft.RKS(cell([1, 3]), xc="PBE"), mf_b, basis_mode="M")
    mf_ainb.conv_tol = 1e-12
    mf_ainb.conv_tol_grad = 1e-9
    mf_ainb.kernel()
    assert mf_ainb.converged
    return mf_ainb


def _host(value):
    return np.asarray(value.get() if hasattr(value, "get") else value)


def _displace(interface, atom, axis, displacement):
    positions = np.asarray(interface.system.positions).copy()
    positions[atom, axis] += displacement
    interface.system.positions = positions


def _fd_force(interface, atom, axis, step):
    original = np.asarray(interface.system.positions).copy()
    try:
        _displace(interface, atom, axis, step)
        plus = interface.compute_energy()
        _displace(interface, atom, axis, -2*step)
        minus = interface.compute_energy()
    finally:
        interface.system.positions = original
    return -(plus-minus)/(2*step)


@pytest.mark.slow
def test_energy_force_mapping_fd_and_frozen_reuse():
    device = os.environ.get("KSCED_BACKEND", "cpu")
    system = build_system()
    qm = QMHamiltonian(interface="pyscf-pbc", **options(device=device))
    calculator = qm[[1, 3]].build_calculator(system)
    interface = calculator.potential
    result = calculator.calculate()
    cached = interface._scf_state()
    interface.compute_forces()
    assert interface._scf_state() is cached, "Force evaluation invalidated the geometry cache"
    reference = _direct_reference(system, device)
    scale = KJMOL_PER_EH*BOHR_PER_ANGSTROM
    assert result.energy == pytest.approx(reference.energy_potential()*KJMOL_PER_EH, abs=2e-6)
    np.testing.assert_allclose(result.forces[[1, 3]],
                               -_host(reference.Gradients().kernel())*scale,
                               atol=5e-4, rtol=0)
    np.testing.assert_array_equal(result.forces[[0, 2]], np.zeros((2, 3)))
    assert sum(result.components.values()) == pytest.approx(result.energy, abs=1e-9)
    frozen = interface.frozen_method[0]
    env = interface.frozen_environment[0]
    frozen_dm = _host(frozen.make_rdm1()).copy()
    def forbidden_kernel(*args, **kwargs):
        pytest.fail("Frozen subsystem B was reconverged")
    frozen.kernel = forbidden_kernel
    for atom in (1, 3):
        for axis in range(3):
            for h in (2e-3, 1e-3, 5e-4):
                fd = _fd_force(interface, atom, axis, h/BOHR_PER_ANGSTROM)
                error = abs(result.forces[atom, axis]-fd)/scale
                print(f"FD atom={atom} axis={axis} h_bohr={h} error_Ha_Bohr={error:.8g}", flush=True)
                assert error < 1e-5
    assert interface.frozen_environment[0] is env
    # B stays in the MM region. Its MM charges and II/III classification do
    # not change its frozen electronic density or the KSCED A-B energy.
    system.subsystems[0] = Subsystem.III
    system.charges[0] = .3
    system.charges[2] = -.3
    assert interface.compute_energy() == pytest.approx(result.energy, abs=2e-6)
    assert interface.frozen_environment[0] is env
    assert interface.frozen_method[0] is frozen
    np.testing.assert_array_equal(_host(frozen.make_rdm1()), frozen_dm)
    _displace(interface, 1, 0, .012)
    moved_energy = interface.compute_energy()
    moved_force = interface.compute_forces()
    fresh = make_interface(system, device=device)
    assert moved_energy == pytest.approx(fresh.compute_energy(), abs=2e-6)
    np.testing.assert_allclose(moved_force, fresh.compute_forces(), atol=5e-4, rtol=0)
    _displace(interface, 2, 1, .1)  # Uncoupled MM movement cannot change E_ainb.
    assert interface.compute_energy() == pytest.approx(moved_energy, abs=2e-6)
    assert interface.frozen_environment[0] is env


@pytest.mark.slow
@pytest.mark.parametrize("case", ["uks_rks", "rks_uks", "smearing"])
def test_spin_and_free_energy(case):
    device = os.environ.get("KSCED_BACKEND", "cpu")
    kwargs = dict(device=device)
    system = build_system("H" if case == "rks_uks" else "He")
    if case == "uks_rks":
        kwargs.update(charge=1, multiplicity=2)
    elif case == "rks_uks":
        kwargs["ksced"] = options()["ksced"] | dict(frozen_multiplicity=2)
    else:
        kwargs.update(sigma=.15, smearing_method="fermi")
    interface = make_interface(system, **kwargs)
    energy = interface.compute_energy()
    force = interface.compute_forces()[1, 0]
    for h in (1e-3, 5e-4):
        fd = _fd_force(interface, 1, 0, h/BOHR_PER_ANGSTROM)
        np.testing.assert_allclose(force, fd, atol=.05, rtol=0)
    if case == "smearing":
        state = interface._scf_state()
        mf_ainb = state.method
        assert mf_ainb.entropy > 1e-5
        assert energy == pytest.approx(mf_ainb.energy_potential()*KJMOL_PER_EH, abs=2e-6)
        assert abs(mf_ainb.energy_potential()-mf_ainb.e_tot) > 1e-5


@pytest.mark.slow
def test_frozen_and_partition_changes_fail_after_a_cached_calculation():
    interface = make_interface(device=os.environ.get("KSCED_BACKEND", "cpu"))
    interface.compute_energy()
    system = interface.system
    original = np.asarray(system.positions).copy()
    _displace(interface, 0, 0, .01)
    with pytest.raises((ValueError, RuntimeError), match="frozen|Frozen|B"):
        interface.compute_energy()
    system.positions = original
    old_box = np.asarray(system.box).copy()
    system.box = old_box*1.01
    with pytest.raises((ValueError, RuntimeError), match="box|lattice|cell"):
        interface.compute_forces()
    system.box = old_box
    system.subsystems[3] = Subsystem.II
    with pytest.raises((ValueError, RuntimeError), match="partition|QM|subsystem"):
        interface.compute_energy()
    system.subsystems[3] = Subsystem.I
    interface.compute_energy()
    interface.potentials.append(object())
    with pytest.raises((ValueError, NotImplementedError), match="KSCED|ksced"):
        interface.compute_energy()


@pytest.mark.slow
def test_nonconverged_b_does_not_become_a_frozen_environment():
    interface = make_interface(max_cycle=0, device=os.environ.get("KSCED_BACKEND", "cpu"))
    with pytest.raises(RuntimeError, match="converg"):
        interface.compute_energy()
    assert interface.frozen_environment[0] is None


def test_pme_sources_exclude_frozen_b_and_require_same_system():
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    interface = make_interface()
    potential = PMEElectronicPotential(interface.system, .4, (12, 12, 12), 4)
    interface.add_electronic_potential(potential)
    periodic = interface.potentials[0]
    assert periodic.excluded_indices == (0,)
    assert periodic._source_indices() == [2]
    interface.system.subsystems[0] = Subsystem.III
    assert periodic._source_indices() == [2]
    other = PMEElectronicPotential(build_system(), .4, (12, 12, 12), 4)
    with pytest.raises(ValueError, match="same system"):
        interface.add_electronic_potential(other)


def test_embedding_configuration_cannot_change_after_scf():
    interface = make_interface()
    interface.density_guess[0] = np.eye(2)
    with pytest.raises(ValueError, match="before the first SCF"):
        interface.configure_electrostatic_embedding(True)
    with pytest.raises(ValueError, match="before the first SCF"):
        interface.add_electronic_potential(object())
