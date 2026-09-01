"""Fast unit tests for the periodic PySCF interface.  No SCF here.

Anything that converges an SCF belongs in pyscf_pbc_integration_test.py
and runs on a compute node.
"""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import (
    pyscf_pbc_interface_factory,
)


def _kwargs(**overrides):
    base = dict(
        basis="gth-szv",
        pseudo="gth-pbe",
        charge=0,
        multiplicity=1,
        functional="pbe",
        ke_cutoff=80.0,
    )
    base.update(overrides)
    return base


def test_pyscf_pbc_is_bundled():
    from pydft_qmmm.interfaces import interfaces
    from pydft_qmmm.utils import TheoryLevel
    assert interfaces["pyscf_pbc"][0] is TheoryLevel.QM


def test_pseudo_is_mandatory(pyscf_pbc_system):
    with pytest.raises(ValueError, match="pseudo"):
        pyscf_pbc_interface_factory(pyscf_pbc_system, **_kwargs(pseudo=None))


def test_ecp_is_rejected(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ecp"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(ecp="def2-svp"),
        )


def test_ke_cutoff_and_mesh_together_raise(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ke_cutoff.*mesh|mesh.*ke_cutoff"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(mesh=(24, 24, 24)),
        )


def test_one_of_ke_cutoff_or_mesh_is_required(pyscf_pbc_system):
    with pytest.raises(ValueError, match="ke_cutoff|mesh"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(ke_cutoff=None),
        )


def test_embedding_sigma_must_be_positive(pyscf_pbc_system):
    with pytest.raises(ValueError, match="embedding_sigma"):
        pyscf_pbc_interface_factory(
            pyscf_pbc_system, **_kwargs(embedding_sigma=0.0),
        )


def test_a_valid_configuration_builds(pyscf_pbc_system):
    interface = pyscf_pbc_interface_factory(pyscf_pbc_system, **_kwargs())
    assert interface.pseudo == "gth-pbe"
    assert interface.ke_cutoff == 80.0
    assert interface.embedding_sigma == 0.3


# ---------------------------------------------------------------------
# The periodic cell
# ---------------------------------------------------------------------


def test_cell_lattice_matches_the_system_box(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    cell, qm_indices = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe",
        80.0, None, 0, 1, 0,
    )
    # cell.a is in Angstrom, as system.box is.
    np.testing.assert_allclose(np.asarray(cell.a), pyscf_pbc_system.box)
    assert qm_indices == tuple(sorted(pyscf_pbc_system.select("subsystem I")))


def test_valence_charges_are_not_atomic_numbers(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import (
        build_cell, valence_charges,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe",
        80.0, None, 0, 1, 0,
    )
    charges = valence_charges(cell)
    # gth-pbe oxygen carries 6 valence electrons, not 8.
    assert charges.sum() == pytest.approx(cell.nelectron)
    assert charges.max() == pytest.approx(6.0)


# ---------------------------------------------------------------------
# The external potential on the uniform grid
# ---------------------------------------------------------------------


def test_near_potential_matches_the_analytic_smeared_coulomb():
    """Smeared charges make sum_j q_j erf(r_j / (sqrt(2) sigma)) / r_j.

    The charges must be neutral overall.  ``poisson_fft`` zeroes the
    G=0 component, so a single charge in a periodic cell sits in a
    neutralizing background and does not follow the isolated form at
    all -- it decays far faster.  A neutral pair has no G=0 component
    to lose, so the near field is the analytic one up to the periodic
    images, which a 20 Angstrom box keeps to about two percent.
    """
    from scipy.special import erf
    from pydft_qmmm.interfaces.vasp.grid_potential import interpolate_at
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        _smeared_potential_grid,
    )
    box = np.eye(3) * 20.0
    mesh = (80, 80, 80)
    sigma = 0.5
    positions = np.array([[9.0, 10.0, 10.0], [11.0, 10.0, 10.0]])
    charges = np.array([1.0, -1.0])
    phi = _smeared_potential_grid(positions, charges, mesh, box, sigma)
    axis = np.linspace(1.2, 3.0, 7)
    probe = np.stack(
        [10.0 + axis, np.full_like(axis, 11.5), np.full_like(axis, 10.0)],
        axis=-1,
    )
    got = interpolate_at(phi, box, probe)
    expected = np.zeros(len(probe))
    for position, charge in zip(positions, charges):
        distance = np.linalg.norm(probe - position, axis=1)
        expected += charge * erf(
            distance / (np.sqrt(2.0) * sigma),
        ) / distance * 14.399645
    np.testing.assert_allclose(got, expected, rtol=3e-2)


def test_grid_weights_sum_to_the_cell_volume(pyscf_pbc_system):
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        grid_coordinates,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe", 80.0, None, 0, 1, 0,
    )
    coords, weights = grid_coordinates(cell)
    assert coords.shape == (weights.size, 3)
    # cell.vol is in Bohr**3, and the weights are the uniform measure.
    assert weights.sum() == pytest.approx(cell.vol)


def test_gaussian_contraction_is_the_adjoint_of_spreading():
    """<spread(q), phi> must differentiate to -contract(phi) on a shift.

    The two halves of the MM force channel are a transpose pair.  If
    they disagree about sigma, the cutoff, or a sign, the MM forces stop
    being the derivative of the embedding energy.  This pins the pair
    directly, without needing an SCF.
    """
    from pydft_qmmm.interfaces.vasp.grid_potential import (
        contract_gaussian_gradient, spread_gaussian,
    )
    box = np.eye(3) * 12.0
    mesh = (48, 48, 48)
    sigma = 0.5
    position = np.array([[5.3, 6.1, 5.8]])
    charge = np.array([0.7])
    # An arbitrary smooth periodic field to contract against.
    axes = [np.arange(n) / n for n in mesh]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    field = np.sin(2.0 * np.pi * grid[..., 0]) * np.cos(
        2.0 * np.pi * grid[..., 1],
    )
    analytic = contract_gaussian_gradient(
        field, position, charge, mesh, box, sigma,
    )[0]
    volume_element = abs(np.linalg.det(box)) / np.prod(mesh)
    step = 1e-3
    numeric = np.zeros(3)
    for axis in range(3):
        shifted = position.copy()
        shifted[0, axis] += step
        plus = np.sum(
            spread_gaussian(shifted, charge, mesh, box, sigma) * field,
        ) * volume_element
        shifted[0, axis] -= 2.0 * step
        minus = np.sum(
            spread_gaussian(shifted, charge, mesh, box, sigma) * field,
        ) * volume_element
        numeric[axis] = -(plus - minus) / (2.0 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------
# The one-electron operator
# ---------------------------------------------------------------------


def test_constant_potential_recovers_the_electron_count(pyscf_pbc_system):
    """Tr(D . V_ao) for constant V must be -N_elec * V.

    A uniform mesh cannot integrate an all-electron density; it can
    integrate a pseudopotential one.  If this drifts, the quadrature is
    no longer resolving the density and every embedding energy is
    wrong.  This is the assertion that catches a 170-electrons-instead
    -of-10 quadrature failure directly.
    """
    import pyscf
    from pyscf import pbc as cpu_pbc
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_cell import build_cell
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_embedding import (
        ao_operator, grid_coordinates,
    )
    cell, _ = build_cell(
        pyscf_pbc_system, "gth-szv", "gth-pbe", 80.0, None, 0, 1, 0,
    )
    kpts = cell.make_kpts([1, 1, 1])
    coords, weights = grid_coordinates(cell)
    constant = 0.25
    potential = np.full(len(coords), constant)
    matrix = ao_operator(pyscf, cell, kpts, coords, weights, potential)
    mf = cpu_pbc.dft.KRKS(cell, kpts=kpts, xc="pbe")
    overlap = mf.get_ovlp()
    # For a constant potential the operator IS the overlap, scaled.
    # Comparing to the analytic overlap rather than to the electron
    # count isolates the quadrature: the initial guess is normalized to
    # 7.9904 rather than 8, which would otherwise show up here as a
    # 0.1% error that has nothing to do with the integration.
    np.testing.assert_allclose(
        matrix, constant * overlap, rtol=0, atol=1e-4,
    )
    dm = mf.get_init_guess()
    trace = np.einsum("kij,kji->", dm, matrix).real
    reference = constant * np.einsum("kij,kji->", dm, overlap).real
    assert trace == pytest.approx(reference, rel=1e-4)
