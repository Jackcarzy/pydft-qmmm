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
