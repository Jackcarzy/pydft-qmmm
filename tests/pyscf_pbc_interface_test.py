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
