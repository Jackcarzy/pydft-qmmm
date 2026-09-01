"""SCF-bearing tests for the periodic PySCF interface.

Every test here runs an SCF and belongs on a compute node.  The login
node has a 4 GB per-user cap and is heavily throttled.  Run with:

    sbatch ../pbc-runs/integration.slurm
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_embedding_shifts_the_energy(pyscf_pbc_embedded_interface):
    """A charged environment must move the QM energy, and sanely."""
    interface, bare = pyscf_pbc_embedded_interface
    embedded = interface.compute_energy()
    assert embedded != pytest.approx(bare, abs=1e-6)
    # An MM environment of ordinary point charges shifts a small QM
    # region by tens of kJ/mol, not by thousands.  A four-figure shift
    # is the signature of a quadrature failure, so this bound is a
    # guard rather than a curiosity: do not widen it to make it pass.
    assert abs(embedded - bare) < 1000.0


def test_components_sum_to_the_energy(pyscf_pbc_embedded_interface):
    interface, _ = pyscf_pbc_embedded_interface
    components = interface.compute_components()
    assert sum(components.values()) == pytest.approx(
        interface.compute_energy(), rel=1e-9,
    )
