"""Regression tests for conservative PME exclusion forces."""
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.potentials.pme_potential import PMEExcludedPotential
from pydft_qmmm.utils import Subsystem


def _excluded_potential() -> PMEExcludedPotential:
    atoms = [
        Atom(
            position=np.array([2.0, 2.0, 2.0]),
            charge=-0.8,
            element="O",
            subsystem=Subsystem.I,
        ),
        Atom(
            position=np.array([3.0, 2.0, 2.0]),
            charge=0.4,
            element="H",
            subsystem=Subsystem.II,
        ),
        Atom(
            position=np.array([7.0, 6.0, 5.0]),
            charge=0.4,
            element="H",
            subsystem=Subsystem.III,
        ),
    ]
    system = System(atoms, box=np.diag([10.0, 10.0, 10.0]))
    return PMEExcludedPotential(system, 0.4, (20, 20, 20), 6)


def test_excluded_pme_force_differentiates_energy_on_field_source() -> None:
    """Moving a field-source atom must produce its energy derivative."""
    potential = _excluded_potential()
    atom = 2
    step = 0.001
    numerical = np.zeros(3)
    for axis in range(3):
        potential.system.positions[atom, axis] += step
        plus = potential.compute_energy()
        potential.system.positions[atom, axis] -= 2 * step
        minus = potential.compute_energy()
        potential.system.positions[atom, axis] += step
        numerical[axis] = -(plus - minus) / (2 * step)

    assert np.abs(numerical).max() > 1e-3
    assert potential.compute_forces()[atom] == pytest.approx(
        numerical, abs=1e-4,
    )
