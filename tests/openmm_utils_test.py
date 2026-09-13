from __future__ import annotations

import openmm
import pytest

from pydft_qmmm.interfaces.openmm.openmm_utils import _exclude_lennard_jones


def _system_with_sigma(sigma: float) -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.addParticle(0.0, sigma, 0.0)
    system.addForce(force)
    return system


def test_exclude_lennard_jones_handles_zero_sigma():
    system = _system_with_sigma(0.0)
    _exclude_lennard_jones(system, frozenset({0}))
    force = system.getForce(0)
    _, sigma, epsilon = force.getParticleParameters(0)
    assert sigma._value == pytest.approx(1.0)
    assert epsilon._value == pytest.approx(0.0)


def test_exclude_lennard_jones_normalizes_nonzero_sigma():
    system = _system_with_sigma(0.3)
    _exclude_lennard_jones(system, frozenset({0}))
    force = system.getForce(0)
    _, sigma, epsilon = force.getParticleParameters(0)
    assert sigma._value == pytest.approx(1.0)
    assert epsilon._value == pytest.approx(0.0)
