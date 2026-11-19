"""OpenMM device defaults and explicit precision overrides."""
from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from pydft_qmmm import MMHamiltonian, QMHamiltonian
from pydft_qmmm.interfaces.openmm import openmm_factory


def particle():
    system = openmm.System()
    system.addParticle(1.)
    return system, SimpleNamespace(positions=np.zeros((1, 3)))


def test_default_cpu_context_uses_reference():
    context = openmm_factory._build_omm_context(*particle())
    assert context.getPlatform().getName() == 'Reference'


def test_explicit_cpu_double_uses_reference():
    with pytest.warns(RuntimeWarning, match='Reference'):
        context = openmm_factory._build_omm_context(*particle(), platform='CPU')
    assert context.getPlatform().getName() == 'Reference'


def test_explicit_cpu_platform_precision():
    context = openmm_factory._build_omm_context(*particle(), platform='CPU', precision=None)
    assert context.getPlatform().getName() == 'CPU'


@pytest.mark.parametrize('platform', ['CUDA', 'OpenCL'])
@pytest.mark.parametrize('precision', ['double', 'mixed', 'single', None])
def test_gpu_precision_properties(monkeypatch, platform, precision):
    selected = SimpleNamespace(getPropertyNames=lambda: ['Precision'])
    monkeypatch.setattr(openmm.Platform, 'getPlatformByName', lambda name: selected)
    seen = []
    def context(system, integrator, chosen, properties):
        assert chosen is selected
        seen.append(properties)
        return SimpleNamespace(setPositions=lambda positions: None)
    monkeypatch.setattr(openmm, 'Context', context)
    openmm_factory._build_omm_context(*particle(), platform=platform, precision=precision)
    assert seen == ([{}] if precision is None else [{'Precision': precision}])


@pytest.mark.parametrize('engine', ['vasp', 'pyscf-pbc', 'psi4', 'sparc'])
@pytest.mark.parametrize('platform', [None, 'Reference', 'CPU', 'OpenCL'])
@pytest.mark.parametrize('mm_first', [False, True])
def test_composite_device_selection(monkeypatch, engine, platform, mm_first):
    seen = []
    monkeypatch.setattr(QMHamiltonian, 'build_calculator', lambda self, system: None)
    def build(self, system):
        seen.append(dict(self.options))
    monkeypatch.setattr(MMHamiltonian, 'build_calculator', build)
    options = {'forcefield': ['test.xml']}
    if platform is not None:
        options['platform'] = platform
    mm = MMHamiltonian(**options)
    qm = QMHamiltonian(interface=engine)
    (mm + qm if mm_first else qm + mm).build_calculator(None)
    expected = dict(options)
    if platform is None and engine in {'vasp', 'pyscf-pbc'}:
        expected['platform'] = 'CUDA'
    assert seen == [expected]
    assert mm.options == options
    # Reusing the same MM Hamiltonian must not retain a GPU default.
    (QMHamiltonian(interface='psi4') + mm).build_calculator(None)
    assert seen[-1] == options
