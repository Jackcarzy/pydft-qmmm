<p align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-light.svg">
  <img alt="The PyDFT-QMMM logo." width=75% src="https://github.com/johnppederson/pydft-qmmm/blob/master/docs/_media/pydft-qmmm-light.svg">
<picture>

</p>

PyDFT-QMMM: A Modular Framework for DFT-QM/MM Simulation
========================================================

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=fff)
[![License](https://img.shields.io/badge/license-LGPL_2.1-blue.svg)](https://opensource.org/license/lgpl-2-1)

[![Build](https://github.com/johnppederson/pydft-qmmm/actions/workflows/test_and_coverage.yml/badge.svg)](https://github.com/johnppederson/pydft-qmmm/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/johnppederson/f0e19ee2c0a71030ace6067046a59ff1/raw/pydft_qmmm.json)](https://github.com/johnppederson/pydft-qmmm/actions)
<!-- [![Deployment](https://github.com/johnppederson/pydft-qmmm/actions/workflows/build_and_deploy.yml/badge.svg)](https://pypi.org/project/pydft-qmmm/) -->
[![Docs](https://github.com/johnppederson/pydft-qmmm/actions/workflows/docs.yml/badge.svg)](https://johnppederson.com/pydft-qmmm/)

</p>

Introduction
------------

PyDFT-QMMM implements QM/MM dynamics for several different PBC QM/MM
approaches, including the QM/MM/Cutoff and
[QM/MM/PME](https://doi.org/10.1063/5.0087386) methods.  Visit our
[website](https://johnppederson.com/pydft-qmmm/) for full documentation.

Requirements
------------
* Python >= 3.10
* [NumPy](https://github.com/numpy/numpy)
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [OpenMM](https://github.com/openmm/openmm)
  [(OpenMM licenses)](https://github.com/openmm/openmm/blob/master/docs-source/licenses/Licenses.txt).
* A QM engine, such as the default [Psi4](https://github.com/psi4/psi4) >= 1.10
  [(LGPL-3.0 license)](https://opensource.org/license/LGPL-3-0).

### Configured environments

The following environments and engine installations describe the current
Georgia Tech setup.  The paths are site-specific and are not general package
requirements.

| Engine | Engine source |
| --- | --- |
| Psi4 |  Psi4 1.11 |
| `pyscf-mol` | PySCF 2.14; GPU4PySCF for GPU calculations |
| `pyscf-pbc` | PySCF 2.14 periodic solver; GPU4PySCF for GPU calculations |
| VASP | External `vasp_std` 6.6.1 executable plus the Python plugin |
| SPARC | `sparc-x-api` plus an external `sparc` executable |

The environment needs these important packages:

* Python 3.10.20
* Psi4 1.11
* OpenMM 8.5.2
* NumPy 2.2.6
* helPME-py 0.2.2
* QCEngine and QCElemental
* ASE 3.29.0
* Pint 0.24.4

#### Required for QM/MM/PME
* [helPME-py](https://github.com/johnppederson/helpmy-py) required for evaluating
  a PME potential on an arbitrary set of coordinates
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Required for Enhanced Sampling
* [PLUMED](https://github.com/plumed/plumed2) required for enhanced sampling
  [(LGPL-3.0 license)](https://opensource.org/license/LGPL-3-0).

#### Required for Optimization
* [geomeTRIC](https://github.com/leeping/geomeTRIC) required for optimization
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).

#### Required for Expanded MM Input File Types
* [ParmEd](https://github.com/ParmEd/ParmEd) required for reading GROMACS, AMBER,
  and CHARMM forcefields/topologies [(LGPL-2.1 license)](https://opensource.org/license/LGPL-2-1).

#### Required for Testing
* [pytest](https://github.com/pytest-dev/pytest) required for performing tests
  [(MIT license)](https://opensource.org/license/MIT).
* [pytest-cov](https://github.com/pytest-dev/pytest-cov) required for performing coverage
  analysis [(MIT license)](https://opensource.org/license/MIT).

#### Required for Development
* [pre-commit](https://github.com/pre-commit/pre-commit) required for maintaining typing and
  formatting standards [(MIT license)](https://opensource.org/license/MIT).

#### Required for Documentation
* [Sphinx](https://github.com/sphinx-doc/sphinx) required for generating documentation
  [(BSD-2-clause license)](https://opensource.org/licenses/BSD-2-Clause).
* [sphinx-autodoc-typehints](https://github.com/tox-dev/sphinx-autodoc-typehints) required for
  generating type hints [(MIT license)](https://opensource.org/licenses/MIT).
* [Furo](https://github.com/pradyunsg/furo) theme for Sphinx documentation
  [(MIT license)](https://opensource.org/license/MIT).

Installation
------------

PyDFT-QMMM can be installed directly from github using ``pip``:

```bash
python -m pip install git+https://github.com/johnppederson/pydft-qmmm
```

Alternatively, you can clone the repository and install using ``pip``:

```bash
git clone https://github.com/johnppederson/pydft-qmmm
cd pydft-qmmm
pip install .
```

Install both PySCF interfaces with QM/MM/PME support from this checkout:

```bash
python -m pip install '.[pyscf,qmmm-pme]'
```

PySCF engines
-------------

Select `pyscf-mol` for a molecular QM region or `pyscf-pbc` for a periodic
QM wavefunction. The old engine names `pyscf` and `pyscf_pbc` are not
accepted; the installation extra remains `pyscf`.

| Setting | `pyscf-mol` | `pyscf-pbc` |
|---|---|---|
| QM model | Molecule in a periodic MM environment | Periodic QM cell matching the simulation box |
| Basis/core treatment | Molecular basis; explicit `ecp` when needed | GTH basis and required `pseudo`; no `ecp` |
| Integration grid | Molecular quadrature: `grid_level` | Uniform FFT grid: `ke_cutoff` or `mesh` |
| Region II | Point charges | Full periodic Gaussian field |
| Region III | Molecular PME with local exclusions | Region III-only reciprocal PME field |

```python
from pydft_qmmm import QMHamiltonian

qm_mol = QMHamiltonian(
    interface="pyscf-mol", basis="def2-svp", functional="pbe",
    charge=0, multiplicity=1, grid_level=5,
)
qm_pbc = QMHamiltonian(
    interface="pyscf-pbc", basis="gth-dzvp", pseudo="gth-pbe",
    functional="pbe", charge=0, multiplicity=1,
    ke_cutoff=200.0, embedding_sigma=0.3,
)
```

Both engines default to `device="cpu"`. Set `device="gpu"` with GPU4PySCF
and a compatible CUDA/CuPy environment. GPU4PySCF is installed separately.

For `pyscf-pbc`, `ke_cutoff` uses Hartree and `embedding_sigma` uses Å.
Supply either `ke_cutoff` or `mesh`, and converge forces as well as energies.
Only Gamma-point sampling is supported; stress is unavailable. Regions I
and II are removed from PME sources, so empty III gives zero reciprocal field.

The examples include MM setup and electrostatic coupling. Run each script
from its example directory.

| Example | Engine | System |
|---|---|---|
| [Case 4](examples/case_4_vasp) | `vasp` | Periodic QM water with PAW embedding |
| [Case 5](examples/case_5_pyscf) | `pyscf-mol` | QM water in SPC/E water |
| [Case 6](examples/case_6_pyscf_ecp) | `pyscf-mol` | Iodide with an ECP |
| [Case 7](examples/case_7_pyscf_gpu) | `pyscf-mol`, GPU | Chloromethane–chloride complex in TIP3P water |
| [Case 8](examples/case_8_pyscf_pbc) | `pyscf-pbc` | Periodic QM water in a 12 Å SPC/E box |
