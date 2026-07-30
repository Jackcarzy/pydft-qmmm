from __future__ import annotations

import json
import os
import pathlib

import pytest

from pydft_qmmm import MMHamiltonian
from pydft_qmmm import QMHamiltonian
from pydft_qmmm import System
from pydft_qmmm import VerletIntegrator
from pydft_qmmm.plugins import SETTLE
from pydft_qmmm.utils import Subsystem


@pytest.fixture
def spce_system():
    return System.load(
        "tests/data/spce_qmmm.pdb",
    )


# @pytest.fixture
# def spce_dimer_system():
#    return System.load(
#        "tests/data/hoh_dimer.pdb",
#    )


@pytest.fixture
def spce_qmmm_system(spce_system):
    with open("tests/data/spce_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        spce_system.subsystems[atom] = Subsystem.II
    return spce_system


@pytest.fixture
def qm_water():
    return QMHamiltonian(
        basis="def2-SVP",
        functional="PBE",
        charge=0,
        multiplicity=1,
        guess="read",
    )


@pytest.fixture
def mm_spce():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce.xml",
            "tests/data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def mm_spce_no_lj():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce_no_lj.xml",
            "tests/data/spce_residues.xml",
        ],
        pme_gridnumber=30,
        pme_alpha=5.0,
    )


@pytest.fixture
def spce_plugins():
    return [SETTLE()]


@pytest.fixture
def verlet():
    return VerletIntegrator(1)


@pytest.fixture
def no_logging():
    return {
        "log_write": False,
        "csv_write": False,
        "dcd_write": False,
    }


VASP_PP_LIBRARY = os.environ.get(
    "VASP_PP_PATH",
    "/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/potpaw_PBE.64",
)


@pytest.fixture
def vasp_pp_library():
    """The POTCAR library used by the VASP interface tests."""
    return VASP_PP_LIBRARY


@pytest.fixture
def vasp_workdir(tmp_path, request):
    """Where VASP-backed tests should put their run directories.

    Defaults to pytest's tmp_path, which keeps the unit tests hermetic.
    On a compute node tmp_path lives in node-local /tmp and evaporates
    when the job ends, taking the OUTCAR with it -- so setting
    PYDFT_QMMM_VASP_TESTDIR redirects runs somewhere durable and makes
    failures diagnosable after the fact.
    """
    root = os.environ.get("PYDFT_QMMM_VASP_TESTDIR")
    if not root:
        return tmp_path
    path = os.path.join(root, request.node.name)
    os.makedirs(path, exist_ok=True)
    return pathlib.Path(path)


@pytest.fixture
def vasp_qmmm_system(spce_system):
    """An SPC/E system with an explicit QM region.

    The spce_qmmm_system fixture only assigns subsystem II; subsystem I
    is normally set when a Hamiltonian builds its calculator.  The VASP
    interface tests exercise the interface directly, so they need the QM
    region assigned up front.
    """
    with open("tests/data/spce_qmmm_region_ii.json") as fh:
        embedding_list = json.load(fh)
    for atom in embedding_list:
        spce_system.subsystems[atom] = Subsystem.II
    # One water molecule as the QM subsystem.
    for atom in range(3):
        spce_system.subsystems[atom] = Subsystem.I
    return spce_system


@pytest.fixture
def vasp_embedded(vasp_qmmm_system, vasp_workdir):
    """A VASP potential with electrostatic embedding switched on."""
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    return vasp_interface_factory(
        vasp_qmmm_system,
        directory=str(vasp_workdir / "vasp"),
        pp_path=VASP_PP_LIBRARY,
        embedding=True,
        # The QM region is one water, but the VASP cell must match the
        # MM box (29.9 A) so the embedded charges land in the right
        # place -- it cannot be shrunk.  At the default ENCUT=400 that
        # is a ~196**3 fine grid, which OOM-killed a 12 GB job.  These
        # tests ask "does the plugin fire and is E consistent with F",
        # not "is the energy converged", so a lighter cutoff is right.
        incar={"ENCUT": 250},
    )
