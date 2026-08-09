from __future__ import annotations

import json
import os
import pathlib

import numpy as np
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
    # System.load() leaves every charge at zero -- charges are normally
    # assigned when an MM Hamiltonian builds its calculator.  Without
    # this the embedding tests build an IDENTICALLY ZERO V_ext and pass
    # while testing no physics whatsoever.  Values are the SPC/E
    # charges from tests/data/spce_no_lj.xml.
    charges = np.asarray(spce_system.charges)
    for atom, element in enumerate(spce_system.elements):
        charges[atom] = -0.8476 if str(element) == "O" else 0.4238
    assert np.count_nonzero(charges) == len(charges), "charges not applied"
    return spce_system


@pytest.fixture
def vasp_three_subsystem_system(vasp_qmmm_system):
    """vasp_qmmm_system with a genuinely populated subsystem III.

    The stock fixture assigns 3 atoms to subsystem I and 1149 to
    subsystem II, leaving subsystem III EMPTY.  That makes any assertion
    of the form "VASP contributes nothing to subsystem III" vacuously
    true, since it ranges over a zero-length array.  Demote the outer
    half of subsystem II so such assertions have atoms to range over.
    """
    near = sorted(vasp_qmmm_system.select("subsystem II"))
    for atom in near[len(near) // 2:]:
        vasp_qmmm_system.subsystems[atom] = Subsystem.III
    assert len(vasp_qmmm_system.select("subsystem III")) > 0
    assert len(vasp_qmmm_system.select("subsystem II")) > 0
    return vasp_qmmm_system


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


@pytest.fixture
def h_constant_field_system():
    """H atom in a constant field.

    One hydrogen at the centre of a 10 x 10 x 30 cell (subsystem I) and
    two oppositely charged sheets at z = 20 and z = 0 (subsystem II).

    The atom is charge neutral and barely polarizable, so the TOTAL
    force must be ~0.  VASP alone reports 1e * E_ext, because it omits
    the core-field interaction; Eq. 4 supplies exactly that.  Unlike a
    QM/MM system, the correction here IS the entire signal.
    """
    system = System.load("tests/data/h_constant_field.pdb")
    SURFACE_CHARGE = 0.001                      # e / Angstrom**2
    sheet = [i for i, e in enumerate(system.elements) if str(e) == "Ne"]
    per_sheet = len(sheet) // 2
    per_site = SURFACE_CHARGE * 100.0 / per_sheet
    charges = np.asarray(system.charges)
    for n, atom in enumerate(sheet):
        charges[atom] = per_site if n < per_sheet else -per_site
    system.subsystems[0] = Subsystem.I
    for atom in sheet:
        system.subsystems[atom] = Subsystem.II
    assert abs(charges.sum()) < 1e-9, "sheets must be net neutral"
    return system


@pytest.fixture
def vasp_pme_system(vasp_qmmm_system):
    """QM/MM partition with a real subsystem III, for the PME path.

    vasp_qmmm_system leaves every atom in I or II, so "not subsystem
    III" covers the whole system and the PME reciprocal sum is almost
    entirely cancelled by its own real-space adjustment.  Assigning the
    remainder to III gives PME something to do: I is the QM region, II
    is handled by the real-space cutoff, and III is what PME sums.
    """
    assigned = set(vasp_qmmm_system.select("subsystem I")) | set(
        vasp_qmmm_system.select("subsystem II"),
    )
    for atom in range(len(vasp_qmmm_system.elements)):
        if atom not in assigned:
            vasp_qmmm_system.subsystems[atom] = Subsystem.III
    assert len(vasp_qmmm_system.select("subsystem III")) > 0
    return vasp_qmmm_system
