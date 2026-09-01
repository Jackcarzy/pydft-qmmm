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


# PySCF fixtures.

# SPC/E charges from tests/data/spce_no_lj.xml.
SPCE_CHARGES = {"O": -0.8476, "H": 0.4238}

# SPC/E water at the origin: r(OH) = 1 A, HOH = 109.47 deg.
WATER_GEOMETRY = np.array([
    [0.00000, 0.0, 0.00000],
    [0.81650, 0.0, 0.57735],
    [-0.81650, 0.0, 0.57735],
])


def build_water_system(placements, box_length=12.0):
    """Build SPC/E waters with assigned subsystems.

    Args:
        placements: Pairs of an origin for the oxygen and the subsystem
            the whole molecule belongs to.
        box_length: The edge length (Angstrom) of the cubic box.

    Returns:
        The system, with SPC/E charges and masses assigned.
    """
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = []
    for residue, (origin, subsystem) in enumerate(placements):
        # Match atom names in tests/data/spce_residues.xml.
        for element, name, offset in zip(
                ("O", "H", "H"), ("O", "H1", "H2"), WATER_GEOMETRY,
        ):
            atoms.append(
                Atom(
                    position=np.asarray(origin, dtype=float) + offset,
                    element=element,
                    name=name,
                    residue=residue,
                    residue_name="HOH",
                    chain="A",
                    charge=SPCE_CHARGES[element],
                    mass=ELEMENT_TO_MASS[element],
                    subsystem=subsystem,
                ),
            )
    system = System(atoms, np.eye(3) * box_length)
    assert np.count_nonzero(system.charges) == len(system.charges)
    return system


@pytest.fixture
def pyscf_water_system():
    """A single QM water in a cubic box."""
    return build_water_system([((6.0, 6.0, 6.0), Subsystem.I)])


@pytest.fixture
def pyscf_pbc_system():
    """A single QM water in a small periodic box.

    The periodic QM cell is the simulation box itself, so the box is
    kept small: at ke_cutoff=80 an 8 Angstrom edge is a ~27**3 mesh,
    where a 12 Angstrom one is over three times the points for no extra
    coverage of the physics under test.
    """
    return build_water_system(
        [((4.0, 4.0, 4.0), Subsystem.I)], box_length=8.0,
    )


def build_pbc_interface(system, device="cpu", **overrides):
    """Build a periodic PySCF interface over a system.

    Args:
        system: The system to tie the interface to.
        device: Either ``cpu`` or ``gpu``.
        overrides: Any factory argument to replace.

    Returns:
        The periodic PySCF interface.
    """
    from pydft_qmmm.interfaces.pyscf_pbc.pbc_factory import (
        pyscf_pbc_interface_factory,
    )
    options = dict(
        basis="gth-szv",
        pseudo="gth-pbe",
        functional="pbe",
        charge=0,
        multiplicity=1,
        ke_cutoff=80.0,
        device=device,
        conv_tol=1e-10,
    )
    options.update(overrides)
    return pyscf_pbc_interface_factory(system, **options)


def build_pbc_embedded_system():
    """A QM water with one subsystem II water in a small periodic box."""
    return build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.3, 5.9, 6.1), Subsystem.II),
        ],
        box_length=8.0,
    )


@pytest.fixture
def pyscf_pbc_embedded_factory():
    """Build an embedded periodic interface on a chosen device.

    Both devices get an identical system, so any difference between
    them is the backend rather than the input.
    """
    def build(device="cpu"):
        interface = build_pbc_interface(
            build_pbc_embedded_system(), device=device,
        )
        interface.configure_electrostatic_embedding(True)
        return interface
    return build


@pytest.fixture
def pyscf_pbc_embedded_interface(pyscf_pbc_embedded_factory):
    """An embedded interface, and the same geometry left unembedded.

    The unembedded energy is computed through a separate interface
    rather than by toggling the flag on this one: the SCF cache keys on
    the system state, not on the embedding flag, so toggling would
    return the stale result.
    """
    interface = pyscf_pbc_embedded_factory("cpu")
    bare = build_pbc_interface(interface.system).compute_energy()
    return interface, bare


@pytest.fixture
def pyscf_triplet_system():
    """Molecular oxygen, whose ground state is a triplet."""
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = [
        Atom(
            position=np.array([6.0, 6.0, 6.0 + z]),
            element="O",
            name="O",
            residue=0,
            residue_name="OXY",
            charge=0.0,
            mass=ELEMENT_TO_MASS["O"],
            subsystem=Subsystem.I,
        )
        for z in (-0.604, 0.604)
    ]
    return System(atoms, np.eye(3) * 12.0)


@pytest.fixture
def pyscf_embedded_water():
    """A QM water and an equivalent bare PySCF embedding."""
    from pyscf import dft, gto, qmmm
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    system = build_water_system([
        ((6.0, 6.0, 6.0), Subsystem.I),
        ((6.4, 7.1, 8.7), Subsystem.II),
    ])
    potential = pyscf_interface_factory(
        system,
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-11,
    )
    qm_indices = sorted(system.select("subsystem I"))
    mm_indices = sorted(system.select("subsystem II"))
    mol = gto.M(
        atom=[
            (str(system.elements[i]), tuple(system.positions[i]))
            for i in qm_indices
        ],
        unit="Angstrom",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=0,
    )
    method = dft.RKS(mol, xc="PBE")
    method.conv_tol = 1e-11
    method.grids.level = 3
    method = qmmm.add_mm_charges(
        method,
        np.asarray(system.positions)[mm_indices],
        np.asarray(system.charges)[mm_indices],
        unit="Angstrom",
    )
    return potential, method, qm_indices, mm_indices


@pytest.fixture
def pyscf_pme_system():
    """Waters spanning all three subsystems in a small box."""
    system = build_water_system(
        [
            ((4.0, 4.0, 4.0), Subsystem.I),
            ((4.5, 5.2, 6.3), Subsystem.II),
            ((7.4, 2.1, 8.0), Subsystem.III),
            ((1.6, 7.7, 2.4), Subsystem.III),
        ],
        box_length=10.0,
    )
    assert len(system.select("subsystem I")) > 0
    assert len(system.select("subsystem II")) > 0
    assert len(system.select("subsystem III")) > 0
    return system


@pytest.fixture
def pyscf_pme_adapter(pyscf_pme_system):
    """A PySCF potential carrying a reciprocal PME electronic potential."""
    from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
    from pydft_qmmm.interfaces.pyscf.pyscf_factory import (
        pyscf_interface_factory,
    )
    potential = pyscf_interface_factory(
        pyscf_pme_system,
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-11,
    )
    potential.add_electronic_potential(
        PMEElectronicPotential(pyscf_pme_system, 0.4, (20, 20, 20), 6),
    )
    return potential


@pytest.fixture
def pyscf_openmm_system():
    """Four SPC/E waters spanning all subsystems in a cubic box."""
    system = build_water_system(
        [
            ((12.0, 12.0, 12.0), Subsystem.I),
            ((12.4, 13.1, 14.7), Subsystem.II),
            ((4.0, 19.0, 5.0), Subsystem.III),
            ((19.5, 4.5, 19.5), Subsystem.III),
        ],
        box_length=24.0,
    )
    return system


@pytest.fixture
def mm_pyscf_spce():
    return MMHamiltonian(
        forcefield=[
            "tests/data/spce_no_lj.xml",
            "tests/data/spce_residues.xml",
        ],
        nonbonded_cutoff=9.0,
        pme_gridnumber=48,
        pme_alpha=3.5,
    )


@pytest.fixture
def qm_pyscf_water():
    return QMHamiltonian(
        interface="pyscf",
        basis="sto-3g",
        functional="PBE",
        charge=0,
        multiplicity=1,
        conv_tol=1e-10,
        # Level 5 limits moving-grid force error to about
        # 0.003 kJ/mol/Angstrom for this fixture.
        grid_level=5,
    )


@pytest.fixture
def pyscf_iodide_system():
    """An iodide ion for effective-core-potential tests."""
    from pydft_qmmm import Atom
    from pydft_qmmm.utils import ELEMENT_TO_MASS
    atoms = [
        Atom(
            position=np.array([6.0, 6.0, 6.0]),
            element="I",
            name="I",
            residue=0,
            residue_name="IOD",
            charge=-1.0,
            mass=ELEMENT_TO_MASS["I"],
            subsystem=Subsystem.I,
        ),
    ]
    return System(atoms, np.eye(3) * 12.0)
