"""QM/MM mechanical-embedding example using VASP for the QM region.

The system is a 4-water cluster in a 12 A periodic box.  The first water
(atoms 0-2) is treated by VASP at the PBE level; the remaining three
waters (atoms 3-11) are MM under SPC/E.  Coupling is mechanical
embedding only -- VASP sees the QM atoms alone and QM/MM coupling is
carried by the MM force field.

Running this script requires a VASP executable on PATH (the ``command``
keyword, or the PYDFT_QMMM_VASP_COMMAND / ASE_VASP_COMMAND environment
variable) and a POTCAR library pointed to by ``pp_path`` or VASP_PP_PATH,
e.g. ``.../pseudopotential/potpaw_PBE.54``.

Note: the VASP interface currently supports mechanical embedding only.
Electrostatic embedding requires a VASP binary compiled with -DPLUGINS
and raises NotImplementedError until that path is wired up.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import SETTLE

# Load the small water cluster.
system = System.load("water4.pdb")

# Assign initial velocities at 300 K.
system.velocities = generate_velocities(
    system.masses,
    300,
    10101,
)

# QM Hamiltonian backed by VASP.  Keyword arguments after ``charge`` are
# forwarded to the interface factory; bare keywords such as ``encut``
# become INCAR tags (upper-cased).  Tags whose names are not Python
# identifiers go through the ``incar`` dict.
qm = QMHamiltonian(
    interface="vasp",
    charge=0,
    directory="./vasp_workdir",
    kpts=(1, 1, 1),
    encut=500,
    ediff=1e-6,
    ismear=0,
    sigma=0.05,
    incar={"GGA": "PE"},
)

# MM Hamiltonian using SPC/E.  Cutoff is kept under half the 12 A box.
mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="CutoffPeriodic",
    nonbonded_cutoff=5.0,
)

# Mechanical embedding for both close- and long-range coupling.
qmmm = QMMMHamiltonian("mechanical", "mechanical")

# QM = first water (atoms 0-2); MM = waters 2-4 (atoms 3-11).
total = qm[:3] + mm[3:] + qmmm

# Verlet integrator with a 1 fs step.
integrator = VerletIntegrator(1)

# SETTLE rigid-water constraint applied to MM waters only (the plugin
# automatically excludes subsystem I).
settle = SETTLE()

simulation = Simulation(
    system=system,
    hamiltonian=total,
    integrator=integrator,
    plugins=[settle],
    output_directory="output_api/",
    log_decimal_places=6,
    csv_decimal_places=6,
)

simulation.run_dynamics(2)
