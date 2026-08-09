"""QM/MM mechanical-embedding example using SPARC for the QM region.

The system is a 4-water cluster in a 12 Å periodic box.  The first
water (atoms 0-2) is treated by SPARC at the PBE level; the remaining
three waters (atoms 3-11) are MM under SPC/E.  Coupling is mechanical
embedding only — SPARC does not see MM point charges.

Running this script requires a working ``sparc`` executable on PATH and
SPARC pseudopotentials available (e.g. via the SPARC_PSP_PATH env var).
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

# QM Hamiltonian backed by SPARC.  Keyword arguments after ``charge``
# are forwarded to ``sparc.calculator.SPARC``.
qm = QMHamiltonian(
    interface="sparc",
    charge=0,
    xc="pbe",
    h=0.25,
    kpts=(1, 1, 1),
    tol_scf=1e-5,
    directory="./sparc_workdir",
)

# MM Hamiltonian using SPC/E.  Cutoff is kept under half the 12 Å box.
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
