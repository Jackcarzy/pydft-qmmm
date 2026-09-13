"""QM/MM electrostatic-embedding example using SPARC for the QM region.

The system is the same 4-water cluster used by the mechanical-embedding
example next door, so the two are directly comparable, as is the Psi4
reference in ../psi4/.  The first water (atoms 0-2) is treated by SPARC
at the PBE level; the remaining three (atoms 3-11) are MM under SPC/E.

Unlike the mechanical case, SPARC sees the MM point charges: they are
spread as Gaussians onto SPARC's grid and enter its effective potential,
and the QM electrostatic potential is contracted back against the same
Gaussians to give the forces on the MM atoms.

Requires a SPARC binary built from the qmmm-embedding branch.  A stock
SPARC will be detected and rejected rather than silently producing an
unembedded result.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import SETTLE

system = System.load("water4.pdb")

system.velocities = generate_velocities(system.masses, 300, 10101)

# FD_GRID is pinned rather than derived from h, so the driver knows the
# grid before SPARC runs and can build V_ext on it.
qm = QMHamiltonian(
    interface="sparc",
    charge=0,
    xc="pbe",
    fd_grid=(60, 60, 60),
    kpts=(1, 1, 1),
    tol_scf=1e-6,
    directory="./sparc_workdir",
    embedding=True,
    embedding_sigma=0.3,
)

mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="CutoffPeriodic",
    nonbonded_cutoff=5.0,
)

# Electrostatic close range, cutoff long range.  Switch the second
# argument to "electrostatic" for full QM/MM/PME.
qmmm = QMMMHamiltonian("electrostatic", "cutoff")

total = qm[:3] + mm[3:] + qmmm

integrator = VerletIntegrator(1)
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
