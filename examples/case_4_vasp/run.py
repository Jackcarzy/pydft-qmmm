"""QM/MM/PME single point with VASP, analytic near field.

One water in a box of 123 SPC/E waters is treated by VASP at PBE/PAW;
the rest are MM.  Waters within 3 A form subsystem II and reach the QM
region through a near field rebuilt from their real positions directly
on VASP's own grid; everything beyond reaches it through PME.

Requires a vasp_std built with -DPLUGINS.
"""

from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition
from pydft_qmmm.utils import center_positions, wrap_positions

# Load system first.
system = System.load("box_128_min.pdb")

system.positions = center_positions(system.positions, system.box, [0, 1, 2])
system.positions = wrap_positions(
    system.positions, system.box, system.residue_map,
)

# Define QM Hamiltonian.
qm = QMHamiltonian(
    interface="vasp",
    charge=0,
    kpts=(1, 1, 1),
    encut=400,
    ediff=1e-6,
    ismear=0,
    sigma=0.05,
    embedding_sigma=0.3,
    incar={
        "GGA": "PE", "PREC": "Accurate", "ISYM": 0, "LREAL": False,
        "NGXF": 108, "NGYF": 108, "NGZF": 108,
    },
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=7.0,
    pme_gridnumber=(40, 40, 40),
    pme_alpha=5.0,
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian(
    "electrostatic", "electrostatic",
    partition=CentroidPartition("all", 3.0),
    pme_gridnumber=(40, 40, 40),
    pme_alpha=0.5,
)

# Define QM/MM Hamiltonian
total = qm[:3] + mm[3:] + qmmm

# Build calculator.
calculator = total.build_calculator(system)

# Run simulation.
results = calculator.calculate()
print(results.energy)