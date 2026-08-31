"""QM/MM/PME single point with PySCF.

One water in the SPC/E box of example case 0 is treated by PySCF at
PBE/def2-SVP; the rest are MM.  Waters within 8 Angstroms form
subsystem II and enter the QM Hamiltonian as point charges, and
everything beyond reaches the QM electrons through the particle-mesh
Ewald reciprocal sum.

PySCF samples that reciprocal potential on the solver's own molecular
quadrature grid and contracts it into a one-electron operator, and the
forces it returns are analytic on both sides: the QM atoms feel the
field, and the MM sites feel the electron density back.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition

# Load system first.
system = System.load("spce.pdb")

# Define QM Hamiltonian.  `grid_level` sets the quadrature used for both
# the exchange-correlation and the embedding integrals.
qm = QMHamiltonian(
    interface="pyscf",
    basis="def2-svp",
    functional="PBE",
    charge=0,
    multiplicity=1,
    conv_tol=1e-10,
    grid_level=5,
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=14.0,
    pme_gridnumber=30,
    pme_alpha=5.0,
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian(
    "electrostatic",
    "electrostatic",
    partition=CentroidPartition("all", 8.0),
)

# Define QM/MM Hamiltonian
total = qm[:3] + mm[3:] + qmmm

# Build calculator.
calculator = total.build_calculator(system)

# Run a single point.
results = calculator.calculate()

print(f"total energy   {results.energy:18.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<12s} {energy:18.6f}")
print()
print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()
print("forces on the QM atoms (kJ/mol/A):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = results.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
