"""QM/MM single point with a PERIODIC PySCF wavefunction.

Unlike case 5, the QM region here is not a molecule in a periodic
environment: the wavefunction itself is periodic, and the QM cell is
the simulation box.  This is the model the VASP interface implements,
run in-process with Gaussian basis functions instead of through a
separate binary.

Because the QM cell IS the box, the box size sets the cost of the QM
calculation.  That is why this case ships a 12 Angstrom box rather than
reusing the 29.9 Angstrom box of cases 0 and 5: at a force-converged
cutoff the larger box needs a mesh several hundred points on a side.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition

# Load system first.  The CRYST1 record sets the periodic QM cell.
system = System.load("spce_small.pdb")

# Define QM Hamiltonian.
#
# pseudo is mandatory: the periodic gradient code rejects all-electron
# cells, so an ECP or a bare all-electron basis will not work here.
#
# ke_cutoff must be converged against a FORCE, not an energy.  On a
# water cell the net force -- which translational invariance requires to
# vanish -- is 93 kJ/mol/Angstrom at ke_cutoff=80 with this basis and
# 0.7 at 200, while the energy already looks settled at 80.
qm = QMHamiltonian(
    interface="pyscf_pbc",
    basis="gth-dzvp",
    pseudo="gth-pbe",
    functional="pbe",
    charge=0,
    multiplicity=1,
    ke_cutoff=200.0,
    conv_tol=1e-10,
    # device="gpu",   # needs GPU4PySCF and a CUDA runtime
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["spce.xml", "spce_residues.xml"],
    nonbonded_method="PME",
    nonbonded_cutoff=5.0,
    pme_gridnumber=24,
    pme_alpha=5.0,
)

# Define IXN Hamiltonian.
qmmm = QMMMHamiltonian(
    "electrostatic",
    "electrostatic",
    partition=CentroidPartition("all", 4.0),
)

# Define QM/MM Hamiltonian.  The first water is the QM subsystem.
total = qm[:3] + mm[3:] + qmmm

# Build calculator.
calculator = total.build_calculator(system)

# Run a single point.
results = calculator.calculate()

print(f"total energy   {results.energy:18.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<28s} {energy:18.6f}")
print()
print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()
print("forces on the QM atoms (kJ/mol/A):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = results.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
print()
print(f"net force      {results.forces.sum(axis=0)}")
