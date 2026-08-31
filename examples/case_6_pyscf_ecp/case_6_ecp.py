"""QM/MM/PME single point on a heavy atom with an effective core potential.

One iodide ion in a box of SPC/E water.  Iodine's def2 basis sets are
parameterized against an effective core potential, and PySCF keeps the
orbital basis and the ECP independent of one another: asking for
def2-SVP alone would put all 54 of iodide's electrons into a
26-function valence basis and converge anyway.  The interface refuses
that combination rather than returning the number, so the ECP has to be
requested explicitly through `ecp`.

The ECP also decides what the reciprocal sum sees.  The wavefunction
carries 26 electrons against a core charge of 25 rather than 53, a net
-1, and the coupling Hamiltonian asks the interface for that effective
charge instead of assuming the atomic number.

Run `python build_iodide_pdb.py` first if `iodide_spce.pdb` is absent.
"""
from __future__ import annotations

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition

# Load system first.
system = System.load("iodide_spce.pdb")

# Define QM Hamiltonian.  Pairing `basis` with `ecp` is what Psi4 does
# for itself when it loads a basis that defines one.
qm = QMHamiltonian(
    interface="pyscf",
    basis="def2-svp",
    ecp="def2-svp",
    functional="PBE",
    charge=-1,
    multiplicity=1,
    conv_tol=1e-10,
    grid_level=5,
)

# Define MM Hamiltonian.
mm = MMHamiltonian(
    forcefield=["iodide_spce.xml", "iodide_residues.xml"],
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

# Define QM/MM Hamiltonian.  The QM region is the single ion.
total = qm[:1] + mm[1:] + qmmm

# Build calculator.
calculator = total.build_calculator(system)

# Run a single point.
results = calculator.calculate()

qm_potential = [
    calc.potential for calc in calculator.calculators
    if type(calc.potential).__name__ == "PySCFPotential"
][0]
mol = qm_potential.method[0].mol

print(f"electrons treated explicitly {mol.nelectron:6d}  (54 without the ECP)")
print(f"effective nuclear charge     {qm_potential.nuclear_charges()[0]:6.1f}"
      "  (53 without the ECP)")
print()
print(f"total energy   {results.energy:18.6f} kJ/mol")
for name, energy in results.components.items():
    if not name.startswith("."):
        print(f"  {name:<12s} {energy:18.6f}")
print()
fx, fy, fz = results.forces[0]
print(f"force on the ion {fx:14.6f} {fy:14.6f} {fz:14.6f} kJ/mol/A")
