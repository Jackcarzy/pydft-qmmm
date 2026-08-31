"""QM/MM/PME single point with PySCF running on a GPU."""
from __future__ import annotations

import time

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition


def single_point():
    """Run a QM/MM/PME single point on the GPU."""
    system = System.load("cmc.pdb")

    # Define QM Hamiltonian.
    qm = QMHamiltonian(
        interface="pyscf",
        basis="6-31G",
        functional="PBE0",
        charge=-1,
        multiplicity=1,
        conv_tol=1e-10,
        grid_level=5,
        device="gpu",
    )

    # Define MM Hamiltonian.
    mm = MMHamiltonian(
        forcefield=["tip3p_cmc_no_intra.xml", "tip3p_cmc_residues.xml"],
        nonbonded_method="PME",
        nonbonded_cutoff=14.0,
        # Two grid points per Angstrom of box edge; a coarser
        # reciprocal mesh than that carries real numerical error.
        pme_gridnumber=80,
        pme_alpha=5.0,
    )

    # Define IXN Hamiltonian.
    qmmm = QMMMHamiltonian(
        "electrostatic",
        "electrostatic",
        partition=CentroidPartition("all", 8.0),
    )

    # Define QM/MM Hamiltonian
    total = qm[:6] + mm[6:] + qmmm

    # Build calculator.
    calculator = total.build_calculator(system)

    # Run a single point.
    start = time.time()
    results = calculator.calculate()
    return system, results, time.time() - start


system, result, elapsed = single_point()
print("=== device: gpu")
print(f"total energy   {result.energy:18.6f} kJ/mol   ({elapsed:.1f} s)")
for name, energy in result.components.items():
    if not name.startswith("."):
        print(f"  {name:<12s} {energy:18.6f}")
print()

print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()

print("forces on the QM atoms (kJ/mol/A):")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = result.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
