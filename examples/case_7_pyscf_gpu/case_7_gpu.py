"""QM/MM/PME single point with PySCF running on a GPU.

The QM/MM/PME coupling of example case 5 applied to the larger system
of example case 1: a chloromethane-chloride complex in 2000 TIP3P
waters, 6006 atoms in a 39.3 Angstrom box.  The six-atom QM region is
what makes the device worth choosing, since the integral and
exchange-correlation work a GPU accelerates grows with the QM region
rather than with the MM environment.

The QM solver is moved onto GPU4PySCF by a single option.  The
calculation is run on both devices so the example checks itself: the
embedding operators are built on the host either way and moved to the
device for the SCF, so the two should agree to SCF precision rather
than differ in physics.
"""
from __future__ import annotations

import time

import numpy as np

from pydft_qmmm import *
from pydft_qmmm.plugins import CentroidPartition


def single_point(device):
    """Run the same QM/MM/PME single point on one device."""
    # Reload so each device starts from the same untouched system.
    system = System.load("cmc.pdb")

    # Define QM Hamiltonian.  `device` is the only difference between
    # the two runs.
    qm = QMHamiltonian(
        interface="pyscf",
        basis="6-31G",
        functional="PBE0",
        charge=-1,
        multiplicity=1,
        conv_tol=1e-10,
        grid_level=5,
        device=device,
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
    # The MCL complex is the first six atoms of the box.
    total = qm[:6] + mm[6:] + qmmm

    # Build calculator.
    calculator = total.build_calculator(system)

    # Run a single point.
    start = time.time()
    results = calculator.calculate()
    return system, results, time.time() - start


results = {}
for device in ("cpu", "gpu"):
    system, result, elapsed = single_point(device)
    results[device] = result
    print(f"=== device: {device}")
    print(f"total energy   {result.energy:18.6f} kJ/mol   ({elapsed:.1f} s)")
    for name, energy in result.components.items():
        if not name.startswith("."):
            print(f"  {name:<12s} {energy:18.6f}")
    print()

print(f"subsystem I   {len(system.select('subsystem I')):6d} atoms")
print(f"subsystem II  {len(system.select('subsystem II')):6d} atoms")
print(f"subsystem III {len(system.select('subsystem III')):6d} atoms")
print()

cpu, gpu = results["cpu"], results["gpu"]
print(f"energy difference   {gpu.energy - cpu.energy:14.6f} kJ/mol")
print(
    "max force difference"
    f" {np.abs(gpu.forces - cpu.forces).max():14.6f} kJ/mol/A",
)
print()
print("forces on the QM atoms (kJ/mol/A), on the GPU:")
for atom in sorted(system.select("subsystem I")):
    fx, fy, fz = gpu.forces[atom]
    print(f"  atom {atom} {fx:14.6f} {fy:14.6f} {fz:14.6f}")
