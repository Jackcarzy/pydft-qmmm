Example Case 5 (PySCF)
======================

Summary
-------
This case serves as an example of using PySCF as the QM engine.  The
system under study is a QM water molecule in a box of MM water modeled
by the SPC/E forcefield, the same box as example case 0.  The
QM/MM/PME algorithm is applied, where waters whose centroid lies within
8.0 Angstroms of the QM region form subsystem II and enter the QM
Hamiltonian as point charges, and everything beyond reaches the QM
electrons through the particle-mesh Ewald reciprocal sum.  A single
energy and set of forces is computed rather than a trajectory.  All
files needed to run this case are in this case directory.

The supported model is a *molecular* QM region embedded in a
periodically replicated, fixed-charge MM environment.  This is
not a `pyscf.pbc` periodic wavefunction, and there are no k-points;
periodicity enters only through the electrostatic environment that
OpenMM and the PME machinery provide.

Running this case requires PySCF, which is installed with the `pyscf`
extra:

```bash
python -m pip install 'pydft_qmmm[pyscf,qmmm-pme]'
```

How to Run
----------
The script using the Python API of PyDFT-QMMM can be run with the
following command:

```bash
python case_5_pme.py
```

What to Expect
--------------
The total energy, its breakdown by calculator, the size of each
subsystem, and the forces on the QM atoms are printed to standard
output.  On this fixture:

```
total energy       -242232.359609 kJ/mol
  PySCF            -200875.929248
  OpenMM            -41880.742893
  PMENuclear           524.312532

subsystem I        3 atoms
subsystem II     189 atoms
subsystem III   2493 atoms
```

The `PySCF` component carries the QM energy including both embedding
terms, and `PMENuclear` is the reciprocal term PyDFT-QMMM retains for
the QM nuclei.


`QMMMHamiltonian` defaults to `coupling_mode="conservative"`, which
removes the QM atoms' force-field charges from the OpenMM force objects
themselves, so the reported forces are the gradient of the reported
energy.  Passing `coupling_mode="force"` instead reproduces the older
assembly, in which those charges stayed in OpenMM's reciprocal sum and
were corrected afterwards by a separate `PMEExcluded` term; that mode
warns, and its total energy differs (here by 0.18 kJ/mol) because the
two removals are not algebraically equivalent.

The reciprocal potential is sampled on the solver's own molecular
quadrature grid rather than on the PME mesh, which at half-Angstrom
spacing is far too coarse to integrate an all-electron density, and the
forces are analytic on both sides: the QM atoms feel the field, and the
MM sites feel the electron density in return.

`grid_level` sets that quadrature.  Because the grid moves with the QM
atoms and the terms that generates are omitted, as PySCF omits them
from the exchange-correlation gradient by default, the level also sets
how closely the analytic QM force tracks the energy.  Level 3 leaves
roughly 0.07 kJ/mol/Angstrom on a single embedded water and level 5
roughly 0.003, which is why this case raises it.
