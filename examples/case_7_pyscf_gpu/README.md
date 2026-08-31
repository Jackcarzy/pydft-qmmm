Example Case 7 (PySCF on a GPU)
===============================

Summary
-------
This case serves as an example of running the QM region on a GPU
through [GPU4PySCF](https://github.com/pyscf/gpu4pyscf).  It applies
the QM/MM/PME coupling of example case 5 to the larger system of
example case 1: a chloromethane-chloride complex in 2000 TIP3P waters,
6006 atoms in a 39.3 Angstrom box, with the six-atom MCL complex as the
QM region at PBE0/6-31G.  Waters whose centroid lies within 8.0
Angstroms form subsystem II and enter the QM Hamiltonian as point
charges; everything beyond reaches it through the reciprocal sum.

The QM region is what makes the device worth choosing.  The integral
and exchange-correlation work a GPU accelerates grows with the QM
region, not with the MM environment, so a six-atom complex has more for
it to do than the single water of case 5.  Selecting the device is a
single option on the QM Hamiltonian:

```python
qm = QMHamiltonian(interface="pyscf", ..., device="gpu")
```

The script runs the same single point on both devices and reports the
difference, so it checks itself rather than asking to be trusted.  The
embedding operators are built on the host in both cases and moved to
the device for the SCF, so the two agree to SCF precision rather than
differing in physics.

All files needed to run this case are in this case directory; unlike
example case 1, no Plumed or ParmEd is needed.  Running it additionally
requires a GPU, CuPy, and a built GPU4PySCF.

How to Run
----------
GPU4PySCF needs its CUDA runtime on the library path, so this case has
to be launched from a batch script rather than run directly.
`submit.slurm` in this directory is a working template:

```bash
sbatch submit.slurm
```

Every path in it is site-specific and must be edited: the CUDA and
compiler modules, the MKL directory PySCF's `libnp_helper` links
against, the `PYTHONPATH` entry pointing at the built GPU4PySCF source
tree, and the Conda environment.

The environment must hold a CuPy matching the CUDA runtime.  Note that
`pyscf-dispersion` installs a real `pyscf/` directory into
site-packages and will shadow an editable PySCF installation, so it
should not be installed alongside one.

What to Expect
--------------
The total energy, its breakdown by calculator, and the timing are
printed for each device, followed by the difference between them and
the forces on the QM atoms.  On a Tesla V100:

```
=== device: cpu
total energy      -2598680.493960 kJ/mol   (18.6 s)
=== device: gpu
total energy      -2598680.493960 kJ/mol   (12.0 s)
  PySCF           -2529617.729254
  OpenMM            -77644.982256
  PMENuclear          8582.217549

subsystem I        6 atoms
subsystem II     216 atoms
subsystem III   5784 atoms

energy difference         0.000000 kJ/mol
max force difference      0.000250 kJ/mol/A
```

`QMMMHamiltonian` defaults to `coupling_mode="conservative"`, so the
reported forces are the gradient of the reported energy and no
`PMEExcluded` component appears.

The two devices should agree to a small fraction of a kJ/mol on the
energy and to well under a tenth of a kJ/mol/Angstrom on the forces.
They will not agree bit for bit: GPU4PySCF builds the same integrals
with different algorithms, so the residual is set by the SCF
convergence rather than by floating-point determinism.

The GPU run takes about two thirds the wall time of the CPU run, for
an energy identical to the printed precision.  The margin is
modest because a six-atom QM region in a split-valence basis is still
small, and because the embedding quadrature and the PME operators are
built on the host for both devices; only the SCF itself moves.  A
larger QM region or a larger basis widens it.

For contrast, example case 5 puts a single water in the QM region, and
there the GPU is the slower of the two: the device cannot pay for its
own setup on 24 basis functions.
