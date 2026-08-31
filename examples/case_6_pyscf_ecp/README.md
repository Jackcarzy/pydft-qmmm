Example Case 6 (PySCF, effective core potential)
================================================

Summary
-------
This case serves as an example of a heavy-atom QM region, where the
basis set is parameterized against an effective core potential.  The
system under study is a single iodide ion in a box of MM water modeled
by the SPC/E forcefield, built by replacing the first water of example
case 5's box with an ion at its oxygen position.  The QM/MM/PME
algorithm is applied, where waters whose centroid lies within 8.0
Angstroms of the ion form subsystem II and enter the QM Hamiltonian as
point charges, and everything beyond reaches it through the
particle-mesh Ewald reciprocal sum.  A single energy and force is
computed rather than a trajectory.  All files needed to run this case
are in this case directory.

Two settings are worth noting before adapting this case.

PySCF keeps the orbital basis and the effective core potential
independent of one another, unlike Psi4, which applies the ECP
automatically when it loads a basis that defines one.  Asking for
`basis="def2-svp"` alone would place all 54 of iodide's electrons into
a 26-function valence basis, and the SCF would converge to a
meaningless energy rather than fail.  The interface refuses that
combination, so the potential is requested explicitly with
`ecp="def2-svp"`.

The ECP also decides what the reciprocal sum sees.  The wavefunction
carries 26 electrons against a core charge of 25 rather than 53, for a
net charge of -1 as iodide should have.  Taking the atomic number
instead would put +53 against those 26 electrons and enter the
reciprocal sum as +27, so the coupling Hamiltonian asks the QM
interface for the effective charge rather than assuming it.

How to Run
----------
The system is shipped ready to use.  To rebuild `iodide_spce.pdb` from
`spce.pdb`, run:

```bash
python build_iodide_pdb.py
```

The case is then run with:

```bash
python case_6_ecp.py
```

A login node is usually capped well below what an SCF over a few
thousand MM sites needs, so on a Slurm cluster run this case from a
batch script; `submit.slurm` in this directory is a working template,
though the environment path in it is site-specific and must be edited.

What to Expect
--------------
The script prints the number of electrons the solver treats explicitly
and the effective nuclear charge, alongside the values that would apply
without the ECP, then the total energy, its breakdown by calculator,
and the force on the ion.  The electron count should be 26 rather than
54 and the nuclear charge 25 rather than 53.  On this fixture:

```
electrons treated explicitly     26  (54 without the ECP)
effective nuclear charge       25.0  (53 without the ECP)

total energy       -822812.800278 kJ/mol
  PySCF            -786309.315844
  OpenMM            -40826.713919
  PMENuclear          4323.229484

force on the ion   -1643.331762   -1563.805733   -1084.317816 kJ/mol/A
```

The `PMENuclear` component is the term the effective nuclear charge
scales.  Computing it from the atomic number instead inflates it by the
ratio 53/25, an error of about 4800 kJ/mol here, so it is a useful
quantity to watch when adapting this case to another heavy element.

The last few digits are not reproducible between runs: OpenMM's
threaded CPU summation order varies, which moves the total by around
1e-4 kJ/mol.  Everything above that is deterministic.

`QMMMHamiltonian` defaults to `coupling_mode="conservative"`, so the
reported forces are the gradient of the reported energy.  A charged QM
region is where that choice shows: under the older
`coupling_mode="force"` assembly this total is 64.9 kJ/mol higher
(-822747.900941) and carries an extra `PMEExcluded` component, because
leaving a net -1 charge in OpenMM's reciprocal sum and correcting it
afterwards is not equivalent to removing it.  The neutral QM water of
example case 5 shifts by only 0.18 kJ/mol under the same change.
