Example Case 3 (SPARC, electrostatic embedding)
===============================================

Summary
-------
The same 4-water cluster as `../sparc/`, but with electrostatic rather
than mechanical embedding. SPARC sees the MM point charges as Gaussians
on its real-space grid; the QM electrostatic potential is contracted
back against those Gaussians to give the MM forces.

With long-range `"electrostatic"` coupling, PME uses only region III
charges and no field exclusions, matching VASP's default FFT embedding.
The periodic Gaussian field supplies region II and its images; SPARC
handles region I periodically. Physical force-field charges are unchanged.
The separate `PMEExcluded` component corrects OpenMM double counting in
the energy only; it adds no forces for engine coupling.

Requirements
------------
- A `sparc` binary built from the `qmmm-embedding` branch of the SPARC
  fork. A stock SPARC is detected and rejected.
- SPARC pseudopotentials, via `SPARC_PSP_PATH`.
- `sparc-x-api`. It is not patched; the interface passes an extended
  parameter schema through the public `sparc_json_file` argument.
- `helpme-py`, only if you switch the long-range scheme to
  `"electrostatic"` for full QM/MM/PME.

How to Run
----------
The launcher depends on the machine. A workstation with a normal MPI
install uses `mpirun`/`mpiexec`; a Slurm cluster without either uses
`srun` instead. Set `ASE_SPARC_COMMAND` accordingly:

```bash
# Workstation with mpirun/mpiexec on PATH:
export ASE_SPARC_COMMAND="mpirun -np 8 /path/to/qmmm-embedding/lib/sparc"

# Slurm cluster with no mpirun/mpiexec (e.g. PACE):
export ASE_SPARC_COMMAND="srun --account=<acct> --partition=<part> -n 8 /path/to/qmmm-embedding/lib/sparc"

python case_3_sparc_electrostatic.py
```

The working directory (`directory="./sparc_workdir"` in the script)
must be on shared storage, not node-local `/tmp`. Under `srun`, ranks
land on compute nodes whose `/tmp` does not contain a login node's
directory, and every rank fails to `chdir` into it. Run this example
from a path on shared storage (e.g. your project or home directory).

What to Expect
--------------
PyDFT-QMMM logs land in `./output_api/` and SPARC working files in
`./sparc_workdir/`, which will also hold `QMMM_VEXT.bin` and
`QMMM_PHI.bin` — the external potential written in and the electrostatic
potential read back.

Comparing to the neighbours
---------------------------
`../sparc/` runs the identical system under mechanical embedding, and
`../psi4/` gives a Psi4 reference. The QM forces should move between the
mechanical and electrostatic cases; the MM forces on atoms 3-11 pick up
a contribution that is identically zero under mechanical embedding.

The settings here favour a fast demo over accuracy. Tighten `fd_grid`
and `tol_scf` for production.

A note on `fd_grid` and force accuracy
---------------------------------------
Embedded runs need a finer `FD_GRID` than energy convergence alone
suggests. Because embedding must retain the genuine net force on the QM
region (the MM to QM momentum transfer), it also retains SPARC's
egg-box discretization artifact, which `Symmetrize_forces` normally
hides on unembedded runs. The magnitude is geometry-dependent, not a
fixed property of the grid spacing, so treat these as measured examples
rather than a budgeting rule:

| System | FD_GRID | spurious net force |
|---|---|---|
| One water, 12 A cell | 48^3 | -5.05e-04 Ha/Bohr (about 2.5 kJ/mol/Angstrom) |
| One water, 12 A cell | 72^3 | -3.18e-05 Ha/Bohr (about 0.15 kJ/mol/Angstrom) |
| Four waters, 12 A cell | 48^3 | about 190 kJ/mol/Angstrom |

On the *same* 48^3 mesh, the four-water system's spurious force is
roughly 75x larger than the single water's — egg-box error depends
strongly on where atoms sit relative to grid points. Do not infer a
convergence rate (e.g. h^7) from the two single-water rows; it does not
hold across geometries. `fd_grid=(60, 60, 60)` in this script is a
fast-demo compromise, not a production setting. Measure this directly
for your own system before trusting embedded forces for dynamics — run
it unembedded at the intended `FD_GRID` and look at the net force —
rather than assume a number from this table applies.

Grid convergence
----------------
The embedded water gradient benchmark used a target spacing of 0.12 Bohr
(247³ for a 15.6358 Å cell), SCF tolerance 1e-8, and central displacements
of 0.0025 Å. Total gradient RMSE was 0.781 kJ/(mol Å); the SPARC component
RMSE was 0.639 kJ/(mol Å). These settings improve on 0.15 Bohr but are
not a universal convergence threshold. Isolated-water errors did not
decrease monotonically between 0.12 and 0.10 Bohr. Check forces as well
as energies when choosing a grid.
