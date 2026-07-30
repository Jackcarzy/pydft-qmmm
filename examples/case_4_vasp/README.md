Example Case 4 (VASP)
=====================

Summary
-------
A small QM/MM mechanical-embedding demo using VASP for the QM region.
The system is a 4-water cluster in a 12 A periodic box.  The first
water (atoms 0-2) is treated by VASP at the PBE level; the remaining
three waters (atoms 3-11) are MM under SPC/E.  Coupling uses
``QMMMHamiltonian("mechanical", "mechanical")`` -- VASP sees only the QM
atoms; QM/MM coupling is the MM force field acting between QM and MM
atoms.

This mirrors ``case_3_sparc`` and ``case_3_psi4`` (in the sibling
pydft-qmmm checkout) on the same 4-water system, so the three QM engines
can be compared directly.

Note: the VASP interface currently supports mechanical embedding only.
Electrostatic embedding (adding the MM electrostatic potential to the
Kohn-Sham Hamiltonian via ``V_ext``) requires a VASP binary compiled
with ``-DPLUGINS`` and is not yet wired up; requesting an electrostatic
scheme raises ``NotImplementedError``.

How the interface works
-----------------------
PyDFT-QMMM drives the dynamics.  Each energy/force evaluation:

1. writes ``POSCAR`` (Cartesian, species-grouped), ``POTCAR`` (species
   concatenated from the library), ``KPOINTS`` (Gamma mesh), and
   ``INCAR`` (single point, ``NSW=0``, ``IBRION=-1``) into the working
   directory;
2. launches VASP as a subprocess;
3. parses ``vasprun.xml`` for the free energy and forces, converts
   eV -> kJ/mol and eV/A -> kJ/mol/A, and un-scrambles the forces back
   into system order.

``WAVECAR`` and ``CHGCAR`` from one step seed the next (``ISTART=1``),
so the sequence of single points behaves like a continued SCF.

Requirements
------------
- A VASP executable reachable by the ``command`` keyword (default
  ``vasp_std``, overridable via ``PYDFT_QMMM_VASP_COMMAND`` or
  ``ASE_VASP_COMMAND``).
- A POTCAR library given by ``pp_path`` or ``VASP_PP_PATH``, holding
  per-species subdirectories, e.g.
  ``.../pseudopotential/potpaw_PBE.54``.  Use ``potcar_map`` to select
  non-default potentials, e.g. ``potcar_map={"Li": "Li_sv"}``.

How to Run
----------
```bash
export VASP_PP_PATH=/path/to/potpaw_PBE.54
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 vasp_std"   # or "srun vasp_std"
python case_4_vasp_api.py
```

On a Slurm GPU cluster, run it from a batch script that loads the VASP
module first; see ``../../../test/submit_interface.slurm`` in this
repository for a single-point template.

What to Expect
--------------
PyDFT-QMMM logs are written to ``./output_api/`` and VASP working files
to ``./vasp_workdir/``.  At each step VASP is invoked once; energies and
forces are combined with the SPC/E MM force field by the standard
PyDFT-QMMM machinery.

The VASP settings here (``ENCUT=500``, Gamma-only, ``EDIFF=1e-6``) are
chosen for a fast demo, not for production accuracy.  Tighten the cutoff
and k-point mesh as needed for real systems.
