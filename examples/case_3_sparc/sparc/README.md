Example Case 3 (SPARC)
======================

Summary
-------
A small QM/MM mechanical-embedding demo using SPARC for the QM region.
The system is a 4-water cluster in a 12 Å periodic box.  The first
water (atoms 0-2) is treated by SPARC at the PBE level via SPARC-X-API;
the remaining three waters (atoms 3-11) are MM under SPC/E.  Coupling
uses ``QMMMHamiltonian("mechanical", "mechanical")`` — SPARC sees only
the QM atoms; QM/MM coupling is the MM force field acting between
QM and MM atoms.

Note: this example uses mechanical embedding for simplicity, but the
SPARC interface also supports electrostatic embedding (point-charge
embedding and QM/MM/PME) when built against the ``qmmm-embedding``
branch of the SPARC fork.  See ``../sparc_electrostatic/`` for that
example, and that fork's ``QMMM.md`` for background.

Requirements
------------
- A working ``sparc`` binary on PATH (and ``mpirun`` if SPARC was built
  with MPI).
- SPARC pseudopotentials available — typically auto-discovered via
  the ``SPARC_PSP_PATH`` environment variable.
- The ``sparc-x-api`` Python package (`pip install sparc-x-api`).

How to Run
----------
```bash
python case_3_sparc_api.py
```

What to Expect
--------------
PyDFT-QMMM logs are written to ``./output_api/`` and SPARC working
files to ``./sparc_workdir/``.  At each step the SPARC binary is
invoked once; energies and forces are returned and combined with the
SPCE MM force field by the standard PyDFT-QMMM machinery.

The SPARC settings in the script (``h=0.25``, ``tol_scf=1e-5``, Γ-only)
are chosen for a fast demo, not for production accuracy.  Tighten
``h``, ``tol_scf``, and add k-points as needed.
