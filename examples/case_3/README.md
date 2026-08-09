Example Case 3 (VASP)
=====================

Summary
-------
This case serves as a basic example of using VASP as the QM engine.  The
system under study is a 4-water cluster in a 12.0 Angstrom periodic box,
where the first water is treated at the QM level of theory with VASP at
the PBE level and the remaining three waters are modeled by the SPC/E
forcefield.  The QM/MM/Mechanical algorithm is applied, so VASP sees the
QM atoms alone and QM/MM coupling is carried by the MM forcefield.
Initial velocities are set according to the Maxwell-Boltzmann
distribution at 300 K using a seed.  Rigid water is enforced through the
SETTLE algorithm, which is implemented as an `IntegratorPlugin` and
which excludes the QM subsystem automatically.  The simulation is run
for 2 steps at the 1 fs step-size using a leap-frog Verlet algorithm.
All files needed to run this case are in this case directory.  Running
this case requires a VASP executable and a POTCAR library.  For
electrostatic embedding with VASP, see `../case_4_vasp/`.

How to Run
----------
The script using the Python API of PyDFT-QMMM can be run with the
following command, where `VASP_PP_PATH` points at a POTCAR library
holding per-species subdirectories:

```bash
export VASP_PP_PATH=/path/to/potpaw_PBE.64
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 vasp_std"   # or "srun vasp_std"
python case_3_vasp_api.py
```

What to Expect
--------------
The logging outputs will be printed out to the `./output_api/`
subdirectory, and the VASP input and output files will be written to the
`./vasp_workdir/` subdirectory.  VASP is invoked once per energy and
force evaluation; the `WAVECAR` and `CHGCAR` left behind by one step
seed the next, so the sequence of single points behaves much like a
continued SCF.  The VASP settings here are chosen for a fast example
rather than for production accuracy.
