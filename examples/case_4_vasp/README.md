Example Case 4 (VASP)
=====================

Summary
-------
This case serves as an example of electrostatic embedding with VASP as
the QM engine.  The system under study is a QM water molecule in a box
of 122 MM waters modeled by the SPC/E forcefield, in a 15.636 Angstrom
periodic box.  The QM/MM/PME algorithm is applied, where waters whose
centroid lies within 3.0 Angstroms of the QM region reach the QM
electrons through a near field rebuilt from their real positions on
VASP's own FFT grid, and everything beyond reaches it through the
particle-mesh Ewald reciprocal sum.  The QM positions are centered in
the box and the remaining residues wrapped around them beforehand.  A
single energy is computed rather than a trajectory.  All files needed to
run this case are in this case directory.  Running this case requires a
VASP executable compiled with `-DPLUGINS`, a POTCAR library, and a
Python environment holding both `numpy` and `helpme_py` for VASP's
embedded interpreter to import.

How to Run
----------
The script can be run with the following command, where `VASP_PP_PATH`
points at a POTCAR library holding per-species subdirectories and
`PYTHONHOME` points at the environment VASP should import the plugin
from:

```bash
export VASP_PP_PATH=/path/to/potpaw_PBE.64
export PYTHONHOME=/path/to/conda/env
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 vasp_std"   # or "srun vasp_std"
python run.py
```

On a Slurm cluster, run it from a batch script that loads the VASP
module first; `submit.slurm` in this directory is a working template,
though the environment and executable paths in it are site-specific and
must be edited.

What to Expect
--------------
The energy of the system is printed to standard output, and the VASP
input and output files will be written to the `./vasp_workdir/`
subdirectory.  The plugin leaves a `PLUGIN_FIRED.txt` sentinel there
recording the grid it used and the minimum of each half of the external
potential; its absence means the embedding never ran, and the interface
raises rather than returning an unembedded energy.  If a callback fails,
the traceback is written to `PLUGIN_ERROR.txt` in the same directory.

Two settings are worth noting before adapting this case, as both fail
quietly.  The external-potential grid is pinned with
`NGXF/NGYF/NGZF = 108`, giving a 0.145 Angstrom spacing against which
the `embedding_sigma = 0.3` Gaussians representing the MM point charges
are comfortably resolved; a sigma below one grid spacing does not
conserve the charge it deposits.  The coupling Hamiltonian's
`pme_alpha = 0.5` is deliberately not the MM forcefield's
`pme_alpha = 5.0`, because the QM/MM reciprocal sum is interpolated onto
VASP's grid and that is only valid while the PME grid resolves the
1/alpha length scale.  Symmetry must be switched off, and requesting
embedding with `ISYM > 0` raises `ValueError`.  The VASP settings here
are chosen for a fast example rather than for production accuracy.
