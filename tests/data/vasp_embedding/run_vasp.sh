#!/usr/bin/bash
# Launcher handed to the VASP interface as PYDFT_QMMM_VASP_COMMAND.
#
# The plugin environment CANNOT be exported by the Slurm script itself:
# PYTHONHOME must point at the vasp_plugin env (python 3.10, matching the
# libpython3.10 the binary links), while pytest and pydft_qmmm live in the
# openmm env (python 3.13).  Setting PYTHONHOME globally breaks the pytest
# interpreter.  Confining it to this wrapper keeps the two apart.
ENV=/storage/home/hcoda1/1/cshao48/.conda/envs/vasp_plugin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV/lib
export PYTHONHOME=$ENV
export PATH=$ENV/bin:$PATH

VASP=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/vasp.6.6.1_nvhpc24/bin/vasp_std
exec mpirun -np 1 "$VASP"
