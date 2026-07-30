#!/usr/bin/bash
#SBATCH -J case4_vasp
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH -t 1:00:00
#SBATCH --mem-per-gpu=12G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -e

module purge
module load nvhpc/25.5
module load openmpi/4.1.8-cuda
module load vasp/6.5.1

export PYTHONPATH=/storage/project/r-jmcdaniel43-0/cshao48/install/pydft-vasp-qmmm/pydft-qmmm
export VASP_PP_PATH=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/potpaw_PBE.64
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 vasp_std"

PY=/storage/home/hcoda1/1/cshao48/.conda/envs/sparc/bin/python

cd /storage/project/r-jmcdaniel43-0/cshao48/install/pydft-vasp-qmmm/pydft-qmmm/examples/case_4_vasp
"$PY" case_4_vasp_api.py
echo "=== case_4 dynamics completed ==="
tail -n +1 output_api/*.csv 2>/dev/null || true
