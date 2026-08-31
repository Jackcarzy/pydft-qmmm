#!/usr/bin/bash
#SBATCH -J pyscf_gpu
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1 -n 8
#SBATCH -t 01:00:00
#SBATCH -q inferno
#SBATCH -p gpu-v100 --gres=gpu:V100:1
#SBATCH --mem-per-cpu=8G
#SBATCH -o case_7.log

# GPU4PySCF needs its CUDA runtime on the library path, so this case
# has to be launched from a batch script rather than run directly.
# Every path below is site-specific and must be edited.

module purge
module load gcc/12.3.0
module load cuda/12.6.1

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUPY_ACCELERATORS=cub,cutensor

# PySCF's libnp_helper links MKL, and `module load` does not populate
# MKLROOT in a batch shell, so point at it directly.
MKL=/usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/intel-oneapi-mkl-2023.1.0-2dpwqvqsqapwjrafwzbfy2yokjn33f3v/mkl/latest/lib/intel64
export LD_LIBRARY_PATH=$MKL:$LD_LIBRARY_PATH

# GPU4PySCF is used from its built source tree rather than installed.
export PYTHONPATH=/storage/home/hcoda1/1/cshao48/r-jmcdaniel43-0/pyscf_install/ksced_test/gpu4pyscf:$PYTHONPATH

ENV=/storage/home/hcoda1/1/cshao48/.conda/envs/pyscf_qmmm
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR}"

nvidia-smi -L
"$ENV/bin/python" case_7_gpu.py
