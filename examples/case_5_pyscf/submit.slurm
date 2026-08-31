#!/usr/bin/bash
#SBATCH -J pyscf_qmmm
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00
#SBATCH --mem=32G

# A login node is usually capped well below what an SCF over a few
# thousand MM sites needs, so run this case from a compute node.  The
# environment path below is site-specific and must be edited.
ENV=/storage/home/hcoda1/1/cshao48/.conda/envs/pyscf_qmmm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PATH="$ENV/bin:$PATH"

cd "${SLURM_SUBMIT_DIR}"

python case_5_pme.py
