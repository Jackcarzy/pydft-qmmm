#!/usr/bin/bash
#SBATCH -J vasp_embed_test
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH -t 2:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Tier 0 only.  The gradient test needs seven VASP launches and gets its
# own job (gradient.slurm) once this one reports the per-run cost.

# VASP 6.6.1 + Python plugins, built with nvhpc 24.1, plugin ACTIVE.
# No hdf5 module -- this build has -DVASP_HDF5 off.
module purge
module load nvhpc-openmpi3/24.1
module load intel-oneapi-mkl/2023.1.0

# NOTE: PYTHONHOME / PATH / LD_LIBRARY_PATH for the plugin environment are
# deliberately NOT exported here.  They live in run_vasp.sh instead, because
# PYTHONHOME must point at vasp_plugin (python 3.10, matching the linked
# libpython3.10) while pytest and pydft_qmmm live in the openmm env (python
# 3.13).  Exporting it globally would break the interpreter running pytest.

# 8 threads is optimal for nvhpc 24.1 builds (76.6 s at 8 vs 84.4 s at 1).
export OMP_NUM_THREADS=8
export OMP_STACKSIZE=512m
ulimit -s unlimited

REPO=/storage/project/r-jmcdaniel43-0/cshao48/install/pydft-vasp-qmmm/pydft-qmmm
cd "$REPO"

export PYDFT_QMMM_VASP_COMMAND="$REPO/tests/data/vasp_embedding/run_vasp.sh"
export VASP_PP_PATH=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/potpaw_PBE.64

# Keep run directories after the job ends.  pytest's tmp_path lives in
# node-local /tmp, which evaporates with the allocation and takes the
# OUTCAR with it -- that cost us the diagnosis on job 11566747.
export PYDFT_QMMM_VASP_TESTDIR=/storage/project/r-jmcdaniel43-0/cshao48/install/pydft-vasp-qmmm/vasp_test_runs/${SLURM_JOB_ID}
mkdir -p "$PYDFT_QMMM_VASP_TESTDIR"

PYTEST=/storage/home/hcoda1/1/cshao48/.conda/envs/openmm/bin/python

echo "=== environment ==="
echo "  VASP command : $PYDFT_QMMM_VASP_COMMAND"
echo "  POTCAR lib   : $VASP_PP_PATH"
echo "  pytest python: $PYTEST"
echo

echo "=== wiring + Tier 0 smoke ==="
date
"$PYTEST" -m pytest tests/vasp_embedding_test.py -v \
    -k "Wiring or Factory or Smoke" -s
status=$?
date
echo "=== exit: $status ==="

echo
echo "=== per-run cost, for sizing the gradient job ==="
for d in "$TMPDIR"/*/vasp*/OUTCAR /tmp/pytest-of-$USER/pytest-*/*/vasp*/OUTCAR; do
    [ -f "$d" ] || continue
    printf "  %-70s %s\n" "$d" \
        "$(grep 'Elapsed time' "$d" 2>/dev/null | awk '{print $4" s"}')"
done 2>/dev/null | head -10
