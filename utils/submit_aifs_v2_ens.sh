#!/bin/bash -l
#SBATCH -J aifs_v2_ens
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=150G
#SBATCH -t 12:00:00
#SBATCH --chdir=/net/scratch2/anustupb/aifs_v2_ens_operational
#SBATCH -o logs/slurm-%x-%j.out

# GPU part of the AIFS ENS v2 pipeline: runs utils/run_aifs_ens.py for one cycle.
# Download, preprocessing and postprocessing run on the CPU in run_aifs_pipeline.sh.
#
# Lives in utils/. Normally submitted by run_aifs_pipeline.sh. To submit by hand
# (from the repo root):
#   sbatch -J aifs_v2_ens_20260923T00 --export=ALL,DATE=20260923T00 utils/submit_aifs_v2_ens.sh
#
# The job's exit code is run_aifs_ens.py's exit code:
#   0 ok, 1 failed, 2 bad arguments, 3 input state missing.
# Details are in logs/AIFS.log; this job's .out file shows only the checks below.
#
# REPO_ROOT is hardcoded on purpose: sbatch runs a spool copy of this script, so the
# script's own location cannot be used to find the repo.

set -uo pipefail

REPO_ROOT="/net/scratch2/anustupb/aifs_v2_ens_operational"
CONDA_ENV="/net/scratch2/marchakitus/conda-envs/AIFS_ENSv2"

# DATE: YYYYMMDDT00 (as passed by the wrapper) or YYYYMMDD.
DATE="${DATE:-}"
DATE="${DATE%T00}"
if [[ ! "$DATE" =~ ^[0-9]{8}$ ]]; then
    echo "ERROR: pass DATE as YYYYMMDDT00, e.g. sbatch --export=ALL,DATE=20260923T00 utils/submit_aifs_v2_ens.sh"
    exit 2
fi
CYCLE="${DATE}T00"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV" || { echo "ERROR: could not activate $CONDA_ENV"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "$REPO_ROOT" || exit 1

echo "=== job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-}) on $(hostname), cycle $CYCLE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

IC_FILE="IC_data/${DATE}/input_state_${CYCLE}_v2.pkl"
if [[ ! -f "$IC_FILE" ]]; then
    echo "ERROR: input state $IC_FILE not found (run utils/preprocess_ic.py --date $CYCLE)"
    exit 3
fi

start=$(date +%s)
echo "=== $(date -u '+%F %T') UTC  inference: starting"
python utils/run_aifs_ens.py --date "$CYCLE"
rc=$?
echo "=== $(date -u '+%F %T') UTC  inference: exit $rc after $(( $(date +%s) - start )) s"
exit $rc