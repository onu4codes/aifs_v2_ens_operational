#!/bin/bash -l
# AIFS ENS v2 operational pipeline for one 00 UTC cycle. Runs on the CPU / login node.
#
#   1. utils/download_ic.py     (CPU)  -> prints the cycle it downloaded, e.g. 20260923T00
#   2. utils/preprocess_ic.py   (CPU)
#   3. utils/submit_aifs_v2_ens.sh (GPU, via sbatch --wait: the wrapper waits for the job)
#   4. utils/postprocess.py     (CPU, only after the GPU job succeeded)
#
# Usage:
#   ./run_aifs_pipeline.sh                    # latest available cycle
#   ./run_aifs_pipeline.sh --date 20260923    # a specific date (00 UTC)
#
# The wrapper waits for the GPU job, which can take hours. Run it from cron, or with
#   nohup ./run_aifs_pipeline.sh --date 20260923 >/dev/null 2>&1 &
# so it is not killed when you log out.
#
# Nothing is printed to the terminal; all messages go to logs/AIFS.log.
# Exit codes:
#   0  done (or nothing to do)       1  a step failed
#   2  bad arguments                 3  input data not available
#   4  another pipeline run is already in progress
# Codes 1-3 are passed through from the step that failed.

set -uo pipefail

REPO_ROOT="/net/scratch2/anustupb/aifs_v2_ens_operational"
CONDA_ENV="/net/scratch2/marchakitus/conda-envs/AIFS_ENSv2"
JOB_SCRIPT="$REPO_ROOT/utils/submit_aifs_v2_ens.sh"
JOB_PREFIX="aifs_v2_ens"
POLL_SECONDS=60                          # how often to check an already-running GPU job

LOG_FILE="$REPO_ROOT/logs/AIFS.log"
LOCK_FILE="$REPO_ROOT/logs/.run_aifs_pipeline.lock"
mkdir -p "$REPO_ROOT/logs"

log() {   # log LEVEL message...   (same layout as the Python scripts)
    local level="$1"; shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') - run_aifs_pipeline - $level - $*" >> "$LOG_FILE"
}

# ── arguments ────────────────────────────────────────────────────────────────
DATE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)
            DATE="${2:-}"; shift 2 || { log ERROR "Invalid arguments: --date needs a value"; exit 2; } ;;
        --date=*)
            DATE="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)
            log ERROR "Invalid arguments: unknown option '$1'"; exit 2 ;;
    esac
done
if [[ -n "$DATE" ]]; then
    if ! [[ "$DATE" =~ ^[0-9]{8}$ ]] || ! date -d "$DATE" >/dev/null 2>&1; then
        log ERROR "Invalid arguments: --date must be a valid YYYYMMDD date, got '$DATE'"
        exit 2
    fi
fi

# From here on, anything a command prints goes to the log, not the terminal.
exec >> "$LOG_FILE" 2>&1

# ── one pipeline at a time ───────────────────────────────────────────────────
exec 9> "$LOCK_FILE"
if ! flock -n 9; then
    log WARNING "Another run_aifs_pipeline.sh is still running (lock $LOCK_FILE). Exiting."
    exit 4
fi

# ── environment ──────────────────────────────────────────────────────────────
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || { log ERROR "Could not activate $CONDA_ENV"; exit 1; }
else
    export PATH="$CONDA_ENV/bin:$PATH"       # e.g. under cron without conda init
fi
cd "$REPO_ROOT" || { log ERROR "Cannot cd to $REPO_ROOT"; exit 1; }

# Check everything the pipeline needs before starting any step.
for f in utils/download_ic.py utils/preprocess_ic.py utils/postprocess.py "$JOB_SCRIPT"; do
    if [[ ! -r "$f" ]]; then
        log ERROR "Required file not found or not readable: $f. Pipeline not started."
        exit 1
    fi
done

log INFO "=== Pipeline started (date: ${DATE:-latest}, host $(hostname), pid $$)"

run_step() {   # run_step NAME command...   -> stops the pipeline on failure
    local name="$1"; shift
    local start rc
    start=$(date +%s)
    log INFO "Step '$name' started"
    "$@"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        log ERROR "Step '$name' failed with exit $rc after $(( $(date +%s) - start )) s. Pipeline stopped."
        exit "$rc"
    fi
    log INFO "Step '$name' finished after $(( $(date +%s) - start )) s"
}

finish() {
    log INFO "=== Pipeline complete for $1: Outputs/postprocessed/init_$1.nc"
    exit 0
}

# Nothing to do at all if the final product already exists for the requested date.
if [[ -n "$DATE" && -f "Outputs/postprocessed/init_${DATE}T00.nc" ]]; then
    log INFO "Outputs/postprocessed/init_${DATE}T00.nc already exists. Nothing to do."
    exit 0
fi

# ── 1. download (prints the cycle) ───────────────────────────────────────────
log INFO "Step 'download' started"
start=$(date +%s)
CYCLE=$(python utils/download_ic.py ${DATE:+--date "$DATE"})
rc=$?
if [[ $rc -ne 0 ]]; then
    log ERROR "Step 'download' failed with exit $rc. Pipeline stopped."
    exit "$rc"
fi
if ! [[ "$CYCLE" =~ ^[0-9]{8}T00$ ]]; then
    log ERROR "download_ic.py returned an unexpected cycle '$CYCLE'. Pipeline stopped."
    exit 1
fi
log INFO "Step 'download' finished after $(( $(date +%s) - start )) s: cycle $CYCLE"

RAW="Outputs/raw/init_${CYCLE}.zarr"
POST="Outputs/postprocessed/init_${CYCLE}.nc"
JOB_NAME="${JOB_PREFIX}_${CYCLE}"

if [[ -f "$POST" ]]; then
    log INFO "$POST already exists. Nothing to do."
    exit 0
fi

# ── 2 + 3. preprocess and GPU inference (skipped if the raw forecast exists) ─
if [[ -d "$RAW" ]]; then
    log INFO "$RAW already exists. Skipping preprocessing and inference."
else
    run_step "preprocess" python utils/preprocess_ic.py --date "$CYCLE"

    existing=$(squeue -h -u "$USER" -n "$JOB_NAME" -o %i 2>/dev/null | head -n 1)
    if [[ -n "$existing" ]]; then
        # e.g. a previous wrapper died while its job kept running: wait for that job
        log INFO "GPU job $existing ($JOB_NAME) is already queued/running. Waiting for it."
        while squeue -h -j "$existing" 2>/dev/null | grep -q .; do
            sleep "$POLL_SECONDS"
        done
        state=$(sacct -n -X -j "$existing" -o State,ExitCode 2>/dev/null | head -n 1 | xargs)
        log INFO "GPU job $existing ended: ${state:-unknown}"
    else
        log INFO "Step 'inference' submitting $JOB_NAME and waiting for it"
        start=$(date +%s)
        sbatch --wait -J "$JOB_NAME" --export=ALL,DATE="$CYCLE" "$JOB_SCRIPT"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            log ERROR "Step 'inference' (job $JOB_NAME) failed with exit $rc after" \
                      "$(( $(date +%s) - start )) s. See logs/slurm-${JOB_NAME}-*.out. Pipeline stopped."
            exit "$rc"
        fi
        log INFO "Step 'inference' finished after $(( $(date +%s) - start )) s"
    fi

    if [[ ! -d "$RAW" ]]; then
        log ERROR "GPU job ended but $RAW was not produced. Pipeline stopped."
        exit 1
    fi
fi

# ── 4. postprocess ───────────────────────────────────────────────────────────
run_step "postprocess" python utils/postprocess.py --date "$CYCLE"

finish "$CYCLE"