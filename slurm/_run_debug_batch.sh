#!/bin/bash
# Shared body sourced by every debug_batch_*.slurm.
# Runs `--only-debug` on output dirs already populated by a stages-1-6 run
# (data/output/MiniMax-M2.7/skip-debug/<paper>/).
#
# Caller exports: BATCH_NAME (e.g. "debug_batch_1") and PAPERS=(...).

set -uo pipefail

: "${BATCH_NAME:?BATCH_NAME must be set by the caller}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY must be exported in the submitting shell}"

# Vista module + venv setup
module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8
source "$WORK/python_venvs/ptc/bin/activate"
echo "python: $(which python)"

cd /scratch/09979/esther_lh/code_gen/ptc_agent_merged

MODEL_TAG="MiniMax-M2.7"
GROUP="skip-debug"          # subfolder under output/<MODEL_TAG>/ where stages 1-6 outputs live
DEBUG_ITER=30

mkdir -p slurm/logs slurm/summaries
SUMMARY="slurm/summaries/${BATCH_NAME}_${SLURM_JOB_ID:-local}.txt"

{
    echo "=== ${BATCH_NAME} — stage 7 only, ${DEBUG_ITER}-iter debug ==="
    echo "started:  $(date -Iseconds)"
    echo "host:     $(hostname)"
    echo "job_id:   ${SLURM_JOB_ID:-(not under slurm)}"
    echo "papers:   ${#PAPERS[@]}"
    echo "list:     ${PAPERS[*]}"
    echo
} | tee "$SUMMARY"

FAILED=()
SUCCEEDED=()
SKIPPED=()

for PAPER in "${PAPERS[@]}"; do
    {
        echo
        echo "============================================"
        echo "[$(date -Iseconds)] Debug: $PAPER"
        echo "============================================"
    } | tee -a "$SUMMARY"

    OUTPUT_DIR="data/output/${MODEL_TAG}/${GROUP}/${PAPER}"

    # Require stages 1-6 to have run (output dir + codebase must exist)
    if [ ! -d "$OUTPUT_DIR/codebase" ]; then
        echo "[$PAPER] SKIP: $OUTPUT_DIR/codebase missing — stages 1-6 must run first" \
            | tee -a "$SUMMARY"
        SKIPPED+=("$PAPER")
        continue
    fi

    LOG="$OUTPUT_DIR/pipeline_debug_${SLURM_JOB_ID:-local}.log"

    if python -u -m src.main \
            --output "$OUTPUT_DIR" \
            --only-debug \
            --debug-iterations "$DEBUG_ITER" \
            2>&1 | tee -a "$LOG"; then
        rc=${PIPESTATUS[0]}
    else
        rc=${PIPESTATUS[0]}
    fi

    if [ "$rc" -eq 0 ]; then
        echo "[$PAPER] DONE" | tee -a "$SUMMARY"
        SUCCEEDED+=("$PAPER")
    else
        echo "[$PAPER] FAILED rc=$rc" | tee -a "$SUMMARY"
        FAILED+=("$PAPER (rc=$rc)")
    fi

    # Smoke-pass-rate snapshot of the unit tests (non-fatal)
    if [ -d "$OUTPUT_DIR/codebase/tests" ]; then
        TEST_LOG="$OUTPUT_DIR/test_results.log"
        ( cd "$OUTPUT_DIR/codebase" && SMOKE=1 python -u -m unittest discover tests -v ) \
            > "$TEST_LOG" 2>&1 || true
    fi
done

{
    echo
    echo "=== SUMMARY ==="
    echo "succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}"
    echo "failed    (${#FAILED[@]}):    ${FAILED[*]}"
    echo "skipped   (${#SKIPPED[@]}):   ${SKIPPED[*]}"
    echo "ended:    $(date -Iseconds)"
} | tee -a "$SUMMARY"

[ ${#FAILED[@]} -eq 0 ]
