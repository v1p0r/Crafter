#!/bin/bash
# Re-run the 10 papers that hit VLM 503s during the original paperbench batch.
# Seeds figure_analysis.json from data/output/MiniMax-M2.7/vlm-retest/<paper>/
# (produced by scripts/vlm_retest.py with the new retry logic), so the pipeline
# skips VLM (cached) and re-does partition + gap + planning + codegen against
# the better figure analysis.
#
# Output: data/output/MiniMax-M2.7/skip-debug-v2/<paper>/

# Note: `set -u` removed — macOS bash 3.2 errors on `${arr[*]}` for an empty
# array under nounset, which made the trailing summary echoes fail and the
# script exit rc=1 even when papers succeeded. API-key checks below still
# enforce that the env vars exist.

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate ptc

INPUT_ROOT="/Users/esther_lh/GitHub/paperbench_data/data/output_v2"
VLM_CACHE_ROOT="data/output/MiniMax-M2.7/vlm-retest"
OUTPUT_ROOT="data/output/MiniMax-M2.7/skip-debug-v2"
BATCH_LOG="$OUTPUT_ROOT/_batch.log"

mkdir -p "$OUTPUT_ROOT"

PAPERS=(
    ftrl
    lca-on-the-line
)

: "${ANTHROPIC_API_KEY:?must be set}"
: "${GEMINI_API_KEY:?must be set}"

echo "============================================" | tee -a "$BATCH_LOG"
echo "[$(date '+%F %T')] Re-running ${#PAPERS[@]} VLM-failed papers" | tee -a "$BATCH_LOG"
echo "Input root:      $INPUT_ROOT" | tee -a "$BATCH_LOG"
echo "VLM cache root:  $VLM_CACHE_ROOT" | tee -a "$BATCH_LOG"
echo "Output root:     $OUTPUT_ROOT" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"

SUCCEEDED=()
FAILED=()
SKIPPED=()

for PAPER in "${PAPERS[@]}"; do
    PAPER_OUT="$OUTPUT_ROOT/$PAPER"
    LOG="$PAPER_OUT/run.log"
    VLM_CACHED_FA="$VLM_CACHE_ROOT/$PAPER/global_memory/figure_analysis.json"

    if [ -d "$PAPER_OUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] output dir exists — skipping" | tee -a "$BATCH_LOG"
        SKIPPED+=("$PAPER")
        continue
    fi

    if [ ! -f "$VLM_CACHED_FA" ]; then
        echo "[$(date '+%F %T')] [$PAPER] ERROR: cached figure_analysis missing at $VLM_CACHED_FA" | tee -a "$BATCH_LOG"
        FAILED+=("$PAPER (no vlm cache)")
        continue
    fi

    INPUT=$(find "$INPUT_ROOT/$PAPER" -name "*_content_list_v2.json" -exec dirname {} \; | head -1)
    if [ -z "$INPUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] ERROR: no content_list_v2.json found" | tee -a "$BATCH_LOG"
        FAILED+=("$PAPER (no input)")
        continue
    fi

    mkdir -p "$PAPER_OUT/global_memory"
    cp "$VLM_CACHED_FA" "$PAPER_OUT/global_memory/figure_analysis.json"

    echo "" | tee -a "$BATCH_LOG"
    echo "[$(date '+%F %T')] [$PAPER] start (input: $INPUT, vlm seeded from $VLM_CACHED_FA)" | tee -a "$BATCH_LOG"

    if python3 -m src.main \
        --input "$INPUT" \
        --output "$PAPER_OUT" \
        --skip-debug > "$LOG" 2>&1; then
        echo "[$(date '+%F %T')] [$PAPER] DONE" | tee -a "$BATCH_LOG"
        SUCCEEDED+=("$PAPER")
    else
        EXIT=$?
        echo "[$(date '+%F %T')] [$PAPER] FAILED (exit $EXIT, see $LOG)" | tee -a "$BATCH_LOG"
        FAILED+=("$PAPER")
    fi
done

echo "" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"
echo "[$(date '+%F %T')] BATCH SUMMARY" | tee -a "$BATCH_LOG"
echo "  Succeeded (${#SUCCEEDED[@]}): ${SUCCEEDED[*]}" | tee -a "$BATCH_LOG"
echo "  Skipped   (${#SKIPPED[@]}):   ${SKIPPED[*]}" | tee -a "$BATCH_LOG"
echo "  Failed    (${#FAILED[@]}):    ${FAILED[*]}" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"

[ ${#FAILED[@]} -eq 0 ] || exit 1
