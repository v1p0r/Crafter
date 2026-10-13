#!/bin/bash
# Run all paperbench papers through the PTC pipeline (VLM on, gap on, debug off).
# Backend: MiniMax-M2.7 (default in src/utils/llm_client.py); VLM: gemini-2.5-flash.
# Output:  data/output/MiniMax-M2.7/skip-debug/<paper>/
# Per-paper log:  data/output/MiniMax-M2.7/skip-debug/<paper>/run.log
# Batch log:      data/output/MiniMax-M2.7/skip-debug/_batch.log

set -u

cd "$(dirname "$0")"

eval "$(conda shell.bash hook)"
conda activate ptc

INPUT_ROOT="/Users/esther_lh/GitHub/paperbench_data/data/output_v2"
OUTPUT_ROOT="data/output/MiniMax-M2.7/skip-debug"
BATCH_LOG="$OUTPUT_ROOT/_batch.log"

mkdir -p "$OUTPUT_ROOT"

for var in ANTHROPIC_API_KEY GEMINI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: $var not set" | tee -a "$BATCH_LOG"
        exit 1
    fi
done

PAPERS=($(ls -1 "$INPUT_ROOT"))

echo "============================================" | tee -a "$BATCH_LOG"
echo "[$(date '+%F %T')] Starting batch — ${#PAPERS[@]} papers" | tee -a "$BATCH_LOG"
echo "Input root:  $INPUT_ROOT" | tee -a "$BATCH_LOG"
echo "Output root: $OUTPUT_ROOT" | tee -a "$BATCH_LOG"
echo "Flags:       --skip-debug (VLM on, gaps on)" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"

SUCCEEDED=()
FAILED=()
SKIPPED=()

for PAPER in "${PAPERS[@]}"; do
    PAPER_OUT="$OUTPUT_ROOT/$PAPER"
    LOG="$PAPER_OUT/run.log"

    if [ -d "$PAPER_OUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] output dir exists — skipping." | tee -a "$BATCH_LOG"
        SKIPPED+=("$PAPER")
        continue
    fi

    INPUT=$(find "$INPUT_ROOT/$PAPER" -name "*_content_list_v2.json" -exec dirname {} \; | head -1)
    if [ -z "$INPUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] ERROR: no content_list_v2.json found." | tee -a "$BATCH_LOG"
        FAILED+=("$PAPER (no input)")
        continue
    fi

    mkdir -p "$PAPER_OUT"

    echo "" | tee -a "$BATCH_LOG"
    echo "[$(date '+%F %T')] [$PAPER] start (input: $INPUT)" | tee -a "$BATCH_LOG"

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
