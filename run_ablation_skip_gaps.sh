#!/bin/bash
# Ablation: run papers with both gap analysis and debug disabled.
#
# Output: data/ablation_output/Claude-sonnet4.6/no_gaps_no_debug/<paper>/
# Pass paper names as arguments to override the five historical defaults.

set -euo pipefail

cd "$(dirname "$0")"

# Vista modules + shared PTC virtual environment.
module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8
PTC_VENV="${PTC_VENV:-${WORK:-/work/09979/esther_lh/vista}/python_venvs/ptc}"
source "$PTC_VENV/bin/activate"
PYTHON_BIN="$PTC_VENV/bin/python"

INPUT_ROOT="${INPUT_ROOT:-data/parsed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/ablation_output/Claude-sonnet4.6/no_gaps_no_debug}"
ABLATION_LOG="$OUTPUT_ROOT/_batch.log"

mkdir -p "$OUTPUT_ROOT"

for var in CLAUDE_API_KEY_2 GEMINI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: $var not set" | tee -a "$ABLATION_LOG"
        exit 1
    fi
done

export LLM_BACKEND=claude
export PYTHONHASHSEED=42
export PTC_RANDOM_SEED=42

MODEL_CHECK=$("$PYTHON_BIN" -c \
    'from src.utils.llm_client import BACKENDS, _current_backend; print("{}:{}".format(_current_backend, BACKENDS[_current_backend]["model"]))')
if [ "$MODEL_CHECK" != "claude:claude-sonnet-4-6" ]; then
    echo "Error: backend/model preflight failed: $MODEL_CHECK" | tee -a "$ABLATION_LOG"
    exit 1
fi

DEFAULT_PAPERS=(
    bam
    all-in-one
    pinn
    mechanistic-understanding
    bbox
)
if [ "$#" -gt 0 ]; then
    PAPERS=("$@")
else
    PAPERS=("${DEFAULT_PAPERS[@]}")
fi

if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: input root not found: $INPUT_ROOT" | tee -a "$ABLATION_LOG"
    exit 1
fi
for PAPER in "${PAPERS[@]}"; do
    if [ -z "$(find "$INPUT_ROOT/$PAPER" -name '*_content_list_v2.json' -print -quit 2>/dev/null)" ]; then
        echo "Error: no *_content_list_v2.json found for $PAPER under $INPUT_ROOT" | tee -a "$ABLATION_LOG"
        exit 1
    fi
done

run_one() {
    local PAPER="$1"
    local PAPER_OUT="$OUTPUT_ROOT/$PAPER"
    local LOG="$PAPER_OUT/run.log"
    local STATUS_FILE="$PAPER_OUT/.exit_status"

    if [ -d "$PAPER_OUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] output dir exists — skipping" >> "$ABLATION_LOG"
        return 0
    fi

    local INPUT_FILE
    local INPUT
    INPUT_FILE=$(find "$INPUT_ROOT/$PAPER" -name "*_content_list_v2.json" -print -quit)
    INPUT=$(dirname "$INPUT_FILE")

    mkdir -p "$PAPER_OUT"
    echo "[$(date '+%F %T')] [$PAPER] start (backend=$LLM_BACKEND, model=claude-sonnet-4-6, --skip-gaps --skip-debug)" >> "$ABLATION_LOG"

    if PTC_PAPER="$PAPER" \
        PTC_TOKEN_USAGE_LOG="$PAPER_OUT/token_usage.jsonl" \
        PTC_TOKEN_USAGE_SUMMARY="$PAPER_OUT/token_usage_summary.json" \
        "$PYTHON_BIN" -m src.main \
        --input "$INPUT" \
        --output "$PAPER_OUT" \
        --seed "$PTC_RANDOM_SEED" \
        --skip-debug \
        --skip-gaps > "$LOG" 2>&1; then
        echo "[$(date '+%F %T')] [$PAPER] DONE" >> "$ABLATION_LOG"
        echo "0" > "$STATUS_FILE"
    else
        local EXIT=$?
        echo "[$(date '+%F %T')] [$PAPER] FAILED (exit $EXIT, see $LOG)" >> "$ABLATION_LOG"
        echo "$EXIT" > "$STATUS_FILE"
    fi
}

echo "============================================" | tee -a "$ABLATION_LOG"
echo "[$(date '+%F %T')] no_gaps_no_debug — ${#PAPERS[@]} papers in parallel" | tee -a "$ABLATION_LOG"
echo "backend/model=$MODEL_CHECK  python=$PYTHON_BIN  seed=$PTC_RANDOM_SEED  PYTHONHASHSEED=$PYTHONHASHSEED" | tee -a "$ABLATION_LOG"
echo "input_root=$INPUT_ROOT  output_root=$OUTPUT_ROOT" | tee -a "$ABLATION_LOG"
for PAPER in "${PAPERS[@]}"; do echo "  $PAPER" | tee -a "$ABLATION_LOG"; done
echo "============================================" | tee -a "$ABLATION_LOG"

for PAPER in "${PAPERS[@]}"; do
    run_one "$PAPER" &
    echo "[$(date '+%F %T')]   [$PAPER] launched (pid $!, backend=$LLM_BACKEND)" | tee -a "$ABLATION_LOG"
done

wait

# Summary
SUCCEEDED=""
FAILED=""
SKIPPED=""
for PAPER in "${PAPERS[@]}"; do
    ST="$OUTPUT_ROOT/$PAPER/.exit_status"
    S=$(cat "$ST" 2>/dev/null || echo missing)
    case "$S" in
        0)        SUCCEEDED="$SUCCEEDED $PAPER" ;;
        skipped)  SKIPPED="$SKIPPED $PAPER" ;;
        *)        FAILED="$FAILED $PAPER(exit:$S)" ;;
    esac
done

echo "" | tee -a "$ABLATION_LOG"
echo "============================================" | tee -a "$ABLATION_LOG"
echo "[$(date '+%F %T')] ABLATION SUMMARY" | tee -a "$ABLATION_LOG"
echo "  Succeeded:$SUCCEEDED" | tee -a "$ABLATION_LOG"
echo "  Skipped:  $SKIPPED" | tee -a "$ABLATION_LOG"
echo "  Failed:   $FAILED" | tee -a "$ABLATION_LOG"
echo "============================================" | tee -a "$ABLATION_LOG"

[ -z "$FAILED" ] || exit 1
