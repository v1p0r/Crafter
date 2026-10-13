#!/bin/bash
# Run papers through stages 1-4 with gaps enabled and debug disabled.
#
# Output: data/ablation_output/Claude-sonnet4.6/gaps_no_debug/<paper>/
# Pass paper names as arguments to override the 18 historical defaults.

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
OUTPUT_ROOT="${OUTPUT_ROOT:-data/ablation_output/Claude-sonnet4.6/gaps_no_debug}"
BATCH_LOG="$OUTPUT_ROOT/_batch.log"

mkdir -p "$OUTPUT_ROOT"

for var in CLAUDE_API_KEY_2 GEMINI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "Error: $var not set" | tee -a "$BATCH_LOG"
        exit 1
    fi
done

export LLM_BACKEND=claude
export PYTHONHASHSEED=42
export PTC_RANDOM_SEED=42

MODEL_CHECK=$("$PYTHON_BIN" -c \
    'from src.utils.llm_client import BACKENDS, _current_backend; print("{}:{}".format(_current_backend, BACKENDS[_current_backend]["model"]))')
if [ "$MODEL_CHECK" != "claude:claude-sonnet-4-6" ]; then
    echo "Error: backend/model preflight failed: $MODEL_CHECK" | tee -a "$BATCH_LOG"
    exit 1
fi

DEFAULT_PAPERS=(
    bbox bridging-data-gaps ftrl lbcs lca-on-the-line
    mechanistic-understanding pinn robust-clip sample-specific-masks sapg
    self-composing-policies self-expansion semantic-self-consistency
    sequential-neural-score-estimation stay-on-topic-with-classifier-free-guidance
    stochastic-interpolant test-time-model-adaptation what-will-my-model-forget
)
if [ "$#" -gt 0 ]; then
    PAPERS=("$@")
else
    PAPERS=("${DEFAULT_PAPERS[@]}")
fi

if [ ! -d "$INPUT_ROOT" ]; then
    echo "Error: input root not found: $INPUT_ROOT" | tee -a "$BATCH_LOG"
    exit 1
fi
for PAPER in "${PAPERS[@]}"; do
    if [ -z "$(find "$INPUT_ROOT/$PAPER" -name '*_content_list_v2.json' -print -quit 2>/dev/null)" ]; then
        echo "Error: no *_content_list_v2.json found for $PAPER under $INPUT_ROOT" | tee -a "$BATCH_LOG"
        exit 1
    fi
done

run_one() {
    local PAPER="$1"
    local PAPER_OUT="$OUTPUT_ROOT/$PAPER"
    local LOG="$PAPER_OUT/run.log"
    local STATUS_FILE="$PAPER_OUT/.exit_status"

    if [ -d "$PAPER_OUT" ]; then
        echo "[$(date '+%F %T')] [$PAPER] output dir exists — skipping." >> "$BATCH_LOG"
        return 0
    fi

    local INPUT_FILE
    local INPUT
    INPUT_FILE=$(find "$INPUT_ROOT/$PAPER" -name "*_content_list_v2.json" -print -quit)
    INPUT=$(dirname "$INPUT_FILE")

    mkdir -p "$PAPER_OUT"
    echo "[$(date '+%F %T')] [$PAPER] start (backend=$LLM_BACKEND, model=claude-sonnet-4-6, input=$INPUT)" >> "$BATCH_LOG"

    if PTC_PAPER="$PAPER" \
        PTC_TOKEN_USAGE_LOG="$PAPER_OUT/token_usage.jsonl" \
        PTC_TOKEN_USAGE_SUMMARY="$PAPER_OUT/token_usage_summary.json" \
        "$PYTHON_BIN" -m src.main \
        --input "$INPUT" \
        --output "$PAPER_OUT" \
        --seed "$PTC_RANDOM_SEED" \
        --skip-debug > "$LOG" 2>&1; then
        echo "[$(date '+%F %T')] [$PAPER] DONE" >> "$BATCH_LOG"
        echo "0" > "$STATUS_FILE"
    else
        local EXIT=$?
        echo "[$(date '+%F %T')] [$PAPER] FAILED (exit $EXIT, see $LOG)" >> "$BATCH_LOG"
        echo "$EXIT" > "$STATUS_FILE"
    fi
}

N=${#PAPERS[@]}
TOTAL_BATCHES=$(( (N + 2) / 3 ))

echo "============================================" | tee -a "$BATCH_LOG"
echo "[$(date '+%F %T')] gaps_no_debug — $N-paper run, 3-parallel batches" | tee -a "$BATCH_LOG"
echo "backend/model=$MODEL_CHECK  python=$PYTHON_BIN  seed=$PTC_RANDOM_SEED  PYTHONHASHSEED=$PYTHONHASHSEED" | tee -a "$BATCH_LOG"
echo "input_root=$INPUT_ROOT  output_root=$OUTPUT_ROOT" | tee -a "$BATCH_LOG"
echo "Papers ($N) in $TOTAL_BATCHES batches of 3" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"

BATCH=1
for ((i=0; i<N; i+=3)); do
    CHUNK=("${PAPERS[@]:i:3}")
    echo "" | tee -a "$BATCH_LOG"
    echo "[$(date '+%F %T')] === BATCH $BATCH/$TOTAL_BATCHES: ${CHUNK[*]} ===" | tee -a "$BATCH_LOG"
    for P in "${CHUNK[@]}"; do
        run_one "$P" &
        echo "[$(date '+%F %T')]   [$P] launched (pid $!)" | tee -a "$BATCH_LOG"
    done
    wait
    echo "[$(date '+%F %T')] === BATCH $BATCH/$TOTAL_BATCHES done ===" | tee -a "$BATCH_LOG"
    BATCH=$((BATCH+1))
done

# Aggregate
SUCCEEDED=""
FAILED=""
SKIPPED=""
N_SUCCEEDED=0
N_FAILED=0
N_SKIPPED=0
for PAPER in "${PAPERS[@]}"; do
    STATUS_FILE="$OUTPUT_ROOT/$PAPER/.exit_status"
    if [ ! -f "$STATUS_FILE" ]; then
        FAILED="$FAILED $PAPER(no-status)"
        N_FAILED=$((N_FAILED + 1))
        continue
    fi
    STATUS=$(cat "$STATUS_FILE")
    case "$STATUS" in
        0)        SUCCEEDED="$SUCCEEDED $PAPER"; N_SUCCEEDED=$((N_SUCCEEDED + 1)) ;;
        skipped)  SKIPPED="$SKIPPED $PAPER"; N_SKIPPED=$((N_SKIPPED + 1)) ;;
        *)        FAILED="$FAILED $PAPER(exit:$STATUS)"; N_FAILED=$((N_FAILED + 1)) ;;
    esac
done

echo "" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"
echo "[$(date '+%F %T')] BATCH SUMMARY ($N papers)" | tee -a "$BATCH_LOG"
echo "  Succeeded ($N_SUCCEEDED):$SUCCEEDED" | tee -a "$BATCH_LOG"
echo "  Skipped   ($N_SKIPPED):$SKIPPED" | tee -a "$BATCH_LOG"
echo "  Failed    ($N_FAILED):$FAILED" | tee -a "$BATCH_LOG"
echo "============================================" | tee -a "$BATCH_LOG"

[ $N_FAILED -eq 0 ] || exit 1
