#!/bin/bash
# Submit all 4 debug-only batches. Pre-req: stages 1-6 outputs already exist
# under data/output/MiniMax-M2.7/skip-debug/<paper>/.
#
# Usage:
#   ANTHROPIC_API_KEY=... ./slurm/submit_debug_all.sh

set -e
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"

cd "$(dirname "$0")/.."

for n in 1 2 3 4; do
    job=$(sbatch --parsable slurm/debug_batch_${n}.slurm)
    echo "submitted debug_batch_${n}: jobid=${job}"
done

echo
squeue -u "$USER"
