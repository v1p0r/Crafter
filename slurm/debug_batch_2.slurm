#!/bin/bash
#SBATCH -A TG-NAIRR250444
#SBATCH --job-name=dbg-merged-b2
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -o slurm/logs/debug_batch_2_%j.out
#SBATCH -e slurm/logs/debug_batch_2_%j.err
#SBATCH --mail-user=fl374@rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

# Submit with:
#   ANTHROPIC_API_KEY=... sbatch slurm/debug_batch_2.slurm

BATCH_NAME="debug_batch_2"
PAPERS=(
    "all-in-one"
    "fre"
    "lca-on-the-line"
    "sample-specific-masks"
    "semantic-self-consistency"
    "test-time-model-adaptation"
)
export BATCH_NAME PAPERS

source /scratch/09979/esther_lh/code_gen/ptc_agent_merged/slurm/_run_debug_batch.sh
