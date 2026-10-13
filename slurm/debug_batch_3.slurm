#!/bin/bash
#SBATCH -A TG-NAIRR250444
#SBATCH --job-name=dbg-merged-b3
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gh
#SBATCH -o slurm/logs/debug_batch_3_%j.out
#SBATCH -e slurm/logs/debug_batch_3_%j.err
#SBATCH --mail-user=fl374@rutgers.edu
#SBATCH --mail-type=ALL
#SBATCH --export=ALL

# Submit with:
#   ANTHROPIC_API_KEY=... sbatch slurm/debug_batch_3.slurm

BATCH_NAME="debug_batch_3"
PAPERS=(
    "bam"
    "ftrl"
    "mechanistic-understanding"
    "robust-clip"
    "sapg"
    "sequential-neural-score-estimation"
)
export BATCH_NAME PAPERS

source /scratch/09979/esther_lh/code_gen/ptc_agent_merged/slurm/_run_debug_batch.sh
