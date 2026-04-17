#!/bin/bash -l
# Launch multi-seed training for the teacher-vs-GT comparison.
#
# Submits 9 jobs: 3 modes × 3 seeds. Each is an independent SLURM job
# using the existing train_guardrail.sbatch with SUPERVISION_TYPE and SEED
# env vars.
#
# Usage:
#   bash slurm/b1/train_multiseed_teacher_vs_gt.sh
#
# This is a launcher script, not an sbatch script — run it directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GUARDRAIL_SBATCH="$SCRIPT_DIR/train_guardrail.sbatch"

MODES=(dense_multi gt_disagree gt_risk)
SEEDS=(42 137 256)

echo "[multi-seed] Launching ${#MODES[@]} modes × ${#SEEDS[@]} seeds = $(( ${#MODES[@]} * ${#SEEDS[@]} )) jobs"

for MODE in "${MODES[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    echo "  sbatch SUPERVISION_TYPE=$MODE SEED=$SEED"
    SUPERVISION_TYPE="$MODE" SEED="$SEED" sbatch "$GUARDRAIL_SBATCH"
  done
done

echo "[multi-seed] All jobs submitted."
