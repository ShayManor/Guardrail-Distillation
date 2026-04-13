#!/bin/bash -l
#
# E2 ablation (paper Table 2): train the Guardrail++ head for mit-b1 in all
# four supervision modes, using a SLURM array job. Each array task is an
# independent 12-hour guardrail training run.
#
# Usage:
#   sbatch slurm/b1/train_guardrail_ablation.sbatch
#
# Outputs land in runs/mit-b1_guard_<supervision_type>_j<jobid> — one
# directory per mode. The matching guardrail.ckpt can then be scored by
# src/eval/full_eval.py on Cityscapes-val and ACDC.
#
#SBATCH --job-name=b1-guard-abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH -o /scratch/gautschi/manors/logs/b1-guard-abl_%A_%a.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-guard-abl_%A_%a.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODES=("scalar_benefit" "dense_disagree" "dense_gap" "dense_multi")
SUPERVISION_TYPE="${MODES[$SLURM_ARRAY_TASK_ID]}"

MODEL=b1
MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

SRC_DIR="${SRC_DIR:-}"
if [ -z "$SRC_DIR" ]; then
  SRC_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SRC_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
  echo "[auto] Found latest SKD run: $SRC_DIR"
fi
[ ! -f "$SRC_DIR/student_skd.ckpt" ] && echo "ERROR: $SRC_DIR/student_skd.ckpt not found." && exit 1

OUT="runs/mit-${MODEL}_guard_${SUPERVISION_TYPE}_j${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "[ablation] task=$SLURM_ARRAY_TASK_ID mode=$SUPERVISION_TYPE output=$OUT"

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --checkpoint-dir "$SRC_DIR" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --skip-sup --skip-kd --skip-skd \
  --epochs-guardrail 35 \
  --batch-size 8 \
  --lr 5e-4 \
  --supervision-type "$SUPERVISION_TYPE" \
  --dense-disagree-weight 1.0 \
  --dense-gap-weight 1.0 \
  --corruption-prob 0.5 \
  --use-student-features \
  --log-every 20

echo "[DONE] Ablation task $SLURM_ARRAY_TASK_ID (${SUPERVISION_TYPE}) complete."
