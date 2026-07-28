#!/bin/bash -l
#SBATCH --job-name=b1-guard-e2efusion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/b1-guard-e2efusion_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-guard-e2efusion_%j.err

# End-to-end fusion: dense_multi guardrail head that ALSO ingests per-pixel
# energy + max-logit as input channels (--use-confidence-features), so a single
# learned score fuses disagreement prediction with confidence magnitude.
# Stage 4 only (student/teacher frozen).
set -euo pipefail
source $SCRATCH/venvs/guardrail/bin/activate
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=~/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODEL="${MODEL:-b1}"
SEED="${SEED:-42}"
MIT_STUDENT=$(find $SCRATCH ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find $SCRATCH ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

SRC_DIR="${SRC_DIR:-}"
if [ -z "$SRC_DIR" ]; then
  SRC_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SRC_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
  echo "[auto] SKD run: $SRC_DIR"
fi
[ ! -f "$SRC_DIR/student_skd.ckpt" ] && echo "ERROR: $SRC_DIR/student_skd.ckpt not found." && exit 1

OUT="runs/mit-${MODEL}_guard_dense_multi_conffeat_s${SEED}_j${SLURM_JOB_ID}"

HF_HOME=$SCRATCH/hf_cache \
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
  --supervision-type dense_multi \
  --dense-disagree-weight 1.0 \
  --dense-gap-weight 1.0 \
  --corruption-prob 0.5 \
  --use-student-features \
  --use-confidence-features \
  --seed "$SEED" \
  --log-every 20

echo "[DONE] e2e-fusion guardrail seed=${SEED}. Output: $OUT  Source: $SRC_DIR"
