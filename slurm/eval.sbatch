#!/bin/bash -l
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/eval-all_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/eval-all_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

ACDC_DIR="$REPO/paper_eval_acdc_j${SLURM_JOB_ID}"
CITY_DIR="$REPO/paper_eval_city_j${SLURM_JOB_ID}"
mkdir -p "$ACDC_DIR" "$CITY_DIR"

for MODEL in b0 b1 b2; do
  echo "============================================"
  echo "[Setup] MODEL=$MODEL"
  echo "============================================"

  MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)

  # Discover checkpoints (overridable: B0_SKD_DIR, B1_GUARD_DIR, etc.)
  skd_var="${MODEL^^}_SKD_DIR"    # e.g. B0_SKD_DIR
  guard_var="${MODEL^^}_GUARD_DIR"
  SKD_DIR="${!skd_var:-}"
  GUARD_DIR="${!guard_var:-}"

  if [ -z "$SKD_DIR" ]; then
    SKD_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
    [ -z "$SKD_DIR" ] && echo "ERROR: No SKD run found for mit-${MODEL}." && exit 1
  fi
  if [ -z "$GUARD_DIR" ]; then
    GUARD_DIR=$(ls -td runs/mit-${MODEL}_guard_*/guardrail.ckpt outputs*/guardrail.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
    [ -z "$GUARD_DIR" ] && echo "ERROR: No guardrail run found for mit-${MODEL}." && exit 1
  fi

  STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
  GUARDRAIL_CKPT="$GUARD_DIR/guardrail.ckpt"

  for f in "$STUDENT_CKPT" "$GUARDRAIL_CKPT"; do
    [ ! -f "$f" ] && echo "ERROR: $f missing" && exit 1
  done

  echo "[auto] SKD dir:   $SKD_DIR"
  echo "[auto] Guard dir: $GUARD_DIR"

  # ── ACDC eval ──
  for DOMAIN in fog night rain snow all; do
    echo "--------------------------------------------"
    echo "[ACDC] $MODEL  domain=$DOMAIN"
    echo "--------------------------------------------"

    HF_HOME=/scratch/gautschi/manors/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "acdc_${MODEL}_skd_${DOMAIN}" \
      --dataset-name acdc \
      --dataset-path "$REPO/data/acdc" \
      --split val \
      --domain "$DOMAIN" \
      --student-name student_skd \
      --train-method skd \
      --student-backbone "$MIT_STUDENT" \
      --student-ckpt "$STUDENT_CKPT" \
      --teacher-backbone "$TEACHER" \
      --guardrail-ckpt "$GUARDRAIL_CKPT" \
      --guardrail-student-name student_skd \
      --mc-dropout-passes 4 \
      --output-dir "$ACDC_DIR" \
      --batch-size 4 \
      --num-workers 4
  done

  # ── Cityscapes eval ──
  echo "--------------------------------------------"
  echo "[City] $MODEL"
  echo "--------------------------------------------"

  HF_HOME=/scratch/gautschi/manors/hf_cache \
  python -u src/eval/full_eval.py eval \
    --run-id "${MODEL}_cityscapes_val" \
    --dataset-name cityscapes \
    --dataset-path "$REPO/data/cityscapes" \
    --split val \
    --domain all \
    --student-name student_skd \
    --train-method skd \
    --student-backbone "$MIT_STUDENT" \
    --student-ckpt "$STUDENT_CKPT" \
    --teacher-backbone "$TEACHER" \
    --guardrail-ckpt "$GUARDRAIL_CKPT" \
    --guardrail-student-name student_skd \
    --mc-dropout-passes 4 \
    --output-dir "$CITY_DIR" \
    --batch-size 4 \
    --num-workers 4
done

echo ""
echo "[DONE] All evals complete."
echo "  ACDC results: $ACDC_DIR/csv/"
echo "  City results: $CITY_DIR/csv/"
