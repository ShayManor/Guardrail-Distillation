#!/bin/bash -l
#SBATCH --job-name=gpp-5-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=0:30:00
#SBATCH -A lilly-rob1
#SBATCH -p highmem
#SBATCH -o /scratch/gautschi/manors/logs/gpp-5-eval-v4_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/gpp-5-eval-v4_%j.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

# ── Change MODEL to switch backbone (b0, b1, b2, …) ──
MODEL=${MODEL:-b0}

MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Auto-discover latest guardrail run for mit-${MODEL} (override with env vars)
GUARD_DIR="${GUARD_DIR:-}"
if [ -z "$GUARD_DIR" ]; then
  GUARD_DIR=$(ls -td runs/mit-${MODEL}_guard_*/guardrail.ckpt outputs*/guardrail.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  if [ -z "$GUARD_DIR" ]; then
    echo "ERROR: No guardrail run found for mit-${MODEL}. Run train_guardrail.sbatch first or set GUARD_DIR."
    exit 1
  fi
  echo "[auto] Found latest guardrail run: $GUARD_DIR"
fi

# Auto-discover latest SKD run for mit-${MODEL}
SKD_DIR="${SKD_DIR:-}"
if [ -z "$SKD_DIR" ]; then
  SKD_DIR=$(ls -td runs/mit-${MODEL}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  if [ -z "$SKD_DIR" ]; then
    echo "ERROR: No SKD run found for mit-${MODEL}. Run train_skd.sbatch first or set SKD_DIR."
    exit 1
  fi
  echo "[auto] Found latest SKD run: $SKD_DIR"
fi

STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
GUARDRAIL_CKPT="$GUARD_DIR/guardrail.ckpt"
EVAL_DIR="$REPO/paper_eval_${GUARD_DIR##*/}"

for f in "$STUDENT_CKPT" "$GUARDRAIL_CKPT"; do
  [ ! -f "$f" ] && echo "ERROR: $f missing" && exit 1
done

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/eval/full_eval.py eval \
  --run-id "guardrailpp_${MODEL}_cityscapes_val" \
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
  --output-dir "$EVAL_DIR" \
  --batch-size 2 \
  --num-workers 4

echo "[DONE] Cityscapes eval complete."
echo "  Student:   $STUDENT_CKPT"
echo "  Guardrail: $GUARDRAIL_CKPT"
echo "  Results:   $EVAL_DIR"
