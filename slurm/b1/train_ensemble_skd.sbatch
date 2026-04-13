#!/bin/bash -l
#
# E4 deep-ensemble baseline: train N independent SKD students for mit-b1,
# each with a different random seed. Used to score predictive-entropy of an
# ensemble as a selective-prediction baseline (Section 5, Paper Table 1).
#
# We reuse the existing mit-b1 sup checkpoint as the common starting point
# (all ensemble members share the same supervised init), then the SKD stage
# is what introduces the diversity (different KD data order, dropout,
# init noise in the pairwise-affinity projection).
#
# Array tasks correspond to ensemble members 1..N. The user asked us NOT to
# retrain the supervised stage multiple times (it would cost 3×12h on its
# own), so we only retrain the SKD stage here.
#
# Usage:
#   sbatch slurm/b1/train_ensemble_skd.sbatch
#   # or override the number of members:
#   sbatch --array=0-4 slurm/b1/train_ensemble_skd.sbatch     # 5 members
#
# Outputs: runs/mit-b1_ensemble_m<member>_j<jobid>/student_skd.ckpt
#
#SBATCH --job-name=b1-ens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
#SBATCH -o /scratch/gautschi/manors/logs/b1-ens_%A_%a.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-ens_%A_%a.err

set -euo pipefail
source /scratch/gautschi/manors/venvs/guardrail/bin/activate

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

MODEL=b1

# Map array index → (member_id, seed). The first member_id=1 uses seed 42 to
# coincide with the existing single-seed run.
MEMBER_ID=$((SLURM_ARRAY_TASK_ID + 1))
SEEDS=(42 137 256 521 1024)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

# Share the supervised stage across every ensemble member.
SUP_SRC="${SUP_SRC:-}"
if [ -z "$SUP_SRC" ]; then
  SUP_SRC=$(ls -td runs/mit-${MODEL}_sup_*/student_sup.ckpt outputs*/student_sup.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SUP_SRC" ] && echo "ERROR: No mit-${MODEL} sup checkpoint found." && exit 1
  echo "[auto] sup source: $SUP_SRC"
fi
[ ! -f "$SUP_SRC/student_sup.ckpt" ] && echo "ERROR: $SUP_SRC/student_sup.ckpt not found." && exit 1

OUT="runs/mit-${MODEL}_ensemble_m${MEMBER_ID}_seed${SEED}_j${SLURM_ARRAY_JOB_ID}"
mkdir -p "$OUT"

echo "[ensemble] member=$MEMBER_ID seed=$SEED output=$OUT"

HF_HOME=/scratch/gautschi/manors/hf_cache \
python -u src/train/run.py train \
  --dataset-path "$REPO/data/cityscapes" \
  --output-dir "$OUT" \
  --checkpoint-dir "$SUP_SRC" \
  --student-model "$MIT_STUDENT" \
  --teacher-model "$TEACHER" \
  --epochs-sup 1 --epochs-kd 1 --epochs-skd 60 --epochs-guardrail 1 \
  --skip-sup --skip-kd --skip-guardrail \
  --batch-size 8 \
  --lr 3e-5 \
  --alpha-kd 1.0 --alpha-struct 0.5 \
  --kd-temperature 2.0 \
  --seed "$SEED" \
  --log-every 20

echo "[DONE] Ensemble member $MEMBER_ID (seed $SEED) complete. Output: $OUT"
