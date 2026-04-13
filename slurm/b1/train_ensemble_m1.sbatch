#!/bin/bash -l
#SBATCH --job-name=b1-ens-m1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH -A lilly-rob1
#SBATCH -p smallgpu
#SBATCH --gres=gpu:1
#SBATCH -o /scratch/gautschi/manors/logs/b1-ens-m1_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/b1-ens-m1_%j.err

# E4 deep-ensemble member 1 (seed 42). Trains one SKD student for mit-b1
# from the shared supervised checkpoint. The paper's ensemble baseline
# averages the softmax of m1+m2+m3 to compute ensemble predictive entropy
# as a selective-prediction score (Paper Table 1, ensemble row).

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
MEMBER_ID=1
SEED=42

MIT_STUDENT=$(find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${MODEL}/snapshots/*" -type d 2>/dev/null | head -n1)
TEACHER=$(find /scratch/gautschi/manors ~/.cache -path '*models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/*' -type d 2>/dev/null | head -n1)

SUP_SRC="${SUP_SRC:-}"
if [ -z "$SUP_SRC" ]; then
  SUP_SRC=$(ls -td runs/mit-${MODEL}_sup_*/student_sup.ckpt outputs*/student_sup.ckpt 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
  [ -z "$SUP_SRC" ] && echo "ERROR: No mit-${MODEL} sup checkpoint found." && exit 1
  echo "[auto] sup source: $SUP_SRC"
fi
[ ! -f "$SUP_SRC/student_sup.ckpt" ] && echo "ERROR: $SUP_SRC/student_sup.ckpt not found." && exit 1

OUT="runs/mit-${MODEL}_ensemble_m${MEMBER_ID}_seed${SEED}_j${SLURM_JOB_ID}"

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

echo "[DONE] Ensemble member ${MEMBER_ID} (seed ${SEED}) complete.  Output: $OUT"
