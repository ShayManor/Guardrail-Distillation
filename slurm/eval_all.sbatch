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

find_mit () {
  local bb="$1"
  find /scratch/gautschi/manors ~/.cache -path "*models--nvidia--mit-${bb}/snapshots/*" -type d 2>/dev/null | head -n1
}

find_skd () {
  local bb="$1"
  ls -td runs/mit-${bb}_skd_*/student_skd.ckpt outputs*/student_skd.ckpt 2>/dev/null \
    | head -1 | xargs -r dirname
}

# Collect (backbone, label, guard_dir) triples.
# b0 / b2: newest guardrail run regardless of naming (legacy names like
#          mit-b0_guard_j<id> OR new names like mit-b0_guard_dense_multi_j<id>).
# b1: one entry per supervision mode that has a checkpoint on disk.
declare -a JOBS

add_job () {
  local bb="$1" label="$2" dir="$3"
  [ -z "$dir" ] && return 0
  [ ! -f "$dir/guardrail.ckpt" ] && return 0
  JOBS+=("${bb}|${label}|${dir}")
  echo "[auto] ${bb}/${label} -> $dir"
}

# b0 and b2: newest guardrail ckpt (any mode/seed label). The glob
# `runs/mit-b0_guard_*/guardrail.ckpt` matches legacy names like
# `mit-b0_guard_j9249359`, new names like `mit-b0_guard_dense_multi_s42_j9300000`,
# and seeded multi-seed names.
for BB in b0 b2; do
  NEW=$(ls -td runs/mit-${BB}_guard_*/guardrail.ckpt 2>/dev/null | head -1 | xargs -r dirname || true)
  add_job "$BB" "latest" "$NEW"
done

# b1: one newest per supervision mode. The `${MODE}_*` glob matches both
# `${MODE}_j<id>` (legacy) and `${MODE}_s<seed>_j<id>` (seeded) forms.
for MODE in dense_multi dense_gap dense_disagree scalar_benefit; do
  D=$(ls -td runs/mit-b1_guard_${MODE}_*/guardrail.ckpt 2>/dev/null | head -1 | xargs -r dirname || true)
  add_job "b1" "$MODE" "$D"
done

if [ ${#JOBS[@]} -eq 0 ]; then
  echo "ERROR: no guardrail checkpoints found" && exit 1
fi

run_one () {
  local DATASET="$1" BB="$2" LABEL="$3" GDIR="$4"
  local MIT_STUDENT SKD_DIR STUDENT_CKPT GUARDRAIL_CKPT EVAL_DIR DOMAIN
  MIT_STUDENT=$(find_mit "$BB")
  SKD_DIR=$(find_skd "$BB")
  [ -z "$SKD_DIR" ] && echo "[skip] no SKD for $BB" && return 0
  STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
  GUARDRAIL_CKPT="$GDIR/guardrail.ckpt"
  [ ! -f "$STUDENT_CKPT" ] && echo "[skip] $STUDENT_CKPT missing" && return 0
  EVAL_DIR="$REPO/paper_eval_${DATASET}_${BB}_${GDIR##*/}"

  if [ "$DATASET" = "city" ]; then
    echo "============================================"
    echo "[CS  ] $BB/$LABEL   out=$EVAL_DIR"
    echo "============================================"
    HF_HOME=/scratch/gautschi/manors/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "city_${BB}_${LABEL}" \
      --dataset-name cityscapes \
      --dataset-path "$REPO/data/cityscapes" \
      --split val \
      --student-name student_skd \
      --train-method skd \
      --student-backbone "$MIT_STUDENT" \
      --student-ckpt "$STUDENT_CKPT" \
      --teacher-backbone "$TEACHER" \
      --guardrail-ckpt "$GUARDRAIL_CKPT" \
      --guardrail-student-name student_skd \
      --mc-dropout-passes 4 \
      --output-dir "$EVAL_DIR" \
      --batch-size 4 \
      --num-workers 4
  else
    for DOMAIN in fog night rain snow all; do
      echo "============================================"
      echo "[ACDC] $BB/$LABEL domain=$DOMAIN   out=$EVAL_DIR"
      echo "============================================"
      HF_HOME=/scratch/gautschi/manors/hf_cache \
      python -u src/eval/full_eval.py eval \
        --run-id "acdc_${BB}_${LABEL}_${DOMAIN}" \
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
        --output-dir "$EVAL_DIR" \
        --batch-size 4 \
        --num-workers 4
    done
  fi
}

for J in "${JOBS[@]}"; do
  IFS='|' read -r BB LABEL DIR <<< "$J"
  run_one city "$BB" "$LABEL" "$DIR"
  run_one acdc "$BB" "$LABEL" "$DIR"
done

echo ""
echo "============================================"
echo "[combine] merging all paper_eval_* CSVs"
echo "============================================"

OUT=src/analysis/combined_all
rm -rf "$OUT" \
       src/analysis/cs_b0_b2_eval \
       src/analysis/acdc_b0_b2_eval \
       src/analysis/combined_ablation_b1
mkdir -p "$OUT"

python - "$OUT" <<'PY'
import sys, re, pathlib
import pandas as pd

out = pathlib.Path(sys.argv[1])
root = pathlib.Path(".")
tables = ["runs","per_image","per_class","risk_coverage","teacher_budget",
          "calibration_bins","confident_failures","latency_samples"]

# Accept both legacy (no mode in name) and new (with mode) dir shapes.
dir_re = re.compile(
    r"^paper_eval_(?P<dataset>city|acdc)_(?P<backbone>b[012])_"
    r"mit-b[012]_guard(?:_(?P<mode>.+?))?_j(?P<jobid>\d+)$"
)

dirs = []
for d in sorted(root.glob("paper_eval_*")):
    m = dir_re.match(d.name)
    if not m:
        continue
    meta = m.groupdict()
    if meta["mode"] is None:
        meta["mode"] = "unknown"
    dirs.append((d, meta))

for table in tables:
    frames = []
    for d, meta in dirs:
        csv = d / "csv" / f"{table}.csv"
        if not csv.is_file():
            continue
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            continue
        # Prefer the supervision_type logged inside runs.csv when we have it,
        # because the dir name for legacy b0/b2 runs is "unknown".
        if table != "runs":
            runs_csv = d / "csv" / "runs.csv"
            if runs_csv.is_file():
                try:
                    rdf = pd.read_csv(runs_csv)
                    if "guardrail_supervision_type" in rdf.columns and len(rdf):
                        val = str(rdf["guardrail_supervision_type"].iloc[0]).strip()
                        if val:
                            meta["mode"] = val
                except Exception:
                    pass
        df.insert(0, "dataset", meta["dataset"])
        df.insert(1, "backbone", meta["backbone"])
        df.insert(2, "supervision_type", meta["mode"])
        df.insert(3, "guard_jobid", meta["jobid"])
        df.insert(4, "source_dir", d.name)
        frames.append(df)
    if not frames:
        print(f"{table}: 0 rows")
        continue
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out / f"{table}.csv", index=False)
    print(f"{table}: {len(combined)} rows, {combined.shape[1]} cols")
PY

echo "[DONE] All runs complete. Combined CSVs under src/analysis/combined_all/"

echo ""
echo "============================================"
echo "[git] committing + pushing combined_all/"
echo "============================================"
cd "$REPO"
git add -A src/analysis/combined_all \
           src/analysis/cs_b0_b2_eval \
           src/analysis/acdc_b0_b2_eval \
           src/analysis/combined_ablation_b1 2>/dev/null || true
if git diff --cached --quiet; then
  echo "[git] no changes to commit"
else
  STAMP=$(date -u +%Y-%m-%dT%H:%MZ)
  git -c user.name="cluster-bot" -c user.email="cluster@local" \
      commit -m "analysis: refresh combined_all/ ($STAMP, job ${SLURM_JOB_ID:-local})"
  git push origin HEAD || echo "[git] push failed (non-fatal)"
fi
