#!/bin/bash -l
#SBATCH --job-name=eval-all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH -A lilly-rob1
#SBATCH --gres=gpu:1
#SBATCH -p smallgpu
#SBATCH -o /scratch/gautschi/manors/logs/eval-all_%j.out
#SBATCH -e /scratch/gautschi/manors/logs/eval-all_%j.err

set -euo pipefail
# Line-buffer stdout/stderr so SLURM logs show the last command before any crash.
exec > >(stdbuf -oL cat) 2> >(stdbuf -oL cat >&2)
trap 'rc=$?; echo "FAILED rc=$rc at line $LINENO: $BASH_COMMAND" >&2' ERR
set -x
source /scratch/gautschi/manors/venvs/guardrail/bin/activate
set +x

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export REPO=/home/manors/Guardrail-Distillation
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
unset DISPLAY
cd "$REPO"

FAIL_LOG="/scratch/gautschi/manors/logs/eval-all_${SLURM_JOB_ID:-local}_failures.log"
: > "$FAIL_LOG"
echo "[fail-log] $FAIL_LOG"

# Run a single eval invocation without aborting the whole job on failure.
# Logs the failing label + rc to $FAIL_LOG and returns 0 so the caller can move on.
try_eval () {
  local label="$1"; shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "[FAIL] $label rc=$rc" | tee -a "$FAIL_LOG" >&2
    return 0
  fi
  echo "[ok  ] $label"
}

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

# Discover every trained guardrail checkpoint on disk and dedupe by
# (backbone, mode, seed). The multi-seed sbatches emit dirs named:
#     runs/mit-b1_guard_dense_multi_s137_j9275012/
# Single-seed sbatches emit:
#     runs/mit-b1_guard_dense_multi_j9249359/          (-> seed=42 by default)
# Legacy dirs:
#     runs/mit-b0_guard_j9249359/                      (-> mode=unknown, seed=42)
#
# For each (backbone, mode, seed), keep the newest by mtime so a resubmit
# shadows the prior run automatically.
declare -a JOBS
mapfile -t JOBS < <(python3 - <<'PY'
import os, re, glob
pat = re.compile(
    r"^runs/mit-(?P<bb>b[012])_guard"
    r"(?:_(?P<mode>dense_multi|dense_gap|dense_disagree|scalar_benefit))?"
    r"(?:_s(?P<seed>\d+))?"
    r"_j(?P<jobid>\d+)$"
)
candidates = []
for d in glob.glob("runs/mit-b*_guard*"):
    if not os.path.isfile(os.path.join(d, "guardrail.ckpt")):
        continue
    m = pat.match(d)
    if not m:
        continue
    # Skip legacy pre-refactor checkpoints (no supervision mode in the dir
    # name). Their state_dicts still have the removed margin_head/family_head
    # keys and fail strict load in full_eval.py::build_guardrail_model.
    if m.group("mode") is None:
        continue
    candidates.append((os.path.getmtime(d), d, m.groupdict()))
candidates.sort(reverse=True)  # newest first

seen = set()
for _, d, g in candidates:
    bb   = g["bb"]
    mode = g["mode"] or "unknown"
    seed = g["seed"] or "42"
    key = (bb, mode, seed)
    if key in seen:
        continue
    seen.add(key)
    print(f"{bb}|{mode}|{seed}|{d}")
PY
)

if [ ${#JOBS[@]} -eq 0 ]; then
  echo "ERROR: no guardrail checkpoints found" && exit 1
fi

echo "[discover] ${#JOBS[@]} guard checkpoints:"
for J in "${JOBS[@]}"; do echo "  $J"; done

run_one () {
  local DATASET="$1" BB="$2" MODE="$3" SEED="$4" GDIR="$5"
  local MIT_STUDENT SKD_DIR STUDENT_CKPT GUARDRAIL_CKPT EVAL_DIR DOMAIN TAG
  MIT_STUDENT=$(find_mit "$BB" || true)
  SKD_DIR=$(find_skd "$BB" || true)
  if [ -z "$MIT_STUDENT" ]; then
    echo "[skip] no mit-$BB backbone dir resolved on this node" >&2
    return 0
  fi
  if [ -z "$SKD_DIR" ]; then
    echo "[skip] no SKD dir for $BB" >&2
    return 0
  fi
  STUDENT_CKPT="$SKD_DIR/student_skd.ckpt"
  GUARDRAIL_CKPT="$GDIR/guardrail.ckpt"
  if [ ! -f "$STUDENT_CKPT" ]; then
    echo "[skip] $STUDENT_CKPT missing" >&2
    return 0
  fi
  TAG="${MODE}_s${SEED}"
  EVAL_DIR="$REPO/paper_eval_${DATASET}_${BB}_${GDIR##*/}"

  # Idempotence: if this guard dir has already been evaluated (runs.csv
  # present) skip unless the caller forces a rerun. Lets us add a new seed
  # without re-running 20+ existing evals.
  if [ -f "$EVAL_DIR/csv/runs.csv" ] && [ -z "${FORCE_EVAL:-}" ]; then
    echo "[skip] $DATASET $BB/$TAG already evaluated at $EVAL_DIR (set FORCE_EVAL=1 to redo)"
    return 0
  fi

  if [ "$DATASET" = "city" ]; then
    echo "============================================"
    echo "[CS  ] $BB/$TAG   out=$EVAL_DIR"
    echo "============================================"
    try_eval "city $BB/$TAG" env HF_HOME=/scratch/gautschi/manors/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "city_${BB}_${TAG}" \
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
  elif [ "$DATASET" = "acdc" ]; then
    for DOMAIN in fog night rain snow all; do
      echo "============================================"
      echo "[ACDC] $BB/$TAG domain=$DOMAIN   out=$EVAL_DIR"
      echo "============================================"
      try_eval "acdc $BB/$TAG/$DOMAIN" env HF_HOME=/scratch/gautschi/manors/hf_cache \
      python -u src/eval/full_eval.py eval \
        --run-id "acdc_${BB}_${TAG}_${DOMAIN}" \
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
  else  # idd
    if [ ! -d "$REPO/data/idd/leftImg8bit/val" ]; then
      echo "[skip] $DATASET $BB/$TAG — $REPO/data/idd not staged yet"
      return 0
    fi
    echo "============================================"
    echo "[IDD ] $BB/$TAG   out=$EVAL_DIR"
    echo "============================================"
    try_eval "idd $BB/$TAG" env HF_HOME=/scratch/gautschi/manors/hf_cache \
    python -u src/eval/full_eval.py eval \
      --run-id "idd_${BB}_${TAG}" \
      --dataset-name idd \
      --dataset-path "$REPO/data/idd" \
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
  fi
}

for J in "${JOBS[@]}"; do
  IFS='|' read -r BB MODE SEED DIR <<< "$J"
  run_one city "$BB" "$MODE" "$SEED" "$DIR"
  run_one acdc "$BB" "$MODE" "$SEED" "$DIR"
  run_one idd  "$BB" "$MODE" "$SEED" "$DIR"
done

echo ""
echo "============================================"
if [ -s "$FAIL_LOG" ]; then
  echo "[failures] $(wc -l < "$FAIL_LOG") run(s) failed — see $FAIL_LOG"
  sed 's/^/  /' "$FAIL_LOG"
else
  echo "[failures] none"
fi
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

# Accept legacy (no mode, no seed), single-seed (mode but no seed), and
# multi-seed (mode + seed) dir shapes. Anchor mode to the known vocabulary
# so "dense_multi_s137" isn't swallowed as mode="dense_multi_s137".
dir_re = re.compile(
    r"^paper_eval_(?P<dataset>city|acdc|idd)_(?P<backbone>b[012])_"
    r"mit-b[012]_guard"
    r"(?:_(?P<mode>dense_multi|dense_gap|dense_disagree|scalar_benefit))?"
    r"(?:_s(?P<seed>\d+))?"
    r"_j(?P<jobid>\d+)$"
)

dirs = []
for d in sorted(root.glob("paper_eval_*")):
    m = dir_re.match(d.name)
    if not m:
        print(f"[warn] skipping unparseable dir: {d.name}")
        continue
    meta = m.groupdict()
    if meta["mode"] is None:
        # Legacy pre-refactor dir — skip entirely.
        print(f"[skip legacy] {d.name}")
        continue
    if meta["seed"] is None:
        meta["seed"] = "42"  # single-seed default for named-mode dirs
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
        # Prefer the supervision_type logged inside runs.csv when available
        # (legacy b0/b2 dirs have mode="unknown" from the regex). Never let
        # this overwrite a seed parsed from the dir name.
        local_meta = dict(meta)
        if table != "runs":
            runs_csv = d / "csv" / "runs.csv"
            if runs_csv.is_file():
                try:
                    rdf = pd.read_csv(runs_csv)
                    if "guardrail_supervision_type" in rdf.columns and len(rdf):
                        val = str(rdf["guardrail_supervision_type"].iloc[0]).strip()
                        if val:
                            local_meta["mode"] = val
                except Exception:
                    pass
        # Drop any columns we're about to inject so .insert() doesn't collide
        # with pre-existing ones written by full_eval.py (e.g. runs.csv already
        # has `seed`, `dataset`, `backbone`).
        injected = ["dataset", "backbone", "supervision_type",
                    "seed", "guard_jobid", "source_dir"]
        df = df.drop(columns=[c for c in injected if c in df.columns])
        df.insert(0, "dataset", local_meta["dataset"])
        df.insert(1, "backbone", local_meta["backbone"])
        df.insert(2, "supervision_type", local_meta["mode"])
        df.insert(3, "seed", int(local_meta["seed"]))
        df.insert(4, "guard_jobid", local_meta["jobid"])
        df.insert(5, "source_dir", d.name)
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
