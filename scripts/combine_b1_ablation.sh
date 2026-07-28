#!/usr/bin/env bash
set -euo pipefail

cd "${REPO:-$HOME/Guardrail-Distillation}"

OUT=src/analysis/combined_ablation_b1
rm -rf "$OUT"
mkdir -p "$OUT"

python - "$OUT" <<'PY'
import sys, re, pathlib
import pandas as pd

out = pathlib.Path(sys.argv[1])
root = pathlib.Path(".")
tables = ["runs","per_image","per_class","risk_coverage","teacher_budget",
          "calibration_bins","confident_failures","latency_samples"]

dir_re = re.compile(
    r"^paper_eval_(?P<dataset>city|acdc)_(?P<backbone>b[012])_"
    r"mit-b[012]_guard_(?P<mode>.+)_j(?P<jobid>\d+)$"
)

dirs = []
for d in sorted(root.glob("paper_eval_*")):
    m = dir_re.match(d.name)
    if not m:
        continue
    dirs.append((d, m.groupdict()))

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
        df.insert(0, "dataset", meta["dataset"])
        df.insert(1, "backbone", meta["backbone"])
        df.insert(2, "supervision_type", meta["mode"])
        df.insert(3, "guard_jobid", meta["jobid"])
        df.insert(4, "source_dir", d.name)
        frames.append(df)
    if not frames:
        print(f"{table}: 0 rows (no matching files)")
        continue
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(out / f"{table}.csv", index=False)
    print(f"{table}: {len(combined)} rows, {combined.shape[1]} cols")
PY

git add "$OUT"
git diff --cached --quiet || git commit -m "analysis(b1): recombine E2 ablation eval CSVs with schema-aware concat"
git push
