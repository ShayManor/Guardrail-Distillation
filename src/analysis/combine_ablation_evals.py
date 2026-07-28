"""Combine scattered paper_eval_* ablation CSVs into a single aggregated dataset.

Scans `paper_eval_{city,acdc}_<backbone>_<guard_dir_basename>/csv/*.csv`,
infers `dataset` (city|acdc), `backbone` (b0|b1|b2), and `supervision_type`
(dense_multi|dense_disagree|dense_gap|scalar) from the directory name, and
writes one combined CSV per table (runs, per_image, per_class, risk_coverage,
teacher_budget, calibration_bins, confident_failures, latency_samples) into
an output directory.

Usage:
    python src/analysis/combine_ablation_evals.py \
        --repo-root . \
        --out src/analysis/combined_ablation/
"""

import argparse
import re
from pathlib import Path

import pandas as pd

TABLES = [
    "runs",
    "per_image",
    "per_class",
    "risk_coverage",
    "teacher_budget",
    "calibration_bins",
    "confident_failures",
    "latency_samples",
]

# paper_eval_{dataset}_{b0|b1|b2}_mit-{b0|b1|b2}_guard_{mode}_j{id}
DIR_RE = re.compile(
    r"^paper_eval_(?P<dataset>city|acdc|bdd|idd)_(?P<backbone>b[012])_"
    r"mit-(?P<backbone2>b[012])_guard_(?P<mode>[a-z_]+?)_j(?P<jobid>\d+)$"
)


def parse_dir(name: str):
    m = DIR_RE.match(name)
    if not m:
        return None
    return {
        "dataset": m.group("dataset"),
        "backbone": m.group("backbone"),
        "supervision_type": m.group("mode"),
        "guard_jobid": m.group("jobid"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    eval_dirs = sorted(args.repo_root.glob("paper_eval_*"))
    if not eval_dirs:
        raise SystemExit(f"No paper_eval_* directories under {args.repo_root.resolve()}")

    per_table: dict[str, list[pd.DataFrame]] = {t: [] for t in TABLES}
    skipped = []
    for d in eval_dirs:
        meta = parse_dir(d.name)
        if meta is None:
            skipped.append(d.name)
            continue
        csv_dir = d / "csv"
        if not csv_dir.is_dir():
            skipped.append(f"{d.name} (no csv/)")
            continue
        for table in TABLES:
            csv_path = csv_dir / f"{table}.csv"
            if not csv_path.is_file():
                continue
            try:
                df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                continue
            for k, v in meta.items():
                df[k] = v
            df["source_dir"] = d.name
            per_table[table].append(df)

    for table, frames in per_table.items():
        if not frames:
            print(f"[skip] {table}: no rows")
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)
        out_path = args.out / f"{table}.csv"
        combined.to_csv(out_path, index=False)
        print(f"[ok]   {table}: {len(combined):>7} rows -> {out_path}")

    if skipped:
        print("\n[warn] skipped dirs:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
