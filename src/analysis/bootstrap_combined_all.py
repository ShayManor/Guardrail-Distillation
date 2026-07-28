"""Dev-only: synthesize `src/analysis/combined_all/` from existing CSVs so the
figure scripts can iterate locally without waiting for the cluster re-eval.

Pulls:
  - b0/b2 rows from `acdc_b0_b2_eval/csv/` and `cs_b0_b2_eval/csv/`
  - b1 rows from `combined_ablation_b1/` (has all 4 supervision modes)

Injects the 5 prefix columns that `eval_all.sbatch`'s combine step adds
(`dataset`, `backbone`, `supervision_type`, `guard_jobid`, `source_dir`) plus
a `seed` column (always 42 for existing data). Once the cluster produces the
real combined_all/ dir, this script is obsolete but harmless.

Run once:
    python src/analysis/bootstrap_combined_all.py
"""

from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "combined_all"
OUT.mkdir(exist_ok=True)

TABLES = ["runs","per_image","per_class","risk_coverage","teacher_budget",
          "calibration_bins","confident_failures","latency_samples"]

def backbone_of(run_id):
    m = re.search(r"\bb([012])\b|_(b[012])[_\b]|(b[012])_", str(run_id))
    if m:
        for g in m.groups():
            if g:
                return g if g.startswith("b") else f"b{g}"
    for tok in str(run_id).split("_"):
        if tok in ("b0","b1","b2"):
            return tok
    return "?"

def load_legacy(base, dataset_name):
    out = {}
    for t in TABLES:
        p = base / f"{t}.csv"
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if "run_id" not in df.columns:
            continue
        df["backbone"] = df["run_id"].map(backbone_of)
        df = df[df["backbone"].isin(["b0","b2"])]  # b1 comes from combined_ablation_b1
        if df.empty:
            continue
        df.insert(0, "dataset", dataset_name)
        df.insert(1, "backbone", df.pop("backbone"))
        df.insert(2, "supervision_type", "dense_multi")
        df.insert(3, "guard_jobid", "legacy")
        df.insert(4, "source_dir", base.parent.name)
        df["seed"] = 42
        out[t] = df
    return out

def load_b1(base):
    out = {}
    for t in TABLES:
        p = base / f"{t}.csv"
        if not p.is_file():
            continue
        try:
            df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            continue
        df = df[df.get("backbone", "b1") == "b1"] if "backbone" in df.columns else df.assign(backbone="b1")
        if df.empty:
            continue
        if "dataset" not in df.columns:
            df.insert(0, "dataset", "city")  # fallback; shouldn't trigger for combined_ablation_b1
        if "seed" not in df.columns:
            df["seed"] = 42
        out[t] = df
    return out

def main():
    acdc_b02 = load_legacy(ROOT / "acdc_b0_b2_eval" / "csv", "acdc")
    cs_b02 = load_legacy(ROOT / "cs_b0_b2_eval" / "csv", "city")
    b1 = load_b1(ROOT / "combined_ablation_b1")

    for t in TABLES:
        frames = []
        for src_name, src in (("acdc_b02", acdc_b02), ("cs_b02", cs_b02), ("b1", b1)):
            if t in src:
                frames.append(src[t])
        if not frames:
            print(f"[skip] {t}: no rows from any source")
            continue
        combined = pd.concat(frames, ignore_index=True, sort=False)

        if "supervision_type" in combined.columns:
            combined["supervision_type"] = (
                combined["supervision_type"].fillna("dense_multi").replace({"": "dense_multi"})
            )
        if "dataset" in combined.columns:
            combined["dataset"] = combined["dataset"].replace({
                "cityscapes": "city", "cs": "city", "Cityscapes": "city",
            })
        if "backbone" in combined.columns:
            combined = combined[combined["backbone"].isin(["b0","b1","b2"])]

        combined.to_csv(OUT / f"{t}.csv", index=False)
        print(f"{t:22s} {len(combined):>7d} rows  {combined.shape[1]:>4d} cols")

if __name__ == "__main__":
    main()
