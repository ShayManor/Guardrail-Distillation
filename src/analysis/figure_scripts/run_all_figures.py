"""Run every figure script's main() in-process and print a timing table.

Each module reads from combined_all/ (override with GD_COMBINED) and writes
into figures/ (override with GD_FIGURES). One bad figure won't stop the rest.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGURES = [
    ("fig_confident_failures",         "F1  — confident-failure AUROC vs msp threshold (all datasets)"),
    ("fig_scaling_across_backbones",   "F2  — scaling: AUROC vs backbone capacity (all datasets)"),
    ("fig_aurc_comparison",            "F3  — AURC bar chart per backbone (all datasets)"),
    ("fig_risk_coverage",              "F4  — risk-coverage curves (all datasets)"),
    ("fig_teacher_budget",             "F5  — teacher-routing Pareto (all datasets)"),
    ("fig_core_insight",               "F6  — student vs teacher risk scatter (ACDC + BDD)"),
    ("fig_per_condition_acdc",         "F7  — ACDC per-condition bars"),
    ("fig_latency",                    "F8  — latency ratios + absolute (all datasets)"),
    ("fig_supervision_ablation",       "F9  — E2 supervision-mode ablation (all datasets)"),
    ("fig_negative_result_decomp",     "F10 — benefit unpredictability (all datasets)"),
    ("fig_acdc_shift_panels",          "F11-14 — ACDC shift panels (mIoU, conf-fail bars, rho, scatter)"),
    ("fig_cross_dataset_headline",     "F15 — cross-dataset headline: CF-AUROC + AURC delta"),
    ("fig_bdd_threshold_sweep",        "F16 — threshold sweep: BDD + IDD + ACDC"),
    ("fig_cross_dataset_correlations", "F17 — negative result replicates across benchmarks"),
    ("fig_shift_severity_spectrum",    "F18 — shift severity vs guardrail advantage"),
]


def main():
    results = []
    for mod_name, description in FIGURES:
        t0 = time.perf_counter()
        status = "OK"
        err = ""
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
        except Exception as exc:  # noqa: BLE001
            status = "FAIL"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        dt = time.perf_counter() - t0
        results.append((mod_name, description, status, dt, err))

    print("\n" + "=" * 78)
    print("Figure generation summary")
    print("=" * 78)
    ok = sum(1 for _, _, s, _, _ in results if s == "OK")
    total = len(results)
    for name, desc, status, dt, err in results:
        marker = "✓" if status == "OK" else "✗"
        print(f"  {marker} {name:30s} {dt:6.2f}s   {desc}")
        if err:
            print(f"      {err}")
    print(f"\n{ok}/{total} figures generated")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
