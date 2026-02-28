"""
Unified guardrail benchmark.

For EVERY student checkpoint, compute AURC using:
    1. MSP (mean max softmax prob)
    2. Entropy (mean prediction entropy)
    3. MC Dropout (if enabled)
    4. Guardrail risk score (only for the student the guardrail was trained on)

Also computes teacher oracle gap for reference.
Outputs: comparison table, risk-coverage curves, calibration plots.
"""

import csv
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IGNORE_INDEX = 255


# ─── Core metrics ─────────────────────────────────────────────────────────────


def compute_aurc(risks, scores):
    """AURC: sort by descending confidence, accumulate risk. Lower = better."""
    order = np.argsort(-scores)
    risks_sorted = risks[order]
    n = len(risks)
    coverages = np.arange(1, n + 1) / n
    cum_risk = np.cumsum(risks_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(cum_risk, coverages))
    return aurc, coverages, cum_risk


def compute_risk_at_coverage(risks, scores, target_coverage=0.8):
    order = np.argsort(-scores)
    risks_sorted = risks[order]
    k = max(1, int(len(risks) * target_coverage))
    return float(risks_sorted[:k].mean())


def image_miou(pred, gt, num_classes):
    valid = gt != IGNORE_INDEX
    if valid.sum() == 0:
        return 0.0
    pred_v, gt_v = pred[valid], gt[valid]
    ious = []
    for c in range(num_classes):
        p_c, l_c = pred_v == c, gt_v == c
        inter = (p_c & l_c).sum().item()
        union = (p_c | l_c).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


# ─── Per-image extraction ─────────────────────────────────────────────────────


@torch.no_grad()
def extract_image_scores(model, dataloader, num_classes, device,
                         teacher=None, guardrail=None, use_student_features=False,
                         mc_dropout_passes=0):
    """
    Run model on val set. For each image, extract:
        - mIoU, risk
        - MSP confidence, entropy confidence
        - (optional) MC dropout uncertainty
        - (optional) guardrail risk score
        - (optional) oracle teacher gap
    """
    model.to(device).eval()
    if teacher is not None:
        teacher.to(device).eval()
    if guardrail is not None:
        guardrail.to(device).eval()

    records = []

    for imgs, lbls in dataloader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        # Student forward
        if use_student_features and guardrail is not None:
            student_logits, student_feat = model(imgs, return_features=True)
        else:
            student_logits = model(imgs)
            student_feat = None

        student_probs = F.softmax(student_logits, dim=1)
        student_preds = student_logits.argmax(dim=1)

        # Teacher forward (if available)
        teacher_preds = None
        if teacher is not None:
            teacher_logits = teacher(imgs)
            teacher_preds = teacher_logits.argmax(dim=1)

        # Guardrail forward (if available)
        guard_out = None
        if guardrail is not None:
            guard_out = guardrail(student_logits, student_feat)

        # MC Dropout (if requested)
        mc_entropy = None
        if mc_dropout_passes > 0:
            model.train()  # enable dropout
            mc_logits = []
            for _ in range(mc_dropout_passes):
                mc_out = model(imgs)
                mc_logits.append(F.softmax(mc_out, dim=1))
            model.eval()
            mc_mean = torch.stack(mc_logits).mean(dim=0)
            eps = 1e-8
            mc_ent = -(mc_mean * (mc_mean + eps).log()).sum(dim=1)  # (B, H, W)
            mc_entropy = []
            for i in range(imgs.shape[0]):
                valid = lbls[i] != IGNORE_INDEX
                mc_entropy.append(mc_ent[i][valid].mean().item() if valid.any() else 0.0)

        # Per-image records
        for i in range(imgs.shape[0]):
            lbl = lbls[i]
            valid = lbl != IGNORE_INDEX
            if valid.sum() == 0:
                continue

            miou = image_miou(student_preds[i], lbl, num_classes)
            risk = 1.0 - miou
            probs_i = student_probs[i][:, valid]

            # MSP: higher = more confident
            msp = probs_i.max(dim=0).values.mean().item()

            # Entropy: lower = more confident
            eps = 1e-8
            entropy = -(probs_i * (probs_i + eps).log()).sum(dim=0).mean().item()

            rec = {"miou": miou, "risk": risk, "msp": msp, "entropy": entropy}

            if mc_entropy is not None:
                rec["mc_entropy"] = mc_entropy[i]

            if guard_out is not None:
                rec["guardrail_risk"] = guard_out["risk_score"][i].item()
                if "gap_heatmap" in guard_out:
                    hm = guard_out["gap_heatmap"][i]
                    rec["guardrail_gap_mean"] = hm[valid].mean().item() if valid.any() else 0.0

            if teacher_preds is not None:
                t_right_s_wrong = (
                    (teacher_preds[i][valid] == lbl[valid]) &
                    (student_preds[i][valid] != lbl[valid])
                ).float().mean().item()
                rec["oracle_gap"] = t_right_s_wrong

            records.append(rec)

        if len(records) % 50 < imgs.shape[0]:
            print(f"    [{len(records)}] images processed...")

    return records


# ─── Main benchmark ───────────────────────────────────────────────────────────


def run_benchmark(
    students: dict[str, nn.Module],
    val_loader,
    num_classes: int = 19,
    device: str = "cuda",
    teacher: Optional[nn.Module] = None,
    guardrail: Optional[nn.Module] = None,
    guardrail_student_name: Optional[str] = None,
    use_student_features: bool = False,
    mc_dropout_passes: int = 0,
    save_dir: str = "results/benchmark",
):
    """
    Unified benchmark across all student checkpoints.

    Args:
        students: {"student_sup": model, "student_kd": model, ...}
        val_loader: validation DataLoader
        teacher: teacher model (for oracle gap)
        guardrail: GuardrailHead (applied only to guardrail_student_name)
        guardrail_student_name: which student the guardrail was trained on
        mc_dropout_passes: 0 = skip MC dropout
        save_dir: output directory
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {student_name: [records]}
    all_summary = []  # flat list for CSV

    for name, model in students.items():
        print(f"\n[Benchmark] Evaluating: {name}")

        # Only attach guardrail to the student it was trained on
        g = guardrail if (guardrail is not None and name == guardrail_student_name) else None

        records = extract_image_scores(
            model, val_loader, num_classes, device,
            teacher=teacher, guardrail=g,
            use_student_features=(use_student_features and g is not None),
            mc_dropout_passes=mc_dropout_passes,
        )
        all_results[name] = records

        # Save per-image CSV
        csv_path = save_path / f"{name}_scores.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

        # Compute AURC for each uncertainty method
        risks = np.array([r["risk"] for r in records])
        mean_miou = np.mean([r["miou"] for r in records])

        methods = {
            "MSP": np.array([r["msp"] for r in records]),
            "Neg-Entropy": -np.array([r["entropy"] for r in records]),
        }
        if mc_dropout_passes > 0 and "mc_entropy" in records[0]:
            methods["MC-Dropout"] = -np.array([r["mc_entropy"] for r in records])
        if "guardrail_risk" in records[0]:
            methods["Guardrail"] = 1.0 - np.array([r["guardrail_risk"] for r in records])
        if "oracle_gap" in records[0]:
            methods["Oracle"] = 1.0 - np.array([r["oracle_gap"] for r in records])

        for method_name, scores in methods.items():
            aurc, _, _ = compute_aurc(risks, scores)
            row = {
                "student": name,
                "method": method_name,
                "miou": mean_miou,
                "aurc": aurc,
                "risk_at_80": compute_risk_at_coverage(risks, scores, 0.8),
                "risk_at_90": compute_risk_at_coverage(risks, scores, 0.9),
                "risk_at_95": compute_risk_at_coverage(risks, scores, 0.95),
            }
            all_summary.append(row)

    # ── Save summary CSV ──
    summary_csv = save_path / "benchmark_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summary[0].keys())
        writer.writeheader()
        writer.writerows(all_summary)

    # ── Print summary table ──
    print(f"\n{'=' * 80}")
    print("UNIFIED BENCHMARK RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Student':<18} {'Method':<16} {'mIoU':>7} {'AURC':>8} {'R@80%':>8} {'R@90%':>8} {'R@95%':>8}")
    print("-" * 83)
    for row in all_summary:
        print(
            f"{row['student']:<18} {row['method']:<16} "
            f"{row['miou']:>7.4f} {row['aurc']:>8.4f} "
            f"{row['risk_at_80']:>8.4f} {row['risk_at_90']:>8.4f} {row['risk_at_95']:>8.4f}"
        )
    print(f"\n[csv] {summary_csv}")

    # ── Plots ──
    _plot_benchmark(all_results, all_summary, save_path)

    return all_summary


def _plot_benchmark(all_results, all_summary, save_path):
    """Generate comparison plots."""
    import pandas as pd
    df = pd.DataFrame(all_summary)
    student_names = df["student"].unique()
    n_students = len(student_names)

    # ── Figure 1: Risk-Coverage per student ──
    fig, axes = plt.subplots(1, n_students, figsize=(6 * n_students, 5), squeeze=False)
    fig.suptitle("Risk-Coverage Curves by Student", fontsize=14, fontweight="bold")

    for idx, sname in enumerate(student_names):
        ax = axes[0, idx]
        records = all_results[sname]
        risks = np.array([r["risk"] for r in records])

        methods = {"MSP": np.array([r["msp"] for r in records]),
                    "Neg-Entropy": -np.array([r["entropy"] for r in records])}
        if "guardrail_risk" in records[0]:
            methods["Guardrail"] = 1.0 - np.array([r["guardrail_risk"] for r in records])
        if "oracle_gap" in records[0]:
            methods["Oracle"] = 1.0 - np.array([r["oracle_gap"] for r in records])
        if "mc_entropy" in records[0]:
            methods["MC-Dropout"] = -np.array([r["mc_entropy"] for r in records])

        for mname, scores in methods.items():
            aurc, covs, cum_risk = compute_aurc(risks, scores)
            style = "--" if mname == "Oracle" else "-"
            ax.plot(covs, cum_risk, style, label=f"{mname} ({aurc:.4f})")

        ax.set(xlabel="Coverage", ylabel="Cumulative Risk", title=sname)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path / "risk_coverage_per_student.png", dpi=150, bbox_inches="tight")
    print(f"[plot] {save_path / 'risk_coverage_per_student.png'}")
    plt.close(fig)

    # ── Figure 2: AURC comparison bar chart ──
    fig, ax = plt.subplots(figsize=(10, 5))
    methods_all = df["method"].unique()
    x = np.arange(len(student_names))
    width = 0.8 / len(methods_all)

    for i, method in enumerate(methods_all):
        sub = df[df["method"] == method]
        aurcs = [sub[sub["student"] == s]["aurc"].values[0] if s in sub["student"].values else 0
                 for s in student_names]
        ax.bar(x + i * width, aurcs, width, label=method, alpha=0.85)

    ax.set_xticks(x + width * len(methods_all) / 2)
    ax.set_xticklabels(student_names, rotation=15)
    ax.set(ylabel="AURC (lower = better)", title="AURC Comparison Across Students & Methods")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path / "aurc_comparison.png", dpi=150, bbox_inches="tight")
    print(f"[plot] {save_path / 'aurc_comparison.png'}")
    plt.close(fig)

    # ── Figure 3: Guardrail calibration ──
    for sname, records in all_results.items():
        if "guardrail_risk" not in records[0]:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        pred_risk = [r["guardrail_risk"] for r in records]
        actual_risk = [r["risk"] for r in records]
        ax.scatter(pred_risk, actual_risk, alpha=0.3, s=10, c="steelblue")
        ax.plot([0, max(actual_risk)], [0, max(actual_risk)], "r--", alpha=0.5, label="Perfect")
        corr = np.corrcoef(pred_risk, actual_risk)[0, 1]
        ax.set(xlabel="Predicted Risk (Guardrail)", ylabel="Actual Risk (1 - mIoU)",
               title=f"Guardrail Calibration ({sname})\nPearson r = {corr:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path / f"calibration_{sname}.png", dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path / f'calibration_{sname}.png'}")
        plt.close(fig)