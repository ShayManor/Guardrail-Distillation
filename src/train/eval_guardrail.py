"""
Unified guardrail benchmark.

For EVERY student checkpoint, compute AURC using:
    1. MSP (mean max softmax prob)
    2. Entropy (mean prediction entropy)
    3. MC Dropout (if enabled)
    4. Guardrail risk score (only for the student the guardrail was trained on)

Also computes teacher oracle gap for reference.

Additional analyses:
    - Confident failure detection (AUROC among high-MSP images)
    - Selective prediction gain (guardrail vs MSP risk reduction at each coverage)
    - Per-class failure detection breakdown
    - "Money plot": MSP vs Risk colored by guardrail score

Outputs: comparison table, risk-coverage curves, calibration plots,
         confident failure analysis, gain curves.
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


def per_class_iou(pred, gt, num_classes):
    """Return dict {class_id: iou} for classes present in gt."""
    valid = gt != IGNORE_INDEX
    if valid.sum() == 0:
        return {}
    pred_v, gt_v = pred[valid], gt[valid]
    result = {}
    for c in range(num_classes):
        p_c, l_c = pred_v == c, gt_v == c
        inter = (p_c & l_c).sum().item()
        union = (p_c | l_c).sum().item()
        if union > 0:
            result[c] = inter / union
    return result


# ─── Confident failure detection ──────────────────────────────────────────────


def compute_confident_failure_detection(records, msp_thresholds=None):
    """
    Among images where student is confident (MSP > threshold),
    how well does each method detect actual failures?

    This is THE key metric: MSP can't distinguish these by definition
    (all above threshold), but guardrail/oracle can.
    """
    if msp_thresholds is None:
        msp_thresholds = [0.85, 0.90, 0.92, 0.95]

    results = {}
    median_risk = np.median([r["risk"] for r in records])

    for thresh in msp_thresholds:
        confident = [r for r in records if r["msp"] > thresh]
        if len(confident) < 10:
            continue

        labels = np.array([r["risk"] > median_risk for r in confident], dtype=float)
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        from sklearn.metrics import roc_auc_score, average_precision_score

        row = {
            "n_confident": len(confident),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "mean_risk": float(np.mean([r["risk"] for r in confident])),
        }

        msp_scores = np.array([-r["msp"] for r in confident])
        try:
            row["MSP_AUROC"] = roc_auc_score(labels, msp_scores)
            row["MSP_AP"] = average_precision_score(labels, msp_scores)
        except ValueError:
            row["MSP_AUROC"] = 0.5
            row["MSP_AP"] = labels.mean()

        ent_scores = np.array([r["entropy"] for r in confident])
        try:
            row["Entropy_AUROC"] = roc_auc_score(labels, ent_scores)
            row["Entropy_AP"] = average_precision_score(labels, ent_scores)
        except ValueError:
            row["Entropy_AUROC"] = 0.5
            row["Entropy_AP"] = labels.mean()

        if "guardrail_risk" in confident[0]:
            guard_scores = np.array([r["guardrail_risk"] for r in confident])
            try:
                row["Guardrail_AUROC"] = roc_auc_score(labels, guard_scores)
                row["Guardrail_AP"] = average_precision_score(labels, guard_scores)
            except ValueError:
                row["Guardrail_AUROC"] = 0.5
                row["Guardrail_AP"] = labels.mean()

        if "oracle_gap" in confident[0]:
            oracle_scores = np.array([r["oracle_gap"] for r in confident])
            try:
                row["Oracle_AUROC"] = roc_auc_score(labels, oracle_scores)
                row["Oracle_AP"] = average_precision_score(labels, oracle_scores)
            except ValueError:
                row["Oracle_AUROC"] = 0.5
                row["Oracle_AP"] = labels.mean()

        results[thresh] = row

    return results


# ─── Selective prediction gain ────────────────────────────────────────────────


def compute_selective_gain(risks, msp_scores, other_scores, method_name="Guardrail"):
    """
    At each coverage level, compute risk reduction of method vs MSP.
    Positive = method is better (lower risk).
    """
    coverages = np.linspace(0.1, 1.0, 50)
    gains = []

    for cov in coverages:
        k = max(1, int(len(risks) * cov))

        msp_order = np.argsort(-msp_scores)
        msp_risk = risks[msp_order[:k]].mean()

        other_order = np.argsort(-other_scores)
        other_risk = risks[other_order[:k]].mean()

        gains.append({
            "coverage": cov,
            "msp_risk": msp_risk,
            f"{method_name}_risk": other_risk,
            "gain": msp_risk - other_risk,
            "relative_gain_pct": 100 * (msp_risk - other_risk) / max(msp_risk, 1e-8),
        })

    return gains


# ─── Per-image extraction ─────────────────────────────────────────────────────


@torch.no_grad()
def extract_image_scores(model, dataloader, num_classes, device,
                         teacher=None, guardrail=None, use_student_features=False,
                         mc_dropout_passes=0):
    """
    Run model on val set. For each image, extract:
        - mIoU, risk, per-class IoU
        - MSP confidence, entropy confidence, pixel-level stats
        - (optional) MC dropout uncertainty
        - (optional) guardrail risk score + pixel-level gap stats
        - (optional) oracle teacher gap + confident-wrong-teacher-right rate
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

        # Teacher forward
        teacher_preds = None
        if teacher is not None:
            teacher_logits = teacher(imgs)
            teacher_preds = teacher_logits.argmax(dim=1)

        # Guardrail forward
        guard_out = None
        if guardrail is not None:
            guard_out = guardrail(student_logits, student_feat)

        # MC Dropout
        mc_entropy = None
        if mc_dropout_passes > 0:
            model.train()
            mc_logits_list = []
            for _ in range(mc_dropout_passes):
                mc_out = model(imgs)
                mc_logits_list.append(F.softmax(mc_out, dim=1))
            model.eval()
            mc_mean = torch.stack(mc_logits_list).mean(dim=0)
            eps = 1e-8
            mc_ent = -(mc_mean * (mc_mean + eps).log()).sum(dim=1)
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
            pci = per_class_iou(student_preds[i], lbl, num_classes)
            risk = 1.0 - miou
            probs_i = student_probs[i][:, valid]

            # MSP stats
            max_probs = probs_i.max(dim=0).values
            msp = max_probs.mean().item()
            msp_std = max_probs.std().item() if max_probs.numel() > 1 else 0.0

            # Entropy stats
            eps = 1e-8
            pixel_ent = -(probs_i * (probs_i + eps).log()).sum(dim=0)
            entropy = pixel_ent.mean().item()
            entropy_std = pixel_ent.std().item() if pixel_ent.numel() > 1 else 0.0

            # Low-confidence pixel fraction
            low_conf_frac = (max_probs < 0.5).float().mean().item()

            rec = {
                "miou": miou,
                "risk": risk,
                "msp": msp,
                "msp_std": msp_std,
                "entropy": entropy,
                "entropy_std": entropy_std,
                "low_conf_pixel_frac": low_conf_frac,
                "per_class_iou": pci,
            }

            if mc_entropy is not None:
                rec["mc_entropy"] = mc_entropy[i]

            if guard_out is not None:
                if "risk_score" in guard_out:
                    rec["guardrail_risk"] = guard_out["risk_score"][i].item()
                if "utility_score" in guard_out:
                    rec["guardrail_risk"] = guard_out["utility_score"][i].item()
                    rec["guardrailpp_utility"] = guard_out["utility_score"][i].item()
                if "margin_vec" in guard_out:
                    margin = guard_out["margin_vec"][i]
                    rec["margin_vec_min"] = float(margin.min().item())
                    for k in range(margin.shape[0]):
                        rec[f"margin_{k}"] = float(margin[k].item())
                if "gap_heatmap" in guard_out:
                    hm = guard_out["gap_heatmap"][i]
                    valid_hm = hm[valid] if hm.shape == lbl.shape else hm.flatten()
                    rec["guardrail_gap_mean"] = valid_hm.mean().item()
                    rec["guardrail_gap_std"] = valid_hm.std().item() if valid_hm.numel() > 1 else 0.0
                    rec["guardrail_gap_max"] = valid_hm.max().item()

            if teacher_preds is not None:
                t_correct = (teacher_preds[i][valid] == lbl[valid])
                s_correct = (student_preds[i][valid] == lbl[valid])
                s_wrong = ~s_correct

                # Oracle gap: teacher right AND student wrong
                t_right_s_wrong = (t_correct & s_wrong).float().mean().item()
                rec["oracle_gap"] = t_right_s_wrong

                # Disagreement metrics
                rec["teacher_acc"] = t_correct.float().mean().item()
                rec["student_acc"] = s_correct.float().mean().item()
                rec["disagreement_rate"] = (
                    teacher_preds[i][valid] != student_preds[i][valid]
                ).float().mean().item()

                # THE MONEY METRIC: student confident AND wrong AND teacher right
                student_confident = max_probs > 0.9
                if student_confident.any():
                    s_pred_conf = student_preds[i][valid][student_confident]
                    t_pred_conf = teacher_preds[i][valid][student_confident]
                    gt_conf = lbl[valid][student_confident]
                    rec["confident_wrong_teacher_right"] = float(
                        ((s_pred_conf != gt_conf) & (t_pred_conf == gt_conf)).float().mean().item()
                    )
                    rec["n_confident_pixels"] = int(student_confident.sum().item())
                else:
                    rec["confident_wrong_teacher_right"] = 0.0
                    rec["n_confident_pixels"] = 0

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
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_summary = []
    all_confident_failure = {}
    all_gain_curves = {}

    for name, model in students.items():
        print(f"\n[Benchmark] Evaluating: {name}")

        g = guardrail if (guardrail is not None and name == guardrail_student_name) else None

        records = extract_image_scores(
            model, val_loader, num_classes, device,
            teacher=teacher, guardrail=g,
            use_student_features=(use_student_features and g is not None),
            mc_dropout_passes=mc_dropout_passes,
        )
        all_results[name] = records

        # Save per-image CSV (exclude per_class_iou dict)
        csv_fields = [k for k in records[0].keys() if k != "per_class_iou"]
        csv_path = save_path / f"{name}_scores.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for r in records:
                writer.writerow({k: v for k, v in r.items() if k != "per_class_iou"})

        # ── Standard AURC ──
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
        if "guardrailpp_utility" in records[0]:
            methods["Guardrail++"] = 1.0 - np.array([r["guardrailpp_utility"] for r in records])
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

        # ── Confident failure detection ──
        cfd = compute_confident_failure_detection(records)
        all_confident_failure[name] = cfd
        if cfd:
            print(f"\n  [Confident Failure Detection] {name}")
            for thresh, metrics in sorted(cfd.items()):
                parts = [f"n={metrics['n_confident']}", f"fails={metrics['n_failures']}"]
                for m in ["MSP_AUROC", "Entropy_AUROC", "Guardrail_AUROC", "Oracle_AUROC"]:
                    if m in metrics:
                        parts.append(f"{m}={metrics[m]:.3f}")
                print(f"    MSP>{thresh}: {', '.join(parts)}")

        # ── Selective gain curves ──
        msp_scores = np.array([r["msp"] for r in records])
        gain_curves = {}
        for method_name, scores in methods.items():
            if method_name == "MSP":
                continue
            gains = compute_selective_gain(risks, msp_scores, scores, method_name)
            gain_curves[method_name] = gains
        all_gain_curves[name] = gain_curves

    # ── Save summary CSV ──
    summary_csv = save_path / "benchmark_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summary[0].keys())
        writer.writeheader()
        writer.writerows(all_summary)

    cfd_rows = []
    for sname, cfd in all_confident_failure.items():
        for thresh, metrics in cfd.items():
            row = {"student": sname, "msp_threshold": thresh}
            row.update(metrics)
            cfd_rows.append(row)
    if cfd_rows:
        # Collect ALL fields across all rows
        all_fields = []
        seen = set()
        for row in cfd_rows:
            for k in row.keys():
                if k not in seen:
                    all_fields.append(k)
                    seen.add(k)
        cfd_csv = save_path / "confident_failure_detection.csv"
        with open(cfd_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cfd_rows)
        print(f"\n[csv] {cfd_csv}")

    # ── Print tables ──
    print(f"\n{'=' * 90}")
    print("UNIFIED BENCHMARK RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Student':<18} {'Method':<16} {'mIoU':>7} {'AURC':>8} {'R@80%':>8} {'R@90%':>8} {'R@95%':>8}")
    print("-" * 83)
    for row in all_summary:
        print(
            f"{row['student']:<18} {row['method']:<16} "
            f"{row['miou']:>7.4f} {row['aurc']:>8.4f} "
            f"{row['risk_at_80']:>8.4f} {row['risk_at_90']:>8.4f} {row['risk_at_95']:>8.4f}"
        )
    print(f"\n[csv] {summary_csv}")

    _print_confident_failure_summary(all_confident_failure)

    # ── All plots ──
    _plot_benchmark(all_results, all_summary, save_path)
    _plot_confident_failures(all_results, all_confident_failure, save_path)
    _plot_selective_gain(all_gain_curves, save_path)
    _plot_confident_wrong_scatter(all_results, save_path)

    # ── wandb logging ──
    try:
        from _wandb_helpers import log_eval_results
        log_eval_results(all_summary, all_confident_failure, save_path=save_path)
    except Exception as e:
        print(f"[wandb] Eval logging skipped: {e}")

    return all_summary


def _print_confident_failure_summary(all_cfd):
    """Print confident failure detection table."""
    print(f"\n{'=' * 90}")
    print("CONFIDENT FAILURE DETECTION (higher AUROC = better at catching confident mistakes)")
    print(f"{'=' * 90}")

    for sname, cfd in all_cfd.items():
        if not cfd:
            continue
        print(f"\n  {sname}:")
        print(f"  {'Threshold':<12} {'N':>5} {'Fails':>6} {'Rate':>7} "
              f"{'MSP':>8} {'Entropy':>8} {'Guard':>8} {'Oracle':>8}")
        print(f"  {'-' * 72}")
        for thresh in sorted(cfd.keys()):
            m = cfd[thresh]
            line = f"  MSP>{thresh:<6.2f} {m['n_confident']:>5} {m['n_failures']:>6} {m['failure_rate']:>7.1%}"
            line += f" {m.get('MSP_AUROC', 0.5):>8.3f}"
            line += f" {m.get('Entropy_AUROC', 0.5):>8.3f}"
            if 'Guardrail_AUROC' in m:
                line += f" {m['Guardrail_AUROC']:>8.3f}"
            else:
                line += f" {'---':>8}"
            if 'Oracle_AUROC' in m:
                line += f" {m['Oracle_AUROC']:>8.3f}"
            else:
                line += f" {'---':>8}"
            print(line)


# ─── Plotting ─────────────────────────────────────────────────────────────────


def _plot_benchmark(all_results, all_summary, save_path):
    """Standard risk-coverage, AURC bars, calibration."""
    import pandas as pd
    df = pd.DataFrame(all_summary)
    student_names = df["student"].unique()
    n_students = len(student_names)

    # Risk-Coverage per student
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
        if "guardrailpp_utility" in records[0]:
            methods["Guardrail++"] = 1.0 - np.array([r["guardrailpp_utility"] for r in records])
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

    # AURC comparison bar chart
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

    # Guardrail calibration
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


def _plot_confident_failures(all_results, all_cfd, save_path):
    """
    KEY PLOT: Among confident images (MSP > threshold), compare
    each method's ability to detect actual failures.
    """
    for sname, cfd in all_cfd.items():
        if not cfd:
            continue

        thresholds = sorted(cfd.keys())
        if len(thresholds) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: AUROC at each threshold
        ax = axes[0]
        method_info = [
            ("MSP_AUROC",       "MSP",       "-o",  "#1f77b4"),
            ("Entropy_AUROC",   "Entropy",   "-s",  "#ff7f0e"),
            ("Guardrail_AUROC", "Guardrail", "-^",  "#2ca02c"),
            ("Oracle_AUROC",    "Oracle",    "--d", "#d62728"),
        ]

        for method_key, label, style, color in method_info:
            vals = [cfd[t].get(method_key, None) for t in thresholds]
            if all(v is None for v in vals):
                continue
            vals = [v if v is not None else 0.5 for v in vals]
            ax.plot(thresholds, vals, style, label=label, color=color, markersize=8, linewidth=2)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.set(xlabel="MSP Confidence Threshold",
               ylabel="AUROC",
               title=f"Confident Failure Detection ({sname})\n"
                     f"How well can each method detect failures\namong images MSP considers safe?")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.35, 1.0)

        # Panel 2: Failure counts and rate
        ax = axes[1]
        n_conf = [cfd[t]["n_confident"] for t in thresholds]
        n_fail = [cfd[t]["n_failures"] for t in thresholds]
        fail_rate = [cfd[t]["failure_rate"] for t in thresholds]

        ax2 = ax.twinx()
        ax.bar(thresholds, n_conf, width=0.03, alpha=0.3, color="#1f77b4", label="N confident")
        ax.bar(thresholds, n_fail, width=0.03, alpha=0.7, color="#d62728", label="N failures")
        ax2.plot(thresholds, [r * 100 for r in fail_rate], "-ok", label="Failure rate %", markersize=8)

        ax.set(xlabel="MSP Confidence Threshold", ylabel="Count",
               title=f"Confident Failures ({sname})")
        ax2.set_ylabel("Failure Rate (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path / f"confident_failures_{sname}.png", dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path / f'confident_failures_{sname}.png'}")
        plt.close(fig)


def _plot_selective_gain(all_gain_curves, save_path):
    """
    Plot risk reduction of guardrail/oracle vs MSP at each coverage level.
    Positive area = method outperforms MSP.
    """
    for sname, gain_curves in all_gain_curves.items():
        if not gain_curves:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Absolute risk at each coverage
        ax = axes[0]
        if gain_curves:
            first_method = list(gain_curves.values())[0]
            covs = [g["coverage"] for g in first_method]
            msp_risks = [g["msp_risk"] for g in first_method]
            ax.plot(covs, msp_risks, "-", color="#1f77b4", linewidth=2, label="MSP")

        colors = {"Neg-Entropy": "#ff7f0e", "Guardrail": "#2ca02c",
                  "Oracle": "#d62728", "MC-Dropout": "#9467bd"}
        for method_name, gains in gain_curves.items():
            covs = [g["coverage"] for g in gains]
            method_risks = [g[f"{method_name}_risk"] for g in gains]
            style = "--" if method_name == "Oracle" else "-"
            ax.plot(covs, method_risks, style, color=colors.get(method_name, "gray"),
                    linewidth=2, label=method_name)

        ax.set(xlabel="Coverage", ylabel="Mean Risk (1 - mIoU)",
               title=f"Selective Prediction Risk ({sname})\nLower = better at each coverage")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Gain over MSP (% risk reduction)
        ax = axes[1]
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)

        for method_name, gains in gain_curves.items():
            covs = [g["coverage"] for g in gains]
            gain_vals = [g["relative_gain_pct"] for g in gains]
            style = "--" if method_name == "Oracle" else "-"
            color = colors.get(method_name, "gray")
            ax.plot(covs, gain_vals, style, color=color, linewidth=2, label=method_name)
            gain_arr = np.array(gain_vals)
            cov_arr = np.array(covs)
            ax.fill_between(cov_arr, 0, gain_arr,
                            where=gain_arr > 0, alpha=0.1, color=color)

        ax.set(xlabel="Coverage",
               ylabel="Risk Reduction vs MSP (%)",
               title=f"Gain Over MSP ({sname})\n"
                     f"Green area = guardrail outperforms MSP")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path / f"selective_gain_{sname}.png", dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path / f'selective_gain_{sname}.png'}")
        plt.close(fig)


def _plot_confident_wrong_scatter(all_results, save_path):
    """
    THE MONEY PLOT: Scatter of MSP vs actual risk, colored by guardrail score.
    Shows guardrail catches high-risk images that MSP thinks are safe.
    """
    for sname, records in all_results.items():
        has_guard = "guardrail_risk" in records[0]
        has_oracle = "oracle_gap" in records[0]
        if not has_guard and not has_oracle:
            continue

        n_panels = 1 + int(has_guard) + int(has_oracle)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        msp = np.array([r["msp"] for r in records])
        risk = np.array([r["risk"] for r in records])
        median_risk = np.median(risk)

        panel_idx = 0

        # Panel 1: MSP vs Risk (baseline — color by risk)
        ax = axes[panel_idx]
        scatter = ax.scatter(msp, risk, c=risk, cmap="RdYlGn_r", s=15, alpha=0.6,
                            vmin=risk.min(), vmax=risk.max())
        ax.axvline(x=0.90, color="red", linestyle="--", alpha=0.5, label="MSP=0.90")
        ax.axhline(y=median_risk, color="blue", linestyle="--", alpha=0.5, label="Median risk")

        # Highlight the dangerous quadrant
        high_msp_high_risk = (msp > 0.90) & (risk > median_risk)
        n_dangerous = high_msp_high_risk.sum()
        ax.scatter(msp[high_msp_high_risk], risk[high_msp_high_risk],
                   facecolors="none", edgecolors="red", s=60, linewidths=1.5,
                   label=f"Confident failures (n={n_dangerous})")

        ax.set(xlabel="MSP (confidence)", ylabel="Risk (1 - mIoU)",
               title=f"MSP vs Risk ({sname})\nRed circles = confident failures MSP misses")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label="Risk")
        panel_idx += 1

        # Panel 2: Same scatter, colored by guardrail risk
        if has_guard:
            ax = axes[panel_idx]
            guard_risk = np.array([r["guardrail_risk"] for r in records])
            scatter = ax.scatter(msp, risk, c=guard_risk, cmap="RdYlGn_r", s=15, alpha=0.6,
                                vmin=guard_risk.min(), vmax=guard_risk.max())
            plt.colorbar(scatter, ax=ax, label="Guardrail risk score")
            ax.axvline(x=0.90, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=median_risk, color="blue", linestyle="--", alpha=0.5)

            if n_dangerous > 0:
                guard_on_dangerous = guard_risk[high_msp_high_risk]
                guard_on_safe = guard_risk[~high_msp_high_risk]
                separation = guard_on_dangerous.mean() - guard_on_safe.mean()
                ax.set_title(
                    f"Guardrail Scores ({sname})\n"
                    f"Confident failures: {guard_on_dangerous.mean():.3f} "
                    f"vs safe: {guard_on_safe.mean():.3f} "
                    f"(Δ={separation:+.3f})"
                )
            else:
                ax.set_title(f"Guardrail Scores ({sname})")
            ax.set(xlabel="MSP (confidence)", ylabel="Risk (1 - mIoU)")
            ax.grid(True, alpha=0.3)
            panel_idx += 1

        # Panel 3: Same scatter, colored by oracle gap
        if has_oracle:
            ax = axes[panel_idx]
            oracle = np.array([r["oracle_gap"] for r in records])
            scatter = ax.scatter(msp, risk, c=oracle, cmap="RdYlGn_r", s=15, alpha=0.6,
                                vmin=0, vmax=max(oracle.max(), 0.01))
            plt.colorbar(scatter, ax=ax, label="Oracle gap")
            ax.axvline(x=0.90, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=median_risk, color="blue", linestyle="--", alpha=0.5)

            if n_dangerous > 0:
                oracle_on_dangerous = oracle[high_msp_high_risk]
                oracle_on_safe = oracle[~high_msp_high_risk]
                ax.set_title(
                    f"Oracle Gap ({sname})\n"
                    f"Confident failures: {oracle_on_dangerous.mean():.3f} "
                    f"vs safe: {oracle_on_safe.mean():.3f}"
                )
            else:
                ax.set_title(f"Oracle Gap ({sname})")
            ax.set(xlabel="MSP (confidence)", ylabel="Risk (1 - mIoU)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path / f"confident_wrong_scatter_{sname}.png", dpi=150, bbox_inches="tight")
        print(f"[plot] {save_path / f'confident_wrong_scatter_{sname}.png'}")
        plt.close(fig)