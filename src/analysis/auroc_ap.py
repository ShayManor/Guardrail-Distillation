import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "./acdc_b0_b2_eval/csv"

# ============================================================
# Configuration
# ============================================================
DATASET = "acdc"
BACKBONE = "nvidia/mit-b2"
TRAIN_METHOD = "skd"

# Silent confident-failure definition
CONFIDENT_MSP_THRESHOLD = 0.90
WRONG_MIOU_THRESHOLD = 0.25

# For deployment-style top-k metrics
ALARM_FRACTION = 0.10

# Detector ordering for plots
DETECTOR_ORDER = ["Guardrail", "Temp-MSP", "MC-Dropout", "Entropy", "MSP"]

# ============================================================
# Helpers
# ============================================================
def recall_precision_fpr_at_top_fraction(y_true, risk_scores, frac=0.10):
    y_true = np.asarray(y_true).astype(int)
    risk_scores = np.asarray(risk_scores)

    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))

    order = np.argsort(-risk_scores)
    chosen = order[:k]

    pred = np.zeros(n, dtype=int)
    pred[chosen] = 1

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    return recall, precision, fpr, k

# ============================================================
# Load raw repo outputs
# ============================================================
per_image = pd.read_csv(os.path.join(BASE, "per_image_acdc.csv"))
runs = pd.read_csv(os.path.join(BASE, "runs_acdc.csv"))

# Exact slice used in the figures
df = per_image[
    (per_image["dataset_name"] == DATASET) &
    (per_image["student_backbone"] == BACKBONE) &
    (per_image["train_method"] == TRAIN_METHOD)
].copy()

if df.empty:
    raise RuntimeError("No rows found for requested dataset/backbone/train_method slice.")

# ============================================================
# Build silent confident-failure subset
# ============================================================
confident = df[df["student_msp"] >= CONFIDENT_MSP_THRESHOLD].copy()
confident["silent_failure"] = (confident["student_miou"] <= WRONG_MIOU_THRESHOLD).astype(int)

n_images = len(confident)
n_pos = int(confident["silent_failure"].sum())
pos_rate = float(confident["silent_failure"].mean())

# ============================================================
# Detector definitions
# ============================================================
detector_specs = {
    "MSP": {
        "column": "student_msp",
        "risk_score": lambda s: -s,
        "latency_source": "single_pass_student",
    },
    "Entropy": {
        "column": "student_entropy",
        "risk_score": lambda s: s,
        "latency_source": "single_pass_student",
    },
    "Temp-MSP": {
        "column": "temp_msp",
        "risk_score": lambda s: -s,
        "latency_source": "single_pass_student",
    },
    "MC-Dropout": {
        "column": "mc_entropy",
        "risk_score": lambda s: s,
        "latency_source": "mc_dropout",
    },
    "Guardrail": {
        "column": "guardrail_risk",
        "risk_score": lambda s: s,
        "latency_source": "student_plus_guardrail",
    },
}

run_slice = runs[
    (runs["dataset_name"] == DATASET) &
    (runs["student_backbone"] == BACKBONE)
].copy()

latency_lookup = {
    "single_pass_student": float(
        run_slice[run_slice["train_method"] == TRAIN_METHOD]["student_latency_ms"].median()
    ),
    "student_plus_guardrail": float(
        (
            run_slice[run_slice["train_method"] == TRAIN_METHOD]["student_latency_ms"] +
            run_slice[run_slice["train_method"] == TRAIN_METHOD]["guardrail_latency_ms"]
        ).median()
    ),
    "mc_dropout": float(confident["mc_dropout_latency_ms"].dropna().median()),
    "teacher": float(run_slice["teacher_latency_ms"].median()),
    "sup_student": float(run_slice[run_slice["train_method"] == "sup"]["student_latency_ms"].median()),
    "kd_student": float(run_slice[run_slice["train_method"] == "kd"]["student_latency_ms"].median()),
    "skd_student": float(run_slice[run_slice["train_method"] == "skd"]["student_latency_ms"].median()),
}

# ============================================================
# Build transparent tradeoff CSV from raw files
# ============================================================
rows = []
for detector_name, spec in detector_specs.items():
    col = spec["column"]
    tmp = confident[["image_id", "silent_failure", col]].dropna().copy()
    if tmp.empty:
        continue

    y_true = tmp["silent_failure"].astype(int).to_numpy()
    risk_scores = spec["risk_score"](tmp[col].to_numpy())

    auroc = float(roc_auc_score(y_true, risk_scores))
    ap = float(average_precision_score(y_true, risk_scores))
    recall10, precision10, fpr10, k = recall_precision_fpr_at_top_fraction(
        y_true, risk_scores, frac=ALARM_FRACTION
    )

    rows.append({
        "dataset_name": DATASET,
        "student_backbone": BACKBONE,
        "train_method": TRAIN_METHOD,
        "detector": detector_name,
        "subset_confident_msp_threshold": CONFIDENT_MSP_THRESHOLD,
        "positive_miou_threshold": WRONG_MIOU_THRESHOLD,
        "n_images_in_subset": len(tmp),
        "n_positive_silent_failures": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "auroc": auroc,
        "ap": ap,
        "recall_at_10pct": float(recall10),
        "precision_at_10pct": float(precision10),
        "fpr_at_10pct": float(fpr10),
        "k": int(k),
        "latency_ms": latency_lookup[spec["latency_source"]],
        "score_column_used": col,
        "score_direction": "higher = riskier" if detector_name in ["Entropy", "MC-Dropout", "Guardrail"] else "lower confidence -> higher risk via negation",
    })

tradeoff = pd.DataFrame(rows)
tradeoff["detector"] = pd.Categorical(tradeoff["detector"], categories=DETECTOR_ORDER, ordered=True)
tradeoff = tradeoff.sort_values("detector").reset_index(drop=True)

tradeoff_csv = os.path.join(BASE, "figure_tradeoff_b2_from_raw.csv")
tradeoff.to_csv(tradeoff_csv, index=False)

# ============================================================
# Figure 1: left-right AUROC and AP
# ============================================================
det = tradeoff.copy()

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=False)

bars1 = axes[0].bar(det["detector"], det["auroc"])
axes[0].set_ylabel("AUROC")
axes[0].set_ylim(0.6, 0.95)
# axes[0].set_title("AUROC", fontsize=12, pad=6)
# axes[0].grid(axis="y", alpha=0.25)
for rect, val in zip(bars1, det["auroc"]):
    axes[0].text(rect.get_x() + rect.get_width()/2, val + 0.008, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)
for spine in ["top", "right"]:
    axes[0].spines[spine].set_visible(False)
plt.setp(axes[0].get_xticklabels(), rotation=18, ha="right")

bars2 = axes[1].bar(det["detector"], det["ap"])
axes[1].set_ylabel("Average Precision")
axes[1].set_ylim(0.25, 0.8)
# axes[1].set_title("Average Precision", fontsize=12, pad=6)
# axes[1].grid(axis="y", alpha=0.25)
for rect, val in zip(bars2, det["ap"]):
    axes[1].text(rect.get_x() + rect.get_width()/2, val + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)
for spine in ["top", "right"]:
    axes[1].spines[spine].set_visible(False)
plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")

fig.suptitle("Silent confident-failure detection on ACDC, MiT-B2", fontsize=13, y=0.9)
fig.text(0.5, 0.81, f"MSP ≥ {CONFIDENT_MSP_THRESHOLD:.2f}; positive = frame mIoU ≤ {WRONG_MIOU_THRESHOLD:.2f}",
         ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])

fig1_path = os.path.join(BASE, "figure1_guardrail_detection_clean_final.png")
fig.savefig(fig1_path, dpi=220, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2: latency chart
# ============================================================
latency_df = pd.DataFrame({
    "method": ["SUP", "KD", "SKD", "SKD + Guardrail", "Teacher", "MC-Dropout"],
    "latency_ms": [
        latency_lookup["sup_student"],
        latency_lookup["kd_student"],
        latency_lookup["skd_student"],
        latency_lookup["student_plus_guardrail"],
        latency_lookup["teacher"],
        latency_lookup["mc_dropout"],
    ],
})

fig, ax = plt.subplots(figsize=(8.8, 4.8))
bars = ax.bar(latency_df["method"], latency_df["latency_ms"])
ax.set_ylabel("Per-frame latency (ms)")
ax.set_title(
    "Guardrail stays near single-pass latency, MC-Dropout does not\n"
    "ACDC, MiT-B2, robust median across runs",
    fontsize=13,
    pad=12,
)
ax.grid(axis="y", alpha=0.25)
ax.set_ylim(0, max(latency_df["latency_ms"]) * 1.16)
for rect, val in zip(bars, latency_df["latency_ms"]):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_height() + 1.0,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()

fig2_path = os.path.join(BASE, "figure2_latency_clean_final.png")
# fig.savefig(fig2_path, dpi=220, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 3: performance-latency tradeoff
# ============================================================
fig, ax = plt.subplots(figsize=(8.0, 5.2))
ax.scatter(det["latency_ms"], det["auroc"], s=85)

ax.set_xlabel("Per-frame latency (ms)")
ax.set_ylabel("AUROC on silent confident failures")
ax.set_title(
    "Performance–latency tradeoff on silent confident failures\nACDC, MiT-B2",
    fontsize=13,
    pad=10,
)
ax.text(
    0.01, 0.97,
    f"Subset: MSP ≥ {CONFIDENT_MSP_THRESHOLD:.2f}; positive = frame mIoU ≤ {WRONG_MIOU_THRESHOLD:.2f}",
    transform=ax.transAxes,
    va="top",
    fontsize=10,
)
ax.grid(alpha=0.25)

offsets = {
    "MSP": (-10, -10),
    "Entropy": (-10, 10),
    "Temp-MSP": (10, 10),
    "Guardrail": (10, 2),
    "MC-Dropout": (10, 8),
}
for _, row in det.iterrows():
    dx, dy = offsets[row["detector"]]
    ax.annotate(
        row["detector"],
        (row["latency_ms"], row["auroc"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha="left" if dx >= 0 else "right",
        va="center",
        fontsize=9,
    )

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()

fig3_path = os.path.join(BASE, "figure3_tradeoff_clean_final.png")
fig.savefig(fig3_path, dpi=220, bbox_inches="tight")
plt.close(fig)

# ============================================================
# README
# ============================================================
readme_path = os.path.join(BASE, "figures_from_raw_README.txt")
# with open(readme_path, "w") as f:
#     f.write(
#         "This script rebuilds the detector tradeoff CSV and all cleaned figures directly from raw repo outputs.\n"
#         f"Slice: dataset={DATASET}, backbone={BACKBONE}, train_method={TRAIN_METHOD}\n"
#         f"Confident subset: student_msp >= {CONFIDENT_MSP_THRESHOLD}\n"
#         f"Silent-failure positive label: student_miou <= {WRONG_MIOU_THRESHOLD}\n"
#         f"Subset size: {n_images}\n"
#         f"Silent failures: {n_pos}\n"
#         f"Positive rate: {pos_rate:.3f}\n"
#     )

print("Saved:")
print(" ", tradeoff_csv)
print(" ", fig1_path)
print(" ", fig2_path)
print(" ", fig3_path)
print(" ", readme_path)
