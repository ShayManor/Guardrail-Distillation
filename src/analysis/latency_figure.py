
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = "./acdc_b0_b2_eval/csv"
BACKBONE = "nvidia/mit-b2"
MSP_THR = 0.90
MIOU_BAD = 0.25

pi = pd.read_csv(f"{BASE}/per_image_acdc.csv").drop_duplicates(
    subset=["student_backbone", "train_method", "image_id"]
)

# Panel A: silent-failure rates by student training method
rows = []
for tm in ["sup", "kd", "skd"]:
    d = pi[(pi.student_backbone == BACKBONE) & (pi.train_method == tm)].copy()
    conf = d[d.student_msp >= MSP_THR].copy()
    bad = conf[conf.student_miou <= MIOU_BAD].copy()
    rows.append(
        {
            "train_method": tm.upper(),
            "n_confident": len(conf),
            "n_fail": len(bad),
            "failure_rate_pct": 100 * len(bad) / len(conf),
            "teacher_gain_on_fail_mean": (bad.teacher_miou - bad.student_miou).mean(),
            "teacher_better_frac_pct": 100 * (bad.teacher_miou > bad.student_miou).mean(),
        }
    )
panelA = pd.DataFrame(rows)

# Panel B: detector ranking quality on the SKD subset
d = pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].copy()
conf = d[d.student_msp >= MSP_THR].copy()
y = (conf.student_miou <= MIOU_BAD).astype(int).values

detectors = {
    "MSP": -conf.student_msp.values,
    "Entropy": conf.student_entropy.values,
    "Temp-MSP": -conf.temp_msp.values,
    "MC-Dropout": conf.mc_entropy.values,
    "Guardrail": conf.guardrail_risk.values,
}

def alarm_stats(y, score, budget=0.10):
    n = len(y)
    k = max(1, int(round(budget * n)))
    idx = np.argsort(-score)
    sel = np.zeros(n, dtype=bool)
    sel[idx[:k]] = True
    tp = ((y == 1) & sel).sum()
    fp = ((y == 0) & sel).sum()
    fn = ((y == 1) & (~sel)).sum()
    tn = ((y == 0) & (~sel)).sum()
    return {
        "recall_at_10pct": tp / max(tp + fn, 1),
        "precision_at_10pct": tp / max(tp + fp, 1),
        "fpr_at_10pct": fp / max(fp + tn, 1),
        "k": int(k),
    }

panelB = []
for name, score in detectors.items():
    panelB.append(
        {
            "detector": name,
            "auroc": roc_auc_score(y, score),
            "ap": average_precision_score(y, score),
            **alarm_stats(y, score, 0.10),
        }
    )
panelB = pd.DataFrame(panelB).sort_values("auroc")

# Figure 1
fig = plt.figure(figsize=(13.5, 5.7), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(panelA))
ax1.bar(x, panelA["failure_rate_pct"], width=0.62)
ax1.set_xticks(x, panelA["train_method"])
ax1.set_ylabel("Silent failure rate among confident frames (%)")
ax1.set_title("A. How often each student fails silently")
ax1.set_ylim(0, max(panelA["failure_rate_pct"]) * 1.42)
for i, row in panelA.iterrows():
    ax1.text(
        i,
        row["failure_rate_pct"] + 0.6,
        f"{row['n_fail']}/{row['n_confident']}\n{row['failure_rate_pct']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax1b = ax1.twinx()
ax1b.plot(x, panelA["teacher_gain_on_fail_mean"], marker="D", linestyle="--")
ax1b.set_ylabel("Teacher mIoU gain on those failure frames")
ax1b.set_ylim(0, max(panelA["teacher_gain_on_fail_mean"]) * 2.2)
for i, row in panelA.iterrows():
    ax1b.text(
        i,
        row["teacher_gain_on_fail_mean"] + 0.004,
        f"+{row['teacher_gain_on_fail_mean']:.3f}\n({row['teacher_better_frac_pct']:.0f}% better)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax1.text(
    0.02,
    0.98,
    f"Confident = MSP ≥ {MSP_THR:.2f}\nVery wrong = frame mIoU ≤ {MIOU_BAD:.2f}",
    transform=ax1.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)

ax2 = fig.add_subplot(gs[0, 1])
order = panelB["detector"].tolist()
vals = panelB["auroc"].values
ypos = np.arange(len(order))
ax2.barh(ypos, vals, height=0.62)
ax2.set_yticks(ypos, order)
ax2.set_xlabel("AUROC for detecting silent failures")
ax2.set_title("B. Which signal best ranks the bad confident frames?")
ax2.set_xlim(0.3, 1.0)

for i, row in enumerate(panelB.itertuples(index=False)):
    ax2.text(
        row.auroc + 0.01,
        i,
        f"AP {row.ap:.3f}   Recall@10% alarms {row.recall_at_10pct:.2f}",
        va="center",
        fontsize=9,
    )

ax2.text(
    0.01,
    0.98,
    f"Subset: SKD, {BACKBONE.split('/')[-1].upper()}, ACDC OOD\n"
    f"{int(conf.shape[0])} confident frames, {int(y.sum())} silent failures\n"
    "Ranking quality: sort frames by risk score and measure AUROC/AP.\n"
    "No fixed detection threshold is used in AUROC/AP.",
    transform=ax2.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)

fig.suptitle("OOD silent failures: distillation creates them, guardrail catches them", fontsize=14, y=1.02)
fig.tight_layout()
# fig.savefig(f"{BASE}/figure1_guardrail_vs_baselines_better.png", bbox_inches="tight")

# Figure 2 latency
lat = pd.DataFrame(
    [
        ("SUP", pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "sup")].student_latency_ms.mean()),
        ("KD", pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "kd")].student_latency_ms.mean()),
        ("SKD", pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].student_latency_ms.mean()),
        (
            "SKD + Guardrail",
            pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].student_latency_ms.mean()
            + pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].guardrail_latency_ms.mean(),
        ),
        ("Teacher", pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].teacher_latency_ms.mean()),
        ("MC-Dropout", pi[(pi.student_backbone == BACKBONE) & (pi.train_method == "skd")].mc_dropout_latency_ms.mean()),
    ],
    columns=["method", "latency_ms"],
)
fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=200)
order = ["SUP", "KD", "SKD", "SKD + Guardrail", "Teacher", "MC-Dropout"]
lat = lat.set_index("method").loc[order].reset_index()
x = np.arange(len(lat))
ax.bar(x, lat["latency_ms"], width=0.65)
ax.set_xticks(x, order, rotation=18, ha="right")
ax.set_ylabel("Latency per frame (ms)")
ax.set_title(f"Latency on ACDC, {BACKBONE.split('/')[-1].upper()}")
ax.set_ylim(0, lat["latency_ms"].max() * 1.18)
for i, row in lat.iterrows():
    ax.text(i, row["latency_ms"] + 0.8, f"{row['latency_ms']:.1f} ms", ha="center", va="bottom", fontsize=9)
base_skd = lat.loc[lat.method == "SKD", "latency_ms"].iloc[0]
gr = lat.loc[lat.method == "SKD + Guardrail", "latency_ms"].iloc[0]
mc = lat.loc[lat.method == "MC-Dropout", "latency_ms"].iloc[0]
ax.text(
    0.02,
    0.96,
    f"Guardrail overhead over SKD: {gr - base_skd:.1f} ms\nMC-Dropout slowdown vs SKD+Guardrail: {mc / gr:.1f}×",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)
fig.tight_layout()
fig.savefig(f"{BASE}/figure2_latency_better.png", bbox_inches="tight")

# Figure 3 tradeoff
trade = panelB.copy()
lat_map = {
    "MSP": lat.loc[lat.method == "SKD", "latency_ms"].iloc[0],
    "Entropy": lat.loc[lat.method == "SKD", "latency_ms"].iloc[0],
    "Temp-MSP": lat.loc[lat.method == "SKD", "latency_ms"].iloc[0],
    "MC-Dropout": lat.loc[lat.method == "MC-Dropout", "latency_ms"].iloc[0],
    "Guardrail": lat.loc[lat.method == "SKD + Guardrail", "latency_ms"].iloc[0],
}
trade["latency_ms"] = trade["detector"].map(lat_map)

fig, ax = plt.subplots(figsize=(7.4, 5.3), dpi=200)
for _, row in trade.iterrows():
    ax.scatter(row["latency_ms"], row["auroc"], s=70)
label_offsets = {
    "MSP": (0.75, 0),
    "Entropy": (0.75, 0),
    "Temp-MSP": (0.75, 0.0),
    "MC-Dropout": (0.75, 0.0),
    "Guardrail": (0.75, 0.0),
}
for _, row in trade.iterrows():
    dx, dy = label_offsets[row["detector"]]
    ax.text(row["latency_ms"] + dx, row["auroc"] + dy, row["detector"], fontsize=10, va="center")
ax.set_xlabel("Latency per frame (ms)")
ax.set_ylabel("AUROC on silent confident failures")
ax.set_title("Performance–latency tradeoff on ACDC, MiT-B2")
ax.set_xlim(14, 65)
ax.set_ylim(0.62, 0.93)
ax.text(
    0.02,
    0.04,
    "MSP ≥ 0.90, mIoU ≤ 0.25",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
)
fig.tight_layout()
fig.savefig(f"{BASE}/figure3_tradeoff_better.png", bbox_inches="tight")
