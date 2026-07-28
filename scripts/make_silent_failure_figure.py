"""Qualitative grid: silent confident-failure under shift.

4 rows (one image per dataset) × 7 columns:
    input | GT | student pred | student error | 1 - student MSP |
    GT-supervised guardrail | teacher-supervised guardrail (ours)

Columns 5-7 are all "higher = worse" severity maps on a shared colormap, each
percentile-normalised so they are compared on spatial structure rather than
absolute scale. The guardrail columns show `gap_pred` — the severity head that
produces every paper number (see the alias table in CLAUDE.md) — for the
dense_multi (ours) and gt_risk (GT control) checkpoints. Both heads share
architecture, training schedule, and corruption aug; only the supervision
target differs.

Image picks come from combined_all/per_image.csv. Percentiles are taken within
a row's own condition; among images that are confidently wrong (student_msp
>= 0.90, risk percentile >= 0.70) and that dense_multi flags (score percentile
>= 0.70), each row is the image maximising
`pct_ours - max(pct_gt_risk, pct_1-msp)`. The `pct=` values below record those
ranks for provenance; they are not drawn. Paths/checkpoints resolve from env
vars (STUDENT_CKPT, GUARD_TEACHER_CKPT, GUARD_GT_CKPT, ACDC_PATH, IDD_PATH,
BDD_PATH, CITY_PATH, MIT_STUDENT).

The four preprocessed model inputs are committed as an asset, so the default
invocation needs only the checkpoints — no datasets, no cluster:

    python scripts/make_silent_failure_figure.py --device cpu

Regenerate that asset where the datasets live (only needed if SELECTION
changes), or cache finished panels to restyle without running models at all:

    python scripts/make_silent_failure_figure.py --dump-samples assets/silent_failure_samples.npz
    python scripts/make_silent_failure_figure.py --dump panels.npz
    python scripts/make_silent_failure_figure.py --panels panels.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SELECTION = [
    dict(
        dataset="acdc",
        domain="fog",
        label="ACDC fog",
        image_id="GOPR0476_frame_000931_rgb_anon",
        pct=dict(msp=0.10, gt=0.71, ours=0.89),
    ),
    dict(
        dataset="acdc",
        domain="snow",
        label="ACDC snow",
        image_id="GOPR0604_frame_000322_rgb_anon",
        pct=dict(msp=0.13, gt=0.56, ours=0.70),
    ),
    dict(
        dataset="acdc",
        domain="night",
        label="ACDC night",
        image_id="GOPR0356_frame_000430_rgb_anon",
        pct=dict(msp=0.20, gt=0.59, ours=0.73),
    ),
    # BDD image_ids are positional indices into the sorted val split. (An IDD
    # row belongs here too, but only 41 of its 981 val images are staged on the
    # cluster, so positional ids no longer resolve — restage IDD to restore it.)
    dict(dataset="bdd", domain="all", label="BDD100K (US)",  image_id="img_000723",
         pct=dict(msp=0.33, gt=0.40, ours=0.76)),
]

COLUMN_TITLES = [
    "Input",
    "Ground truth",
    "Student pred",
    "Student error",
    "1 − MSP",
    "GT-head",
    "Ours",
]

# Standard Cityscapes 19-class palette (trainId -> RGB).
CS_PALETTE = np.array(
    [
        [128,  64, 128],
        [244,  35, 232],
        [ 70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [ 70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [  0,   0, 142],
        [  0,   0,  70],
        [  0,  60, 100],
        [  0,  80, 100],
        [  0,   0, 230],
        [119,  11,  32],
    ],
    dtype=np.uint8,
)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
NUM_CLASSES = 19
IGNORE = 255


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_dataset_path(ds_name: str) -> Path:
    env_key = {
        "acdc": "ACDC_PATH",
        "idd": "IDD_PATH",
        "bdd": "BDD_PATH",
        "city": "CITY_PATH",
    }[ds_name]
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    default = {
        "acdc": "data/acdc",
        "idd": "data/idd",
        "bdd": "data/bdd100k",
        "city": "data/cityscapes",
    }[ds_name]
    return repo_root() / default


def resolve_checkpoint(env_var: str, *globs: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"{env_var}={p!s} does not exist")
        return p
    for pattern in globs:
        matches = sorted(repo_root().glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for m in matches:
            if m.is_file():
                return m
    raise FileNotFoundError(
        f"Could not find checkpoint for {env_var}. Set {env_var}=... or "
        f"place a checkpoint matching one of: {globs}"
    )


def resolve_backbone(model_tag: str, env_key: str) -> str:
    override = os.environ.get(env_key)
    if override:
        return override
    # Mirror the HF-snapshot discovery the slurm scripts use.
    candidates = []
    scratch = os.environ.get("SCRATCH")
    roots = [scratch] if scratch else []
    roots.append(str(Path.home() / ".cache"))
    for root in roots:
        candidates.extend(Path(root).glob(f"**/models--nvidia--{model_tag}/snapshots/*"))
    for c in candidates:
        if c.is_dir():
            return str(c)
    raise FileNotFoundError(
        f"Could not locate HuggingFace snapshot for nvidia/{model_tag}. Set {env_key}=..."
    )


def fetch_sample(dataset: str, domain: str, image_id: str, dataset_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (image[C,H,W] normalized, label[H,W]) for one target image."""
    sys.path.insert(0, str(repo_root()))
    if dataset == "acdc":
        return _fetch_acdc(domain, image_id, dataset_path)
    if dataset == "idd":
        from src.train.data import IDDDataset

        ds = IDDDataset(str(dataset_path), split="val", crop_size=512)
    elif dataset == "bdd":
        from src.train.data import BDDDataset

        ds = BDDDataset(str(dataset_path), split="val", crop_size=512)
    elif dataset == "city":
        from src.train.data import CityscapesDataset

        ds = CityscapesDataset(str(dataset_path), split="val", crop_size=512)
    else:
        raise ValueError(dataset)

    idx = int(image_id.replace("img_", "").lstrip("0") or "0")
    img, lbl = ds[idx][0], ds[idx][1]
    return img, lbl


def _fetch_acdc(domain: str, image_id: str, dataset_path: Path):
    """Reuse full_eval's ACDC loader so the label LUT matches the paper run."""
    sys.path.insert(0, str(repo_root()))
    from src.eval.full_eval import EvalConfig, _build_acdc_loader

    cfg = EvalConfig(
        dataset_name="acdc",
        dataset_path=str(dataset_path),
        split="val",
        domain=domain,
        batch_size=1,
        # The loader is scanned linearly for one image_id; workers keep that
        # from serialising 100 image decodes per row.
        num_workers=8,
        num_classes=NUM_CLASSES,
        device="cpu",
        student_backbone="",
        teacher_backbone=None,
        temperature=1.0,
    )
    loader = _build_acdc_loader(cfg)
    for batch in loader:
        img, lbl, meta = batch
        m = meta[0] if isinstance(meta, list) else {k: v[0] for k, v in meta.items()}
        if m.get("image_id") == image_id:
            return img[0], lbl[0]
    raise LookupError(f"ACDC image_id={image_id!r} (domain={domain}) not found under {dataset_path}")


def load_models(
    student_ckpt: Path, teacher_head_ckpt: Path, gt_head_ckpt: Path, device: str
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, bool, bool]:
    sys.path.insert(0, str(repo_root()))
    from src.eval.full_eval import EvalConfig, build_guardrail_model, build_student_model

    student_backbone = resolve_backbone("mit-b1", "MIT_STUDENT")

    cfg = EvalConfig(
        dataset_name="cityscapes",
        dataset_path="",
        split="val",
        domain="all",
        batch_size=1,
        num_workers=0,
        num_classes=NUM_CLASSES,
        device=device,
        student_backbone=student_backbone,
        teacher_backbone=None,
        temperature=1.0,
    )
    student = build_student_model(cfg, str(student_ckpt))
    teacher_head = build_guardrail_model(cfg, str(teacher_head_ckpt))
    gt_head = build_guardrail_model(cfg, str(gt_head_ckpt))
    teacher_uses_feat = bool(getattr(teacher_head, "_use_student_features", False))
    gt_uses_feat = bool(getattr(gt_head, "_use_student_features", False))
    return student, teacher_head, gt_head, teacher_uses_feat, gt_uses_feat


@torch.no_grad()
def run_forward(
    img: torch.Tensor,
    lbl: torch.Tensor,
    student: torch.nn.Module,
    teacher_head: torch.nn.Module,
    gt_head: torch.nn.Module,
    teacher_uses_feat: bool,
    gt_uses_feat: bool,
    device: str,
) -> Dict[str, np.ndarray]:
    x = img.unsqueeze(0).to(device)
    need_feat = teacher_uses_feat or gt_uses_feat

    if need_feat:
        logits, feat = student(x, return_features=True)
    else:
        logits = student(x)
        feat = None

    probs = F.softmax(logits, dim=1)
    msp, pred = probs.max(dim=1)

    def head_map(head, uses_feat):
        # gap_pred is the severity head every paper number is computed from.
        out = head(logits, student_features=feat if uses_feat else None)
        return out["gap_pred"].squeeze(0).cpu().numpy()

    teacher_map = head_map(teacher_head, teacher_uses_feat)
    gt_map = head_map(gt_head, gt_uses_feat)

    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
    lbl_np = lbl.cpu().numpy()
    pred_np = pred[0].cpu().numpy()
    msp_np = msp[0].cpu().numpy()
    valid = lbl_np != IGNORE
    err = (pred_np != lbl_np) & valid

    return {
        "input": img_np,
        "label": lbl_np,
        "pred": pred_np,
        "error": err,
        "msp": msp_np,
        "teacher_map": teacher_map,
        "gt_map": gt_map,
        "valid": valid,
    }


def colorize_label(lbl: np.ndarray) -> np.ndarray:
    out = np.zeros((lbl.shape[0], lbl.shape[1], 3), dtype=np.uint8)
    valid = (lbl >= 0) & (lbl < NUM_CLASSES)
    out[valid] = CS_PALETTE[lbl[valid]]
    return out


def robust_norm(a: np.ndarray, valid: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Scale to [0,1] by percentiles of the valid pixels (structure, not scale)."""
    ref = a[valid] if valid.any() else a
    v0, v1 = np.percentile(ref, [lo, hi])
    if v1 <= v0:
        v1 = v0 + 1e-6
    return np.clip((a - v0) / (v1 - v0), 0.0, 1.0)


def render_grid(rows: List[Tuple[dict, Dict[str, np.ndarray]]], output_path: Path) -> None:
    n_rows = len(rows)
    n_cols = len(COLUMN_TITLES)

    # Row heights follow each image's aspect so non-square rows (BDD) don't
    # leave a gap under the square ACDC rows.
    ratios = [a["input"].shape[0] / a["input"].shape[1] for _, a in rows]
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.3 * n_cols, 2.3 * sum(ratios)),
        squeeze=False,
        gridspec_kw={"height_ratios": ratios},
    )
    err_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.15, 0.15, 1.0)])
    score_cmap = "magma"

    for r, (meta, arrs) in enumerate(rows):
        valid = arrs["valid"]
        axes[r, 0].imshow(arrs["input"])
        axes[r, 1].imshow(colorize_label(arrs["label"]))
        axes[r, 2].imshow(colorize_label(arrs["pred"]))
        axes[r, 3].imshow(arrs["input"])
        axes[r, 3].imshow(arrs["error"].astype(np.uint8), cmap=err_cmap, vmin=0, vmax=1, alpha=0.75)
        # Columns 5-7 share a colormap and an orientation: brighter = worse.
        axes[r, 4].imshow(robust_norm(1.0 - arrs["msp"], valid), cmap=score_cmap, vmin=0.0, vmax=1.0)
        axes[r, 5].imshow(robust_norm(arrs["gt_map"], valid), cmap=score_cmap, vmin=0.0, vmax=1.0)
        axes[r, 6].imshow(robust_norm(arrs["teacher_map"], valid), cmap=score_cmap, vmin=0.0, vmax=1.0)

        axes[r, 0].set_ylabel(meta["label"], fontsize=11, rotation=0, ha="right", va="center", labelpad=14)

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)

    for c, title in enumerate(COLUMN_TITLES):
        axes[0, c].set_title(title, fontsize=11)

    fig.suptitle(
        "Silent confident failure under shift (mit-b1). Right three columns: brighter = worse, "
        "percentile-normalised per panel.\n"
        "GT-head is the same 66.8K-param head trained on ground truth instead of the teacher.",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


PANEL_KEYS = ("input", "label", "pred", "error", "msp", "teacher_map", "gt_map", "valid")

SAMPLES_ASSET = str(
    Path(__file__).resolve().parent.parent
    / "src" / "analysis" / "figure_scripts" / "assets" / "silent_failure_samples.npz"
)


def save_panels(rows: List[Tuple[dict, Dict[str, np.ndarray]]], path: Path) -> None:
    """Cache panels at half resolution so styling can be iterated off-cluster."""
    blob = {}
    for i, (meta, arrs) in enumerate(rows):
        blob[f"{i}_label_text"] = np.array(meta["label"])
        for k in PANEL_KEYS:
            a = arrs[k]
            a = a[::2, ::2] if a.ndim == 2 else a[::2, ::2, :]
            if a.dtype == np.float32 or a.dtype == np.float64:
                a = a.astype(np.float16)
            blob[f"{i}_{k}"] = a
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_rows=np.array(len(rows)), **blob)
    print(f"[dump] panels saved: {path}")


def load_panels(path: Path) -> List[Tuple[dict, Dict[str, np.ndarray]]]:
    d = np.load(path, allow_pickle=False)
    rows = []
    for i in range(int(d["n_rows"])):
        arrs = {k: d[f"{i}_{k}"] for k in PANEL_KEYS}
        arrs = {k: (v.astype(np.float32) if v.dtype == np.float16 else v) for k, v in arrs.items()}
        arrs["valid"] = arrs["valid"].astype(bool)
        arrs["error"] = arrs["error"].astype(bool)
        rows.append(({"label": str(d[f"{i}_label_text"])}, arrs))
    return rows


def save_samples(samples: List[Tuple[dict, torch.Tensor, torch.Tensor]], path: Path) -> None:
    """Cache the preprocessed model inputs so the figure regenerates offline.

    Stores the normalised image tensor exactly as the eval loaders produce it,
    so a local re-run reproduces cluster outputs rather than approximating them.
    """
    blob = {}
    for i, (meta, img, lbl) in enumerate(samples):
        blob[f"{i}_label_text"] = np.array(meta["label"])
        blob[f"{i}_img"] = img.cpu().numpy().astype(np.float16)
        blob[f"{i}_lbl"] = lbl.cpu().numpy().astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, n_rows=np.array(len(samples)), **blob)
    print(f"[dump] samples saved: {path}")


def load_samples(path: Path) -> List[Tuple[dict, torch.Tensor, torch.Tensor]]:
    d = np.load(path, allow_pickle=False)
    out = []
    for i in range(int(d["n_rows"])):
        meta = {"label": str(d[f"{i}_label_text"])}
        img = torch.from_numpy(d[f"{i}_img"].astype(np.float32))
        lbl = torch.from_numpy(d[f"{i}_lbl"].astype(np.int64))
        out.append((meta, img, lbl))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(repo_root() / "src" / "analysis" / "figures" / "fig_silent_confident_failure.png"),
    )
    parser.add_argument("--dump", default=None, help="write computed panels to this .npz")
    parser.add_argument("--panels", default=None, help="render from a cached .npz (no models needed)")
    parser.add_argument("--dump-samples", default=None,
                        help="write the preprocessed model inputs to this .npz (needs the datasets)")
    parser.add_argument("--samples", default=SAMPLES_ASSET, help="read model inputs from this .npz "
                        "instead of the datasets; defaults to the committed asset when present")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.panels:
        rows = load_panels(Path(args.panels))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render_grid(rows, output_path)
        print(f"[done] figure saved: {output_path.resolve()}")
        return 0

    student_ckpt = resolve_checkpoint(
        "STUDENT_CKPT",
        "runs/mit-b1_skd_*/student_skd.ckpt",
        "outputs*/student_skd.ckpt",
    )
    teacher_head_ckpt = resolve_checkpoint(
        "GUARD_TEACHER_CKPT",
        "runs/mit-b1_guard_dense_multi_*/guardrail.ckpt",
    )
    gt_head_ckpt = resolve_checkpoint(
        "GUARD_GT_CKPT",
        "runs/mit-b1_guard_gt_risk_*/guardrail.ckpt",
    )
    print(f"[ckpt] student      : {student_ckpt}")
    print(f"[ckpt] teacher-head : {teacher_head_ckpt}")
    print(f"[ckpt] gt-head      : {gt_head_ckpt}")

    student, teacher_head, gt_head, teacher_uses_feat, gt_uses_feat = load_models(
        student_ckpt, teacher_head_ckpt, gt_head_ckpt, args.device
    )
    print(f"[head] teacher uses_student_features={teacher_uses_feat}  gt uses_student_features={gt_uses_feat}")

    use_cache = args.samples and Path(args.samples).is_file() and not args.dump_samples
    if use_cache:
        print(f"[img ] using cached inputs: {args.samples}")
        samples = load_samples(Path(args.samples))
    else:
        samples = []
        for sel in SELECTION:
            ds_path = resolve_dataset_path(sel["dataset"])
            print(f"[img ] {sel['label']}: dataset_path={ds_path}  image_id={sel['image_id']}")
            img, lbl = fetch_sample(sel["dataset"], sel["domain"], sel["image_id"], ds_path)
            samples.append((sel, img, lbl))
        if args.dump_samples:
            save_samples(samples, Path(args.dump_samples))

    rows = []
    for meta, img, lbl in samples:
        arrs = run_forward(
            img, lbl, student, teacher_head, gt_head,
            teacher_uses_feat, gt_uses_feat, args.device,
        )
        rows.append((meta, arrs))

    if args.dump:
        save_panels(rows, Path(args.dump))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_grid(rows, output_path)
    print(f"[done] figure saved: {output_path}")
    print(f"[done]   absolute : {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
