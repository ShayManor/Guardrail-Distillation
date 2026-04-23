"""Qualitative figure: silent confident failure under shift.

4 rows (one image per dataset) x 7 columns:
    input | GT | student pred | student error | student MSP |
    teacher-supervised guardrail heatmap | GT-supervised guardrail heatmap

Both guardrail heads share architecture (GuardrailPlusHead, 66.8K params),
training schedule, and corruption augmentation. The only difference is
the supervision target: teacher-student disagreement vs student-vs-GT
disagreement. At inference both produce sigmoid(disagree_logits) in [0,1].

Image selection criterion (reproduced from src/analysis/combined_all/per_image.csv,
backbone=b1): for each dataset, pick the image that maximises
(dense_multi util - gt_disagree util) among (student_msp>=0.85, student_risk>=0.25).
That is, the image most representative of teacher-head's edge.

Paths and checkpoints resolve through env vars (SKD_DIR, GUARD_TEACHER_DIR,
GUARD_GT_DIR, *_PATH for datasets) so this is safe to run on the cluster
or locally if checkpoints are staged.
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


# ---------------------------------------------------------------------------
# Selection (hard-coded from the per_image.csv analysis; see module docstring).
# ---------------------------------------------------------------------------

SELECTION = [
    dict(
        dataset="acdc",
        domain="fog",
        label="ACDC fog",
        image_id="GOPR0476_frame_000781_rgb_anon",
    ),
    dict(
        dataset="acdc",
        domain="snow",
        label="ACDC snow",
        image_id="GOPR0122_frame_000332_rgb_anon",
    ),
    dict(
        dataset="idd",
        domain="all",
        label="IDD (India)",
        image_id="img_000214",  # index 214 in sorted val split
    ),
    dict(
        dataset="bdd",
        domain="all",
        label="BDD100K (US)",
        image_id="img_000338",  # index 338 in sorted val split
    ),
]

COLUMN_TITLES = [
    "Input",
    "Ground truth",
    "Student pred",
    "Student error",
    "Student MSP",
    "Teacher-head",
    "GT-head (same arch)",
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
    # Mirror the auto-discovery used in slurm scripts.
    candidates = []
    for root in ("/scratch/gautschi/manors", str(Path.home() / ".cache")):
        candidates.extend(Path(root).glob(f"**/models--nvidia--{model_tag}/snapshots/*"))
    for c in candidates:
        if c.is_dir():
            return str(c)
    raise FileNotFoundError(
        f"Could not locate HuggingFace snapshot for nvidia/{model_tag}. Set {env_key}=..."
    )


# ---------------------------------------------------------------------------
# Image fetching per dataset.
# ---------------------------------------------------------------------------


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
    """Reuse full_eval's ACDC loader so the label-LUT path is identical."""
    sys.path.insert(0, str(repo_root()))
    from src.eval.full_eval import EvalConfig, _build_acdc_loader

    cfg = EvalConfig(
        dataset_name="acdc",
        dataset_path=str(dataset_path),
        split="val",
        domain=domain,
        batch_size=1,
        num_workers=0,
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


# ---------------------------------------------------------------------------
# Model loading (reuse full_eval builders).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forward + visualisation.
# ---------------------------------------------------------------------------


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
    msp, pred = probs.max(dim=1)  # [1,H,W], [1,H,W]

    def head_map(head, uses_feat):
        out = head(logits, student_features=feat if uses_feat else None)
        return torch.sigmoid(out["disagree_logits"]).squeeze(0).cpu().numpy()

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


def render_grid(rows: List[Tuple[dict, Dict[str, np.ndarray]]], output_path: Path) -> None:
    n_rows = len(rows)
    n_cols = len(COLUMN_TITLES)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.3 * n_cols, 2.3 * n_rows),
        squeeze=False,
    )
    err_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.15, 0.15, 1.0)])

    for r, (meta, arrs) in enumerate(rows):
        axes[r, 0].imshow(arrs["input"])
        axes[r, 1].imshow(colorize_label(arrs["label"]))
        axes[r, 2].imshow(colorize_label(arrs["pred"]))
        axes[r, 3].imshow(arrs["input"])
        axes[r, 3].imshow(arrs["error"].astype(np.uint8), cmap=err_cmap, vmin=0, vmax=1, alpha=0.75)
        axes[r, 4].imshow(arrs["msp"], cmap="viridis", vmin=0.0, vmax=1.0)
        axes[r, 5].imshow(arrs["teacher_map"], cmap="magma", vmin=0.0, vmax=1.0)
        axes[r, 6].imshow(arrs["gt_map"], cmap="magma", vmin=0.0, vmax=1.0)

        axes[r, 0].set_ylabel(meta["label"], fontsize=11, rotation=0, ha="right", va="center", labelpad=14)

        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_visible(False)

    for c, title in enumerate(COLUMN_TITLES):
        axes[0, c].set_title(title, fontsize=11)

    fig.suptitle(
        "Silent confident failure under shift: teacher-supervised vs GT-supervised guardrail "
        "(same 66.8K-param architecture, mit-b1 student)",
        fontsize=12, y=1.01,
    )
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(repo_root() / "src" / "analysis" / "figures" / "fig_silent_confident_failure.png"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
        "runs/mit-b1_guard_gt_disagree_*/guardrail.ckpt",
    )
    print(f"[ckpt] student      : {student_ckpt}")
    print(f"[ckpt] teacher-head : {teacher_head_ckpt}")
    print(f"[ckpt] gt-head      : {gt_head_ckpt}")

    student, teacher_head, gt_head, teacher_uses_feat, gt_uses_feat = load_models(
        student_ckpt, teacher_head_ckpt, gt_head_ckpt, args.device
    )
    print(f"[head] teacher uses_student_features={teacher_uses_feat}  gt uses_student_features={gt_uses_feat}")

    rows = []
    for sel in SELECTION:
        ds_path = resolve_dataset_path(sel["dataset"])
        print(f"[img ] {sel['label']}: dataset_path={ds_path}  image_id={sel['image_id']}")
        img, lbl = fetch_sample(sel["dataset"], sel["domain"], sel["image_id"], ds_path)
        arrs = run_forward(
            img, lbl, student, teacher_head, gt_head,
            teacher_uses_feat, gt_uses_feat, args.device,
        )
        rows.append((sel, arrs))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_grid(rows, output_path)
    print(f"[done] figure saved: {output_path}")
    print(f"[done]   absolute : {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
