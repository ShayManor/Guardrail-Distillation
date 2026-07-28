"""Precompute the real panels used by src/analysis/figure_scripts/fig_architecture.py.

Runs one image through the frozen student and a dense_multi guardrail head and
stores the input RGB, the disagreement map sigmoid(disagree_logits), the risk
map gap_pred, and the post-hoc energy map -logsumexp(student logits).
Preprocessing mirrors full_eval's ACDC loader (resize to 512x512 bilinear +
ImageNet normalisation), so the maps match eval-time inputs.

    python scripts/dump_arch_panels.py \
        --image data/acdc/rgb_anon/fog/val/GOPR0476/GOPR0476_frame_000788_rgb_anon.png \
        --student mit-b1/student_skd.ckpt \
        --guardrail mit-b1/dense_multi/guardrail.ckpt \
        --output src/analysis/figure_scripts/assets/arch_panels.npz

Maps are stored at half resolution (256x256); they are smooth enough that this
is visually identical at figure scale and keeps the asset small.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--student", required=True, type=Path)
    p.add_argument("--guardrail", required=True, type=Path)
    p.add_argument("--output", type=Path,
                   default=REPO / "src/analysis/figure_scripts/assets/arch_panels.npz")
    p.add_argument("--backbone", default="nvidia/mit-b1",
                   help="HF id or local snapshot dir for the student backbone")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src.eval.full_eval import (EvalConfig, build_guardrail_model,
                                    build_student_model)

    backbone = args.backbone
    if not Path(backbone).is_dir():
        from huggingface_hub import snapshot_download
        backbone = snapshot_download(backbone)

    cfg = EvalConfig(
        dataset_name="acdc", dataset_path="", split="val", domain="all",
        batch_size=1, num_workers=0, num_classes=19, device=args.device,
        student_backbone=backbone, teacher_backbone=None, temperature=1.0,
    )
    student = build_student_model(cfg, str(args.student))
    guard = build_guardrail_model(cfg, str(args.guardrail))
    uses_feat = bool(getattr(guard, "_use_student_features", False))

    img = Image.open(args.image).convert("RGB")
    resized = TF.resize(img, [512, 512],
                        interpolation=TF.InterpolationMode.BILINEAR)
    x = TF.normalize(TF.to_tensor(resized), MEAN, STD).unsqueeze(0).to(args.device)

    with torch.no_grad():
        if uses_feat:
            logits, feat = student(x, return_features=True)
        else:
            logits, feat = student(x), None
        out = guard(logits, student_features=feat if uses_feat else None)
        d = torch.sigmoid(out["disagree_logits"])[0].cpu().numpy()
        r = out["gap_pred"][0].cpu().numpy()
        e = (-torch.logsumexp(logits[0], dim=0)).cpu().numpy()

    half = lambda a: a.reshape(256, 2, 256, 2).mean(axis=(1, 3)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rgb=np.asarray(resized, np.uint8)[::2, ::2],
                        d=half(d), r=half(r), e=half(e))
    print(f"[done] {args.output}  d=[{d.min():.3f},{d.max():.3f}] "
          f"r=[{r.min():.3f},{r.max():.3f}] e=[{e.min():.3f},{e.max():.3f}]")


if __name__ == "__main__":
    main()
