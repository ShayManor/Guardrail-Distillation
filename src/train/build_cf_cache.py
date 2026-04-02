"""Build cached utility / counterfactual margin targets for Guardrail++."""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from config import Config
from data import build_dataloaders
from models import HFSegModelWrapper
from train_guardrail import _apply_corruption, _teacher_benefit_weighted, _teacher_benefit


def parse_args():
    p = argparse.ArgumentParser(description="Build Guardrail++ target cache")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--teacher-model", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    p.add_argument("--student-model", default="nvidia/mit-b0")
    p.add_argument("--student-ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cf-delta", type=float, default=0.05)
    p.add_argument("--cf-severities", default="0.0,0.25,0.5,0.75,1.0")
    return p.parse_args()


def make_student(model_name, num_classes, ckpt_path, device):
    from transformers import AutoModelForSemanticSegmentation, SegformerConfig, SegformerForSemanticSegmentation
    from utils import load_checkpoint

    backbone = AutoModelForSemanticSegmentation.from_pretrained(model_name, local_files_only=True)
    s_cfg = SegformerConfig.from_pretrained(model_name, local_files_only=True)
    s_cfg.num_labels = num_classes
    student = SegformerForSemanticSegmentation(s_cfg)
    student.segformer.load_state_dict(backbone.base_model.state_dict(), strict=False)
    wrapped = HFSegModelWrapper(student, num_classes)
    load_checkpoint(wrapped, ckpt_path, device=device)
    return wrapped.to(device).eval()


def make_teacher(model_name, num_classes, device):
    from transformers import AutoModelForSemanticSegmentation

    raw = AutoModelForSemanticSegmentation.from_pretrained(model_name, local_files_only=True)
    return HFSegModelWrapper(raw, num_classes).to(device).eval()


def main():
    args = parse_args()
    severities = [float(x.strip()) for x in args.cf_severities.split(",") if x.strip()]

    cfg = Config(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir="unused",
        device=args.device,
    )
    train_loader, _ = build_dataloaders(cfg)

    student = make_student(args.student_model, cfg.num_classes, args.student_ckpt, cfg.device)
    teacher = make_teacher(args.teacher_model, cfg.num_classes, cfg.device)

    families = ["underexposure", "motion_blur", "noise", "fog"]
    rows = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc="cache")):
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            s_logits = student(imgs)
            t_logits = teacher(imgs)

            utility = _teacher_benefit_weighted(s_logits, t_logits, labels, weights=(0.5, 0.25, 0.25)).clamp(0.0, 1.0)
            bsz = imgs.shape[0]
            margins = torch.ones((bsz, len(families)), device=cfg.device)

            for k, fam in enumerate(families):
                crossed = torch.zeros(bsz, device=cfg.device, dtype=torch.bool)
                for sev in severities:
                    cor = _apply_corruption(imgs, fam, sev)
                    s_cor = student(cor)
                    t_cor = teacher(cor)
                    benefit, _, _ = _teacher_benefit(s_cor, t_cor, labels)
                    hit = benefit >= args.cf_delta
                    update = (~crossed) & hit
                    margins[update, k] = sev
                    crossed = crossed | hit

            for i in range(bsz):
                rows.append({
                    "batch_idx": batch_idx,
                    "item_idx": i,
                    "utility_target": float(utility[i].item()),
                    "margin_vec": margins[i].cpu(),
                })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": rows, "families": families, "severities": severities, "cf_delta": args.cf_delta}, out)
    print(f"Saved cache: {out} ({len(rows)} samples)")


if __name__ == "__main__":
    main()
