#!/usr/bin/env python3
"""
Guardrail Distillation Pipeline
================================
Usage:
    python train.py \
        --dataset_path /data/cityscapes \
        --teacher_arch resnet101 \
        --student_arch mobilenet \
        --epochs_sup 100 --epochs_kd 100 --epochs_skd 100 --epochs_guardrail 50

    # Or with a HuggingFace dataset:
    python train.py --dataset_path hf://scene_parse_150

Stages:
    1. train student_sup        (supervised baseline)
    2. train student_kd         (+ KL distillation from teacher)
    3. train student_skd        (+ structural pairwise affinity)
    4. train guardrail          (teacher-grounded risk prediction)

All stages can be individually skipped with --skip_sup, --skip_kd, etc.
"""

import argparse
import os
import sys
import torch

from config import Config
from models import build_teacher, build_student, GuardrailHead
from data import build_dataloaders
from utils import load_checkpoint, timer

from train_supervised import train_supervised
from train_kd import train_kd
from train_skd import train_skd
from train_guardrail import train_guardrail


def parse_args():
    p = argparse.ArgumentParser(description="Guardrail Distillation Pipeline")

    # ── Required ──
    p.add_argument("--dataset_path", type=str, required=True,
                    help="Path to Cityscapes root or hf://dataset_name")

    # ── Model ──
    p.add_argument("--teacher_arch", type=str, default="resnet101")
    p.add_argument("--student_arch", type=str, default="mobilenet")
    p.add_argument("--teacher_ckpt", type=str, default=None,
                    help="Path to pre-trained teacher checkpoint (if None, uses torchvision pretrained)")
    p.add_argument("--num_classes", type=int, default=19)

    # ── Training ──
    p.add_argument("--epochs_sup", type=int, default=100)
    p.add_argument("--epochs_kd", type=int, default=100)
    p.add_argument("--epochs_skd", type=int, default=100)
    p.add_argument("--epochs_guardrail", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")

    # ── Loss weights ──
    p.add_argument("--alpha_ce", type=float, default=1.0)
    p.add_argument("--alpha_dice", type=float, default=0.5)
    p.add_argument("--alpha_kd", type=float, default=1.0)
    p.add_argument("--alpha_struct", type=float, default=0.5)
    p.add_argument("--kd_temperature", type=float, default=4.0)

    # ── Guardrail ──
    p.add_argument("--guardrail_mode", type=str, default="gap",
                    choices=["gap", "binary", "both"])
    p.add_argument("--guardrail_use_features", action="store_true",
                    help="Feed student backbone features to guardrail (heavier but richer)")

    # ── Data ──
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)

    # ── Skip stages ──
    p.add_argument("--skip_sup", action="store_true")
    p.add_argument("--skip_kd", action="store_true")
    p.add_argument("--skip_skd", action="store_true")
    p.add_argument("--skip_guardrail", action="store_true")

    # ── Resume / pre-trained student ──
    p.add_argument("--student_ckpt", type=str, default=None,
                    help="Pre-trained student checkpoint to skip sup/kd/skd and go straight to guardrail")

    # ── Misc ──
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1)

    args = p.parse_args()
    if args.no_fp16:
        args.fp16 = False
    return args


def args_to_config(args) -> Config:
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()
    cfg = args_to_config(args)

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    cfg.device = device
    print(f"Device: {device}")

    # ── Build data ──
    with timer("Data loading"):
        train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # ── Build models ──
    teacher = build_teacher(cfg.teacher_arch, cfg.num_classes)
    if args.teacher_ckpt:
        load_checkpoint(teacher, args.teacher_ckpt, device=device)
    teacher.to(device).eval()
    print(f"Teacher: {cfg.teacher_arch} ({sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params)")

    def fresh_student():
        return build_student(cfg.student_arch, cfg.num_classes)

    student_params = sum(p.numel() for p in fresh_student().parameters())
    print(f"Student: {cfg.student_arch} ({student_params/1e6:.1f}M params)")

    # ── Track best student checkpoint across stages ──
    best_student_path = args.student_ckpt  # can be None

    # ── Stage 1: Supervised ──
    if not args.skip_sup and not args.student_ckpt:
        student_sup = fresh_student()
        with timer("Stage 1 (Supervised)"):
            path_sup = train_supervised(student_sup, train_loader, val_loader, cfg)
        best_student_path = path_sup
        del student_sup
        torch.cuda.empty_cache()

    # ── Stage 2: KD ──
    if not args.skip_kd and not args.student_ckpt:
        student_kd = fresh_student()
        with timer("Stage 2 (KD)"):
            path_kd = train_kd(student_kd, teacher, train_loader, val_loader, cfg)
        # Keep whichever is better (compare mIoU from checkpoints)
        best_student_path = _pick_best(best_student_path, path_kd, device)
        del student_kd
        torch.cuda.empty_cache()

    # ── Stage 3: Structured KD ──
    if not args.skip_skd and not args.student_ckpt:
        student_skd = fresh_student()
        with timer("Stage 3 (Structured KD)"):
            path_skd = train_skd(student_skd, teacher, train_loader, val_loader, cfg)
        best_student_path = _pick_best(best_student_path, path_skd, device)
        del student_skd
        torch.cuda.empty_cache()

    # ── Stage 4: Guardrail ──
    if not args.skip_guardrail:
        assert best_student_path is not None, (
            "No student checkpoint available for guardrail training. "
            "Run at least one student stage or pass --student_ckpt."
        )

        print(f"\nUsing student checkpoint for guardrail: {best_student_path}")
        best_student = fresh_student()
        load_checkpoint(best_student, best_student_path, device=device)
        best_student.to(device).eval()

        # Determine guardrail input channels
        feat_channels = 0
        if args.guardrail_use_features:
            # Probe feature dim
            with torch.no_grad():
                dummy = torch.randn(1, 3, cfg.crop_size, cfg.crop_size, device=device)
                _, feat = best_student(dummy, return_features=True)
                feat_channels = feat.shape[1]
            print(f"Guardrail uses student features ({feat_channels} channels)")

        guardrail = GuardrailHead(
            num_classes=cfg.num_classes,
            feat_channels=feat_channels,
            mode=cfg.guardrail_mode,
        )

        with timer("Stage 4 (Guardrail)"):
            path_guard = train_guardrail(
                guardrail, best_student, teacher,
                train_loader, val_loader, cfg,
                use_student_features=args.guardrail_use_features,
            )

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {cfg.output_dir}")
    for f in sorted(os.listdir(cfg.output_dir)):
        fpath = os.path.join(cfg.output_dir, f)
        size = os.path.getsize(fpath) / 1e6
        print(f"  {f} ({size:.1f} MB)")


def _pick_best(path_a, path_b, device):
    """Return the checkpoint path with higher mIoU."""
    if path_a is None:
        return path_b
    if path_b is None:
        return path_a

    ckpt_a = torch.load(path_a, map_location=device, weights_only=False)
    ckpt_b = torch.load(path_b, map_location=device, weights_only=False)
    miou_a = ckpt_a.get("miou", 0)
    miou_b = ckpt_b.get("miou", 0)
    winner = path_a if miou_a >= miou_b else path_b
    print(f"  Best student: {winner} (mIoU={max(miou_a, miou_b):.4f})")
    return winner


if __name__ == "__main__":
    main()