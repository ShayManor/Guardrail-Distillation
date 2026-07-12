"""Stage 4: train the GuardrailPlusHead on a frozen student + teacher.

Primary supervision is per-pixel teacher/student disagreement (BCE) plus a
per-pixel signed risk-gap regression (smooth-L1). The scalar image-level
benefit head is wired in only as the negative-result baseline; the GT-target
modes (gt_disagree, gt_risk) drop the teacher and use ground-truth labels.
"""

import random
import time

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from losses import GuardrailPlusLoss
from utils import MetricTracker, build_scheduler
from _wandb_helpers import wandb_log, log_system_metrics

IGNORE_INDEX = 255

# Online corruption families (loosely ImageNet-C aligned). Clean Cityscapes
# alone has too little student/teacher disagreement to fit a selective-
# prediction head, so we synthesise OOD-ish inputs at train time.
CORRUPTION_FAMILIES = ("underexposure", "motion_blur", "noise", "fog")


def _apply_corruption(imgs, family, severity):
    """Apply one corruption family to a batch. severity in [0, 1]."""
    severity = float(max(0.0, min(1.0, severity)))
    if severity == 0.0:
        return imgs
    x = imgs

    if family == "underexposure":
        return x * (1.0 - 0.95 * severity)

    if family == "motion_blur":
        k = 3 + int(round(12 * severity))
        if k % 2 == 0:
            k += 1
        kernel = torch.zeros(1, 1, 1, k, device=x.device)
        kernel[..., :] = 1.0 / k
        B, C, H, W = x.shape
        x_pad = F.pad(x.reshape(B * C, 1, H, W), (k // 2, k // 2, 0, 0), mode="reflect")
        return F.conv2d(x_pad, kernel).reshape(B, C, H, W)

    if family == "noise":
        sigma = 0.02 + 0.28 * severity
        gaussian = torch.randn_like(x) * sigma
        shot = torch.randn_like(x) * (x.abs().sqrt() * 0.5 * severity)
        return x + gaussian + shot

    if family == "fog":
        B, C, H, W = x.shape
        density = 0.1 + 0.7 * severity
        depth = (
            torch.linspace(1.0, 0.2, H, device=x.device)
            .view(1, 1, H, 1)
            .expand(B, 1, H, W)
        )
        transmission = torch.exp(-density * depth * 3.0)
        return x * transmission + 0.8 * (1.0 - transmission)

    return x


def _valid_mean(v, valid_mask):
    denom = valid_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
    return (v * valid_mask.float()).sum(dim=(1, 2)) / denom


def _teacher_benefit_scalar(student_logits, teacher_logits, labels):
    ignore = labels == IGNORE_INDEX
    safe = labels.clone()
    safe[ignore] = 0
    student_ce = F.cross_entropy(student_logits, safe, reduction="none")
    teacher_ce = F.cross_entropy(teacher_logits, safe, reduction="none")
    valid = ~ignore
    return (_valid_mean(student_ce, valid) - _valid_mean(teacher_ce, valid)).clamp(min=0.0)


def _teacher_disagreement_map(student_logits, teacher_logits, labels):
    student_pred = student_logits.argmax(dim=1)
    teacher_pred = teacher_logits.argmax(dim=1)
    disagree = (student_pred != teacher_pred).float()
    valid = (labels != IGNORE_INDEX).float()
    return disagree, valid


def _teacher_risk_gap_map(student_logits, teacher_logits, labels):
    """Signed gap = student_ce − teacher_ce, no clamping."""
    ignore = labels == IGNORE_INDEX
    safe = labels.clone()
    safe[ignore] = 0
    student_ce = F.cross_entropy(student_logits, safe, reduction="none")
    teacher_ce = F.cross_entropy(teacher_logits, safe, reduction="none")
    return student_ce - teacher_ce, (~ignore).float()


def _gt_disagreement_map(student_logits, labels):
    """student_argmax ≠ ground_truth — teacher-free counterpart of the BCE target."""
    student_pred = student_logits.argmax(dim=1)
    valid = (labels != IGNORE_INDEX).float()
    disagree = (student_pred != labels).float() * valid
    return disagree, valid


def _gt_risk_map(student_logits, labels):
    """Per-pixel student CE against GT — teacher-free counterpart of the gap target."""
    ignore = labels == IGNORE_INDEX
    safe = labels.clone()
    safe[ignore] = 0
    student_ce = F.cross_entropy(student_logits, safe, reduction="none")
    valid = (~ignore).float()
    return student_ce * valid, valid


def _build_targets(student_logits, teacher_logits, labels, supervision_type="dense_multi"):
    """Build every target the loss might consume.

    GT modes ignore the teacher; teacher modes use both. utility_target is a
    placeholder zero for GT modes since the scalar head is dead under those.
    """
    valid = (labels != IGNORE_INDEX).float()

    if supervision_type in ("gt_disagree", "gt_risk"):
        gt_disagree_target, _ = _gt_disagreement_map(student_logits, labels)
        gt_risk_target, _ = _gt_risk_map(student_logits, labels)
        return {
            "utility_target": torch.zeros(student_logits.shape[0], device=student_logits.device),
            "disagree_target": gt_disagree_target,
            "disagree_valid": valid,
            "gap_target": gt_risk_target,
            "gap_valid": valid,
        }

    # Teacher-based modes
    disagree_target, _ = _teacher_disagreement_map(
        student_logits, teacher_logits, labels
    )
    gap_signed, _ = _teacher_risk_gap_map(student_logits, teacher_logits, labels)
    gap_signed = gap_signed * valid  # zero-out ignored pixels

    utility_target = _teacher_benefit_scalar(
        student_logits, teacher_logits, labels
    ).clamp(0.0, 1.0)

    return {
        "utility_target": utility_target,
        "disagree_target": disagree_target,
        "disagree_valid": valid,
        "gap_target": gap_signed,
        "gap_valid": valid,
    }


def train_guardrail(
    guardrail,
    student,
    teacher,
    train_loader,
    val_loader,
    cfg,
    use_student_features=True,
    global_step=0,
):
    """Returns (best_checkpoint_path, global_step). Student + teacher stay frozen."""
    print("\n" + "=" * 60)
    print(f"STAGE 4: Guardrail training  supervision_type={cfg.supervision_type}")
    print("=" * 60)

    device = cfg.device
    guardrail = guardrail.to(device).train()
    student = student.to(device).eval()
    teacher = teacher.to(device).eval()

    for p in student.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    criterion = GuardrailPlusLoss(
        supervision_type=cfg.supervision_type,
        dense_disagree_weight=cfg.dense_disagree_weight,
        dense_gap_weight=cfg.dense_gap_weight,
        scalar_weight=cfg.scalar_benefit_weight,
    )

    optimizer = torch.optim.AdamW(
        guardrail.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs_guardrail * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_loss = float("inf")
    best_path = f"{cfg.output_dir}/guardrail.ckpt"

    corruption_prob = float(cfg.corruption_prob)
    log_interval_start = time.time()

    for epoch in range(cfg.epochs_guardrail):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            if corruption_prob > 0:
                for i in range(imgs.shape[0]):
                    if random.random() < corruption_prob:
                        fam = random.choice(CORRUPTION_FAMILIES)
                        sev = 0.2 + 0.6 * random.random()
                        imgs[i] = _apply_corruption(imgs[i : i + 1], fam, sev).squeeze(0)

            with autocast("cuda", enabled=cfg.fp16):
                with torch.no_grad():
                    if use_student_features:
                        student_logits, student_feat = student(
                            imgs, return_features=True
                        )
                    else:
                        student_logits = student(imgs)
                        student_feat = None
                    teacher_logits = teacher(imgs)

                targets = _build_targets(
                    student_logits, teacher_logits, lbls,
                    supervision_type=cfg.supervision_type,
                )
                preds = guardrail(student_logits, student_feat)
                loss, loss_info = criterion(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            grad_norm = None
            if (step + 1) % cfg.log_every == 0:
                _sc = scaler.get_scale()
                _gn = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in guardrail.parameters()
                    if p.grad is not None
                )
                grad_norm = (_gn**0.5) / max(_sc, 1e-12)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            tracker.update("loss", loss.item())
            for k, v in loss_info.items():
                tracker.update(k, v)

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                elapsed = time.time() - log_interval_start
                parts = " ".join(f"{k}={v:.4f}" for k, v in s.items())
                print(
                    f"  [Guard] Epoch {epoch+1}/{cfg.epochs_guardrail} "
                    f"Step {step+1} {parts}"
                )

                wb = {f"guardrail/{k}": v for k, v in s.items()}
                wb["guardrail/learning_rate"] = optimizer.param_groups[0]["lr"]
                wb["guardrail/epoch"] = epoch + 1
                wb["guardrail/grad_norm"] = grad_norm
                wb["guardrail/grad_scaler"] = scaler.get_scale()
                wb["perf/step_time_sec"] = elapsed / cfg.log_every
                wb["perf/throughput_img_per_sec"] = (
                    cfg.log_every * cfg.batch_size / max(elapsed, 1e-6)
                )

                dt = targets["disagree_target"]
                dv = targets["disagree_valid"]
                denom = dv.sum().clamp(min=1.0)
                wb["guardrail/disagree_target_rate"] = float((dt * dv).sum().item() / denom.item())
                wb["guardrail/disagree_pred_mean"] = float(
                    torch.sigmoid(preds["disagree_logits"]).mean().item()
                )
                wb["guardrail/gap_target_mean"] = float(
                    (targets["gap_target"] * targets["gap_valid"]).sum().item() / denom.item()
                )
                wb["guardrail/gap_pred_mean"] = float(preds["gap_pred"].mean().item())
                wb["guardrail/gap_pred_std"] = float(preds["gap_pred"].std().item())
                wb["guardrail/utility_target_mean"] = float(targets["utility_target"].mean().item())
                wb["guardrail/utility_pred_mean"] = float(preds["utility_score"].mean().item())

                wandb_log(wb, step=global_step)
                log_system_metrics(global_step)
                log_interval_start = time.time()

        val_loss = _eval_guardrail(
            guardrail, student, teacher, val_loader, cfg, use_student_features
        )
        print(f"  [Guard] Epoch {epoch+1} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "model": guardrail.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "supervision_type": cfg.supervision_type,
                    "use_student_features": bool(use_student_features),
                    "use_confidence_features": bool(
                        getattr(guardrail, "use_confidence_features", False)),
                    "dense_disagree_weight": cfg.dense_disagree_weight,
                    "dense_gap_weight": cfg.dense_gap_weight,
                    "seed": cfg.seed,
                },
                best_path,
            )
            print(f"  → Saved guardrail: {best_path}")

        wandb_log(
            {
                "guardrail/val_loss": val_loss,
                "guardrail/best_val_loss": best_loss,
                "guardrail/epoch": epoch + 1,
            },
            step=global_step,
        )

    print(f"  [Guard] Best val loss={best_loss:.4f}")
    return best_path, global_step


@torch.no_grad()
def _eval_guardrail(guardrail, student, teacher, val_loader, cfg, use_student_features):
    guardrail.eval()
    criterion = GuardrailPlusLoss(
        supervision_type=cfg.supervision_type,
        dense_disagree_weight=cfg.dense_disagree_weight,
        dense_gap_weight=cfg.dense_gap_weight,
        scalar_weight=cfg.scalar_benefit_weight,
    )

    total_loss = 0.0
    count = 0

    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(cfg.device), lbls.to(cfg.device)

        if use_student_features:
            student_logits, student_feat = student(imgs, return_features=True)
        else:
            student_logits = student(imgs)
            student_feat = None
        teacher_logits = teacher(imgs)

        targets = _build_targets(
            student_logits, teacher_logits, lbls,
            supervision_type=cfg.supervision_type,
        )
        preds = guardrail(student_logits, student_feat)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

    guardrail.train()
    return total_loss / max(count, 1)
