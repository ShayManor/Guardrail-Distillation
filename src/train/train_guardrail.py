"""Stage 4: Train guardrail head on frozen student with teacher-grounded labels."""

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from losses import GuardrailLoss, GuardrailPlusLoss, compute_guardrail_targets
from utils import MetricTracker, build_scheduler

IGNORE_INDEX = 255
DYNAMIC_CLASS_IDS = [11, 12, 13, 14, 15, 17, 18]


def _apply_corruption(imgs, family, severity):
    """Differentiation-free corruption operator for counterfactual target construction."""
    severity = float(max(0.0, min(1.0, severity)))
    x = imgs

    if family == "underexposure":
        # Darken image in normalized space.
        factor = 1.0 - 0.7 * severity
        return x * factor

    if family == "motion_blur":
        # Approximate blur using average pooling then blend.
        k = 3 + int(round(6 * severity))
        k = k + 1 if k % 2 == 0 else k
        blurred = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        return (1.0 - severity) * x + severity * blurred

    if family == "noise":
        sigma = 0.05 + 0.25 * severity
        noise = torch.randn_like(x) * sigma
        return x + noise

    if family == "fog":
        # Low-contrast haze blend.
        fog_level = 0.15 + 0.55 * severity
        gray = x.mean(dim=1, keepdim=True)
        return (1.0 - fog_level) * x + fog_level * gray

    return x


def _valid_mean(v, valid_mask):
    denom = valid_mask.float().sum(dim=(1, 2)).clamp(min=1.0)
    return (v * valid_mask.float()).sum(dim=(1, 2)) / denom


def _teacher_benefit(student_logits, teacher_logits, labels):
    ignore = labels == IGNORE_INDEX
    safe = labels.clone()
    safe[ignore] = 0

    student_ce = F.cross_entropy(student_logits, safe, reduction="none")
    teacher_ce = F.cross_entropy(teacher_logits, safe, reduction="none")

    valid = ~ignore
    student_risk = _valid_mean(student_ce, valid)
    teacher_risk = _valid_mean(teacher_ce, valid)
    benefit = (student_risk - teacher_risk).clamp(min=0.0)
    return benefit, student_ce, teacher_ce


def _teacher_benefit_weighted(student_logits, teacher_logits, labels, weights):
    w0, w1, w2 = weights
    benefit, student_ce, teacher_ce = _teacher_benefit(student_logits, teacher_logits, labels)

    valid = labels != IGNORE_INDEX

    h = labels.shape[-2]
    roi_start = int(h * (2.0 / 3.0))
    near_mask = torch.zeros_like(valid)
    near_mask[:, roi_start:, :] = True
    near_mask = near_mask & valid

    dyn_mask = torch.zeros_like(valid)
    for c in DYNAMIC_CLASS_IDS:
        dyn_mask |= (labels == c)
    dyn_mask = dyn_mask & valid

    near_benefit = (
        _valid_mean(student_ce, near_mask) - _valid_mean(teacher_ce, near_mask)
    ).clamp(min=0.0)
    dyn_benefit = (
        _valid_mean(student_ce, dyn_mask) - _valid_mean(teacher_ce, dyn_mask)
    ).clamp(min=0.0)

    utility = (w0 * benefit + w1 * near_benefit + w2 * dyn_benefit).clamp(min=0.0)
    return utility


def _build_guardrailpp_targets(student, teacher, imgs, labels, cfg):
    with torch.no_grad():
        student_logits = student(imgs)
        teacher_logits = teacher(imgs)

    utility_target = _teacher_benefit_weighted(
        student_logits,
        teacher_logits,
        labels,
        weights=(cfg.utility_w0, cfg.utility_w1, cfg.utility_w2),
    )

    out = {"utility_target": utility_target.clamp(0.0, 1.0)}

    if cfg.guardrail_mode in ("margin", "guardrailpp"):
        families = ["underexposure", "motion_blur", "noise", "fog"]
        severity_grid = cfg.cf_severities
        margins = []

        for fam in families:
            crossed = torch.zeros(imgs.shape[0], device=imgs.device, dtype=torch.bool)
            fam_margin = torch.ones(imgs.shape[0], device=imgs.device)

            for sev in severity_grid:
                corrupted = _apply_corruption(imgs, fam, sev)
                with torch.no_grad():
                    s_cor = student(corrupted)
                    t_cor = teacher(corrupted)
                benefit_cor, _, _ = _teacher_benefit(s_cor, t_cor, labels)
                hit = benefit_cor >= cfg.cf_delta
                update_mask = (~crossed) & hit
                fam_margin[update_mask] = float(sev)
                crossed = crossed | hit

            margins.append(fam_margin)

        margin_target = torch.stack(margins, dim=1)
        out["margin_target"] = margin_target
        out["family_target"] = torch.argmin(margin_target, dim=1)

    return out


def train_guardrail(guardrail, student, teacher, train_loader, val_loader, cfg,
                    use_student_features=False):
    """Train guardrail or Guardrail++ head. Student and teacher are frozen."""
    print("\n" + "=" * 60)
    print("STAGE 4: Guardrail Training")
    print("=" * 60)

    device = cfg.device
    guardrail = guardrail.to(device).train()
    student = student.to(device).eval()
    teacher = teacher.to(device).eval()

    for p in student.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    plus_modes = {"utility", "margin", "guardrailpp"}
    if cfg.guardrail_mode in plus_modes:
        criterion = GuardrailPlusLoss(
            utility_weight=cfg.utility_loss_weight,
            margin_weight=cfg.margin_loss_weight,
            family_weight=cfg.family_loss_weight,
            margin_loss=cfg.margin_loss,
        )
    else:
        criterion = GuardrailLoss(mode=cfg.guardrail_mode)

    optimizer = torch.optim.AdamW(guardrail.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_guardrail * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_loss = float("inf")
    best_path = f"{cfg.output_dir}/guardrail.ckpt"

    for epoch in range(cfg.epochs_guardrail):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                with torch.no_grad():
                    if use_student_features:
                        student_logits, student_feat = student(imgs, return_features=True)
                    else:
                        student_logits = student(imgs)
                        student_feat = None

                    teacher_logits = teacher(imgs)

                if cfg.guardrail_mode in plus_modes:
                    targets = _build_guardrailpp_targets(student, teacher, imgs, lbls, cfg)
                else:
                    targets = compute_guardrail_targets(
                        student_logits, teacher_logits, lbls, mode=cfg.guardrail_mode
                    )

                preds = guardrail(student_logits, student_feat)
                loss, loss_info = criterion(preds, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tracker.update("loss", loss.item())
            for k, v in loss_info.items():
                tracker.update(k, v)

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                parts = " ".join(f"{k}={v:.4f}" for k, v in s.items())
                print(f"  [Guard] Epoch {epoch+1}/{cfg.epochs_guardrail} Step {step+1} {parts}")

        val_loss = _eval_guardrail(guardrail, student, teacher, val_loader, cfg,
                                   use_student_features)
        print(f"  [Guard] Epoch {epoch+1} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model": guardrail.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "guardrail_mode": cfg.guardrail_mode,
            }, best_path)
            print(f"  → Saved guardrail: {best_path}")

    print(f"  [Guard] Best val loss={best_loss:.4f}")
    return best_path


@torch.no_grad()
def _eval_guardrail(guardrail, student, teacher, val_loader, cfg, use_student_features):
    guardrail.eval()
    plus_modes = {"utility", "margin", "guardrailpp"}
    if cfg.guardrail_mode in plus_modes:
        criterion = GuardrailPlusLoss(
            utility_weight=cfg.utility_loss_weight,
            margin_weight=cfg.margin_loss_weight,
            family_weight=cfg.family_loss_weight,
            margin_loss=cfg.margin_loss,
        )
    else:
        criterion = GuardrailLoss(mode=cfg.guardrail_mode)

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

        if cfg.guardrail_mode in plus_modes:
            targets = _build_guardrailpp_targets(student, teacher, imgs, lbls, cfg)
        else:
            targets = compute_guardrail_targets(
                student_logits, teacher_logits, lbls, mode=cfg.guardrail_mode
            )

        preds = guardrail(student_logits, student_feat)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

    guardrail.train()
    return total_loss / max(count, 1)
