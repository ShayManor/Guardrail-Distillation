"""Stage 4: Train guardrail head on frozen student with teacher-grounded labels."""

import random

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from losses import GuardrailLoss, GuardrailPlusLoss, compute_guardrail_targets
from utils import MetricTracker, build_scheduler

IGNORE_INDEX = 255
DYNAMIC_CLASS_IDS = [11, 12, 13, 14, 15, 17, 18]


def _apply_corruption(imgs, family, severity):
    """ImageNet-C / Cityscapes-C aligned corruption families for driving.

    Families map to physically motivated driving hazards:
      underexposure -> brightness (INet-C #10)
      motion_blur   -> motion_blur (INet-C #6)
      noise         -> gaussian_noise + shot_noise (INet-C #1-2)
      fog           -> fog (INet-C #9)
    """
    severity = float(max(0.0, min(1.0, severity)))
    if severity == 0.0:
        return imgs
    x = imgs

    if family == "underexposure":
        # Matches ImageNet-C brightness: multiply by factor in [0.05, 0.95]
        # severity 0->1 maps to factor 0.95->0.05
        factor = 1.0 - 0.95 * severity
        return x * factor

    if family == "motion_blur":
        # Directional blur approximation (horizontal, driving-relevant)
        # Kernel size scales with severity: 3->15
        k = 3 + int(round(12 * severity))
        k = k if k % 2 == 1 else k + 1
        # Horizontal 1D blur (motion along driving direction)
        kernel = torch.zeros(1, 1, 1, k, device=x.device)
        kernel[..., :] = 1.0 / k
        B, C, H, W = x.shape
        x_pad = F.pad(x.reshape(B * C, 1, H, W), (k // 2, k // 2, 0, 0), mode='reflect')
        blurred = F.conv2d(x_pad, kernel).reshape(B, C, H, W)
        return blurred

    if family == "noise":
        # Gaussian + shot noise blend (INet-C #1-2 aligned)
        sigma = 0.02 + 0.28 * severity  # range [0.02, 0.30]
        gaussian = torch.randn_like(x) * sigma
        # Shot noise component: Poisson-like via sqrt scaling
        shot_scale = 0.5 * severity
        shot = torch.randn_like(x) * (x.abs().sqrt() * shot_scale)
        return x + gaussian + shot

    if family == "fog":
        # INet-C fog: diamond-shaped fog density + atmospheric scattering
        B, C, H, W = x.shape
        fog_density = 0.1 + 0.7 * severity
        # Depth-dependent: thicker at top (far), thinner at bottom (near)
        depth = torch.linspace(1.0, 0.2, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        transmission = torch.exp(-fog_density * depth * 3.0)
        airlight = 0.8  # atmospheric light (gray)
        return x * transmission + airlight * (1.0 - transmission)

    return x


MARGIN_FAMILY_NAMES = ["underexposure", "motion_blur", "noise", "fog"]


def apply_adaptive_preprocessing(imgs, margin_vec, threshold=0.5):
    """Apply corrective preprocessing based on predicted vulnerability margins.

    Low margin for a family => apply the inverse correction for that family.
    Returns preprocessed images (no grad needed, runs at inference).
    """
    x = imgs.clone()
    # margin_vec: (B, num_families) in [0,1], low = vulnerable
    for k, family in enumerate(MARGIN_FAMILY_NAMES):
        vulnerable = margin_vec[:, k] < threshold  # (B,)
        if not vulnerable.any():
            continue
        strength = (threshold - margin_vec[:, k]).clamp(0, 1)  # higher = more correction

        if family == "underexposure":
            # Brighten: gamma correction approx
            factor = 1.0 + 0.5 * strength  # up to 1.5x
            x[vulnerable] = x[vulnerable] * factor[vulnerable, None, None, None]

        elif family == "motion_blur":
            # Sharpen via unsharp mask
            blurred = F.avg_pool2d(x[vulnerable], kernel_size=3, stride=1, padding=1)
            alpha = 0.3 * strength[vulnerable, None, None, None]
            x[vulnerable] = x[vulnerable] + alpha * (x[vulnerable] - blurred)

        elif family == "noise":
            # Denoise via light smoothing
            sigma = 0.5 * strength[vulnerable, None, None, None]
            smoothed = F.avg_pool2d(x[vulnerable], kernel_size=3, stride=1, padding=1)
            x[vulnerable] = (1 - sigma) * x[vulnerable] + sigma * smoothed

        elif family == "fog":
            # Dehaze: boost contrast
            factor = 1.0 + 0.4 * strength[vulnerable, None, None, None]
            mean = x[vulnerable].mean(dim=(2, 3), keepdim=True)
            x[vulnerable] = mean + factor * (x[vulnerable] - mean)

    return x.clamp(-3, 3)  # stay within normalized range

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


def _build_guardrailpp_targets(student, teacher, imgs, labels, cfg,
                               student_logits=None, teacher_logits=None):
    """Build Guardrail++ targets: utility, gap map, margin vector.

    Uses relative margin thresholds: the margin for a corruption family is the
    minimum severity at which the *additional* teacher benefit (above the current
    input's baseline) exceeds cf_delta. This prevents clean-data teacher benefit
    from trivially triggering the threshold at severity=0.
    """
    with torch.no_grad():
        if student_logits is None:
            student_logits = student(imgs)
        if teacher_logits is None:
            teacher_logits = teacher(imgs)

    utility_target = _teacher_benefit_weighted(
        student_logits,
        teacher_logits,
        labels,
        weights=(cfg.utility_w0, cfg.utility_w1, cfg.utility_w2),
    )

    out = {"utility_target": utility_target.clamp(0.0, 1.0)}
    ignore = labels == IGNORE_INDEX
    safe = labels.clone()
    safe[ignore] = 0
    s_ce = F.cross_entropy(student_logits, safe, reduction="none")
    t_ce = F.cross_entropy(teacher_logits, safe, reduction="none")
    gap = (s_ce - t_ce).clamp(min=0)
    gap[ignore] = 0
    gap = gap.clamp(max=5.0) / 5.0
    out["gap_map"] = gap

    if cfg.guardrail_mode in ("margin", "guardrailpp"):
        families = ["underexposure", "motion_blur", "noise", "fog"]
        severity_grid = cfg.cf_severities
        margins = []

        # Baseline benefit on the (possibly already corrupted) input
        baseline_benefit, _, _ = _teacher_benefit(student_logits, teacher_logits, labels)

        for fam in families:
            crossed = torch.zeros(imgs.shape[0], device=imgs.device, dtype=torch.bool)
            fam_margin = torch.ones(imgs.shape[0], device=imgs.device)

            for sev in severity_grid:
                if sev == 0.0:
                    continue  # skip clean — measured via baseline
                corrupted = _apply_corruption(imgs, fam, sev)
                with torch.no_grad():
                    s_cor = student(corrupted)
                    t_cor = teacher(corrupted)
                benefit_cor, _, _ = _teacher_benefit(s_cor, t_cor, labels)
                # Relative threshold: additional benefit above baseline
                additional_benefit = benefit_cor - baseline_benefit
                hit = additional_benefit >= cfg.cf_delta
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

    corruption_prob = getattr(cfg, "corruption_prob", 0.5)

    for epoch in range(cfg.epochs_guardrail):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Per-image online corruption augmentation for guardrail++ modes
            if cfg.guardrail_mode in plus_modes and corruption_prob > 0:
                for i in range(imgs.shape[0]):
                    if random.random() < corruption_prob:
                        fam = random.choice(MARGIN_FAMILY_NAMES)
                        sev = 0.2 + 0.6 * random.random()  # uniform [0.2, 0.8]
                        imgs[i] = _apply_corruption(
                            imgs[i:i+1], fam, sev
                        ).squeeze(0)

            with autocast("cuda", enabled=cfg.fp16):
                with torch.no_grad():
                    if use_student_features:
                        student_logits, student_feat = student(imgs, return_features=True)
                    else:
                        student_logits = student(imgs)
                        student_feat = None

                    teacher_logits = teacher(imgs)

                if cfg.guardrail_mode in plus_modes:
                    targets = _build_guardrailpp_targets(
                        student, teacher, imgs, lbls, cfg,
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                    )
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
    if cfg.guardrail_mode in plus_modes:
        from utils import load_checkpoint as _lc
        best_state = torch.load(best_path, map_location=device, weights_only=False)
        guardrail.load_state_dict(best_state["model"])
        calib_path = calibrate_guardrail(guardrail, student, teacher, val_loader, cfg, use_student_features)

    return best_path

def calibrate_guardrail(guardrail, student, teacher, val_loader, cfg, use_student_features=False):
    """Post-hoc isotonic calibration of utility scores against true teacher benefit."""
    from sklearn.isotonic import IsotonicRegression
    import pickle

    device = cfg.device
    guardrail.eval(); student.eval(); teacher.eval()

    pred_utils, true_utils = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            student_logits = student(imgs)
            student_feat = None
            if use_student_features:
                student_logits, student_feat = student(imgs, return_features=True)
            teacher_logits = teacher(imgs)

            preds = guardrail(student_logits, student_feat)
            true_benefit = _teacher_benefit_weighted(
                student_logits, teacher_logits, lbls,
                weights=(cfg.utility_w0, cfg.utility_w1, cfg.utility_w2),
            ).clamp(0.0, 1.0)

            pred_utils.append(preds["utility_score"].cpu())
            true_utils.append(true_benefit.cpu())

    pred_arr = torch.cat(pred_utils).numpy()
    true_arr = torch.cat(true_utils).numpy()

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(pred_arr, true_arr)

    calib_path = f"{cfg.output_dir}/guardrail_calibrator.pkl"
    with open(calib_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"  [Calibration] Saved isotonic calibrator: {calib_path}")
    return calib_path

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
            targets = _build_guardrailpp_targets(
                student, teacher, imgs, lbls, cfg,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
            )
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
