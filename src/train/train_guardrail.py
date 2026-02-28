"""Stage 4: Train guardrail head on frozen student with teacher-grounded labels."""

import torch
from torch.amp import GradScaler, autocast
from losses import GuardrailLoss, compute_guardrail_targets
from utils import MetricTracker, save_checkpoint, build_scheduler


def train_guardrail(guardrail, student, teacher, train_loader, val_loader, cfg,
                    use_student_features=False):
    """
    Train guardrail head. Student and teacher are frozen.

    Args:
        guardrail: GuardrailHead module
        student: frozen student model (best checkpoint)
        teacher: frozen teacher model
        use_student_features: if True, also feed backbone features to guardrail
    Returns:
        path to best guardrail checkpoint
    """
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

    criterion = GuardrailLoss(mode=cfg.guardrail_mode)
    optimizer = torch.optim.AdamW(guardrail.parameters(), lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay)
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
                # Generate inputs and targets
                with torch.no_grad():
                    if use_student_features:
                        student_logits, student_feat = student(imgs, return_features=True)
                    else:
                        student_logits = student(imgs)
                        student_feat = None
                    teacher_logits = teacher(imgs)

                targets = compute_guardrail_targets(
                    student_logits, teacher_logits, lbls, mode=cfg.guardrail_mode
                )

                # Forward guardrail
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

        # Validation: compute avg loss
        val_loss = _eval_guardrail(guardrail, student, teacher, val_loader, cfg,
                                    use_student_features)
        print(f"  [Guard] Epoch {epoch+1} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model": guardrail.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, best_path)
            print(f"  → Saved guardrail: {best_path}")

    print(f"  [Guard] Best val loss={best_loss:.4f}")
    return best_path


@torch.no_grad()
def _eval_guardrail(guardrail, student, teacher, val_loader, cfg, use_student_features):
    guardrail.eval()
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

        targets = compute_guardrail_targets(
            student_logits, teacher_logits, lbls, mode=cfg.guardrail_mode
        )
        preds = guardrail(student_logits, student_feat)
        loss, _ = criterion(preds, targets)
        total_loss += loss.item() * imgs.shape[0]
        count += imgs.shape[0]

    guardrail.train()
    return total_loss / max(count, 1)