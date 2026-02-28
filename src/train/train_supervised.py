"""Stage 1: Train student with supervised loss (CE + Dice)."""

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou


def train_supervised(student, train_loader, val_loader, cfg):
    """Train student_sup. Returns path to best checkpoint."""
    print("\n" + "=" * 60)
    print("STAGE 1: Supervised Student Training")
    print("=" * 60)

    device = cfg.device
    student = student.to(device).train()

    criterion = SegLoss(alpha_ce=cfg.alpha_ce, alpha_dice=cfg.alpha_dice)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_sup * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_sup.ckpt"

    for epoch in range(cfg.epochs_sup):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                logits = student(imgs)
                loss = criterion(logits, lbls)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == prev_scale:
                scheduler.step()

            tracker.update("loss", loss.item())
            preds = logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                print(f"  [Sup] Epoch {epoch+1}/{cfg.epochs_sup} Step {step+1} "
                      f"loss={s['loss']:.4f} mIoU={s['miou']:.4f}")

        # Eval
        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [Sup] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)

    print(f"  [Sup] Best val mIoU={best_miou:.4f}")
    return best_path