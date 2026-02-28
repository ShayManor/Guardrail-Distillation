"""Stage 2: Train student with KD (supervised + KL distillation)."""

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss, KDLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou


def train_kd(student, teacher, train_loader, val_loader, cfg):
    """Train student_kd with teacher frozen. Returns path to best checkpoint."""
    print("\n" + "=" * 60)
    print("STAGE 2: Knowledge Distillation Training")
    print("=" * 60)

    device = cfg.device
    student = student.to(device).train()
    teacher = teacher.to(device).eval()

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    seg_loss = SegLoss(alpha_ce=cfg.alpha_ce, alpha_dice=cfg.alpha_dice)
    kd_loss = KDLoss(temperature=cfg.kd_temperature)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_kd * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_kd.ckpt"

    for epoch in range(cfg.epochs_kd):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                student_logits = student(imgs)
                with torch.no_grad():
                    teacher_logits = teacher(imgs)

                loss_sup = seg_loss(student_logits, lbls)
                loss_kd = kd_loss(student_logits, teacher_logits)
                loss = loss_sup + cfg.alpha_kd * loss_kd

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tracker.update("loss", loss.item())
            tracker.update("loss_sup", loss_sup.item())
            tracker.update("loss_kd", loss_kd.item())
            preds = student_logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                print(f"  [KD] Epoch {epoch+1}/{cfg.epochs_kd} Step {step+1} "
                      f"loss={s['loss']:.4f} sup={s['loss_sup']:.4f} kd={s['loss_kd']:.4f} "
                      f"mIoU={s['miou']:.4f}")

        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [KD] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)

    print(f"  [KD] Best val mIoU={best_miou:.4f}")
    return best_path