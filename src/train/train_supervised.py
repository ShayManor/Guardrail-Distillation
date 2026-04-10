"""Stage 1: Train student with supervised loss (CE + Dice)."""

import time

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou
from wandb_utils import wandb_log, log_system_metrics


def train_supervised(student, train_loader, val_loader, cfg, global_step=0):
    """Train student_sup. Returns (path, global_step)."""
    print("\n" + "=" * 60)
    print("STAGE 1: Supervised Student Training")
    print("=" * 60)

    device = cfg.device
    student = student.to(device).train()

    class_weights = torch.ones(cfg.num_classes, device=device)
    rare_classes = [3, 4, 5, 6, 12, 14, 15, 16, 17]
    # 3=wall, 4=fence, 5=pole, 6=traffic light, 12=rider, 14=truck, 15=bus, 16=train, 17=motorcycle
    for c in rare_classes:
        class_weights[c] = 3.0
    criterion = SegLoss(alpha_ce=cfg.alpha_ce, alpha_dice=cfg.alpha_dice, class_weights=class_weights)
    backbone_params = [p for n, p in student.named_parameters() if "decode_head" not in n and p.requires_grad]
    head_params = [p for n, p in student.named_parameters() if "decode_head" in n and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr},
        {"params": head_params, "lr": cfg.lr * 20},
    ], weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_sup * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_sup.ckpt"
    log_interval_start = time.time()

    for epoch in range(cfg.epochs_sup):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                logits = student(imgs)
                loss = criterion(logits, lbls)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient norm (unscaled) for monitoring
            grad_norm = None
            if (step + 1) % cfg.log_every == 0:
                _s = scaler.get_scale()
                _gn = sum(p.grad.data.norm(2).item() ** 2 for p in student.parameters() if p.grad is not None)
                grad_norm = (_gn ** 0.5) / max(_s, 1e-12)

            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() == prev_scale:
                scheduler.step()
            global_step += 1

            tracker.update("loss", loss.item())
            preds = logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                elapsed = time.time() - log_interval_start
                print(f"  [Sup] Epoch {epoch+1}/{cfg.epochs_sup} Step {step+1} "
                      f"loss={s['loss']:.4f} mIoU={s['miou']:.4f}")
                wandb_log({
                    "supervised/loss": s["loss"],
                    "supervised/miou": s["miou"],
                    "supervised/learning_rate": optimizer.param_groups[0]["lr"],
                    "supervised/epoch": epoch + 1,
                    "supervised/grad_norm": grad_norm,
                    "supervised/grad_scaler": scaler.get_scale(),
                    "perf/step_time_sec": elapsed / cfg.log_every,
                    "perf/throughput_img_per_sec": cfg.log_every * cfg.batch_size / max(elapsed, 1e-6),
                }, step=global_step)
                log_system_metrics(global_step)
                log_interval_start = time.time()

        # Eval
        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [Sup] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)
            wandb_log({
                "supervised/val_miou": miou,
                "supervised/best_val_miou": best_miou,
                "supervised/epoch": epoch + 1,
            }, step=global_step)

    print(f"  [Sup] Best val mIoU={best_miou:.4f}")
    return best_path, global_step