"""Stage 2: Train student with KD (supervised + KL distillation)."""

import time

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss, KDLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou
from wandb_utils import wandb_log, log_system_metrics


def train_kd(student, teacher, train_loader, val_loader, cfg, global_step=0):
    """Train student_kd with teacher frozen. Returns (path, global_step)."""
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
    backbone_params = [p for n, p in student.named_parameters() if "decode_head" not in n and p.requires_grad]
    head_params = [p for n, p in student.named_parameters() if "decode_head" in n and p.requires_grad]
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_kd * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_kd.ckpt"
    log_interval_start = time.time()

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

            grad_norm = None
            if (step + 1) % cfg.log_every == 0:
                _s = scaler.get_scale()
                _gn = sum(p.grad.data.norm(2).item() ** 2 for p in student.parameters() if p.grad is not None)
                grad_norm = (_gn ** 0.5) / max(_s, 1e-12)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            tracker.update("loss", loss.item())
            tracker.update("loss_sup", loss_sup.item())
            tracker.update("loss_kd", loss_kd.item())
            preds = student_logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                elapsed = time.time() - log_interval_start
                print(f"  [KD] Epoch {epoch+1}/{cfg.epochs_kd} Step {step+1} "
                      f"loss={s['loss']:.4f} sup={s['loss_sup']:.4f} kd={s['loss_kd']:.4f} "
                      f"mIoU={s['miou']:.4f}")
                wandb_log({
                    "kd/loss": s["loss"],
                    "kd/loss_supervised": s["loss_sup"],
                    "kd/loss_kd": s["loss_kd"],
                    "kd/miou": s["miou"],
                    "kd/learning_rate": optimizer.param_groups[0]["lr"],
                    "kd/epoch": epoch + 1,
                    "kd/grad_norm": grad_norm,
                    "kd/grad_scaler": scaler.get_scale(),
                    "perf/step_time_sec": elapsed / cfg.log_every,
                    "perf/throughput_img_per_sec": cfg.log_every * cfg.batch_size / max(elapsed, 1e-6),
                }, step=global_step)
                log_system_metrics(global_step)
                log_interval_start = time.time()

        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [KD] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)
            wandb_log({
                "kd/val_miou": miou,
                "kd/best_val_miou": best_miou,
                "kd/epoch": epoch + 1,
            }, step=global_step)

    print(f"  [KD] Best val mIoU={best_miou:.4f}")
    return best_path, global_step