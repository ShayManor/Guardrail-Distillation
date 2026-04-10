"""Stage 3: Train student with structured KD (sup + KL + pairwise affinity)."""

import time

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss, KDLoss, PairwiseAffinityLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou
from _wandb_helpers import wandb_log, log_system_metrics


def train_skd(student, teacher, train_loader, val_loader, cfg, global_step=0):
    """Train student_skd. Returns (path, global_step)."""
    print("\n" + "=" * 60)
    print("STAGE 3: Structured Knowledge Distillation Training")
    print("=" * 60)

    device = cfg.device
    student = student.to(device).train()
    teacher = teacher.to(device).eval()

    for p in teacher.parameters():
        p.requires_grad = False

    seg_loss = SegLoss(alpha_ce=cfg.alpha_ce, alpha_dice=cfg.alpha_dice)
    kd_loss = KDLoss(temperature=cfg.kd_temperature)
    struct_loss = PairwiseAffinityLoss()
    backbone_params = [p for n, p in student.named_parameters() if "decode_head" not in n and p.requires_grad]
    head_params = [p for n, p in student.named_parameters() if "decode_head" in n and p.requires_grad]
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_skd * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_skd.ckpt"
    log_interval_start = time.time()

    for epoch in range(cfg.epochs_skd):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                student_logits, student_feat = student(imgs, return_features=True)
                with torch.no_grad():
                    teacher_logits, teacher_feat = teacher(imgs, return_features=True)

                l_sup = seg_loss(student_logits, lbls)
                l_kd = kd_loss(student_logits, teacher_logits)
                l_struct = struct_loss(student_feat, teacher_feat)
                loss = l_sup + cfg.alpha_kd * l_kd + cfg.alpha_struct * l_struct

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            grad_norm = None
            if (step + 1) % cfg.log_every == 0:
                _sc = scaler.get_scale()
                _gn = sum(p.grad.data.norm(2).item() ** 2 for p in student.parameters() if p.grad is not None)
                grad_norm = (_gn ** 0.5) / max(_sc, 1e-12)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            tracker.update("loss", loss.item())
            tracker.update("l_sup", l_sup.item())
            tracker.update("l_kd", l_kd.item())
            tracker.update("l_struct", l_struct.item())
            preds = student_logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                elapsed = time.time() - log_interval_start
                print(f"  [SKD] Epoch {epoch+1}/{cfg.epochs_skd} Step {step+1} "
                      f"loss={s['loss']:.4f} sup={s['l_sup']:.4f} kd={s['l_kd']:.4f} "
                      f"struct={s['l_struct']:.4f} mIoU={s['miou']:.4f}")
                wandb_log({
                    "skd/loss": s["loss"],
                    "skd/loss_supervised": s["l_sup"],
                    "skd/loss_kd": s["l_kd"],
                    "skd/loss_structural": s["l_struct"],
                    "skd/miou": s["miou"],
                    "skd/learning_rate": optimizer.param_groups[0]["lr"],
                    "skd/epoch": epoch + 1,
                    "skd/grad_norm": grad_norm,
                    "skd/grad_scaler": scaler.get_scale(),
                    "perf/step_time_sec": elapsed / cfg.log_every,
                    "perf/throughput_img_per_sec": cfg.log_every * cfg.batch_size / max(elapsed, 1e-6),
                }, step=global_step)
                log_system_metrics(global_step)
                log_interval_start = time.time()

        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [SKD] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)
            wandb_log({
                "skd/val_miou": miou,
                "skd/best_val_miou": best_miou,
                "skd/epoch": epoch + 1,
            }, step=global_step)

    print(f"  [SKD] Best val mIoU={best_miou:.4f}")
    return best_path, global_step
