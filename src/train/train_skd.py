"""Stage 3: Train student with structured KD (sup + KL + pairwise affinity)."""

import torch
from torch.amp import GradScaler, autocast
from losses import SegLoss, KDLoss, PairwiseAffinityLoss
from utils import MetricTracker, save_checkpoint, evaluate, build_scheduler, compute_miou


def train_skd(student, teacher, train_loader, val_loader, cfg):
    """Train student_skd. Returns path to best checkpoint."""
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
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs_skd * len(train_loader)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    best_miou = 0.0
    best_path = f"{cfg.output_dir}/student_skd.ckpt"

    for epoch in range(cfg.epochs_skd):
        tracker = MetricTracker()

        for step, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)

            with autocast("cuda", enabled=cfg.fp16):
                student_logits, student_feat = student(imgs, return_features=True)
                with torch.no_grad():
                    teacher_logits = teacher(imgs, return_features=False)
                    teacher_feat = teacher_logits

                l_sup = seg_loss(student_logits, lbls)
                l_kd = kd_loss(student_logits, teacher_logits)
                l_struct = struct_loss(student_feat, teacher_feat)
                loss = l_sup + cfg.alpha_kd * l_kd + cfg.alpha_struct * l_struct

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            tracker.update("loss", loss.item())
            tracker.update("l_sup", l_sup.item())
            tracker.update("l_kd", l_kd.item())
            tracker.update("l_struct", l_struct.item())
            preds = student_logits.argmax(dim=1)
            for i in range(preds.shape[0]):
                tracker.update("miou", compute_miou(preds[i], lbls[i], cfg.num_classes))

            if (step + 1) % cfg.log_every == 0:
                s = tracker.summary()
                print(f"  [SKD] Epoch {epoch+1}/{cfg.epochs_skd} Step {step+1} "
                      f"loss={s['loss']:.4f} sup={s['l_sup']:.4f} kd={s['l_kd']:.4f} "
                      f"struct={s['l_struct']:.4f} mIoU={s['miou']:.4f}")

        if (epoch + 1) % cfg.eval_every == 0:
            miou = evaluate(student, val_loader, cfg.num_classes, device)
            print(f"  [SKD] Epoch {epoch+1} val mIoU={miou:.4f}")
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(student, optimizer, epoch, miou, best_path)

    print(f"  [SKD] Best val mIoU={best_miou:.4f}")
    return best_path