"""Loss functions for all training stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


# ──────────────────────────────────────────────
# Stage 1: Supervised losses
# ──────────────────────────────────────────────

class CELoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce(logits, target)


class DiceLoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        mask = target != self.ignore_index
        target_clean = target.clone()
        target_clean[~mask] = 0
        one_hot = F.one_hot(target_clean, num_classes).permute(0, 3, 1, 2).float()

        mask = mask.unsqueeze(1).expand_as(one_hot)
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """Combined CE + Dice for supervised training."""

    def __init__(self, alpha_ce=1.0, alpha_dice=0.5):
        super().__init__()
        self.ce = CELoss()
        self.dice = DiceLoss()
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice

    def forward(self, logits, target):
        return self.alpha_ce * self.ce(logits, target) + self.alpha_dice * self.dice(logits, target)


# ──────────────────────────────────────────────
# Stage 2: Knowledge Distillation loss
# ──────────────────────────────────────────────

class KDLoss(nn.Module):
    """KL divergence on softened logits."""

    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        loss = F.kl_div(s, t, reduction="batchmean") * (self.T ** 2)
        return loss


# ──────────────────────────────────────────────
# Stage 3: Structured KD losses
# ──────────────────────────────────────────────

class PairwiseAffinityLoss(nn.Module):
    """
    Distill pairwise pixel relations from teacher features to student features.
    Computes cosine similarity matrices and matches them.
    """

    def __init__(self, subsample=512):
        super().__init__()
        self.subsample = subsample

    def forward(self, student_feat, teacher_feat):
        B, C_s, H, W = student_feat.shape
        _, C_t, _, _ = teacher_feat.shape

        # Align spatial dims
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        # Flatten to (B, C, N)
        s = student_feat.flatten(2)  # (B, C_s, N)
        t = teacher_feat.flatten(2)  # (B, C_t, N)
        N = s.shape[2]

        # Subsample pixels for memory efficiency
        if N > self.subsample:
            idx = torch.randperm(N, device=s.device)[:self.subsample]
            s = s[:, :, idx]
            t = t[:, :, idx]

        # Pairwise cosine similarity
        s_norm = F.normalize(s, dim=1)
        t_norm = F.normalize(t, dim=1)
        s_aff = torch.bmm(s_norm.transpose(1, 2), s_norm)  # (B, n, n)
        t_aff = torch.bmm(t_norm.transpose(1, 2), t_norm)

        return F.mse_loss(s_aff, t_aff)


# ──────────────────────────────────────────────
# Stage 4: Guardrail losses
# ──────────────────────────────────────────────

class GuardrailLoss(nn.Module):
    """
    Loss for training the guardrail head.

    Targets:
        gap_map: per-pixel |loss_student - loss_teacher| (regression)
        binary_map: (teacher_correct & student_wrong) mask (binary CE)
        risk_label: image-level scalar (mean gap or fraction of failures)
    """

    def __init__(self, mode="gap"):
        super().__init__()
        self.mode = mode

    def forward(self, preds, targets):
        loss = 0.0
        info = {}

        # Image-level risk
        if "risk_score" in preds and "risk_label" in targets:
            risk_loss = F.mse_loss(preds["risk_score"].squeeze(), targets["risk_label"])
            loss = loss + risk_loss
            info["risk_loss"] = risk_loss.item()

        # Gap heatmap (regression)
        if "gap_heatmap" in preds and "gap_map" in targets:
            gap_loss = F.smooth_l1_loss(preds["gap_heatmap"], targets["gap_map"])
            loss = loss + gap_loss
            info["gap_loss"] = gap_loss.item()

        # Binary heatmap
        if "binary_heatmap" in preds and "binary_map" in targets:
            bin_loss = F.binary_cross_entropy(
                preds["binary_heatmap"], targets["binary_map"].float()
            )
            loss = loss + bin_loss
            info["binary_loss"] = bin_loss.item()

        return loss, info


def compute_guardrail_targets(student_logits, teacher_logits, gt, mode="gap"):
    """
    Compute guardrail supervision from teacher/student predictions and GT.

    Returns dict of target tensors.
    """
    ignore_mask = gt == IGNORE_INDEX
    gt_safe = gt.clone()
    gt_safe[ignore_mask] = 0

    student_pred = student_logits.argmax(dim=1)
    teacher_pred = teacher_logits.argmax(dim=1)

    student_correct = (student_pred == gt) & ~ignore_mask
    teacher_correct = (teacher_pred == gt) & ~ignore_mask

    targets = {}

    if mode in ("gap", "both"):
        # Per-pixel CE loss difference
        student_ce = F.cross_entropy(student_logits, gt_safe, reduction="none")
        teacher_ce = F.cross_entropy(teacher_logits, gt_safe, reduction="none")
        gap = (student_ce - teacher_ce).clamp(min=0)
        gap[ignore_mask] = 0
        targets["gap_map"] = gap
        targets["risk_label"] = gap.mean(dim=(1, 2))  # image-level

    if mode in ("binary", "both"):
        binary = (teacher_correct & ~student_correct).float()
        binary[ignore_mask] = 0
        targets["binary_map"] = binary
        if "risk_label" not in targets:
            valid = (~ignore_mask).float().sum(dim=(1, 2)).clamp(min=1)
            targets["risk_label"] = binary.sum(dim=(1, 2)) / valid

    return targets