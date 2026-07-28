"""Losses for all four training stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


class CELoss(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)

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
    """Stage-1 supervised loss: CE + Dice."""

    def __init__(self, alpha_ce=1.0, alpha_dice=0.5, class_weights=None):
        super().__init__()
        self.ce = CELoss(weight=class_weights)
        self.dice = DiceLoss()
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice

    def forward(self, logits, target):
        return self.alpha_ce * self.ce(logits, target) + self.alpha_dice * self.dice(logits, target)


class KDLoss(nn.Module):
    """Stage-2 KD: KL on softened logits, normalised by spatial size."""

    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits):
        s = F.log_softmax(student_logits / self.T, dim=1)
        t = F.softmax(teacher_logits / self.T, dim=1)
        loss = F.kl_div(s, t, reduction="batchmean") * (self.T ** 2)
        return loss / (student_logits.shape[2] * student_logits.shape[3])


class PairwiseAffinityLoss(nn.Module):
    """Stage-3 structured KD: match pairwise cosine-similarity matrices."""

    def __init__(self, subsample=512):
        super().__init__()
        self.subsample = subsample

    def forward(self, student_feat, teacher_feat):
        B, C_s, H, W = student_feat.shape

        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        s = student_feat.flatten(2)
        t = teacher_feat.flatten(2)
        N = s.shape[2]

        if N > self.subsample:
            idx = torch.randperm(N, device=s.device)[:self.subsample]
            s = s[:, :, idx]
            t = t[:, :, idx]

        s_norm = F.normalize(s, dim=1)
        t_norm = F.normalize(t, dim=1)
        s_aff = torch.bmm(s_norm.transpose(1, 2), s_norm)
        t_aff = torch.bmm(t_norm.transpose(1, 2), t_norm)
        return F.mse_loss(s_aff, t_aff)


class GuardrailPlusLoss(nn.Module):
    """Stage-4 guardrail loss. Routes to the heads specified by supervision_type.

    dense_multi (default) sums BCE on disagree_logits and smooth-L1 on gap_pred.
    Single-head modes drop one of those terms. scalar_benefit is the legacy
    image-level regression on utility_score, kept only as the ablation row.
    GT modes use the same heads but with ground-truth-derived targets.
    """

    def __init__(
        self,
        supervision_type="dense_multi",
        dense_disagree_weight=1.0,
        dense_gap_weight=1.0,
        scalar_weight=1.0,
    ):
        super().__init__()
        assert supervision_type in (
            "scalar_benefit", "dense_disagree", "dense_gap", "dense_multi",
            "gt_disagree", "gt_risk", "gt_multi",
        ), f"unknown supervision_type: {supervision_type}"
        self.supervision_type = supervision_type
        self.dense_disagree_weight = float(dense_disagree_weight)
        self.dense_gap_weight = float(dense_gap_weight)
        self.scalar_weight = float(scalar_weight)

    @staticmethod
    def _masked_mean(x, mask):
        denom = mask.sum().clamp(min=1.0)
        return (x * mask).sum() / denom

    def forward(self, preds, targets):
        loss = torch.zeros((), device=preds["disagree_logits"].device)
        info = {}

        st = self.supervision_type
        use_scalar = st == "scalar_benefit"
        use_disagree = st in ("dense_disagree", "dense_multi", "gt_disagree", "gt_multi")
        use_gap = st in ("dense_gap", "dense_multi", "gt_risk", "gt_multi")

        if use_scalar:
            l_utility = F.smooth_l1_loss(preds["utility_score"], targets["utility_target"])
            loss = loss + self.scalar_weight * l_utility
            info["utility_loss"] = float(l_utility.item())

        if use_disagree:
            per_pix = F.binary_cross_entropy_with_logits(
                preds["disagree_logits"], targets["disagree_target"], reduction="none"
            )
            l_disagree = self._masked_mean(per_pix, targets["disagree_valid"])
            loss = loss + self.dense_disagree_weight * l_disagree
            info["dense_disagree_loss"] = float(l_disagree.item())

        if use_gap:
            per_pix = F.smooth_l1_loss(preds["gap_pred"], targets["gap_target"], reduction="none")
            l_gap = self._masked_mean(per_pix, targets["gap_valid"])
            loss = loss + self.dense_gap_weight * l_gap
            info["dense_gap_loss"] = float(l_gap.item())

        info["loss"] = float(loss.item())
        return loss, info
