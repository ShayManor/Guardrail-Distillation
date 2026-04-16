"""Loss functions for all training stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = 255


# ──────────────────────────────────────────────
# Stage 1: Supervised losses
# ──────────────────────────────────────────────

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
    """Combined CE + Dice for supervised training."""

    def __init__(self, alpha_ce=1.0, alpha_dice=0.5, class_weights=None):
        super().__init__()
        self.ce = CELoss(weight=class_weights)
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
        loss = loss / (student_logits.shape[2] * student_logits.shape[3])
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
# Stage 4: Guardrail++ loss (dense per-pixel supervision)
# ──────────────────────────────────────────────


class GuardrailPlusLoss(nn.Module):
    """Training loss for the Guardrail++ selective-prediction head.

    The active supervision signal is selected by ``supervision_type``:

    * ``'scalar_benefit'`` — image-level regression of ``utility_score``
      against the scalar teacher-benefit target ``student_risk − teacher_risk``
      (legacy baseline; used for the ablation in Table 2).
    * ``'dense_disagree'`` — masked BCE on the per-pixel ``disagree_logits``
      head against the teacher/student argmax disagreement mask.
    * ``'dense_gap'`` — masked smooth-L1 on the per-pixel ``gap_pred`` head
      against ``student_ce − teacher_ce``.
    * ``'dense_multi'`` (default) — both dense losses summed with weights.

    The scalar path is retained only so we can ablate against it; the paper's
    primary method is ``dense_multi``.
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
            "gt_disagree", "gt_risk",
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
        use_disagree = st in ("dense_disagree", "dense_multi", "gt_disagree")
        use_gap = st in ("dense_gap", "dense_multi", "gt_risk")

        if use_scalar:
            l_utility = F.smooth_l1_loss(
                preds["utility_score"], targets["utility_target"]
            )
            loss = loss + self.scalar_weight * l_utility
            info["utility_loss"] = float(l_utility.item())

        if use_disagree:
            logits = preds["disagree_logits"]
            tgt = targets["disagree_target"]
            valid = targets["disagree_valid"]
            per_pix = F.binary_cross_entropy_with_logits(
                logits, tgt, reduction="none"
            )
            l_disagree = self._masked_mean(per_pix, valid)
            loss = loss + self.dense_disagree_weight * l_disagree
            info["dense_disagree_loss"] = float(l_disagree.item())

        if use_gap:
            pred = preds["gap_pred"]
            tgt = targets["gap_target"]
            valid = targets["gap_valid"]
            per_pix = F.smooth_l1_loss(pred, tgt, reduction="none")
            l_gap_dense = self._masked_mean(per_pix, valid)
            loss = loss + self.dense_gap_weight * l_gap_dense
            info["dense_gap_loss"] = float(l_gap_dense.item())

        info["loss"] = float(loss.item())
        return loss, info
