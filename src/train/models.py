"""Model definitions: teacher, student, and the Guardrail++ head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    deeplabv3_resnet50,
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_ResNet101_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


# ── Wrapper that exposes intermediate features ──

class SegModel(nn.Module):
    """Wraps a torchvision DeepLabV3 model to expose backbone features + logits."""

    def __init__(self, base_model, num_classes=19):
        super().__init__()
        self.backbone = base_model.backbone
        self.classifier = base_model.classifier
        last_conv = self.classifier[-1]
        if last_conv.out_channels != num_classes:
            self.classifier[-1] = nn.Conv2d(
                last_conv.in_channels, num_classes, kernel_size=1
            )

    def forward(self, x, return_features=False):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        feat = features["out"]
        logits = self.classifier(feat)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        if return_features:
            return logits, feat
        return logits


def build_teacher(arch="resnet101", num_classes=19, pretrained=True):
    """Build teacher segmentation model."""
    if arch == "resnet101":
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet101(weights=weights)
    elif arch == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet50(weights=weights)
    else:
        raise ValueError(f"Unknown teacher arch: {arch}")
    return SegModel(base, num_classes)


class HFSegModelWrapper(nn.Module):
    """Wraps a HuggingFace segmentation model to match our interface."""

    def __init__(self, hf_model, num_classes=19):
        super().__init__()
        self.model = hf_model
        self.proj = None  # set lazily

    def forward(self, x, return_features=False):
        input_shape = x.shape[-2:]
        out = self.model(x, output_hidden_states=return_features)
        logits = out.logits if hasattr(out, "logits") else out
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)

        if return_features:
            if hasattr(out, "hidden_states") and out.hidden_states:
                feat = out.hidden_states[-1]
                if feat.shape[-2:] != input_shape:
                    feat = F.interpolate(feat, size=input_shape, mode="bilinear", align_corners=False)
            else:
                feat = logits
            return logits, feat
        return logits


def build_student(arch="mobilenet", num_classes=19, pretrained=True):
    """Build student segmentation model."""
    if arch.startswith("hf://") or "/" in arch:
        from transformers import AutoModelForSemanticSegmentation
        try:
            base = AutoModelForSemanticSegmentation.from_pretrained(arch)
            return HFSegModelWrapper(base, num_classes)
        except Exception:
            raise ValueError(f"Could not load HF model: {arch}")
    elif arch == "mobilenet":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = deeplabv3_mobilenet_v3_large(weights=weights)
    elif arch == "resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet50(weights=weights)
    elif arch == "resnet18":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet50(weights=weights)
        print("[WARN] resnet18 student not natively supported, using resnet50 as proxy")
    else:
        raise ValueError(f"Unknown student arch: {arch}")
    return SegModel(base, num_classes)


# ──────────────────────────────────────────────────────────────────────────
# Guardrail++ Head
# ──────────────────────────────────────────────────────────────────────────

class GuardrailPlusHead(nn.Module):
    """Light-weight selective-prediction head for a frozen segmentation student.

    Three outputs share a small 3-conv dense encoder fed by the student's
    (detached) logits and, optionally, its backbone features:

      - ``utility_score`` — image-level scalar in [0, 1], retained for the
        ``scalar_benefit`` ablation.
      - ``disagree_logits`` — per-pixel BCE logits trained against the teacher
        / student argmax disagreement mask.
      - ``gap_pred`` — per-pixel linear prediction of ``student_ce − teacher_ce``.

    The paper's primary selective-prediction score is
    ``sigmoid(disagree_logits).mean()`` (== ``guardrailpp_utility_dense_bce``)
    or ``gap_pred.mean()`` (== ``guardrailpp_utility_dense_gap``), chosen
    empirically on the validation set.

    The head adds <3% latency on top of the student and is fully decoupled
    from the (frozen) student / teacher weights.
    """

    def __init__(self, num_classes=19, feat_channels=0):
        super().__init__()
        in_ch = num_classes + feat_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Scalar image-level head (trained only under supervision_type='scalar_benefit').
        self.utility_head = nn.Linear(32, 1)

        # Dense per-pixel heads (trained under supervision_type in
        # {'dense_disagree', 'dense_gap', 'dense_multi'}).
        self.disagree_head = nn.Conv2d(32, 1, 1)
        self.gap_head = nn.Conv2d(32, 1, 1)

    def forward(self, student_logits, student_features=None):
        x = student_logits.detach()
        if student_features is not None:
            feat = F.interpolate(
                student_features.detach(),
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, feat], dim=1)

        enc = self.encoder(x)
        pooled = self.pool(enc).flatten(1)

        return {
            "utility_score": torch.sigmoid(self.utility_head(pooled)).squeeze(1),
            "disagree_logits": self.disagree_head(enc).squeeze(1),
            "gap_pred": self.gap_head(enc).squeeze(1),
        }
