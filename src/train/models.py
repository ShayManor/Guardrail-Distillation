"""Teacher, student, and the GuardrailPlusHead."""

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


class SegModel(nn.Module):
    """torchvision DeepLabV3 wrapper that can return backbone features."""

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
        feat = self.backbone(x)["out"]
        logits = self.classifier(feat)
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        if return_features:
            return logits, feat
        return logits


def build_teacher(arch="resnet101", num_classes=19, pretrained=True):
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
    """HuggingFace segmentation model wrapped to match SegModel's interface."""

    def __init__(self, hf_model, num_classes=19):
        super().__init__()
        self.model = hf_model
        self.proj = None

    def forward(self, x, return_features=False):
        input_shape = x.shape[-2:]
        out = self.model(x, output_hidden_states=return_features)
        logits = out.logits if hasattr(out, "logits") else out
        logits = F.interpolate(logits, size=input_shape, mode="bilinear", align_corners=False)
        if not return_features:
            return logits

        if hasattr(out, "hidden_states") and out.hidden_states:
            feat = out.hidden_states[-1]
            if feat.shape[-2:] != input_shape:
                feat = F.interpolate(feat, size=input_shape, mode="bilinear", align_corners=False)
        else:
            feat = logits
        return logits, feat


def build_student(arch="mobilenet", num_classes=19, pretrained=True):
    if arch.startswith("hf://") or "/" in arch:
        from transformers import AutoModelForSemanticSegmentation
        base = AutoModelForSemanticSegmentation.from_pretrained(arch)
        return HFSegModelWrapper(base, num_classes)
    if arch == "mobilenet":
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = deeplabv3_mobilenet_v3_large(weights=weights)
    elif arch in ("resnet50", "resnet18"):
        # resnet18 isn't shipped by torchvision for DeepLabV3; fall back to resnet50.
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet50(weights=weights)
        if arch == "resnet18":
            print("[WARN] resnet18 not natively supported, using resnet50")
    else:
        raise ValueError(f"Unknown student arch: {arch}")
    return SegModel(base, num_classes)


class GuardrailPlusHead(nn.Module):
    """Selective-prediction head bolted onto a frozen segmentation student.

    Three outputs share a 3-conv encoder over the (detached) student logits,
    optionally concatenated with detached backbone features:

      utility_score    image-level scalar in [0, 1]; trained only under scalar_benefit
      disagree_logits  per-pixel BCE logit; trained under dense_{multi,disagree}, gt_disagree
      gap_pred         per-pixel real value; trained under dense_{multi,gap}, gt_risk

    Inference scores: sigmoid(disagree_logits).mean() and gap_pred.mean(),
    aliased into per_image.csv as guardrailpp_utility_dense_{bce,gap}.
    """

    def __init__(self, num_classes=19, feat_channels=0, use_confidence_features=False):
        super().__init__()
        # End-to-end fusion: feed per-pixel energy (-logsumexp) and max-logit as
        # explicit input channels so a single learned score can combine dense
        # disagreement prediction with confidence magnitude. Off by default.
        self.use_confidence_features = use_confidence_features
        conf_channels = 2 if use_confidence_features else 0
        in_ch = num_classes + feat_channels + conf_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.utility_head = nn.Linear(32, 1)
        self.disagree_head = nn.Conv2d(32, 1, 1)
        self.gap_head = nn.Conv2d(32, 1, 1)

    def forward(self, student_logits, student_features=None):
        # Detach so the guardrail can never backprop into the student.
        logits = student_logits.detach()
        parts = [logits]
        if student_features is not None:
            feat = F.interpolate(
                student_features.detach(),
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            parts.append(feat)
        if self.use_confidence_features:
            energy = -torch.logsumexp(logits, dim=1, keepdim=True)
            max_logit = logits.max(dim=1, keepdim=True).values
            parts.extend([energy, max_logit])
        x = torch.cat(parts, dim=1) if len(parts) > 1 else logits

        enc = self.encoder(x)
        pooled = self.pool(enc).flatten(1)
        return {
            "utility_score": torch.sigmoid(self.utility_head(pooled)).squeeze(1),
            "disagree_logits": self.disagree_head(enc).squeeze(1),
            "gap_pred": self.gap_head(enc).squeeze(1),
        }
