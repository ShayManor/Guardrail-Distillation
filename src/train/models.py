"""Model definitions: teacher, student, and guardrail head."""

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
        # Replace final conv if num_classes differs
        last_conv = self.classifier[-1]
        if last_conv.out_channels != num_classes:
            self.classifier[-1] = nn.Conv2d(
                last_conv.in_channels, num_classes, kernel_size=1
            )

    def forward(self, x, return_features=False):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        feat = features["out"]  # backbone output features
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
        # Probe output channels and add projection if needed
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
        # Load from HuggingFace
        import segmentation_models_pytorch as smp
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
        # No built-in DeepLabV3+ResNet18, use resnet50 as fallback
        # For a true resnet18 student you'd swap the backbone manually
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        base = deeplabv3_resnet50(weights=weights)
        print("[WARN] resnet18 student not natively supported, using resnet50 as proxy")
    else:
        raise ValueError(f"Unknown student arch: {arch}")
    return SegModel(base, num_classes)


# ── Guardrail Network ──

class GuardrailHead(nn.Module):
    """
    Lightweight head that predicts teacher-student risk gap.

    Inputs: student logits (C,H,W) and optionally student backbone features.
    Outputs:
        - risk_score: scalar image-level risk (sigmoid)
        - pixel_heatmap: per-pixel failure probability (H,W)
    """

    def __init__(self, num_classes=19, feat_channels=0, mode="gap"):
        """
        Args:
            num_classes: number of seg classes (student logit channels)
            feat_channels: student backbone feature channels (0 = don't use features)
            mode: 'gap' | 'binary' | 'both'
        """
        super().__init__()
        self.mode = mode
        in_ch = num_classes + feat_channels

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )

        # Per-pixel heatmap head
        if mode in ("gap", "both"):
            self.gap_head = nn.Conv2d(32, 1, 1)  # regression: gap magnitude
        if mode in ("binary", "both"):
            self.binary_head = nn.Conv2d(32, 1, 1)  # binary: student wrong & teacher right

        # Image-level risk score (global pool → scalar)
        self.risk_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, student_logits, student_features=None):
        """
        Args:
            student_logits: (B, C, H, W) raw logits from student
            student_features: (B, F, h, w) backbone features (optional)
        """
        x = student_logits.detach()  # don't backprop into student

        if student_features is not None:
            feat = F.interpolate(
                student_features.detach(), size=x.shape[-2:],
                mode="bilinear", align_corners=False
            )
            x = torch.cat([x, feat], dim=1)

        enc = self.encoder(x)

        out = {"risk_score": self.risk_head(enc)}

        if hasattr(self, "gap_head"):
            out["gap_heatmap"] = torch.sigmoid(self.gap_head(enc).squeeze(1))  # (B, H, W)
        if hasattr(self, "binary_head"):
            out["binary_heatmap"] = torch.sigmoid(self.binary_head(enc).squeeze(1))

        return out


class GuardrailPlusHead(nn.Module):
    """
    Guardrail++ head for utility / counterfactual margin prediction.

    Outputs:
        - utility_score: image-level fallback utility in [0, 1]
        - margin_vec: per-family intervention margin in [0, 1]
        - family_prob (optional): corruption family distribution
    """

    def __init__(
        self,
        num_classes=19,
        feat_channels=0,
        num_families=4,
        predict_family_prob=True,
    ):
        super().__init__()
        in_ch = num_classes + feat_channels
        self.num_families = num_families
        self.predict_family_prob = predict_family_prob

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gap_head = nn.Conv2d(32, 1, 1)
        self.utility_head = nn.Linear(32, 1)
        self.margin_head = nn.Linear(32, num_families)
        if self.predict_family_prob:
            self.family_head = nn.Linear(32, num_families)

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

        out = {
            "utility_score": torch.sigmoid(self.utility_head(pooled)).squeeze(1),
            "margin_vec": torch.sigmoid(self.margin_head(pooled)),
            "gap_heatmap": torch.sigmoid(self.gap_head(enc).squeeze(1)),
        }
        if self.predict_family_prob:
            out["family_prob"] = torch.softmax(self.family_head(pooled), dim=1)
        return out
