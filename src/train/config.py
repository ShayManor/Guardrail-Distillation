"""Central configuration for the guardrail distillation pipeline."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ── Paths ──
    teacher_path: str = ""            # path or torchvision model name
    student_arch: str = "mobilenet"   # mobilenet | resnet18 | resnet50
    teacher_arch: str = "resnet101"   # resnet101 | resnet50
    dataset_path: str = ""            # local path or hf://dataset_name
    output_dir: str = "outputs"
    num_classes: int = 19             # cityscapes default

    # ── Training ──
    epochs_sup: int = 100
    epochs_kd: int = 100
    epochs_skd: int = 100
    epochs_guardrail: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-3
    lr_scheduler: str = "cosine"      # cosine | step | poly
    warmup_epochs: int = 5

    # ── Loss weights ──
    alpha_ce: float = 1.0             # supervised CE weight
    alpha_dice: float = 0.5           # Dice loss weight
    alpha_kd: float = 1.0             # KL distillation weight
    alpha_struct: float = 0.5         # structural/affinity loss weight
    kd_temperature: float = 4.0       # softmax temperature for KD

    # ── Guardrail ──
    guardrail_mode: str = "gap"       # gap | binary | both | utility | margin | guardrailpp
    guardrail_threshold: float = 0.5  # risk score threshold for flagging
    utility_w0: float = 0.5
    utility_w1: float = 0.25
    utility_w2: float = 0.25
    cf_delta: float = 0.02
    cf_severities: tuple = field(default_factory=lambda: (0.25, 0.5, 0.75, 1.0))
    utility_loss_weight: float = 1.0
    margin_loss_weight: float = 1.0
    family_loss_weight: float = 0.0
    margin_loss: str = "huber"
    corruption_prob: float = 0.5
    use_student_features: bool = False
    composite_risk_weight: float = 0.0   # 0 = pure benefit target (default); >0 mixes student risk into utility target
                                         # e.g. 0.8 → target = 0.2*benefit + 0.8*student_risk

    # ── Data ──
    crop_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    # ── Misc ──
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    log_every: int = 50
    eval_every: int = 1               # eval every N epochs
    resume: Optional[str] = None
