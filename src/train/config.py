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
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"      # cosine | step | poly
    warmup_epochs: int = 5

    # ── Loss weights ──
    alpha_ce: float = 1.0             # supervised CE weight
    alpha_dice: float = 0.5           # Dice loss weight
    alpha_kd: float = 1.0             # KL distillation weight
    alpha_struct: float = 0.5         # structural/affinity loss weight
    kd_temperature: float = 4.0       # softmax temperature for KD

    # ── Guardrail ──
    guardrail_mode: str = "gap"       # gap | binary | both
    guardrail_threshold: float = 0.5  # risk score threshold for flagging

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