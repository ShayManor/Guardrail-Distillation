"""Central configuration for the guardrail distillation pipeline."""

from dataclasses import dataclass
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

    # ── Distillation loss weights (stages 1-3) ──
    alpha_ce: float = 1.0             # supervised CE weight
    alpha_dice: float = 0.5           # Dice loss weight
    alpha_kd: float = 1.0             # KL distillation weight
    alpha_struct: float = 0.5         # structural/affinity loss weight
    kd_temperature: float = 4.0       # softmax temperature for KD

    # ── Guardrail++ head (stage 4) ──
    # Primary method (paper): dense per-pixel supervision. The scalar_benefit
    # path is retained for ablations only.
    supervision_type: str = "dense_multi"   # scalar_benefit | dense_disagree | dense_gap | dense_multi
    dense_disagree_weight: float = 1.0
    dense_gap_weight: float = 1.0
    scalar_benefit_weight: float = 1.0
    use_student_features: bool = True       # feed student backbone features into the head
    corruption_prob: float = 0.5            # fraction of training batches that get online corruption

    # ── Data ──
    crop_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    # ── Misc ──
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    log_every: int = 50
    eval_every: int = 1
    resume: Optional[str] = None
