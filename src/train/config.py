"""Defaults for the four-stage training pipeline.

CLI flags on `run.py` override every field here.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    teacher_path: str = ""
    student_arch: str = "mobilenet"
    teacher_arch: str = "resnet101"
    dataset_path: str = ""
    output_dir: str = "outputs"
    num_classes: int = 19

    epochs_sup: int = 100
    epochs_kd: int = 100
    epochs_skd: int = 100
    epochs_guardrail: int = 50
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-3
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5

    alpha_ce: float = 1.0
    alpha_dice: float = 0.5
    alpha_kd: float = 1.0
    alpha_struct: float = 0.5
    kd_temperature: float = 4.0

    # Stage 4. dense_multi is the paper headline; the rest are ablation rows.
    # Modes: dense_multi | dense_disagree | dense_gap | gt_disagree | gt_risk | gt_multi | scalar_benefit
    supervision_type: str = "dense_multi"
    dense_disagree_weight: float = 1.0
    dense_gap_weight: float = 1.0
    scalar_benefit_weight: float = 1.0
    use_student_features: bool = True
    use_confidence_features: bool = False
    corruption_prob: float = 0.5

    crop_size: int = 512
    num_workers: int = 4
    pin_memory: bool = True

    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    log_every: int = 50
    eval_every: int = 1
    resume: Optional[str] = None
