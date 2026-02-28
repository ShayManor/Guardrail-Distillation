from transformers import AutoModelForSemanticSegmentation
from models import HFSegModelWrapper, GuardrailHead
from config import Config
from data import build_dataloaders
from utils import load_checkpoint
from train_supervised import train_supervised
from train_kd import train_kd
from train_skd import train_skd
from train_guardrail import train_guardrail
import torch

cfg = Config(
    dataset_path="/workspace/data/cityscapes",
    num_classes=19,
    batch_size=8,
    epochs_sup=4,
    epochs_kd=4,
    epochs_skd=4,
    epochs_guardrail=8,
    lr=1e-4,
    eval_every=1,
    log_every=50,
    output_dir="outputs",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

train_loader, val_loader = build_dataloaders(cfg)

# Load from local HF cache
teacher_raw = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
teacher = HFSegModelWrapper(teacher_raw, cfg.num_classes).to(cfg.device).eval()

def fresh_student():
    raw = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
    return HFSegModelWrapper(raw, cfg.num_classes)

# Stages 1–3
path_sup = train_supervised(fresh_student(), train_loader, val_loader, cfg)
path_kd = train_kd(fresh_student(), teacher, train_loader, val_loader, cfg)
path_skd = train_skd(fresh_student(), teacher, train_loader, val_loader, cfg)

# Stage 4
best_student = fresh_student()
load_checkpoint(best_student, path_skd, device=cfg.device)
best_student.to(cfg.device).eval()

guardrail = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
train_guardrail(guardrail, best_student, teacher, train_loader, val_loader, cfg)