import os

from transformers import AutoModelForSemanticSegmentation
from models import HFSegModelWrapper, GuardrailHead
from config import Config
from data import build_dataloaders
from utils import load_checkpoint
from train_supervised import train_supervised
from train_kd import train_kd
from train_skd import train_skd
from train_guardrail import train_guardrail
from src.eval.eval import run_eval
from src.eval.data import CITYSCAPES_LABELID_TO_TRAINID
from src.eval.analysis import plot_results
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

cfg = Config(
    dataset_path="/root/Guardrail-Distillation/data/cityscapes",
    num_classes=19,
    batch_size=2,
    epochs_sup=3,
    epochs_kd=3,
    epochs_skd=3,
    epochs_guardrail=4,
    lr=3e-4,
    eval_every=1,
    alpha_kd=1.0,
    alpha_struct=0.5,
    kd_temperature=2.0,
    log_every=5,
    output_dir="outputs",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

train_loader, val_loader = build_dataloaders(cfg)

# Load from local HF cache
teacher_raw = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024", local_files_only=True)
teacher = HFSegModelWrapper(teacher_raw, cfg.num_classes).to(cfg.device).eval()

def fresh_student():
    raw = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024", local_files_only=True)
    return HFSegModelWrapper(raw, cfg.num_classes)

# Stages 1–3
path_sup = train_supervised(fresh_student(), train_loader, val_loader, cfg)
print(f"  [KD] Starting training ({len(train_loader)} steps/epoch)...")
path_kd = train_kd(fresh_student(), teacher, train_loader, val_loader, cfg)
path_skd = train_skd(fresh_student(), teacher, train_loader, val_loader, cfg)

# Stage 4
best_student = fresh_student()
load_checkpoint(best_student, path_skd, device=cfg.device)
best_student.to(cfg.device).eval()

guardrail = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
train_guardrail(guardrail, best_student, teacher, train_loader, val_loader, cfg)

checkpoints = {
    "student_sup": ("nvidia/segformer-b0-finetuned-cityscapes-512-1024", "outputs/student_sup.ckpt"),
    "student_kd":  ("nvidia/segformer-b0-finetuned-cityscapes-512-1024", "outputs/student_kd.ckpt"),
    "student_skd": ("nvidia/segformer-b0-finetuned-cityscapes-512-1024", "outputs/student_skd.ckpt"),
    "guardrail":   ("nvidia/segformer-b0-finetuned-cityscapes-512-1024", "outputs/guardrail.ckpt"),
}

all_csvs = []
for name, (arch, ckpt_path) in checkpoints.items():
    if not os.path.exists(ckpt_path):
        print(f"[Eval] Skipping {name} — checkpoint not found")
        continue
    csv_path = f"results/{name}.csv"
    print(f"\n[Eval] {name}: {ckpt_path}")
    run_eval(
        model_tag=f"{arch}::{ckpt_path}",
        dataset_path=cfg.dataset_path,
        output_csv=csv_path,
        images_subdir="leftImg8bit/val",
        labels_subdir="gtFine/val",
        label_map=CITYSCAPES_LABELID_TO_TRAINID,
        model_name=name,
        dataset_name="cityscapes-val",
    )
    all_csvs.append(csv_path)

if all_csvs:
    plot_results(all_csvs, save_dir="results/figures")

from src.train.eval_guardrail import run_benchmark

# Load all student checkpoints
students = {}
for name, ckpt_path in [("student_sup", "outputs/student_sup.ckpt"),
                          ("student_kd", "outputs/student_kd.ckpt"),
                          ("student_skd", "outputs/student_skd.ckpt")]:
    if os.path.exists(ckpt_path):
        model = fresh_student()
        load_checkpoint(model, ckpt_path, device=cfg.device)
        students[name] = model

# Load guardrail
guardrail_head = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
guard_ckpt = torch.load("outputs/guardrail.ckpt", map_location=cfg.device, weights_only=False)
guardrail_head.load_state_dict(guard_ckpt["model"])

run_benchmark(
    students=students,
    val_loader=val_loader,
    num_classes=cfg.num_classes,
    device=cfg.device,
    teacher=teacher,
    guardrail=guardrail_head,
    guardrail_student_name="student_skd",  # whichever student the guardrail was trained on
    mc_dropout_passes=0,                    # set to 5-10 to include MC Dropout baseline
    save_dir="results/benchmark",
)