import os
import argparse
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Guardrail Distillation Pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── Shared args ────────────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset-path",   default="/root/Guardrail-Distillation/data/cityscapes")
    shared.add_argument("--output-dir",     default="outputs-mit-b0")
    shared.add_argument("--num-classes",    type=int,   default=19)
    shared.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    shared.add_argument("--num-workers",    type=int,   default=0)
    shared.add_argument("--batch-size",     type=int,   default=10)
    shared.add_argument("--teacher-model",  default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    shared.add_argument("--student-model",  default="nvidia/mit-b0")

    # ── Train mode ─────────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", parents=[shared], help="Run training pipeline then eval")
    train_p.add_argument("--epochs-sup",       type=int,   default=10)
    train_p.add_argument("--epochs-kd",        type=int,   default=10)
    train_p.add_argument("--epochs-skd",       type=int,   default=10)
    train_p.add_argument("--epochs-guardrail", type=int,   default=20)
    train_p.add_argument("--lr",               type=float, default=5e-5)
    train_p.add_argument("--eval-every",       type=int,   default=1)
    train_p.add_argument("--alpha-kd",         type=float, default=1.0)
    train_p.add_argument("--alpha-struct",     type=float, default=0.5)
    train_p.add_argument("--kd-temperature",   type=float, default=2.0)
    train_p.add_argument("--log-every",        type=int,   default=50)
    train_p.add_argument("--guardrail-mode",   default="confidence")
    train_p.add_argument("--mc-dropout-passes",type=int,   default=0)
    train_p.add_argument("--skip-sup",         action="store_true")
    train_p.add_argument("--skip-kd",          action="store_true")
    train_p.add_argument("--skip-skd",         action="store_true")
    train_p.add_argument("--skip-guardrail",   action="store_true")

    # ── Eval mode ──────────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser("eval", parents=[shared], help="Run eval from existing checkpoints")
    eval_p.add_argument("--guardrail-mode",    default="confidence")
    eval_p.add_argument("--mc-dropout-passes", type=int, default=0)
    eval_p.add_argument(
        "--checkpoints",
        nargs="+",
        metavar="NAME:PATH",
        help="Override checkpoint paths, e.g. student_sup:outputs/student_sup.ckpt",
        default=None,
    )

    return parser.parse_args()


def build_cfg(args):
    from config import Config
    return Config(
        dataset_path=args.dataset_path,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=getattr(args, "lr", 5e-5),
        epochs_sup=getattr(args, "epochs_sup", 10),
        epochs_kd=getattr(args, "epochs_kd", 10),
        epochs_skd=getattr(args, "epochs_skd", 10),
        epochs_guardrail=getattr(args, "epochs_guardrail", 20),
        eval_every=getattr(args, "eval_every", 1),
        alpha_kd=getattr(args, "alpha_kd", 1.0),
        alpha_struct=getattr(args, "alpha_struct", 0.5),
        kd_temperature=getattr(args, "kd_temperature", 2.0),
        log_every=getattr(args, "log_every", 50),
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=args.device,
        guardrail_mode=getattr(args, "guardrail_mode", "confidence"),
    )


def make_fresh_student(args, cfg):
    from transformers import AutoModelForSemanticSegmentation, SegformerForSemanticSegmentation, SegformerConfig
    from models import HFSegModelWrapper

    backbone = AutoModelForSemanticSegmentation.from_pretrained(args.student_model, local_files_only=True)
    config = SegformerConfig.from_pretrained(args.student_model, local_files_only=True)
    config.num_labels = cfg.num_classes
    model = SegformerForSemanticSegmentation(config)
    model.segformer.load_state_dict(backbone.base_model.state_dict(), strict=False)
    return HFSegModelWrapper(model, cfg.num_classes)


def load_teacher(args, cfg):
    from transformers import AutoModelForSemanticSegmentation
    from models import HFSegModelWrapper

    raw = AutoModelForSemanticSegmentation.from_pretrained(args.teacher_model, local_files_only=True)
    return HFSegModelWrapper(raw, cfg.num_classes).to(cfg.device).eval()


def default_checkpoints(output_dir):
    return {
        "student_sup": os.path.join(output_dir, "student_sup.ckpt"),
        "student_kd":  os.path.join(output_dir, "student_kd.ckpt"),
        "student_skd": os.path.join(output_dir, "student_skd.ckpt"),
        "guardrail":   os.path.join(output_dir, "guardrail.ckpt"),
    }


def run_eval_pipeline(args, cfg, checkpoint_map):
    from models import GuardrailHead
    from utils import load_checkpoint
    from src.eval.eval import run_eval
    from src.eval.data import CITYSCAPES_LABELID_TO_TRAINID
    from src.eval.analysis import plot_results
    from src.train.eval_guardrail import run_benchmark
    from data import build_dataloaders

    _, val_loader = build_dataloaders(cfg)
    teacher = load_teacher(args, cfg)

    results_dir = os.path.join(args.output_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    benchmark_dir = os.path.join(results_dir, "benchmark")
    os.makedirs(results_dir, exist_ok=True)

    all_csvs = []
    for name, ckpt_path in checkpoint_map.items():
        if name == "guardrail":
            continue
        if not os.path.exists(ckpt_path):
            print(f"[Eval] Skipping {name} — checkpoint not found at {ckpt_path}")
            continue
        csv_path = os.path.join(results_dir, f"{name}.csv")
        print(f"\n[Eval] {name}: {ckpt_path}")
        run_eval(
            model_tag=f"{args.student_model}::{ckpt_path}",
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
        plot_results(all_csvs, save_dir=figures_dir)

    # Benchmark
    students = {}
    for name, ckpt_path in checkpoint_map.items():
        if name == "guardrail":
            continue
        if os.path.exists(ckpt_path):
            model = make_fresh_student(args, cfg)
            load_checkpoint(model, ckpt_path, device=cfg.device)
            students[name] = model

    guardrail_ckpt = checkpoint_map.get("guardrail", "")
    if os.path.exists(guardrail_ckpt):
        guardrail_head = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
        guard_state = torch.load(guardrail_ckpt, map_location=cfg.device, weights_only=False)
        guardrail_head.load_state_dict(guard_state["model"])

        run_benchmark(
            students=students,
            val_loader=val_loader,
            num_classes=cfg.num_classes,
            device=cfg.device,
            teacher=teacher,
            guardrail=guardrail_head,
            guardrail_student_name="student_skd",
            mc_dropout_passes=args.mc_dropout_passes,
            save_dir=benchmark_dir,
        )
    else:
        print("[Eval] Guardrail checkpoint not found, skipping benchmark.")


def run_train_pipeline(args, cfg):
    from models import GuardrailHead
    from utils import load_checkpoint
    from data import build_dataloaders
    from train_supervised import train_supervised
    from train_kd import train_kd
    from train_skd import train_skd
    from train_guardrail import train_guardrail

    train_loader, val_loader = build_dataloaders(cfg)
    teacher = load_teacher(args, cfg)
    fresh = lambda: make_fresh_student(args, cfg)

    ckpts = default_checkpoints(cfg.output_dir)

    if not args.skip_sup:
        ckpts["student_sup"] = train_supervised(fresh(), train_loader, val_loader, cfg)

    if not args.skip_kd:
        print(f"  [KD] Starting training ({len(train_loader)} steps/epoch)...")
        ckpts["student_kd"] = train_kd(fresh(), teacher, train_loader, val_loader, cfg)

    if not args.skip_skd:
        ckpts["student_skd"] = train_skd(fresh(), teacher, train_loader, val_loader, cfg)

    if not args.skip_guardrail:
        best_student = fresh()
        load_checkpoint(best_student, ckpts["student_skd"], device=cfg.device)
        best_student.to(cfg.device).eval()
        guardrail = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
        train_guardrail(guardrail, best_student, teacher, train_loader, val_loader, cfg)
        ckpts["guardrail"] = os.path.join(cfg.output_dir, "guardrail.ckpt")

    return ckpts


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        ckpts = run_train_pipeline(args, cfg)
        run_eval_pipeline(args, cfg, ckpts)

    elif args.mode == "eval":
        if args.checkpoints:
            ckpts = {}
            for entry in args.checkpoints:
                name, path = entry.split(":", 1)
                ckpts[name] = path
        else:
            ckpts = default_checkpoints(args.output_dir)
        run_eval_pipeline(args, cfg, ckpts)


if __name__ == "__main__":
    main()