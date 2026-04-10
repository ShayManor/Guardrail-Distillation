import os
import argparse
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from wandb_utils import setup_wandb, wandb_log, log_eval_results, finish_wandb


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
    train_p.add_argument(
        "--guardrail-mode",
        default="utility",
        choices=["gap", "binary", "both", "utility", "margin", "guardrailpp"],
    )
    train_p.add_argument("--utility-w0",       type=float, default=0.5)
    train_p.add_argument("--utility-w1",       type=float, default=0.25)
    train_p.add_argument("--utility-w2",       type=float, default=0.25)
    train_p.add_argument("--cf-delta",         type=float, default=0.02)
    train_p.add_argument("--cf-severities",    default="0.25,0.5,0.75,1.0")
    train_p.add_argument("--utility-loss-weight", type=float, default=1.0)
    train_p.add_argument("--margin-loss-weight",  type=float, default=1.0)
    train_p.add_argument("--family-loss-weight",  type=float, default=0.0)
    train_p.add_argument("--margin-loss",      default="huber", choices=["huber", "mse"])
    train_p.add_argument("--mc-dropout-passes",type=int,   default=0)
    train_p.add_argument("--corruption-prob",   type=float, default=0.5)
    train_p.add_argument("--use-student-features", action="store_true")
    train_p.add_argument("--skip-sup",         action="store_true")
    train_p.add_argument("--skip-kd",          action="store_true")
    train_p.add_argument("--skip-skd",         action="store_true")
    train_p.add_argument("--skip-guardrail",   action="store_true")

    # wandb
    train_p.add_argument("--wandb-project",    default="guardrail-distillation")
    train_p.add_argument("--wandb-group",      default=None)
    train_p.add_argument("--wandb-name",       default=None)
    train_p.add_argument("--no-wandb",         action="store_true", help="Disable wandb logging")

    # ── Eval mode ──────────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser("eval", parents=[shared], help="Run eval from existing checkpoints")
    eval_p.add_argument(
        "--guardrail-mode",
        default="utility",
        choices=["gap", "binary", "both", "utility", "margin", "guardrailpp"],
    )
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
    mode = getattr(args, "guardrail_mode", "utility")
    cf_severities = tuple(float(x.strip()) for x in getattr(args, "cf_severities", "0.25,0.5,0.75,1.0").split(",") if x.strip())
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
        alpha_kd=getattr(args, "alpha_kd", 0.5),
        alpha_struct=getattr(args, "alpha_struct", 0.5),
        kd_temperature=getattr(args, "kd_temperature", 2.0),
        log_every=getattr(args, "log_every", 50),
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=args.device,
        guardrail_mode=mode,
        utility_w0=getattr(args, "utility_w0", 0.5),
        utility_w1=getattr(args, "utility_w1", 0.25),
        utility_w2=getattr(args, "utility_w2", 0.25),
        cf_delta=getattr(args, "cf_delta", 0.02),
        cf_severities=cf_severities,
        utility_loss_weight=getattr(args, "utility_loss_weight", 1.0),
        margin_loss_weight=getattr(args, "margin_loss_weight", 1.0),
        family_loss_weight=getattr(args, "family_loss_weight", 0.0),
        margin_loss=getattr(args, "margin_loss", "huber"),
        corruption_prob=getattr(args, "corruption_prob", 0.5),
        use_student_features=getattr(args, "use_student_features", False),
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
    from models import GuardrailHead, GuardrailPlusHead
    from utils import load_checkpoint
    from src.eval.eval import run_eval
    from src.eval.data import CITYSCAPES_LABELID_TO_TRAINID
    from src.eval.analysis import plot_results
    from src.train.eval_guardrail import run_benchmark
    from data import build_dataloaders

    eval_cfg = build_cfg(args)
    eval_cfg.num_workers = 0
    _, val_loader = build_dataloaders(eval_cfg)
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
        guard_state = torch.load(guardrail_ckpt, map_location=cfg.device, weights_only=False)
        # Infer feat_channels from saved encoder weight shape
        enc_weight = guard_state["model"]["encoder.0.weight"]
        feat_ch = enc_weight.shape[1] - cfg.num_classes  # total_in - logit_channels
        print(f"[Eval] Guardrail feat_channels={feat_ch} (from checkpoint)")
        if cfg.guardrail_mode in ("utility", "margin", "guardrailpp"):
            guardrail_head = GuardrailPlusHead(num_classes=cfg.num_classes, feat_channels=feat_ch, num_families=4)
        else:
            guardrail_head = GuardrailHead(num_classes=cfg.num_classes, feat_channels=feat_ch, mode=cfg.guardrail_mode)
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
            use_student_features=(feat_ch > 0),
        )
    else:
        print("[Eval] Guardrail checkpoint not found, skipping benchmark.")


def run_train_pipeline(args, cfg):
    from models import GuardrailHead, GuardrailPlusHead
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
    global_step = 0

    if not args.skip_sup:
        ckpts["student_sup"], global_step = train_supervised(
            fresh(), train_loader, val_loader, cfg, global_step=global_step)

    if not args.skip_kd:
        print(f"  [KD] Starting training ({len(train_loader)} steps/epoch)...")
        cfg_kd = build_cfg(args)
        cfg_kd.lr = cfg_kd.lr * 0.5
        student_kd = fresh()
        load_checkpoint(student_kd, ckpts["student_sup"], device=cfg_kd.device)
        ckpts["student_kd"], global_step = train_kd(
            student_kd, teacher, train_loader, val_loader, cfg_kd, global_step=global_step)

    if not args.skip_skd:
        cfg_skd = build_cfg(args)
        cfg_skd.lr = cfg_skd.lr * 0.5
        student_skd = fresh()
        load_checkpoint(student_skd, ckpts["student_sup"], device=cfg_skd.device)
        ckpts["student_skd"], global_step = train_skd(
            student_skd, teacher, train_loader, val_loader, cfg_skd, global_step=global_step)

    if not args.skip_guardrail:
        best_student = fresh()
        load_checkpoint(best_student, ckpts["student_skd"], device=cfg.device)
        best_student.to(cfg.device).eval()

        # Probe feature dim (only used if --use-student-features is set)
        feat_ch = 0
        if cfg.use_student_features:
            with torch.no_grad():
                dummy = torch.randn(1, 3, cfg.crop_size, cfg.crop_size * 2, device=cfg.device)
                _, feat = best_student(dummy, return_features=True)
                feat_ch = feat.shape[1]
                print(f"  [Guard] Student feature channels: {feat_ch}")
        else:
            print(f"  [Guard] Logits-only mode (feat_channels=0)")

        if cfg.guardrail_mode in ("utility", "margin", "guardrailpp"):
            guardrail = GuardrailPlusHead(num_classes=cfg.num_classes, feat_channels=feat_ch, num_families=4)
        else:
            guardrail = GuardrailHead(num_classes=cfg.num_classes, feat_channels=0, mode=cfg.guardrail_mode)
        _, global_step = train_guardrail(
            guardrail, best_student, teacher, train_loader, val_loader, cfg,
            use_student_features=cfg.use_student_features, global_step=global_step)
        ckpts["guardrail"] = os.path.join(cfg.output_dir, "guardrail.ckpt")

    return ckpts


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train":
        setup_wandb(cfg, args)
        ckpts = run_train_pipeline(args, cfg)
        run_eval_pipeline(args, cfg, ckpts)
        finish_wandb()

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
