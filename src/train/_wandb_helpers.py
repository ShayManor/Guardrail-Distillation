"""Weights & Biases integration for guardrail distillation training."""

import os
import tempfile
import datetime
from dataclasses import asdict


def setup_wandb(cfg, args):
    """Initialize a wandb run with full hyperparameter tracking.

    All Config fields and CLI-only args (model names, skip flags) are logged
    as wandb config so every hyperparameter is searchable and filterable.

    Returns True if wandb was initialised, False otherwise.
    """
    if getattr(args, "no_wandb", False):
        return False

    try:
        import wandb
    except ImportError:
        print("[wandb] Not installed (pip install wandb). Skipping.")
        return False

    # ── Build config dict from dataclass + CLI extras ──
    config_dict = asdict(cfg)
    config_dict["student_model"] = args.student_model
    config_dict["teacher_model"] = args.teacher_model
    config_dict["skip_sup"] = getattr(args, "skip_sup", False)
    config_dict["skip_kd"] = getattr(args, "skip_kd", False)
    config_dict["skip_skd"] = getattr(args, "skip_skd", False)
    config_dict["skip_guardrail"] = getattr(args, "skip_guardrail", False)

    # ── Experiment name ──
    exp_name = getattr(args, "wandb_name", None)
    if not exp_name:
        parts = [args.student_model.split("/")[-1], cfg.supervision_type]
        slurm_job = os.environ.get("SLURM_JOB_ID")
        if slurm_job:
            slurm_proc = os.environ.get("SLURM_PROCID", "0")
            parts.append(f"j{slurm_job}.{slurm_proc}")
        parts.append(datetime.datetime.now().strftime("%m%d_%H%M%S"))
        exp_name = "_".join(parts)

    try:
        wandb.init(
            project=getattr(args, "wandb_project", None) or "guardrail-distillation",
            group=getattr(args, "wandb_group", None),
            name=exp_name,
            config=config_dict,
            save_code=True,
            dir=tempfile.mkdtemp(),
            settings=wandb.Settings(start_method="thread"),
            tags=[cfg.supervision_type, args.student_model.split("/")[-1]],
        )
    except Exception as e:
        print(f"[wandb] Init failed ({e}). Training will continue without wandb.")
        return False

    # Tell wandb which logged value to use as x-axis for each metric group.
    wandb.define_metric("global_step")
    for prefix in ("supervised", "kd", "skd", "guardrail", "system", "perf"):
        wandb.define_metric(f"{prefix}/*", step_metric="global_step")

    print(f"[wandb] Run: {wandb.run.name} | {wandb.run.get_url()}")
    return True


# ── Logging helpers ──────────────────────────────────────────────────────────


def wandb_log(metrics, step=None):
    """Log *metrics* to wandb.  No-op when wandb is not active or on error."""
    try:
        import wandb

        if wandb.run is None:
            return
        if step is not None:
            metrics["global_step"] = step
        wandb.log(metrics)
    except ImportError:
        pass
    except Exception:
        # Network blip, auth expiry, etc. — never crash training.
        pass


def log_system_metrics(step):
    """Log torch CUDA memory usage.

    wandb's built-in system monitor already tracks GPU utilization, CPU %,
    RAM, disk I/O, and network.  Here we add torch-level memory stats which
    reflect the Python process specifically (useful when multiple jobs share
    a GPU).
    """
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return

    metrics = {}
    try:
        import torch

        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_gb"] = (
                torch.cuda.memory_allocated() / 1e9
            )
            metrics["system/gpu_memory_reserved_gb"] = (
                torch.cuda.memory_reserved() / 1e9
            )
            total = torch.cuda.get_device_properties(0).total_mem
            metrics["system/gpu_memory_utilization_pct"] = (
                100.0 * torch.cuda.memory_allocated() / total
            )
    except Exception:
        pass

    if metrics:
        metrics["global_step"] = step
        try:
            wandb.log(metrics)
        except Exception:
            pass


# ── Evaluation logging ───────────────────────────────────────────────────────


def log_eval_results(benchmark_summary, confident_failure_results=None,
                     save_path=None):
    """Log evaluation benchmark results to wandb.

    - Scalar metrics under ``eval/{student}/{method}/...``
    - ``wandb.Table`` for the full benchmark grid
    - Confident-failure-detection table
    - Plots from *save_path* as ``wandb.Image``
    - ``wandb.run.summary`` entries for best AURC per student
    """
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return

    try:
        # ── Scalar metrics ──
        eval_metrics = {}
        for row in benchmark_summary:
            student = row["student"]
            method = row["method"].replace(" ", "_").replace("-", "_")
            pfx = f"eval/{student}/{method}"
            eval_metrics[f"{pfx}/miou"] = row["miou"]
            eval_metrics[f"{pfx}/aurc"] = row["aurc"]
            eval_metrics[f"{pfx}/risk_at_80"] = row["risk_at_80"]
            eval_metrics[f"{pfx}/risk_at_90"] = row["risk_at_90"]
            eval_metrics[f"{pfx}/risk_at_95"] = row["risk_at_95"]
        wandb.log(eval_metrics)

        # ── Benchmark table ──
        cols = list(benchmark_summary[0].keys())
        table = wandb.Table(columns=cols)
        for row in benchmark_summary:
            table.add_data(*[row[c] for c in cols])
        wandb.log({"eval/benchmark_table": table})

        # ── Confident-failure table ──
        if confident_failure_results:
            cfd_rows = []
            for sname, cfd in confident_failure_results.items():
                for thresh, m in cfd.items():
                    r = {"student": sname, "msp_threshold": thresh}
                    r.update(m)
                    cfd_rows.append(r)
            if cfd_rows:
                all_cols = list(dict.fromkeys(k for r in cfd_rows for k in r.keys()))
                cfd_table = wandb.Table(columns=all_cols)
                for r in cfd_rows:
                    cfd_table.add_data(*[r.get(c, None) for c in all_cols])
                wandb.log({"eval/confident_failure_table": cfd_table})

        # ── Plots ──
        if save_path:
            from pathlib import Path

            images = {}
            for png in sorted(Path(save_path).glob("*.png")):
                images[f"eval/plots/{png.stem}"] = wandb.Image(str(png))
            if images:
                wandb.log(images)

        # ── Summary: best AURC per student ──
        best = {}
        for row in benchmark_summary:
            s = row["student"]
            if s not in best or row["aurc"] < best[s]["aurc"]:
                best[s] = row
        for s, row in best.items():
            wandb.run.summary[f"best/{s}/aurc"] = row["aurc"]
            wandb.run.summary[f"best/{s}/method"] = row["method"]
            wandb.run.summary[f"best/{s}/miou"] = row["miou"]

    except Exception as e:
        print(f"[wandb] Eval logging failed ({e}). Results are still saved locally.")


def finish_wandb():
    """Finish the wandb run (flush + upload)."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
