# Guardrail-Distillation

Selective prediction for edge-deployed semantic-segmentation students that
have been distilled from a large vision teacher.

A small SegFormer student (mit-b0/b1/b2, distilled from
`nvidia/segformer-b5-finetuned-cityscapes-1024-1024`) is paired with a
lightweight **GuardrailPlusHead** that emits a per-image reliability score.
The score is used for (i) abstention and deferral to the teacher and
(ii) detecting confident failures under real domain shift (ACDC fog / night
/ rain / snow, plus IDD and BDD as cross-dataset shifts). The head adds
under 3% latency on top of the student forward pass and never backpropagates
into the student.

## Headline result

Per-pixel teacher/student disagreement supervision beats every common
confidence baseline on confident-failure AUROC under domain shift, at
negligible inference cost. Concretely (mit-b1, MSP threshold 0.85):

| dataset            | MSP    | TempMSP | Energy | MaxLogit | MC-Drop | Ours (dense_multi) |
|--------------------|--------|---------|--------|----------|---------|--------------------|
| Cityscapes (in-d.) | ~0.66  | ~0.66   | ~0.65  | ~0.66    | ~0.66   | **~0.80**          |
| ACDC               | ~0.56  | ~0.57   | ~0.59  | ~0.59    | ~0.58   | **~0.80**          |

Numbers come from the per-paper CSVs under `src/analysis/combined_all/`.

## Why dense supervision

The image-level scalar `teacher_benefit = student_risk − teacher_risk` is
*structurally* unpredictable from student features:
`corr(student_risk, teacher_risk) ≈ 0.81` on Cityscapes-val and ~0.78 on
ACDC, which bounds the R² of any scalar benefit predictor below ~0.06.
Dense per-pixel disagreement signals sidestep this trap (~1M pixels of
training signal per image instead of a single residual scalar) and are what
the paper actually trains the head on.

## Pipeline

Four sequential stages, each a separate SLURM job. Stages 1–3 train the
student; stage 4 trains the guardrail head on a *frozen* student and
*frozen* teacher.

```
student_sup.ckpt → student_kd.ckpt | student_skd.ckpt → guardrail.ckpt
   (CE + Dice)      (KL on softened    (KL + pairwise      (frozen student
                     teacher logits)    feature affinity)    + frozen teacher)
```

Stage 4 emits three heads from a shared 3-conv encoder over (detached)
student logits — optionally concatenated with detached student features:

| head              | shape         | trained when                                       |
|-------------------|---------------|----------------------------------------------------|
| `disagree_logits` | `(B, H, W)`   | `dense_multi`, `dense_disagree`, `gt_disagree`     |
| `gap_pred`        | `(B, H, W)`   | `dense_multi`, `dense_gap`, `gt_risk`              |
| `utility_score`   | `(B,)` scalar | `scalar_benefit` only (negative-result baseline)   |

Inference averages the per-pixel dense outputs over valid pixels into a
single scalar per image. `full_eval.py` aliases that into
`guardrailpp_utility` based on which head was actually trained for the
checkpoint's `supervision_type`, so the alias never reads from an untrained
output.

## Supervision modes

Set via `--supervision-type` on `run.py` or `SUPERVISION_TYPE=...` env var
in SLURM:

| mode             | target                                  | role                 |
|------------------|-----------------------------------------|----------------------|
| `dense_multi`    | both dense targets, summed              | **paper headline**   |
| `dense_disagree` | teacher_argmax ≠ student_argmax (BCE)   | ablation             |
| `dense_gap`      | student_ce − teacher_ce (smooth-L1)     | ablation             |
| `gt_disagree`    | student_argmax ≠ ground_truth (BCE)     | teacher-vs-GT control |
| `gt_risk`        | per-pixel student CE vs GT              | teacher-vs-GT control |
| `scalar_benefit` | image-level teacher_benefit (smooth-L1) | negative-result row  |

## Quickstart

Install once:

```bash
pip install -r requirements.txt
```

Always run the CPU smoke test before queueing a 12h training job — 68
tests, finishes in under 5 seconds:

```bash
python tests/test_guardrail_head.py
```

### Train

```bash
# Stages 1–3 (student): one 12h job each, sequential.
sbatch slurm/b1/train_sup.sbatch
sbatch slurm/b1/train_skd.sbatch          # branches off student_sup
sbatch slurm/b1/train_kd.sbatch           # alternative to skd; only needed for the KD-only baseline row

# Stage 4 (guardrail head): primary run + ablation rows.
sbatch slurm/b1/train_guardrail.sbatch                  # dense_multi (default)
sbatch slurm/b1/train_guardrail_dense_disagree.sbatch
sbatch slurm/b1/train_guardrail_dense_gap.sbatch
sbatch slurm/b1/train_guardrail_gt_disagree.sbatch
sbatch slurm/b1/train_guardrail_gt_risk.sbatch
sbatch slurm/b1/train_guardrail_scalar.sbatch
```

The same five ablation files exist for `b0/` and `b2/`. Multi-seed
guardrail retrains live under `slurm/multi/` (seeds 137 and 256 on top of
the default seed 42). One 12h job per seed; queue independently.

### Evaluate

Paper numbers come from `src/eval/full_eval.py`, which consumes a student
checkpoint + an optional guardrail checkpoint + an optional teacher and
upserts CSVs into `src/analysis/<experiment>/csv/`:
`runs.csv`, `per_image.csv`, `per_class.csv`, `risk_coverage.csv`,
`teacher_budget.csv`, `confident_failures.csv`, `calibration_bins.csv`,
`latency_samples.csv`.

```bash
sbatch slurm/b1/eval_city.sbatch          # Cityscapes val (in-domain)
sbatch slurm/b1/eval_acdc.sbatch          # ACDC fog/night/rain/snow + pooled
sbatch slurm/b1/eval_all_ablations.sbatch # sweep newest checkpoint per (backbone, mode)
```

For an ablation eval run, **always** pass `GUARD_DIR=` explicitly:

```bash
GUARD_DIR=runs/mit-b1_guard_dense_gap_j12346 sbatch slurm/b1/eval_acdc.sbatch
```

`full_eval.py` accepts `--seed` and `--seeds 42,137,256` for eval-only
multi-seed aggregation; `--seeds` overrides `--seed`.

### Render figures

```bash
python src/analysis/figure_scripts/run_all_figures.py
sbatch slurm/b1/figure_silent_failure.sbatch   # qualitative grid
```

Figures land in `src/analysis/figures/`.

## Repository layout

```
src/train/
  config.py            dataclass defaults; CLI flags on run.py override
  models.py            SegModel / HFSegModelWrapper / GuardrailPlusHead
  losses.py            SegLoss, KDLoss, PairwiseAffinityLoss, GuardrailPlusLoss
  data.py              Cityscapes / IDD / BDD / HF segmentation datasets
  utils.py             seeding, mIoU, checkpoints, schedulers, evaluator
  _wandb_helpers.py    optional W&B integration; no-op when wandb missing
  train_supervised.py  stage 1
  train_kd.py          stage 2
  train_skd.py         stage 3
  train_guardrail.py   stage 4 (dense supervision + corruption augmentation)
  run.py               single CLI for all four stages

src/eval/
  full_eval.py         authoritative paper evaluator → CSVs + figures
  eval.py              fast per-image driver used by run.py's sanity pass
  data.py              local / HF / Kaggle iterators for eval.py
  analysis.py          per-image metrics + 6-panel sanity plot

src/analysis/
  figure_scripts/      one script per paper figure; reads combined_all/
  *.py                 standalone figure generators (predecessors of figure_scripts/)

scripts/
  make_silent_failure_figure.py  qualitative 4×7 grid; called by sbatch

slurm/
  {b0,b1,b2}/          per-backbone training and eval jobs
  multi/               multi-seed guardrail retrains (seeds 137, 256)
  eval.sbatch          single-checkpoint eval; eval_all.sbatch sweeps everything

tests/
  test_guardrail_head.py   68 CPU tests, MUST pass before any 12h job
```

## SLURM rules

Single task per file, no arrays, 12h cap. Output naming prevents collisions
across ablations:

- Single seed: `runs/mit-b<N>_<stage>_<mode>_j<jobid>/`
- Multi-seed:  `runs/mit-b<N>_guard_dense_multi_s<seed>_j<jobid>/`

Eval jobs auto-discover the newest student/guardrail checkpoint by mtime,
so always pass `GUARD_DIR=` and (if needed) `SKD_DIR=` when targeting a
specific ablation.
