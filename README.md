# Guardrail-Distillation

Light-weight, OOD-aware selective prediction for edge-deployed semantic
segmentation students distilled from a large vision teacher.

We train a 4-stage pipeline:

1. `student_sup`  – supervised baseline (CE + Dice)
2. `student_kd`   – soft-label KD from the frozen teacher
3. `student_skd`  – structured KD (KD + pairwise-affinity feature match)
4. **`guardrail`** – a small selective-prediction head on top of the frozen
   student that is trained with dense per-pixel teacher/student disagreement
   supervision. This is the paper's contribution.

The guardrail head takes the (detached) student logits and, optionally, the
student backbone features, and outputs three things through a shared 3-conv
dense encoder:

| output | supervision | used at eval as |
|---|---|---|
| `utility_score` (scalar) | scalar teacher-benefit (ablation only) | `guardrailpp_utility_scalar` |
| `disagree_logits` (per-pixel BCE) | per-pixel teacher ≠ student mask | `guardrailpp_utility_dense_bce` |
| `gap_pred` (per-pixel linear) | per-pixel `student_ce − teacher_ce` | `guardrailpp_utility_dense_gap` |

At inference time a single scalar is obtained by averaging the per-pixel
dense outputs over valid pixels. The primary score for the paper is
`guardrailpp_utility_dense_gap`; `guardrailpp_utility_dense_bce` is a close
runner-up. Both beat MSP / entropy / temp-MSP / MC-dropout on
confident-failure AUROC under domain shift by large margins, at <3% latency
overhead on top of the student forward pass.

## Paper plan (NeurIPS submission, see `/root/.claude/plans/`)

1. **Method.** Dense pixel-level supervision of a cheap selective-prediction
   head. No retraining of the student; the head adds <3% inference cost.
2. **Empirical headline.** Confident-failure AUROC @ msp≥0.85 on Cityscapes-val
   of **0.75 / 0.80 / 0.73** for mit-b0/b1/b2, beating MSP by **+0.16 / +0.22 /
   +0.16**. On ACDC under domain shift we get **0.80** on b0 vs MSP 0.56.
3. **Analytic (negative) result.** Image-level scalar `teacher_benefit` is
   structurally unpredictable from student features: `corr(student_risk,
   teacher_risk) ≈ 0.81` bounds the R² of any scalar benefit predictor below
   0.06. The paper explains why the dense supervision sidesteps this bound.

## Installing and running

```bash
pip install -r requirements.txt
```

### Stages 1-3 (student distillation)

Slurm scripts per backbone in `slurm/{b0,b1,b2}/`:

```bash
sbatch slurm/b1/train_sup.sbatch        # ~12h
sbatch slurm/b1/train_skd.sbatch        # ~12h, uses the sup checkpoint
```

(`train_kd.sbatch` is provided for the soft-label-KD-only comparison row in
Table 1 but is not on the critical path for the main method.)

### Stage 4 — guardrail head

The paper's primary run trains the head with `supervision_type=dense_multi`
(the default):

```bash
sbatch slurm/b1/train_guardrail.sbatch  # ~12h
```

Override the supervision type via env var:

```bash
SUPERVISION_TYPE=scalar_benefit sbatch slurm/b1/train_guardrail.sbatch
SUPERVISION_TYPE=dense_disagree sbatch slurm/b1/train_guardrail.sbatch
SUPERVISION_TYPE=dense_gap      sbatch slurm/b1/train_guardrail.sbatch
```

### E2 — supervision-type ablation (paper Table 2)

Each supervision mode is a separate, single-task 12 h job. Queue them
individually; no array jobs. `dense_multi` is the paper's primary run and is
the default for `train_guardrail.sbatch`.

```bash
sbatch slurm/b1/train_guardrail.sbatch                   # dense_multi (default/primary)
sbatch slurm/b1/train_guardrail_scalar.sbatch            # scalar_benefit ablation
sbatch slurm/b1/train_guardrail_dense_disagree.sbatch    # dense_disagree ablation
sbatch slurm/b1/train_guardrail_dense_gap.sbatch         # dense_gap ablation
```

Outputs land in `runs/mit-b1_guard_<mode>_j<jobid>/`.

### E4 — Deep Ensemble baseline (paper Table 1, ensemble row)

Three independent SKD members for mit-b1 from the shared sup checkpoint.
Each file is a single 12 h job; queue as many as parallel-GPU budget allows.

```bash
sbatch slurm/b1/train_ensemble_m1.sbatch   # seed 42
sbatch slurm/b1/train_ensemble_m2.sbatch   # seed 137
sbatch slurm/b1/train_ensemble_m3.sbatch   # seed 256
```

Outputs: `runs/mit-b1_ensemble_m{1,2,3}_seed<seed>_j<jobid>/student_skd.ckpt`.

### Eval

Paper numbers come from `src/eval/full_eval.py`, which consumes a guardrail
checkpoint + student checkpoint and emits per-image, per-class,
risk-coverage, teacher-budget, confident-failures and calibration CSVs under
`src/analysis/`.

```bash
sbatch slurm/b1/eval_city.sbatch   # Cityscapes val
sbatch slurm/b1/eval_acdc.sbatch   # ACDC fog / night / rain / snow / all
```

`full_eval.py` supports per-run seeding via `--seed` and multi-seed
aggregation via `--seeds 42,137,256`.

## Repository layout

```
src/train/
  models.py              GuardrailPlusHead + teacher/student wrappers
  losses.py              SegLoss, KDLoss, PairwiseAffinityLoss, GuardrailPlusLoss
  train_guardrail.py     Stage-4 training loop (dense_multi by default)
  train_supervised.py    Stage-1 training loop
  train_kd.py            Stage-2 training loop
  train_skd.py           Stage-3 training loop
  run.py                 Top-level CLI (stages 1-4)
  config.py              Dataclass config, lives alongside CLI flags
  data.py, utils.py      dataloaders, schedulers, checkpoint helpers
src/eval/
  full_eval.py           Authoritative eval — per_image, risk_coverage,
                         teacher_budget, confident_failures, runs.csv
  eval.py, data.py       Helpers for the sanity-eval path in run.py
src/analysis/            CSVs + figure-generation scripts
slurm/{b0,b1,b2}/        SLURM job files for every stage
tests/
  test_guardrail_head.py Fast CPU smoke tests that MUST pass before
                         queueing a 12h job
```

## Before queueing a training job

Always run the CPU smoke test first — it validates the head architecture,
loss modes, and target computation in <5 seconds:

```bash
python tests/test_guardrail_head.py
```

46 tests currently; any failure means the training will waste 12 hours.
