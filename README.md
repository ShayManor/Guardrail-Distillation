# Guardrail Distillation

Selective prediction for distilled semantic-segmentation students under domain shift.

This repository contains the code, evaluation scripts, and reviewer-facing checkpoints for a paper on **teacher-guided dense failure detection**. A compact SegFormer student is paired with a lightweight guardrail head that predicts whether the student is likely to fail, without calling the teacher at test time.

For anonymous review, author/institution identifiers and public release links should be removed. The camera-ready release can restore the public GitHub and Hugging Face links.

## Main idea

Knowledge distillation can preserve segmentation accuracy while leaving compact students overconfident on shifted or adverse inputs. This project trains a small guardrail head on **dense teacher-derived targets** so it can detect high-confidence student failures under domain shift.

The key comparison is:

- **Teacher-supervised guardrail:** trained from teacher-student disagreement / teacher-relative dense risk.
- **GT-supervised guardrail:** trained from student-ground-truth error / dense student CE.
- **Post-hoc baselines:** MSP, temperature-scaled MSP, entropy, MaxLogit, MC-Dropout, and related confidence scores.

## Pipeline

```text
Cityscapes -> student_sup.ckpt -> student_kd.ckpt / student_skd.ckpt -> frozen student + frozen teacher -> guardrail.ckpt
```

Stages 1–3 train the student. Stage 4 freezes the student and teacher, then trains the guardrail head on detached student logits.

## Guardrail supervision modes

| Mode | Target | Role |
|---|---|---|
| `dense_multi` | teacher-student disagreement + teacher-relative dense risk | main method |
| `dense_disagree` | teacher argmax differs from student argmax | ablation |
| `dense_gap` | student CE − teacher CE | ablation |
| `gt_disagree` | student argmax differs from ground truth | GT control |
| `gt_risk` | student CE against ground truth | GT control |
| `scalar_benefit` | image-level teacher benefit | negative-control baseline |

## Repository layout

```text
src/train/
  run.py               training CLI
  config.py            default hyperparameters
  data.py              Cityscapes / IDD / BDD loaders
  models.py            SegFormer wrapper + GuardrailPlusHead
  losses.py            segmentation, KD, SKD, guardrail losses
  train_*.py           four training stages

src/eval/
  full_eval.py         authoritative paper evaluator
  eval.py              quick sanity evaluator
  analysis.py          metric/plot helpers

src/analysis/
  figure_scripts/      paper figure scripts

slurm/
  b0/, b1/, b2/        backbone-specific training/eval jobs
  multi/               multi-seed guardrail runs

tests/
  test_guardrail_head.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Raw datasets are **not included**. Download them from their official sources and place them under:

```text
data/
  cityscapes/
    leftImg8bit/{train,val}/
    gtFine/{train,val}/
  acdc/
    rgb_anon/{fog,night,rain,snow}/val/
    gt/{fog,night,rain,snow}/val/
  idd/
    leftImg8bit/val/
    gtFine/val/
  bdd100k/
    seg/images/val/
    seg/labels/val/
```

The code expects Cityscapes-compatible 19-class train IDs for evaluation. Cityscapes raw label IDs are mapped internally.

See `ASSETS_AND_LICENSES.md` for dataset/model terms.

## Evaluation

All paper metrics flow through:

```bash
python src/eval/full_eval.py eval ...
```

Example Cityscapes evaluation:

```bash
export REPO="$(pwd)"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

python -u src/eval/full_eval.py eval   --run-id mit_b1_city_dense_multi   --dataset-name cityscapes   --dataset-path "$REPO/data/cityscapes"   --split val   --domain all   --student-name student_skd   --train-method skd   --student-backbone "$REPO/checkpoints/upstream/nvidia_mit-b1"   --student-ckpt "$REPO/checkpoints/mit-b1/student_skd.ckpt"   --teacher-backbone "$REPO/checkpoints/upstream/nvidia_segformer_b5_cityscapes"   --guardrail-ckpt "$REPO/checkpoints/mit-b1/dense_multi/guardrail.ckpt"   --guardrail-student-name student_skd   --mc-dropout-passes 4   --output-dir "$REPO/results/mit_b1_city_dense_multi"   --batch-size 4   --num-workers 4
```

Example ACDC evaluation:

```bash
for DOMAIN in fog night rain snow all; do
  python -u src/eval/full_eval.py eval     --run-id "mit_b1_acdc_${DOMAIN}_dense_multi"     --dataset-name acdc     --dataset-path "$REPO/data/acdc"     --split val     --domain "$DOMAIN"     --student-name student_skd     --train-method skd     --student-backbone "$REPO/checkpoints/upstream/nvidia_mit-b1"     --student-ckpt "$REPO/checkpoints/mit-b1/student_skd.ckpt"     --teacher-backbone "$REPO/checkpoints/upstream/nvidia_segformer_b5_cityscapes"     --guardrail-ckpt "$REPO/checkpoints/mit-b1/dense_multi/guardrail.ckpt"     --guardrail-student-name student_skd     --mc-dropout-passes 4     --output-dir "$REPO/results/mit_b1_acdc_dense_multi"     --batch-size 4     --num-workers 4
done
```

The same command pattern works for `idd` and `bdd` by changing `--dataset-name` and `--dataset-path`.

## Evaluator outputs

`full_eval.py` writes:

```text
csv/runs.csv
csv/per_image.csv
csv/per_class.csv
csv/risk_coverage.csv
csv/teacher_budget.csv
csv/calibration_bins.csv
csv/confident_failures.csv
csv/latency_samples.csv
```

These files are used to generate the paper tables, threshold sweeps, risk-coverage curves, and qualitative analyses.

## Training

SLURM scripts are provided for full retraining.

```bash
sbatch slurm/b1/train_sup.sbatch
sbatch slurm/b1/train_skd.sbatch
sbatch slurm/b1/train_guardrail.sbatch
```

Ablations:

```bash
sbatch slurm/b1/train_guardrail_dense_disagree.sbatch
sbatch slurm/b1/train_guardrail_dense_gap.sbatch
sbatch slurm/b1/train_guardrail_gt_disagree.sbatch
sbatch slurm/b1/train_guardrail_gt_risk.sbatch
sbatch slurm/b1/train_guardrail_scalar.sbatch
```

Before running on a new cluster, edit account, partition, virtualenv path, repository path, cache path, and dataset paths in the `.sbatch` files.

## Intended use

This artifact is for reproducing the paper’s experiments and analyzing semantic-segmentation failure detection under domain shift.

It is not a deployment artifact, safety certificate, autonomous-driving safety monitor, or commercial product. The guardrail is a failure-risk score and can produce both false negatives and false positives.
