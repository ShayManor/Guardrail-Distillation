---
license: other
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - semantic-segmentation
  - failure-detection
  - selective-prediction
  - domain-shift
  - knowledge-distillation
  - segformer
---

# Teacher-Guided Guardrail Checkpoints

Checkpoint documentation for the submitted paper on teacher-guided failure detection for distilled semantic segmentation.

## Model

A SegFormer student predicts semantic segmentation logits. A lightweight guardrail head consumes frozen student logits, optionally with detached student features, and predicts dense reliability maps. These maps are averaged into an image-level failure score.

The teacher is used during guardrail training only. The default guardrail does **not** invoke the teacher at test time.

## Included checkpoints

Recommended layout:

```text
checkpoints/
  mit-b1/
    student_skd.ckpt
    dense_multi/guardrail.ckpt
    dense_disagree/guardrail.ckpt
    dense_gap/guardrail.ckpt
    gt_disagree/guardrail.ckpt
    gt_risk/guardrail.ckpt
    scalar_benefit/guardrail.ckpt
```

| Checkpoint | Purpose |
|---|---|
| `student_skd.ckpt` | distilled SegFormer student |
| `dense_multi/guardrail.ckpt` | main teacher-supervised guardrail |
| `dense_disagree/guardrail.ckpt` | teacher-disagreement ablation |
| `dense_gap/guardrail.ckpt` | teacher-gap ablation |
| `gt_disagree/guardrail.ckpt` | ground-truth disagreement control |
| `gt_risk/guardrail.ckpt` | ground-truth dense-risk control |
| `scalar_benefit/guardrail.ckpt` | scalar-target negative control |

## Training data

Training uses Cityscapes-derived semantic-segmentation supervision. Guardrail training uses frozen student/teacher predictions on Cityscapes images with corruption-aware augmentation.

Raw datasets are not included.

## Evaluation data

The paper evaluates on:

- Cityscapes validation for in-domain evaluation;
- ACDC validation for fog/night/rain/snow;
- IDD validation for geographic and semantic shift;
- BDD100K validation for broader driving-scene shift.

## Intended use

Use these checkpoints to reproduce paper metrics, supervision ablations, risk-coverage curves, threshold sweeps, and qualitative failure analyses.

## Out-of-scope use

Do not use these checkpoints for:

- production deployment;
- autonomous-driving safety certification;
- standalone safety monitoring;
- commercial use that violates upstream dataset/model terms;
- unsupported tasks or datasets without additional validation.

## Loading

For direct inspection:

```python
import torch

student = torch.load("checkpoints/mit-b1/student_skd.ckpt", map_location="cpu")
guardrail = torch.load("checkpoints/mit-b1/dense_multi/guardrail.ckpt", map_location="cpu")
```

For correct preprocessing, reconstruction, and metrics, use `src/eval/full_eval.py`.

## Limitations

The guardrail is a learned risk score, not a guarantee. It can miss failures and can also over-flag correct predictions. Performance varies across domain shifts and adverse conditions. Results should not be assumed to transfer to other sensors, geographies, tasks, model families, or deployment settings without further validation.

## Upstream assets

The checkpoints depend on upstream datasets and model assets:

- Cityscapes;
- ACDC;
- IDD;
- BDD100K;
- SegFormer / NVIDIA Hugging Face checkpoints.

Raw datasets are not redistributed. See `ASSETS_AND_LICENSES.md`.

## Citation

Citation information is omitted for anonymous review.
