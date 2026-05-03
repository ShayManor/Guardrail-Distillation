# Assets and Licenses

This artifact uses existing driving-scene datasets and pretrained model assets. Raw dataset images and labels are **not redistributed**.

## Existing datasets

| Asset | Used for | Redistributed? | Terms                     |
|---|---|---:|---------------------------|
| Cityscapes | training and in-domain validation | No | official Cityscapes terms |
| ACDC | adverse weather/lighting evaluation | No | official ACDC license/terms |
| IDD | geographic and semantic-shift evaluation | No | official IDD portal terms |
| BDD100K | broader driving-scene evaluation | No | official BDD100K terms    |

Users must download each dataset from its official source and comply with the original terms.

## Existing model assets

| Asset | Used for | Redistributed? | Terms |
|---|---|---:|---|
| SegFormer model family | teacher/student architecture | No | cite upstream SegFormer paper |
| `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` | teacher model | No unless explicitly included | upstream Hugging Face/NVIDIA model-card terms |
| `nvidia/mit-b0`, `nvidia/mit-b1`, `nvidia/mit-b2` | student backbone initialization | No unless explicitly included | upstream Hugging Face/NVIDIA model-card terms |

The evaluator expects upstream Hugging Face assets to be available locally.

## New assets in this artifact

The artifact may include:

- distilled student checkpoints;
- teacher-supervised guardrail checkpoints;
- ground-truth-supervised control checkpoints;
- scalar-target control checkpoints;
- training/evaluation/analysis code.

These assets are provided for anonymous review and reproducibility. They do not include raw dataset images or labels.

## Code license

```text
Apache License 2.0
```

This license applies to repository code only. It does not override dataset, checkpoint, or upstream model terms.

## Checkpoint terms

```text
Research-use checkpoint release. Subject to upstream dataset and model terms.
No raw dataset images or labels are included.
Not certified for safety-critical deployment.
```

## Intended use

The released checkpoints and scripts are intended for reproducing the paper’s experiments and analyzing semantic-segmentation failure detection under domain shift.

They are not intended for autonomous-driving deployment, safety certification, or unsupported commercial use.
