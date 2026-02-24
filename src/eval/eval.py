import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from data import DEFAULT_TRANSFORM, LABEL_TRANSFORM, is_hf_path, load_hf_stream, load_local_dataset
from analysis import ImageMetrics, compute_metrics


def load_model(model_tag: str, device: torch.device, num_classes: int = 19):
    """
    Load segmentation model from HF tag (must be cached locally).
    Returns (model, processor_or_None).
    """
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

    print(f"[model] Loading {model_tag} (local cache)")
    try:
        model = AutoModelForSemanticSegmentation.from_pretrained(model_tag, local_files_only=True)
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_tag, local_files_only=True)

    model.to(device).eval()

    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(model_tag, local_files_only=True)
    except Exception:
        pass

    cfg_classes = getattr(model.config, "num_labels", None)
    if cfg_classes and cfg_classes != num_classes:
        print(f"[model] WARNING: model has {cfg_classes} classes, expected {num_classes}")

    return model, processor


def _hf_processor_transform(processor):
    def transform(pil_img):
        inputs = processor(images=pil_img, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)

    return transform


@torch.no_grad()
def run_eval(
        model_tag: str,
        dataset_path: str,
        output_csv: str,
        num_classes: int = 19,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_samples: Optional[int] = None,
        image_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        use_hf_processor: bool = True,
        hf_image_key: str = "image",
        hf_label_key: str = "label",
        hf_split: Optional[str] = None,
        images_subdir: str = "images",
        labels_subdir: str = "labels",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
) -> Path:
    """Run eval, write per-image metrics CSV. Returns output path."""
    dev = torch.device(device)
    model, processor = load_model(model_tag, dev, num_classes)

    if image_transform:
        img_tf = image_transform
    elif use_hf_processor and processor:
        img_tf = _hf_processor_transform(processor)
    else:
        img_tf = DEFAULT_TRANSFORM
    lbl_tf = label_transform or LABEL_TRANSFORM

    if is_hf_path(dataset_path):
        print(f"[eval] Streaming: {dataset_path}")
        data_iter = load_hf_stream(dataset_path, hf_image_key, hf_label_key, hf_split, max_samples)
    else:
        print(f"[eval] Local: {dataset_path}")
        data_iter = load_local_dataset(dataset_path, images_subdir, labels_subdir, max_samples)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mn = model_name or model_tag.split("/")[-1]
    dn = dataset_name or dataset_path
    results: list[ImageMetrics] = []
    total_time = 0.0

    for img_id, pil_img, pil_lbl in data_iter:
        img_tensor = img_tf(pil_img)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(dev)

        lbl_np = np.array(lbl_tf(pil_lbl)).astype(np.int64)
        lbl_tensor = torch.from_numpy(lbl_np).long().to(dev)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        out = model(img_tensor)
        if hasattr(out, "logits"):
            logits = out.logits
        elif isinstance(out, dict):
            logits = out.get("out") or out.get("logits") or next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        if logits.shape[-2:] != lbl_tensor.shape[-2:]:
            logits = F.interpolate(logits, size=lbl_tensor.shape[-2:], mode="bilinear", align_corners=False)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append(compute_metrics(logits, lbl_tensor, num_classes, img_id, elapsed_ms))
        total_time += elapsed_ms

        if len(results) % 50 == 0:
            print(f"  [{len(results)}] mIoU={np.mean([r.miou for r in results]):.4f}  {elapsed_ms:.1f}ms")

    count = len(results)
    if count == 0:
        print("[eval] No images processed.")
        return out_path

    fieldnames = ["model_name", "dataset_name"] + list(ImageMetrics.__dataclass_fields__.keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["model_name"] = mn
            row["dataset_name"] = dn
            writer.writerow(row)

    agg_miou = np.mean([r.miou for r in results])
    print(f"\n  {mn} on {dn}: N={count}  mIoU={agg_miou:.4f}  avg={total_time / count:.1f}ms")
    print(f"  -> {out_path}\n")
    return out_path


if __name__ == "__main__":
    import argparse
    from analysis import get_worst_k, plot_results

    parser = argparse.ArgumentParser(description="Guardrail eval pipeline")
    sub = parser.add_subparsers(dest="cmd")

    ep = sub.add_parser("eval")
    ep.add_argument("--model", required=True, help="HuggingFace model tag (cached locally)")
    ep.add_argument("--dataset", required=True, help="Local path or hf://org/name[/split]")
    ep.add_argument("--output", default="results/eval.csv")
    ep.add_argument("--num-classes", type=int, default=19)
    ep.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ep.add_argument("--max-samples", type=int, default=None)
    ep.add_argument("--model-name", default=None)
    ep.add_argument("--dataset-name", default=None)
    ep.add_argument("--no-hf-processor", action="store_true")
    ep.add_argument("--hf-image-key", default="image")
    ep.add_argument("--hf-label-key", default="label")
    ep.add_argument("--hf-split", default=None)

    pp = sub.add_parser("plot")
    pp.add_argument("--csvs", nargs="+", required=True)
    pp.add_argument("--save-dir", default="results/figures")

    wp = sub.add_parser("worst")
    wp.add_argument("--csv", required=True)
    wp.add_argument("--k", type=float, default=5.0)
    wp.add_argument("--sort-by", default="miou")
    wp.add_argument("--descending", action="store_true")
    wp.add_argument("--output", default=None)

    args = parser.parse_args()

    if args.cmd == "eval":
        run_eval(
            model_tag=args.model, dataset_path=args.dataset, output_csv=args.output,
            num_classes=args.num_classes, device=args.device, max_samples=args.max_samples,
            use_hf_processor=not args.no_hf_processor, model_name=args.model_name,
            dataset_name=args.dataset_name, hf_image_key=args.hf_image_key,
            hf_label_key=args.hf_label_key, hf_split=args.hf_split,
        )
    elif args.cmd == "plot":
        plot_results(args.csvs, save_dir=args.save_dir)
    elif args.cmd == "worst":
        worst = get_worst_k(args.csv, k_percent=args.k, sort_by=args.sort_by, ascending=not args.descending)
        if args.output:
            worst.to_csv(args.output, index=False)
            print(f"Saved to {args.output}")
        else:
            print(worst[["image_id", "miou", "pixel_acc", "msp_mean", "entropy_mean"]].to_string())
    else:
        parser.print_help()
