import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from src.eval.data import DEFAULT_TRANSFORM, LABEL_TRANSFORM, is_hf_path, is_kaggle_path, load_hf_stream, load_local_dataset, load_kaggle_dataset
from src.eval.analysis import ImageMetrics, compute_metrics

def sliding_window_inference(model, img_tensor, crop_size=(1024, 1024), stride=(768, 768), num_classes=19):
    """Sliding window with overlapping crops, averaging logits."""
    B, C, H, W = img_tensor.shape
    logits = img_tensor.new_zeros((B, num_classes, H, W))
    count = img_tensor.new_zeros((B, 1, H, W))

    for y in range(0, H, stride[0]):
        for x in range(0, W, stride[1]):
            y1 = min(y, H - crop_size[0])
            x1 = min(x, W - crop_size[1])
            y2, x2 = y1 + crop_size[0], x1 + crop_size[1]

            crop = img_tensor[:, :, y1:y2, x1:x2]
            out = model(crop)
            crop_logits = out.logits if hasattr(out, 'logits') else out[0]
            crop_logits = F.interpolate(crop_logits, size=crop_size, mode='bilinear', align_corners=False)

            logits[:, :, y1:y2, x1:x2] += crop_logits
            count[:, :, y1:y2, x1:x2] += 1

    return logits / count

def load_model(model_tag: str, device: torch.device, num_classes: int = 19, do_resize=False):
    """
    Load segmentation model from HF tag (must be cached locally).
    Returns (model, processor_or_None).
    """
    from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

    ckpt_path = None
    arch_tag = model_tag

    # Check if it's a checkpoint path
    if "::" in model_tag:
        # Format: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024::/path/to/finetuned.ckpt"
        arch_tag, ckpt_path = model_tag.split("::", 1)
    elif any(model_tag.endswith(ext) for ext in (".ckpt", ".pth", ".pt", ".bin", ".safetensors")):
        ckpt_path = model_tag
        arch_tag = None

    if arch_tag:
        print(f"[model] Loading architecture {arch_tag}")
        try:
            model = AutoModelForSemanticSegmentation.from_pretrained(arch_tag, local_files_only=True)
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(arch_tag, local_files_only=True)
    else:
        raise ValueError(
            f"Cannot infer architecture from checkpoint path alone: {model_tag}\n"
            f"Use format: 'hf_model_tag::/path/to/weights.ckpt'"
        )

    if ckpt_path:
        print(f"[model] Loading weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle common checkpoint formats
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "model" in state:
                state = state["model"]
        # Strip "model." prefix if present (common in lightning checkpoints)
        cleaned = {}
        for k, v in state.items():
            k = k.removeprefix("model.")
            cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[model] Missing keys: {len(missing)} (first 5: {missing[:5]})")
        if unexpected:
            print(f"[model] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    model.to(device).eval()

    processor = None
    try:
        tag_for_proc = arch_tag or model_tag.split("::")[0]
        processor = AutoImageProcessor.from_pretrained(
            tag_for_proc, local_files_only=True, do_resize=do_resize, use_fast=False
        )
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
    use_sliding_window: bool = False,
    do_resize: bool = False,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    label_map: Optional[dict[int, int]] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """Run eval, write per-image metrics CSV. Returns output path.

    dataset_path formats:
      - Local:  "./data/cityscapes/val"
      - HF:    "hf://org/dataset[/split]"    (streamed, no disk usage)
      - Kaggle: "kaggle://owner/dataset[/split]" (downloaded once to ~/.cache/kaggle_datasets/)
    """
    dev = torch.device(device)
    model, processor = load_model(model_tag, dev, num_classes, do_resize)
    print(f"[eval] Using transform: {'hf_processor' if use_hf_processor and processor else 'default'}")

    if image_transform:
        img_tf = image_transform
    elif use_hf_processor and processor:
        img_tf = _hf_processor_transform(processor)
    else:
        img_tf = DEFAULT_TRANSFORM
    lbl_tf = label_transform or LABEL_TRANSFORM

    if is_hf_path(dataset_path):
        print(f"[eval] Pulling: {dataset_path}")
        data_iter = load_hf_stream(dataset_path, hf_image_key, hf_label_key, hf_split, max_samples)
    elif is_kaggle_path(dataset_path):
        print(f"[eval] Kaggle: {dataset_path}")
        data_iter = load_kaggle_dataset(dataset_path, images_subdir, labels_subdir, max_samples=max_samples)
    else:
        print(f"[eval] Local: {dataset_path}")
        data_iter = load_local_dataset(dataset_path, images_subdir, labels_subdir, max_samples)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mn = model_name or model_tag.split("/")[-1]
    dn = dataset_name or dataset_path
    results: list[ImageMetrics] = []
    total_time = 0.0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for img_id, pil_img, pil_lbl in data_iter:
        img_tensor = img_tf(pil_img)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(dev)
        lbl_np = np.array(lbl_tf(pil_lbl)).astype(np.int64)
        if lbl_np.ndim == 3:
            lbl_np = lbl_np[:, :, 0]
        if label_map is not None:
            if isinstance(next(iter(label_map)), tuple):
                palette = np.array(list(label_map.keys()), dtype=np.float32)
                ids = np.array(list(label_map.values()), dtype=np.int64)
                pixels = lbl_np[:, :, :3].reshape(-1, 3).astype(np.float32)
                dists = np.linalg.norm(pixels[:, None] - palette[None, :], axis=2)
                nearest_idx = dists.argmin(axis=1)
                min_dist = dists[np.arange(len(pixels)), nearest_idx]
                mapped = ids[nearest_idx]
                mapped[min_dist > 30] = 255
                lbl_np = mapped.reshape(lbl_np.shape[:2])
            else:
                lbl_np = np.vectorize(lambda x: label_map.get(x, 255))(lbl_np)
        # lbl_np[(lbl_np >= num_classes) & (lbl_np != 255)] = 255
        if not results:
            total = lbl_np.size
            ignored = (lbl_np == 255).sum()
            if ignored / total > 0.3:
                print(f"[WARN] shape={lbl_np.shape} ignored={ignored}/{total} ({100 * ignored / total:.1f}%)")
            print(f"[INFO] unique classes: {np.unique(lbl_np)}")

        lbl_np[lbl_np >= num_classes] = 255
        lbl_tensor = torch.from_numpy(lbl_np).long().to(dev)

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if use_sliding_window:
            logits = sliding_window_inference(model, img_tensor, num_classes=num_classes)
        else:
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

        if use_sliding_window:
            pred = logits.argmax(dim=1).squeeze(0)
            lab = lbl_tensor.squeeze()
            valid = lab != 255
            idx = (lab[valid] * num_classes + pred[valid]).view(-1).cpu()
            conf_mat += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

        results.append(compute_metrics(logits, lbl_tensor, num_classes, img_id, elapsed_ms))
        total_time += elapsed_ms

        if len(results) % 50 == 0:
            if use_sliding_window:
                inter = conf_mat.diag().float()
                union = conf_mat.sum(0).float() + conf_mat.sum(1).float() - inter
                miou = (inter / union.clamp(min=1))[union > 0].mean().item()
            else:
                miou = np.mean([r.miou for r in results])
            print(f"  [{len(results)}] mIoU={miou:.4f}  {elapsed_ms:.1f}ms")

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

    # agg_miou = np.mean([r.miou for r in results])
    if use_sliding_window:
        inter = conf_mat.diag().float()
        union = conf_mat.sum(0).float() + conf_mat.sum(1).float() - inter
        agg_miou = (inter / union.clamp(min=1))[union > 0].mean().item()
    else:
        agg_miou = np.mean([r.miou for r in results])
    print(f"\n  {mn} on {dn}: N={count}  mIoU={agg_miou:.4f}  avg={total_time/count:.1f}ms")
    print(f"  -> {out_path}\n")
    return out_path


if __name__ == "__main__":
    import argparse
    from analysis import get_worst_k, plot_results

    parser = argparse.ArgumentParser(description="Guardrail eval pipeline")
    sub = parser.add_subparsers(dest="cmd")

    ep = sub.add_parser("eval")
    ep.add_argument("--model", required=True, help="HuggingFace model tag (cached locally)")
    ep.add_argument("--dataset", required=True, help="Local path, hf://org/name[/split], or kaggle://owner/dataset[/split]")
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
    ep.add_argument("--images-subdir", default="images")
    ep.add_argument("--labels-subdir", default="labels")
    ep.add_argument("--sliding-window", action="store_true")

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
            images_subdir=args.images_subdir,
            labels_subdir=args.labels_subdir,
            use_sliding_window=args.sliding_window,
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
