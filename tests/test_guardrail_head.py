#!/usr/bin/env python3
"""Smoke tests for the simplified Guardrail++ head and loss.

These are fast (<5s on CPU) sanity checks that must pass before launching
12-hour slurm jobs. They cover:
  - forward-pass shapes for every output
  - all four supervision_type modes produce finite loss
  - all four modes yield non-zero gradients on the expected heads
  - checkpoint save/load round-trip
  - config defaults are sane

Run with:  python tests/test_guardrail_head.py
"""
import os
import sys
import tempfile
import traceback

import torch
import torch.nn.functional as F

# Make src/train importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))

from models import GuardrailPlusHead  # noqa: E402
from losses import GuardrailPlusLoss   # noqa: E402
from config import Config              # noqa: E402


PASS = 0
FAIL = 0


def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))


def make_inputs(B=2, C=19, H=32, W=64, feat_ch=32, device="cpu"):
    logits = torch.randn(B, C, H, W, device=device)
    feats = torch.randn(B, feat_ch, H // 4, W // 4, device=device)
    labels = torch.randint(0, C, (B, H, W), device=device)
    # Inject some ignore pixels
    labels[:, 0, :8] = 255
    return logits, feats, labels


def make_targets(B=2, H=32, W=64, device="cpu"):
    disagree = torch.randint(0, 2, (B, H, W), device=device).float()
    gap = torch.randn(B, H, W, device=device) * 0.5
    valid = torch.ones(B, H, W, device=device)
    valid[:, 0, :8] = 0.0
    utility = torch.rand(B, device=device)
    return {
        "disagree_target": disagree,
        "disagree_valid": valid,
        "gap_target": gap * valid,
        "gap_valid": valid,
        "utility_target": utility,
    }


# ──────────────────────────────────────────────────────────────────────
# Section 1 — Head forward pass
# ──────────────────────────────────────────────────────────────────────
print("\n[1] GuardrailPlusHead forward pass")

head = GuardrailPlusHead(num_classes=19, feat_channels=0)
logits, _, _ = make_inputs(feat_ch=0)
out = head(logits)
test("forward returns dict", isinstance(out, dict))
test("has utility_score", "utility_score" in out)
test("has disagree_logits", "disagree_logits" in out)
test("has gap_pred", "gap_pred" in out)
test("utility_score shape == (B,)", out["utility_score"].shape == (2,))
test("disagree_logits shape matches input HW",
     out["disagree_logits"].shape == (2, 32, 64),
     f"got {tuple(out['disagree_logits'].shape)}")
test("gap_pred shape matches input HW",
     out["gap_pred"].shape == (2, 32, 64),
     f"got {tuple(out['gap_pred'].shape)}")

# With student features
head2 = GuardrailPlusHead(num_classes=19, feat_channels=32)
logits, feats, _ = make_inputs(feat_ch=32)
out2 = head2(logits, feats)
test("with features, disagree_logits shape",
     out2["disagree_logits"].shape == (2, 32, 64))
test("with features, gap_pred shape",
     out2["gap_pred"].shape == (2, 32, 64))


# ──────────────────────────────────────────────────────────────────────
# Section 2 — Loss modes
# ──────────────────────────────────────────────────────────────────────
print("\n[2] GuardrailPlusLoss modes")

targets = make_targets()
for st in ("scalar_benefit", "dense_disagree", "dense_gap", "dense_multi"):
    head_fresh = GuardrailPlusHead(num_classes=19, feat_channels=0)
    logits, _, _ = make_inputs(feat_ch=0)
    logits.requires_grad_(False)
    preds = head_fresh(logits)
    criterion = GuardrailPlusLoss(supervision_type=st)
    loss, info = criterion(preds, targets)
    test(f"[{st}] loss is finite", torch.isfinite(loss).item(),
         f"loss={loss.item()}")
    test(f"[{st}] loss is positive", loss.item() >= 0.0,
         f"loss={loss.item()}")

    # Gradient check: the right heads should get nonzero grad.
    loss.backward()
    disagree_grad = head_fresh.disagree_head.weight.grad
    gap_grad = head_fresh.gap_head.weight.grad
    utility_grad = head_fresh.utility_head.weight.grad

    if st == "scalar_benefit":
        test(f"[{st}] utility_head has grad", utility_grad is not None and utility_grad.abs().sum() > 0)
    if st in ("dense_disagree", "dense_multi"):
        test(f"[{st}] disagree_head has grad", disagree_grad is not None and disagree_grad.abs().sum() > 0)
    if st in ("dense_gap", "dense_multi"):
        test(f"[{st}] gap_head has grad", gap_grad is not None and gap_grad.abs().sum() > 0)

    test(f"[{st}] info contains loss key", "loss" in info)


# ──────────────────────────────────────────────────────────────────────
# Section 3 — Checkpoint round-trip
# ──────────────────────────────────────────────────────────────────────
print("\n[3] Checkpoint save/load round-trip")
h1 = GuardrailPlusHead(num_classes=19, feat_channels=32)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "guardrail.ckpt")
    torch.save(
        {
            "model": h1.state_dict(),
            "supervision_type": "dense_multi",
            "use_student_features": True,
            "dense_disagree_weight": 1.0,
            "dense_gap_weight": 1.0,
            "seed": 42,
        },
        path,
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    test("checkpoint has supervision_type", state.get("supervision_type") == "dense_multi")
    test("checkpoint has seed", state.get("seed") == 42)
    test("checkpoint has use_student_features", state.get("use_student_features") is True)

    h2 = GuardrailPlusHead(num_classes=19, feat_channels=32)
    h2.load_state_dict(state["model"])
    logits, feats, _ = make_inputs(feat_ch=32)
    o1 = h1(logits, feats)
    o2 = h2(logits, feats)
    test("round-trip disagree_logits match",
         torch.allclose(o1["disagree_logits"], o2["disagree_logits"], atol=1e-6))
    test("round-trip gap_pred match",
         torch.allclose(o1["gap_pred"], o2["gap_pred"], atol=1e-6))


# ──────────────────────────────────────────────────────────────────────
# Section 4 — Config defaults
# ──────────────────────────────────────────────────────────────────────
print("\n[4] Config defaults")
cfg = Config()
test("supervision_type default", cfg.supervision_type == "dense_multi",
     f"got {cfg.supervision_type}")
test("use_student_features default", cfg.use_student_features is True,
     f"got {cfg.use_student_features}")
test("dense_disagree_weight default", cfg.dense_disagree_weight == 1.0)
test("dense_gap_weight default", cfg.dense_gap_weight == 1.0)
test("corruption_prob default", cfg.corruption_prob == 0.5)
test("seed default", cfg.seed == 42)


# ──────────────────────────────────────────────────────────────────────
# Section 5 — End-to-end train-step shape check
# ──────────────────────────────────────────────────────────────────────
print("\n[5] Train-step shape check (mimics train_guardrail loop body)")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "train"))
try:
    from train_guardrail import _build_targets  # noqa: E402
    B, C, H, W = 2, 19, 32, 64
    student_logits = torch.randn(B, C, H, W)
    teacher_logits = torch.randn(B, C, H, W)
    labels = torch.randint(0, C, (B, H, W))
    labels[0, 0, :10] = 255  # ignore pixels
    tgt = _build_targets(student_logits, teacher_logits, labels)
    test("targets has disagree_target", "disagree_target" in tgt)
    test("targets has disagree_valid", "disagree_valid" in tgt)
    test("targets has gap_target", "gap_target" in tgt)
    test("targets has utility_target", "utility_target" in tgt)
    test("disagree_target shape matches labels",
         tgt["disagree_target"].shape == labels.shape)
    test("valid mask marks ignore as 0",
         tgt["disagree_valid"][0, 0, 5].item() == 0.0)

    # Full head + loss step
    head = GuardrailPlusHead(num_classes=C, feat_channels=0)
    criterion = GuardrailPlusLoss(supervision_type="dense_multi")
    preds = head(student_logits)
    loss, info = criterion(preds, tgt)
    loss.backward()
    test("end-to-end loss is finite", torch.isfinite(loss).item())
    test("end-to-end info has dense_disagree_loss", "dense_disagree_loss" in info)
    test("end-to-end info has dense_gap_loss", "dense_gap_loss" in info)
except Exception as exc:
    traceback.print_exc()
    test("end-to-end step raised", False, str(exc))


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n== {PASS} passed, {FAIL} failed ==")
sys.exit(0 if FAIL == 0 else 1)
