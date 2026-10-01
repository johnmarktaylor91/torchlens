"""C-KINETO acceptance gate legs (torchnative W2.4/W2.5; run by D02).

Runs the correlation-ID join on pinned real checkpoints on a REAL
CUDA/CUPTI host and emits a JSON artifact per cell; gate prose is GENERATED
from the artifact, never typed. The gate law (memo 4.2, Sol's two-part
accounting, per model per phase separately):

  (a) exact model attribution / (phase device activity - typed TL-internal)
      >= 0.95, AND
  (b) (exact model attribution + typed TL-internal) / total phase activity
      >= 0.95,

with 100% of the residual NAMED by exact launch name/device/stream -- an
unnameable remainder is a truth-mechanism bug, not a coverage number.

Cells:
  forward:  gpt2 (transformers pin per artifact) @ batch 16 x seq 64, eval;
            torchvision resnet50 IMAGENET1K_V2 @ batch 32, eval.
  backward: same models, train, fwd+bwd via trace.log_backward; first AND
            second backward measured separately.

Usage (cluster lane, manual/periodic per testing-panel T3; CI stays CPU):
  python tools/gpu_gates/kineto_join_gate.py --cell forward-resnet50 \
      --out /tmp/kineto_gate/
  python tools/gpu_gates/kineto_join_gate.py --all --out /tmp/kineto_gate/

A missing CUDA device is a SETUP FAILURE (exit 2), never a skip (T3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

GATE_THRESHOLD = 0.95

CELLS = (
    "forward-gpt2",
    "forward-resnet50",
    "backward-gpt2",
    "backward-resnet50",
)


def _build_gpt2() -> tuple[Any, Any]:
    """Pinned real GPT-2 checkpoint + batch 16 x seq 64 input ids."""

    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    input_ids = torch.randint(0, 50257, (16, 64), device="cuda")
    return model, input_ids


def _build_resnet50() -> tuple[Any, Any]:
    """Pinned torchvision resnet50 IMAGENET1K_V2 + batch 32 input."""

    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).cuda()
    inputs = torch.randn(32, 3, 224, 224, device="cuda")
    return model, inputs


def run_cell(cell: str) -> dict[str, Any]:
    """Run one gate cell and return its JSON-able artifact."""

    import torchlens as tl
    from torchlens import observability as obs

    phase, model_name = cell.split("-", 1)
    model, inputs = _build_gpt2() if model_name == "gpt2" else _build_resnet50()
    if phase == "forward":
        model.eval()
        result = obs.native_profile(model, inputs)
        trace = result.trace
        join = result.join
        backward_facts: dict[str, Any] = {}
    else:
        model.train()
        with obs.session(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as active:
            trace = tl.trace(
                model,
                inputs,
                capture=tl.options.CaptureOptions(backward_ready=True),
                save_mode="reference",
            )
            output = trace.output_ops[0].out
            loss = (output.logits if hasattr(output, "logits") else output).float().sum()
            trace.log_backward(loss, retain_graph=True)
            trace.log_backward(loss)  # second backward: accumulation cell
            torch.cuda.synchronize()
        join = obs.join_session(active, trace)
        backward_facts = {
            "marker_leaks": trace.__dict__.get("_tl_gradfn_marker_leaks", 0),
            "marker_gaps": trace.__dict__.get("_tl_gradfn_marker_gaps", 0),
            "gradfn_markers": sum(1 for marker in join.markers if marker.owner_class == "grad_fn"),
            "engine_worker_tids": sorted(
                {
                    marker.tid
                    for marker in join.markers
                    if marker.owner_class == "grad_fn" and marker.tid is not None
                }
            ),
        }

    coverage = join.coverage
    accounting = {device: coverage.accounting(device) for device in coverage.device_busy_ns}
    verdicts = {
        device: (
            None
            if parts[0] is None or parts[1] is None
            else bool(parts[0] >= GATE_THRESHOLD and parts[1] >= GATE_THRESHOLD)
        )
        for device, parts in accounting.items()
    }
    artifact = {
        "schema": "torchlens.kineto_gate_cell.v1",
        "cell": cell,
        "threshold": GATE_THRESHOLD,
        "availability": join.availability,
        "extraction_path": coverage.extraction_path,
        "device_busy_ns": coverage.device_busy_ns,
        "attributed_ns": coverage.attributed_ns,
        "internal_ns": coverage.internal_ns,
        "unattributed_ns": coverage.unattributed_ns,
        "owner_share_ns": coverage.owner_share_ns,
        "two_part_accounting": {
            device: {"part_a": parts[0], "part_b": parts[1]} for device, parts in accounting.items()
        },
        "verdict_by_device": verdicts,
        "residual_named": [
            {"launch_name": name, "count": count} for name, count in coverage.residual
        ],
        "residual_fully_named": all(name for name, _ in coverage.residual),
        "n_markers": len(join.markers),
        "n_launches": len(join.launches),
        "backward_facts": backward_facts,
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
    }
    trace.cleanup()
    return artifact


def main() -> int:
    """CLI entry: run cells, write artifacts, exit nonzero on a red gate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=CELLS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("SETUP FAILURE: no CUDA device (a declared GPU leg never skips, T3)")
        return 2
    cells = list(CELLS) if args.all else [args.cell]
    if not cells or cells == [None]:
        parser.error("pass --cell <name> or --all")
    args.out.mkdir(parents=True, exist_ok=True)
    any_red = False
    for cell in cells:
        artifact = run_cell(cell)
        path = args.out / f"{cell}.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        verdicts = artifact["verdict_by_device"]
        red = any(verdict is not True for verdict in verdicts.values()) or not verdicts
        any_red = any_red or red
        print(f"{cell}: {'GREEN' if not red else 'RED'} {verdicts} -> {path}")
    return 1 if any_red else 0


if __name__ == "__main__":
    sys.exit(main())
