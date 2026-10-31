"""V-JEPA 2 full-clip capture and validation leg (GPU venue; U55.4).

Runs ``facebook/vjepa2-vitl-fpc64-256`` at its native 64 frames x 256 px
through ``tl.trace`` and ``tl.validate(scope="forward")`` on one CUDA device
and writes a receipt with the environment fingerprint. CPU CI covers video
with the VideoMAE registry row instead; that row never stands in for this
receipt. Usage::

    python tests/real_model/gpu_legs/vjepa2_full_leg.py OUT_DIR [--frames 64] [--job-id ID]

The checkpoint comes from the registry row ``r2-vjepa2-vitl-fpc64-256`` (model
id, pinned revision, weight digest); the weights must already be in the
Hugging Face cache, because the leg runs offline.
Each stage writes ``stage<A|B>.json``; a rerun skips a stage whose receipt
exists, so a timeout loses at most one stage. Run stage B in a fresh process
(rerun after stage A): the caching allocator keeps stage A's GPU memory, and
the save budget is a fraction of what is free. ``--validate-frames N`` splits
validation onto a shorter clip when the full clip does not fit; the receipt
then says the full leg did NOT pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ARTIFACT_ID = "r2-vjepa2-vitl-fpc64-256"
WEIGHT_FILE = "model.safetensors"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
CROP = 256
SEED = 0


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _driver() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _row() -> Any:
    from tests.real_model.registry import load_registry

    return load_registry().checkpoint_evidence(ARTIFACT_ID)


def fingerprint(job_id: str) -> dict[str, Any]:
    """Return the GPU environment fingerprint the receipt carries."""

    import torch
    import transformers

    import torchlens

    cuda = torch.cuda.is_available()
    return {
        "job_id": job_id,
        "python": platform.python_version(),
        "glibc": "-".join(platform.libc_ver()),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "cuda_available": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
        "driver": _driver(),
        "transformers": transformers.__version__,
        "torchlens": getattr(torchlens, "__version__", "unknown"),
        "torchlens_commit": os.environ.get("TORCHLENS_COMMIT", "unknown"),
        "artifact_id": ARTIFACT_ID,
        "model_id": _row().model_id,
        "revision": _row().revision,
    }


def _clip(frames: int) -> Any:
    import torch

    generator = torch.Generator().manual_seed(SEED)
    raw = torch.rand(1, frames, 3, CROP, CROP, generator=generator)
    mean = torch.tensor(IMAGE_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(1, 1, 3, 1, 1)
    return (raw - mean) / std


def _load(device: str) -> Any:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    row = _row()
    pinned = next(blob.sha256 for blob in row.blobs if blob.filename == WEIGHT_FILE)
    weights = hf_hub_download(
        row.model_id, WEIGHT_FILE, revision=row.revision, local_files_only=True
    )
    actual = _sha256(weights)
    if actual != pinned:
        raise SystemExit(f"weight sha256 {actual} != the registry's {pinned}")
    model = AutoModel.from_pretrained(
        row.model_id, revision=row.revision, dtype=torch.float32, attn_implementation="sdpa"
    )
    return model.to(device).eval()


def _peaks() -> dict[str, float]:
    import torch

    peak = {"peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_available():
        peak["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / (1 << 20)
    return peak


def stage_a(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Capture every op, activations to host; traced output == direct forward."""

    import torch

    import torchlens as tl
    from torchlens.options import CaptureOptions

    with torch.no_grad():
        direct = model(**inputs).last_hidden_state.detach().cpu()
    start = time.monotonic()
    trace = tl.trace(model, (), dict(inputs), capture=CaptureOptions(output_device="cpu"))
    seconds = time.monotonic() - start
    try:
        matches = [
            op
            for op in trace.output_ops
            if isinstance(op.out, torch.Tensor) and op.out.shape == direct.shape
        ]
        equal = any(torch.equal(op.out.cpu(), direct) for op in matches)
        max_abs = min((float((op.out.cpu() - direct).abs().max()) for op in matches), default=None)
        return {
            "status": trace.outcome.status.name,
            "traced_equals_direct": equal,
            "max_abs_diff_vs_direct": max_abs,
            "num_ops": int(trace.num_ops),
            "num_layers": int(len(trace.layers)),
            "num_modules": int(trace.num_modules),
            "saved_activation_memory_bytes": int(trace.cost.saved_activation_memory),
            "trace_seconds": round(seconds, 1),
            **_peaks(),
        }
    finally:
        trace.cleanup()


def stage_b(model: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Forward replay validation on the same clip."""

    import torch

    import torchlens as tl

    # Inference-mode validation: the capture records grad mode off and replay
    # honors it, so no autograd graph of the reference forward stays alive.
    start = time.monotonic()
    with torch.no_grad():
        ok = tl.validate(model, (), dict(inputs), scope="forward", output_device="cpu")
    return {
        "validate_forward": bool(ok),
        "grad_mode": "no_grad",
        "validate_seconds": round(time.monotonic() - start, 1),
        **_peaks(),
    }


def _run_stage(
    args: Any, name: str, stage: Any, model: Any, frames: int, refused: Any = None
) -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = {"pixel_values_videos": _clip(frames).to("cuda")}
    facts = {"frames": frames, "job_id": args.job_id, **stage(model, inputs)}
    if refused:
        facts["refused_longer_clips"] = refused
    (args.out_dir / f"stage{name}.json").write_text(json.dumps(facts, indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument(
        "--validate-frames",
        default=None,
        help="comma-separated clip lengths to try for stage B, longest first",
    )
    parser.add_argument("--job-id", default="unknown")
    args = parser.parse_args(argv)

    import torch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    env = fingerprint(args.job_id)
    (args.out_dir / "fingerprint.json").write_text(json.dumps(env, indent=1))
    if not env["cuda_available"]:
        print("no CUDA device: this leg runs on a GPU venue only", file=sys.stderr)
        return 3
    torch.manual_seed(SEED)
    model = _load("cuda")
    tries = (
        [int(n) for n in args.validate_frames.split(",")] if args.validate_frames else [args.frames]
    )
    receipt: dict[str, Any] = {
        "fingerprint": env,
        "frames": args.frames,
        "crop": CROP,
        "seed": SEED,
    }
    if not (args.out_dir / "stageA.json").exists():
        _run_stage(args, "A", stage_a, model, args.frames)
    if not (args.out_dir / "stageB.json").exists():
        from torchlens._save_budget import SaveBudgetExceededError

        refused: list[dict[str, Any]] = []
        for frames in tries:
            # Only a memory refusal moves to a shorter clip; a validation failure
            # or any other error propagates and fails the leg.
            try:
                _run_stage(args, "B", stage_b, model, frames, refused=refused)
                break
            except (SaveBudgetExceededError, torch.cuda.OutOfMemoryError) as exc:
                refused.append(
                    {"frames": frames, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
                )
        (args.out_dir / "stageB_refused.json").write_text(json.dumps(refused, indent=1))
    for name in ("A", "B"):
        path = args.out_dir / f"stage{name}.json"
        if not path.exists():
            print(f"stage {name} did not complete; see stage{name}_refused.json", file=sys.stderr)
            return 1
        receipt[f"stage{name}"] = json.loads(path.read_text())
    stage_a_ok = (
        receipt["stageA"]["status"] == "COMPLETE" and receipt["stageA"]["traced_equals_direct"]
    )
    stage_b_ok = receipt["stageB"]["validate_forward"]
    full_clip = receipt["stageB"].get("frames", args.frames) == args.frames
    receipt["passed"] = bool(stage_a_ok and stage_b_ok and full_clip)
    receipt["split_validation"] = None if full_clip else receipt["stageB"].get("frames")
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({"passed": receipt["passed"], "job_id": args.job_id}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
