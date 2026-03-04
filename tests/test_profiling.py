"""Profiling report: measures TorchLens overhead vs raw forward pass.

Generates tests/test_outputs/profiling_report.txt with timing data for:
  - Raw forward pass (baseline)
  - log_forward_pass (initial logging)
  - save_new_activations (fast re-logging with new input)
  - validate_forward_pass (perturbation validation)

Run:  pytest tests/test_profiling.py -v -s
"""

import os
import signal
import time
from os.path import join as opj

import torch
import torch.nn as nn

from conftest import TEST_OUTPUTS_DIR

import example_models
from torchlens import log_forward_pass, validate_forward_pass
from torchlens.validation.invariants import MetadataInvariantError


class _ValidationTimeout(Exception):
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPORT_PATH = opj(TEST_OUTPUTS_DIR, "profiling_report.txt")

# Illustrative sampling across architecture families: toy models first,
# then real-world torchvision models for realistic overhead measurement.
PROFILING_MODELS = [
    # --- Toy models (fast, cover core patterns) ---
    (
        "SimpleFF",
        lambda: example_models.SimpleFF(),
        lambda: torch.rand(5, 5),
        "Minimal functional model (no modules)",
    ),
    (
        "SimpleBranching",
        lambda: example_models.SimpleBranching(),
        lambda: torch.rand(5, 5),
        "Diverging / merging data flow",
    ),
    (
        "NestedModules",
        lambda: example_models.NestedModules(),
        lambda: torch.rand(5, 5),
        "Deep module hierarchy",
    ),
    (
        "ResidualBlock",
        lambda: example_models.ResidualBlockModel(),
        lambda: torch.rand(1, 16, 8, 8),
        "Conv-BN-ReLU residual skip connection",
    ),
    (
        "MultiheadAttention",
        lambda: example_models.MultiheadAttentionModel(),
        lambda: torch.rand(10, 2, 16),
        "nn.MultiheadAttention (seq=10, batch=2, dim=16)",
    ),
    (
        "LSTM",
        lambda: example_models.LSTMModel(),
        lambda: torch.rand(5, 5, 5),
        "LSTM + linear classifier",
    ),
    (
        "RecurrentParamsSimple",
        lambda: example_models.RecurrentParamsSimple(),
        lambda: torch.rand(5, 5),
        "Recurrent / looping parameters",
    ),
    # --- Real-world models (torchvision, larger graphs) ---
    (
        "ResNet18",
        lambda: __import__("torchvision").models.resnet18(),
        lambda: torch.rand(1, 3, 224, 224),
        "Classic residual CNN (11.7M params)",
    ),
    (
        "MobileNetV2",
        lambda: __import__("torchvision").models.mobilenet_v2(),
        lambda: torch.rand(1, 3, 224, 224),
        "Inverted residual + depthwise separable convs (3.4M params)",
    ),
    (
        "EfficientNet_B0",
        lambda: __import__("torchvision").models.efficientnet_b0(),
        lambda: torch.rand(1, 3, 224, 224),
        "Compound-scaled efficient CNN (5.3M params)",
    ),
    (
        "Swin_T",
        lambda: __import__("torchvision").models.swin_t(),
        lambda: torch.rand(1, 3, 224, 224),
        "Shifted-window vision transformer (28.3M params)",
    ),
    (
        "VGG16",
        lambda: __import__("torchvision").models.vgg16(),
        lambda: torch.rand(1, 3, 224, 224),
        "Deep sequential CNN (138M params)",
    ),
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _sync():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_fn(fn, *args, **kwargs):
    """Time a single function call. Returns (result, elapsed_seconds)."""
    _sync()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync()
    elapsed = time.perf_counter() - start
    return result, elapsed


def _time_raw_forward(model, input_tensor, num_warmup=2, num_runs=5):
    """Average raw forward-pass time with warmup."""
    with torch.no_grad():
        for _ in range(num_warmup):
            model(input_tensor)
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            _, t = _time_fn(model, input_tensor)
            times.append(t)
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Per-model profiling
# ---------------------------------------------------------------------------


def _profile_model(name, model, input_tensor, description):
    """Profile a single model. Returns a results dict."""
    raw_time = _time_raw_forward(model, input_tensor)

    log, lfp_time = _time_fn(log_forward_pass, model, input_tensor, random_seed=42)

    # save_new_activations requires the computational graph to match the
    # original log_forward_pass exactly.  For some models (e.g. AlexNet) this
    # can fail even with identical inputs due to counter-alignment issues in
    # the fast path.  Gracefully degrade to N/A when that happens.
    try:
        _, sna_time = _time_fn(
            log.save_new_activations, model, input_tensor.clone(), random_seed=42
        )
    except ValueError:
        sna_time = None

    # validate_forward_pass may fail for models that trigger known bugs
    # (e.g. Bug #79: loop detection param-sharing fragmentation) or hang
    # on large tensor comparisons.  Use a 60s signal-based timeout.
    def _alarm_handler(signum, frame):
        raise _ValidationTimeout()

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        signal.alarm(60)
        _, val_time = _time_fn(validate_forward_pass, model, input_tensor, random_seed=42)
    except (MetadataInvariantError, _ValidationTimeout, Exception):
        val_time = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    safe_raw = raw_time if raw_time > 0 else 1e-9
    return {
        "name": name,
        "description": description,
        "num_layers": log.num_operations,
        "raw_time": raw_time,
        "lfp_time": lfp_time,
        "sna_time": sna_time,
        "val_time": val_time,
        "lfp_ratio": lfp_time / safe_raw,
        "sna_ratio": sna_time / safe_raw if sna_time is not None else None,
        "val_ratio": val_time / safe_raw if val_time is not None else None,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _fmt_time(seconds):
    if seconds is None:
        return "N/A"
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}\u00b5s"
    if seconds < 1:
        return f"{seconds * 1_000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}min"


def _fmt_ratio(ratio):
    if ratio is None:
        return "N/A"
    return f"{ratio:.1f}x"


def _generate_report(results):
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append("TORCHLENS PROFILING REPORT")
    lines.append("=" * W)
    lines.append("")
    lines.append(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"Device    : {'CUDA (' + torch.cuda.get_device_name() + ')' if torch.cuda.is_available() else 'CPU'}"
    )
    lines.append(f"PyTorch   : {torch.__version__}")
    lines.append(f"Models    : {len(results)}")
    lines.append("")

    # ---- Summary table ----
    hdr = (
        f"{'Model':<25} {'Layers':>6}  "
        f"{'Raw':>9} {'log_fwd':>9} {'save_new':>9} {'validate':>9}  "
        f"{'log_fwd':>8} {'save_new':>8} {'validate':>8}"
    )
    sub = (
        f"{'':25} {'':>6}  "
        f"{'(time)':>9} {'(time)':>9} {'(time)':>9} {'(time)':>9}  "
        f"{'(ratio)':>8} {'(ratio)':>8} {'(ratio)':>8}"
    )
    lines.append(hdr)
    lines.append(sub)
    lines.append("-" * W)

    for r in results:
        row = (
            f"{r['name']:<25} {r['num_layers']:>6}  "
            f"{_fmt_time(r['raw_time']):>9} "
            f"{_fmt_time(r['lfp_time']):>9} "
            f"{_fmt_time(r['sna_time']):>9} "
            f"{_fmt_time(r['val_time']):>9}  "
            f"{_fmt_ratio(r['lfp_ratio']):>8} "
            f"{_fmt_ratio(r['sna_ratio']):>8} "
            f"{_fmt_ratio(r['val_ratio']):>8}"
        )
        lines.append(row)

    lines.append("-" * W)
    lines.append("")

    # ---- Per-model detail ----
    for r in results:
        lines.append(f"--- {r['name']} ---")
        lines.append(f"  {r['description']}")
        lines.append(f"  Layers logged: {r['num_layers']}")
        lines.append(f"  Raw forward pass           : {_fmt_time(r['raw_time']):>10}")
        lines.append(
            f"  log_forward_pass           : {_fmt_time(r['lfp_time']):>10}  "
            f"({_fmt_ratio(r['lfp_ratio'])} overhead)"
        )
        lines.append(
            f"  save_new_activations       : {_fmt_time(r['sna_time']):>10}  "
            f"({_fmt_ratio(r['sna_ratio'])} overhead)"
        )
        lines.append(
            f"  validate_forward_pass : {_fmt_time(r['val_time']):>10}  "
            f"({_fmt_ratio(r['val_ratio'])} overhead)"
        )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test entry point
# ---------------------------------------------------------------------------


def test_profiling_report():
    """Run profiling across all models and write report to test_outputs/."""
    results = []
    for name, model_factory, input_factory, description in PROFILING_MODELS:
        model = model_factory()
        model.eval()
        input_tensor = input_factory()
        result = _profile_model(name, model, input_tensor, description)
        results.append(result)

    report = _generate_report(results)

    os.makedirs(TEST_OUTPUTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    # Print to stdout when running with -s
    print("\n" + report)
