"""Time the metadata pickle seam of save and load, with deterministic counters.

Measures, per model and repeat: the portable ``trace.save`` wall time, the
``tl.load`` wall time, and the time to read the saved ``metadata.pkl`` through the
restricted unpickler (and through the pure-Python interpreter when the build has
one). Deterministic counters: metadata bytes, opcode count, the unpickler's
``find_class`` calls and guarded REDUCE calls, and which machine ran the load.

Usage::

    python benchmarks/metadata_unpickle_timing.py --repeats 3 --chain 300
"""

from __future__ import annotations

import argparse
import io
import pickletools
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._io import _safe_unpickle
from torchlens.options import CaptureOptions


def _chain(depth: int) -> nn.Module:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Linear(16, 16), nn.ReLU()))
    return nn.Sequential(*layers)


def _models(chain_depth: int) -> dict[str, tuple[nn.Module, torch.Tensor]]:
    models: dict[str, tuple[nn.Module, torch.Tensor]] = {
        f"chain_{chain_depth}": (_chain(chain_depth), torch.randn(2, 16)),
    }
    try:
        import torchvision

        models["resnet18"] = (torchvision.models.resnet18(weights=None), torch.randn(1, 3, 64, 64))
    except Exception:  # noqa: BLE001 -- torchvision is optional for this benchmark
        pass
    return models


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def _time_unpickle(factory: Any, data: bytes, reps: int) -> tuple[float, Any]:
    samples: list[float] = []
    unpickler = None
    for _ in range(reps):
        unpickler = factory(io.BytesIO(data))
        start = time.perf_counter()
        unpickler.load()
        samples.append(time.perf_counter() - start)
    return _median_ms(samples), unpickler


def _measure(name: str, model: nn.Module, example: torch.Tensor, repeats: int) -> dict[str, Any]:
    model.eval()
    trace = tl.trace(model, example, capture=CaptureOptions(cache=False))
    save_samples: list[float] = []
    load_samples: list[float] = []
    metadata = b""
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(repeats):
            path = Path(tmp) / f"{name}_{index}.tlspec"
            start = time.perf_counter()
            trace.save(path)
            save_samples.append(time.perf_counter() - start)
            start = time.perf_counter()
            tl.load(path)
            load_samples.append(time.perf_counter() - start)
            metadata = (path / "metadata.pkl").read_bytes()
    row: dict[str, Any] = {
        "model": name,
        "ops": len(trace.layers),
        "save_ms": _median_ms(save_samples),
        "load_ms": _median_ms(load_samples),
        "metadata_kib": len(metadata) / 1024.0,
        "opcodes": sum(1 for _ in pickletools.genops(metadata)),
    }
    restricted_ms, unpickler = _time_unpickle(_safe_unpickle.SafeBundleUnpickler, metadata, 5)
    row["restricted_unpickle_ms"] = restricted_ms
    row["load_path"] = getattr(unpickler, "load_path", "pure_python (pre C path)")
    row["global_lookups"] = getattr(unpickler, "global_lookups", None)
    row["guarded_calls"] = getattr(unpickler, "guarded_calls", None)
    interpreter = getattr(_safe_unpickle, "_PurePythonBundleUnpickler", None)
    if interpreter is not None:
        row["interpreter_unpickle_ms"], _ = _time_unpickle(interpreter, metadata, 5)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chain", type=int, default=300)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = [
        _measure(name, model, example, args.repeats)
        for name, (model, example) in _models(args.chain).items()
    ]
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        cells = [f"{row[k]:.1f}" if isinstance(row[k], float) else str(row[k]) for k in keys]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
