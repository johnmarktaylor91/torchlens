"""Measure retained TorchLens trace memory before and after Op slotting."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import random
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelCase:
    """Benchmark fixture definition.

    Attributes
    ----------
    name:
        Stable model name for result matching.
    build:
        Callable returning an initialized model and example input.
    """

    name: str
    build: Callable[[], tuple[nn.Module, torch.Tensor]]


@dataclass(frozen=True)
class Measurement:
    """Measurement for one model fixture.

    Attributes
    ----------
    model:
        Stable model name.
    n_ops:
        Number of traced operation records in ``trace.layer_list``.
    total_bytes:
        Retained bytes for the full TorchLens trace object.
    bytes_per_op:
        ``total_bytes / n_ops``.
    error:
        Error text when measurement failed.
    """

    model: str
    n_ops: int
    total_bytes: int
    bytes_per_op: float
    error: str | None = None


class SmallMLP(nn.Module):
    """Small fully connected network fixture."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Input tensor with shape ``(batch, 16)``.

        Returns
        -------
        torch.Tensor
            Output tensor with shape ``(batch, 8)``.
        """

        return self.net(x)


class SmallCNN(nn.Module):
    """Small convolutional network fixture."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN.

        Parameters
        ----------
        x:
            Image batch with shape ``(batch, 3, 16, 16)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 10)``.
        """

        features = self.features(x)
        return self.head(features.flatten(1))


class SmallTransformerBlock(nn.Module):
    """Small transformer block fixture."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(16)
        self.ffn = nn.Sequential(
            nn.Linear(16, 32),
            nn.GELU(),
            nn.Linear(32, 16),
        )
        self.norm2 = nn.LayerNorm(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block.

        Parameters
        ----------
        x:
            Sequence tensor with shape ``(batch, sequence, channels)``.

        Returns
        -------
        torch.Tensor
            Block output with the same shape as ``x``.
        """

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


def _seed_everything() -> None:
    """Set deterministic seeds for model construction and inputs."""

    random.seed(1701)
    torch.manual_seed(1701)
    torch.set_num_threads(1)


def _build_mlp() -> tuple[nn.Module, torch.Tensor]:
    """Build the MLP fixture.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Model and input tensor.
    """

    _seed_everything()
    return SmallMLP().eval(), torch.randn(4, 16)


def _build_cnn() -> tuple[nn.Module, torch.Tensor]:
    """Build the CNN fixture.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Model and input tensor.
    """

    _seed_everything()
    return SmallCNN().eval(), torch.randn(2, 3, 16, 16)


def _build_transformer() -> tuple[nn.Module, torch.Tensor]:
    """Build the transformer fixture.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Model and input tensor.
    """

    _seed_everything()
    return SmallTransformerBlock().eval(), torch.randn(2, 6, 16)


def _fixtures() -> list[ModelCase]:
    """Return benchmark fixtures.

    Returns
    -------
    list[ModelCase]
        Ordered benchmark fixtures.
    """

    return [
        ModelCase("small_mlp", _build_mlp),
        ModelCase("small_cnn", _build_cnn),
        ModelCase("small_transformer_block", _build_transformer),
    ]


def _install_repo_import(repo_root: Path) -> None:
    """Make ``repo_root`` the first import location for ``torchlens``.

    Parameters
    ----------
    repo_root:
        Checkout root to measure.
    """

    repo_root = repo_root.resolve()
    sys.path = [path for path in sys.path if Path(path or ".").resolve() != repo_root]
    sys.path.insert(0, str(repo_root))


def _trace_total_bytes(trace: Any) -> tuple[int, str]:
    """Measure retained memory for a trace.

    Parameters
    ----------
    trace:
        TorchLens trace object.

    Returns
    -------
    tuple[int, str]
        Total bytes and sizing method name.
    """

    try:
        from pympler import asizeof
    except ImportError:
        return _fallback_total_bytes(trace), "fallback_sys_getsizeof_walk"
    return int(asizeof.asizeof(trace)), "pympler.asizeof"


def _fallback_total_bytes(root: Any) -> int:
    """Fallback retained-size walk for common Python containers and slots.

    Parameters
    ----------
    root:
        Root object to size.

    Returns
    -------
    int
        Best-effort retained bytes.
    """

    seen: set[int] = set()

    def walk(obj: Any) -> int:
        """Walk one object and its reachable Python attributes.

        Parameters
        ----------
        obj:
            Object to size.

        Returns
        -------
        int
            Best-effort retained bytes for ``obj``.
        """

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        total = sys.getsizeof(obj, 0)
        if isinstance(obj, dict):
            for key, value in obj.items():
                total += walk(key) + walk(value)
            return total
        if isinstance(obj, (list, tuple, set, frozenset)):
            return total + sum(walk(item) for item in obj)
        instance_dict = getattr(obj, "__dict__", None)
        if instance_dict is not None:
            total += walk(instance_dict)
        for cls in type(obj).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"} or not hasattr(obj, slot):
                    continue
                total += walk(getattr(obj, slot))
        return total

    return walk(root)


def _measure_case(case: ModelCase, tl_module: Any) -> Measurement:
    """Measure one model fixture.

    Parameters
    ----------
    case:
        Fixture to measure.
    tl_module:
        Imported ``torchlens`` module from the target checkout.

    Returns
    -------
    Measurement
        Measurement row, or an error row if tracing failed.
    """

    try:
        model, example_input = case.build()
        with torch.no_grad():
            trace = tl_module.trace(model, example_input)
        gc.collect()
        total_bytes, _ = _trace_total_bytes(trace)
        n_ops = len(trace.layer_list)
        return Measurement(
            model=case.name,
            n_ops=n_ops,
            total_bytes=total_bytes,
            bytes_per_op=total_bytes / n_ops if n_ops else 0.0,
        )
    except Exception as exc:  # noqa: BLE001
        return Measurement(
            model=case.name,
            n_ops=0,
            total_bytes=0,
            bytes_per_op=0.0,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_measurement(repo_root: Path, label: str) -> dict[str, Any]:
    """Run measurements against one checkout.

    Parameters
    ----------
    repo_root:
        Checkout root to import.
    label:
        Human-readable state label.

    Returns
    -------
    dict[str, Any]
        JSON-serializable measurement payload.
    """

    _install_repo_import(repo_root)
    import torchlens as tl

    method = "unknown"
    rows = []
    for case in _fixtures():
        row = _measure_case(case, tl)
        if row.error is None:
            method = _trace_total_bytes(object())[1]
        rows.append(row)
    return {
        "label": label,
        "repo_root": str(repo_root.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchlens": getattr(tl, "__version__", "unknown"),
        "sizing_method": method,
        "rows": [asdict(row) for row in rows],
    }


def _format_int(value: int) -> str:
    """Format an integer with thousands separators.

    Parameters
    ----------
    value:
        Integer value.

    Returns
    -------
    str
        Formatted integer.
    """

    return f"{value:,}"


def _format_float(value: float) -> str:
    """Format a floating-point value for Markdown.

    Parameters
    ----------
    value:
        Floating-point value.

    Returns
    -------
    str
        Formatted value.
    """

    return f"{value:,.1f}"


def render_markdown(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Render a before/after Markdown report.

    Parameters
    ----------
    before:
        Pre-slots measurement payload.
    after:
        Slotted measurement payload.

    Returns
    -------
    str
        Markdown report.
    """

    before_rows = {row["model"]: row for row in before["rows"]}
    after_rows = {row["model"]: row for row in after["rows"]}
    lines = [
        "# Op __slots__ retained-memory baseline",
        "",
        "| Model | n_ops | Before B/op | After B/op | Reduction % | Before total | After total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fixture_order = [case.name for case in _fixtures()]
    extra_models = sorted((before_rows.keys() | after_rows.keys()) - set(fixture_order))
    for model in fixture_order + extra_models:
        before_row = before_rows.get(model)
        after_row = after_rows.get(model)
        if before_row is None or after_row is None or before_row["error"] or after_row["error"]:
            lines.append(f"| {model} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        reduction = (
            (before_row["bytes_per_op"] - after_row["bytes_per_op"])
            / before_row["bytes_per_op"]
            * 100.0
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _format_int(int(after_row["n_ops"])),
                    _format_float(float(before_row["bytes_per_op"])),
                    _format_float(float(after_row["bytes_per_op"])),
                    _format_float(reduction),
                    _format_int(int(before_row["total_bytes"])),
                    _format_int(int(after_row["total_bytes"])),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            (
                "Each row traces an inline deterministic PyTorch fixture with "
                "`tl.trace(model, input)` and measures retained "
                "memory for the resulting full TorchLens trace object. The before state is "
                "the parent of the Op `__slots__` commit; the after state is the current "
                "checkout. Byte counts use the same script and environment for both states."
            ),
            "",
            "## Environment",
            "",
            f"- Before checkout: `{before['repo_root']}`",
            f"- After checkout: `{after['repo_root']}`",
            f"- Python: `{after['python']}`",
            f"- Torch: `{after['torch']}`",
            f"- TorchLens before: `{before['torchlens']}`",
            f"- TorchLens after: `{after['torchlens']}`",
            f"- Sizing method: `{after['sizing_method']}`",
            f"- Platform: `{after['platform']}`",
        ]
    )
    failures = [
        (payload["label"], row["model"], row["error"])
        for payload in (before, after)
        for row in payload["rows"]
        if row["error"]
    ]
    if failures:
        lines.extend(["", "## Measurement failures", ""])
        for label, model, error in failures:
            lines.append(f"- `{label}` `{model}`: {error}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--label", default="after")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--before-json", type=Path)
    parser.add_argument("--after-json", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run measurement or render a paired Markdown report."""

    args = parse_args()
    if args.before_json and args.after_json:
        before = json.loads(args.before_json.read_text())
        after = json.loads(args.after_json.read_text())
        markdown = render_markdown(before, after)
        if args.markdown_out:
            args.markdown_out.write_text(markdown)
        else:
            print(markdown, end="")
        return

    payload = run_measurement(args.repo_root, args.label)
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
