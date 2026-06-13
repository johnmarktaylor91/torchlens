"""Baseline Op object memory and construction timing for slotting work."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchlens as tl  # noqa: E402
from torchlens.constants import LAYER_PASS_LOG_FIELD_ORDER  # noqa: E402
from torchlens.data_classes.op import Op  # noqa: E402
from torchlens.utils.display import _timed_phase  # noqa: E402


@dataclass(frozen=True)
class BenchmarkFixture:
    """Model/input builder for one Op slotting baseline fixture.

    Attributes
    ----------
    name:
        Display name.
    build:
        Callable returning a model and input.
    """

    name: str
    build: Callable[[], tuple[nn.Module, Any]]


@dataclass(frozen=True)
class BaselineRow:
    """Measured Op slotting baseline row.

    Attributes
    ----------
    fixture:
        Fixture display name.
    op_count:
        Number of layer-pass entries in the trace.
    total_object_bytes:
        Sum of ``sys.getsizeof(op) + sys.getsizeof(op.__dict__)``.
    bytes_per_op:
        Mean shallow Op object bytes.
    construction_total_ms:
        Total ``object_construction:op`` timing in milliseconds.
    construction_us_per_op:
        Mean construction timing per constructed op in microseconds.
    construction_count:
        Number of timed construction samples.
    """

    fixture: str
    op_count: int
    total_object_bytes: int
    bytes_per_op: float
    construction_total_ms: float
    construction_us_per_op: float
    construction_count: int


class ParityMLP(nn.Module):
    """Small deterministic MLP matching the backend-parity fixture shape."""

    def __init__(self) -> None:
        """Initialize fixed-shape layers."""

        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear-ReLU-linear forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        hidden = torch.relu(self.fc1(x))
        return self.fc2(hidden)


class SmallResNetish(nn.Module):
    """Small residual convolutional fixture without torchvision dependency."""

    def __init__(self) -> None:
        """Initialize a compact residual stack."""

        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
        )
        self.skip2 = nn.Conv2d(8, 16, kernel_size=1, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(16, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a compact residual image model.

        Parameters
        ----------
        x:
            Image batch with shape ``(batch, 3, 16, 16)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 5)``.
        """

        y = torch.relu(self.stem(x))
        y = torch.relu(self.block1(y) + y)
        y = torch.relu(self.block2(y) + self.skip2(y))
        y = self.pool(y).flatten(1)
        return self.head(y)


def _seed_everything() -> None:
    """Set deterministic seeds for benchmark construction."""

    random.seed(1701)
    torch.manual_seed(1701)
    torch.set_num_threads(1)


def _build_parity_mlp() -> tuple[nn.Module, torch.Tensor]:
    """Build the parity MLP fixture.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Model and deterministic input.
    """

    _seed_everything()
    return ParityMLP().eval(), torch.randn(2, 3)


def _build_small_resnetish() -> tuple[nn.Module, torch.Tensor]:
    """Build the small ResNet-ish fixture.

    Returns
    -------
    tuple[nn.Module, torch.Tensor]
        Model and deterministic input.
    """

    _seed_everything()
    return SmallResNetish().eval(), torch.randn(2, 3, 16, 16)


def _op_object_bytes(trace: Any) -> int:
    """Return shallow Op object plus ``__dict__`` bytes for a trace.

    Parameters
    ----------
    trace:
        TorchLens trace with ``layer_list``.

    Returns
    -------
    int
        Sum of shallow object and dict sizes.
    """

    return sum(sys.getsizeof(op) + sys.getsizeof(op.__dict__) for op in trace.layer_list)


def _construction_stats(trace: Any) -> tuple[float, int]:
    """Return object-construction timing stats from a profiled trace.

    Parameters
    ----------
    trace:
        TorchLens trace with ``_phase_timings``.

    Returns
    -------
    tuple[float, int]
        Total seconds and sample count for ``object_construction:op``.
    """

    stats = getattr(trace, "_phase_timings", {}).get("object_construction:op", {})
    return float(stats.get("total_s", 0.0)), int(stats.get("count", 0))


def measure_fixture(fixture: BenchmarkFixture) -> BaselineRow:
    """Measure one fixture.

    Parameters
    ----------
    fixture:
        Fixture builder.

    Returns
    -------
    BaselineRow
        Measured baseline row.
    """

    model, x = fixture.build()
    with torch.no_grad():
        trace = tl.trace(model, x, profile=True, save=lambda _ctx: False)
    op_count = len(trace.layer_list)
    total_bytes = _op_object_bytes(trace)
    construction_total_s, construction_count = _construction_stats(trace)
    return BaselineRow(
        fixture=fixture.name,
        op_count=op_count,
        total_object_bytes=total_bytes,
        bytes_per_op=total_bytes / op_count if op_count else 0.0,
        construction_total_ms=construction_total_s * 1000.0,
        construction_us_per_op=(construction_total_s * 1_000_000.0 / construction_count)
        if construction_count
        else 0.0,
        construction_count=construction_count,
    )


def _seed_fields_for_synthetic_ops() -> dict[str, Any]:
    """Return a complete Op field dict from a real traced seed op.

    Returns
    -------
    dict[str, Any]
        Complete field dictionary suitable for ``Op`` construction.
    """

    model, x = _build_parity_mlp()
    with torch.no_grad():
        trace = tl.trace(model, x, profile=True, save=lambda _ctx: False)
    seed_op = next(op for op in trace.layer_list if op.func_name not in {None, "none"})
    return {
        field_name: seed_op.__dict__.get(field_name) for field_name in LAYER_PASS_LOG_FIELD_ORDER
    }


def _synthetic_field_dict(seed_fields: dict[str, Any], index: int) -> dict[str, Any]:
    """Return a unique field dict for one synthetic Op.

    Parameters
    ----------
    seed_fields:
        Complete seed fields from a real traced Op.
    index:
        Synthetic op index.

    Returns
    -------
    dict[str, Any]
        Complete field dictionary for ``Op`` construction.
    """

    fields = dict(seed_fields)
    fields["_label_raw"] = f"synthetic_add_{index}_raw"
    fields["_layer_label_raw"] = f"synthetic_add_{index}_raw"
    fields["label"] = f"synthetic_add_{index}"
    fields["layer_label"] = f"synthetic_add_{index}"
    fields["raw_index"] = index
    return fields


def measure_synthetic_op_trace(num_ops: int) -> BaselineRow:
    """Measure a synthetic trace containing many freshly constructed Ops.

    Parameters
    ----------
    num_ops:
        Number of Op instances to construct.

    Returns
    -------
    BaselineRow
        Measured synthetic baseline row.
    """

    seed_fields = _seed_fields_for_synthetic_ops()
    model, x = _build_parity_mlp()
    with torch.no_grad():
        trace = tl.trace(model, x, profile=True, save=lambda _ctx: False)
    trace._phase_timings = {}
    synthetic_ops = []
    for index in range(num_ops):
        fields = _synthetic_field_dict(seed_fields, index)
        with _timed_phase(trace, "object_construction:op"):
            synthetic_ops.append(Op(fields))
    trace.layer_list = synthetic_ops
    total_bytes = _op_object_bytes(trace)
    construction_total_s, construction_count = _construction_stats(trace)
    return BaselineRow(
        fixture=f"synthetic_{num_ops:,}_ops",
        op_count=len(trace.layer_list),
        total_object_bytes=total_bytes,
        bytes_per_op=total_bytes / len(trace.layer_list) if trace.layer_list else 0.0,
        construction_total_ms=construction_total_s * 1000.0,
        construction_us_per_op=(construction_total_s * 1_000_000.0 / construction_count)
        if construction_count
        else 0.0,
        construction_count=construction_count,
    )


def render_table(rows: list[BaselineRow]) -> str:
    """Render measured rows as a Markdown table.

    Parameters
    ----------
    rows:
        Baseline rows.

    Returns
    -------
    str
        Markdown table.
    """

    lines = [
        "| Fixture | Ops | Total Op+dict bytes | Bytes/op | object_construction:op ms | us/op |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.fixture,
                    f"{row.op_count:,}",
                    f"{row.total_object_bytes:,}",
                    f"{row.bytes_per_op:.1f}",
                    f"{row.construction_total_ms:.3f}",
                    f"{row.construction_us_per_op:.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_fixtures() -> list[BenchmarkFixture]:
    """Build the ordered fixture list.

    Returns
    -------
    list[BenchmarkFixture]
        Ordered benchmark fixtures.
    """

    return [
        BenchmarkFixture("parity_mlp", _build_parity_mlp),
        BenchmarkFixture("small_resnetish", _build_small_resnetish),
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-ops",
        type=int,
        default=100_000,
        help="Number of torch.add calls in the synthetic fixture.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Op slotting baseline and print a Markdown table."""

    args = parse_args()
    rows = [measure_fixture(fixture) for fixture in build_fixtures()]
    rows.append(measure_synthetic_op_trace(args.synthetic_ops))
    print(render_table(rows))


if __name__ == "__main__":
    main()
