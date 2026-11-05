"""Collapse renders of models whose module tree holds never-called containers.

``nn.ModuleList`` and ``nn.ModuleDict`` are never called, so their addresses
(``encoder.layers``, ``body.stages.a``) have no module record and no cluster.
An edge between two rendered clusters that share only such a container prefix
must land in the nearest ancestor that does have a record; naming the container
itself made ``draw(collapse="max")`` raise ``KeyError`` for every model holding
an ``nn.TransformerEncoder``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import Trace

#: Every collapse setting ``Trace.draw`` accepts: the named modes plus the ladder.
COLLAPSE_SETTINGS: tuple[Any, ...] = ("none", "auto", "max", 0.0, 0.25, 0.5, 0.75, 1.0)

#: Every ``show_containers`` value ``Trace.draw`` accepts.
SHOW_CONTAINERS: tuple[Any, ...] = (False, "labels", "collapsed", "nodes")


class _TinyTransformer(nn.Module):
    """Two-layer ``nn.TransformerEncoder``: ``encoder.layers`` is a ModuleList."""

    def __init__(self) -> None:
        """Build the encoder stack."""

        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder."""

        return self.encoder(x)


class _Body(nn.Module):
    """Called module holding a ModuleDict of ModuleLists of Sequential blocks."""

    def __init__(self) -> None:
        """Build the nested containers."""

        super().__init__()
        self.stages = nn.ModuleDict(
            {
                "a": nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.ReLU()) for _ in range(3)]),
                "b": nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(3)]),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every block of every stage in order."""

        for name in ("a", "b"):
            for block in self.stages[name]:
                x = block(x)
        return x


class _NestedContainers(nn.Module):
    """Root whose only child is a called module full of container addresses."""

    def __init__(self) -> None:
        """Build the body."""

        super().__init__()
        self.body = _Body()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the body."""

        return self.body(x)


def _trace_model(name: str) -> Trace:
    """Trace one of the container models by name."""

    torch.manual_seed(0)
    if name == "transformer":
        return tl.trace(_TinyTransformer().eval(), torch.randn(1, 6, 16))
    return tl.trace(_NestedContainers().eval(), torch.randn(2, 8))


@pytest.fixture(scope="module", params=("transformer", "nested"))
def container_trace(request: pytest.FixtureRequest) -> Iterator[Trace]:
    """One trace per container model, released at module teardown."""

    with torch.no_grad():
        trace = _trace_model(request.param)
    try:
        yield trace
    finally:
        trace.cleanup()


def _assert_container_addresses_unrecorded(trace: Trace) -> None:
    """Pin the premise: the model's container prefixes have no module record."""

    recorded = {module.address for module in trace.modules}
    if "encoder" in recorded:
        assert "encoder.layers" not in recorded
        assert "encoder.layers.0" in recorded
    else:
        assert "body" in recorded
        assert "body.stages" not in recorded
        assert "body.stages.a" not in recorded
        assert "body.stages.a.0" in recorded


@pytest.mark.parametrize("rolled", (False, True))
@pytest.mark.parametrize("collapse", COLLAPSE_SETTINGS)
def test_every_collapse_setting_draws_container_models(
    container_trace: Trace, collapse: Any, rolled: bool, tmp_path: Path
) -> None:
    """Each collapse setting renders, rolled and unrolled, past container prefixes."""

    _assert_container_addresses_unrecorded(container_trace)
    drawing = container_trace.draw(
        path=str(tmp_path / "graph"), collapse=collapse, rolled=rolled, file_format="dot"
    )
    assert drawing.dot.lstrip().startswith(("digraph", "strict digraph"))


@pytest.mark.parametrize("rolled", (False, True))
@pytest.mark.parametrize("show_containers", SHOW_CONTAINERS)
def test_max_collapse_draws_with_every_container_display(
    container_trace: Trace, show_containers: Any, rolled: bool, tmp_path: Path
) -> None:
    """``collapse="max"`` renders under every container display option."""

    drawing = container_trace.draw(
        path=str(tmp_path / "graph"),
        collapse="max",
        rolled=rolled,
        show_containers=show_containers,
        file_format="dot",
    )
    assert drawing.dot.lstrip().startswith(("digraph", "strict digraph"))


def test_edge_owner_backs_off_past_container_prefixes() -> None:
    """The edge cluster is the nearest common ancestor that has a module record."""

    from torchlens._visualization._render_module_scope import (
        _lowest_common_rendered_owner_key,
    )

    with torch.no_grad():
        transformer = _trace_model("transformer")
        nested = _trace_model("nested")
    try:
        layer0, layer1_attn = "encoder.layers.0:1", "encoder.layers.1.self_attn:1"
        assert (
            _lowest_common_rendered_owner_key(transformer, layer0, layer1_attn, "unrolled")
            == "encoder:1"
        )
        assert (
            _lowest_common_rendered_owner_key(
                transformer, "encoder.layers.0", "encoder.layers.1.self_attn", "rolled"
            )
            == "encoder"
        )
        # A recorded common prefix is kept exactly as before.
        assert (
            _lowest_common_rendered_owner_key(
                transformer, "encoder.layers.0.linear1:1", "encoder.layers.0.linear2:1", "unrolled"
            )
            == "encoder.layers.0:1"
        )
        assert (
            _lowest_common_rendered_owner_key(
                nested, "body.stages.a.0:1", "body.stages.b.2:1", "unrolled"
            )
            == "body:1"
        )
        assert (
            _lowest_common_rendered_owner_key(
                nested, "body.stages.a.0", "body.stages.a.2", "rolled"
            )
            == "body"
        )
        # No recorded common ancestor: the edge is drawn at top level.
        assert _lowest_common_rendered_owner_key(nested, "body:1", None, "unrolled") == -1
    finally:
        transformer.cleanup()
        nested.cleanup()
