"""Regression tests for r18rf -- recurrent ``draw_combined`` render-flow crash.

Root cause: ``_render_flow._get_max_call_depth`` crawled the module hierarchy by
indexing ``module_edge_dict[module]`` directly. For a recurrent (multi-pass)
model the combined-graph edge payloads only carry the FIRST-pass call key
(``fc:1``) while ``top_modules``/``module_submodule_dict`` still expose every
per-pass key (``fc:1``, ``fc:2``, ``fc:3``). Popping a phantom per-pass key
therefore raised ``KeyError: 'fc:3'`` and aborted the whole render. r18j's
correspondence-edge fix unmasked this pre-existing deeper bug.

Fix: treat any module key with no recorded edge payload as edge-empty (a
``.get`` guard) so the depth crawl stays robust. Keys that DO carry a payload
resolve byte-identically to the historical direct lookup, so feedforward
rendering (and the locked render-identity oracle) is unchanged.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.visualization._render_flow import _get_max_call_depth

os.environ.setdefault("MPLBACKEND", "Agg")


class RecurrentLinear(nn.Module):
    """A 3-pass recurrent model: one Linear reused three times with tanh."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(self.fc(x))
        return x


class RecurrentReLU(nn.Module):
    """A 3-pass recurrent model reusing a nested module."""

    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.relu(x) + 0.1
        return x


class FeedForward(nn.Module):
    """A plain feedforward model (no recurrence)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def _combined_trace(model: nn.Module) -> tl.Trace:
    log = tl.trace(model, torch.randn(2, 8, requires_grad=True))
    loss = log[log.output_layers[0]].out.sum()
    log.log_backward(loss)
    return log


def _assert_combined_render(dot: object, svg_path: str) -> None:
    """Assert ``draw_combined`` produced real DOT source and a real SVG artifact.

    A no-op renderer would return ``""`` (still ``is not None``) and write no
    file; the original ``assert dot is not None`` did not catch that. This does.
    """

    assert isinstance(dot, str) and "digraph" in dot, "expected non-empty DOT source"
    path = f"{svg_path}.svg"
    assert os.path.exists(path), f"expected SVG artifact at {path}"
    with open(path, encoding="utf-8") as fh:
        svg_text = fh.read()
    assert "<svg" in svg_text, f"expected {path} to contain SVG markup"


# --------------------------------------------------------------------------- #
# Direct unit coverage of the robustness fix
# --------------------------------------------------------------------------- #


def test_get_max_call_depth_tolerates_phantom_per_pass_keys() -> None:
    """Per-pass keys missing from the edge payloads must not raise KeyError.

    This mirrors the exact recurrent combined-graph shapes captured live:
    ``top_modules`` and the child map reference ``fc:1/2/3`` but only ``fc:1``
    carries an edge payload.
    """

    top_modules = ["fc:1", "fc:2", "fc:3"]
    module_edge_dict = {"fc:1": {"edges": [("a", "b")], "nodes": ()}}
    module_submodule_dict = {"fc:1": [], "fc:2": [], "fc:3": []}

    depth = _get_max_call_depth(top_modules, module_edge_dict, module_submodule_dict)

    # fc:1 carries an edge (depth 1); the phantom passes contribute nothing.
    assert depth == 1


def test_get_max_call_depth_present_keys_match_direct_lookup() -> None:
    """When every key has a payload the result equals the historical crawl.

    Guards behavior-preservation for the feedforward path that the render
    identity oracle depends on.
    """

    top_modules = ["outer:1"]
    module_edge_dict = {
        "outer:1": {"edges": [], "nodes": ()},
        "inner:1": {"edges": [("x", "y")], "nodes": ()},
    }
    module_submodule_dict = {"outer:1": ["inner:1"], "inner:1": []}

    # outer (depth 1) has no edges but a child; inner (depth 2) has an edge.
    assert _get_max_call_depth(top_modules, module_edge_dict, module_submodule_dict) == 2


def test_get_max_call_depth_missing_submodule_key_no_crash() -> None:
    """A submodule referenced by a parent but absent from the child map is safe."""

    top_modules = ["root:1"]
    module_edge_dict = {"root:1": {"edges": [("a", "b")], "nodes": ()}}
    module_submodule_dict = {"root:1": ["orphan:2"]}  # orphan:2 has no own entry

    assert _get_max_call_depth(top_modules, module_edge_dict, module_submodule_dict) == 1


# --------------------------------------------------------------------------- #
# End-to-end render regression (the reported bug)
# --------------------------------------------------------------------------- #


def test_recurrent_draw_combined_renders(tmp_path) -> None:
    """The reported crash: recurrent ``draw_combined`` raised KeyError: 'fc:3'."""

    log = _combined_trace(RecurrentLinear())
    out = str(tmp_path / "rec_combined")
    dot = log.draw_combined(
        vis_outpath=out,
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_combined_render(dot, out)


def test_recurrent_nested_module_draw_combined_renders(tmp_path) -> None:
    """Class coverage: a recurrent reuse of a nested module also renders."""

    log = _combined_trace(RecurrentReLU())
    out = str(tmp_path / "recrelu_combined")
    dot = log.draw_combined(
        vis_outpath=out,
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_combined_render(dot, out)


def test_feedforward_draw_combined_still_renders(tmp_path) -> None:
    """Behavior-preservation smoke: the non-recurrent combined path is unchanged."""

    log = _combined_trace(FeedForward())
    out = str(tmp_path / "ff_combined")
    dot = log.draw_combined(
        vis_outpath=out,
        vis_save_only=True,
        vis_fileformat="svg",
    )
    _assert_combined_render(dot, out)
