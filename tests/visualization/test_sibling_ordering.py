"""Regression tests for Graphviz sibling-ordering post-processing."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
import torchvision.models as tv_models

import torchlens as tl
from torchlens.visualization.rendering import _strip_sibling_rank_groups


def _plain_nodes(dot_source: str) -> dict[str, tuple[float, float]]:
    """Return node coordinates parsed from ``dot -Tplain``.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.

    Returns
    -------
    dict[str, tuple[float, float]]
        Rendered node coordinates keyed by Graphviz node name.
    """

    proc = subprocess.run(
        ["dot", "-Tplain"],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    nodes: dict[str, tuple[float, float]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] == "node":
            nodes[parts[1]] = (float(parts[2]), float(parts[3]))
    return nodes


def _sibling_chains(dot_source: str) -> list[list[str]]:
    """Return sibling chains parsed from TorchLens invisible ordering edges.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.

    Returns
    -------
    list[list[str]]
        Ordered rendered node names for each emitted sibling chain.
    """

    chains: list[list[str]] = []
    current_edges: list[tuple[str, str]] = []
    in_group = False
    for line in dot_source.splitlines():
        if "tl:sibling-order:start" in line:
            in_group = True
            current_edges = []
            continue
        if "tl:sibling-order:end" in line:
            in_group = False
            if current_edges:
                chain = [current_edges[0][0], *[right for _, right in current_edges]]
                chains.append(chain)
            continue
        if in_group and "->" in line and "tl:sibling-order" in line:
            left, rest = line.strip().split(" -> ", 1)
            right = rest.split(" ", 1)[0]
            current_edges.append((left.strip('"'), right.strip('"')))
    return chains


def _draw_source(log: tl.Trace, tmp_path: Path, name: str, **kwargs: object) -> str:
    """Draw ``log`` and return the DOT source.

    Parameters
    ----------
    log:
        Trace to render.
    tmp_path:
        Temporary directory.
    name:
        Output file stem.
    **kwargs:
        Extra draw kwargs.

    Returns
    -------
    str
        DOT source returned by ``Trace.draw``.
    """

    return str(
        log.draw(
            vis_outpath=str(tmp_path / name),
            vis_save_only=True,
            vis_fileformat="pdf",
            **kwargs,
        )
    )


class ResidualToy(torch.nn.Module):
    """Residual merge toy whose candidate fanout should be skipped."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual toy forward pass."""

        source = x + 1
        branch = source.relu()
        merged = branch + source
        return merged.sigmoid()


class CodexDistorter(torch.nn.Module):
    """Unequal-depth distorter from the sibling-ordering findings."""

    def __init__(self, depth: int = 5) -> None:
        """Initialize the side-chain depth."""

        super().__init__()
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the unequal-depth distorter forward pass."""

        parent = x + 1
        source = parent.relu()
        left = source + 1
        right = source + 2
        side = parent
        for _ in range(self.depth):
            side = side.sigmoid() + 1
        right = right + side
        return (left + 1) + (right + 1)


class TinyTransformer(torch.nn.Module):
    """Small transformer used for real-model calibration coverage."""

    def __init__(self) -> None:
        """Initialize the tiny encoder."""

        super().__init__()
        layer = torch.nn.TransformerEncoderLayer(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the tiny encoder."""

        return self.encoder(x)


@pytest.mark.heavy
def test_googlenet_sibling_chains_are_execution_ordered(tmp_path: Path) -> None:
    """GoogLeNet's nine Inception fanouts render in execution order left-to-right."""

    with torch.no_grad():
        log = tl.trace(
            tv_models.googlenet(weights=None, aux_logits=False, init_weights=False).eval(),
            torch.randn(1, 3, 224, 224),
        )
        try:
            dot_source = _draw_source(log, tmp_path, "googlenet")
            decision = log._last_sibling_ordering_decision
        finally:
            log.cleanup()

    baseline_nodes = _plain_nodes(_strip_sibling_rank_groups(dot_source))
    ordered_nodes = _plain_nodes(dot_source)
    chains = _sibling_chains(dot_source)

    assert len(chains) == 9
    assert decision.candidate_count == 9
    assert decision.survivor_count == 9
    assert len(ordered_nodes) == len(baseline_nodes)
    for chain in chains:
        assert chain == sorted(chain, key=lambda node: ordered_nodes[node][0])


def test_distorter_chain_drops_by_ratio_cap(tmp_path: Path) -> None:
    """The L=5 unequal-depth distorter drops the bad chain by stretch ratio."""

    with torch.no_grad():
        log = tl.trace(CodexDistorter(depth=5).eval(), torch.randn(1, 3))
        try:
            dot_source = _draw_source(log, tmp_path, "distorter")
            decision = log._last_sibling_ordering_decision
        finally:
            log.cleanup()

    assert decision.candidate_count == 2
    assert decision.survivor_count == 1
    assert max(decision.ratios.values()) > 4.5
    assert dot_source.count("tl:sibling-order:start") == 1


@pytest.mark.heavy
def test_mixed_googlenet_and_distorter_decisions(tmp_path: Path) -> None:
    """Mixed case keeps all GoogLeNet chains and drops the distorter chain independently."""

    with torch.no_grad():
        google = tl.trace(
            tv_models.googlenet(weights=None, aux_logits=False, init_weights=False).eval(),
            torch.randn(1, 3, 224, 224),
        )
        distorter = tl.trace(CodexDistorter(depth=5).eval(), torch.randn(1, 3))
        try:
            google_source = _draw_source(google, tmp_path, "mixed_google")
            distorter_source = _draw_source(distorter, tmp_path, "mixed_distorter")
            google_decision = google._last_sibling_ordering_decision
            distorter_decision = distorter._last_sibling_ordering_decision
        finally:
            google.cleanup()
            distorter.cleanup()

    assert google_decision.survivor_count == 9
    assert google_source.count("tl:sibling-order:start") == 9
    assert distorter_decision.candidate_count == 2
    assert distorter_decision.survivor_count == 1
    assert distorter_source.count("tl:sibling-order:start") == 1


def test_residual_toy_is_safe_noop(tmp_path: Path) -> None:
    """Residual merges are skipped by the sole-parent guard."""

    with torch.no_grad():
        log = tl.trace(ResidualToy().eval(), torch.randn(1, 3))
        try:
            ordered_source = _draw_source(log, tmp_path, "residual_ordered")
            baseline_source = _draw_source(
                log,
                tmp_path,
                "residual_baseline",
                order_siblings=False,
            )
            decision = log._last_sibling_ordering_decision
        finally:
            log.cleanup()

    assert decision.candidate_count == 0
    assert ordered_source == baseline_source


@pytest.mark.slow
def test_densenet_is_safe_noop(tmp_path: Path) -> None:
    """DenseNet fanouts are skipped and match the baseline dot layout."""

    with torch.no_grad():
        log = tl.trace(
            tv_models.densenet121(weights=None).eval(),
            torch.randn(1, 3, 224, 224),
        )
        try:
            # DenseNet's dense skips push the auto engine to rank layout; pin
            # dot so this test keeps exercising the sibling-ordering post-pass.
            ordered_source = _draw_source(
                log, tmp_path, "densenet_ordered", vis_node_placement="dot"
            )
            baseline_source = _draw_source(
                log,
                tmp_path,
                "densenet_baseline",
                order_siblings=False,
                vis_node_placement="dot",
            )
            decision = log._last_sibling_ordering_decision
        finally:
            log.cleanup()

    assert decision.candidate_count == 0
    assert ordered_source == baseline_source


def test_sibling_ordering_is_deterministic(tmp_path: Path) -> None:
    """Two renders of the same trace make identical sibling-order decisions."""

    with torch.no_grad():
        log = tl.trace(CodexDistorter(depth=5).eval(), torch.randn(1, 3))
        try:
            first_source = _draw_source(log, tmp_path, "first")
            first_decision = log._last_sibling_ordering_decision
            second_source = _draw_source(log, tmp_path, "second")
            second_decision = log._last_sibling_ordering_decision
        finally:
            log.cleanup()

    assert first_source == second_source
    assert first_decision == second_decision


def test_lr_direction_places_first_exec_topmost(tmp_path: Path) -> None:
    """LR direction orders siblings with first execution topmost."""

    with torch.no_grad():
        log = tl.trace(CodexDistorter(depth=5).eval(), torch.randn(1, 3))
        try:
            dot_source = _draw_source(log, tmp_path, "lr", direction="leftright")
        finally:
            log.cleanup()

    nodes = _plain_nodes(dot_source)
    for chain in _sibling_chains(dot_source):
        assert chain == sorted(chain, key=lambda node: nodes[node][1], reverse=True)


@pytest.mark.heavy
def test_real_model_calibration_ratios_stay_below_cap(tmp_path: Path) -> None:
    """Real-model calibration keeps legitimate chains under the selected cap."""

    cases = [
        (
            "resnet18",
            tv_models.resnet18(weights=None).eval(),
            torch.randn(1, 3, 224, 224),
        ),
        (
            "resnet50",
            tv_models.resnet50(weights=None).eval(),
            torch.randn(1, 3, 224, 224),
        ),
        ("tiny_transformer", TinyTransformer().eval(), torch.randn(2, 4, 16)),
    ]
    for name, model, inputs in cases:
        with torch.no_grad():
            log = tl.trace(model, inputs)
            try:
                _draw_source(log, tmp_path, name)
                decision = log._last_sibling_ordering_decision
            finally:
                log.cleanup()
        survivor_ratios = [decision.ratios[key] for key in decision.surviving_keys]
        assert max(survivor_ratios, default=0.0) <= 4.5
