"""Sibling-order rank groups never span Graphviz clusters, and a failed ordered layout falls back.

Graphviz 16 fails ("trouble in init_rank") or segfaults on a ``rank=same`` set whose members sit
in a different cluster than the set itself; Graphviz 2.43 only warns and pulls the node out of its
cluster. The model tests assert on the emitted DOT, so they need no particular Graphviz version.
"""

from __future__ import annotations

import re
import subprocess
import warnings
from collections import defaultdict
from pathlib import Path

import example_models
import pytest
import torch

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.visualization import _render_ordering
from torchlens.visualization._render_common import SiblingOrderChain

_SUBGRAPH_OPEN = re.compile(r'^\s*subgraph\s+"?([^"{]+?)"?\s*\{\s*$')
_NAME = r'(?:"(?:[^"\\]|\\.)*"|[A-Za-z0-9_.:]+)'
_EDGE = re.compile(rf"^\s*({_NAME})\s*->\s*({_NAME})")
_NODE = re.compile(rf"^\s*({_NAME})\s*(?:\[|$)")


def _cluster_layout(dot_source: str) -> tuple[dict[str, set[tuple[str, ...]]], list[tuple]]:
    """Return node cluster paths and each sibling-order group's cluster and members.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source as emitted by TorchLens.

    Returns
    -------
    tuple
        ``(node_paths, groups)``: the cluster paths at which each node is named outside the
        sibling-order groups, and ``(enclosing_cluster_path, members)`` per group.
    """

    stack: list[str] = []
    node_paths: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    groups: list[tuple] = []
    group_members: list[str] | None = None
    for line in dot_source.splitlines():
        clusters = tuple(name for name in stack if name.startswith("cluster"))
        if "tl:sibling-order:start" in line:
            group_members = []
            group_path = clusters
            continue
        if "tl:sibling-order:end" in line:
            groups.append((group_path, tuple(group_members or ())))
            group_members = None
            continue
        if group_members is not None:
            node = _NODE.match(line)
            if node and "=" not in line:
                group_members.append(node.group(1).strip('"'))
            continue
        subgraph = _SUBGRAPH_OPEN.match(line)
        if subgraph:
            stack.append(subgraph.group(1))
        elif line.strip() == "{":
            stack.append("")
        elif line.strip() == "}":
            if stack:
                stack.pop()
        elif "=" not in line.split("[", 1)[0]:
            match = _EDGE.match(line) or _NODE.match(line)
            if match:
                for name in match.groups():
                    if name not in {"graph", "node", "edge"}:
                        node_paths[name.strip('"')].add(clusters)
    return node_paths, groups


def _spanning_groups(dot_source: str) -> list[tuple]:
    """Return the sibling-order groups emitted outside their members' own cluster."""

    node_paths, groups = _cluster_layout(dot_source)
    bad = []
    for group_path, members in groups:
        member_paths = {max(node_paths[member], key=len) for member in members}
        if member_paths != {group_path}:
            bad.append((group_path, members, sorted(member_paths)))
    return bad


def _draw(model: torch.nn.Module, inputs: object, tmp_path: Path, name: str, **kwargs) -> str:
    """Trace ``model`` and return the final DOT source of an unrolled draw."""

    log = tl.trace(model, inputs)
    try:
        return str(
            log.draw(
                vis_outpath=str(tmp_path / name),
                vis_save_only=True,
                vis_fileformat="pdf",
                **kwargs,
            )
        )
    finally:
        log.cleanup()


_FAILING_MODELS = {
    "transformer_encoder": (example_models.TransformerEncoderModel, lambda: torch.rand(5, 2, 16)),
    "simple_normalizing_flow": (example_models.SimpleNormalizingFlow, lambda: torch.rand(2, 8)),
    "dueling_dqn": (example_models.DuelingDQN, lambda: torch.rand(4, 8)),
    "stop_grad": (example_models.StopGradientModel, lambda: torch.rand(4, 16)),
}


@pytest.mark.parametrize("name", sorted(_FAILING_MODELS))
def test_sibling_rank_groups_stay_inside_member_cluster(name: str, tmp_path: Path) -> None:
    """The four models that broke Graphviz 16 emit no cross-cluster rank group."""

    model_cls, make_input = _FAILING_MODELS[name]
    torch.manual_seed(0)
    dot_source = _draw(model_cls(), make_input(), tmp_path, name)

    assert _spanning_groups(dot_source) == []


def test_spanning_checker_flags_the_dueling_dqn_shape() -> None:
    """The test-side checker flags the cross-cluster group seen in the Graphviz 16 repro."""

    dot_source = _DUELING_SKELETON.replace("{INJECT}", _GROUP_A_B)

    assert _spanning_groups(dot_source) == [((), ("a", "b"), [("cluster_l",), ("cluster_r",)])]


def _chain(targets: tuple[str, ...], lca_key: str | int = -1) -> SiblingOrderChain:
    """Build a sibling chain over ``targets`` emitted at ``lca_key``."""

    return SiblingOrderChain(
        source_label="src",
        source_name="src",
        targets=targets,
        target_labels=targets,
        lca_key=lca_key,
    )


_DUELING_SKELETON = """digraph M {
\tgraph [label=<x>]
\tnode [ordering=out]
\tsrc [label=<src>]
\tsrc -> a [style=solid]
\tsrc -> b [style=solid]
\ttop1 [label=<t>]
\ttop2 [label=<t>]
\tsubgraph cluster_l {
\t\tfillcolor=white label=<l>
\t\ta -> a2 [style=solid]
\t\tc -> d [style=solid]
\t}
\tsubgraph cluster_l {
\t\tsubgraph "cluster_l.inner_pass1" {
\t\t\tfillcolor=white label=<i>
\t\t\td -> e [style=solid]
\t\t}
\t}
\tsubgraph cluster_r {
\t\tfillcolor=white label=<r>
\t\tb -> b2 [style=solid]
\t}
{INJECT}}
"""
_GROUP_A_B = """\t// tl:sibling-order:start
\t{
\t\trank=same
\t\ta
\t\tb
\t\ta -> b [comment="tl:sibling-order" style=invis weight=100]
\t}
\t// tl:sibling-order:end
"""


def test_member_cluster_filter_keeps_only_same_cluster_groups() -> None:
    """Groups survive only when emitted into the innermost cluster of every member."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "")
    cross = _chain(("a", "b"))
    same_cluster = _chain(("a2", "c"), "l")
    top_level_of_clustered = _chain(("a2", "c"))
    nested_member = _chain(("c", "e"), "l")
    nested_cluster = _chain(("d", "e"), "l.inner:1")
    top_level = _chain(("top1", "top2"))
    chains = (cross, same_cluster, top_level_of_clustered, nested_member, nested_cluster, top_level)

    kept = _render_ordering._filter_sibling_chains_to_member_cluster(chains, baseline)

    # ``d`` is named in cluster_l and in the nested cluster, so it lives in the nested one.
    assert kept == (same_cluster, nested_cluster, top_level)


def test_member_cluster_filter_drops_everything_on_unbalanced_dot() -> None:
    """An unparseable baseline conservatively disables sibling ordering (invariant 11)."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "").rstrip().rstrip("}")

    assert (
        _render_ordering._filter_sibling_chains_to_member_cluster(
            (_chain(("top1", "top2")),), baseline
        )
        == ()
    )


def test_member_cluster_filter_marks_nodes_in_unrelated_clusters_ambiguous() -> None:
    """A node named from two unrelated clusters matches no group."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "").replace("b -> b2", "a -> b2")

    assert (
        _render_ordering._filter_sibling_chains_to_member_cluster(
            (_chain(("a", "a2"), "l"),), baseline
        )
        == ()
    )


class _Fanout(torch.nn.Module):
    """Top-level fanout whose two branches are orderable siblings."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fanout."""

        source = (x + 1).relu()
        return (source + 1).sigmoid() * (source + 2).tanh()


def _fail_ordered_layout(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> list[str]:
    """Make every ``dot -Tplain`` layout of a source with rank groups raise ``exc``."""

    real_layout = _render_ordering._layout_dot_plain
    calls: list[str] = []

    def layout(source: str, rankdir: str, captured_edges: list) -> object:
        calls.append(source)
        if "tl:sibling-order" in source:
            raise exc
        return real_layout(source, rankdir, captured_edges)

    monkeypatch.setattr(_render_ordering, "_layout_dot_plain", layout)
    monkeypatch.setattr(_render_ordering, "_SIBLING_ORDER_WARNING_EMITTED", False)
    return calls


@pytest.mark.parametrize(
    "exc",
    [
        subprocess.CalledProcessError(1, ["dot"], stderr="Error: trouble in init_rank"),
        subprocess.CalledProcessError(-11, ["dot"]),
    ],
    ids=["init_rank", "segfault"],
)
def test_failed_ordered_layout_falls_back_to_plain_layout(
    exc: BaseException, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing ordered layout renders the plain layout with a coded warning."""

    plain = _draw(_Fanout(), torch.randn(1, 3), tmp_path, "plain", order_siblings=False)
    ordered = _draw(_Fanout(), torch.randn(1, 3), tmp_path, "ordered")
    assert "tl:sibling-order:start" in ordered
    monkeypatch.setenv("TORCHLENS_COLLAPSE_STRICT", "0")
    calls = _fail_ordered_layout(monkeypatch, exc)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fallback = _draw(_Fanout(), torch.randn(1, 3), tmp_path, "fallback")

    assert any("tl:sibling-order" in source for source in calls)
    assert fallback == plain
    assert (tmp_path / "fallback.pdf").stat().st_size > 0
    codes = [w.message.fields["code"] for w in caught if isinstance(w.message, TorchLensWarning)]
    assert codes == ["sibling_order_fallback"]


def test_empty_ordered_layout_falls_back_to_plain_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-exit ordered layout with no nodes is a failure, not a node-population crash."""

    plain = _draw(_Fanout(), torch.randn(1, 3), tmp_path, "plain", order_siblings=False)
    real_layout = _render_ordering._layout_dot_plain
    monkeypatch.setenv("TORCHLENS_COLLAPSE_STRICT", "0")
    monkeypatch.setattr(_render_ordering, "_SIBLING_ORDER_WARNING_EMITTED", False)

    def layout(source: str, rankdir: str, captured_edges: list) -> object:
        if "tl:sibling-order" in source:
            return _render_ordering.PlainLayout(nodes={}, edge_spans={})
        return real_layout(source, rankdir, captured_edges)

    monkeypatch.setattr(_render_ordering, "_layout_dot_plain", layout)
    with pytest.warns(TorchLensWarning, match="produced no layout"):
        fallback = _draw(_Fanout(), torch.randn(1, 3), tmp_path, "fallback")

    assert fallback == plain


def test_failed_ordered_layout_still_raises_under_strict_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suite's strict tripwire keeps surfacing ordered-layout failures."""

    monkeypatch.setenv("TORCHLENS_COLLAPSE_STRICT", "1")
    _fail_ordered_layout(monkeypatch, subprocess.CalledProcessError(1, ["dot"]))

    with pytest.raises(subprocess.CalledProcessError):
        _draw(_Fanout(), torch.randn(1, 3), tmp_path, "strict")
