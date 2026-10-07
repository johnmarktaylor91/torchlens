"""Sibling-order rank groups sit in a cluster holding all members; failed ordered layouts fall back.

Graphviz 16 fails ("trouble in init_rank"), crashes, or warns "already in a rankset, deleted from
cluster" on a top-level ``rank=same`` set whose members sit inside clusters; a set emitted inside a
cluster may span that cluster's nested clusters. The model tests assert on the emitted DOT, so they
need no particular Graphviz version.
"""

from __future__ import annotations

import re
import subprocess
import warnings
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import example_models
import pytest
import torch

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.visualization import _render_ordering
from torchlens.visualization._render_common import SiblingOrderChain
from torchlens.visualization._render_flow import _inject_sibling_rank_groups

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
    """Return the sibling-order groups whose cluster does not hold every member.

    A top-level group may only hold top-level nodes; a group inside a cluster may hold nodes of
    that cluster and of its nested clusters.
    """

    node_paths, groups = _cluster_layout(dot_source)
    bad = []
    for group_path, members in groups:
        member_paths = sorted({max(node_paths[member], key=len) for member in members})
        if not all(
            path[: len(group_path)] == group_path and (group_path or not path)
            for path in member_paths
        ):
            bad.append((group_path, members, member_paths))
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
def test_sibling_rank_groups_stay_inside_member_cluster(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The four models that broke Graphviz 16 emit each chain once and in a holding cluster."""

    from torchlens.visualization import _render_dot

    real_verify = _render_dot._verify_and_apply_sibling_ordering
    emitted: list[tuple[str, int]] = []

    def verify(source: str, chains: tuple, *args: object) -> object:
        emitted.append((source, len(chains)))
        return real_verify(source, chains, *args)

    monkeypatch.setattr(_render_dot, "_verify_and_apply_sibling_ordering", verify)
    model_cls, make_input = _FAILING_MODELS[name]
    torch.manual_seed(0)
    dot_source = _draw(model_cls(), make_input(), tmp_path, name)

    assert _spanning_groups(dot_source) == []
    # The renderer emits each candidate chain exactly once (a cluster chain used to be
    # re-emitted at top level too, which put clustered nodes in a root rankset).
    assert [source.count("tl:sibling-order:start") for source, _ in emitted] == [
        count for _, count in emitted
    ]


def test_spanning_checker_flags_the_dueling_dqn_shape() -> None:
    """The test-side checker flags the cross-cluster group seen in the Graphviz 16 repro."""

    dot_source = _DUELING_SKELETON.replace("{INJECT}", _GROUP_A_B)

    assert _spanning_groups(dot_source) == [((), ("a", "b"), [("cluster_l",), ("cluster_r",)])]
    nested_group = (
        _GROUP_A_B.replace("\ta", "\tc").replace("\tb", "\te").replace("a -> b", "c -> e")
    )
    nested_source = _DUELING_SKELETON.replace(
        "\t\tc -> d [style=solid]\n", "\t\tc -> d [style=solid]\n" + nested_group
    ).replace("{INJECT}", "")
    assert "tl:sibling-order" in nested_source
    assert _spanning_groups(nested_source) == []


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


def _fit(chains: tuple[SiblingOrderChain, ...], baseline: str) -> tuple[SiblingOrderChain, ...]:
    """Fit ``chains`` to the clusters of ``baseline``."""

    return _render_ordering._fit_sibling_chains_to_clusters(
        chains, _render_ordering._dot_node_clusters(baseline)
    )


def test_cluster_fit_keeps_moves_or_drops_each_group() -> None:
    """Groups stay when their cluster holds every member, move to the shared cluster, or drop."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "")
    cross = _chain(("a", "b"))
    same_cluster = _chain(("a2", "c"), "l")
    top_level_of_clustered = _chain(("a2", "c"))
    nested_member = _chain(("c", "e"), "l")
    nested_cluster = _chain(("d", "e"), "l.inner:1")
    outer_member = _chain(("c", "e"), "l.inner:1")
    sibling_cluster = _chain(("a2", "b2"), "r")
    top_level = _chain(("top1", "top2"))
    top_level_of_nested = _chain(("d", "e"))
    unknown_target = _chain(("top1", "ghost"))
    chains = (
        cross,
        same_cluster,
        top_level_of_clustered,
        nested_member,
        nested_cluster,
        outer_member,
        sibling_cluster,
        top_level,
        top_level_of_nested,
        unknown_target,
    )

    fitted = _fit(chains, baseline)

    # ``d`` is named in cluster_l and in the nested cluster, so it lives in the nested one.
    assert fitted == (
        same_cluster,
        replace(top_level_of_clustered, lca_key="l"),
        nested_member,
        nested_cluster,
        replace(outer_member, lca_key="l"),
        top_level,
        replace(top_level_of_nested, lca_key="l.inner_pass1"),
    )
    injected = _inject_sibling_rank_groups(baseline, fitted)
    assert injected.count("tl:sibling-order:start") == len(fitted)
    assert _spanning_groups(injected) == []


@pytest.mark.parametrize(
    "breakage",
    ["unbalanced", "multiline_label"],
)
def test_cluster_fit_drops_everything_on_unreadable_dot(breakage: str) -> None:
    """DOT outside the one-statement-per-line shape disables sibling ordering (invariant 11)."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "")
    if breakage == "unbalanced":
        baseline = baseline.rstrip().rstrip("}")
    else:
        # A label spanning lines could hide a closing brace from the brace tracking.
        baseline = baseline.replace("label=<r>", "label=<r\n}\n{\n>")

    assert _fit((_chain(("top1", "top2")),), baseline) == ()


def test_cluster_fit_drops_nodes_named_in_unrelated_clusters() -> None:
    """A node named from two unrelated clusters matches no group."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "").replace("b -> b2", "a -> b2")

    assert _fit((_chain(("a", "a2"), "l"),), baseline) == ()


def test_group_scope_check_catches_insertion_into_a_label() -> None:
    """A group that string insertion lands outside its intended cluster disables the pass."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "").replace(
        "\tsrc [label=<src>]", '\tsrc [label="subgraph cluster_l {"]'
    )
    clusters = _render_ordering._dot_node_clusters(baseline)
    fitted = _render_ordering._fit_sibling_chains_to_clusters((_chain(("a2", "c")),), clusters)
    assert fitted == (replace(_chain(("a2", "c")), lca_key="l"),)

    injected = _inject_sibling_rank_groups(baseline, fitted)

    assert _spanning_groups(injected) != []
    assert not _render_ordering._sibling_groups_fit(injected, clusters)
    proper = _inject_sibling_rank_groups(_DUELING_SKELETON.replace("{INJECT}", ""), fitted)
    assert _render_ordering._sibling_groups_fit(proper, clusters)


def test_verify_rebuilds_when_a_chain_loses_a_rendered_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chain dropped for a missing rendered node never stays in the laid-out source."""

    baseline = _DUELING_SKELETON.replace("{INJECT}", "")
    kept, ghost = _chain(("top1", "top2")), _chain(("top1", "ghost"))
    source = _inject_sibling_rank_groups(baseline, (kept, ghost))
    names = ("src", "a", "b", "a2", "b2", "c", "d", "e", "top1", "top2")
    layout = _render_ordering.PlainLayout(nodes=dict.fromkeys(names, (0.0, 0.0)), edge_spans={})
    monkeypatch.setattr(_render_ordering, "_layout_dot_plain", lambda *args: layout)

    final, decision = _render_ordering._verify_and_apply_sibling_ordering(
        source, (kept, ghost), [], "TB"
    )

    assert "ghost" not in final
    assert final.count("tl:sibling-order:start") == 1
    assert decision.surviving_keys == (("src", ("top1", "top2")),)


class _Branches(torch.nn.Module):
    """Two parallel branches inside one parent module, like an inception block."""

    def __init__(self) -> None:
        """Build the branches."""

        super().__init__()
        self.left = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        self.right = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both branches and merge them."""

        return torch.cat([self.left(x), self.right(x)], dim=1)


class _InceptionToy(torch.nn.Module):
    """A top-level stem feeding a two-branch block."""

    def __init__(self) -> None:
        """Build the stem and block."""

        super().__init__()
        self.stem = torch.nn.Linear(4, 4)
        self.block = _Branches()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stem, then the block."""

        return self.block(self.stem(x).relu())


def test_root_level_chain_moves_into_the_branches_parent_cluster(tmp_path: Path) -> None:
    """A fanout from outside a block into its branches is ordered inside the block's cluster."""

    log = tl.trace(_InceptionToy(), torch.randn(2, 4))
    try:
        dot_source = str(
            log.draw(
                vis_outpath=str(tmp_path / "inception"), vis_save_only=True, vis_fileformat="pdf"
            )
        )
        decision = log._last_sibling_ordering_decision
    finally:
        log.cleanup()

    _, groups = _cluster_layout(dot_source)
    assert decision.survivor_count == 1
    assert [path for path, _ in groups] == [("cluster_block_pass1",)]
    assert _spanning_groups(dot_source) == []


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
    assert codes.count("sibling_order_fallback") == 1


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


class _Distorter(torch.nn.Module):
    """Two fanouts; the unequal-depth one stretches edges and is dropped by the ratio cap."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the distorter."""

        parent = x + 1
        source = parent.relu()
        left = source + 1
        right = source + 2
        side = parent
        for _ in range(5):
            side = side.sigmoid() + 1
        right = right + side
        return (left + 1) + (right + 1)


def test_empty_retry_layout_falls_back_to_plain_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty layout on a post-stretch retry also falls back instead of passing as unstretched."""

    plain = _draw(_Distorter(), torch.randn(1, 3), tmp_path, "plain", order_siblings=False)
    real_layout = _render_ordering._layout_dot_plain
    ordered_calls: list[str] = []
    monkeypatch.setenv("TORCHLENS_COLLAPSE_STRICT", "0")
    monkeypatch.setattr(_render_ordering, "_SIBLING_ORDER_WARNING_EMITTED", False)

    def layout(source: str, rankdir: str, captured_edges: list) -> object:
        if "tl:sibling-order" in source:
            ordered_calls.append(source)
            if len(ordered_calls) > 1:
                return _render_ordering.PlainLayout(nodes={}, edge_spans={})
        return real_layout(source, rankdir, captured_edges)

    monkeypatch.setattr(_render_ordering, "_layout_dot_plain", layout)
    with pytest.warns(TorchLensWarning, match="produced no layout"):
        fallback = _draw(_Distorter(), torch.randn(1, 3), tmp_path, "fallback")

    assert len(ordered_calls) == 2
    assert fallback == plain
