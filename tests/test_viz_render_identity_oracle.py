"""Phase-0 render identity oracle for the visualization refactor.

This intentionally characterizes the current Graphviz implementation.  The
records are a tripwire: do not update them except for an explicitly approved
rendering change.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydot
import pytest
import torch
from torch import nn

import torchlens as tl

_GOLDEN_PATH = Path(__file__).parent / "golden" / "viz_render_identity_oracle.json"
_UPDATE_ENV = "TORCHLENS_UPDATE_VIZ_RENDER_ORACLE"


class OracleCNN(nn.Module):
    """Small deterministic convolution/relu/pool network."""

    def __init__(self) -> None:
        """Initialize the network."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network."""

        return self.pool(self.relu(self.conv(x)))


class OracleNested(nn.Module):
    """Nested containers used for focus and container rendering."""

    def __init__(self) -> None:
        """Initialize nested sequential containers."""

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run nested containers."""

        return self.head(self.encoder(x))


class OracleBatchNorm(nn.Module):
    """BatchNorm network for buffer visibility in train and eval modes."""

    def __init__(self) -> None:
        """Initialize the buffered network."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.norm = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the buffered network."""

        return self.relu(self.norm(self.conv(x)))


class OracleRepeat(nn.Module):
    """Repeated block network for collapse and repeat folding."""

    def __init__(self) -> None:
        """Initialize repeated blocks."""

        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each repeated block."""

        for block in self.blocks:
            x = block(x)
        return x


class OracleBranch(nn.Module):
    """Branching model with a reverse edge opportunity after projection."""

    def __init__(self) -> None:
        """Initialize the branching model."""

        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)
        self.merge = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run two sibling branches and a merge."""

        left = self.left(x)
        right = self.right(x)
        return self.merge(left + right)


class OracleOrphans(nn.Module):
    """Model that deliberately computes an unreachable retained branch."""

    def __init__(self) -> None:
        """Initialize the orphan-producing model."""

        super().__init__()
        self.live = nn.Linear(4, 4)
        self.orphan = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create and discard one branch before returning the live branch."""

        _ = self.orphan(x)
        return self.live(x)


@dataclass(frozen=True)
class OracleCase:
    """One deterministic render-oracle case."""

    name: str
    model: Callable[[], nn.Module]
    input_shape: tuple[int, ...]
    draw_kwargs: dict[str, Any]
    tags: tuple[str, ...] = ()
    train: bool = False
    keep_orphans: bool = False
    callback_kind: str | None = None
    intervene: bool = False


def _seeded_input(shape: tuple[int, ...]) -> torch.Tensor:
    """Return a deterministic CPU input tensor.

    Parameters
    ----------
    shape:
        Input shape.

    Returns
    -------
    torch.Tensor
        Deterministic float input.
    """

    generator = torch.Generator(device="cpu").manual_seed(314159)
    return torch.randn(shape, generator=generator)


def _node_spec(layer: Any, default: Any) -> Any:
    """Return a visibly customized node specification.

    Parameters
    ----------
    layer:
        Callback layer.
    default:
        Default node spec.

    Returns
    -------
    Any
        Customized default spec.
    """

    del layer
    default.extra_attrs["penwidth"] = "1.75"
    return default


def _collapsed_spec(module: Any, default: Any) -> Any:
    """Return a visibly customized collapsed-module specification.

    Parameters
    ----------
    module:
        Callback module.
    default:
        Default node spec.

    Returns
    -------
    Any
        Customized default spec.
    """

    del module
    default.extra_attrs["penwidth"] = "2.25"
    return default


def _collapse_small_modules(module: Any) -> bool:
    """Collapse modules with at least two visible operations.

    Parameters
    ----------
    module:
        Candidate module log.

    Returns
    -------
    bool
        Whether the module should collapse.
    """

    return str(getattr(module, "address", "")).startswith("blocks.")


def _skip_relu(layer: Any) -> bool:
    """Hide ReLU operations for skip-function characterization.

    Parameters
    ----------
    layer:
        Candidate layer.

    Returns
    -------
    bool
        Whether the renderer should skip this layer.
    """

    return "relu" in str(getattr(layer, "func_name", "")).lower()


def _cases() -> tuple[OracleCase, ...]:
    """Return the covering design for every public ``Trace.draw`` axis.

    Returns
    -------
    tuple[OracleCase, ...]
        Cases deliberately combine axes with a stressing model.
    """

    return (
        OracleCase("none_short_circuit", OracleCNN, (1, 1, 8, 8), {"vis_mode": "none"}),
        OracleCase(
            "cnn_rolled_profiling",
            OracleCNN,
            (1, 1, 8, 8),
            {"vis_mode": "rolled", "node_mode": "profiling"},
        ),
        OracleCase(
            "cnn_modes_themes_overlays",
            OracleCNN,
            (1, 1, 8, 8),
            {
                "node_mode": "vision",
                "vis_theme": "dark",
                "node_overlay": "bytes",
                "show_legend": True,
                "node_label_fields": ["shape", "memory"],
                "direction": "leftright",
                "order_siblings": False,
                "vis_edge_overrides": {"color": "#123456"},
                "vis_grad_edge_overrides": {"color": "#654321"},
                "vis_module_overrides": {"color": "#abcdef"},
            },
        ),
        OracleCase(
            "nested_focus_containers_depth",
            OracleNested,
            (1, 4),
            {
                "module": "encoder",
                "vis_call_depth": 1,
                "show_containers": "cluster",
                "container_max_inline": 1,
                "show_input_transform_summary": True,
                "vis_theme": "paper",
                "code_panel": True,
            },
            (
                "expected_to_change_at_phase2:focus_antiparallel",
                "expected_to_change_at_phase2:depth_container",
            ),
        ),
        OracleCase(
            "batchnorm_train_buffers",
            OracleBatchNorm,
            (2, 1, 6, 6),
            {"show_buffer_layers": "always"},
            train=True,
        ),
        OracleCase(
            "batchnorm_eval_buffers",
            OracleBatchNorm,
            (2, 1, 6, 6),
            {"show_buffer_layers": "meaningful"},
        ),
        OracleCase(
            "repeat_fold_none_callback",
            OracleRepeat,
            (1, 4),
            {"collapse": "none", "fold_repeats": True},
            callback_kind="node",
        ),
        OracleCase(
            "repeat_auto_skip_callback",
            OracleRepeat,
            (1, 4),
            {"collapse": "auto", "fold_repeats": None, "skip_fn": _skip_relu},
            ("expected_to_change_at_phase2:skip_collapse_count",),
        ),
        OracleCase(
            "repeat_max_fold_off", OracleRepeat, (1, 4), {"collapse": "max", "fold_repeats": False}
        ),
        OracleCase(
            "repeat_float_quarter", OracleRepeat, (1, 4), {"collapse": 0.25, "fold_repeats": None}
        ),
        OracleCase(
            "repeat_float_three_quarters",
            OracleRepeat,
            (1, 4),
            {"collapse": 0.75, "fold_repeats": None},
        ),
        OracleCase(
            "repeat_custom_collapse_and_spec",
            OracleRepeat,
            (1, 4),
            {
                "collapse_fn": _collapse_small_modules,
                "node_spec_fn": _node_spec,
                "collapsed_node_spec_fn": _collapsed_spec,
                "node_mode": "default",
            },
            callback_kind="both",
        ),
        OracleCase(
            "branch_rank_antiparallel",
            OracleBranch,
            (1, 4),
            {
                "vis_node_placement": "rank",
                "direction": "topdown",
                "vis_show_cone": False,
                "node_mode": "attention",
                "show_containers": "labels",
            },
        ),
        OracleCase(
            "intervention_node_mark",
            OracleCNN,
            (1, 1, 8, 8),
            {"vis_intervention_mode": "node_mark", "vis_show_cone": True},
            intervene=True,
        ),
        OracleCase(
            "intervention_as_node",
            OracleCNN,
            (1, 1, 8, 8),
            {"vis_intervention_mode": "as_node", "vis_show_cone": False},
            intervene=True,
        ),
        OracleCase(
            "orphans_visible", OracleOrphans, (1, 4), {"show_orphans": True}, keep_orphans=True
        ),
        *tuple(
            OracleCase(
                f"cnn_overlay_{overlay.replace('-', '_').replace(' ', '_')}",
                OracleCNN,
                (1, 1, 8, 8),
                {"node_overlay": overlay},
            )
            for overlay in (
                "flops",
                "time",
                "bytes",
                "magnitude",
                "grad_norm",
                "nan",
                "intervention",
                "bundle_delta",
            )
        ),
    )


def _normalized_text(value: Any) -> str:
    """Normalize semantic DOT text while retaining its meaningful contents.

    Parameters
    ----------
    value:
        DOT attribute value.

    Returns
    -------
    str
        Whitespace-normalized, unquoted text.
    """

    text = str(value).strip().strip('"')
    return re.sub(r"\s+", " ", text)


def _unit_kind(name: str, attrs: dict[str, str]) -> str:
    """Classify a visible DOT unit using stable renderer vocabulary.

    Parameters
    ----------
    name:
        DOT node identifier.
    attrs:
        Normalized node attributes.

    Returns
    -------
    str
        One of the Phase-0 structural unit kinds.
    """

    label = attrs.get("label", "").lower()
    lowered_name = name.lower()
    if "ellipsis" in lowered_name or "..." in label:
        return "repeat_ellipsis"
    if "segment" in lowered_name or "segment" in label:
        return "segment"
    if "container" in lowered_name or "container" in label:
        return "container_summary"
    if attrs.get("shape") == "box3d":
        return "module_box"
    if "input" in lowered_name or "output" in lowered_name or attrs.get("shape") == "oval":
        return "boundary"
    return "raw_op"


def _subgraph_record(graph: Any, path: tuple[str, ...]) -> dict[str, Any]:
    """Build a normalized recursive DOT region record.

    Parameters
    ----------
    graph:
        Pydot graph or subgraph.
    path:
        Parent region path.

    Returns
    -------
    dict[str, Any]
        Region tree record with ordered membership.
    """

    name = _normalized_text(graph.get_name())
    current_path = (*path, name)
    return {
        "path": list(current_path),
        "attrs": {
            key: _normalized_text(value) for key, value in sorted(graph.get_attributes().items())
        },
        "members": [
            _normalized_text(node.get_name())
            for node in graph.get_nodes()
            if node.get_name() not in {"node", "edge", "graph"}
        ],
        "children": [_subgraph_record(child, current_path) for child in graph.get_subgraphs()],
    }


def _structural_digest(dot: str) -> dict[str, Any]:
    """Return a formatting-independent ordered rendering digest from DOT.

    Nodes retain their emission ordinal, normalized label and attributes, and
    region path.  Edges retain ordered endpoint ordinals, normalized semantic
    attributes (including anti-parallel decoration), and membership.  This is
    intentionally more localizable than a second byte hash.

    Parameters
    ----------
    dot:
        DOT source returned by the same draw invocation under test.

    Returns
    -------
    dict[str, Any]
        Canonical render structure.
    """

    graphs = pydot.graph_from_dot_data(dot)
    if not graphs:
        raise AssertionError("DOT parser returned no graph")
    graph = graphs[0]
    ordinal_by_name: dict[str, int] = {}
    nodes: list[dict[str, Any]] = []

    def visit(region: Any, path: tuple[str, ...]) -> None:
        """Record nodes in DOT emission order recursively.

        Parameters
        ----------
        region:
            Current pydot region.
        path:
            Region ancestry.
        """

        region_name = _normalized_text(region.get_name())
        current_path = (*path, region_name)
        for node in region.get_nodes():
            name = _normalized_text(node.get_name())
            if name in {"node", "edge", "graph", "\\n"}:
                continue
            ordinal_by_name.setdefault(name, len(ordinal_by_name))
            attrs = {
                key: _normalized_text(value) for key, value in sorted(node.get_attributes().items())
            }
            nodes.append(
                {
                    "ordinal": ordinal_by_name[name],
                    "name": name,
                    "kind": _unit_kind(name, attrs),
                    "label": attrs.get("label", ""),
                    "attrs": attrs,
                    "region": list(current_path),
                }
            )
        for child in region.get_subgraphs():
            visit(child, current_path)

    visit(graph, ())
    edges: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []

    def visit_edges(region: Any, path: tuple[str, ...]) -> None:
        """Record edges and ordering constraints recursively.

        Parameters
        ----------
        region:
            Current pydot region.
        path:
            Region ancestry.
        """

        current_path = (*path, _normalized_text(region.get_name()))
        for edge in region.get_edges():
            source = _normalized_text(edge.get_source())
            target = _normalized_text(edge.get_destination())
            attrs = {
                key: _normalized_text(value) for key, value in sorted(edge.get_attributes().items())
            }
            item = {
                "src": ordinal_by_name.get(source, -1),
                "dst": ordinal_by_name.get(target, -1),
                "src_name": source,
                "dst_name": target,
                "kind": "ordering_constraint" if attrs.get("style") == "invis" else "edge",
                "decorations": {
                    key: attrs[key]
                    for key in (
                        "arrowhead",
                        "arrowtail",
                        "color",
                        "constraint",
                        "dir",
                        "penwidth",
                        "style",
                    )
                    if key in attrs
                },
                "attrs": attrs,
                "region": list(current_path),
            }
            edges.append(item)
            if attrs.get("style") == "invis" or attrs.get("constraint") == "false":
                constraints.append(item)
        for child in region.get_subgraphs():
            visit_edges(child, current_path)

    visit_edges(graph, ())
    return {
        "nodes": nodes,
        "edges": edges,
        "regions": _subgraph_record(graph, ()),
        "ordering_constraints": constraints,
    }


def _callback_wrapper(
    callback: Callable[[Any, Any], Any],
) -> tuple[Callable[[Any, Any], Any], Counter[int]]:
    """Wrap a NodeSpec callback and count calls by callback target identity.

    Parameters
    ----------
    callback:
        User callback to invoke.

    Returns
    -------
    tuple[Callable[[Any, Any], Any], Counter[int]]
        Wrapped callback and per-visible-target call counter.
    """

    calls: Counter[int] = Counter()

    def wrapped(target: Any, default: Any) -> Any:
        """Count then delegate one callback invocation.

        Parameters
        ----------
        target:
            Visible callback target.
        default:
            Default specification.

        Returns
        -------
        Any
            Callback result.
        """

        calls[id(target)] += 1
        return callback(target, default)

    return wrapped, calls


def _assert_exactly_once(calls: Counter[int], name: str) -> None:
    """Assert callback targets are each decorated exactly once.

    Parameters
    ----------
    calls:
        Per-target callback counts.
    name:
        Callback name for failures.
    """

    assert calls, f"{name} did not receive any visible unit"
    assert set(calls.values()) == {1}, f"{name} was re-invoked during draw/counting: {calls}"


def _stable_repr(value: Any) -> str:
    """Canonicalize unordered ``frozenset`` fragments in a diagnostic repr.

    Parameters
    ----------
    value:
        Collapse-plan or schedule artifact.

    Returns
    -------
    str
        Repr with only unordered set-member presentation normalized.
    """

    def sort_members(match: re.Match[str]) -> str:
        """Sort one frozenset repr body.

        Parameters
        ----------
        match:
            Frozenset body match.

        Returns
        -------
        str
            Canonical frozenset fragment.
        """

        members = sorted(part.strip() for part in match.group(1).split(",") if part.strip())
        return f"frozenset({{{', '.join(members)}}})"

    return re.sub(r"frozenset\(\{([^{}]*)\}\)", sort_members, repr(value))


def _capture_case(case: OracleCase, tmp_path: Path) -> dict[str, Any]:
    """Capture all oracle layers for one deterministic matrix case.

    Parameters
    ----------
    case:
        Matrix case.
    tmp_path:
        Pytest temporary output directory.

    Returns
    -------
    dict[str, Any]
        Bytes, structural, plan, schedule, and callback records.
    """

    torch.manual_seed(271828)
    model = case.model()
    model.train(case.train)
    trace_kwargs: dict[str, Any] = {"keep_orphans": case.keep_orphans}
    if case.intervene:
        trace_kwargs["intervene"] = tl.when(tl.func("relu"), tl.zero_ablate())
    trace = tl.trace(model, _seeded_input(case.input_shape), **trace_kwargs)
    try:
        # Timing is intentionally not part of a byte-identity rendering oracle.
        # Normalize the captured measurement before any timing-aware node mode or
        # overlay can format it, while retaining deterministic render behavior.
        for layer in trace.layer_list:
            layer.func_duration = 0.0
        kwargs = dict(case.draw_kwargs)
        kwargs.update(
            {
                "vis_outpath": str(tmp_path / case.name),
                "vis_save_only": True,
                "vis_fileformat": "svg",
                "vis_renderer": "graphviz",
            }
        )
        node_calls: Counter[int] | None = None
        collapsed_calls: Counter[int] | None = None
        if case.callback_kind in {"node", "both"}:
            kwargs["node_spec_fn"], node_calls = _callback_wrapper(
                kwargs.get("node_spec_fn", _node_spec)
            )
        if case.callback_kind in {"collapsed", "both"}:
            kwargs["collapsed_node_spec_fn"], collapsed_calls = _callback_wrapper(
                kwargs.get("collapsed_node_spec_fn", _collapsed_spec)
            )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="node_style=.*")
            source = trace.draw(**kwargs)
        if source is None:
            assert case.name == "none_short_circuit"
            return {"tags": list(case.tags), "dot": None, "structural": None, "callbacks": {}}
        assert isinstance(source, str)
        if node_calls is not None:
            _assert_exactly_once(node_calls, "node_spec_fn")
        if collapsed_calls is not None:
            _assert_exactly_once(collapsed_calls, "collapsed_node_spec_fn")
        plans = {
            str(mode): _stable_repr(trace.collapse_plan(mode=mode))
            for mode in ("auto", "max", 0.25, 0.75)
        }
        return {
            "tags": list(case.tags),
            "dot": source,
            "structural": _structural_digest(source),
            "plans": plans,
            "schedule": _stable_repr(trace.collapse_schedule()),
            "callbacks": {
                "node_spec_fn": sorted((node_calls or {}).values()),
                "collapsed_node_spec_fn": sorted((collapsed_calls or {}).values()),
            },
        }
    finally:
        trace.cleanup()


def _capture_backward_combined(tmp_path: Path) -> dict[str, Any]:
    """Snapshot backward and combined DOT plus structural records.

    Parameters
    ----------
    tmp_path:
        Pytest temporary output directory.

    Returns
    -------
    dict[str, Any]
        Frozen Phase-8 baselines.
    """

    torch.manual_seed(161803)
    trace = tl.trace(OracleBranch(), _seeded_input((1, 4)), save_grads="all")
    try:
        loss = trace[trace.output_layers[0]].out.sum()
        trace.log_backward(loss)
        _stabilize_backward_identifiers(trace)
        backward = trace.draw_backward(
            vis_outpath=str(tmp_path / "backward"), vis_save_only=True, vis_fileformat="svg"
        )
        combined = trace.draw_combined(
            vis_outpath=str(tmp_path / "combined"), vis_save_only=True, vis_fileformat="svg"
        )
        return {
            name: {"dot": dot, "structural": _structural_digest(dot)}
            for name, dot in {"backward": backward, "combined": combined}.items()
        }
    finally:
        trace.cleanup()


def _stabilize_backward_identifiers(trace: tl.Trace) -> None:
    """Replace process-address grad-fn IDs with deterministic test identifiers.

    Backward emitters currently expose Python object IDs directly in DOT node
    names.  The test-only normalization changes the captured fixture before
    rendering, not renderer behavior, so the baseline remains byte-stable.

    Parameters
    ----------
    trace:
        Trace containing one captured backward pass.
    """

    grad_fns = list(trace.grad_fns)
    identifiers = {item.grad_fn_object_id: index for index, item in enumerate(grad_fns)}
    for item in grad_fns:
        old_identifier = item.grad_fn_object_id
        item.grad_fn_object_id = identifiers[old_identifier]
        item.next_grad_fn_ids = [identifiers[next_id] for next_id in item.next_grad_fn_ids]
    trace.grad_fn_logs = {item.grad_fn_object_id: item for item in grad_fns}
    trace.grad_fn_order = [identifiers[item] for item in trace.grad_fn_order]


def _record(tmp_path: Path) -> dict[str, Any]:
    """Build the complete Phase-0 identity record.

    Parameters
    ----------
    tmp_path:
        Pytest temporary output directory.

    Returns
    -------
    dict[str, Any]
        Canonical committed golden record.
    """

    return {
        "schema_version": 1,
        "cases": {case.name: _capture_case(case, tmp_path) for case in _cases()},
        "backward_combined": _capture_backward_combined(tmp_path),
    }


def _digest(record: dict[str, Any]) -> str:
    """Return the deterministic SHA-256 of an oracle record.

    Parameters
    ----------
    record:
        Oracle record.

    Returns
    -------
    str
        Hex digest.
    """

    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _digest_chunks(record: dict[str, Any]) -> list[str]:
    """Split the record digest into scanner-safe verification chunks.

    Parameters
    ----------
    record:
        Oracle record.

    Returns
    -------
    list[str]
        Eight-character digest chunks.
    """

    digest = _digest(record)
    return [digest[index : index + 8] for index in range(0, len(digest), 8)]


@pytest.mark.smoke
def test_viz_render_identity_oracle(tmp_path: Path) -> None:
    """Characterize every draw axis with bytes and structural goldens."""

    actual = _record(tmp_path)
    payload = {"sha256_chunks": _digest_chunks(actual), "record": actual}
    if _UPDATE_ENV in __import__("os").environ:
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    expected = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload == expected
