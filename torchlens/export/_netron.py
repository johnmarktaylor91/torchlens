"""The enriched netron exporter entry (lane F14, netron memo B1-B6).

``tl.export.netron`` writes an honest custom-domain ONNX-JSON artifact netron
lights up fully: typed values and native graph I/O (netron's oldest open ask,
#71), a curated accurately-worded properties panel, the module hierarchy as
nested per-call FunctionProtos with one-click drill-down, buffer density via
the existing tri-state policy, the ``netron:attachment`` metrics companion,
and a serve/notebook one-liner. Every kwarg spelling here is a placeholder
for the naming sprint (DOCUMENTED-UNSTABLE).

The file-export default granularity is ``"module"`` (ruling N2); colour
re-domaining is not built (ruling N1) -- structure replaces colour.
"""

from __future__ import annotations

import json as _json
import warnings
from pathlib import Path
from typing import Any

from .._capture_honesty import capture_honesty_facts
from .._literals import BufferVisibilityLiteral
from .._options_validation import _validate_buffer_visibility
from ..errors import ConfigurationError, TorchLensWarning
from ..utils.display import atomic_write_text
from ._netron_attachment import write_attachment
from ._netron_emit import NETRON_SCHEMA_VERSION, emit_model_json
from ._netron_module import project_module
from ._netron_records import (
    NETRON_MODULE_DOMAIN,
    NetronProjection,
    project_op,
    project_rolled,
)

__tl_layer__ = "L8"

#: Disclaimer embedded in the netron export's model and graph doc strings.
NETRON_DISCLAIMER = (
    "TorchLens lossy graph export: not a runnable ONNX model; graph inspection "
    "only. Ops keep their captured TorchLens names under the ai.torchlens.lossy "
    "domain and carry no standard-ONNX execution semantics."
)

#: Closed granularity vocabulary (memo D-11/D-13; spellings [UI-SPRINT]).
_GRANULARITIES = frozenset({"op", "module", "rolled"})

#: Extent budget in netron ranks: ~27 ranks keeps node labels >= 7 px at fit
#: zoom in a 1600x1000 viewport at netron's ~55 px/rank (memo D-12). The
#: 7 px legibility floor is a judgment constant; a different floor re-derives
#: this budget in one line.
NETRON_EXTENT_RANK_BUDGET = 27


def _validate_options(granularity: str, depth: int, show_buffers: BufferVisibilityLiteral) -> None:
    """Refuse unknown option tokens with teaching messages (typed)."""

    if granularity not in _GRANULARITIES:
        raise ConfigurationError(
            f"Netron export granularity {granularity!r} is not a supported "
            f"projection; the closed vocabulary is {sorted(_GRANULARITIES)}. "
            "Remedy: pass granularity='module' (the architecture view, the "
            "default), 'op' (every leaf op), or 'rolled' (repeated passes "
            "merged).",
            code="netron_granularity_invalid",
            granularity=str(granularity),
        )
    if not isinstance(depth, int) or isinstance(depth, bool) or depth < 1:
        raise ConfigurationError(
            f"Netron export depth {depth!r} is not a positive int; depth is "
            "the module level shown in the root graph (1 = the shallow "
            "architecture diagram). Remedy: pass an int >= 1.",
            code="netron_granularity_invalid",
            depth=str(depth),
        )
    _validate_buffer_visibility(show_buffers)


def _assert_function_dag(projection: NetronProjection) -> bool:
    """Explicit DAG check over the function-call graph (memo D-11).

    Per-call keying makes cycles structurally impossible, but a cycle is not
    a degraded render -- it kills netron (stack overflow, dead file, zero
    console errors) -- so the named assertion stays as a belt.
    """

    calls: dict[str, set[str]] = {}
    for function in projection.functions:
        calls[function.name] = {
            node.op_type for node in function.nodes if node.domain == NETRON_MODULE_DOMAIN
        }
    visiting: set[str] = set()
    done: set[str] = set()

    def _acyclic(name: str) -> bool:
        """DFS one function; False when its call chain revisits itself."""

        if name in done:
            return True
        if name in visiting:
            return False
        visiting.add(name)
        for callee in calls.get(name, ()):
            if callee in calls and not _acyclic(callee):
                return False
        visiting.discard(name)
        done.add(name)
        return True

    return all(_acyclic(name) for name in calls)


def _build_projection(
    log: Any, granularity: str, depth: int, show_buffers: str
) -> NetronProjection:
    """Build the selected projection, fail-soft on a module call-graph cycle."""

    if granularity == "op":
        return project_op(log, show_buffers)
    if granularity == "rolled":
        return project_rolled(log, show_buffers)
    projection = project_module(log, show_buffers, depth)
    if projection.functions and not _assert_function_dag(projection):
        warnings.warn(
            TorchLensWarning(
                "Netron module projection produced a function-call cycle, "
                "which netron cannot open (a dead file, not a degraded "
                "render); falling back to the valid op projection. Remedy: "
                "export with granularity='op', and report this trace -- "
                "per-call keying should make cycles impossible.",
                code="netron_module_projection_fallback",
            ),
            stacklevel=3,
        )
        projection = project_op(log, show_buffers)
        projection.fallback_reason = "module_call_cycle"
    return projection


def _warn_extent(projection: NetronProjection) -> None:
    """Warn-plus-remedy when the root graph exceeds the extent budget (D-12)."""

    if projection.extent_ranks <= NETRON_EXTENT_RANK_BUDGET:
        return
    remedies = {
        "op": "export with granularity='module' (the default architecture view)",
        "module": "reduce depth= toward 1, or export a sub-module's trace",
        "rolled": "export with granularity='module'",
    }
    warnings.warn(
        TorchLensWarning(
            f"Netron export root graph is ~{projection.extent_ranks} ranks "
            f"deep, past the ~{NETRON_EXTENT_RANK_BUDGET}-rank budget where "
            "node labels stay legible at fit zoom (about one 1600x1000 "
            "screen); netron will open it, but as a rope, not a diagram. "
            f"Remedy: {remedies[projection.granularity]}.",
            code="netron_extent_budget_exceeded",
        ),
        stacklevel=3,
    )


def _capture_outcome_marker(log: Any) -> str:
    """Return the capture-outcome status string, best-effort (memo D-19).

    The exporters sit OUTSIDE the capture-outcome gates by design, so this
    marker is the only place a halted or unverified trace can say so.
    """

    try:
        outcome = getattr(log, "outcome", None)
        status = getattr(outcome, "status", None)
        value = getattr(status, "value", None)
        return str(value if value is not None else status or "")
    except (AttributeError, TypeError, ValueError):
        return ""


def _metadata_props(
    log: Any, projection: NetronProjection, show_buffers: str, baseline: str | None
) -> list[tuple[str, str]]:
    """Assemble the ordered honesty metadata rows (memo D-19)."""

    rows: list[tuple[str, str]] = [
        ("torchlens.lossy_export", "true"),
        ("torchlens.runnable", "false"),
        ("torchlens.netron_schema", NETRON_SCHEMA_VERSION),
        ("torchlens.capture_honesty", _json.dumps(capture_honesty_facts(log))),
    ]
    outcome = _capture_outcome_marker(log)
    if outcome:
        rows.append(("torchlens.capture_outcome", outcome))
    rows.append(("torchlens.model_class", str(getattr(log, "model_class_name", ""))))
    rows.append(("torchlens.granularity", projection.granularity))
    if projection.module_depth is not None:
        rows.append(("torchlens.module_depth", str(projection.module_depth)))
    rows.append(("torchlens.buffer_policy", show_buffers))
    rows.append(("torchlens.node_count", str(len(projection.nodes))))
    rows.append(("torchlens.edge_count", str(projection.edge_count)))
    rows.append(("torchlens.function_count", str(len(projection.functions))))
    rows.append(("torchlens.omitted_value_count", str(projection.omitted_value_count)))
    hidden_buffer_count = sum(len(v) for v in projection.hidden_buffers.values())
    rows.append(("torchlens.hidden_buffer_count", str(hidden_buffer_count)))
    rows.append(("torchlens.hidden_op_count", str(projection.hidden_op_count)))
    if getattr(log, "intervention_audit", None):
        rows.append(("torchlens.intervened", "true"))
    if baseline is not None:
        # Reserved diff-plumbing slot (netron #1544; compo row C4).
        rows.append(("torchlens.baseline", str(baseline)))
    if projection.fallback_reason:
        rows.append(("torchlens.granularity_fallback_reason", projection.fallback_reason))
    return rows


def _serve(name: str, payload: bytes, *, browse: bool) -> tuple[str, int]:
    """Serve the artifact through the netron package (memo D-20).

    Gotchas locked by execution: ``bytearray`` not ``bytes`` (bytes falls
    into an experimental sniffing branch), the served route is
    ``/data/<basename>``, the name must end ``.json``; loopback-only default
    with an ephemeral port.
    """

    try:
        import netron as netron_package
    except ImportError as error:
        raise ConfigurationError(
            "Opening in netron needs the netron package, which is not "
            "installed (the artifact file itself needs no extra -- open it "
            "at netron.app or with the desktop app). Remedy: pip install "
            "'torchlens[netron]'.",
            code="netron_serve_unavailable",
        ) from error
    address = netron_package.serve(
        name, bytearray(payload), address=("127.0.0.1", 0), browse=browse
    )
    _maybe_widget(netron_package, address)
    return address


def _maybe_widget(netron_package: Any, address: tuple[str, int]) -> None:
    """Render the notebook widget when IPython is displaying (memo D-20)."""

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            netron_package.widget(address)
    except ImportError:
        pass


def netron(  # noqa: PLR0913 -- the memo-ruled public exporter surface (D-11..D-20)
    log: Any,
    path: str | Path | None = None,
    *,
    granularity: str = "module",
    depth: int = 1,
    show_buffers: BufferVisibilityLiteral = "meaningful",
    attachment: bool = False,
    open: bool = False,  # noqa: A002 - the memo-ruled one-liner spelling
    baseline: str | None = None,
) -> Path | None:
    """Export a lossy ONNX ``ModelProto`` JSON graph that netron opens.

    The payload is valid ONNX protobuf JSON (irVersion 10, camelCase field
    names, strict-parseable into ``onnx.ModelProto`` and green under
    ``onnx.checker.check_model(full_check=True)``) -- netron's ProtoReader
    acceptance contract. It is deliberately NOT runnable: ops keep their
    captured TorchLens names under ``ai.torchlens.lossy``, module calls
    become navigable FunctionProtos under ``ai.torchlens.module``, and the
    disclaimer rides ``docString`` and ``metadataProps``. Every kwarg
    spelling is DOCUMENTED-UNSTABLE pending the naming sprint.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` to export.
    path:
        Destination JSON path; must end ``.json`` for netron's sniffer.
        ``None`` serves the artifact from memory (requires ``open=True``).
    granularity:
        ``"module"`` (default, ruling N2): per-call FunctionProto hierarchy,
        one-click drill-down. ``"op"``: every leaf op flat. ``"rolled"``:
        repeated passes merged by rolled identity, variance-gated types.
    depth:
        Module level shown in the root graph (module granularity only);
        the interim default 1 is the shallow architecture diagram (memo
        DISSENT D1), recorded in metadata.
    show_buffers:
        The existing tri-state buffer policy (``never|meaningful|always``);
        ``meaningful`` also drops counter-update chains and disclosed
        hidden counts ride the owning function docString and the model
        metadata.
    attachment:
        Also write the ``<stem>.attachment.json`` metrics companion
        (netron's own sidecar channel; drop it onto the open model).
    open:
        Serve through the installed netron package (loopback, ephemeral
        port) and open a browser tab / notebook widget.
    baseline:
        Optional baseline trace identifier for the reserved
        ``torchlens.baseline`` diff-plumbing slot (netron #1544).

    Returns
    -------
    Path | None
        The written JSON path, or ``None`` for the memory-only serve mode.
    """

    _validate_options(granularity, depth, show_buffers)
    if path is None and not open:
        raise ConfigurationError(
            "Netron export got path=None without open=True, leaving it "
            "nothing to do. Remedy: pass a destination path, or open=True "
            "to serve the artifact from memory.",
            code="netron_path_or_open_required",
        )
    if path is None and attachment:
        raise ConfigurationError(
            "The netron attachment companion is a second FILE netron merges "
            "beside the artifact, so it needs a real path. Remedy: pass "
            "path= together with attachment=True.",
            code="netron_attachment_invalid",
        )
    projection = _build_projection(log, granularity, depth, show_buffers)
    _warn_extent(projection)
    graph_name = str(getattr(log, "model_class_name", "TorchLens graph"))
    text = emit_model_json(
        projection,
        graph_name=graph_name,
        disclaimer=NETRON_DISCLAIMER,
        metadata_props=_metadata_props(log, projection, show_buffers, baseline),
    )
    if path is None:
        _serve(f"{graph_name or 'torchlens'}.json", text.encode(), browse=True)
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, text)
    if attachment:
        write_attachment(projection, destination)
    if open:
        _serve(destination.name, text.encode(), browse=True)
    return destination
