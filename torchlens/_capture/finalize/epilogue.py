"""Door epilogue of the FINALIZE stage: ``finish_trace`` and the layer visualizers.

The bodies moved verbatim from the torch door (``torchlens._user_funcs``, then
``torchlens._capture.prepare.door``); the door calls ``finish_trace`` once.
``finish_trace`` reads the door facts from ``CaptureRequest.admission``
(``stop_after_site``, ``stop_after_selector_shaped``, ``stop_after_ambient``,
``save_grads_policy``, ``uses_selective_layers_to_save``,
``episode_resolved``, ``meta_admission``), the user's knobs from
``CaptureRequest.options`` (``input_transform``,
``save_args``, ``layer_visualizers``, ``unwrap_when_done``) and the cache
``key``, ``path``, ``secret`` and ``store`` callable from ``CaptureRequest.cache``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..._data_classes._preprocessing_provenance import stamp_user_transform_provenance
from ..._data_classes.trace import (
    Trace,
)
from ..._errors import InvalidArgumentError
from ..._utils.display import _vprint, ensure_trace_visualizer_dir
from ...errors import TorchLensWarning
from .._episode_ledger import (
    EPISODE_ANNOTATIONS_KEY,
    EpisodeLedger,
    EpisodeLedgerRow,
    RowStatus,
    _build_header,
    _fidelity_basis,
    write_episode_ledger,
)

if TYPE_CHECKING:
    from .._episode_ledger import ResolvedEpisode
    from ..stages import CaptureRequest, ForwardResult


def _render_layer_visualizers(
    trace: Trace,
    layer_visualizers: dict[Any, Callable[..., Any]],
) -> None:
    """Render configured per-layer visualizer thumbnails after capture.

    Parameters
    ----------
    trace:
        Completed trace whose layer outs may be rendered.
    layer_visualizers:
        Mapping from TorchLens site selectors to visualizer callables.
    """

    # Deferred: the site resolver sits above the FINALIZE layer.
    from ..._selection.resolver import _resolve_sites

    output_dir = ensure_trace_visualizer_dir(trace)
    visualizer_dir = output_dir / "visualizers"
    visualizer_dir.mkdir(parents=True, exist_ok=True)
    max_fanout = max(1, len(trace._layer_list))

    for selector, visualizer in layer_visualizers.items():
        try:
            selected_ops = tuple(_resolve_sites(trace, selector, max_fanout=max_fanout))
        except Exception as exc:
            warnings.warn(
                f"Skipping layer visualizer for selector {selector!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for op in selected_ops:
            _render_one_layer_visualizer(visualizer_dir, op, visualizer)


def _render_one_layer_visualizer(
    visualizer_dir: Path,
    op: Any,
    visualizer: Callable[..., Any],
) -> None:
    """Render one op visualizer and store its output path on the op.

    Parameters
    ----------
    visualizer_dir:
        Directory that receives rendered files.
    op:
        Layer operation to render.
    visualizer:
        Callable accepting ``(tensor, *, layer_label=None)``.
    """

    if not bool(getattr(op, "has_saved_activation", False)) or getattr(op, "out", None) is None:
        return
    try:
        rendered = visualizer(op.out, layer_label=getattr(op, "layer_label", None))
        if rendered is None:
            return
        safe_label = _safe_visualizer_filename(str(op.layer_label))
        if isinstance(rendered, str):
            output_path = visualizer_dir / f"{safe_label}.html"
            output_path.write_text(rendered, encoding="utf-8")
        else:
            output_path = visualizer_dir / f"{safe_label}.png"
            rendered.save(output_path)
        op.visualizer_path = str(output_path)
    except Exception as exc:
        warnings.warn(
            f"Layer visualizer failed for {getattr(op, 'layer_label', '<unknown>')}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _safe_visualizer_filename(label: str) -> str:
    """Return a filesystem-safe visualizer basename for a layer label.

    Parameters
    ----------
    label:
        Layer label to encode.

    Returns
    -------
    str
        Safe filename stem.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label.replace(":", "pass")).strip("_") or "layer"


def finish_trace(result: ForwardResult, request: CaptureRequest) -> Trace:
    """Run the door epilogue on a finished capture and return its trace.

    Order (pinned): stop_after never fired; on_nonfinite="warn";
    input_transform provenance; selective save; summary print; layer
    visualizers; ``unwrap_when_done``; cache store; episode ledger;
    weightsfree settlement. ``request.admission`` carries the door facts,
    ``request.options`` the user's knobs (module docstring); ``cache``
    carries ``key``, ``path``, ``secret`` and the ``store`` callable.

    Parameters
    ----------
    result:
        Forward result of the run; only ``trace`` is read.
    request:
        The request that started the run.

    Returns
    -------
    Trace
        The finished trace.
    """
    trace = result.trace
    facts = request.admission or {}
    options = request.options
    _check_stop_after_fired(trace, facts)
    if getattr(options, "on_nonfinite", "ignore") == "warn":
        _warn_once_nonfinite_detected(trace)
    input_transform = getattr(options, "input_transform", None)
    if input_transform is not None:
        stamp_user_transform_provenance(trace, input_transform)
    trace._options_save_grads = facts.get("save_grads_policy")
    if facts.get("uses_selective_layers_to_save"):
        _restrict_layer_nums_to_saved(trace, bool(getattr(options, "save_args", False)))

    # Print final summary.
    _vprint(
        trace,
        f"Done: {len(trace._layer_logs)} layers, "
        f"{trace._num_saved_ops} saved, "
        f"{trace._cost_total_activation_memory}",
    )

    layer_visualizers = getattr(options, "layer_visualizers", None)
    if layer_visualizers:
        _render_layer_visualizers(trace, layer_visualizers)

    _settle_epilogue_tail(trace, request)
    return trace


def _settle_epilogue_tail(trace: Trace, request: CaptureRequest) -> None:
    """Unwrap when asked, store the cache, write the episode ledger, settle weightsfree."""
    facts = request.admission or {}
    if getattr(request.options, "unwrap_when_done", False):
        from ...backends.torch.wrappers import unwrap_torch

        unwrap_torch()

    _store_capture_cache(trace, request.cache)

    episode_resolved = facts.get("episode_resolved")
    if episode_resolved is not None:
        # Settlement-time episode ledger (S7): written ONCE, after postprocess
        # and outcome settlement, from the settled record + module-call truth.
        write_episode_ledger(trace, episode_resolved)

    # Weightsfree settlement (F33): wrap-generation stamp on every capture
    # (W1-ORD preflight input), D22 invariant net + persisted evidence
    # envelope on structure-only captures.
    from .._structure_evidence import settle_weightsfree_capture

    settle_weightsfree_capture(trace, facts.get("meta_admission"))


def _check_stop_after_fired(trace: Trace, facts: dict[str, Any]) -> None:
    """Refuse or warn when ``stop_after`` never fired, by the site's provenance.

    Never-fired policy split by provenance: a selector-shaped site that never
    fired is a typo-shaped wrong result (the full forward ran; the caller asked
    for a frontier) and refuses typed; an exploratory callable or ambient
    context-manager site legitimately may never fire and warns with a durable
    ledger entry.
    """
    # brainpipe D-16.
    stop_after_site = facts.get("stop_after_site")
    if stop_after_site is None or bool(getattr(trace, "is_stopped", False)):
        return
    site_repr = repr(stop_after_site)
    if facts.get("stop_after_selector_shaped") and not facts.get("stop_after_ambient"):
        raise InvalidArgumentError(
            f"stop_after={site_repr} never fired: the capture ran the full "
            "forward and completed without halting, so the result is not the "
            "requested frontier",
            code="stop_after_never_fired",
            remedy=(
                "check the site against the model's module addresses "
                "(model.named_modules()) or use tl.func(...)/tl.module(...); "
                "an exploratory predicate that may legitimately never fire "
                "belongs in a callable"
            ),
            argument="stop_after",
        )
    stop_annotations = getattr(trace, "annotations", None)
    if isinstance(stop_annotations, dict):
        stop_annotations.setdefault("unmatched_capture_selectors", []).append(
            {"slot": "stop_after", "selector": site_repr}
        )
    warnings.warn(
        TorchLensWarning(
            f"stop_after={site_repr} never fired; the capture ran the full "
            "forward and the outputs are the model's real outputs, not a "
            "frontier. Remedy: check the stop_after site against the executed "
            "model, or drop stop_after",
            code="stop_after_never_fired_callable",
        ),
        stacklevel=4,
    )


def _warn_once_nonfinite_detected(trace: Trace) -> None:
    """Emit the one ``on_nonfinite="warn"`` warning when any op went non-finite."""

    from ..._data_classes._nonfinite import nonfinite_op_labels

    labels = nonfinite_op_labels(trace)
    if not labels:
        return
    shown = ", ".join(labels[:3]) + (", ..." if len(labels) > 3 else "")
    warnings.warn(
        TorchLensWarning(
            f"{len(labels)} op output(s) were non-finite (NaN or Inf) during capture: "
            f"{shown}. Remedy: inspect trace.nonfinite_ops, or capture with "
            "CaptureOptions(on_nonfinite='raise') to stop at the first one",
            code="capture_nonfinite_detected",
        ),
        stacklevel=4,
    )


def _restrict_layer_nums_to_saved(trace: Trace, save_args: bool) -> None:
    """Narrow the saved-layer index list to the ops a selective save kept."""
    trace._layer_nums_to_save = [
        op.raw_index
        for op in trace._layer_list
        if op.has_saved_activation and op.type not in {"input", "output"}
    ]
    if not save_args:
        trace._replay_arg_version_data_complete = False


def _store_capture_cache(trace: Trace, cache: dict[str, Any] | None) -> None:
    """Stamp the cache identity on the trace and store it through the request's callable."""
    if cache is None:
        return
    cache_path, cache_key, cache_secret = cache.get("path"), cache.get("key"), cache.get("secret")
    if cache_path is None or cache_key is None or cache_secret is None:
        return
    trace._provenance_cache_hit = False
    trace._provenance_cache_key = cache_key
    trace._provenance_cache_path = str(cache_path)
    cache["store"](trace, cache_path, cache_secret)


def attach_failed_episode_ledger(exc: BaseException, resolved: ResolvedEpisode) -> None:
    """Best-effort ledger attach on the FAILED path (exc.partial_trace).

    Failures here only warn -- the user's exception is never masked.

    FAILED-path episode ledger attach (lane F40a; split out of the ledger, folded
    into the FINALIZE epilogue by the capture rebuild). A failed forward's
    partial product has a settled FAILED outcome but no finished module-call
    records, so step truth is recovered from the surviving module enter/exit
    event lanes (exact: an entered call that ran zero traced ops is still a
    started step; only an exit witnesses a return), falling back to the raw op
    module stacks (non-upgrading). Split from ``_episode_ledger.py`` at the R43
    size seam: the failed path reads the settlement writers' building blocks
    but nothing reads back into it.
    """

    partial = getattr(exc, "partial_trace", None)
    trace = getattr(partial, "trace", None)
    if trace is None:
        return
    try:
        from .._episode_join import _live_step_join_envelope

        started, returned = _failed_step_counts(exc, trace, resolved)
        n_total = max(resolved.n_steps or started, started)
        header = _build_header(
            trace,
            resolved,
            fidelity=_fidelity_basis(resolved, None),
            step_join=_live_step_join_envelope(resolved, n_total, started),
        )
        complete_steps = returned if returned is not None else max(started - 1, 0)
        complete_steps = min(complete_steps, started)
        from .._episode_coupling import settlement_fire_counts

        fire_counts = settlement_fire_counts(
            resolved.coupling_session, started=started, n_total=n_total
        )
        rows = _failed_ledger_rows(n_total, complete_steps, started, fire_counts)
        ledger = EpisodeLedger(header, rows)
        trace.annotations[EPISODE_ANNOTATIONS_KEY] = ledger.to_payload()
    except Exception as attach_exc:  # noqa: BLE001 - never mask the user's failure
        warnings.warn(
            "episode ledger could not be attached to the failed partial "
            f"product: {type(attach_exc).__name__}: {attach_exc}",
            TorchLensWarning,
            stacklevel=2,
        )


def _failed_ledger_rows(
    n_total: int, complete_steps: int, started: int, fire_counts: Any
) -> list[EpisodeLedgerRow]:
    """Build the FAILED-path ledger rows: complete, then interrupted, then absent steps."""
    rows: list[EpisodeLedgerRow] = []
    for step in range(n_total):
        if step < complete_steps:
            row_status: RowStatus = "complete"
        elif step < started:
            row_status = "interrupted"
        else:
            row_status = "absent"
        rows.append(
            EpisodeLedgerRow(
                episode_step=step,
                role="prefill" if step == 0 else "decode",
                status=row_status,
                coord={"member_call_index": step + 1, "pass_range": None},
                fire_count=(fire_counts[step] if fire_counts is not None else None),
            )
        )
    return rows


def _failed_step_counts(
    exc: BaseException, trace: Any, resolved: ResolvedEpisode
) -> tuple[int, int | None]:
    """Derive ``(started, returned)`` stepped-call counts on the FAILED path.

    Exact where module enter/exit event lanes survive; non-upgrading raw-op
    module-stack fallback otherwise. Strict-arm truth (lane F40c): the join
    hooks fire BEFORE the wrapped forward, so a feed-closed halt at step k's
    entry raises before the module-enter event -- the step's input arrived
    (it ENTERED in the join sense) but no event witnesses it. The typed
    refusal carries ``break_step``; honor it so row k reads interrupted,
    never absent.
    """

    started = 0
    returned: int | None = None
    try:
        events = trace._capture_events
    except AttributeError:
        events = None
    events = events or getattr(trace, "capture_events", None)
    enter_events = getattr(events, "module_enter_events", None) if events else None
    if enter_events:
        started = sum(
            1 for event in enter_events if getattr(event, "address", None) == resolved.address
        )
        exit_events = getattr(events, "module_exit_events", None) or ()
        returned = sum(
            1 for event in exit_events if getattr(event, "address", None) == resolved.address
        )
    else:
        try:
            raw_ws = trace._raw_graph_ws
        except AttributeError:
            raw_ws = None
        raw_layers = getattr(raw_ws, "raw_layer_dict", None) or {}
        for raw_op in raw_layers.values():
            for entry in getattr(raw_op, "modules", ()) or ():
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == resolved.address:
                    started = max(started, int(entry[1]))
    break_step = getattr(exc, "fields", {}).get("break_step")
    if isinstance(break_step, int) and not isinstance(break_step, bool):
        started = max(started, break_step + 1)
        returned = min(returned, break_step) if returned is not None else break_step
    return started, returned
