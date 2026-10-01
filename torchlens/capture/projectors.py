"""Sibling projections over a sealed capture run core."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from ..ir.capture_events import _clone_op_event_for_replay
from ..ir.events import OpEvent
from .session import CapturedRunCore

if TYPE_CHECKING:
    from ..fastlog.types import ActivationRecord

_REFRESH_SOURCES: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()

#: Pinned refresh graph-change message term. Historical ``save_new_outs`` callers
#: match on the literal phrase "computational graph changed", so every projector
#: refusal -- the generic signature arm and the typed D18 buffer-sink arms alike
#: -- carries this exact base message. The phrase is a compatibility floor, not
#: the branching contract: the generic arm stamps ``refresh_graph_change_detail``
#: on the raised ``ValueError`` (AUD-HONESTY H3) so the live ``run()`` provider
#: settles it through the divergence spine (``PathDivergenceError`` /
#: ``RunnableErrorCode.CALL_STRUCTURE_MISMATCH``, ``on_divergence`` honored), and
#: public code branches on ``exc.fields["code"]`` / the run report, never text.
_GRAPH_CHANGE_MESSAGE = (
    "The computational graph changed for this forward pass compared to the original "
    "call to trace (either due to different inputs or a different "
    "random seed). Live-model state mutation across run() calls (for example "
    "BatchNorm running stats, caches, or counters) is another likely cause. "
    "For an explicitly static feature-extraction loop, use run(inputs=..., fast=True); "
    "otherwise save_new_outs failed. Please "
    "re-run trace with the desired inputs."
)


def _distinct_label_index_keys(label: str, raw_label: str | None) -> tuple[str, ...]:
    """Return the distinct label keys that should index one activation record.

    Parameters
    ----------
    label
        Primary public label for the retained record.
    raw_label
        Optional raw label alias for the same retained record.

    Returns
    -------
    tuple[str, ...]
        Unique label keys that should reference the record exactly once.
    """

    if raw_label is None or raw_label == label:
        return (label,)
    return (label, raw_label)


def _signature_sort_key(value: Any) -> tuple[str, str]:
    """Return a stable comparable token for graph-signature sort keys.

    Parameters
    ----------
    value
        Parent-position key to normalize for signature comparison.

    Returns
    -------
    tuple[str, str]
        Type-qualified token that keeps incomparable key types sortable.
    """

    return (f"{type(value).__module__}.{type(value).__qualname__}", repr(value))


def _normalized_parent_arg_positions(
    parent_arg_positions: dict[str, dict[Any, str]] | None,
) -> tuple[tuple[str, tuple[tuple[Any, str], ...]], ...]:
    """Return a deterministic signature view of parent argument positions.

    Parameters
    ----------
    parent_arg_positions
        Layer parent routing metadata keyed by argument domain.

    Returns
    -------
    tuple[tuple[str, tuple[tuple[Any, str], ...]], ...]
        Stable, comparable representation of argument and keyword parent routing.
    """

    positions = parent_arg_positions or {}
    return tuple(
        (
            arg_domain,
            tuple(
                sorted(
                    (positions.get(arg_domain, {}) or {}).items(),
                    key=lambda item: _signature_sort_key(item[0]),
                )
            ),
        )
        for arg_domain in ("args", "kwargs")
    )


def _format_parent_arg_positions(
    parent_arg_positions: tuple[tuple[str, tuple[tuple[Any, str], ...]], ...],
) -> str:
    """Format normalized parent routing for graph-drift error details.

    Parameters
    ----------
    parent_arg_positions
        Normalized parent routing metadata from ``_normalized_parent_arg_positions``.

    Returns
    -------
    str
        Compact human-readable routing summary.
    """

    rendered_parts: list[str] = []
    for arg_domain, entries in parent_arg_positions:
        rendered_parts.append(f"{arg_domain}={list(entries)!r}")
    return ", ".join(rendered_parts)


def _cloned_core_events(core: CapturedRunCore) -> tuple[OpEvent, ...]:
    """Return sealed core events with independent mutable dict fields.

    Projection consumers may mutate ``transform_config`` and
    ``parent_arg_positions`` in place, so each event is re-created with fresh
    copies of those two dicts; every other field is shared by reference.
    """

    return tuple(_clone_op_event_for_replay(event) for event in core.events)


@dataclass(frozen=True, slots=True)
class RefreshProjector:
    """Project a same-graph captured rerun onto an existing Trace."""

    target: Any
    layer_nums_to_save: str | tuple[int, ...] = "all"
    grad_layer_nums_to_save: str | tuple[int, ...] = "all"

    _DYNAMIC_OP_FIELDS = frozenset(
        {
            "out",
            "transformed_out",
            "has_saved_activation",
            "saved_args",
            "saved_kwargs",
            "has_saved_args",
            "shape",
            "dtype",
            "dtype_ref",
            "device_ref",
            "activation_memory",
            "transformed_out_shape",
            "transformed_out_dtype",
            "transformed_activation_memory",
            "grad",
            "transformed_grad",
            "has_grad",
            "grad_shape",
            "grad_dtype",
            "gradient_memory",
            "transformed_grad_shape",
            "transformed_grad_dtype",
            "transformed_gradient_memory",
            "grad_fn_class_name",
            "grad_fn_class_qualname",
            "grad_fn_object_id",
            "grad_fn_handle",
            "autograd_memory",
            "num_autograd_tensors",
            "bytes_delta_at_call",
            "bytes_peak_at_call",
            "func_duration",
            "func_rng_states",
            "func_autocast_state",
            "non_tensor_pos_args",
            "non_tensor_kwargs",
            "func_non_tensor_args",
            "func_config",
            "has_out_variations",
            "out_versions_by_child",
            "is_scalar_bool",
            "bool_value",
            "flops_forward",
            "flops_backward",
        }
    )

    def project(self, refreshed: Any) -> None:
        """Validate and apply refreshed payloads without replacing graph containers.

        Parameters
        ----------
        refreshed
            Fully postprocessed Trace captured by the fixed-order kernel.

        Raises
        ------
        ValueError
            If the refreshed computational graph differs from the target graph.
        """

        # D18: the typed buffer-sink arms run BEFORE the generic signature arm so
        # every buffer-sink-shaped refusal (train-mode writer, claim/evidence
        # contradiction, refresh write tripwire, evidence asymmetry) raises the
        # TYPED BufferSinkRoutingError -- a mode flip between runs changes the
        # graph shape too, and routing it through the untyped generic arm would
        # void the D-ruling's "fails typed" obligation.
        self._check_buffer_sink_routing(refreshed)
        target_signature = self._graph_signature(self.target)
        refreshed_signature = self._graph_signature(refreshed)
        if target_signature != refreshed_signature:
            raise self._graph_change_error(
                refreshed,
                self._graph_signature_mismatch_detail(refreshed),
            )
        refreshed_by_raw = {layer._layer_label_raw: layer for layer in refreshed.layer_list}
        # B8-36: one aggregated warning per refresh. A per-layer warning with the
        # label interpolated into the message defeats Python's warning dedup
        # (hundreds of warnings on a real CNN, and ``-W error`` aborts the
        # refresh at layer one), so labels are collected during the sweep and
        # reported once.
        shape_changed: list[str] = []
        for layer in self.target.layer_list:
            new_shape = refreshed_by_raw[layer._layer_label_raw].shape
            if layer.shape is not None and new_shape != layer.shape:
                shape_changed.append(
                    f"'{layer.layer_label}' (expected {layer.shape}, got {new_shape})"
                )
        if shape_changed:
            preview = "; ".join(shape_changed[:3])
            remainder = len(shape_changed) - 3
            suffix = f"; and {remainder} more layer(s)" if remainder > 0 else ""
            warnings.warn(
                f"Tensor shape changed for {len(shape_changed)} layer(s): "
                f"{preview}{suffix}. "
                "The computational graph may have changed between ops."
            )
        from ..data_classes._state_adapter import state_items

        preserved_states = [dict(state_items(layer)) for layer in self.target.layer_list]
        if not self.target._refresh_matching_rerun_state_from(refreshed):
            raise self._graph_change_error(
                refreshed,
                "raw/final layer labels no longer align during refresh projection",
            )
        for layer, preserved in zip(self.target.layer_list, preserved_states, strict=False):
            for field_name, value in preserved.items():
                if field_name not in self._DYNAMIC_OP_FIELDS:
                    layer._internal_set(field_name, value)
        refreshed._refresh_projection_target_ref = weakref.ref(self.target)
        _REFRESH_SOURCES[self.target] = refreshed
        self.target._layer_nums_to_save = self.layer_nums_to_save
        self.target._grad_op_nums_to_save = self.grad_layer_nums_to_save
        self._rebind_backward_hooks()
        self._separate_output_payloads()
        if self.layer_nums_to_save != "all":
            selected = set(self.layer_nums_to_save)
            for output_label in self.target.output_layers:
                output = self.target.layer_dict_all_keys[output_label]
                selected.add(output.raw_index)
                selected.update(
                    self.target.layer_dict_all_keys[parent].raw_index for parent in output.parents
                )
            for layer in self.target.layer_list:
                if layer.raw_index not in selected:
                    self._clear_payload(layer)

    def project_prefix(self, refreshed: Any, plan: Any) -> None:
        """Prefix-scoped projection for a truncated (until=) live refresh (L4 2.3).

        The refreshed argument is the HALTED internal capture covering the
        executed prefix. All projector tripwires run at full strength on that
        prefix -- the graph signature over executed non-output ops, and the
        typed D18 buffer-sink arms scoped to in-prefix sinks -- and a prefix
        mismatch refuses exactly like a full mismatch (the projector is never
        loosened to make a truncated projection pass). Target ops beyond the
        stop are retained STRUCTURE-ONLY: their value payloads are cleared so a
        stale capture-time tensor can never read as a fresh value, and the
        skipped set is recorded on the fork for the run-level disclosure.
        Frontier-synthesized output nodes on the partial are excluded from the
        comparison (the target's real output nodes are in the skipped set).
        """

        executed = frozenset(plan.executed_raw_labels)
        self._check_buffer_sink_routing(refreshed, executed_raw_labels=executed)
        target_prefix = [
            layer
            for layer in self.target.layer_list
            if layer.layer_type != "output" and layer._layer_label_raw in executed
        ]
        refreshed_prefix = [layer for layer in refreshed.layer_list if layer.layer_type != "output"]

        def _prefix_signature(layers: list[Any]) -> tuple[tuple[Any, ...], ...]:
            """Return the structural (label, type, parents) prefix signature."""

            return tuple(
                (
                    layer._layer_label_raw,
                    layer.layer_type,
                    tuple(layer.parents),
                    _normalized_parent_arg_positions(layer.parent_arg_positions),
                )
                for layer in layers
            )

        target_signature = _prefix_signature(target_prefix)
        refreshed_signature = _prefix_signature(refreshed_prefix)
        if target_signature != refreshed_signature:
            raise self._graph_change_error(
                refreshed,
                "the executed prefix of the truncated rerun does not match the "
                f"recorded prefix (expected {len(target_signature)} prefix op(s), "
                f"got {len(refreshed_signature)}; first divergence at "
                f"{next((expected[0] for expected, actual in zip(target_signature, refreshed_signature, strict=False) if expected != actual), 'op count')!r})",
            )
        from ..data_classes._state_adapter import state_items

        preserved_states = [dict(state_items(layer)) for layer in target_prefix]
        for layer, new_layer in zip(target_prefix, refreshed_prefix, strict=True):
            self.target._refresh_rerun_op_from(layer, new_layer)
        for layer, preserved in zip(target_prefix, preserved_states, strict=True):
            for field_name, value in preserved.items():
                if field_name not in self._DYNAMIC_OP_FIELDS:
                    layer._internal_set(field_name, value)
        # 3.3.3 sanitation: skipped records are retained structure-only; value
        # payloads are cleared at fork-finalization so omission never reads as
        # a fresh value.
        for layer in self.target.layer_list:
            if layer.layer_type == "output" or layer._layer_label_raw not in executed:
                self._clear_payload(layer)
        self.target.__dict__["_run_truncation_skipped_raw_labels"] = tuple(plan.skipped_raw_labels)
        self.target._layer_nums_to_save = self.layer_nums_to_save
        self.target._grad_op_nums_to_save = self.grad_layer_nums_to_save
        self._rebind_backward_hooks(raw_labels=executed)

    @staticmethod
    def _graph_change_error(refreshed: Any, detail: str | None = None) -> ValueError:
        """Build the legacy graph-change exception with partial-capture metadata."""

        from ..partial import PartialTrace

        detail_suffix = "" if detail is None else f" Detail: {detail}."
        error = ValueError(f"{_GRAPH_CHANGE_MESSAGE}{detail_suffix}")
        error.partial_log = PartialTrace(refreshed, error)  # type: ignore[attr-defined]
        # H3 marker: the live provider identifies the generic graph-change arm by
        # this attribute (never by message text) and routes it through the shared
        # divergence-policy spine. The D18 buffer-sink arms are already typed and
        # deliberately do NOT carry it.
        error.refresh_graph_change_detail = (  # type: ignore[attr-defined]
            "graph signature changed" if detail is None else str(detail)
        )
        return error

    @staticmethod
    def _buffer_sink_routing_error(refreshed: Any, detail: str) -> Exception:
        """Build the typed D18 buffer-sink routing refusal.

        All four D18 arms (train-mode writer, unproven evidence, mode-claim
        contradiction, refresh write tripwire / evidence asymmetry) raise this
        one typed class with the one frozen ``RunnableErrorCode`` member.
        ``ValueError`` stays in the MRO and the message keeps the pinned
        "computational graph changed" term, so historical callers survive.
        """

        from ..errors.runnable import BufferSinkRoutingError
        from ..partial import PartialTrace
        from ..runnable import RunnableErrorCode

        error = BufferSinkRoutingError(
            f"{_GRAPH_CHANGE_MESSAGE} Detail: {detail}. "
            "Remedy: put the model in eval mode (or otherwise stop value-changing "
            "buffer writes) and re-capture, then refresh",
            code=RunnableErrorCode.BUFFER_SINK_ROUTING_MUTABLE.value,
            detection_stage="refresh_buffer_sink_routing",
        )
        error.partial_log = PartialTrace(refreshed, error)  # type: ignore[attr-defined]
        return error

    @staticmethod
    def _buffer_sink_evidence(trace: Any) -> tuple[tuple[str, Any, Any], ...]:
        """Return the ordered buffer-sink write-evidence tuple for one Trace.

        Each row is ``(raw_label, buffer_value_changed, buffer_write_kind)`` for
        every buffer-typed internal sink op. ``buffer_value_changed`` is derived
        write evidence (byte comparison in the buffer-write journal), never a
        self-declared flag; ``None`` means unproven and fails closed downstream.
        """

        rows: list[tuple[str, Any, Any]] = []
        for label in getattr(trace, "internal_sink_ops", ()):
            layer = trace.layer_dict_all_keys[label]
            if layer.layer_type != "buffer":
                continue
            rows.append(
                (
                    layer._layer_label_raw,
                    layer.buffer_value_changed,
                    layer.buffer_write_kind,
                )
            )
        return tuple(rows)

    @staticmethod
    def _buffer_sink_mode_claims(trace: Any, sink_layer: Any) -> list[tuple[str, bool]]:
        """Return the recorded mode claims for one buffer sink's producing op.

        Two independent mode authorities are consulted where they exist: the
        recorded literal mode argument (``training`` / ``use_input_stats``) on
        the producing mode-sensitive call, and the capture-recorded
        ``module_training_modes`` entry for the producing op's innermost
        containing module. Each claim is cross-checked against the write
        evidence by the caller; absent facts yield no claim (the primary
        evidence key still governs).
        """

        from ..runnable import is_mode_sensitive_qualname

        claims: list[tuple[str, bool]] = []
        qualname = getattr(sink_layer, "buffer_source_func_name", None)
        if not is_mode_sensitive_qualname(qualname):
            return claims
        source_label = getattr(sink_layer, "buffer_source", None)
        if not source_label or source_label not in trace.layer_dict_all_keys:
            return claims
        source = trace.layer_dict_all_keys[source_label]
        tail = (qualname or "").rsplit(".", 1)[-1].removesuffix("_")
        mode_argument = "use_input_stats" if tail.endswith("instance_norm") else "training"
        kwargs = getattr(source, "non_tensor_kwargs", None) or {}
        literal = kwargs.get(mode_argument)
        if not isinstance(literal, bool):
            # The mode flag is the first boolean among the positional non-tensor
            # arguments for every torch batch_norm/instance_norm signature
            # (momentum/eps are floats; cudnn_enabled trails the mode flag).
            literal = next(
                (
                    value
                    for value in getattr(source, "non_tensor_pos_args", None) or ()
                    if isinstance(value, bool)
                ),
                None,
            )
        if isinstance(literal, bool):
            claims.append((f"recorded literal {mode_argument!r} argument", literal))
        modules = tuple(getattr(source, "modules", ()) or ())
        if modules:
            address = str(modules[-1]).rsplit(":", 1)[0]
            modes = getattr(getattr(trace, "_runnable", None), "module_training_modes", None) or {}
            if address in modes:
                claims.append((f"module_training_modes[{address!r}] record", bool(modes[address])))
        return claims

    def _check_buffer_sink_routing(
        self, refreshed: Any, executed_raw_labels: frozenset[str] | None = None
    ) -> None:
        """Enforce the D18 mode-aware buffer-sink routing contract (typed).

        Refuse iff any buffer sink carries ``buffer_value_changed is not False``
        (``True`` = a write happened; ``None`` = unproven, fail closed), a
        recorded mode claim contradicts the write evidence, the refreshed
        rerun's own journal recorded a value-changing buffer write (O1), or the
        target-vs-refreshed evidence tuples diverge (O2). Eval-mode BatchNorm
        (all sinks ``False`` with agreeing eval claims) passes.

        ``executed_raw_labels`` scopes the check to a truncated run's executed
        prefix: in-prefix sinks are compared at FULL strength including the O2
        evidence tuple, while sinks whose producing calls were cut are excluded
        (disclosed via the truncation record, not compared).
        """

        target_sinks = [
            self.target.layer_dict_all_keys[label]
            for label in getattr(self.target, "internal_sink_ops", ())
            if self.target.layer_dict_all_keys[label].layer_type == "buffer"
            and (
                executed_raw_labels is None
                or self.target.layer_dict_all_keys[label]._layer_label_raw in executed_raw_labels
            )
        ]
        target_rows = tuple(
            (layer._layer_label_raw, layer.buffer_value_changed, layer.buffer_write_kind)
            for layer in target_sinks
        )
        # Belt: recorded mode claims must agree with the write evidence where
        # both exist; a contradiction refuses typed, never resolved permissively.
        for sink_layer, (raw_label, value_changed, _) in zip(
            target_sinks, target_rows, strict=True
        ):
            if value_changed is None:
                continue
            for authority, claim in self._buffer_sink_mode_claims(self.target, sink_layer):
                if claim is not bool(value_changed):
                    raise self._buffer_sink_routing_error(
                        refreshed,
                        f"the {authority} for buffer sink {raw_label!r} claims "
                        f"{'train' if claim else 'eval'}-mode behavior but the "
                        f"capture-time write evidence records buffer_value_changed="
                        f"{value_changed!r} -- a tampered or incoherent mode claim",
                    )
        # Primary key: capture-time write evidence. True = train-mode writer;
        # None = unproven, fail closed. Eval-mode sinks (False) pass.
        written = [row for row in target_rows if row[1] is not False]
        if written:
            names = ", ".join(repr(label) for label, _, _ in written)
            unproven = all(value_changed is None for _, value_changed, _ in written)
            reason = (
                "carries unproven buffer write evidence (buffer_value_changed=None, fail closed)"
                if unproven
                else "recorded value-changing buffer writes (train-mode buffer writers)"
            )
            raise self._buffer_sink_routing_error(
                refreshed,
                f"refresh target {reason} on buffer sink op(s) {names}, "
                "whose live routing can change across reruns",
            )
        # O1 refresh write tripwire: on the newly-allowed no-write path the
        # refreshed rerun's OWN journal must also record no value-changing
        # buffer write -- independent fresh evidence a tampered stored bit
        # cannot buy a pass against.
        refreshed_rows = self._buffer_sink_evidence(refreshed)
        refreshed_written = [row for row in refreshed_rows if row[1] is not False]
        if refreshed_written:
            names = ", ".join(repr(label) for label, _, _ in refreshed_written)
            raise self._buffer_sink_routing_error(
                refreshed,
                "the refreshed rerun's own buffer-write journal recorded a "
                f"value-changing (or unproven) buffer write on sink op(s) {names} "
                "-- the model's training mode changed between runs or a stored "
                "write-evidence claim was tampered",
            )
        # O2 signature widening, evaluated in the typed arm: any
        # target-vs-refreshed buffer-sink evidence asymmetry refuses typed.
        if target_rows != refreshed_rows:
            raise self._buffer_sink_routing_error(
                refreshed,
                "buffer-sink evidence diverged between the refresh target "
                f"{target_rows!r} and the refreshed rerun {refreshed_rows!r}",
            )

    def _graph_signature_mismatch_detail(self, refreshed: Any) -> str:
        """Describe the first graph-signature fact that changed across reruns.

        Parameters
        ----------
        refreshed
            Fully postprocessed rerun candidate that failed the graph signature.

        Returns
        -------
        str
            Human-readable reason that names the changed graph fact.
        """

        target_layers = self.target.layer_list
        refreshed_layers = refreshed.layer_list
        if len(target_layers) != len(refreshed_layers):
            return (
                f"layer count changed: expected {len(target_layers)}, got {len(refreshed_layers)}"
            )
        for target_layer, refreshed_layer in zip(target_layers, refreshed_layers, strict=True):
            if target_layer._layer_label_raw != refreshed_layer._layer_label_raw:
                return (
                    "raw label order changed: "
                    f"expected {target_layer._layer_label_raw!r}, "
                    f"got {refreshed_layer._layer_label_raw!r}"
                )
            if target_layer.layer_type != refreshed_layer.layer_type:
                return (
                    f"layer_type changed for {target_layer._layer_label_raw!r}: "
                    f"expected {target_layer.layer_type!r}, "
                    f"got {refreshed_layer.layer_type!r}"
                )
            expected_parents = tuple(target_layer.parents)
            actual_parents = tuple(refreshed_layer.parents)
            expected_positions = _normalized_parent_arg_positions(target_layer.parent_arg_positions)
            actual_positions = _normalized_parent_arg_positions(
                refreshed_layer.parent_arg_positions
            )
            if expected_parents != actual_parents or expected_positions != actual_positions:
                changed_fields: list[str] = []
                if expected_parents != actual_parents:
                    changed_fields.append(
                        f"parents expected {expected_parents!r}, got {actual_parents!r}"
                    )
                if expected_positions != actual_positions:
                    changed_fields.append(
                        "parent_arg_positions expected "
                        f"{_format_parent_arg_positions(expected_positions)}, got "
                        f"{_format_parent_arg_positions(actual_positions)}"
                    )
                return (
                    f"operand routing changed for {target_layer._layer_label_raw!r}: "
                    + "; ".join(changed_fields)
                )
        return "graph signature changed"

    def _rebind_backward_hooks(self, raw_labels: frozenset[str] | None = None) -> None:
        """Bind refreshed live tensors and grad-fn registry entries to the target Trace.

        ``raw_labels`` restricts the rebind to a truncated run's executed prefix:
        skipped ops carry only stale capture-time grad-fn handles and cleared
        payloads, so registering them would bind dead autograd state.
        """

        from ..backends.torch.backward import _register_forward_grad_fn
        from ..backends.torch.tensor_tracking import _add_tensor_backward_hook

        self.target.__dict__["_tl_backward_hooked_tensor_keys"] = set()
        self.target.__dict__["_tl_grad_hook_owner_by_label"] = {}
        for layer in self.target.layer_list:
            if raw_labels is not None and layer._layer_label_raw not in raw_labels:
                continue
            _register_forward_grad_fn(
                self.target,
                layer.grad_fn_handle,
                layer._layer_label_raw,
            )
            if layer.out is not None:
                _add_tensor_backward_hook(self.target, layer.out, layer._layer_label_raw)

    def _separate_output_payloads(self) -> None:
        """Settle refreshed output-node payloads against their producer's payload.

        HONESTY 13-R1 (same rule as plain capture, ``graph_traversal``): when
        the refreshed output value IS the producer's retained payload, the
        output pseudo-row rides that ONE payload; a differing value keeps its
        own physical copy, so a refresh never hides a variation and never
        re-acquires the redundant duplicate the byte model charged.
        """

        from ..postprocess.graph_traversal import output_payload_aliases_parent
        from ..utils.tensor_utils import safe_copy

        for output_label in self.target.output_layers:
            output = self.target.layer_dict_all_keys[output_label]
            if not output.parents or output.out is None:
                continue
            parent_out = self.target.layer_dict_all_keys[output.parents[0]].out
            if output_payload_aliases_parent(output.out, parent_out):
                output._internal_set("out", parent_out)
                continue
            output._internal_set(
                "out",
                safe_copy(
                    output.out,
                    detach_tensor=self.target.detach_saved_activations,
                ),
            )

    @staticmethod
    def _clear_payload(layer: Any) -> None:
        """Clear refresh payload fields for one unselected operation.

        Parameters
        ----------
        layer
            Existing operation whose refreshed payload was not requested.
        """

        layer._internal_set("out", None)
        layer._internal_set("transformed_out", None)
        layer.transformed_out_shape = None
        layer.transformed_out_dtype = None
        layer.transformed_activation_memory = None
        layer.has_saved_activation = False
        layer.has_out_variations = False
        layer.out_versions_by_child = {}

    @staticmethod
    def _graph_signature(trace: Any) -> tuple[tuple[Any, ...], ...]:
        """Return the refresh graph-change tripwire signature for a Trace.

        Parameters
        ----------
        trace
            Completed Trace whose operation topology should be summarized.

        Returns
        -------
        tuple[tuple[Any, ...], ...]
            Ordered raw identity, type, and parent facts for every operation.
        """

        return tuple(
            (
                layer._layer_label_raw,
                layer.layer_type,
                tuple(layer.parents),
                _normalized_parent_arg_positions(layer.parent_arg_positions),
            )
            for layer in trace.layer_list
        )


@dataclass(frozen=True, slots=True)
class RecordingProjection:
    """Sparse activation records and lookup indexes projected from sealed facts."""

    records: tuple[ActivationRecord, ...]
    by_pass: dict[int, list[int]]
    by_label: dict[str, list[tuple[int, int]]]
    by_address: dict[str, list[int]]
    events: tuple[OpEvent, ...]
    capture_events: object | None
    output_tensors: tuple[object, ...]
    output_tensor_addresses: tuple[str, ...]
    output_labels: tuple[str | None, ...]
    trace_facts: dict[str, object]
    buffer_layers: tuple[str, ...]
    internal_source_ops: tuple[str, ...]

    def prepare_trace(self, trace: Any) -> None:
        """Apply sealed run facts required before Trace postprocessing.

        Parameters
        ----------
        trace
            Fresh Trace projection shell to initialize from core facts.
        """

        trace.buffer_layers = list(self.buffer_layers)
        trace.internal_source_ops = list(self.internal_source_ops)
        trace.capture_start_time = self.trace_facts["capture_start_time"]
        trace.setup_duration = self.trace_facts["setup_duration"]
        trace.forward_duration = self.trace_facts["forward_duration"]
        trace.forward_peak_memory = self.trace_facts["forward_peak_memory"]
        trace.forward_memory_backend = self.trace_facts["forward_memory_backend"]
        trace._source_model_ref = self.trace_facts["source_model_ref"]
        trace.random_seed = self.trace_facts["random_seed"]
        trace._raw_graph_ws.layer_counter = cast(int, self.trace_facts["layer_counter"])

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        for tensor, label in zip(self.output_tensors, self.output_labels, strict=False):
            if label is not None and get_tensor_label(tensor) is None:
                set_tensor_label(tensor, label)

    def bind_halt_frontier(self, tensor: object, label: str) -> None:
        """Restore a core-attributed halt-frontier label before projection.

        Parameters
        ----------
        tensor
            Retained halt-frontier tensor selected from projected records.
        label
            Authoritative raw event label for that tensor.
        """

        from ..backends.torch._tl import get_tensor_label, set_tensor_label

        if get_tensor_label(tensor) is None:
            set_tensor_label(tensor, label)


class RecordingProjector:
    """Build Recording activation indexes directly from sealed run cores."""

    def project(self, cores: Iterable[CapturedRunCore]) -> RecordingProjection:
        """Build the sparse activation projection without first building a Trace.

        Parameters
        ----------
        cores
            Sealed cores in Recorder pass order.

        Returns
        -------
        RecordingProjection
            Retained records and the current public lookup indexes.
        """

        from .projections import activation_record_from_event

        records: list[ActivationRecord] = []
        by_pass: dict[int, list[int]] = {}
        by_label: dict[str, list[tuple[int, int]]] = {}
        by_address: dict[str, list[int]] = {}
        all_events: list[OpEvent] = []
        captured_cores = tuple(cores)
        for core in captured_cores:
            core_events = _cloned_core_events(core)
            all_events.extend(core_events)
            stored_records = core.projection_facts.get("records", ())
            if stored_records:
                projected_records = stored_records
            else:
                projected_records = tuple(
                    record
                    for event in core_events
                    if (record := activation_record_from_event(event)) is not None
                )
            for record in projected_records:
                index = len(records)
                records.append(record)
                by_pass.setdefault(record.ctx.pass_index, []).append(index)
                for label_key in _distinct_label_index_keys(record.ctx.label, record.ctx.raw_label):
                    by_label.setdefault(label_key, []).append((record.ctx.pass_index, index))
                if record.ctx.address is not None:
                    by_address.setdefault(record.ctx.address, []).append(index)
        last_facts = captured_cores[-1].projection_facts if captured_cores else {}
        selected = next(
            (
                (core, core.projection_facts.get("capture_events"))
                for core in captured_cores
                if core.projection_facts.get("capture_events") is not None
            ),
            None,
        )
        capture_events: Any = None
        if selected is not None and selected[1] is not None:
            source_core, capture_events = selected
            # The projection clone was made BEFORE the capture-end seal, so
            # its watermark normally arrives via the seal's dual-stamp; the
            # sealed core's own watermark is the authoritative fallback
            # (reviewer note S-N2 — either transport alone suffices, both are
            # kept so a future re-ordering of snapshot/seal cannot silently
            # strand the filter anchor).
            if (
                getattr(capture_events, "core_seal_watermark", None) is None
                and source_core.amendment_watermark is not None
            ):
                capture_events.core_seal_watermark = source_core.amendment_watermark
            capture_events = capture_events.copy_for_replay(projected_op_events=all_events)
            capture_events.raw_layer_counter = max(
                (event.raw_index for event in all_events), default=0
            )
        return RecordingProjection(
            tuple(records),
            by_pass,
            by_label,
            by_address,
            tuple(all_events),
            capture_events,
            tuple(last_facts.get("output_tensors", ())),
            tuple(last_facts.get("output_tensor_addresses", ())),
            tuple(last_facts.get("output_labels", ())),
            dict(last_facts),
            tuple(event.label_raw for event in all_events if event.layer_type == "buffer"),
            tuple(
                event.label_raw
                for event in all_events
                if event.layer_type != "input" and not event.parents
            ),
        )
