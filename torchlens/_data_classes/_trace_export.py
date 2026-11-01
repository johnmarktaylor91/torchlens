"""Trace export mixin."""

from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    import pandas as pd

    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._errors import CaptureNotActiveError, InvalidArgumentError, PayloadUnavailableError
from .._utils.collections import _path_component_key
from .interface import _format_conditional_branch_stack


def _output_role_from_container_path(path: Any) -> str | None:
    """Return the final output-container role for a dataframe row.

    Parameters
    ----------
    path:
        Captured typed container path.

    Returns
    -------
    str | None
        Last key/index/field component, or ``None`` for non-container rows.
    """

    components = tuple(path or ())
    if not components:
        return None
    return str(_path_component_key(components[-1]))


def _is_batch_topk_decoded(value: Any) -> bool:
    """Return whether ``value`` is the typed decoded batch top-k representation.

    Parameters
    ----------
    value:
        Candidate decoded output value.

    Returns
    -------
    bool
        Whether the value has ``{"kind": "batch_topk", "rows": [...]}`` shape.
    """

    return (
        isinstance(value, Mapping)
        and value.get("kind") == "batch_topk"
        and isinstance(value.get("rows"), list)
    )


def _decoded_batch_topk_rows(value: Any) -> list[dict[str, Any]] | None:
    """Return decoded batch top-k rows from typed or legacy representations.

    Parameters
    ----------
    value:
        Decoded output value.

    Returns
    -------
    list[dict[str, Any]] | None
        Batch top-k rows, or ``None`` when the decoded value is another kind.
    """

    if _is_batch_topk_decoded(value):
        rows = value["rows"]
    elif isinstance(value, list):
        rows = value
    else:
        return None
    if all(
        isinstance(row, dict) and {"batch_item", "rank", "label", "prob"} <= set(row)
        for row in rows
    ):
        return cast(list[dict[str, Any]], rows)
    return None


def _normalize_batch_items(
    batch_items: int | Sequence[int] | None, rows: Sequence[Mapping[str, Any]]
) -> set[int] | None:
    """Normalize an output-table batch item selector.

    Parameters
    ----------
    batch_items:
        ``None``, a count, or explicit batch item indices.
    rows:
        Decoded rows used to infer available item indices.

    Returns
    -------
    set[int] | None
        Selected batch item indices, or ``None`` for all.
    """

    if batch_items is None:
        return None
    available = sorted({int(row.get("batch_item", 0)) for row in rows})
    if isinstance(batch_items, int):
        if batch_items < 0:
            raise InvalidArgumentError(
                f"batch_items count must be >= 0; received {batch_items!r}",
                code="batch_items_invalid",
                remedy="pass a non-negative integer batch_items count",
                argument="batch_items",
            )
        return set(available[:batch_items])
    return {int(item) for item in batch_items}


def _retained_output_logits(trace: "Trace") -> torch.Tensor | None:
    """Return a retained output tensor suitable for classification re-decode.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    torch.Tensor | None
        Retained logits tensor, if exactly one output tensor is available.
    """

    tensors = [
        getattr(op, "out", None)
        for op in getattr(trace, "output_ops", [])
        if isinstance(getattr(op, "out", None), torch.Tensor)
    ]
    if len(tensors) != 1:
        return None
    return tensors[0]


# Op fields deliberately omitted from ``Trace.to_pandas()`` columns. Every
# field in ``LAYER_PASS_LOG_FIELD_ORDER`` must either appear as a dataframe
# column or be listed here -- ``tests/test_io_pandas.py`` enforces this so new
# Op fields can never silently fail to reach the layer-pass table again
# (regression gate for TO-PANDAS-NEW-FIELDS).
_TO_PANDAS_EXCLUDED_OP_FIELDS: frozenset[str] = frozenset(
    {
        # Internal / private bookkeeping (not part of the user-facing table):
        "_label_raw",
        "_layer_label_raw",
        "_tracing_finished",
        "_construction_done",
        "_param_barcodes",
        "_param_logs",
        "_edge_uses",
        "_address_normalized",
        "source_trace",
        # Tensor payloads (bulk data; access via ``layer.out`` / ``layer.grad``):
        "out",
        "transformed_out",
        "transformed_grad",
        "input_activations",
        "saved_args",
        "saved_kwargs",
        "out_versions_by_child",
        # Live objects, callables, and rich records (not scalar table cells):
        "input_ops",  # live accessor over parent Op records; needs an attached
        # source trace and raises on disk-loaded logs -- the `parents` label
        # column already carries the relationship.
        "func",
        "grad_fn",
        "grad_fn_handle",
        "grad_fn_object_id",
        "activation_transform",
        "edits",
        "code_context",
        "transform_fn_source",
        "args_template",
        "kwargs_template",
        "container_spec",
        "parent_param_ops",
        "atomic_module_call",
        "module_entry_arg_keys",
        "annotations",
        # Live-trace-derived views (raise on detached/rehydrated logs; the
        # ``parents`` column plus parent rows carry the same information):
        "input_shapes",
        "input_dtypes",
        "input_memory",
        # Bulky state snapshots and internal viz paths:
        "func_rng_states",
        "func_autocast_state",
        "visualizer_path",
    }
)


def _decode_door() -> ModuleType:
    """Return the output-decode module (the one lazy seam into ``_autoroute``).

    ``_autoroute`` sits above this layer, so the import is deferred to the first
    decode and shared by every decode path of the mixin.
    """

    from .._autoroute import _builtin_output

    return _builtin_output


def _recompute_or_refuse(trace: "Trace", k: int | None) -> Any:
    """Recompute the decoded output when none was captured, or refuse when nothing is retained.

    Parameters
    ----------
    trace:
        Trace whose ``decoded_output`` is ``None``.
    k:
        Requested maximum rank per batch item, or ``None``.

    Returns
    -------
    Any
        The recomputed decoded output.

    Raises
    ------
    PayloadUnavailableError
        ``decoded_output_unavailable`` when no output tensor is retained.
    """

    recomputed = trace._recompute_decoded_output(top_n=k)
    if recomputed is None:
        raise PayloadUnavailableError(
            "no decoded output was captured and no output tensor is retained",
            code="decoded_output_unavailable",
            remedy=(
                "decode on demand with decode_output(style=..., head=...) while the "
                "output activations are retained (the default save policy)"
            ),
        )
    return recomputed


def _recompute_above_captured(trace: "Trace", k: int, captured_top_n: int) -> Any:
    """Recompute decoded output for a ``k`` above the capture-time bound, or refuse.

    Parameters
    ----------
    trace:
        Trace with captured decoded output.
    k:
        Requested maximum rank per batch item.
    captured_top_n:
        The capture-time bound ``k`` exceeds.

    Returns
    -------
    Any
        The recomputed decoded output.

    Raises
    ------
    PayloadUnavailableError
        ``decoded_output_unavailable`` when the logits were not retained.
    """

    recomputed = trace._recompute_decoded_output(top_n=k)
    if recomputed is not None:
        return recomputed
    raise PayloadUnavailableError(
        f"logits not retained; re-decode unavailable above captured top_n={captured_top_n}",
        code="decoded_output_unavailable",
        remedy=(
            f"request top_n <= {captured_top_n}, or keep the output activations "
            "retained to recompute"
        ),
        captured_top_n=captured_top_n,
    )


def _rank_filtered(decoded: Any, k: int) -> Any:
    """Return captured decoded output keeping only rows ranked ``k`` or better.

    Parameters
    ----------
    decoded:
        Captured decoded output: a batch top-k mapping, a row list, or any
        other value (returned unchanged).
    k:
        Maximum rank to keep per batch item.

    Returns
    -------
    Any
        The filtered decoded output, in the shape it was captured.
    """

    if _is_batch_topk_decoded(decoded):
        return {
            "kind": "batch_topk",
            "rows": [
                row
                for row in decoded["rows"]
                if isinstance(row, dict) and int(row.get("rank", 1)) <= k
            ],
        }
    if not isinstance(decoded, list):
        return decoded
    return [row for row in decoded if not isinstance(row, dict) or int(row.get("rank", 1)) <= k]


def _filtered_topk_rows(
    rows: list[dict[str, Any]], k: int, selected_items: set[int] | None
) -> list[dict[str, Any]]:
    """Return typed ``output_table`` rows ranked ``k`` or better for the selected items.

    Parameters
    ----------
    rows:
        Batch top-k decoded rows.
    k:
        Maximum rank to keep per batch item.
    selected_items:
        Batch item indices to keep, or ``None`` for all.

    Returns
    -------
    list[dict[str, Any]]
        Rows with ``batch_item``, ``rank``, ``label`` and ``prob``.
    """

    return [
        {
            "batch_item": int(row["batch_item"]),
            "rank": int(row["rank"]),
            "label": str(row["label"]),
            "prob": float(row["prob"]),
        }
        for row in rows
        if int(row.get("rank", 0)) <= k
        and (selected_items is None or int(row.get("batch_item", -1)) in selected_items)
    ]


_TO_PANDAS_COLUMN_TYPES: dict[str, type] = {
    "type_index": int,
    "step_index": int,
    "num_passes": int,
    "pass_number": int,
    "is_inplace": bool,
    "is_input": bool,
    "is_output": bool,
    "is_buffer": bool,
    "in_multi_output": bool,
    "has_parents": bool,
    "has_children": bool,
    "has_siblings": bool,
    "has_co_parents": bool,
    "has_params": bool,
    "num_params": int,
    "param_memory": int,
    "activation_memory": int,
    "is_module_input": bool,
    "is_module_output": bool,
    "conditional_branch_depth": int,
    "is_terminal_conditional_bool": bool,
}


class TraceExportMixin(_TraceMixinBase):
    """``Trace`` export surface: dataframe, dict, and tabular projections."""

    @property
    def logged_values(self: "Trace") -> dict[str, Any]:
        """Return values recorded via ``log_value`` during this capture.

        DOCUMENTED-UNSTABLE spelling (naming ratification pending). This is
        the read-back for ``torchlens.observers.log_value``, which was a
        writer nothing read back: values land in
        ``annotations["logged_values"]`` and persist with the trace.
        Returns a copy -- write through ``log_value`` during capture, never by
        mutating this view.

        Returns
        -------
        dict[str, Any]
            Name-to-value mapping of capture-time logged values (empty when
            none were recorded).
        """

        values = (getattr(self, "annotations", {}) or {}).get("logged_values", {})
        return dict(values) if isinstance(values, dict) else {}

    def to_agent_json(self: "Trace", *, max_ops: int | None = None) -> dict[str, Any]:
        """Return a self-describing machine-readable dump of this trace.

        DOCUMENTED-UNSTABLE spelling (naming ratification pending). The dump
        is a navigation aid over the SAME public surface a human drives: it
        embeds a ``guide`` block mapping every record back to the live
        spelling to call next (``trace[label]``, ``trace.explain``, ...),
        and carries the capture outcome/verification honesty facts verbatim.
        Tensor payloads are never inlined.

        Parameters
        ----------
        max_ops:
            Optional cap on emitted op rows (execution order, first ``max_ops``
            kept). Any omission is disclosed in the ``truncation`` block.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dump under the ``torchlens.agent_trace.v1``
            schema.
        """

        from ..report._agent_json import build_agent_json

        return build_agent_json(self, max_ops=max_ops)

    def to_pandas(self: "Trace", include_decoded_output_summary: bool = False) -> "pd.DataFrame":
        """Return a dataframe containing one row per layer pass.

        Parameters
        ----------
        include_decoded_output_summary:
            If ``True``, add a gated ``decoded_output_summary`` column populated
            only on output-node rows. The default remains schema-stable.

        Returns
        -------
        pd.DataFrame
            Layer-pass table for this model log.

        Raises
        ------
        RuntimeError
            If called before the forward pass has completed.
        ValueError
            If an exported field such as ``grad`` is ambiguous across multiple
            backward passes and must be accessed through a pass-specific record.
        """
        from .._runnable import refuse_poisoned_trace

        refuse_poisoned_trace(self, "tabular export")
        if not self._tracing_finished:
            raise CaptureNotActiveError(
                "to_pandas() cannot be called before the forward pass is complete",
                code="trace_not_finished",
                remedy="wait until trace(...) has returned before exporting",
                operation="to_pandas",
            )
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with "
                "`pip install torchlens[tabular]`."
            ) from e

        # Identity columns lead the table; the remaining columns follow the
        # canonical Op field order minus the documented exclusions, plus the
        # handful of property-backed convenience columns that are not stored
        # fields. ``dict.fromkeys`` removes duplicates while keeping the lead
        # block's positions.
        lead_columns = [
            "layer_label",
            "label",
            "short_layer_label",
            "short_label",
            "layer_type",
        ]
        property_columns = [
            "has_parents",
            "siblings",
            "has_siblings",
            "co_parents",
            "has_co_parents",
            "has_params",
            "is_module_input",
            "output_role",
            "num_saved_grads",
        ]
        if include_decoded_output_summary:
            property_columns.append("decoded_output_summary")
        fields_for_df = list(
            dict.fromkeys(
                lead_columns
                + [
                    field_name
                    for field_name in LAYER_PASS_LOG_FIELD_ORDER
                    if field_name not in _TO_PANDAS_EXCLUDED_OP_FIELDS
                ]
                + property_columns
            )
        )

        model_df_dictlist: list[dict[str, Any]] = []
        for layer_entry in self._layer_list:
            layer_dict: dict[str, Any] = {}
            for field_name in fields_for_df:
                if field_name == "conditional_branch_stack":
                    layer_dict[field_name] = _format_conditional_branch_stack(
                        layer_entry.conditional_branch_stack
                    )
                elif field_name == "layer_type":
                    # The table keeps its layer_type column; an Op names it type.
                    layer_dict[field_name] = layer_entry.type
                elif field_name == "grad":
                    layer_dict[field_name] = getattr(layer_entry, "grad")
                elif field_name == "num_saved_grads":
                    layer_dict[field_name] = len(getattr(layer_entry, "_grad_records", ()))
                elif field_name == "output_role":
                    layer_dict[field_name] = _output_role_from_container_path(
                        getattr(layer_entry, "container_path", ())
                    )
                elif field_name == "decoded_output_summary":
                    layer_dict[field_name] = (
                        self._decoded_output_summary()
                        if bool(getattr(layer_entry, "is_output", False))
                        else None
                    )
                else:
                    layer_dict[field_name] = getattr(layer_entry, field_name)
            model_df_dictlist.append(layer_dict)
        model_df = pd.DataFrame(model_df_dictlist)

        for column_name, column_type in _TO_PANDAS_COLUMN_TYPES.items():
            model_df[column_name] = model_df[column_name].astype(column_type)
        model_df["terminal_conditional_id"] = model_df["terminal_conditional_id"].astype("Int64")

        from .._capture_honesty import attach_dataframe_honesty

        return attach_dataframe_honesty(model_df, self)

    def decode_output(
        self: "Trace",
        k: int | None = None,
        *,
        style: str | None = None,
        head: str | None = None,
        tokenizer: Any | None = None,
    ) -> Any:
        """Return decoded output rows: the captured rows, or a decode on demand.

        With no keywords this returns the rows the automatic capture-time
        decode produced. Naming a ``style``, a ``head`` or a ``tokenizer``
        decodes again from the retained output tensors (the output ops'
        values), the persisted ``output_id2label`` and the shipped ImageNet
        bank, without touching ``decoded_output``.

        Parameters
        ----------
        k:
            Optional maximum rank to return per batch item.
        style:
            Decode style: ``"classification"``, ``"imagenet"``,
            ``"imagenet1k"``, ``"hf_text"`` or a registered custom style.
            ``None`` lets the registered detectors pick one.
        head:
            Output head to decode, as an output container path such as
            ``"logits"``, ``"0"`` or ``"0.logits"``.
        tokenizer:
            Tokenizer with a ``decode`` method, required by ``"hf_text"``.

        Returns
        -------
        Any
            JSON-primitive decoded output.

        Raises
        ------
        ValueError
            If decoded output was not captured or the requested ``k`` exceeds
            the capture-time bound (``decoded_output_unavailable``); a style
            no detector resolves (``decode_style_unknown``); a head that is
            not an output head (``decode_head_unknown``) or an ambiguous head
            (``decode_head_ambiguous``).
        """

        # Capability gate (quickstart memo D7, F17): decoded labels on
        # synthesized input values would be a statement about random numbers
        # presented as a statement about data. One chokepoint for every
        # decode door (output_table, to_pandas decoded summary, direct calls).
        from .._quickstart._gate import require_gold

        require_gold(self, "decoded output labels")
        if style is not None or head is not None or tokenizer is not None:
            door = _decode_door()
            return door.decode_trace_output(
                self, style=style, head=head, tokenizer=tokenizer, top_n=k or door._TOP_N_CAPTURED
            )
        if self._decoded_output is None:
            return _recompute_or_refuse(self, k)
        if k is None:
            return self._decoded_output
        captured_top_n = getattr(self._provenance_output_postprocessor, "top_n_captured", None)
        if captured_top_n is not None and k > captured_top_n:
            return _recompute_above_captured(self, k, captured_top_n)
        return _rank_filtered(self._decoded_output, k)

    def output_table(
        self: "Trace",
        k: int = 5,
        batch_items: int | Sequence[int] | None = None,
        *,
        style: str | None = None,
        head: str | None = None,
    ) -> "pd.DataFrame":
        """Return decoded output as a batch top-k dataframe.

        Parameters
        ----------
        k:
            Maximum number of class rows per batch item.
        batch_items:
            ``None`` returns all captured batch items, an integer returns the
            first ``N`` batch items, and a sequence selects explicit batch item
            indices.
        style:
            Optional decode style for a decode on demand (see ``decode_output``).
        head:
            Optional output head for a decode on demand (see ``decode_output``).

        Returns
        -------
        pd.DataFrame
            Table with columns ``batch_item``, ``rank``, ``label``, and ``prob``.

        Raises
        ------
        ValueError
            If classification logits were not decoded or cannot be recomputed.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        if k < 1:
            raise InvalidArgumentError(
                f"top_n must be >= 1; received {k!r}",
                code="top_n_invalid",
                remedy="pass a positive integer top_n",
                argument="top_n",
            )
        decoded = self.decode_output(k=k, style=style, head=head)
        rows = _decoded_batch_topk_rows(decoded)
        if rows is None:
            raise PayloadUnavailableError(
                "decoded output is not a batch top-k classification table",
                code="decoded_output_not_classification",
                remedy=(
                    "decode on demand with output_table(style='classification') to produce "
                    "a batch top-k table"
                ),
            )
        filtered_rows = _filtered_topk_rows(rows, k, _normalize_batch_items(batch_items, rows))
        return pd.DataFrame(filtered_rows, columns=["batch_item", "rank", "label", "prob"])

    def _recompute_decoded_output(self: "Trace", top_n: int | None) -> Any | None:
        """Best-effort decoded-output recompute from retained output logits.

        Parameters
        ----------
        top_n:
            Requested number of top rows per batch item.

        Returns
        -------
        Any | None
            Typed decoded representation when recompute succeeds, otherwise
            ``None``.
        """

        postprocessor = self._provenance_output_postprocessor
        if postprocessor is None or getattr(postprocessor, "style", None) == "hf_text":
            return None
        logits = _retained_output_logits(self)
        if logits is None:
            return None
        door = _decode_door()
        labels = door._labels_for_resolved(postprocessor)
        if labels is None:
            return None
        rows = door._decode_classification(logits, labels, top_n=top_n or 5)
        if rows is None:
            return None
        return {"kind": "batch_topk", "rows": rows}

    def _decoded_output_summary(self: "Trace") -> str | None:
        """Return a compact decoded-output summary for gated dataframe export.

        Returns
        -------
        str | None
            First decoded row summary, or ``None`` when unavailable.
        """

        rows = _decoded_batch_topk_rows(
            None if self.__dict__.get("_tl_cleaned_up", False) else self._decoded_output
        )
        if not rows:
            return None
        first_item = int(rows[0].get("batch_item", 0))
        item_rows = [row for row in rows if int(row.get("batch_item", -1)) == first_item][:3]
        return "; ".join(
            f"{int(row.get('rank', 0))}. {row.get('label')} ({float(row.get('prob', 0.0)):.1%})"
            for row in item_rows
        )
