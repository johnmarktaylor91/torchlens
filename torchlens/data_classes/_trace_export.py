"""Trace export mixin."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    import pandas as pd

    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from ..constants import LAYER_PASS_LOG_FIELD_ORDER
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
    component = components[-1]
    for attr_name in ("index", "key", "name"):
        if hasattr(component, attr_name):
            return str(getattr(component, attr_name))
    return str(component)


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
            raise ValueError("batch_items count must be >= 0.")
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
        "interventions",
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
        "input_ops",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        # Bulky state snapshots and internal viz paths:
        "func_rng_states",
        "func_autocast_state",
        "visualizer_path",
    }
)


class TraceExportMixin(_TraceMixinBase):
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
        """
        if not self._tracing_finished:
            raise RuntimeError(
                "to_pandas() cannot be called before the forward pass is complete. "
                "Please wait until trace has returned."
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
            "layer_label_short",
            "label_short",
            "layer_type",
        ]
        property_columns = [
            "has_parents",
            "siblings",
            "has_siblings",
            "co_parents",
            "has_co_parents",
            "uses_params",
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

        fields_to_change_type = {
            "type_index": int,
            "step_index": int,
            "num_passes": int,
            "pass_index": int,
            "is_inplace": bool,
            "is_input": bool,
            "is_output": bool,
            "is_buffer": bool,
            "in_multi_output": bool,
            "has_parents": bool,
            "has_children": bool,
            "has_siblings": bool,
            "has_co_parents": bool,
            "uses_params": bool,
            "num_params": int,
            "param_memory": int,
            "activation_memory": int,
            "is_module_input": bool,
            "is_module_output": bool,
            "conditional_branch_depth": int,
            "is_terminal_conditional_bool": bool,
        }

        model_df_dictlist: list[dict[str, Any]] = []
        for layer_entry in self.layer_list:
            layer_dict: dict[str, Any] = {}
            for field_name in fields_for_df:
                if field_name == "conditional_branch_stack":
                    layer_dict[field_name] = _format_conditional_branch_stack(
                        layer_entry.conditional_branch_stack
                    )
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

        for column_name in fields_to_change_type:
            model_df[column_name] = model_df[column_name].astype(fields_to_change_type[column_name])
        model_df["terminal_conditional_id"] = model_df["terminal_conditional_id"].astype("Int64")

        return model_df

    def decode_output(self: "Trace", top_n: int | None = None) -> Any:
        """Return captured decoded output rows when available.

        Parameters
        ----------
        top_n:
            Optional maximum rank to return per batch item.

        Returns
        -------
        Any
            JSON-primitive decoded output captured during tracing.

        Raises
        ------
        ValueError
            If decoded output was not captured or the requested ``top_n`` exceeds
            the capture-time bound.
        """

        if self.decoded_output is None:
            recomputed = self._recompute_decoded_output(top_n=top_n)
            if recomputed is None:
                raise ValueError(
                    "logits not retained; re-decode unavailable. Capture with semantic "
                    "output decoding enabled and retained logits to recompute."
                )
            return recomputed
        if top_n is None:
            return self.decoded_output
        captured_top_n = getattr(self.output_postprocessor, "top_n_captured", None)
        if captured_top_n is not None and top_n > captured_top_n:
            recomputed = self._recompute_decoded_output(top_n=top_n)
            if recomputed is not None:
                return recomputed
            raise ValueError(
                f"logits not retained; re-decode unavailable above captured top_n={captured_top_n}."
            )
        if _is_batch_topk_decoded(self.decoded_output):
            return {
                "kind": "batch_topk",
                "rows": [
                    row
                    for row in self.decoded_output["rows"]
                    if isinstance(row, dict) and int(row.get("rank", 1)) <= top_n
                ],
            }
        if not isinstance(self.decoded_output, list):
            return self.decoded_output
        return [
            row
            for row in self.decoded_output
            if not isinstance(row, dict) or int(row.get("rank", 1)) <= top_n
        ]

    def output_table(
        self: "Trace", top_n: int = 5, batch_items: int | Sequence[int] | None = None
    ) -> "pd.DataFrame":
        """Return decoded output as a batch top-k dataframe.

        Parameters
        ----------
        top_n:
            Maximum number of class rows per batch item.
        batch_items:
            ``None`` returns all captured batch items, an integer returns the
            first ``N`` batch items, and a sequence selects explicit batch item
            indices.

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

        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        decoded = self.decode_output(top_n=top_n)
        rows = _decoded_batch_topk_rows(decoded)
        if rows is None:
            raise ValueError("decoded output is not a batch top-k classification table.")
        selected_items = _normalize_batch_items(batch_items, rows)
        filtered_rows = [
            {
                "batch_item": int(row["batch_item"]),
                "rank": int(row["rank"]),
                "label": str(row["label"]),
                "prob": float(row["prob"]),
            }
            for row in rows
            if int(row.get("rank", 0)) <= top_n
            and (selected_items is None or int(row.get("batch_item", -1)) in selected_items)
        ]
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

        postprocessor = getattr(self, "output_postprocessor", None)
        if postprocessor is None or getattr(postprocessor, "style", None) == "hf_text":
            return None
        logits = _retained_output_logits(self)
        if logits is None:
            return None
        from ..autoroute._builtin_output import _decode_classification, _labels_for_resolved

        labels = _labels_for_resolved(postprocessor)
        if labels is None:
            return None
        rows = _decode_classification(logits, labels, top_n=top_n or 5)
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

        rows = _decoded_batch_topk_rows(getattr(self, "decoded_output", None))
        if not rows:
            return None
        first_item = int(rows[0].get("batch_item", 0))
        item_rows = [row for row in rows if int(row.get("batch_item", -1)) == first_item][:3]
        return "; ".join(
            f"{int(row.get('rank', 0))}. {row.get('label')} ({float(row.get('prob', 0.0)):.1%})"
            for row in item_rows
        )
