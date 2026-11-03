"""Trace export mixin."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from .trace import Trace

    _TraceMixinBase = Trace
else:
    _TraceMixinBase = object
from . import _doors


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

        return _doors.report()._agent_json.build_agent_json(self, max_ops=max_ops)

    def _attach_dataframe_honesty(self: "Trace", frame: Any) -> Any:
        """Stamp ``frame.attrs`` with this trace's capture-honesty facts.

        The one place the records reach ``_capture_honesty`` for tabular exports:
        ``to_pandas`` here and ``_accessor_base.attach_source_honesty`` both call it.
        """

        from .._capture_honesty import attach_dataframe_honesty

        return attach_dataframe_honesty(frame, self)

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

        return _doors.export()._trace_tables.to_pandas(
            self, include_decoded_output_summary=include_decoded_output_summary
        )

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
        decodes again from the retained output tensors, the persisted
        ``output_id2label`` and the ImageNet bank, never ``decoded_output``.

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
            Tokenizer with a ``decode`` method, used by ``"hf_text"``; without
            one the rows carry the argmax ``token_ids`` and ``text`` is ``None``.

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

        return _doors.export()._trace_tables.decode_output(
            self, k=k, style=style, head=head, tokenizer=tokenizer
        )

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

        return _doors.export()._trace_tables.output_table(
            self, k=k, batch_items=batch_items, style=style, head=head
        )
