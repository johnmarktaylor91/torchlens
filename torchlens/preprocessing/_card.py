"""The on-request preprocessing card check.

:func:`check_card` compares the preprocessing a capture RECORDED
(``Trace.input_preprocessor``) with the model's known card -- its HF image
processor, torchvision weights preset, timm config or open_clip config, or an
authority the caller names -- and warns when they disagree. Nothing in
TorchLens calls it automatically: it runs only when asked, matching the
opt-in tier of :func:`torchlens.preprocessing.diagnose`.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from ._audit import VERDICT_MISMATCH, PreprocessingAudit, audit
from ._records import COMPARABLE_FIELDS, DeclaredPreprocessing, Resolution

if TYPE_CHECKING:
    from torchlens._data_classes.trace import ResolvedPreprocessing


def check_card(
    trace_or_record: Any,
    *,
    model: Any = None,
    authority: Any = None,
    offline: bool | None = None,
) -> PreprocessingAudit:
    """Compare recorded preprocessing with the model's card; warn on mismatch.

    Parameters
    ----------
    trace_or_record:
        A ``Trace`` (its ``input_preprocessor`` is read) or a
        ``ResolvedPreprocessing`` record.
    model:
        Model whose card is the reference, resolved through
        :func:`torchlens.preprocessing.resolve` (``model=``). Defaults to the
        trace's live source model when ``authority`` is not given.
    authority:
        Explicit card instead of the model's own: a weights preset, image
        processor, data config, preprocess ``Compose`` or declaration
        mapping, or an already resolved :class:`Resolution`.
    offline:
        Fetch policy for the HF tier of model discovery (see ``resolve``).

    Returns
    -------
    PreprocessingAudit
        The field-level report. ``applied_source`` is ``"recorded"`` when the
        record declares fields; an opaque or missing record compares as
        ``unknown``, never as a match.

    Raises
    ------
    InvalidArgumentError
        ``preprocessing_card_unresolved`` when neither ``authority`` nor a
        model (given or live on the trace) is available.

    Warns
    -----
    TorchLensWarning
        ``preprocessing_card_mismatch`` when any compared field disagrees.
    """

    from ._authorities import resolve

    record = _record_of(trace_or_record)
    if authority is not None:
        card = authority if isinstance(authority, Resolution) else resolve(authority)
    else:
        source = model if model is not None else _live_model(trace_or_record)
        if source is None:
            from torchlens._errors import InvalidArgumentError

            raise InvalidArgumentError(
                "check_card needs a card to compare against: no authority= was "
                "given and the trace has no live source model.",
                code="preprocessing_card_unresolved",
                remedy="pass model=<the model> or authority=<its preprocessing object>",
            )
        card = resolve(model=source, offline=offline)
    report = audit(card, declared_from_record(record))
    report = _relabel_recorded(report, record)
    if report.verdict == VERDICT_MISMATCH:
        from torchlens.errors import TorchLensWarning

        rows = "; ".join(
            f"{name}: card={finding.authority_value!r} recorded={finding.applied_value!r}"
            for name in report.mismatched_fields
            for finding in report.findings
            if finding.field == name
        )
        warnings.warn(
            TorchLensWarning(
                f"recorded preprocessing does not match the model's card "
                f"({card.preprocessing.identifier}): {rows}",
                code="preprocessing_card_mismatch",
            ),
            stacklevel=2,
        )
    return report


def declared_from_record(record: ResolvedPreprocessing | None) -> DeclaredPreprocessing | None:
    """Rebuild the declared fields a provenance record carries, if any.

    Parameters
    ----------
    record:
        A ``ResolvedPreprocessing`` record, or ``None``.

    Returns
    -------
    DeclaredPreprocessing | None
        The comparable fields the record's config declares, or ``None`` when
        the record is missing or declares none (opaque transforms, unknown).
    """

    config = getattr(record, "config", None)
    if not isinstance(config, dict):
        return None
    fields: dict[str, Any] = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in config.items()
        if name in COMPARABLE_FIELDS and value is not None
    }
    if not fields:
        return None
    return DeclaredPreprocessing(modality=str(config.get("modality", "image")), **fields)


def _record_of(trace_or_record: Any) -> ResolvedPreprocessing | None:
    """Return the provenance record of a trace, or the record itself."""

    from torchlens._data_classes.trace import ResolvedPreprocessing

    if isinstance(trace_or_record, ResolvedPreprocessing):
        return trace_or_record
    return getattr(trace_or_record, "input_preprocessor", None)


def _live_model(trace_or_record: Any) -> Any:
    """Return the trace's live source model, or ``None``."""

    ref = getattr(trace_or_record, "_source_model_ref", None)
    return ref() if callable(ref) else None


def _relabel_recorded(
    report: PreprocessingAudit, record: ResolvedPreprocessing | None
) -> PreprocessingAudit:
    """Name the applied side ``recorded`` when the record declared fields."""

    from dataclasses import replace

    if declared_from_record(record) is None:
        return report
    return replace(report, applied_source="recorded")


__all__ = ["check_card", "declared_from_record"]
