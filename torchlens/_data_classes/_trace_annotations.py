"""Trace annotation family: user notes and tags on a finished trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from ..debug._audit import HealthReport
    from .trace import Trace

__all__ = [
    "annotate",
    "annotate_in_place",
    "annotation_data_kind",
    "validate_annotation_tensor",
    "validate_annotation_json",
    "annotation_key_for_site",
    "user_annotation_blobs",
    "store_annotation_blob",
    "store_annotation_breadcrumb",
    "user_annotation_dict",
    "mark_annotations_mutated",
    "audit",
    "using",
    "outcome",
]


def annotate(  # noqa: PLR0913 - mirrors Trace.annotate: every axis is a named keyword
    trace: Trace,
    selector: Any,
    *,
    data: Any = None,
    image: str | Path | None = None,
    max_fanout: int = 1_000_000,
    copy: bool = False,
) -> Trace:
    """Attach user-owned annotation data to selected graph nodes.

    Parameters
    ----------
    trace:
        The owning trace.
    selector:
        Selector, target spec, frozen target spec, or non-strict bare string.
    data:
        JSON-serializable metadata or a portable tensor payload. Tensor
        payloads are persisted under ``_annotation_blobs`` and are supported
        only for torch traces.
    image:
        Optional local image path used by the existing ``NodeSpec.image``
        render hook.
    max_fanout:
        Explicit maximum number of selected sites. The default is
        intentionally large so annotation can fan out across many layers.
    copy:
        If ``True``, annotate and return an owned fork instead of mutating
        this trace.

    Returns
    -------
    Trace
        The annotated trace. This is ``self`` unless ``copy=True``.
    """

    target = trace.fork(name=None) if copy else trace
    annotate_in_place(target, selector, data=data, image=image, max_fanout=max_fanout)
    return target


def annotate_in_place(
    trace: Trace,
    selector: Any,
    *,
    data: Any,
    image: str | Path | None,
    max_fanout: int,
) -> None:
    """Apply annotation updates directly to this trace.

    Parameters
    ----------
    trace:
        The owning trace.
    selector:
        Selector resolved against this trace.
    data:
        Annotation data supplied by the caller.
    image:
        Optional image path supplied by the caller.
    max_fanout:
        Explicit fan-out limit passed to the site resolver.

    Returns
    -------
    None
        This trace is mutated in place.
    """

    if data is None and image is None:
        raise InvalidArgumentError(
            "annotate() requires data=, image=, or both",
            code="annotation_payload_missing",
            remedy="pass data=, image=, or both to annotate()",
        )
    sites = trace._resolve_sites(selector, max_fanout=max_fanout)
    image_value = str(image) if image is not None else None
    data_kind = annotation_data_kind(trace, data)
    for site in sites:
        key = annotation_key_for_site(trace, site)
        if data is not None:
            if data_kind == "blob":
                store_annotation_blob(trace, key, data)
            else:
                store_annotation_breadcrumb(trace, site, "data", data)
        if image_value is not None:
            store_annotation_breadcrumb(trace, site, "image", image_value)
    mark_annotations_mutated(trace)


def annotation_data_kind(trace: Trace, data: Any) -> str:
    """Classify and validate annotation data.

    Parameters
    ----------
    trace:
        The owning trace.
    data:
        Candidate annotation payload.

    Returns
    -------
    str
        ``"none"``, ``"blob"``, or ``"json"``.
    """

    if data is None:
        return "none"
    if isinstance(data, torch.Tensor):
        validate_annotation_tensor(trace, data)
        return "blob"
    validate_annotation_json(data)
    return "json"


def validate_annotation_tensor(trace: Trace, tensor: torch.Tensor) -> None:
    """Validate that a tensor annotation matches the active payload codec.

    Parameters
    ----------
    trace:
        The owning trace.
    tensor:
        Tensor annotation candidate.

    Returns
    -------
    None
        Raises if the tensor is not portable for this trace.
    """

    backend_name = str(trace._provenance_backend)
    if backend_name != "torch":
        raise InvalidArgumentError(
            "annotate(data=torch.Tensor) is supported only for torch traces in this "
            f"release; this trace uses backend={backend_name!r}",
            code="annotation_backend_unsupported",
            remedy="pass JSON-serializable annotation data for non-torch traces",
            backend=backend_name,
        )
    from .._io.payload_codec import get_payload_codec

    decision = get_payload_codec(backend_name).validate_for_save(tensor, strict=True)
    if decision.__class__.__name__ != "Ok":
        reason = getattr(decision, "text", "unsupported tensor payload")
        raise InvalidArgumentError(
            f"Annotation tensor is not portable for backend {backend_name!r}: {reason}",
            code="annotation_tensor_not_portable",
            remedy="pass a dense, codec-supported tensor as annotation data",
            backend=backend_name,
            reason=reason,
        )


def validate_annotation_json(data: Any) -> None:
    """Validate that an annotation breadcrumb can be persisted as JSON data.

    Parameters
    ----------
    data:
        Candidate JSON breadcrumb.

    Returns
    -------
    None
        Raises if ``data`` is not JSON-serializable.
    """

    try:
        json.dumps(data, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidArgumentError(
            "annotate(data=...) must be JSON-serializable or a torch.Tensor; "
            "values are never silently stringified",
            code="annotation_not_json_serializable",
            remedy="convert arrays to torch.Tensor for blob persistence",
            argument="data",
        ) from exc


def annotation_key_for_site(trace: Trace, site: Any) -> str:
    """Return the persistent annotation blob key for a resolved site.

    Parameters
    ----------
    trace:
        The owning trace.
    site:
        Resolved layer-pass record.

    Returns
    -------
    str
        ``layer:<layer_label>`` for single-pass layers, otherwise
        ``op:<op.label>``.
    """

    layer_label = str(getattr(site, "layer_label"))
    if trace._layer_num_calls.get(layer_label, 1) == 1:
        return f"layer:{layer_label}"
    return f"op:{getattr(site, 'label')}"


def user_annotation_blobs(trace: Trace) -> dict[str, Any] | None:
    """Return only user-attached annotation blobs.

    Returns
    -------
    dict[str, Any] | None
        Blobs outside every derived render namespace, or ``None`` when no
        user blob exists (matching the unannotated default).
    """

    blobs = trace._annotation_blobs
    if not isinstance(blobs, dict):
        return None
    kept = {
        key: value
        for key, value in blobs.items()
        if not key.startswith(trace._DERIVED_ANNOTATION_BLOB_PREFIXES)
    }
    return kept or None


def store_annotation_blob(trace: Trace, key: str, data: Any) -> None:
    """Store a blob annotation under ``_annotation_blobs``.

    Parameters
    ----------
    trace:
        The owning trace.
    key:
        Namespaced annotation key.
    data:
        Codec-validated payload.

    Returns
    -------
    None
        This trace's blob mapping is mutated in place.
    """

    if trace._annotation_blobs is None:
        trace._annotation_blobs = {}
    trace._annotation_blobs[key] = data


def store_annotation_breadcrumb(trace: Trace, site: Any, name: str, value: Any) -> None:
    """Store a small user breadcrumb on the selected Op and Layer.

    Parameters
    ----------
    trace:
        The owning trace.
    site:
        Resolved layer-pass record.
    name:
        User-namespace field name.
    value:
        JSON-compatible breadcrumb value.

    Returns
    -------
    None
        The selected Op and aggregate Layer annotation dicts are mutated.
    """

    user_annotation_dict(site.annotations)[name] = value
    layer_log = trace._layer_logs.get(str(getattr(site, "layer_label")))
    if layer_log is not None:
        user_annotation_dict(layer_log.annotations)[name] = value


def user_annotation_dict(annotations: dict[str, Any]) -> dict[str, Any]:
    """Return the reserved user annotation namespace.

    Parameters
    ----------
    annotations:
        Op, Layer, or Trace annotation mapping.

    Returns
    -------
    dict[str, Any]
        Mutable ``annotations["user"]`` mapping.
    """

    user_annotations = annotations.setdefault("user", {})
    if not isinstance(user_annotations, dict):
        raise InvalidArgumentError(
            'annotations["user"] must be a dict to store user annotations; '
            f"found {type(user_annotations).__name__}",
            code="annotation_namespace_invalid",
            remedy='restore annotations["user"] to a dict before annotating',
        )
    return user_annotations


def mark_annotations_mutated(trace: Trace) -> None:
    """Invalidate render-only caches after an annotation mutation.

    Returns
    -------
    None
        The cached sibling-ordering decision is discarded.
    """

    trace.__dict__.pop("_last_sibling_ordering_decision", None)
    trace.__dict__.pop("_last_encoding_state", None)
    trace.__dict__.pop("_last_render_geometry", None)


def audit(trace: Trace) -> HealthReport:
    """Return an honest one-call health report for this completed trace.

    Payload-dependent and gradient-dependent diagnostics are listed as
    skipped when this capture cannot support them, rather than being used
    to claim a clean result.

    Returns
    -------
    torchlens.report.HealthReport
        Prioritized findings plus checks run and skipped-check reasons.
    """

    from ..debug._audit import audit_trace

    return audit_trace(trace)


def using(trace: Trace, name: str) -> Any:
    """Return the appliance registered as ``name``, bound to this trace.

    The one accessor door for appliances (DOCUMENTED-UNSTABLE); register with
    ``torchlens.appliances.register``. An unknown name refuses typed
    (``appliance_unknown``) and lists the registered names.
    """

    from ..appliances._appliances import open_appliance

    return open_appliance(trace, name)


def outcome(trace: Trace) -> Any | None:
    """Return the settled capture outcome for this trace, when one exists.

    Returns
    -------
    CaptureOutcome | None
        The settlement authority's typed record: an attested settle stamp
        for live products, an adopted or lattice-derived record for loaded
        artifacts (load derivation writes the sidecar), or ``None`` on a
        live trace that has not settled yet (the documented pre-settlement
        state; capability gates independently treat it as UNKNOWN).
    """

    from .._capture.outcome import CaptureOutcome, CaptureStatus, outcome_for

    if trace.__dict__.get("_tl_cleaned_up", False):
        # cleanup() deletes the settled sidecar with every other field.
        # The frozen vocabulary designates UNKNOWN as the most-restrictive
        # value and the save gate already lands there; ``None`` is outside
        # the vocabulary and was the one unsettled read (b6-opus R25).
        return CaptureOutcome(status=CaptureStatus.UNKNOWN, is_derived=True)
    return outcome_for(trace)
