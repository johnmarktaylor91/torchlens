"""The ``fed_by`` relation kind: content provenance between two traced calls.

"critique fed_by draft" records that the LATER member's prompt contains the
EARLIER member's output. Like every ordering kind the row's ``from`` is the
later member (the reader) and ``to`` the earlier member (the source), so
``ordering_path`` and the upstream/downstream walks read it in the same
direction as ``successor_of``. ``fed_by`` is CONTENT provenance only: a router
whose decision selects a call without its output entering that call's prompt
is chronology (``successor_of``) plus application metadata, never ``fed_by``.

Params (all optional, per-kind closed): ``token_spans`` (rows of four ints:
source start, source end, destination start, destination end; half-open),
``basis`` (``"declared"``, ``"verified"``, ``"matched"``), ``coverage`` (a
fraction of the source's ELIGIBLE tokens, see below) and ``label`` (free
human text that TorchLens never interprets).

BINDING. Four integers alone are not a verifiable claim: they mean nothing
until bound to the token streams they index. A member's binding is a
``token_boundary_v1`` descriptor in the reserved ``torchlens.boundary_facts``
sidecar family on that member (``trace.annotations["sidecar"]``): which
captured stream (``"input"``: the first integer-dtype input tensor;
``"output"``: the first integer-dtype output tensor), the batch row, the row
length, the SHA-256 digest of the row's token ids, the member's own
completion range inside that stream, the token ids outside the eligible set
(padding, special tokens), and the tokenizer and template identity when known.
TorchLens derives the descriptor from retained ids when it can (batch of one,
no completion range, identities unknown); users attach a richer one through
``torchlens.io.attach_sidecar`` before relating.

BASIS. A row reads ``"verified"`` only when TorchLens checked it in this
process: both endpoints bound, each binding's digest equal to the member's
retained ids, every source span inside the source completion range, and the
token ids equal span by span. Otherwise the row is ``"declared"``. A claimed
``"verified"``/``"matched"`` row arriving through any door (relate, the
constructor, an artifact) is re-checked: unverifiable evidence (ids not
retained, no binding) clamps to ``"declared"``; contradicting evidence
refuses typed (``bundle_relation_span_unverified``). ``"matched"`` is written
by the automatic span finder, which is not available yet.

COVERAGE is computed on verified rows whose source binding names its
completion range: the covered fraction of the source's eligible tokens, where
the eligible set is that completion range minus positions whose id is listed
in ``excluded_ids`` (padding, special tokens). Without a bound completion range
the eligible set is unknown and a verified row carries no coverage. On declared
rows a user-supplied coverage is a declaration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from .._io.sidecar import (
    SIDECAR_ANNOTATIONS_KEY,
    SidecarFamily,
    attach_sidecar,
    list_sidecar_families,
    read_sidecar,
    register_sidecar_family,
)
from ..errors.episode import BundleRelationError

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

FED_BY_KIND = "fed_by"
FED_BY_BASES = frozenset({"declared", "verified", "matched"})
#: Bases TorchLens must earn; any door re-checks them against the members.
EARNED_BASES = frozenset({"verified", "matched"})
TOKEN_BOUNDARY_FAMILY_ID = "torchlens.boundary_facts"
TOKEN_BOUNDARY_SCHEMA_ID = "token_boundary_v1"
_TOKEN_STREAMS = frozenset({"input", "output"})
_DESCRIPTOR_REQUIRED = frozenset({"stream", "batch_row", "length", "ids_digest"})
_DESCRIPTOR_OPTIONAL = frozenset({"completion", "excluded_ids", "tokenizer", "template"})
_DIGEST_PREFIX = "sha256:"
_LABEL_MAX_CHARS = 256


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_range(value: Any, *, what: str) -> list[int]:
    """Return a validated half-open ``[lo, hi]`` pair with ``0 <= lo <= hi``."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or not all(_is_int(item) for item in value)
        or not 0 <= value[0] <= value[1]
    ):
        raise ValueError(f"{what} must be a [start, end] pair of ints with 0 <= start <= end")
    return [int(value[0]), int(value[1])]


def normalize_token_spans(value: Any) -> list[list[int]]:
    """Validate ``token_spans`` rows and return them as lists of four ints.

    Raises
    ------
    ValueError
        On ``"auto"`` (a request, never a persisted value), a non-list, or a
        row that is not four non-negative ints with ``start <= end`` per side.
    """

    if value == "auto":
        raise ValueError(
            "token_spans='auto' is a request to Bundle.relate, never a persisted "
            "value; persisted rows are lists of four ints"
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"fed_by 'token_spans' must be a list of four-int rows, got {type(value).__name__}"
        )
    rows: list[list[int]] = []
    for index, row in enumerate(value):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 4
            or not all(_is_int(item) and item >= 0 for item in row)
        ):
            raise ValueError(
                f"fed_by token_spans row {index} must be four non-negative ints "
                f"[src_start, src_end, dst_start, dst_end], got {row!r}"
            )
        if row[0] > row[1] or row[2] > row[3]:
            raise ValueError(f"fed_by token_spans row {index} has start > end: {list(row)!r}")
        rows.append([int(item) for item in row])
    return rows


def validate_fed_by_param(key: str, value: Any) -> Any:
    """Validate one ``fed_by`` param value; return its persisted normal form."""

    if key == "token_spans":
        return normalize_token_spans(value)
    if key == "basis" and value not in FED_BY_BASES:
        raise ValueError(
            f"fed_by 'basis' is {value!r}, outside the closed vocabulary {sorted(FED_BY_BASES)}"
        )
    if key == "coverage" and (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"fed_by 'coverage' must be a number in [0, 1], got {value!r}")
    if key == "label" and (
        not isinstance(value, str) or not value or len(value) > _LABEL_MAX_CHARS
    ):
        raise ValueError(
            f"fed_by 'label' must be a non-empty string of at most {_LABEL_MAX_CHARS} "
            f"characters, got {value!r}"
        )
    return value


def check_fed_by_params(params: Mapping[str, Any]) -> None:
    """Cross-param rule: an earned basis needs at least one span row."""

    if params.get("basis") in EARNED_BASES and not params.get("token_spans"):
        raise ValueError(
            f"fed_by basis {params['basis']!r} needs at least one token_spans row "
            "(a verified correspondence without spans claims nothing)"
        )


# ---------------------------------------------------------------------------
# Member bindings (the token_boundary_v1 descriptor).
# ---------------------------------------------------------------------------


class _TokenBoundaryFamily(SidecarFamily):
    """``torchlens.boundary_facts`` family: one ``token_boundary_v1`` descriptor."""

    def validate_payload(self, payload: Any) -> None:
        validate_descriptor(payload)


def validate_descriptor(payload: Any) -> dict[str, Any]:
    """Validate a ``token_boundary_v1`` descriptor, FAIL-CLOSED on unknown keys."""

    if not isinstance(payload, Mapping):
        raise ValueError("token_boundary_v1 descriptor must be a mapping")
    keys = set(payload)
    if not _DESCRIPTOR_REQUIRED <= keys <= _DESCRIPTOR_REQUIRED | _DESCRIPTOR_OPTIONAL:
        raise ValueError(
            "token_boundary_v1 descriptor keys must include "
            f"{sorted(_DESCRIPTOR_REQUIRED)} and may add {sorted(_DESCRIPTOR_OPTIONAL)}; "
            f"got {sorted(keys)}"
        )
    if payload["stream"] not in _TOKEN_STREAMS:
        raise ValueError(f"token_boundary_v1 'stream' must be one of {sorted(_TOKEN_STREAMS)}")
    for key in ("batch_row", "length"):
        if not _is_int(payload[key]) or payload[key] < 0:
            raise ValueError(f"token_boundary_v1 {key!r} must be a non-negative int")
    digest = payload["ids_digest"]
    if not isinstance(digest, str) or not digest.startswith(_DIGEST_PREFIX):
        raise ValueError(f"token_boundary_v1 'ids_digest' must be a '{_DIGEST_PREFIX}<hex>' string")
    completion = payload.get("completion")
    if completion is not None and _int_range(completion, what="completion")[1] > payload["length"]:
        raise ValueError("token_boundary_v1 'completion' ends past the stream length")
    excluded = payload.get("excluded_ids", [])
    if not isinstance(excluded, list) or not all(_is_int(item) for item in excluded):
        raise ValueError("token_boundary_v1 'excluded_ids' must be a list of ints")
    for key in ("tokenizer", "template"):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise ValueError(f"token_boundary_v1 {key!r} must be a string or null")
    return dict(payload)


if TOKEN_BOUNDARY_FAMILY_ID not in list_sidecar_families():
    register_sidecar_family(
        _TokenBoundaryFamily(
            family_id=TOKEN_BOUNDARY_FAMILY_ID,
            schema_id=TOKEN_BOUNDARY_SCHEMA_ID,
            version=1,
            owner="torchlens",
        )
    )


def ids_digest(ids: Sequence[int]) -> str:
    """Return the ``sha256:`` digest of one token-id row (int64 little-endian)."""

    hasher = hashlib.sha256(b"token_ids_v1:")
    for token in ids:
        hasher.update(int(token).to_bytes(8, "little", signed=True))
    return _DIGEST_PREFIX + hasher.hexdigest()


def _stream_op(trace: Trace, stream: str) -> Any | None:
    """Return the first integer-dtype input (or output) op of ``trace``, if any."""

    try:
        ops = list(trace.input_ops if stream == "input" else trace.output_ops)
    except Exception:  # noqa: BLE001 - an unreadable stream is simply unbound
        return None
    for op in ops:
        dtype = getattr(op, "dtype", None)
        if dtype is not None and not (
            dtype.is_floating_point or dtype.is_complex or dtype is torch.bool
        ):
            return op
    return None


def _stream_shape(trace: Trace, stream: str) -> tuple[int, ...] | None:
    op = _stream_op(trace, stream)
    shape = getattr(op, "shape", None) if op is not None else None
    return tuple(int(size) for size in shape) if shape is not None else None


def _stream_ids(trace: Trace, stream: str, batch_row: int) -> list[int] | None:
    """Return one retained token-id row of ``stream``, or ``None`` when unavailable."""

    op = _stream_op(trace, stream)
    if op is None or not getattr(op, "has_saved_activation", False):
        return None
    try:
        payload: torch.Tensor = op.out
    except Exception:  # noqa: BLE001 - unretained ids leave the row unverifiable
        return None
    if getattr(payload, "ndim", None) == 1 and batch_row == 0:
        return [int(token) for token in payload.tolist()]
    if getattr(payload, "ndim", None) == 2 and batch_row < int(payload.shape[0]):
        return [int(token) for token in payload[batch_row].tolist()]
    return None


def member_binding(trace: Trace) -> dict[str, Any] | None:
    """Return the member's validated ``token_boundary_v1`` descriptor, if attached."""

    namespace = getattr(trace, "annotations", {}).get(SIDECAR_ANNOTATIONS_KEY)
    if not isinstance(namespace, dict) or TOKEN_BOUNDARY_FAMILY_ID not in namespace:
        return None
    return validate_descriptor(read_sidecar(trace, TOKEN_BOUNDARY_FAMILY_ID))


def bind_member(trace: Trace) -> dict[str, Any] | None:
    """Return the member's binding, deriving and attaching a default one if possible.

    The default descriptor binds the ``"input"`` stream of a batch-of-one
    capture whose ids are retained; anything else stays unbound.
    """

    existing = member_binding(trace)
    if existing is not None:
        return existing
    shape = _stream_shape(trace, "input")
    if shape is None or len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] != 1):
        return None
    ids = _stream_ids(trace, "input", 0)
    if ids is None:
        return None
    descriptor = {
        "stream": "input",
        "batch_row": 0,
        "length": len(ids),
        "ids_digest": ids_digest(ids),
    }
    attach_sidecar(trace, TOKEN_BOUNDARY_FAMILY_ID, descriptor)
    return descriptor


def _stream_length(trace: Trace, binding: Mapping[str, Any] | None) -> int | None:
    if binding is not None:
        return int(binding["length"])
    shape = _stream_shape(trace, "input")
    return shape[-1] if shape else None


def check_span_bounds(
    spans: Sequence[Sequence[int]], reader: Trace, source: Trace, *, row_index: int
) -> None:
    """Refuse span rows outside the members' known token streams.

    Unbound members are measured on their input stream (the teacher-forced
    convention); a member with no readable stream length is not checked.
    """

    for side, trace, lo_index in (("source", source, 0), ("destination", reader, 2)):
        length = _stream_length(trace, member_binding(trace))
        if length is None:
            continue
        for span in spans:
            if span[lo_index + 1] > length:
                raise BundleRelationError(
                    f"fed_by token_spans row {list(span)!r} reaches {side} position "
                    f"{span[lo_index + 1]}, past the {side} stream length {length}.",
                    code="bundle_relation_span_out_of_bounds",
                    row_index=row_index,
                    row_kind=FED_BY_KIND,
                    side=side,
                    stream_length=length,
                )


def _bound_ids(trace: Trace, binding: Mapping[str, Any]) -> tuple[str, list[int] | None]:
    """Return ``(state, ids)``: ``"ok"``, ``"unverifiable"`` or ``"contradicted"``."""

    ids = _stream_ids(trace, str(binding["stream"]), int(binding["batch_row"]))
    if ids is None:
        return "unverifiable", None
    if len(ids) != binding["length"] or ids_digest(ids) != binding["ids_digest"]:
        return "contradicted", None
    return "ok", ids


def assess_spans(
    spans: Sequence[Sequence[int]], reader: Trace, source: Trace
) -> tuple[str, float | None, str]:
    """Check declared spans against the bound streams of both members.

    Returns
    -------
    tuple[str, float | None, str]
        ``(state, coverage, reason)`` with state ``"verified"``,
        ``"unverifiable"`` (evidence absent) or ``"contradicted"`` (evidence
        disagrees), the computed coverage for verified rows, and a reason.
    """

    source_binding, reader_binding = member_binding(source), member_binding(reader)
    if source_binding is None or reader_binding is None or not spans:
        return "unverifiable", None, "an endpoint carries no token_boundary_v1 binding"
    source_state, source_ids = _bound_ids(source, source_binding)
    reader_state, reader_ids = _bound_ids(reader, reader_binding)
    for state, who in ((source_state, "source"), (reader_state, "destination")):
        if state != "ok":
            return state, None, f"the {who} binding is {state} against its retained ids"
    source_ids, reader_ids = cast("list[int]", source_ids), cast("list[int]", reader_ids)
    bound_completion = source_binding.get("completion")
    completion = bound_completion or [0, len(source_ids)]
    for src_lo, src_hi, dst_lo, dst_hi in spans:
        if not completion[0] <= src_lo <= src_hi <= completion[1]:
            return "contradicted", None, f"source span {[src_lo, src_hi]} leaves the completion"
        if source_ids[src_lo:src_hi] != reader_ids[dst_lo:dst_hi]:
            span = [src_lo, src_hi, dst_lo, dst_hi]
            return "contradicted", None, f"token ids differ on span {span}"
    if bound_completion is None:
        # No bound completion range: the eligible set is unknown, so no coverage.
        return "verified", None, "verified"
    excluded = set(source_binding.get("excluded_ids", []))
    eligible = [p for p in range(*completion) if source_ids[p] not in excluded]
    covered = {p for src_lo, src_hi, _, _ in spans for p in range(src_lo, src_hi)}
    coverage = len(covered.intersection(eligible)) / len(eligible) if eligible else None
    return "verified", coverage, "verified"


def settle_fed_by_params(
    params: Mapping[str, Any],
    reader: Trace,
    source: Trace,
    *,
    row_index: int,
    assess_unclaimed: bool,
) -> dict[str, Any]:
    """Return the row's params with the basis TorchLens can stand behind.

    ``assess_unclaimed`` (the positional ``relate`` door with ``basis``
    omitted) binds the members and verifies the spans when it can.
    """

    settled = dict(params)
    spans = settled.get("token_spans") or []
    check_span_bounds(spans, reader, source, row_index=row_index)
    claimed = settled.get("basis")
    if claimed not in EARNED_BASES and not (assess_unclaimed and claimed is None):
        return settled
    if assess_unclaimed and claimed is None and spans:
        bind_member(source)
        bind_member(reader)
    state, coverage, reason = assess_spans(spans, reader, source)
    if state == "contradicted" and claimed in EARNED_BASES:
        raise BundleRelationError(
            f"fed_by row claims basis {claimed!r} but its evidence disagrees: {reason}. "
            "A verified correspondence must re-check against the members it names.",
            code="bundle_relation_span_unverified",
            row_index=row_index,
            row_kind=FED_BY_KIND,
            claimed_basis=claimed,
            reason=reason,
        )
    if state != "verified":
        settled["basis"] = "declared"
        return settled
    if claimed is None:
        settled["basis"] = "verified"
    if coverage is not None:
        settled["coverage"] = coverage
    else:
        settled.pop("coverage", None)
    return settled
