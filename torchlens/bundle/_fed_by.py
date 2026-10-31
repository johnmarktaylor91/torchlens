"""The ``fed_by`` relation kind: content provenance between two traced calls.

"critique fed_by draft" records that the LATER member's prompt contains the
EARLIER member's output. Like every ordering kind the row's ``from`` is the
later member (the reader) and ``to`` the earlier member (the source), so
``ordering_path`` and the upstream/downstream walks read it in the same
direction as ``successor_of``. ``fed_by`` is CONTENT provenance only: a router
whose decision selects a call without its output entering that call's prompt
is ``selected_by`` (:mod:`torchlens.bundle._selected_by`), never ``fed_by``.

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
length, the content digest of the row's token ids, the member's own
completion range inside that stream, the token ids outside the eligible set
(padding, special tokens), and the tokenizer and template identity when known.
Users attach one through ``torchlens.io.attach_sidecar`` before relating and
may leave out the measured ``length``/``ids_digest``, which the positional
``relate`` fills from the retained ids; with no descriptor it derives a
default input-stream binding for a batch-of-one capture. Without a bound
completion range a source cannot verify, because TorchLens cannot tell its
output from its prompt.

BASIS. A row reads ``"verified"`` only when TorchLens checked it in this
process: both endpoints bound, each binding's digest equal to the member's
retained ids, the source binding naming its completion range, every source
span inside it, and the token ids equal span by span. Otherwise the row is
``"declared"``. A claimed ``"verified"``/``"matched"`` row arriving through
any door (relate, the constructor, an artifact) is re-checked: unverifiable
evidence (ids not retained, no binding, no completion range) clamps to
``"declared"``; contradicting evidence refuses typed
(``bundle_relation_span_unverified``). ``"matched"`` means the automatic span
finder (:mod:`torchlens.bundle._token_spans`) found the spans: on top of the
``"verified"`` checks, re-running the matcher on the members must return the
same spans, else the row settles as ``"verified"`` (checked, not found).

COVERAGE is computed on verified rows: the covered fraction of the source's
eligible tokens, where the eligible set is the bound completion range minus
positions whose id is listed in ``excluded_ids`` (padding, special tokens). On
declared rows a user-supplied coverage is a declaration.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import torch

from .._io.sidecar import (
    SIDECAR_ANNOTATIONS_KEY,
    SidecarError,
    SidecarFamily,
    attach_sidecar,
    list_sidecar_families,
    read_sidecar,
    register_sidecar_family,
)
from ..errors.episode import BundleRelationError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

FED_BY_KIND = "fed_by"
FED_BY_BASES = frozenset({"declared", "verified", "matched"})
#: Bases TorchLens must earn; any door re-checks them against the members.
EARNED_BASES = frozenset({"verified", "matched"})
TOKEN_BOUNDARY_FAMILY_ID = "torchlens.boundary_facts"
TOKEN_BOUNDARY_SCHEMA_ID = "token_boundary_v1"
_TOKEN_STREAMS = frozenset({"input", "output"})
_DESCRIPTOR_REQUIRED = frozenset({"stream", "batch_row"})
#: ``length`` and ``ids_digest`` are measured: a user may omit them and
#: ``bind_member`` completes them from the retained ids before relating.
_DESCRIPTOR_OPTIONAL = frozenset(
    {"length", "ids_digest", "completion", "excluded_ids", "tokenizer", "template"}
)
_DIGEST_PREFIX = "sha256:"
_LABEL_MAX_CHARS = 256


def _is_int(value: Any) -> bool:
    """Return whether ``value`` is a plain int (``bool`` excluded)."""

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
        if row[0] >= row[1] or row[2] >= row[3]:
            raise ValueError(
                f"fed_by token_spans row {index} must have start < end on both sides "
                f"(an empty span claims nothing): {list(row)!r}"
            )
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
        """Refuse a malformed descriptor typed, at attach and at every read."""

        try:
            validate_descriptor(payload)
        except ValueError as exc:
            raise SidecarError(
                f"Sidecar payload for family {self.family_id!r} is not a valid "
                f"token_boundary_v1 descriptor: {exc}",
                code="sidecar_payload_invalid",
                family_id=self.family_id,
                remedy="Re-attach the descriptor through torchlens.io.attach_sidecar.",
            ) from exc


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
        if key in payload and (not _is_int(payload[key]) or payload[key] < 0):
            raise ValueError(f"token_boundary_v1 {key!r} must be a non-negative int")
    digest = payload.get("ids_digest", _DIGEST_PREFIX)
    if not isinstance(digest, str) or not digest.startswith(_DIGEST_PREFIX):
        raise ValueError(f"token_boundary_v1 'ids_digest' must be a '{_DIGEST_PREFIX}<hex>' string")
    _validate_descriptor_optional(payload)
    return dict(payload)


def _validate_descriptor_optional(payload: Mapping[str, Any]) -> None:
    """Validate the optional descriptor keys: completion, exclusions, identities."""

    completion = payload.get("completion")
    if completion is not None:
        end = _int_range(completion, what="completion")[1]
        if "length" in payload and end > payload["length"]:
            raise ValueError("token_boundary_v1 'completion' ends past the stream length")
    excluded = payload.get("excluded_ids", [])
    if not isinstance(excluded, list) or not all(_is_int(item) for item in excluded):
        raise ValueError("token_boundary_v1 'excluded_ids' must be a list of ints")
    for key in ("tokenizer", "template"):
        if payload.get(key) is not None and not isinstance(payload[key], str):
            raise ValueError(f"token_boundary_v1 {key!r} must be a string or null")


# The sidecar door imports this module on its first consultation
# (``torchlens._io.sidecar._ensure_builtin_families``), so the family is
# present in a fresh process whatever was imported first.
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
    """Return the prefixed hex digest of one token-id row (eight-byte little-endian ints)."""

    hasher = hashlib.sha256(b"token_ids_v1:")
    for token in ids:
        hasher.update(int(token).to_bytes(8, "little", signed=True))
    return _DIGEST_PREFIX + hasher.hexdigest()


def _stream_op(trace: Trace, stream: str) -> Any | None:
    """Return the first integer-dtype input (or output) op of ``trace``, if any."""

    for op in trace.input_ops if stream == "input" else trace.output_ops:
        dtype = getattr(op, "dtype", None)
        if dtype is not None and not (
            dtype.is_floating_point or dtype.is_complex or dtype is torch.bool
        ):
            return op
    return None


def _stream_shape(trace: Trace, stream: str) -> tuple[int, ...] | None:
    """Return the shape of ``stream``'s token-id tensor, or ``None`` when absent."""

    op = _stream_op(trace, stream)
    shape = getattr(op, "shape", None) if op is not None else None
    return tuple(int(size) for size in shape) if shape is not None else None


def _stream_ids(trace: Trace, stream: str, batch_row: int) -> list[int] | None:
    """Return one retained token-id row of ``stream``, or ``None`` when unavailable."""

    op = _stream_op(trace, stream)
    if op is None or not getattr(op, "has_saved_activation", False):
        return None
    payload = op.out  # a lazily loaded member materializes its ids on this read
    if not isinstance(payload, torch.Tensor):
        return None
    if payload.ndim == 1 and batch_row == 0:
        return [int(token) for token in payload.tolist()]
    if payload.ndim == 2 and batch_row < int(payload.shape[0]):
        return [int(token) for token in payload[batch_row].tolist()]
    return None


def member_binding(trace: Trace) -> dict[str, Any] | None:
    """Return the member's validated ``token_boundary_v1`` descriptor, if attached.

    Raises
    ------
    SidecarError
        ``sidecar_payload_invalid`` for a malformed descriptor.
    """

    namespace = getattr(trace, "annotations", {}).get(SIDECAR_ANNOTATIONS_KEY)
    if not isinstance(namespace, dict) or TOKEN_BOUNDARY_FAMILY_ID not in namespace:
        return None
    return validate_descriptor(read_sidecar(trace, TOKEN_BOUNDARY_FAMILY_ID))


def bind_member(trace: Trace) -> dict[str, Any] | None:
    """Return the member's binding with its measured keys filled in, if possible.

    An attached descriptor missing ``length``/``ids_digest`` is completed
    from the retained ids of the stream it names. With no descriptor, the
    default binds the ``"input"`` stream of a batch-of-one capture (no
    completion range, identities unknown). Anything else stays as it is.
    """

    existing = member_binding(trace)
    if existing is None:
        shape = _stream_shape(trace, "input")
        if shape is None or len(shape) not in (1, 2) or (len(shape) == 2 and shape[0] != 1):
            return None
        existing = {"stream": "input", "batch_row": 0}
    if "length" in existing and "ids_digest" in existing:
        return existing
    ids = _stream_ids(trace, str(existing["stream"]), int(existing["batch_row"]))
    if ids is None:
        return existing if member_binding(trace) is not None else None
    completed = {**existing, "length": len(ids), "ids_digest": ids_digest(ids)}
    attach_sidecar(trace, TOKEN_BOUNDARY_FAMILY_ID, dict(completed))
    return completed


def check_span_bounds(
    spans: Sequence[Sequence[int]], reader: Trace, source: Trace, *, row_index: int
) -> None:
    """Refuse span rows outside the members' bound token streams.

    Only a binding that names its ``length`` says what the coordinates index;
    an unbound member's coordinates are a declaration and are not measured.
    """

    for side, trace, lo_index in (("source", source, 0), ("destination", reader, 2)):
        binding = member_binding(trace)
        if binding is None or "length" not in binding:
            continue
        length = int(binding["length"])
        for span in spans:
            if span[lo_index + 1] > length:
                raise BundleRelationError(
                    f"fed_by token_spans row {list(span)!r} reaches {side} position "
                    f"{span[lo_index + 1]}, past the {side} stream length {length}.",
                    code="bundle_relation_span_out_of_bounds",
                    remedy=(
                        "index the member's bound token stream (its token_boundary_v1 "
                        "length), or bind the stream the coordinates really index"
                    ),
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


def stream_ids(trace: Trace, binding: Mapping[str, Any]) -> list[int] | None:
    """Return the bound row's token ids when they agree with the binding, else ``None``."""

    if "length" not in binding or "ids_digest" not in binding:
        return None
    state, ids = _bound_ids(trace, binding)
    return ids if state == "ok" else None


def retained_ids(trace: Trace, binding: Mapping[str, Any]) -> list[int] | None:
    """Return the bound row's ids: checked against a measured binding, else as retained."""

    if "length" in binding and "ids_digest" in binding:
        return stream_ids(trace, binding)
    return _stream_ids(trace, str(binding["stream"]), int(binding["batch_row"]))


def _span_contradiction(
    span: Sequence[int],
    completion: Sequence[int],
    prompt_end: int,
    source_ids: Sequence[int],
    reader_ids: Sequence[int],
) -> str | None:
    """Return why one span contradicts the bound streams, or ``None`` when it agrees."""

    src_lo, src_hi, dst_lo, dst_hi = span
    if not completion[0] <= src_lo < src_hi <= completion[1]:
        return f"source span {[src_lo, src_hi]} leaves the completion"
    if dst_hi > prompt_end:
        return f"destination span {[dst_lo, dst_hi]} reaches the reader's own output"
    if list(source_ids[src_lo:src_hi]) != list(reader_ids[dst_lo:dst_hi]):
        return f"token ids differ on span {list(span)}"
    return None


def assess_spans(
    spans: Sequence[Sequence[int]], reader: Trace, source: Trace
) -> tuple[str, float | None, str]:
    """Check declared spans against the bound streams of both members.

    Verification needs both members bound (``length`` and ``ids_digest``
    measured) and the SOURCE binding to name its ``completion`` range: only
    then is "the earlier output reached the later prompt" checkable, rather
    than "two token streams share some ids". When the READER binding names
    its own completion, destination spans must end before it (the text
    reached the prompt, not the reader's output).

    Returns
    -------
    tuple[str, float | None, str]
        ``(state, coverage, reason)`` with state ``"verified"``,
        ``"unverifiable"`` (evidence absent) or ``"contradicted"`` (evidence
        disagrees), the computed coverage for verified rows, and a reason.
    """

    source_binding, reader_binding = member_binding(source), member_binding(reader)
    for binding, who in ((source_binding, "source"), (reader_binding, "destination")):
        if binding is None or "ids_digest" not in binding or "length" not in binding:
            return "unverifiable", None, f"the {who} carries no measured token_boundary_v1 binding"
    source_binding, reader_binding = (
        cast("dict[str, Any]", source_binding),
        cast("dict[str, Any]", reader_binding),
    )
    completion = source_binding.get("completion")
    if completion is None:
        return "unverifiable", None, "the source binding names no completion range"
    source_state, source_ids = _bound_ids(source, source_binding)
    reader_state, reader_ids = _bound_ids(reader, reader_binding)
    for state, who in ((source_state, "source"), (reader_state, "destination")):
        if state != "ok":
            return state, None, f"the {who} binding is {state} against its retained ids"
    source_ids, reader_ids = cast("list[int]", source_ids), cast("list[int]", reader_ids)
    reader_completion = reader_binding.get("completion")
    prompt_end = reader_completion[0] if reader_completion is not None else len(reader_ids)
    for span in spans:
        contradiction = _span_contradiction(span, completion, prompt_end, source_ids, reader_ids)
        if contradiction is not None:
            return "contradicted", None, contradiction
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
    omitted) first binds both members (filling measured keys), then verifies
    the spans when it can. A claimed ``matched`` that verifies stays
    ``matched`` only when the matcher, re-run on the members, finds the same
    spans; otherwise it settles as ``verified`` (checked, not found).
    """

    settled = dict(params)
    spans = settled.get("token_spans") or []
    claimed = settled.get("basis")
    assess = bool(spans) and (claimed in EARNED_BASES or (assess_unclaimed and claimed is None))
    if assess and claimed is None:
        bind_member(source)
        bind_member(reader)
    check_span_bounds(spans, reader, source, row_index=row_index)
    if not assess:
        return settled
    state, coverage, reason = assess_spans(spans, reader, source)
    if state == "contradicted" and claimed in EARNED_BASES:
        raise BundleRelationError(
            f"fed_by row claims basis {claimed!r} but its evidence disagrees: {reason}. "
            "A verified correspondence must re-check against the members it names.",
            code="bundle_relation_span_unverified",
            remedy=(
                "drop the claimed basis so the row records a declaration, or fix the "
                "spans or bindings so the token ids agree"
            ),
            row_index=row_index,
            row_kind=FED_BY_KIND,
            claimed_basis=claimed,
            reason=reason,
        )
    if state != "verified":
        settled["basis"] = "declared"
        return settled
    from ._token_spans import rederives_as_matched

    found = claimed == "matched" and rederives_as_matched(spans, reader, source)
    settled["basis"] = "matched" if found else "verified"
    if coverage is not None:
        settled["coverage"] = coverage
    else:
        settled.pop("coverage", None)
    return settled
