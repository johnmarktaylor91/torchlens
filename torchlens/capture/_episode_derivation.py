"""Episode step-output derivation: the declaration-driven evidence readers.

The F40b derivation family (foldA D8), split from
:mod:`torchlens.capture._episode_ledger` along the derivation/validation seam
(R43 file-size ratchet): per-step evidence derives from the DECLARED
``step_output_kind`` / ``step_output_from`` / ``step_axis``, never from
root-output-shape guessing, with per-step positions TAIL-ALIGNED along the
declared axis. The capture digest minted here binds a ledger to ITS product
(the F42 attested-coupling input). Ledger construction, settlement wiring,
and load validation stay in the ledger module; this leaf reads only the
settled trace and the resolved declaration.

Every spelling here is DOCUMENTED-UNSTABLE pending the rolling naming
session; semantics are pinned by the ratified S2/S6/S7 contracts and the
F40b declared-derivation doc of record (``docs/reference/episode_capture.md``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._episode_ledger import ResolvedEpisode

__all__ = [
    "CAPTURE_DIGEST_SCHEMA",
    "mint_capture_digest",
    "recompute_capture_digests",
    "rederive_step_evidence",
    "rederived_evidence_matches",
    "validate_step_output_positions",
]

#: Schema tag of the CONTENT-BINDING capture digest (W051, audit 3.2): the
#: product's recomputable identity facts PLUS the finished ledger content
#: (header minus the digest slot, every row). The v1 form hashed identity
#: alone, so two equal-length prompts minted identical digests and a rewritten
#: ledger (grades, step_output) still attested bound.
CAPTURE_DIGEST_SCHEMA = "episode_capture_digest_v2"
_LEGACY_CAPTURE_DIGEST_SCHEMA = "episode_capture_digest_v1"


def _declaration_error(message: str, *, code: str | None = None) -> Exception:
    """Build the typed entry refusal (import-local to keep the leaf light)."""

    from ..errors.episode import EpisodeDeclarationError

    return EpisodeDeclarationError(
        message,
        code=code if code is not None else "episode_declaration_invalid",
    )


def _call_returned(call: Any) -> bool:
    """Whether a ModuleCall shows exit evidence (the forward returned).

    ``ModuleCall`` carries no explicit exit flag; output ops, a positive
    forward duration, and a captured output structure are each written only
    by the module-exit event, so any of them witnesses a returned call.
    """

    if getattr(call, "output_ops", None):
        return True
    if getattr(call, "forward_duration", 0.0) and call.forward_duration > 0:
        return True
    return getattr(call, "output_structure", None) is not None


def _pass_range_for_call(call: Any) -> tuple[int, int] | None:
    """Derive the [min, max] op pass-coordinate range inside one call."""

    passes: list[int] = []
    for op_label in getattr(call, "ops", ()) or ():
        text = str(op_label)
        if ":" in text:
            suffix = text.rsplit(":", 1)[1]
            if suffix.isdigit():
                passes.append(int(suffix))
    if not passes:
        return None
    return (min(passes), max(passes))


def _render_container_path(path: Any) -> str:
    """Render one output op's container path as the declared-slot spelling.

    Dot-separated segments: dict keys and namedtuple/ModelOutput field names
    by name, tuple/list positions by index (``"sequences"``, ``"0"``,
    ``"0.logits"``). The bare single-tensor root renders as ``""`` (nothing
    to declare; the default source is the whole output).
    """

    parts: list[str] = []
    for segment in tuple(path or ()):
        for attr in ("key", "name", "index"):
            value = getattr(segment, attr, None)
            if value is not None:
                parts.append(str(value))
                break
        else:
            parts.append(str(segment))
    return ".".join(parts)


def _resolve_step_output_source(trace: Any, resolved: ResolvedEpisode) -> Any:
    """Resolve the DECLARED step-output source to one root-output payload.

    ``step_output_from=None`` keeps the single-output default (exactly one
    root output layer); a declared path selects among the root output's
    tensor leaves by container position (``Op.container_path``), so
    dict/ModelOutput/tuple roots resolve without shape guessing.

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` when the source is ambiguous
        (multiple outputs, nothing declared), the declared path names no
        output leaf, or the resolved payload was not retained.
    """

    source_op = _resolve_step_output_op(trace, resolved)
    out = source_op.out
    if out is None:
        raise _declaration_error(
            "the episode step-output source payload was not retained by this "
            "capture's save= selection. Remedy: include the root output in "
            "save= (or keep the default save policy), or declare "
            "step_output_kind='none' for a status-only ledger.",
            code="episode_declaration_invalid",
        )
    return out


def _resolve_step_output_op(trace: Any, resolved: Any) -> Any:
    """Resolve the declared step-output source to its root-output OP record.

    The op-level half of :func:`_resolve_step_output_source`: the retention
    check is the caller's, so the re-derivation cross-check can tell "payload
    not retained" (nothing to compare) from "declaration cannot resolve on
    this product" (a mismatch signal) without parsing messages.
    """

    output_ops = list(trace.output_ops)
    paths = {op: _render_container_path(getattr(op, "container_path", ())) for op in output_ops}
    if resolved.step_output_from is None:
        if len(output_ops) != 1:
            available = sorted(p for p in paths.values() if p)
            raise _declaration_error(
                "episode step-output derivation found "
                f"{len(output_ops)} root output layers and no declared source; "
                "declare EpisodeSpec.step_output_from naming the slot the "
                "per-step evidence derives from"
                + (f" (available slots: {available})" if available else "")
                + ".",
                code="episode_declaration_invalid",
            )
        source_op = output_ops[0]
    else:
        matches = [op for op, path in paths.items() if path == resolved.step_output_from]
        if not matches:
            available = sorted(p for p in paths.values() if p)
            raise _declaration_error(
                f"EpisodeSpec.step_output_from={resolved.step_output_from!r} "
                "names no root-output slot on this capture"
                + (
                    f"; available slots: {available}"
                    if available
                    else "; the root returned a single bare tensor (drop "
                    "step_output_from to use it)"
                )
                + ".",
                code="episode_declaration_invalid",
            )
        source_op = matches[0]
    return source_op


def _flat_cpu_snapshot(value: Any) -> Any:
    """Detached, flat CPU copy of a boundary payload for digest/evidence reads.

    Routes through the ``safe_copy`` chokepoint (the training-mode detach
    guardrail) instead of a bare ``.detach()``; the copy is consumed by byte
    reinterpretation or ``tolist`` and never re-enters capture state.
    """

    from ..utils.tensor_utils import safe_copy

    return safe_copy(value, detach_tensor=True).cpu().reshape(-1)


def _step_slice_digest(step_slice: Any) -> str:
    """Content digest of one per-step source slice (``kind="digest"``).

    ``sha256:`` prefix over dtype + shape + raw little-endian bytes of the
    detached CPU-contiguous slice; deterministic across processes for equal
    payloads. Construction is DOCUMENTED-UNSTABLE (naming session) but
    byte-stable within a release.
    """

    import torch

    # A fresh buffer forces literal stride-1 storage: a size-1 slice of a
    # wider tensor reads as "contiguous" while keeping its parent stride
    # (clone/contiguous preserve it), which the byte reinterpretation
    # below rejects.
    source = _flat_cpu_snapshot(step_slice)
    flat = torch.empty(source.shape, dtype=source.dtype)
    flat.copy_(source)
    hasher = hashlib.sha256()
    hasher.update(str(step_slice.dtype).encode("utf-8"))
    hasher.update(str(tuple(step_slice.shape)).encode("utf-8"))
    if flat.numel():
        hasher.update(flat.view(torch.uint8).numpy().tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _step0_entry_width(resolved: ResolvedEpisode) -> int | None:
    """The measured position width of the FIRST stepped call's entry, or None.

    Read from the live join session's step-0 boundary snapshot (lane F40c
    records ``entry_shape`` for token-id-shaped integer entries). ``None``
    when no session ran, the entry was not snapshotted (float entry,
    oversize, no tensor argument), or the entry rank has no defensible
    position axis (only rank 1/2 -- ``[positions]`` / ``[batch,
    positions]`` -- read as a token entry).
    """

    session = getattr(resolved, "join_session", None)
    boundaries = getattr(session, "boundaries", None) or ()
    if not boundaries:
        return None
    shape = getattr(boundaries[0], "entry_shape", None)
    if shape is None or len(shape) not in (1, 2):
        return None
    return int(shape[-1])


def _measured_entry_rows(
    resolved: ResolvedEpisode, n_rows: int
) -> list[list[tuple[int, ...]]] | None:
    """Per-step token-id entry rows from the live join session, or ``None``.

    ``None`` unless the session snapshotted EXACT ids for every started
    step's entry (integer, token-id-shaped, under the snapshot ceiling); the
    chain arm never reasons over a partially measured chain.
    """

    from ._episode_join import _entry_rows

    session = getattr(resolved, "join_session", None)
    boundaries = getattr(session, "boundaries", None) or ()
    if len(boundaries) < n_rows:
        return None
    rows_by_step = [_entry_rows(boundary) for boundary in boundaries[:n_rows]]
    if any(rows is None for rows in rows_by_step):
        return None
    return [list(rows) for rows in rows_by_step if rows is not None]


def _chain_widths(
    entries: list[list[tuple[int, ...]]], root_rows: list[tuple[int, ...]], crossings: set[int]
) -> list[int] | None:
    """Per-step entry widths of a root-reproduced chain, or ``None`` (see the arm)."""

    widths: list[int] = []
    for step, rows in enumerate(entries):
        width = len(rows[0])
        prefix_ok = all(len(row) == width for row in rows) and all(
            root_row[:width] == row for root_row, row in zip(root_rows, rows, strict=True)
        )
        surplus = width - widths[-1] - 1 if step else 0
        if not prefix_ok or surplus < 0 or (surplus > 0 and step not in crossings):
            return None
        widths.append(width)
    return widths if len(root_rows[0]) == widths[-1] + 1 else None


def _measured_chain_positions(
    resolved: ResolvedEpisode, out: Any, axis: int, n_rows: int
) -> list[int] | None:
    """The DECLARED-CROSSING chain arm: per-step emission positions, or ``None``.

    Licensed when the root output reproduces the MEASURED entry chain and
    every surplus position is a declared crossing (W051 FIX2):

    - the join session snapshotted exact token ids for every step's entry,
      one id row per batch row matching the root's rows, and the source is
      an integer tensor whose ``step_axis`` is its position (last) axis;
    - every step's entry is a PREFIX of the root output row-for-row
      (``out[r, :w_k] == entry_k[r]``): the root returns the full fed id
      sequence, so position ``w_k`` -- right after step k's measured entry
      -- is step k's emission, and the positions past it up to the next
      entry's width are what the next step received from outside;
    - each join appends at least the one emission (``w_{k+1} >= w_k + 1``);
      a SURPLUS (``w_{k+1} - w_k - 1 > 0`` positions) is admitted only when
      step k+1 is a DECLARED crossing (``EpisodeSpec(crossings=...)``, the
      tool-call disclosure) -- an undeclared surplus is indistinguishable
      from multi-token decoding and stays refused; and the root ends
      exactly one emission after the last entry (``W == w_{n-1} + 1``).

    Returns ``[w_0, ..., w_{n-1}]``; on a pure append chain this IS the
    tail alignment. ``None`` when the premise is unmet (the caller refuses).
    """

    token_shaped = not (out.is_floating_point() or out.is_complex()) and out.dim() in (1, 2)
    if not token_shaped or (axis if axis >= 0 else out.dim() + axis) != out.dim() - 1:
        return None
    entries = _measured_entry_rows(resolved, n_rows)
    if entries is None:
        return None
    flat = _flat_cpu_snapshot(out).reshape(-1, out.shape[-1]).tolist()
    root_rows = [tuple(int(v) for v in row) for row in flat]
    if any(len(rows) != len(root_rows) for rows in entries):
        return None
    crossings = set(getattr(resolved, "crossings", ()) or ())
    return _chain_widths(entries, root_rows, crossings)


def _licensed_positions(resolved: ResolvedEpisode, out: Any, axis: int, n_rows: int) -> list[int]:
    """The per-step evidence positions the license admits (W051, 3.1 + FIX2).

    Tail alignment reads the LAST ``n_rows`` positions as one emission per
    step. That is only licensed when the source carries exactly the
    emissions (``axis_size == n_rows``) or exactly the measured step-0 entry
    plus one emission per step (``axis_size == entry_width + n_rows``, the
    prompt+completion shape). Any other width means more than one position
    per step (multi-token decoding, draft+verify loops, a stepped module
    called several times per iteration) or a source that is not the
    per-step emission column at all; deriving anyway misattributes tokens
    to steps and the join re-grade then reports FALSE exogenous breaks.
    Without a measured entry width only the emitted-only shape can be
    certified. The one further admission is the DECLARED-CROSSING chain arm
    (:func:`_measured_chain_positions`): a root returning the full fed id
    sequence whose measured entry chain it reproduces, with every surplus
    position a declared crossing, derives at the measured chain positions
    (disclosed in the header's ``step_output_positions``). Anything else is
    refused with the widths named.

    SCOPE: the license gates TOKEN-SHAPED episodes -- ``kind="tokens"``, or
    any kind whose step-0 entry was a token-id-shaped integer tensor (the
    only entries the join session snapshots a width for). A ``digest``
    column over a FLOAT carried state (fixed-point / diffusion loops, where
    the state keeps one shape and no width is measured) is a disclosed
    positional slice convention keyed by the header's ``step_output_from``
    / ``step_axis`` -- recomputable by any reader and consumed by no join
    re-grade -- so it keeps the historical ``>= n_rows`` admission.
    """

    axis_size = int(out.shape[axis])
    tail = list(range(axis_size - n_rows, axis_size))
    entry_width = _step0_entry_width(resolved)
    if resolved.step_output_kind != "tokens" and entry_width is None:
        return tail
    if axis_size == n_rows:
        return tail
    if entry_width is not None and axis_size == entry_width + n_rows:
        return tail
    chain = _measured_chain_positions(resolved, out, axis, n_rows)
    if chain is not None:
        return chain
    licensed = (
        f"{n_rows} (emitted-only root) or {entry_width + n_rows} (the measured "
        f"{entry_width}-position step-0 entry plus one emission per step)"
        if entry_width is not None
        else f"{n_rows} (emitted-only root; the step-0 entry width was not "
        "measured, so the prompt+completion shape cannot be certified)"
    )
    raise _declaration_error(
        f"the step-output source carries {axis_size} positions along "
        f"step_axis={axis} but the episode ran {n_rows} steps; tail-aligned "
        "per-step evidence (one emitted position per step) is licensed only "
        f"when the source carries exactly {licensed}. The source does not "
        "match ONE position per step (multi-token decoding, a draft+verify "
        "loop, a stepped module called more than once per iteration, or "
        "positions injected from outside mid-loop). Positions a tool call "
        "injected at step k's entry are admitted when the root returns the "
        "full fed id sequence (every measured entry as a prefix) and "
        "EpisodeSpec(crossings=(k, ...)) declares the crossing; otherwise "
        "declare step_output_from naming a per-step slot, declare "
        "step_output_kind='none' for a status-only ledger, or restructure the "
        "root to return one position per step.",
        code="episode_declaration_invalid",
    )


def _evidence_positions(
    resolved: Any, out: Any, axis: int, n_rows: int, *, check_license: bool
) -> list[int]:
    """Resolve the per-step evidence positions along the step axis.

    A PERSISTED disclosure (``step_output_positions`` on a re-derivation
    view of a loaded header) is consumed as-is after a fit check -- the
    positions must address exactly ``n_rows`` in-range positions, else the
    ledger does not describe this product. Without one, settlement runs the
    license; re-derivation of a pre-disclosure artifact reads the tail.
    """

    axis_size = int(out.shape[axis])
    persisted = getattr(resolved, "step_output_positions", None)
    if persisted is not None:
        positions = [int(p) for p in persisted]
        if len(positions) != n_rows or any(not 0 <= p < axis_size for p in positions):
            raise _declaration_error(
                f"the persisted step_output_positions {positions} do not address "
                f"{n_rows} positions inside the step-output source's {axis_size} "
                f"positions along step_axis={axis}; the ledger does not describe "
                "this product.",
                code="episode_declaration_invalid",
            )
        return positions
    if not check_license:
        return list(range(axis_size - n_rows, axis_size))
    return _licensed_positions(resolved, out, axis, n_rows)


def validate_step_output_positions(
    positions: Any,
    *,
    step_output_kind: str,
    n_rows: int | None = None,
    rows_with_output: int | None = None,
) -> tuple[int, ...] | None:
    """Validate a header ``step_output_positions`` slot FAIL-CLOSED.

    ``None`` is admitted (kind ``none``, a truncated ledger that derived no
    column, or a pre-disclosure grammar-v2 artifact). Present, the slot is a
    sequence of non-negative ints (bools refused), strictly increasing along
    the step axis, never under kind ``none``, one per ledger row when
    ``n_rows`` is known, and only on a ledger whose rows ALL carry
    ``step_output`` when ``rows_with_output`` is known. Returns the tuple
    form. Raises ``ValueError`` (callers map onto
    ``episode_ledger_incoherent``).
    """

    if positions is None:
        return None
    if rows_with_output is not None and n_rows is not None and rows_with_output != n_rows:
        raise ValueError(
            "episode_ledger_incoherent: step_output_positions disclosed on a ledger "
            f"whose rows do not all carry step_output ({rows_with_output} of {n_rows})"
        )
    if step_output_kind == "none":
        raise ValueError(
            "episode-ledger step_output_kind='none' derives no evidence column "
            "and cannot carry step_output_positions"
        )
    if isinstance(positions, (str, bytes)) or not isinstance(positions, Sequence):
        raise ValueError("episode-ledger step_output_positions must be a list of ints or null")
    values = tuple(positions)
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"episode-ledger step_output_positions must be non-negative ints, got {value!r}"
            )
    if any(later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ValueError(
            "episode-ledger step_output_positions must strictly increase along "
            f"the step axis, got {list(values)}"
        )
    if n_rows is not None and len(values) != n_rows:
        raise ValueError(
            f"episode-ledger step_output_positions lists {len(values)} positions "
            f"for {n_rows} ledger rows"
        )
    return values


def _derive_step_evidence(
    trace: Any, resolved: Any, n_rows: int, *, check_license: bool = True
) -> list[tuple[int, ...]] | list[str] | None:
    """The evidence column alone (see :func:`_derive_step_column`)."""

    return _derive_step_column(trace, resolved, n_rows, check_license=check_license)[0]


def _derive_step_column(
    trace: Any, resolved: Any, n_rows: int, *, check_license: bool = True
) -> tuple[list[tuple[int, ...]] | list[str] | None, tuple[int, ...] | None]:
    """Derive per-step evidence under the DECLARED kind and source (foldA D8).

    Returns ``(evidence_by_row, positions)``: the per-row evidence column and
    the step-axis positions it was read from (both ``None`` under kind
    ``none``). The positions are persisted in the header
    (``step_output_positions``) so any reader re-derives the column without
    the live join session.

    The derivation is declaration-driven, never root-output-shape guessing:

    - ``kind="none"``: no evidence at all (status-only ledger; the shape for
      roots with no per-step output structure, e.g. diffusion images).
    - ``kind="tokens"``: integer per-step token ids read from the declared
      source.
    - ``kind="digest"``: per-step content digests of the declared source
      (any dtype -- float/hidden-state roots are first-class).

    Per-step positions are TAIL-ALIGNED along the declared ``step_axis``:
    the LAST ``n_rows`` positions are the per-step emissions (one appended
    position per step), so a root returning what real ``generate()`` returns
    -- prompt+completion -- keeps its prompt prefix out of the evidence
    column, and a hand-shaped emitted-only root is the ``axis size ==
    n_rows`` special case of the same rule. The one other alignment is the
    DECLARED-CROSSING chain arm (:func:`_measured_chain_positions`): with
    tool-call positions declared via ``EpisodeSpec(crossings=...)`` the
    emissions sit right after each MEASURED entry, not at the tail.

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` when the declared source is
        ambiguous/unretained/not a tensor, the declared kind disagrees with
        the source dtype (``tokens`` on a float root teaches
        ``digest``/``none``), the axis is out of range, the source
        carries fewer positions than the episode ran steps, or the source
        width is neither the emitted-only, the measured entry-plus-one-
        per-step, nor the declared-crossing chain shape (alignment
        unlicensed; W051, audit 3.1 + FIX2).
    """

    import torch

    if resolved.step_output_kind == "none":
        return None, None
    out = _resolve_step_output_source(trace, resolved)
    if not isinstance(out, torch.Tensor):
        raise _declaration_error(
            "the episode step-output source must resolve to a tensor; got "
            f"{type(out).__name__}. Remedy: point step_output_from at a "
            "tensor slot, or declare step_output_kind='none'.",
            code="episode_declaration_invalid",
        )
    if resolved.step_output_kind == "tokens" and (out.is_floating_point() or out.is_complex()):
        raise _declaration_error(
            "step_output_kind='tokens' requires an integer token tensor as "
            f"the step-output source; got {out.dtype}. Remedy: declare "
            "step_output_kind='digest' (per-step content digests, any dtype) "
            "or 'none' (status-only ledger) for non-token roots.",
            code="episode_declaration_invalid",
        )
    axis = resolved.step_axis
    try:
        axis_size = out.shape[axis]
    except IndexError:
        raise _declaration_error(
            f"step_axis={axis} is out of range for the step-output source "
            f"shape {tuple(out.shape)}.",
            code="episode_declaration_invalid",
        ) from None
    if axis_size < n_rows:
        raise _declaration_error(
            f"the step-output source carries {axis_size} positions along "
            f"step_axis={axis} but the episode ran {n_rows} steps; per-step "
            f"evidence derives from the LAST {n_rows} positions (one emitted "
            "position per step), so the source must carry at least that many. "
            "The declaration does not match the executed episode.",
            code="episode_declaration_invalid",
        )
    positions = _evidence_positions(resolved, out, axis, n_rows, check_license=check_license)
    normalized_axis = axis if axis >= 0 else out.dim() + axis
    if resolved.step_output_kind == "digest":
        digests = [_step_slice_digest(out.select(normalized_axis, p)) for p in positions]
        return digests, tuple(positions)
    tokens = [
        tuple(int(v) for v in out.select(normalized_axis, p).reshape(-1).tolist())
        for p in positions
    ]
    return tokens, tuple(positions)


def _product_identity(trace: Any, address: str, started: int) -> dict[str, Any]:
    """The D4-clause recomputable identity facts of one episode product."""

    labels = [str(getattr(op, "label", op)) for op in (getattr(trace, "layer_list", None) or ())]
    return {
        "entry_seed": getattr(trace, "random_seed", None),
        "stepped_module": address,
        "stepped_calls": started,
        "op_labels": labels,
    }


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    """sha256 over the canonical (sorted, compact) JSON form of ``payload``."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ledger_content(ledger_payload: Mapping[str, Any]) -> dict[str, Any]:
    """The ledger content the v2 digest binds: header sans digest slot + rows."""

    header = dict(ledger_payload.get("header") or {})
    header.pop("capture_digest", None)
    # The episode_id is a per-capture random identifier (never a fact about
    # the execution); identical re-runs must mint identical digests (the F42
    # compare-digests contract), so it is excluded from the bound content.
    header.pop("episode_id", None)
    return {"header": header, "rows": list(ledger_payload.get("rows") or [])}


def mint_capture_digest(
    trace: Any, address: str, started: int, ledger_payload: Mapping[str, Any] | None = None
) -> str:
    """Mint the capture digest binding an episode ledger to ITS product.

    Hex SHA-256 over canonical JSON. With ``ledger_payload`` (the finished
    ledger's ``to_payload()`` form, digest slot ignored) the digest is the
    CONTENT-BINDING v2 form (W051, audit 3.2): the product's recomputable
    identity facts -- ordered recorded op labels, stepped-module address and
    call count, the managed entry seed (foldA D7, the F40b middle third) --
    PLUS every persisted ledger fact (header declaration and disclosure
    fields, the step_join envelope, the intervention digest, every row's
    status/coord/step_output/witness slots). A consumer (lane F42's attested
    coupling, the load anchor) recomputes it from the product and its
    persisted ledger and compares; any post-mint rewrite of ledger content
    reads unbound. It is an INTEGRITY binding, not a signature: a forger who
    re-mints is caught only by the evidence re-derivation cross-check
    (:func:`rederived_evidence_matches`), which re-reads the retained root
    output. Without ``ledger_payload`` the LEGACY v1 identity-only form is
    minted (kept so pre-v2 artifacts are recognized as pre-binding, never
    misreported as foreign). Spelling and construction are
    DOCUMENTED-UNSTABLE (naming session).
    """

    identity = _product_identity(trace, address, started)
    if ledger_payload is None:
        return _canonical_digest({"schema": _LEGACY_CAPTURE_DIGEST_SCHEMA, **identity})
    return _canonical_digest(
        {
            "schema": CAPTURE_DIGEST_SCHEMA,
            "identity": identity,
            "ledger": _ledger_content(ledger_payload),
        }
    )


def recompute_capture_digests(trace: Any, ledger_payload: Mapping[str, Any]) -> tuple[str, str]:
    """Recompute ``(v2, legacy_v1)`` capture digests from a product + payload."""

    header = ledger_payload.get("header") or {}
    rows = ledger_payload.get("rows") or []
    started = sum(1 for row in rows if isinstance(row, Mapping) and row.get("status") != "absent")
    address = str(header.get("stepped_module"))
    return (
        mint_capture_digest(trace, address, started, ledger_payload),
        mint_capture_digest(trace, address, started),
    )


@dataclass(frozen=True)
class _HeaderResolved:
    """The derivation-facing view of a PERSISTED header (re-derivation only)."""

    step_output_kind: str
    step_output_from: str | None
    step_axis: int
    join_session: Any = None
    step_output_positions: tuple[int, ...] | None = None


def rederive_step_evidence(
    trace: Any, header: Mapping[str, Any], n_rows: int
) -> list[tuple[int, ...]] | list[str] | None:
    """Re-derive the evidence column from the product's RETAINED root output.

    Returns ``None`` when there is nothing to compare: ``kind="none"``, no
    rows, or the declared source payload is not retained on this product
    (selective saves). Raises ``EpisodeDeclarationError`` when the persisted
    declaration cannot derive on this product at all (no such slot, a
    tokens declaration over a float root, axis out of range, too few
    positions) -- on a genuine product the same derivation succeeded at
    settlement, so a failure here is a mismatch signal, never noise. The
    alignment license is NOT re-checked (it was checked at settlement
    against the live entry chain, which no loaded product carries); the
    column is read at the header's persisted ``step_output_positions`` when
    disclosed, else at the tail (pre-disclosure artifacts).
    """

    kind = header.get("step_output_kind")
    if kind == "none" or n_rows <= 0:
        return None
    source = header.get("step_output_from")
    axis = header.get("step_axis")
    positions = header.get("step_output_positions")
    resolved = _HeaderResolved(
        step_output_kind=str(kind),
        step_output_from=None if source in (None, "output") else str(source),
        step_axis=int(axis) if axis is not None else -1,
        step_output_positions=(
            tuple(int(p) for p in positions) if isinstance(positions, Sequence) else None
        ),
    )
    import torch

    out = _resolve_step_output_op(trace, resolved).out
    if not isinstance(out, torch.Tensor):
        # None (not retained by save=) or a not-yet-materialized payload
        # handle (a .tlspec load inside __setstate__, before blob attach):
        # nothing to compare, never a mismatch.
        return None
    return _derive_step_evidence(trace, resolved, n_rows, check_license=False)


def rederived_evidence_matches(trace: Any, ledger_payload: Mapping[str, Any]) -> bool | None:
    """Cross-check the persisted evidence column against the product's payload.

    ``True``: the column re-derived from the retained root output equals the
    persisted rows (the ledger describes THIS execution's output). ``None``:
    nothing to compare (kind ``none``, incomplete rows -- settlement derives
    no column then -- or the source payload is not retained). ``False``: the
    column differs, or the declaration cannot derive on this product -- the
    ledger does not describe this product (a swapped, grafted, or re-minted
    ledger). Digests bind content; this check binds the content to the
    product's own values.
    """

    from ..errors.episode import EpisodeDeclarationError

    rows = list(ledger_payload.get("rows") or [])
    if not rows or any(
        not isinstance(row, Mapping) or row.get("status") != "complete" for row in rows
    ):
        return None
    header = ledger_payload.get("header") or {}
    try:
        derived = rederive_step_evidence(trace, header, len(rows))
    except EpisodeDeclarationError:
        return False
    if derived is None:
        return None
    persisted = [row.get("step_output") for row in rows]
    normalized = [list(item) if isinstance(item, tuple) else item for item in derived]
    return normalized == [list(item) if isinstance(item, tuple) else item for item in persisted]
