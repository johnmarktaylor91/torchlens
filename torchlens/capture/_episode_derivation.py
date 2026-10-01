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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._episode_ledger import ResolvedEpisode

__all__ = [
    "mint_capture_digest",
]


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
    source = step_slice.detach().cpu().reshape(-1)
    flat = torch.empty(source.shape, dtype=source.dtype)
    flat.copy_(source)
    hasher = hashlib.sha256()
    hasher.update(str(step_slice.dtype).encode("utf-8"))
    hasher.update(str(tuple(step_slice.shape)).encode("utf-8"))
    if flat.numel():
        hasher.update(flat.view(torch.uint8).numpy().tobytes())
    return f"sha256:{hasher.hexdigest()}"


def _derive_step_evidence(
    trace: Any, resolved: ResolvedEpisode, n_rows: int
) -> list[tuple[int, ...]] | list[str] | None:
    """Derive per-step evidence under the DECLARED kind and source (foldA D8).

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
    n_rows`` special case of the same rule.

    Raises
    ------
    EpisodeDeclarationError
        ``episode_declaration_invalid`` when the declared source is
        ambiguous/unretained/not a tensor, the declared kind disagrees with
        the source dtype (``tokens`` on a float root teaches
        ``digest``/``none``), the axis is out of range, or the source
        carries fewer positions than the episode ran steps.
    """

    import torch

    if resolved.step_output_kind == "none":
        return None
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
    normalized_axis = axis if axis >= 0 else out.dim() + axis
    offset = axis_size - n_rows
    if resolved.step_output_kind == "digest":
        return [
            _step_slice_digest(out.select(normalized_axis, offset + step)) for step in range(n_rows)
        ]
    return [
        tuple(int(v) for v in out.select(normalized_axis, offset + step).reshape(-1).tolist())
        for step in range(n_rows)
    ]


def mint_capture_digest(trace: Any, address: str, started: int) -> str:
    """Mint the capture digest binding an episode ledger to ITS product.

    Hex SHA-256 over the canonical-JSON encoding of the product's
    recomputable identity facts (foldA D7, the F40b middle third): the
    D4-clause facts ordered recorded op labels and stepped-module call
    count, plus the managed entry seed and the stepped-module address, under
    the schema tag ``episode_capture_digest_v1``. Every input persists on
    the product, so a consumer (lane F42's attested coupling) recomputes the
    digest from the product alone and compares -- the positive "this ledger
    belongs to this product" claim (``address``/``started`` come from the
    resolved declaration at settlement and from the persisted header/rows at
    attestation). Root output bytes are deliberately NOT an input: they are
    already disclosed per step through the evidence column, and
    unretained-payload products must still bind. Spelling and construction
    are DOCUMENTED-UNSTABLE (naming session).
    """

    labels = [str(getattr(op, "label", op)) for op in (getattr(trace, "layer_list", None) or ())]
    canonical = json.dumps(
        {
            "schema": "episode_capture_digest_v1",
            "entry_seed": getattr(trace, "random_seed", None),
            "stepped_module": address,
            "stepped_calls": started,
            "op_labels": labels,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
