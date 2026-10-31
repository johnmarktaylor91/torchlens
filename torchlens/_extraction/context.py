"""Batch context, row validation, and user-key validation (extract item 4).

The :class:`BatchContext` is the one per-batch fact carrier the store
pipeline threads through pooling, transforms, and the post-store hook seam
(the plumbing named in extract memo section 8 for future stats/telemetry
sinks). It carries identity and geometry facts — batch/global row ranges,
the ID slice, mask and true extents, position-ID source, device — and NEVER
retains raw examples after commit.

Row validation is the D13 rule: every selected output and auxiliary row
carrier must retain the INPUT batch's row count after postprocessing, or
refuse typed (the motivating regression: a transform's feature axis
reinterpreted as stimuli, completing with the wrong count).

User output keys are validated at CALL time — refuse empty, path
separators, ``..``, NUL, over-length — BEFORE any exporter can turn them
into paths, and :func:`sanitize_key` is the collision-free filesystem
encoding every file-per-key exporter shares.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import torch

from .._errors import InvalidArgumentError

__tl_layer__ = "L5"

__all__ = [
    "BatchContext",
    "sanitize_key",
    "validate_output_keys",
    "validate_row_counts",
]

#: Longest accepted user output key (a bounded key stays renderable in
#: refusal text and exportable as a filename on every mainstream fs).
_MAX_KEY_LENGTH = 200

#: Characters banned from user output keys: path separators and NUL make a
#: key path-traversal-capable the moment an exporter renders it.
_BANNED_KEY_CHARS = ("/", "\\", "\x00")


@dataclasses.dataclass(frozen=True)
class BatchContext:
    """Per-batch fact carrier for the store pipeline (extract D13).

    Attributes
    ----------
    batch_index:
        Zero-based shard/batch index.
    row_start:
        Global row index of this batch's first stimulus.
    row_count:
        INPUT-derived number of stimulus rows (the row-truth).
    stimulus_ids:
        This batch's ordered ID slice, when the run carries IDs.
    mask:
        The collator's two-dimensional attention mask, when supplied.
    row_extents:
        Per-row ``(start, extent)`` gather facts derived from the mask
        (``None`` when no mask exists or a row is non-contiguous).
    padding_side:
        Disclosed pad geometry of this batch: ``"right"`` / ``"left"`` /
        ``"mixed"`` / ``"none"``; ``None`` without a mask.
    position_ids_source:
        ``"derived"`` / ``"caller"`` / ``"model_default"`` — how this
        batch's positions reached the model (extract D5 rule 4).
    device:
        Device the batch was moved to, as a string, when one was requested.
    collate_disclosure:
        The batch envelope's JSON-portable collate disclosure.
    """

    batch_index: int
    row_start: int
    row_count: int
    stimulus_ids: tuple[str, ...] | None
    mask: torch.Tensor | None
    row_extents: tuple[tuple[int, int], ...] | None
    padding_side: str | None
    position_ids_source: str
    device: str | None
    collate_disclosure: Mapping[str, Any]


#: Post-store hook seam (extract memo s8 plumbing): callables invoked after
#: each shard commits, with the batch context and the committed ledger row.
#: PRIVATE; future stats/telemetry sinks register here.
_POST_STORE_HOOKS: list[Callable[[BatchContext, Mapping[str, Any]], None]] = []


def _run_post_store_hooks(context: BatchContext, ledger_row: Mapping[str, Any]) -> None:
    """Invoke every registered post-store hook (best-effort, never fatal).

    Parameters
    ----------
    context:
        The committed batch's context.
    ledger_row:
        The appended ledger row.
    """

    import contextlib

    for hook in list(_POST_STORE_HOOKS):
        # A telemetry sink must never fail a commit.
        with contextlib.suppress(Exception):
            hook(context, ledger_row)


def validate_output_keys(keys: Any) -> None:
    """Validate user output keys at call time (extract item 4).

    Today any string — including path traversal — is accepted verbatim;
    this gate runs BEFORE model movement, directory creation, or any
    forward, so no exporter can ever turn a hostile key into a path.

    Parameters
    ----------
    keys:
        Iterable of user output keys (a mapping ``layers=``'s keys).

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_output_key_invalid`` naming the offending key and the
        exact violated rule (empty, path separator, ``..``, NUL,
        over-length, non-string).
    """

    for key in keys:
        if not isinstance(key, str):
            raise InvalidArgumentError(
                f"Output key {key!r} is {type(key).__name__}, not str; keys "
                "name manifest entries and export members.",
                code="extraction_output_key_invalid",
                remedy="use string output keys",
                key=repr(key),
            )
        problem = None
        if not key:
            problem = "is empty"
        elif any(banned in key for banned in _BANNED_KEY_CHARS):
            problem = "contains a path separator or NUL byte"
        elif ".." in key:
            problem = "contains '..'"
        elif len(key) > _MAX_KEY_LENGTH:
            problem = f"exceeds {_MAX_KEY_LENGTH} characters"
        if problem is not None:
            raise InvalidArgumentError(
                f"Output key {key[:80]!r} {problem}; keys become manifest "
                "entries and exporter filenames, so path-capable or "
                "unbounded keys are refused at call time.",
                code="extraction_output_key_invalid",
                remedy=(
                    "rename the key: non-empty, under "
                    f"{_MAX_KEY_LENGTH} characters, no '/', '\\\\', '..', or NUL"
                ),
                key=key[:80],
            )


def sanitize_key(key: str) -> str:
    """Encode one output key as a collision-free filename stem.

    The encoding is INJECTIVE: every byte outside ``[A-Za-z0-9_.-]`` —
    including ``%`` itself — becomes ``%XX``, so two distinct keys can
    never sanitize to the same stem. Case-insensitive-filesystem collisions
    across a concrete key SET are the exporter's job (it disambiguates and
    records the full map in its contract file).

    Parameters
    ----------
    key:
        Validated user output key.

    Returns
    -------
    str
        Filesystem-safe stem.
    """

    safe = []
    for char in key:
        if char.isascii() and (char.isalnum() or char in "_.-"):
            safe.append(char)
        else:
            safe.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
    return "".join(safe)


def validate_row_counts(context: BatchContext, processed: Mapping[str, torch.Tensor | Any]) -> None:
    """Enforce the D13 row-truth rule on one batch's stored payloads.

    Every selected output must retain the INPUT batch's row count after
    pooling and postprocessing. The historical engine derived ``n_rows``
    from the first output — exactly the fail-open this rule kills (a
    transform reinterpreting a feature axis as stimuli completed with the
    wrong count).

    Parameters
    ----------
    context:
        The batch's context (its ``row_count`` is the truth).
    processed:
        Stored per-key payloads: tensors, or objects exposing
        ``row_count``.

    Raises
    ------
    torchlens.errors.InvalidArgumentError
        ``extraction_row_count_mismatch`` naming every offending key with
        its observed and required counts; refused BEFORE the shard commit.
    """

    offenders: dict[str, int] = {}
    for key, payload in processed.items():
        observed = getattr(payload, "row_count", None)
        if observed is None and isinstance(payload, torch.Tensor):
            observed = int(payload.shape[0]) if payload.ndim >= 1 else 0
        if observed is None:
            observed = -1
        if observed != context.row_count:
            offenders[key] = int(observed)
    if offenders:
        raise InvalidArgumentError(
            f"Batch {context.batch_index} collated {context.row_count} "
            f"stimulus rows, but the stored payloads disagree: "
            f"{offenders} (observed rows per offending key). Row i of every "
            "shard must be stimulus i — a row-count drift after pooling or "
            "postprocessing would mislabel every stored row, so the shard "
            "is refused BEFORE its commit.",
            code="extraction_row_count_mismatch",
            remedy=(
                "keep every transform row-preserving over the stimulus "
                "axis (reduce over non-batch axes only), or fix the "
                "collate row_count"
            ),
            batch_index=context.batch_index,
            expected_rows=context.row_count,
            observed=offenders,
        )
