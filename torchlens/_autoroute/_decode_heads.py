"""Torch-only helpers of the output decoders.

Split from ``_builtin_output``: explicit-head lookup, logits-candidate sniffing,
the Hugging Face text decode, and the metadata normalisers for cache keys.
None of them raises a refusal (those stay in ``_builtin_output``), and the
module imports nothing from torchlens, so it stays out of the package's
import cycle.
"""

from __future__ import annotations

from typing import Any

import torch


def _select_tensor_by_head(outputs: Any, output_head: Any) -> torch.Tensor | None:
    """Select a tensor by explicit live-output head name.

    Parameters
    ----------
    outputs:
        Live model outputs.
    output_head:
        Requested head.

    Returns
    -------
    torch.Tensor | None
        Selected tensor, if found.
    """

    if output_head is None:
        return None
    if isinstance(output_head, str):
        if hasattr(outputs, output_head):
            value = getattr(outputs, output_head)
            if isinstance(value, torch.Tensor):
                return value
        if isinstance(outputs, dict):
            value = outputs.get(output_head)
            if isinstance(value, torch.Tensor):
                return value
        if output_head.isdigit() and isinstance(outputs, (tuple, list)):
            index = int(output_head)
            if 0 <= index < len(outputs) and isinstance(outputs[index], torch.Tensor):
                return outputs[index]
    return None


def _tensor_candidates(
    outputs: Any, *, expected_width: int | None
) -> list[tuple[torch.Tensor, str]]:
    """Collect plausible logits candidates from a live output object.

    Parameters
    ----------
    outputs:
        Live model outputs.
    expected_width:
        Optional expected last dimension.

    Returns
    -------
    list[tuple[torch.Tensor, str]]
        Candidate tensors and head labels.
    """

    candidates: list[tuple[torch.Tensor, str]] = []
    if isinstance(outputs, dict):
        iterable = [(str(key), value) for key, value in outputs.items()]
    elif isinstance(outputs, (tuple, list)):
        iterable = [(str(index), value) for index, value in enumerate(outputs)]
    else:
        iterable = []
    for name, value in iterable:
        if not isinstance(value, torch.Tensor):
            continue
        try:
            if value.is_nested or value.ndim < 2:
                continue
            width = int(value.shape[-1])
        except Exception:
            # Exotic tensor subclasses may refuse metadata reads; they are
            # not logits candidates and must never abort the sniff (R65).
            continue
        if expected_width is None or width == expected_width:
            candidates.append((value, name))
    return candidates


def _safe_int(value: Any) -> int | None:
    """Return ``value`` as int when possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _decode_hf_text(logits: torch.Tensor, meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Decode Hugging Face text logits: argmax token ids, plus text when a tokenizer is present.

    Parameters
    ----------
    logits:
        Live logits tensor.
    meta:
        Detector metadata. Without a usable ``tokenizer`` (one with a
        ``decode`` method) the rows carry the token ids and ``text`` is
        ``None``: the trace was captured without a tokenizer and none was
        passed at read time.

    Returns
    -------
    list[dict[str, Any]] | None
        JSON-primitive rows; ``None`` when the logits are not rank-2 or rank-3.
    """

    tokenizer = meta.get("tokenizer")
    if tokenizer is not None and not hasattr(tokenizer, "decode"):
        tokenizer = None
    logits_copy = logits.detach().clone()
    if logits_copy.ndim == 2:
        # Unbatched `[seq_len, vocab]`: treat as a single batch item.
        logits_copy = logits_copy.unsqueeze(0)
    elif logits_copy.ndim != 3:
        return None
    # `[batch, seq_len, vocab]`: decode each batch item's own token sequence
    # independently, never flattening across the batch dimension.
    token_ids_per_item = torch.argmax(logits_copy, dim=-1).tolist()
    rows: list[dict[str, Any]] = []
    for batch_index, token_ids in enumerate(token_ids_per_item):
        text = None if tokenizer is None else str(tokenizer.decode(token_ids))
        rows.append(
            {
                "batch_item": int(batch_index),
                "rank": 1,
                "text": text,
                "token_ids": [int(i) for i in token_ids],
            }
        )
    return rows


def _normalized_id2label(value: Any) -> dict[int, str] | None:
    """Normalize an id-to-label mapping.

    Parameters
    ----------
    value:
        Candidate mapping.

    Returns
    -------
    dict[int, str] | None
        Normalized mapping.
    """

    if not isinstance(value, dict):
        return None
    normalized: dict[int, str] = {}
    for key, label in value.items():
        try:
            normalized[int(key)] = str(label)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _normalized_mapping(value: Any) -> dict[str, str] | None:
    """Normalize a small string mapping for cache keys."""

    if not isinstance(value, dict):
        return None
    return {str(key): str(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}


def _normalized_sequence(value: Any) -> tuple[str, ...] | None:
    """Normalize a string sequence for cache keys."""

    if not isinstance(value, (list, tuple)):
        return None
    return tuple(str(item) for item in value)


def _safe_str(value: Any) -> str | None:
    """Return ``value`` as str when present."""

    if value is None:
        return None
    return str(value)
