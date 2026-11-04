"""Per-step attention pictures from a recorded generation (tviz memo D22).

A generation recorded with ``tl.record(model, input_ids, save=tl.func("softmax"),
generate_kwargs={...})`` keeps every attention softmax each forward ran. The
prefill forward computes the prompt's rows; each cached decode forward
computes exactly one new row, the newest position, against every key so far.
:func:`generation_attention_view` assembles step ``s``'s picture from those
records -- the prefill rows plus one row per decode step up to ``s`` -- into
an :class:`AttentionView` whose episode coordinates name the step. The
existing renderer (``tl.viz.attention``) draws it and outlines the row the
decode step computed. Nothing is recomputed: every row is the softmax output
the model produced at the step that computed that position.

A row computed at an earlier step is shorter than the keys at step ``s``;
the cells to its right are keys that did not exist yet. They hold zero, and
the view carries no mask source unless the caller passes ``mask=`` (D6:
zero-valued cells are never read as a mask).

Spellings are private until the naming ruling (the generation door is
documented-unstable).
"""

from __future__ import annotations

from typing import Any

import torch

from ._errors import refuse
from ._records import AttentionView, EpisodeCoordinates, MaskInfo, TokenAxis

__all__ = ["generation_attention_view", "generation_attention_views"]

#: The function name every eager attention pattern is the output of.
_PATTERN_FUNC = "softmax"

_RECORD_REMEDY = (
    "record the generation with the attention softmax saved: tl.record(model, input_ids, "
    'save=tl.func("softmax"), generate_kwargs={...}), with eager attention '
    '(attn_implementation="eager"); fused kernels never materialize the pattern'
)


def _inside(address: Any, layer: str) -> bool:
    """Return whether a record's nearest module address is ``layer`` or below it."""

    text = "" if address is None else str(address)
    return text == layer or text.startswith(layer + ".")


def _is_pattern(record: Any) -> bool:
    """Return whether a record is a retained four-dimensional softmax output."""

    payload = getattr(record, "ram_payload", None)
    return (
        getattr(record.ctx, "func_name", None) == _PATTERN_FUNC
        and isinstance(payload, torch.Tensor)
        and payload.ndim == 4
    )


def _pattern_layers(generation: Any) -> list[str]:
    """Return the module addresses holding a softmax pattern in the prefill, in order."""

    layers: list[str] = []
    for record in generation.records_at(0):
        address = getattr(record.ctx, "address", None)
        if _is_pattern(record) and address is not None and str(address) not in layers:
            layers.append(str(address))
    return layers


def _require_resident(generation: Any, step: int) -> None:
    """Refuse when a forward up to ``step`` was evicted from memory."""

    evicted = {index: how for index, how in dict(generation.evicted_steps).items() if index <= step}
    if evicted:
        refuse(
            code="tv_payload_missing",
            message=f"Generation steps {sorted(evicted)} were evicted from memory "
            f"({sorted(set(evicted.values()))}); step {step}'s picture needs every row "
            "from the prefill to that step.",
            remedy="record without GenerationOptions(keep_last=), or picture a step "
            "whose whole history is still in memory",
        )


def _step_rows(generation: Any, layer: str, step: int, batch_index: int) -> torch.Tensor:
    """Return one forward's softmax rows for ``layer`` as ``[head, query, key]``."""

    hits = [
        record
        for record in generation.records_at(step)
        if _is_pattern(record) and _inside(record.ctx.address, layer)
    ]
    if len(hits) != 1:
        available = _pattern_layers(generation)
        refuse(
            code="tv_facet_missing" if not hits else "tv_record_invalid",
            message=f"Generation step {step} holds {len(hits)} retained softmax patterns "
            f"inside {layer!r}; a picture needs exactly one.",
            remedy=f"pick one of the pattern-bearing modules {available[:8]}; " + _RECORD_REMEDY,
            available=available,
        )
    payload = hits[0].ram_payload
    if not 0 <= batch_index < payload.shape[0]:
        refuse(
            code="tv_record_invalid",
            message=f"batch_index {batch_index} is outside the recorded batch of "
            f"{payload.shape[0]}.",
            remedy=f"pass batch_index from 0 to {payload.shape[0] - 1}",
        )
    return payload[batch_index].detach().to("cpu", dtype=torch.float32)


def _assembled_pattern(generation: Any, layer: str, step: int, batch_index: int) -> torch.Tensor:
    """Place every row computed up to ``step`` at its absolute position."""

    n_keys = generation.steps[step].positions.stop
    pattern: torch.Tensor | None = None
    for entry in generation.steps[: step + 1]:
        rows = _step_rows(generation, layer, entry.step, batch_index)
        positions = entry.positions
        if rows.shape[1] != len(positions) or rows.shape[2] != positions.stop:
            refuse(
                code="tv_record_invalid",
                message=f"Step {entry.step}'s pattern at {layer!r} has shape "
                f"{tuple(rows.shape)}, but the step computed positions {positions.start} to "
                f"{positions.stop - 1} against {positions.stop} keys; a sliding-window or "
                "compressed cache does not keep one key per earlier position.",
                remedy="picture a model whose cache keeps every earlier key",
            )
        if pattern is None:
            pattern = rows.new_zeros((rows.shape[0], n_keys, n_keys))
        pattern[:, positions.start : positions.stop, : positions.stop] = rows
    if pattern is None:
        refuse(
            code="tv_payload_missing",
            message="This generation recorded no forwards.",
            remedy=_RECORD_REMEDY,
        )
    return pattern


def _axis(tokens: Any, n_keys: int, role: str) -> TokenAxis:
    """Return one token axis over the first ``n_keys`` positions of the sequence."""

    if tokens is None:
        return TokenAxis(role=role, tokens=tuple(str(index) for index in range(n_keys)))
    labels = tuple(str(token) for token in tokens)
    if len(labels) < n_keys:
        refuse(
            code="tv_record_invalid",
            message=f"tokens= has {len(labels)} entries; the picture needs {n_keys} (the "
            "prompt plus every token generated before this step).",
            remedy="pass one label per position of generation.sequences",
        )
    return TokenAxis(role=role, tokens=labels[:n_keys])


def generation_attention_view(  # noqa: PLR0913 -- the attention_view keywords plus step
    generation: Any,
    layer: str,
    step: int,
    *,
    tokens: Any = None,
    batch_index: int = 0,
    mask: torch.Tensor | None = None,
) -> AttentionView:
    """Return one attention layer's pattern as it stood at one generation step.

    Parameters
    ----------
    generation:
        A generation recording (``tl.record(..., generate_kwargs=...)``) that
        saved the attention softmax.
    layer:
        Module address of the attention module.
    step:
        0 for the prefill forward, ``k`` for the k-th decode forward.
    tokens:
        One label per position of ``generation.sequences`` (or at least the
        positions up to this step); defaults to position numbers.
    batch_index:
        Batch element to extract.
    mask:
        Explicit bool ``[n_keys, n_keys]`` mask metadata, ``True`` = masked out.

    Returns
    -------
    AttentionView
        ``[head, query, key]`` over every position up to the step, with
        ``episode`` naming the step; a decode step's row is the last one.
    """

    generation.records_at(step)  # refuses a step that does not exist, naming the range
    _require_resident(generation, step)
    kind = generation.steps[step].kind
    pattern = _assembled_pattern(generation, layer, step, batch_index)
    n_heads, n_keys, _ = pattern.shape
    return AttentionView(
        pattern=pattern,
        query_tokens=_axis(tokens, n_keys, "query"),
        key_tokens=_axis(tokens, n_keys, "key"),
        heads=tuple(range(n_heads)),
        layer=layer,
        provenance="captured",
        mask=None if mask is None else MaskInfo(mask=mask.bool(), source="user_metadata"),
        episode=EpisodeCoordinates(step=step, role=kind),
    )


def generation_attention_views(
    generation: Any,
    step: int,
    *,
    layers: list[str] | None = None,
    tokens: Any = None,
    batch_index: int = 0,
) -> list[AttentionView]:
    """Return every (or the named) attention layer's view at one generation step.

    Parameters
    ----------
    generation:
        A generation recording that saved the attention softmax.
    step:
        0 for the prefill forward, ``k`` for the k-th decode forward.
    layers:
        Module addresses; defaults to every module whose prefill holds a
        softmax pattern.
    tokens:
        One label per position of ``generation.sequences``.
    batch_index:
        Batch element to extract.

    Returns
    -------
    list[AttentionView]
        One view per layer, in execution order.
    """

    if layers is None:
        layers = _pattern_layers(generation)
    if not layers:
        refuse(
            code="tv_facet_missing",
            message="No forward of this generation retained an attention softmax pattern.",
            remedy=_RECORD_REMEDY,
        )
    return [
        generation_attention_view(generation, layer, step, tokens=tokens, batch_index=batch_index)
        for layer in layers
    ]
