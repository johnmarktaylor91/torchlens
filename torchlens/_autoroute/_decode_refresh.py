"""Keep a trace's automatic decoded output in step with its current output values.

The automatic decode pass (``_builtin_output.decode_outputs_for_trace``) runs
once, on the live forward return at the end of a capture, and stores JSON rows
in ``trace._decoded_output``. Any later step that rewrites the output values of
the SAME trace -- a ``do()`` replay committing a perturbed cone, a fast live
run writing a new forward's outputs -- must re-derive those rows, or
``decode_output()``, ``output_table()``, the summary preview and the rendered
output node keep describing the pre-edit forward while ``trace[out].out`` holds
the edited values.

The refresh re-decodes the retained output tensors under the capture's own
resolved postprocessor (same style, head, label bank and ``top_n``), so an
unchanged output yields identical rows. Text rows need the session tokenizer,
which settlement drops: a row whose token id the captured rows already decoded
keeps that text, a token id new to the rows gets ``text=None`` (the id is the
decode; ``decode_output(style="hf_text", tokenizer=...)`` re-renders text).
When the current output cannot be decoded the stale rows are dropped, never
kept, so the read doors recompute or refuse typed.
"""

from __future__ import annotations

from typing import Any

import torch

from torchlens._autoroute._builtin_output import (
    _TOP_N_CAPTURED,
    _decode_selected,
    _retained_output_view,
    _select_logits,
)


def refresh_decoded_output(trace: Any) -> None:
    """Re-derive ``trace._decoded_output`` from the trace's current output values.

    No-op when the capture decoded nothing. Called after a step rewrote output
    values in place; never runs a detector, so the resolved postprocessor
    recorded at capture stays the provenance of the refreshed rows.

    Parameters
    ----------
    trace:
        Trace whose output values were just rewritten.
    """

    previous = trace._decoded_output
    resolved = trace._provenance_output_postprocessor
    if previous is None or resolved is None:
        return
    trace._decoded_output = _redecode_current_output(trace, resolved, previous)


def _redecode_current_output(trace: Any, resolved: Any, previous: Any) -> Any | None:
    """Decode the retained output under ``resolved``; ``None`` when it does not decode."""

    view, _heads = _retained_output_view(trace)
    if view is None:
        return None
    selected = _select_logits(view, resolved, {})
    if selected is None or not isinstance(selected[0], torch.Tensor):
        return None
    top_n = getattr(resolved, "top_n_captured", None) or _TOP_N_CAPTURED
    decoded = _decode_selected(selected[0], resolved, {}, top_n=top_n)
    if resolved.style == "hf_text" and isinstance(decoded, list):
        return _carry_unchanged_text(decoded, previous)
    return decoded


def _carry_unchanged_text(rows: list[dict[str, Any]], previous: Any) -> list[dict[str, Any]]:
    """Keep the captured text of every token id the captured rows already decoded.

    A row's text is the decode of its one token id, so it depends on the id
    alone, never on the row's rank or batch item.
    """

    if not isinstance(previous, list):
        return rows
    captured = {
        row.get("token_id"): row.get("text")
        for row in previous
        if isinstance(row, dict) and row.get("text") is not None
    }
    return [{**row, "text": captured.get(row["token_id"], row["text"])} for row in rows]


__all__ = ("refresh_decoded_output",)
