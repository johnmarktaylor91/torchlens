"""Raise-time branch-kind classification for a scalar escape.

Shared by the structure-only teaching refusal (``structure_only_belt``) and the
plain scalar-escape advisory (``_completeness_dispatch_names``). It is a leaf:
nothing from the package is imported at module level, so both callers import it
at the top. When it lived in ``structure_only_belt``, a cold import of that
module reached ``_completeness_dispatch_names`` through the completeness
modules, which then asked the still-loading belt for this function.
"""

from __future__ import annotations


def _classify_consumer_kind(filename: str, line: int) -> str:
    """Raise-time branch-kind classification (memo sec 2.3).

    Uses the standalone ``ast_branches.classify_bool`` lookup — no
    postprocess dependency; its ``"unknown"`` maps to ``"scalar_escape"`` (an
    escape the classifier cannot tie to a branch context is by definition a
    bare escape). The vocabulary of record is ``BoolConsumer.kind``.
    """

    try:
        from ..._postprocess.ast_branches import classify_bool

        kind = classify_bool(filename, line).kind
    except Exception:  # noqa: BLE001 — classification must never mask the refusal
        return "scalar_escape"
    return "scalar_escape" if kind == "unknown" else kind


def _source_unreadable(filename: str) -> bool:
    """Return whether conditional tracking cannot read ``filename``'s source.

    Step 5 ties a bool read to an ``if`` statement through the AST of the
    file on disk, or of the text ``linecache`` holds for a notebook cell.
    Code compiled from an unregistered string or piped on stdin has neither,
    and a cell name reused for different text fails closed, so no branch in
    it is recorded.

    Parameters
    ----------
    filename:
        Code-object filename of the bool read's user frame.

    Returns
    -------
    bool
        ``True`` when no AST index can be built for ``filename``.
    """

    from ..._postprocess.ast_branches import get_file_index

    return get_file_index(filename) is None
