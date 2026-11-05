"""The one never-fired ``stop_after`` policy, shared by torch and the previews.

A selector-shaped site that never fired is a typo-shaped wrong result (the full
forward ran; the caller asked for a frontier) and refuses typed with
``stop_after_never_fired``. An exploratory callable or an ambient
context-manager site legitimately may never fire: it warns with
``stop_after_never_fired_callable`` and leaves a durable ledger entry.

The torch door reaches this from the finalize epilogue with its provenance
facts; the preview backends, which never pass the torch door, call it with the
same facts derived from the site they halt on. This module imports only the
error types, so it stays a leaf of the import graph.
"""

from __future__ import annotations

import warnings
from typing import Any

from .._errors import InvalidArgumentError
from ..errors import TorchLensWarning


def settle_unfired_stop_after(
    trace: Any,
    stop_after_site: Any,
    *,
    selector_shaped: bool,
    ambient: bool = False,
    stacklevel: int = 5,
) -> None:
    """Refuse (``InvalidArgumentError``) or warn when ``stop_after`` never fired.

    Parameters
    ----------
    trace
        Trace just captured; a halted trace (``is_stopped``) passes silently.
    stop_after_site
        The requested ``stop_after`` site, or ``None`` when none was requested.
    selector_shaped
        Whether the site is a selector or a selector string (not a callable).
    ambient
        Whether the site came from an ambient context manager.
    stacklevel
        Warning stack level, counted from this function.
    """
    # brainpipe D-16.
    if stop_after_site is None or bool(getattr(trace, "is_stopped", False)):
        return
    site_repr = repr(stop_after_site)
    if selector_shaped and not ambient:
        raise InvalidArgumentError(
            f"stop_after={site_repr} never fired: the capture ran the full "
            "forward and completed without halting, so the result is not the "
            "requested frontier",
            code="stop_after_never_fired",
            remedy=(
                "check the site against the model's module addresses "
                "(model.named_modules()) or use tl.func(...)/tl.module(...); "
                "an exploratory predicate that may legitimately never fire "
                "belongs in a callable"
            ),
            argument="stop_after",
        )
    stop_annotations = getattr(trace, "annotations", None)
    if isinstance(stop_annotations, dict):
        stop_annotations.setdefault("unmatched_capture_selectors", []).append(
            {"slot": "stop_after", "selector": site_repr}
        )
    warnings.warn(
        TorchLensWarning(
            f"stop_after={site_repr} never fired; the capture ran the full "
            "forward and the outputs are the model's real outputs, not a "
            "frontier. Remedy: check the stop_after site against the executed "
            "model, or drop stop_after",
            code="stop_after_never_fired_callable",
        ),
        stacklevel=stacklevel,
    )
