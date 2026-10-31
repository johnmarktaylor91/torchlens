"""The guarded PEP 678 exception-note helper, shared by every crash-note site.

``BaseException.add_note`` exists from Python 3.11. On 3.10 (a supported
version) a bare ``exc.add_note(...)`` raises ``AttributeError``; the crash-note
sites wrapped it in a blanket suppress, so the note telling the user a partial
capture exists silently never appeared there. This helper adds the note where
notes exist and falls back to a ``RuntimeWarning`` carrying the same text where
they do not.
"""

from __future__ import annotations

import warnings

__tl_layer__ = "L1"


def annotate_guarded(exc: BaseException, note: str, *, stacklevel: int = 3) -> None:
    """Attach ``note`` to ``exc``; on Python 3.10 surface it as a warning.

    Parameters
    ----------
    exc:
        The in-flight exception to annotate.
    note:
        Note text.
    stacklevel:
        ``warnings.warn`` stack level for the 3.10 fallback.
    """

    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:  # Python 3.10: no PEP 678 notes -- surface via warning.
        warnings.warn(note, RuntimeWarning, stacklevel=stacklevel)
