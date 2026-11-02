"""Repr-independent discovery of ``torch.return_types`` structseq field names.

The ONLY sanctioned sources for a torch structseq's field names are the TYPE's
``__match_args__`` declaration and, for types that do not declare one, an identity
round-trip over the type's member descriptors. ``repr()`` parsing is FORBIDDEN:
console wrap position injects phantom fields (``dtype=`` / ``grad_fn=`` at line
start), flipping witness verdicts on tensor size alone. Re-exported by
:mod:`torchlens._utils._torch_compat`, the sanctioned cross-version boundary.
"""

from __future__ import annotations

from typing import Any


def torch_structseq_field_names(value: Any) -> tuple[str, ...]:
    """Return the exact ordered public field names of a torch structseq value.

    Parameters
    ----------
    value:
        Candidate ``torch.return_types.*`` (PyStructSequence) instance.

    Returns
    -------
    tuple[str, ...]
        The validated ordered field names, or ``()`` when ``value`` is not a
        fully named torch structseq or its declaration cannot be PROVEN
        (producers refuse such a structseq; runtimes report typed
        unavailability -- never a guess, never a repr parse).
    """

    cls = type(value)
    if cls.__module__ != "torch.return_types" or not isinstance(value, tuple):
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    match_args = getattr(cls, "__match_args__", None)
    if (
        isinstance(match_args, tuple)
        and len(match_args) == n_fields
        and len(set(match_args)) == n_fields
        and all(
            isinstance(name, str) and name.isidentifier() and not name.startswith("_")
            for name in match_args
        )
    ):
        return tuple(str(name) for name in match_args)
    # No usable ``__match_args__``: derive the name -> index bijection by IDENTITY
    # round-trip over the type's descriptors (``getattr(value, name) is
    # value[i]`` for exactly one ``i``). Refuse on any ambiguity or
    # non-bijection -- fail closed, never repr.
    candidates = [
        name
        for name, member in vars(cls).items()
        if not name.startswith("_") and hasattr(member, "__get__") and not callable(member)
    ]
    if len(candidates) != n_fields or len(value) != n_fields:
        return ()
    by_index: dict[int, str] = {}
    for name in candidates:
        try:
            attribute = getattr(value, name)
        except Exception:
            return ()
        matches = [index for index in range(n_fields) if value[index] is attribute]
        if len(matches) != 1 or matches[0] in by_index:
            return ()
        by_index[matches[0]] = name
    if len(by_index) != n_fields:
        return ()
    return tuple(by_index[index] for index in range(n_fields))
