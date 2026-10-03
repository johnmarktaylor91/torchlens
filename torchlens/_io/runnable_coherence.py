"""Registry-coherence names for sparse runnable readiness (r83 S3).

A loaded artifact names each computational call twice: the execution authority
``run.callable_registry[].key`` and the op's independently recorded function name.
These helpers decide when the two agree, so readiness can refuse a
self-contradictory artifact without false-refusing legitimate spellings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..intervention.types import FunctionRegistryKey

# Private builtin modules whose callables torch re-exports under a public namespace with
# this name prefix dropped (``torch.fft.rfft is torch._C._fft.fft_rfft``).
_PRIVATE_BUILTIN_PUBLIC_PREFIXES: Mapping[str, str] = {
    "torch._C._fft": "fft_",
    "torch._C._linalg": "linalg_",
    "torch._C._special": "special_",
}


def _normalized_callable_name(name: str | None) -> str | None:
    """Return a callable name in the one spelling both records agree on (r83 S3).

    ``sites[].function_path`` (derived from ``layer_list[].func_name``) and
    ``run.callable_registry[].key.qualname`` are persisted independently and
    agree verbatim for every op measured across dunder, in-place, method and
    function dispatch, with three exceptions where the site keeps the operator
    dunder while the registry records the torch method (``__neg__``/``neg``,
    ``__pow__``/``pow``, and ``__ipow__``/``pow_``). Stripping ordinary operator
    dunder underscores and mapping the one in-place power spelling collapses
    exactly those, and nothing else: ``__iadd__`` and ``relu_`` keep their
    distinguishing characters, so an in-place op can never normalize onto its
    out-of-place sibling.

    Parameters
    ----------
    name:
        Persisted callable name from either record.

    Returns
    -------
    str | None
        Normalized name, or ``None`` when the record states no opinion.
    """

    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped or stripped == "none":
        return None
    if stripped == "__ipow__":
        return "pow_"
    if stripped.startswith("__") and stripped.endswith("__") and len(stripped) > 4:
        return stripped[2:-2]
    return stripped


def _callable_registry_contradiction(
    registry_key: FunctionRegistryKey,
    affected_ops: tuple[str, ...],
    recorded_func_names: Mapping[str, str | None],
) -> tuple[str, str] | None:
    """Return the first op whose recorded name contradicts the registry name (r83 S3).

    The callable a loaded artifact EXECUTES comes from
    ``manifest.json -> run.callable_registry[].key``. The SAME call is named
    independently by ``manifest.json -> sites[].function_path`` and by
    ``metadata.pkl -> layer_list[].func_name`` / ``.func_id.qualname``. Nothing
    reconciled them, so editing one JSON field (``Tensor.__add__`` ->
    ``__sub__``) left an internally SELF-CONTRADICTORY artifact that ran happily
    and reported ``VERIFIED`` with different numbers -- three other persisted
    fields still said ``add``.

    Called from the resolver's success sink, so it fires ONLY for a key that
    RESOLVES to a real, signature-compatible callable -- the S3 threat, a swap
    between two valid ops that would otherwise silently run. An unresolvable or
    signature-incompatible key never reaches here, keeping the resolver's own
    richer readiness/``ReattachError`` path. Compares against the persisted
    registry qualname (not the resolved one) so a version-alias move never
    false-refuses, and normalizes the operator dunder so the measured
    record-spelling divergences (``__neg__``/``neg``, ``__pow__``/``pow``,
    ``__ipow__``/``pow_``) are not read as contradictions. A private torch builtin
    key (``torch._C._fft:fft_rfft``) also accepts its public binding name
    (``rfft``, see ``_registry_name_spellings``), the name the op records. A record
    that states no name (a source op's ``"none"``, a missing label) is no opinion,
    never a contradiction.

    Parameters
    ----------
    registry_key:
        The persisted execution authority for the resolved registry entry.
    affected_ops:
        Op labels driven by this registry entry (``"add_1_2:1"`` form).
    recorded_func_names:
        Per-op function names implied by the REHYDRATED trace, keyed by layer
        label (``"add_1_2"``).

    Returns
    -------
    tuple[str, str] | None
        ``(recorded_func_name, op_label)`` for the first contradicting op, or
        ``None`` when every op that states a name agrees with the authority.
    """

    authorities = {
        name
        for name in map(_normalized_callable_name, _registry_name_spellings(registry_key))
        if name is not None
    }
    if not authorities:
        return None
    for op_label in affected_ops:
        raw_recorded = recorded_func_names.get(op_label.split(":")[0])
        recorded = _normalized_callable_name(raw_recorded)
        if recorded is None or recorded in authorities:
            continue
        # Sanctioned canonicalization pair (round-31 M6, r28 reconcile): the
        # ``Tensor.data`` surface records the canonical ``detach`` callable as
        # its execution authority -- the getter under the ``detach`` op name,
        # the setter (``t.data = rhs``, logged over the RHS only) under the
        # user-facing ``"data"`` op name. A ``data``-named op whose authority
        # is ``detach`` is therefore the artifact's own documented recording,
        # not a self-contradiction. One direction only: any OTHER authority for
        # a ``data`` op, and a ``detach``-named op with a non-detach authority,
        # still refuse.
        if recorded == "data" and "detach" in authorities:
            continue
        return str(raw_recorded), op_label
    return None


def _registry_name_spellings(key: FunctionRegistryKey) -> tuple[str, ...]:
    """Return the callable names an op driven by ``key`` may legitimately record.

    Torch binds each private builtin of ``torch._C._fft`` / ``_linalg`` / ``_special``
    under its public namespace with the namespace prefix dropped
    (``torch.fft.rfft is torch._C._fft.fft_rfft``), and capture records the public
    name as the op's ``func_name``. The registry key keeps the private qualname, so
    both spellings name the same callable. Every other key has exactly one spelling.

    Parameters
    ----------
    key:
        Persisted registry key of the resolved entry.

    Returns
    -------
    tuple[str, ...]
        The registry qualname, plus the public binding name for a private builtin key.
    """

    if key.namespace != "custom" or key.import_path is None:
        return (key.qualname,)
    module_name, separator, qualname = key.import_path.partition(":")
    prefix = _PRIVATE_BUILTIN_PUBLIC_PREFIXES.get(module_name)
    if (
        separator != ":"
        or prefix is None
        or qualname != key.qualname
        or not qualname.startswith(prefix)
        or len(qualname) == len(prefix)
    ):
        return (key.qualname,)
    return (key.qualname, qualname[len(prefix) :])
