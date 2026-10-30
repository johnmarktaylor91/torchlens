"""Facet families: facets a registered codec (an SAE's latents) serves on its home op.

A family is registered under a name by ``torchlens.bridge.sae.register`` and
consulted live by every ``FacetView``, so a family registered after capture
is visible on existing traces. Its facets are spelled ``"<name>"`` (every
latent), ``"<name>[k]"`` (one) and ``"<name>[j,k]"`` or ``"<name>[k,]"``
(several, as one write). A family object provides ``facet_spec(record,
index)`` (``None`` off its home, coded refusals for a home it cannot place
or a latent it lacks), ``listed_spec(record)`` (never raises) and
``selector_name(index)`` (validates a selector spelling).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .facets import FacetSpec

#: Registered families by name (filled by ``torchlens.bridge.sae.register``).
_FACET_FAMILIES: dict[str, Any] = {}
_FAMILY_KEY = re.compile(r"^(?P<base>[^\[\],\s]+)(?:\[(?P<index>\d+(?:,\d+)*,?)\])?$")


def family_key(name: Any) -> tuple[Any, int | tuple[int, ...] | None] | None:
    """Return ``(family, index)`` when ``name`` spells a registered family facet.

    ``"res"`` is every latent (index ``None``), ``"res[3]"`` latent 3 and
    ``"res[3,7]"`` (or ``"res[3,]"``) the listed latents as one tuple.
    """

    match = _FAMILY_KEY.match(name) if isinstance(name, str) else None
    family = _FACET_FAMILIES.get(match["base"]) if match is not None else None
    if match is None or family is None:
        return None
    raw = match["index"]
    if raw is None:
        return family, None
    parts = tuple(int(part) for part in raw.rstrip(",").split(","))
    return family, parts if "," in raw else parts[0]


def family_spec(record: Any, name: Any) -> FacetSpec | None:
    """Return the family facet ``name`` names on ``record``, or ``None`` off its home."""

    key = family_key(name)
    if key is None:
        return None
    family, index = key
    return cast("FacetSpec | None", family.facet_spec(record, index))


def family_entries(record: Any) -> dict[str, FacetSpec]:
    """Return every registered family's all-latents facet whose home is ``record``."""

    entries: dict[str, FacetSpec] = {}
    for family_name, family in list(_FACET_FAMILIES.items()):
        spec = family.listed_spec(record)
        if spec is not None:
            entries[family_name] = spec
    return entries


def family_facet_name(base: Any, index: Any) -> str:
    """Return the facet name of latents ``index`` of family ``base``, validated.

    Parameters
    ----------
    base:
        Registered family name (the name an SAE was registered under).
    index:
        A latent index, or a non-empty sequence of them.

    Returns
    -------
    str
        ``"<base>[k]"`` or ``"<base>[j,k]"``.

    Raises
    ------
    InvalidArgumentError
        ``sae_latent_family_unknown`` when no family has that name; the
        family's own refusals for an index it does not have.
    """

    from .._errors import InvalidArgumentError

    family = _FACET_FAMILIES.get(base) if isinstance(base, str) else None
    if family is None:
        registered = sorted(_FACET_FAMILIES)
        raise InvalidArgumentError(
            f"tl.facet({base!r}).latent(...) names no registered SAE; registered SAEs: "
            f"{registered if registered else 'none'}",
            code="sae_latent_family_unknown",
            remedy=f"register the SAE under that name first, "
            f"tl.bridge.sae.register({base!r}, sae, site='<site>'), or use one of the "
            "registered names; see docs/field-guide/pretrained_saes.md",
        )
    return cast(str, family.selector_name(index))
