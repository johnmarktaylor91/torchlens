"""Lens preset registry-as-data and preset resolution (themes memo, lane C05).

Two axes (themes memo architecture item 1): a SEMANTIC LENS (which question
the picture answers -- which draw settings, channels, and label rows) and a
COSMETIC SKIN (palette and typography, today's ``theme=`` presets in
``themes.py``). Every spelling here is DOCUMENTED-UNSTABLE pending the
naming session; the public draw() kwargs do not consume this registry yet
(the ``theme=``/``skin=`` spelling is a named [UI-SPRINT] fork), so the
substrate ships entry-dark: importable, validated, and pinned by tests.

Design rules implemented from the memo:

- **Registry as inspectable data** (N12): a lens is a frozen record whose
  member names EQUAL draw() parameter names, so :func:`describe` teaches the
  channel API; built-ins are instances, keyed by render surface
  (``"graphviz-draw"`` now; fastlog/html reserved); registration validates at
  registration time; records are plain-serialisable.
- **Defaults-not-overrides** (N1): resolution merges ``lens members <
  explicit user kwargs``, built on the same explicit-field discipline as
  ``options.py`` -- a lens member NEVER overrides a kwarg the caller
  explicitly passed, and ``preset_spec_fn`` rides a dedicated internal slot
  applied BEFORE the user's ``node_spec_fn`` so the lens can carry label
  rows without consuming the user's documented last-word slot.
- **HEADLINE / SECONDARY member semantics** (N2, record half): a lens
  declares which capture-time evidence is its HEADLINE (missing -> typed
  refusal naming the capture remedy) and which members are SECONDARY
  (missing -> degrade with a mandatory rendered notice). The evidence
  checks themselves bind when the perf lenses land (themes items 9-15).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .._errors import InvalidArgumentError
from .request import ResolvedRenderRequest

__all__ = [
    "GRAPHVIZ_DRAW_SURFACE",
    "LensPreset",
    "describe_lens",
    "get_lens",
    "list_lenses",
    "register_lens",
    "resolve_lens_request",
]

#: The one render surface lenses target today. fastlog/html keys reserved.
GRAPHVIZ_DRAW_SURFACE = "graphviz-draw"

#: draw() parameter vocabulary a lens member may set: exactly the resolved
#: render-request fields (their names ARE the draw parameter names).
_DRAW_PARAMETER_NAMES = frozenset(
    request_field.name for request_field in fields(ResolvedRenderRequest)
)


@dataclass(frozen=True)
class LensPreset:
    """One semantic lens: a question, its draw settings, and its honesty terms.

    Attributes
    ----------
    name:
        Registry row name (``"overview"``, ``"blueprint"``, ...).
    question:
        The one question this lens answers, verbatim from the roster.
    members:
        Draw settings the lens applies as DEFAULTS. Keys must be draw()
        parameter names (validated at registration); explicit user kwargs
        always win over members.
    surface:
        Render surface key this row targets.
    headline_evidence:
        Name of the capture-time evidence family whose absence makes the
        picture answer a different question (``None`` for lenses with no
        headline -- overview/blueprint/debug never refuse on evidence).
    secondary_members:
        Member names that DEGRADE with a mandatory rendered notice when
        their evidence is missing; silence is never an option.
    disclosure:
        Mandatory rendered disclosure text fragments this lens carries.
    preset_spec_fn:
        Internal node-spec slot applied BEFORE the user's ``node_spec_fn``
        so a lens can carry its label rows without consuming the user's
        documented last-word slot.
    """

    name: str
    question: str
    members: Mapping[str, Any] = field(default_factory=dict)
    surface: str = GRAPHVIZ_DRAW_SURFACE
    headline_evidence: str | None = None
    secondary_members: tuple[str, ...] = ()
    disclosure: tuple[str, ...] = ()
    preset_spec_fn: Callable[..., Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the plain-serialisable registry row (callables excluded).

        A ``.tlspec`` round-trip renders identically because the row is data:
        every member value is a plain literal, and the one callable slot is
        disclosed by name only.
        """

        return {
            "name": self.name,
            "question": self.question,
            "members": dict(self.members),
            "surface": self.surface,
            "headline_evidence": self.headline_evidence,
            "secondary_members": list(self.secondary_members),
            "disclosure": list(self.disclosure),
            "preset_spec_fn": None
            if self.preset_spec_fn is None
            else getattr(self.preset_spec_fn, "__qualname__", repr(self.preset_spec_fn)),
        }


_REGISTRY: dict[tuple[str, str], LensPreset] = {}


def register_lens(lens: LensPreset) -> LensPreset:
    """Validate and register one lens row; return it for chaining.

    Raises
    ------
    InvalidArgumentError
        If a member name is not a draw() parameter, a secondary member is
        not declared in ``members``, or the (surface, name) key is taken.
    """

    unknown = sorted(set(lens.members) - _DRAW_PARAMETER_NAMES)
    if unknown:
        raise InvalidArgumentError(
            f"lens {lens.name!r} declares members that are not draw() parameters: "
            f"{', '.join(unknown)}",
            code="lens_member_unknown",
            remedy="use draw() parameter names as lens member keys",
            argument="members",
        )
    missing_secondary = sorted(set(lens.secondary_members) - set(lens.members))
    if missing_secondary:
        raise InvalidArgumentError(
            f"lens {lens.name!r} marks SECONDARY members it does not set: "
            f"{', '.join(missing_secondary)}",
            code="lens_secondary_undeclared",
            remedy="declare every SECONDARY member in the lens members mapping",
            argument="secondary_members",
        )
    key = (lens.surface, lens.name)
    if key in _REGISTRY:
        raise InvalidArgumentError(
            f"lens {lens.name!r} is already registered for surface {lens.surface!r}",
            code="lens_name_taken",
            remedy="pick an unregistered lens name or surface",
            argument="name",
        )
    _REGISTRY[key] = lens
    return lens


def get_lens(name: str, surface: str = GRAPHVIZ_DRAW_SURFACE) -> LensPreset:
    """Return one registered lens row, refusing typed with the roster."""

    lens = _REGISTRY.get((surface, name))
    if lens is None:
        roster = ", ".join(sorted(row_name for _, row_name in _REGISTRY)) or "(empty)"
        raise InvalidArgumentError(
            f"no lens named {name!r} is registered for surface {surface!r}; "
            f"registered lenses: {roster}",
            code="lens_unknown",
            remedy="pick a registered lens name",
            argument="name",
        )
    return lens


def list_lenses(surface: str = GRAPHVIZ_DRAW_SURFACE) -> tuple[LensPreset, ...]:
    """Return every registered lens for one surface, name-ordered."""

    return tuple(
        lens for (lens_surface, _), lens in sorted(_REGISTRY.items()) if lens_surface == surface
    )


def describe_lens(name: str, surface: str = GRAPHVIZ_DRAW_SURFACE) -> str:
    """Return the teaching description of one lens: its exact settings.

    The roster TEACHES the channel API (memo N12): every printed member is a
    real draw() parameter with the exact value the lens applies.
    """

    lens = get_lens(name, surface)
    lines = [f"lens {lens.name!r} -- {lens.question}"]
    for member, value in sorted(lens.members.items()):
        marker = " (SECONDARY)" if member in lens.secondary_members else ""
        lines.append(f"  {member}={value!r}{marker}")
    if lens.headline_evidence is not None:
        lines.append(f"  HEADLINE evidence: {lens.headline_evidence}")
    for disclosure in lens.disclosure:
        lines.append(f"  discloses: {disclosure}")
    return "\n".join(lines)


def resolve_lens_request(
    lens: LensPreset | str,
    user_kwargs: Mapping[str, Any],
    surface: str = GRAPHVIZ_DRAW_SURFACE,
) -> dict[str, Any]:
    """Merge lens members under explicit user kwargs (defaults-not-overrides).

    Parameters
    ----------
    lens:
        Registered lens name or record.
    user_kwargs:
        EXPLICITLY-passed draw() kwargs only -- the caller's half of the
        explicit-field discipline (``options.py`` ``_specified_fields``): a
        kwarg the user did not pass must not appear here.
    surface:
        Render surface key.

    Returns
    -------
    dict[str, Any]
        Effective draw kwargs: lens members first, explicit user kwargs
        winning on every conflict. The lens's ``preset_spec_fn`` rides the
        ``"preset_spec_fn"`` key for the internal pre-user slot.
    """

    resolved_lens = lens if isinstance(lens, LensPreset) else get_lens(lens, surface)
    unknown = sorted(set(user_kwargs) - _DRAW_PARAMETER_NAMES)
    if unknown:
        raise InvalidArgumentError(
            f"unknown draw() parameter(s) in explicit kwargs: {', '.join(unknown)}",
            code="lens_user_kwarg_unknown",
            remedy="pass draw() parameter names only",
            argument="user_kwargs",
        )
    effective: dict[str, Any] = dict(resolved_lens.members)
    effective.update(user_kwargs)
    if resolved_lens.preset_spec_fn is not None:
        effective["preset_spec_fn"] = resolved_lens.preset_spec_fn
    return effective


# ---------------------------------------------------------------------------
# Built-in rows (PROVISIONAL: the roster proper is themes item 15; these two
# never-refuse rows exist to prove the machinery and pin the memo's exact
# settings for the default and the everything view).
# ---------------------------------------------------------------------------

OVERVIEW_LENS = register_lens(
    LensPreset(
        name="overview",
        question="what is this model?",
        members={
            "vis_mode": "rolled",
            "collapse": "auto",
            "fold_repeats": None,
            "node_mode": "default",
            "show_buffer_layers": "meaningful",
            "show_containers": "auto",
            "show_legend": None,
        },
        disclosure=(
            "visible/hidden counts",
            "fallback policy BY NAME",
            "the blueprint escape",
        ),
    )
)

BLUEPRINT_LENS = register_lens(
    LensPreset(
        name="blueprint",
        question="show me everything",
        members={
            # Fully PINNED row: today's bare-draw contract by name, so a
            # future default flip cannot silently change it.
            "vis_mode": "unrolled",
            "collapse": "none",
            "fold_repeats": False,
            "show_containers": False,
            "show_buffer_layers": "meaningful",
        },
        disclosure=("may hit the existing typed size/timeout refusal; never silently compacts",),
    )
)
