"""The renderer registry: every picture passes one capability gate (C01 item 16).

Architecture memo 6.3 seam 2: renderers are registered units with REQUIRED
capability rows; appliances produce typed display records and ALL pictures
render through registered renderers. This door closes the measured defect
that ``dagua`` CAN bypass the capability gate -- its draw path short-circuits
before ``RenderIR`` exists, so ``RendererCapabilities.require`` never ran.
``Trace.draw`` now resolves EVERY renderer name here and runs the row-based
gate before dispatch; the graphviz deep pipeline keeps its existing
``RendererCapabilities.require`` check on the built ``RenderIR`` (two layers
of the same gate, not a replacement).

Builtins register through the same public function an out-of-tree renderer
uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._registry import (
    TORCHLENS_PROVIDER,
    ProviderInfo,
    RegistrationInfo,
    create_registry,
)
from .renderers.base import UnsupportedRendererCapabilityError

__tl_layer__ = "L7"


@dataclass(frozen=True)
class RendererEntry:
    """One registered renderer's dispatch identity.

    Parameters
    ----------
    name:
        Stable renderer name (the ``vis_renderer=`` spelling).
    kind:
        ``"render_ir"`` for renderers consuming the neutral RenderIR through
        the deep pipeline, ``"trace_direct"`` for renderers with their own
        trace-consuming entry (dagua today).
    """

    name: str
    kind: str


_RENDERER_REGISTRY = create_registry(
    "renderers",
    kind_label="renderer",
    missing_provider_behavior="typed_refusal",
)


def register_renderer(
    name: str,
    entry: RendererEntry,
    *,
    capabilities: Mapping[str, Any],
    provider: ProviderInfo | None = None,
    replace: bool = False,
    conformance_ref: str | None = None,
) -> RegistrationInfo:
    """Register one renderer through the public door (capability rows required)."""

    return _RENDERER_REGISTRY.register(
        name,
        entry,
        capabilities=capabilities,
        provider=provider or TORCHLENS_PROVIDER,
        replace=replace,
        conformance_ref=conformance_ref,
    )


def unregister_renderer(name: str) -> None:
    """Remove one renderer registration."""

    _RENDERER_REGISTRY.unregister(name)


def renderer_names() -> tuple[str, ...]:
    """Return the registered renderer names."""

    return _RENDERER_REGISTRY.list_ids()


def renderer_info(name: str) -> RegistrationInfo:
    """Return one renderer's registration metadata (capability rows included)."""

    return _RENDERER_REGISTRY.info(name)


def resolve_renderer(name: str) -> RendererEntry:
    """Return one registered renderer entry; unknown names refuse teaching."""

    return _RENDERER_REGISTRY.get(name)


def require_renderer_capabilities(name: str, needed: Mapping[str, bool]) -> None:
    """Run the row-based capability gate for one renderer at the draw entry.

    Parameters
    ----------
    name:
        Registered renderer name.
    needed:
        Capability name -> whether this draw request needs it. Only ``True``
        entries are checked.

    Raises
    ------
    UnsupportedRendererCapabilityError
        If the renderer's registered capability rows lack a needed
        capability. Same typed class and stable code
        (``renderer_capability_unsupported``) as the deep RenderIR gate.
    """

    rows = _RENDERER_REGISTRY.info(name).capabilities
    missing = sorted(
        capability
        for capability, required in needed.items()
        if required and not bool(rows.get(capability, False))
    )
    if missing:
        raise UnsupportedRendererCapabilityError(
            f"renderer {name!r} lacks required capabilities: {', '.join(missing)}",
            code="renderer_capability_unsupported",
            renderer=name,
            missing_capabilities=tuple(missing),
        )


# ---------------------------------------------------------------------------
# Builtin registrations, through the same door third parties use.
# ---------------------------------------------------------------------------
register_renderer(
    "graphviz",
    RendererEntry(name="graphviz", kind="render_ir"),
    capabilities={
        "encoding_channels": True,
        "nested_regions": True,
        "rank_groups": True,
        "images": True,
        "formats": ("pdf", "png", "svg", "dot"),
    },
)
register_renderer(
    "dagua",
    RendererEntry(name="dagua", kind="trace_direct"),
    capabilities={
        # Honest rows for the experimental GPU layout engine: the encoding
        # channels refuse today (raise_encoding_dagua_refusal) -- the row is
        # now the authority the draw entry consults.
        "encoding_channels": False,
        "nested_regions": False,
        "rank_groups": False,
        "images": False,
        "formats": ("pdf", "png", "svg"),
        "experimental_opt_in_required": True,
    },
)
