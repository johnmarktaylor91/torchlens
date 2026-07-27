"""Backend-neutral renderer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..render_ir import RenderIR
from ..request import RenderTarget


class UnsupportedRendererCapabilityError(RuntimeError):
    """Raised when a RenderIR requires a capability absent from its renderer."""


@dataclass(frozen=True)
class RendererCapabilities:
    """Features a renderer can execute without semantic approximation."""

    nested_regions: bool = False
    ordering_constraints: bool = False
    html_labels: bool = False
    layout_execution: bool = False

    def require(self, required: "RendererCapabilities", renderer_name: str) -> None:
        """Validate that every requested renderer feature is supported.

        Parameters
        ----------
        required:
            Capabilities required by a render operation.
        renderer_name:
            Renderer name included in failure diagnostics.

        Raises
        ------
        UnsupportedRendererCapabilityError
            If any required capability is unavailable.
        """

        missing = tuple(
            name
            for name in self.__dataclass_fields__
            if getattr(required, name) and not getattr(self, name)
        )
        if missing:
            raise UnsupportedRendererCapabilityError(
                f"Renderer {renderer_name!r} lacks required capabilities: {', '.join(missing)}"
            )


@dataclass(frozen=True)
class RenderReport:
    """Result of renderer execution, kept separate from immutable RenderIR."""

    source: str
    source_path: Path | None = None
    output_path: Path | None = None


@runtime_checkable
class Renderer(Protocol):
    """Trace-free backend boundary for decision-complete RenderIR."""

    name: str
    capabilities: RendererCapabilities

    def render(self, ir: RenderIR, target: RenderTarget) -> RenderReport:
        """Serialize and optionally execute layout for ``ir``.

        Parameters
        ----------
        ir:
            Decision-complete, host-object-free render description.
        target:
            Output destination and format.

        Returns
        -------
        RenderReport
            Serialized source and output paths produced by the renderer.
        """
