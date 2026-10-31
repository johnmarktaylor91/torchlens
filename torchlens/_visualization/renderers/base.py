"""Backend-neutral renderer contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from ..._errors import _actionable_message, _ActionableErrorMixin
from ...errors._base import ConfigurationError
from ..render_ir import RenderIR
from ..request import RenderTarget


class UnsupportedRendererCapabilityError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when a RenderIR requires a capability absent from its renderer.

    Keeps its historical ``RuntimeError`` base while joining the taxonomy
    with a stable code and a default remedy, so single-message raise sites
    stay valid.
    """

    code: str = "renderer_capability_unsupported"
    default_remedy: str = (
        "render with the graphviz renderer or drop the option that requires the missing capability"
    )

    def __init__(
        self,
        problem: str,
        *,
        remedy: str | None = None,
        code: str | None = None,
        **context: object,
    ) -> None:
        """Initialize an actionable renderer-capability refusal.

        Parameters
        ----------
        problem:
            Description of the renderer and its missing capabilities.
        remedy:
            Concrete caller action. The class default is used when omitted.
        code:
            Stable refusal code; the class attribute when omitted. Raise
            sites may spell it inline so the S-17 census sees the code at
            the site.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        resolved_remedy = remedy or type(self).default_remedy
        super().__init__(
            _actionable_message(problem, resolved_remedy),
            code=code or type(self).code,
            remedy=resolved_remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class RendererCapabilities:
    """Features a renderer can execute without semantic approximation.

    ``encodings`` is the N14 capability bit (themes memo item 6): whether the
    renderer can execute value-encoding channels (color_by/size_by ramps and
    legends). "Refuse by capability, never draw an unencoded imitation" needs
    the vocabulary to say so.
    """

    nested_regions: bool = False
    ordering_constraints: bool = False
    html_labels: bool = False
    layout_execution: bool = False
    encodings: bool = False

    def require(self, required: RendererCapabilities, renderer_name: str) -> None:
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
    """Result of renderer execution, kept separate from immutable RenderIR.

    ``engine`` and ``layout_stderr`` disclose what actually executed
    (vizmech D24): the engine binary invoked and everything it wrote to
    stderr even on exit 0 (the cairo clamp warning class).
    """

    source: str
    source_path: Path | None = None
    output_path: Path | None = None
    engine: str = ""
    layout_stderr: str = ""


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
