"""Dumb Graphviz serializer for decision-complete RenderIR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import graphviz

from .. import _render_utils
from ..render_execution import atomic_render_target, surface_layout_stderr
from ..render_ir import RenderIR, RenderIRDotStatement
from ..request import RenderTarget
from .base import RendererCapabilities, RenderReport


class GraphvizRenderer:
    """Serialize ordered IR statements and execute Graphviz layout."""

    name = "graphviz"
    capabilities = RendererCapabilities(
        nested_regions=True,
        ordering_constraints=True,
        html_labels=True,
        layout_execution=True,
        encodings=True,
    )

    def emit(self, ir: RenderIR, dot: graphviz.Digraph) -> None:
        """Append the IR's already-resolved statements to ``dot`` in order.

        Parameters
        ----------
        ir:
            Host-object-free render IR.
        dot:
            Graphviz object receiving serialized statements.
        """

        self.capabilities.require(ir.required_capabilities(), self.name)
        self._emit_statements(dot, ir.dot_statements)

    def render(self, ir: RenderIR, target: RenderTarget) -> RenderReport:
        """Serialize ``ir`` and execute the requested Graphviz layout.

        Parameters
        ----------
        ir:
            Host-object-free render IR.
        target:
            Output destination and format.

        Returns
        -------
        RenderReport
            DOT source and generated artifact locations.
        """

        dot = graphviz.Digraph(
            name=target.graph_name,
            comment=target.graph_comment,
            format=target.fileformat,
        )
        self.emit(ir, dot)
        source_path = Path(dot.save(target.outpath))
        output_path = Path(f"{target.outpath}.{target.fileformat}")
        # Atomic publish + exit-0 stderr surfacing (vizmech D20/D24): a
        # failed layout leaves nothing at the user's path, and the cairo
        # clamp warning is never silently discarded.
        with atomic_render_target(str(output_path)) as temp_output_path:
            completed = _render_utils.run_bounded_subprocess(
                [dot.engine, f"-T{target.fileformat}", "-o", temp_output_path, str(source_path)],
                timeout=target.timeout,
            )
        stderr_text = surface_layout_stderr(completed.stderr, engine=dot.engine)
        return RenderReport(
            dot.source,
            source_path,
            output_path,
            engine=dot.engine,
            layout_stderr=stderr_text,
        )

    def _emit_statements(
        self,
        dot: graphviz.Digraph,
        statements: tuple[RenderIRDotStatement, ...],
    ) -> None:
        """Serialize an ordered statement tuple recursively.

        Parameters
        ----------
        dot:
            Graph or subgraph receiving statements.
        statements:
            Ordered backend-ready statements.
        """

        for statement in statements:
            kwargs: dict[str, Any] = dict(statement.attrs)
            if statement.kind == "node":
                dot.node(*statement.args, **kwargs)
            elif statement.kind == "edge":
                # Edge endpoints may contain intentional Graphviz port separators.
                # Their node-name portions are resolved before this boundary.
                dot.edge(*statement.args, **kwargs)
            elif statement.kind == "attr":
                dot.attr(*statement.args, **kwargs)
            elif statement.kind == "raw":
                dot.body.append(*statement.args)
            else:
                with dot.subgraph(*statement.args, **kwargs) as subgraph:
                    self._emit_statements(subgraph, statement.children)
