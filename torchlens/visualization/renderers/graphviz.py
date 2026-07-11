"""Dumb Graphviz serializer for decision-complete RenderIR."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import graphviz

from ..render_ir import RenderIR, RenderIRDotStatement
from ..request import RenderTarget
from .base import RenderReport, RendererCapabilities


class GraphvizRenderer:
    """Serialize ordered IR statements and execute Graphviz layout."""

    name = "graphviz"
    capabilities = RendererCapabilities(
        nested_regions=True,
        ordering_constraints=True,
        html_labels=True,
        layout_execution=True,
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

        dot = graphviz.Digraph(format=target.fileformat)
        self.emit(ir, dot)
        source_path = Path(dot.save(target.outpath))
        output_path = Path(f"{target.outpath}.{target.fileformat}")
        subprocess.run(
            [dot.engine, f"-T{target.fileformat}", "-o", str(output_path), str(source_path)],
            check=True,
            capture_output=True,
        )
        return RenderReport(dot.source, source_path, output_path)

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
            else:
                with dot.subgraph(*statement.args, **kwargs) as subgraph:
                    self._emit_statements(subgraph, statement.children)
