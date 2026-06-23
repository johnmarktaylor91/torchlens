"""Shared Literal aliases for public TorchLens option strings."""

from typing import Literal

OutputDeviceLiteral = Literal["same", "cpu", "cuda"]
BufferVisibilityLiteral = Literal["never", "meaningful", "always"]
CollapseLiteral = Literal["none", "auto", "max"]
VisModeLiteral = Literal["none", "rolled", "unrolled"]
VisNodeModeLiteral = Literal["default", "profiling", "vision", "attention"]
VisDirectionLiteral = Literal["bottomup", "topdown", "leftright"]
VisNodePlacementLiteral = Literal["auto", "dot", "rank"]
VisRendererLiteral = Literal["graphviz", "dagua"]
VisInterventionModeLiteral = Literal["node_mark", "as_node"]
