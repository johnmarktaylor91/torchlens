"""Shared Literal aliases for public TorchLens option strings."""

from typing import Literal

OutputDeviceLiteral = Literal["same", "cpu", "cuda"]
VisModeLiteral = Literal["none", "rolled", "unrolled"]
VisNodeModeLiteral = Literal["default", "profiling", "vision", "attention"]
VisDirectionLiteral = Literal["bottomup", "topdown", "leftright"]
VisNodePlacementLiteral = Literal["auto", "dot", "elk", "sfdp"]
VisRendererLiteral = Literal["graphviz", "dagua"]
