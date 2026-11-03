"""Shared Literal aliases for public TorchLens option strings."""

from collections.abc import Callable
from typing import Literal

import torch

OutputDeviceLiteral = Literal["same", "cpu", "cuda"]
OnNonfiniteLiteral = Literal["ignore", "track", "warn", "raise"]
BufferVisibilityLiteral = Literal["never", "meaningful", "always"]
CollapseLiteral = Literal["none", "auto", "max"] | float
FoldRepeatsLiteral = Literal[None, True, False]
VisModeLiteral = Literal["none", "rolled", "unrolled"]
VisNodeModeLiteral = Literal["default", "profiling"]
VisDirectionLiteral = Literal["bottomup", "topdown", "leftright", "rightleft"]
VisNodePlacementLiteral = Literal["auto", "dot", "rank"]
VisRendererLiteral = Literal["graphviz", "dagua"]
VisInterventionModeLiteral = Literal["node_mark", "as_node"]

#: Postfunc callables applied to saved activations and gradients; private
#: aliases shared by the capture engine and the ``types`` facade.
_ActivationPostfunc = Callable[[torch.Tensor], torch.Tensor]
_GradientPostfunc = Callable[[torch.Tensor], torch.Tensor]

SaveBudgetOption = str | int | float | None
"""Accepted ``save_budget`` spellings: ``"auto"``, a fraction, bytes, or ``None``."""
#: ``Trace.draw(highlight=...)`` values: the one place they are spelled.
DrawHighlightLiteral = Literal["interventions"]
