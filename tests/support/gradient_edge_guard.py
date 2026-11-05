"""The one-backward capability route, shared by every test that reads attributions.

One-backward reads (``tl.attribution``, attribution graphs and everything built
on them) need torch's working ``GradientEdge`` surface, which postdates the
torch 2.1 floor; ``get_gradient_edge_support`` probes it and backs the
``HAS_GRADIENT_EDGE`` flag in the doctor snapshot. On a torch without it the
read refuses typed (``onebackward_torch_unsupported``), and
``tests/test_onebackward_refusals.py::TestTorchBandGate`` asserts that refusal
on every torch, so the tests marked here route by capability rather than
checking a version.
"""

from __future__ import annotations

import pytest

from torchlens._utils._torch_compat import get_gradient_edge_support

requires_gradient_edge = pytest.mark.skipif(
    not get_gradient_edge_support(),
    reason="one-backward reads require torch.autograd.graph.GradientEdge (HAS_GRADIENT_EDGE)",
)
