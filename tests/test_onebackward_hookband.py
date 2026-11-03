"""HOOK-BAND spike: torch-version-band assertions for the read engine (F04).

M(reads) section 4's release-gate spike, CPU half: the five torch facts the
one-backward design sits on, asserted on the SUPPORTED band so a torch
upgrade that changes any of them fails loudly here rather than corrupting
reads silently. The CUDA half (auto-B crossover) is the C-READ cluster row
(D02); until it lands, the auto plan's CUDA default stays sequential --
pinned below.

Pure torch: no TorchLens capture in this module.
"""

from __future__ import annotations

import pytest
import torch

from torchlens._utils._torch_compat import HAS_NODE_PREHOOK, get_gradient_edge_support

_HAS_GRADIENT_EDGE = get_gradient_edge_support()

if _HAS_GRADIENT_EDGE:
    from torch.autograd.graph import GradientEdge
else:  # torch < 2.4: GradientEdge absent or non-functional (see get_gradient_edge_support).
    GradientEdge = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    not (_HAS_GRADIENT_EDGE and HAS_NODE_PREHOOK),
    reason="GradientEdge / Node.register_prehook postdate the torch 2.1 floor",
)


def _one_node_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``y = sum(sin(x * 2))`` and return (x, h, m) with live nodes."""

    x = torch.randn(3, requires_grad=True)
    h = x * 2
    m = h.sin()
    return x, h, m


def test_cuda_auto_default_stays_sequential_pending_c_read() -> None:
    """The auto-B CUDA default is pinned to 1 until the C-READ row lands."""

    from torchlens.attribution.onebackward._engine import resolve_batch_plan

    plan = resolve_batch_plan("auto", "cuda")
    assert plan.batch_size == 1
    assert "c_read" in plan.reason
    cpu_plan = resolve_batch_plan("auto", "cpu")
    assert cpu_plan.batch_size > 1
