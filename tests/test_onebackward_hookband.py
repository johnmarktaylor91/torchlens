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

from torchlens.utils._torch_compat import HAS_NODE_PREHOOK, get_gradient_edge_support

_HAS_GRADIENT_EDGE = get_gradient_edge_support()

if _HAS_GRADIENT_EDGE:
    from torch.autograd.graph import GradientEdge
else:  # torch < 2.4: GradientEdge absent or non-functional (see get_gradient_edge_support).
    GradientEdge = None  # type: ignore[assignment,misc]

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not (_HAS_GRADIENT_EDGE and HAS_NODE_PREHOOK),
        reason="GradientEdge / Node.register_prehook postdate the torch 2.1 floor",
    ),
]


def _one_node_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``y = sum(sin(x * 2))`` and return (x, h, m) with live nodes."""

    x = torch.randn(3, requires_grad=True)
    h = x * 2
    m = h.sin()
    return x, h, m


def test_band_flags_are_true_on_the_supported_band() -> None:
    """The two capability flags hold on every torch this suite runs under."""

    assert _HAS_GRADIENT_EDGE, "GradientEdge disappeared from torch.autograd.graph"
    assert HAS_NODE_PREHOOK, "Node.register_prehook disappeared"


def test_gradient_edge_input_seeding() -> None:
    """autograd.grad accepts GradientEdge INPUTS (the read's addressing)."""

    x, h, m = _one_node_graph()
    (grad,) = torch.autograd.grad(
        m.sum(), [GradientEdge(h.grad_fn, h.output_nr)], retain_graph=True
    )
    assert torch.allclose(grad, h.detach().cos())


def test_gradient_edge_output_seeding_with_cotangent() -> None:
    """autograd.grad accepts GradientEdge OUTPUTS with an explicit cotangent."""

    x, h, m = _one_node_graph()
    (grad,) = torch.autograd.grad(
        [GradientEdge(h.grad_fn, h.output_nr)],
        [x],
        grad_outputs=[torch.ones(3)],
        retain_graph=True,
    )
    assert torch.allclose(grad, torch.full((3,), 2.0))


def test_is_grads_batched_composes_with_edge_seeding() -> None:
    """Stacked cotangents batch through an edge-seeded VJP."""

    x, h, m = _one_node_graph()
    cotangents = torch.stack([torch.ones(3), 2 * torch.ones(3)])
    (grad,) = torch.autograd.grad(
        [GradientEdge(h.grad_fn, h.output_nr)],
        [x],
        grad_outputs=[cotangents],
        is_grads_batched=True,
        retain_graph=True,
    )
    assert grad.shape == (2, 3)
    assert torch.allclose(grad[1], 2 * grad[0])


def test_materialize_grads_raise_condition_pin() -> None:
    """materialize_grads=True REFUSES GradientEdge inputs; False serves mixed.

    Stronger than the panel's mixed-population claim: torch refuses the
    combination outright, so ``False`` cannot be "simplified" away -- silent
    zero-materialization (the 438-fabricated-rows counterfactual) is
    unreachable through this engine.
    """

    x, h, m = _one_node_graph()
    unrelated = torch.randn(2, requires_grad=True) + 1
    edge = GradientEdge(h.grad_fn, h.output_nr)
    unreachable_edge = GradientEdge(unrelated.grad_fn, unrelated.output_nr)
    with pytest.raises(RuntimeError, match="materialize_grads"):
        torch.autograd.grad(
            m.sum(),
            [edge, unreachable_edge],
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
    reachable, unreachable = torch.autograd.grad(
        m.sum(),
        [edge, unreachable_edge],
        retain_graph=True,
        allow_unused=True,
        materialize_grads=False,
    )
    assert reachable is not None
    assert unreachable is None, "not-upstream must surface as None, never zeros"


def test_mixed_cone_batching_fabricates_zero_rows() -> None:
    """The measured hazard that forces cone-grouped chunks (engine design).

    Two targets with disjoint cones in ONE batched call: the not-upstream
    (target, input) pair comes back as a ZERO ROW, not None --
    indistinguishable from a real zero. If torch ever changes this to
    per-row None, the engine's cone grouping can be relaxed; until then this
    pin documents why chunks group by cone.
    """

    a = torch.randn(2, requires_grad=True)
    b = torch.randn(2, requires_grad=True)
    pa, pb = a * 3.0, b * 5.0
    ta, tb = pa.sum(), pb.sum()
    grads = torch.autograd.grad(
        [GradientEdge(ta.grad_fn, 0), GradientEdge(tb.grad_fn, 0)],
        [GradientEdge(pa.grad_fn, 0), GradientEdge(pb.grad_fn, 0)],
        grad_outputs=[torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        is_grads_batched=True,
        retain_graph=True,
        allow_unused=True,
    )
    assert grads[0] is not None and grads[1] is not None
    assert torch.equal(grads[0][1], torch.zeros(2)), (
        "torch now returns something other than fabricated zeros for the "
        "not-upstream row of a mixed-cone batch -- revisit engine cone grouping"
    )
    assert torch.equal(grads[1][0], torch.zeros(2))


def test_node_prehook_slot_zeroing_and_bitexact_removal() -> None:
    """Prehooks zero downstream flow only; removal restores bit-exact.

    The frozen-site's OWN edge still reads its incoming gradient (the D6
    documented semantics: post-freeze applies to sites UPSTREAM of the
    frozen node).
    """

    x, h, m = _one_node_graph()
    target = m.sum()
    edge_h = GradientEdge(h.grad_fn, h.output_nr)
    (own_before,) = torch.autograd.grad(target, [edge_h], retain_graph=True)
    (x_before,) = torch.autograd.grad(target, [x], retain_graph=True)
    handle = h.grad_fn.register_prehook(
        lambda grads: tuple(g * 0 if g is not None else None for g in grads)
    )
    try:
        (own_frozen,) = torch.autograd.grad(target, [edge_h], retain_graph=True)
        (x_frozen,) = torch.autograd.grad(target, [x], retain_graph=True)
    finally:
        handle.remove()
    assert torch.equal(own_frozen, own_before), (
        "a frozen site's own row keeps its incoming gradient"
    )
    assert torch.equal(x_frozen, torch.zeros(3)), "upstream flow must be stopped"
    (own_after,) = torch.autograd.grad(target, [edge_h], retain_graph=True)
    (x_after,) = torch.autograd.grad(target, [x], retain_graph=True)
    assert torch.equal(own_after, own_before)
    assert torch.equal(x_after, x_before), "hook removal must restore bit-exact"


def test_cuda_auto_default_stays_sequential_pending_c_read() -> None:
    """The auto-B CUDA default is pinned to 1 until the C-READ row lands."""

    from torchlens.attribution.onebackward._engine import resolve_batch_plan

    plan = resolve_batch_plan("auto", "cuda")
    assert plan.batch_size == 1
    assert "c_read" in plan.reason
    cpu_plan = resolve_batch_plan("auto", "cpu")
    assert cpu_plan.batch_size > 1
