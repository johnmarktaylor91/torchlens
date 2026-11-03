"""Guarded fast live run: modules reached by a direct ``module.forward(x)`` call.

A direct ``forward()`` call skips ``Module.__call__`` and so every torch hook,
which the fast live run uses to collect sites, steer and fingerprint. Capture
records the modules it entered that way, and the fast run routes exactly those
through a per-run ``forward`` wrapper; every other module stays on hooks.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.classics_corpus._loader import build_entry, corpus_entries
from torchlens._fast_live_steer import refusal_code
from torchlens._utils._torch_compat import HAS_MODULE_CALL_DISPATCH_FRAME
from torchlens.backends.torch._tl import get_module_meta
from torchlens.errors import PathDivergenceError

pytestmark = pytest.mark.skipif(
    not HAS_MODULE_CALL_DISPATCH_FRAME, reason="torch build cannot tell a direct forward() call"
)


class DirectParent(nn.Module):
    """Reaches its child only by a direct ``forward()`` call."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the child without ``__call__``."""

        return self.inner.forward(x)


class BothWaysParent(nn.Module):
    """Calls its child once through ``__call__`` and once directly."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Hooked call, then a direct call on its result."""

        return self.inner.forward(torch.relu(self.inner(x)))


class Root(nn.Module):
    """Root holding one parent; adds one op after it."""

    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self.parent = parent
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the parent, then the head."""

        return self.head(self.parent(x) + 1)


def _build(parent_type: type[nn.Module]) -> Root:
    """Build a seeded root around one parent type."""

    torch.manual_seed(0)
    return Root(parent_type()).eval()


def _marked(model: nn.Module) -> set[str]:
    """Return the addresses whose module a capture marked as called directly."""

    return {
        name
        for name, module in model.named_modules()
        if (meta := get_module_meta(module)) is not None and meta.direct_forward_called
    }


def _hook_count(model: nn.Module) -> int:
    """Count forward and forward-pre hooks across every module."""

    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _assert_no_routed_forward(model: nn.Module) -> None:
    """Assert no module holds a leftover instance ``forward`` (identity restored)."""

    for module in model.modules():
        assert "forward" not in module.__dict__
        assert module.forward.__func__ is type(module).forward


def _leaves(output: Any) -> list[torch.Tensor]:
    """Flatten a model output into its tensor leaves in a stable order."""

    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, dict):
        return [leaf for key in sorted(output) for leaf in _leaves(output[key])]
    if isinstance(output, (tuple, list)):
        return [leaf for item in output for leaf in _leaves(item)]
    return []


def _assert_same(left: Any, right: Any) -> None:
    """Assert two outputs hold exactly equal tensor leaves."""

    left_leaves, right_leaves = _leaves(left), _leaves(right)
    assert left_leaves and len(left_leaves) == len(right_leaves)
    for want, have in zip(left_leaves, right_leaves, strict=True):
        assert torch.equal(want.detach(), have.detach())


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_plans.py, backends/torch/prehook_provenance.py.
@pytest.mark.parametrize("parent_type", [BothWaysParent])
def test_fast_run_collects_a_direct_forward_site_on_a_new_input(
    parent_type: type[nn.Module],
) -> None:
    """The fast run returns the native output and refreshes the direct-call site."""

    model = _build(parent_type)
    baseline = _hook_count(model)
    trace = tl.trace(model, torch.randn(2, 3))
    new_x = torch.randn(2, 3)
    result = trace.run(inputs=new_x, fast=True)
    product = result.trace
    assert product.last_run["engine"] == "guarded_fast", product.last_run.get("fast_refused")
    with torch.no_grad():
        native = model(new_x)
    _assert_same(native, result.output)
    fresh = tl.trace(model, new_x)
    sites = list(product.find_sites(tl.module("parent.inner")))
    fresh_sites = list(fresh.find_sites(tl.module("parent.inner")))
    assert sites and len(sites) == len(fresh_sites)
    for site, fresh_site in zip(sites, fresh_sites, strict=True):
        assert torch.equal(site.out, fresh_site.out)
    _assert_no_routed_forward(model)
    trace.__dict__["_fast_run_session"].close()
    assert _hook_count(model) == baseline


class HookedParent(nn.Module):
    """Reaches its child through ``__call__`` (the ordinary hooked path)."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the child normally."""

        return self.inner(x)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_plans.py.
@pytest.mark.parametrize("parent_type", [DirectParent])
def test_fast_run_steers_through_a_module(parent_type: type[nn.Module]) -> None:
    """A staged steer fires in the fast run as in the full run, direct call or hooked.

    The default save-all capture gives the root's ``+ 1`` a function plan named
    ``__add__``; the steer's own arithmetic is one intervention-replacement op in
    the graph, so the live session must not read it as that model event.
    """

    model = _build(parent_type)
    x = torch.randn(2, 3)
    baseline = _hook_count(model)
    with torch.no_grad():
        plain = model(x)
    direction = torch.tensor([1.0, -2.0, 0.5])
    trace = tl.trace(model, x)
    trace.attach_hooks(
        tl.module("parent.inner"), tl.steer(direction, strength=3.0, dim=-1), confirm_mutation=True
    )
    trace.run(model, x)
    assert trace.last_run["engine"] == "rerun"
    full = [op.out.clone() for op in trace.output_ops]
    trace.run(model, x)
    assert trace.last_run["engine"] == "guarded_fast", trace.last_run.get("fast_refused")
    assert not trace.last_run.get("hooks_unfired")
    fast = [op.out for op in trace.output_ops]
    _assert_same(full, fast)
    assert not torch.equal(plain, fast[0])
    _assert_no_routed_forward(model)
    assert _hook_count(model) == baseline


def _urbandriver() -> tuple[nn.Module, Any]:
    """Build the corpus model that ends ``return self.embedding.forward(indices)``."""

    (entry,) = [e for e in corpus_entries() if e.id == "urbandriver"]
    return build_entry(entry)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _fast_live_steer.py.
def test_fast_run_routes_urbandriver_past_its_direct_forward_call() -> None:
    """urbandriver's directly called embedding is routed and its sites collected.

    Before routing, the fast run refused at that call
    (``fast_live_module_hook_unobserved``). It now collects the embedding and
    its parent and runs on; a later, unrelated divergence on this model (see
    the strict xfail below) is the only refusal left.
    """

    model, inputs = _urbandriver()
    trace = tl.trace(model, inputs)
    assert any(address.endswith("embedding") for address in _marked(model))
    try:
        trace.run(inputs=inputs, fast=True)
    except PathDivergenceError as exc:
        assert refusal_code(exc) == "conditional_arm_mismatch:fast_live_function_path"
    session = trace.__dict__["_fast_run_session"]
    assert session.module_index >= 2
    _assert_no_routed_forward(model)


@pytest.mark.xfail(
    strict=True,
    raises=PathDivergenceError,
    reason=(
        "after the routed embedding, the live functional path diverges from the "
        "captured function plan around LocalMLP's F.relu(x, inplace=True) "
        "(expected '__getitem__', got 'relu'); not a direct-forward() gap"
    ),
)
def test_fast_run_reruns_urbandriver() -> None:
    """The corpus model reruns fast and equals the full rerun."""

    model, inputs = _urbandriver()
    trace = tl.trace(model, inputs)
    full_trace = tl.trace(model, inputs)
    fast = trace.run(inputs=inputs, fast=True)
    full = full_trace.run(inputs=inputs)
    _assert_same(full.output, fast.output)
