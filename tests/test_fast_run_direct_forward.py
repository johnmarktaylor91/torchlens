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
from torchlens._utils._module_call_dispatch import forward_entered_through_module_call
from torchlens._utils._torch_compat import HAS_MODULE_CALL_DISPATCH_FRAME
from torchlens.backends.torch._tl import get_module_meta

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


def test_caller_frame_tells_a_direct_forward_call_apart() -> None:
    """``module(x)`` reads as a module call, ``module.forward(x)`` as direct, hooked or not."""

    import sys

    seen: list[bool] = []

    class Probe(nn.Module):
        def forward(self) -> None:
            seen.append(forward_entered_through_module_call(sys._getframe(1)))

    probe = Probe()
    probe()
    probe.forward()
    handle = probe.register_forward_hook(lambda *_: None)
    probe()
    probe.forward()
    handle.remove()
    assert seen == [True, False, True, False]


def test_capture_marks_only_modules_entered_by_a_direct_forward_call() -> None:
    """The mark lands on the directly called module and nothing reached by ``__call__``."""

    x = torch.randn(2, 3)
    direct = _build(DirectParent)
    tl.trace(direct, x)
    assert _marked(direct) == {"parent.inner"}
    both = _build(BothWaysParent)
    tl.trace(both, x)
    assert _marked(both) == {"parent.inner"}

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    plain = _build(Plain)
    tl.trace(plain, x)
    assert _marked(plain) == set()
    # A hooked ``module(x)`` runs forward from torch's nested hooked-call
    # closure; it is still a ``__call__`` call, never marked direct.
    hooked = _build(Plain)
    hooked.parent.inner.register_forward_hook(lambda *_: None)
    hooked.parent.register_forward_pre_hook(lambda *_: None)
    tl.trace(hooked, x)
    assert _marked(hooked) == set()


@pytest.mark.parametrize("parent_type", [DirectParent, BothWaysParent])
def test_fast_run_collects_a_direct_forward_site_on_a_new_input(
    parent_type: type[nn.Module],
) -> None:
    """The fast run returns the native output and refreshes the direct-call site."""

    model = _build(parent_type)
    baseline = _hook_count(model)
    trace = tl.trace(model, torch.randn(2, 3))
    new_x = torch.randn(2, 3)
    result = trace.run(inputs=new_x, fast=True)
    assert trace.last_run["engine"] == "guarded_fast", trace.last_run.get("fast_refused")
    with torch.no_grad():
        native = model(new_x)
    _assert_same(native, result.output)
    fresh = tl.trace(model, new_x)
    sites = list(trace.find_sites(tl.module("parent.inner")))
    fresh_sites = list(fresh.find_sites(tl.module("parent.inner")))
    assert sites and len(sites) == len(fresh_sites)
    for site, fresh_site in zip(sites, fresh_sites, strict=True):
        assert torch.equal(site.out, fresh_site.out)
    _assert_no_routed_forward(model)
    trace.__dict__["_fast_run_session"].close()
    assert _hook_count(model) == baseline


def test_fast_run_matches_the_full_run_on_the_same_input() -> None:
    """Same input: the fast run equals the full (capture-engine) rerun."""

    model = _build(BothWaysParent)
    x = torch.randn(2, 3)
    fast_trace = tl.trace(model, x)
    full_trace = tl.trace(model, x)
    fast = fast_trace.run(inputs=x, fast=True)
    full = full_trace.run(inputs=x)
    assert fast_trace.last_run["engine"] == "guarded_fast", fast_trace.last_run.get("fast_refused")
    _assert_same(full.output, fast.output)


class HookedParent(nn.Module):
    """Reaches its child through ``__call__`` (the ordinary hooked path)."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call the child normally."""

        return self.inner(x)


@pytest.mark.parametrize("parent_type", [DirectParent, HookedParent])
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
