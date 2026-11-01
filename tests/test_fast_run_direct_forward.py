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
from torchlens._utils._module_call_dispatch import forward_entered_through_module_call
from torchlens._utils._torch_compat import HAS_MODULE_CALL_DISPATCH_FRAME

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
    for want, have in zip(left_leaves, right_leaves):
        assert torch.equal(want.detach(), have.detach())


def test_caller_frame_tells_a_direct_forward_call_apart() -> None:
    """``module(x)`` reads as a module call, ``module.forward(x)`` as direct."""

    import sys

    seen: list[bool] = []

    class Probe(nn.Module):
        def forward(self) -> None:
            seen.append(forward_entered_through_module_call(sys._getframe(1)))

    probe = Probe()
    probe()
    probe.forward()
    assert seen == [True, False]


def test_capture_records_only_modules_entered_by_a_direct_forward_call() -> None:
    """The record holds the directly called module and nothing reached by ``__call__``."""

    x = torch.randn(2, 3)
    direct = tl.trace(_build(DirectParent), x)
    assert direct._module_capture_ws.direct_forward_addresses == {"parent.inner"}  # noqa: SLF001
    both = tl.trace(_build(BothWaysParent), x)
    assert both._module_capture_ws.direct_forward_addresses == {"parent.inner"}  # noqa: SLF001

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    plain = tl.trace(_build(Plain), x)
    assert plain._module_capture_ws.direct_forward_addresses == set()  # noqa: SLF001


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
    assert trace.last_run["engine"] == "guarded_fast"
    with torch.no_grad():
        native = model(new_x)
    _assert_same(native, result.output)
    fresh = tl.trace(model, new_x)
    sites = list(trace.find_sites(tl.module("parent.inner")))
    fresh_sites = list(fresh.find_sites(tl.module("parent.inner")))
    assert sites and len(sites) == len(fresh_sites)
    for site, fresh_site in zip(sites, fresh_sites):
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
    assert fast_trace.last_run["engine"] == "guarded_fast"
    _assert_same(full.output, fast.output)


def test_fast_run_steers_through_a_directly_called_module() -> None:
    """A staged steer at a direct-call site fires in the fast run as in the full run."""

    model = _build(DirectParent)
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
    assert trace.last_run["engine"] == "guarded_fast"
    assert not trace.last_run.get("hooks_unfired")
    fast = [op.out for op in trace.output_ops]
    _assert_same(full, fast)
    assert not torch.equal(plain, fast[0])
    _assert_no_routed_forward(model)
    assert _hook_count(model) == baseline


def test_fast_run_reruns_urbandriver() -> None:
    """The corpus model ending ``return self.embedding.forward(indices)`` reruns fast."""

    (entry,) = [e for e in corpus_entries() if e.id == "urbandriver"]
    model, inputs = build_entry(entry)
    trace = tl.trace(model, inputs)
    assert "embedding" in " ".join(trace._module_capture_ws.direct_forward_addresses)  # noqa: SLF001
    full_trace = tl.trace(model, inputs)
    fast = trace.run(inputs=inputs, fast=True)
    full = full_trace.run(inputs=inputs)
    assert trace.last_run["engine"] == "guarded_fast"
    _assert_same(full.output, fast.output)
    _assert_no_routed_forward(model)
