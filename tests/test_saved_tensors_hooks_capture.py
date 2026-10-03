"""r32-abc Fix A pins: saved-tensors pack/unpack hooks must not corrupt capture.

Round-34 Finding A (MED, pre-existing at e7f036fe): a torch op executed inside a
``torch.autograd.graph.saved_tensors_hooks`` PACK hook fires DURING the producing
user op's dispatch, before that op is logged. Two graph corruptions followed:

* a same-object return (``t.cpu()`` on a CPU tensor) labeled the producer's
  not-yet-logged output, so the producer (relu) was silently DELETED from the
  graph (``_output_should_be_logged`` judged it a higher-level wrapper) and the
  hook op survived as a parentless node;
* a pack over an already-labeled tensor (linear packing its input) spliced the
  hook op INTO the forward dataflow (input -> cpu -> linear).

Hook results feed autograd's saved-for-backward storage, never the forward
dataflow, so the honest forward-trace model is to scope user hook bodies as
autograd-internal bookkeeping: ``saved_tensors_hooks.__init__`` is patched at
wrap time (feature-detected via ``HAS_SAVED_TENSORS_HOOKS_PATCHABLE``) so hook
bodies run with capture logging paused on the owner thread. ``save_on_cpu`` and
the non-reentrant checkpoint hook subclass the same class and are covered.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.utils._torch_compat import HAS_SAVED_TENSORS_HOOKS_PATCHABLE

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

EXPECTED_LABELS = ["input_1", "linear_1_1", "relu_1_2", "mul_1_3", "output_1"]


class OffloadModel(nn.Module):
    """Standard activation-offload idiom: pack/unpack hooks around linear+relu."""

    def __init__(self, pack: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.pack = pack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t):
            y = self.lin(x).relu()
        return y * 2.0


class SaveOnCpuModel(nn.Module):
    """The built-in ``save_on_cpu`` subclass of ``saved_tensors_hooks``."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autograd.graph.save_on_cpu():
            y = self.lin(x).relu()
        return y * 2.0


def _assert_clean_offload_graph(trace: tl.Trace) -> None:
    """The forward graph must be exactly input -> linear -> relu -> mul -> output."""
    assert trace.layer_labels == EXPECTED_LABELS
    assert trace["relu_1_2"].parents == ("linear_1_1",)
    assert trace["relu_1_2"].children == ("mul_1_3",)
    assert trace["linear_1_1"].parents == ("input_1",)
    for label in trace.layer_labels:
        op = trace[label]
        if label != "input_1":
            assert op.parents, f"{label} is parentless"


@pytest.mark.parametrize(
    "pack_name",
    ["cpu_same_object", "clone_new_object", "identity_no_op"],
)
def test_pack_hook_op_does_not_delete_producer(pack_name: str) -> None:
    """A pack-hook torch op must never cannibalize the producing user op.

    ``t.cpu()`` on a CPU tensor is the same-object vehicle that deleted relu;
    ``t.clone()`` is the new-object control that used to leave a dangling clone
    node; identity executes no op at all. All three must yield the identical
    clean forward graph, a real-output match, and validate True.
    """
    packs: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "cpu_same_object": lambda t: t.cpu(),
        "clone_new_object": lambda t: t.clone(),
        "identity_no_op": lambda t: t,
    }
    torch.manual_seed(0)
    model = OffloadModel(packs[pack_name]).eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        real = model(x)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    _assert_clean_offload_graph(trace)
    assert torch.equal(trace["output_1"].out, real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True


def test_save_on_cpu_subclass_is_covered() -> None:
    """``save_on_cpu`` routes through the patched base-class ``__init__``."""
    torch.manual_seed(0)
    model = SaveOnCpuModel().eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        real = model(x)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    _assert_clean_offload_graph(trace)
    assert torch.equal(trace["output_1"].out, real)


def test_pack_hook_still_functions_after_capture() -> None:
    """Scoping must not break the hooks themselves: backward still works."""
    torch.manual_seed(0)
    model = OffloadModel(lambda t: t.cpu())
    x = torch.randn(2, 4, requires_grad=True)
    tl.trace(model, x.detach(), capture=tl.options.CaptureOptions(layers_to_save="all"))
    y = model(x).sum()
    y.backward()
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


def test_unwrap_restores_saved_tensors_hooks_init() -> None:
    """``unwrap_torch`` restores the original ``__init__``; rewrap re-installs."""
    if not HAS_SAVED_TENSORS_HOOKS_PATCHABLE:
        pytest.skip("saved_tensors_hooks init not patchable on this torch runtime")
    from torchlens.backends.torch import backward as bwd
    from torchlens.backends.torch.wrappers import unwrap_torch

    torch.manual_seed(0)
    tl.trace(OffloadModel(lambda t: t.cpu()).eval(), torch.randn(2, 4))
    assert bwd._SAVED_TENSORS_HOOKS_INIT_PATCHED
    try:
        unwrap_torch()
        assert not bwd._SAVED_TENSORS_HOOKS_INIT_PATCHED
        assert (
            torch.autograd.graph.saved_tensors_hooks.__init__
            is bwd._ORIGINAL_SAVED_TENSORS_HOOKS_INIT
        )
        assert (
            torch.autograd.graph.saved_tensors_hooks.__enter__
            is bwd._ORIGINAL_SAVED_TENSORS_HOOKS_ENTER
        )
    finally:
        # The next capture auto-rewraps; confirm the patch reinstalls cleanly.
        trace = tl.trace(OffloadModel(lambda t: t.cpu()).eval(), torch.randn(2, 4))
        assert bwd._SAVED_TENSORS_HOOKS_INIT_PATCHED
        _assert_clean_offload_graph(trace)


class PrebuiltCtxModel(nn.Module):
    """A hook context constructed in module ``__init__`` and reused per-forward."""

    def __init__(self, ctx: torch.autograd.graph.saved_tensors_hooks) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.ctx = ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with self.ctx:
            y = self.lin(x).relu()
        return y * 2.0


def test_prebuilt_hook_context_is_scoped_at_use_time() -> None:
    """PIN (round-35 R1): a context built BEFORE the first capture is covered.

    The round-5 fix scoped hooks in ``saved_tensors_hooks.__init__``, so an
    instance constructed before ``wrap_torch()`` ran (a model storing the
    context in its own ``__init__`` is a legal idiom) carried raw hooks
    forever, and the same-object ``t.cpu()`` pack op re-deleted relu. The
    ``__enter__`` patch re-scopes hooks at use time, so construction order no
    longer matters. Simulate pre-wrap construction faithfully by building the
    context while torch is unwrapped (the original ``__init__`` is live).
    """
    if not HAS_SAVED_TENSORS_HOOKS_PATCHABLE:
        pytest.skip("saved_tensors_hooks not patchable on this torch runtime")
    from torchlens.backends.torch.wrappers import unwrap_torch

    unwrap_torch()
    ctx = torch.autograd.graph.saved_tensors_hooks(lambda t: t.cpu(), lambda t: t)
    assert not getattr(ctx.pack_hook, "__tl_saved_tensors_hook_scoped__", False)

    torch.manual_seed(0)
    model = PrebuiltCtxModel(ctx).eval()
    x = torch.randn(2, 4)
    with torch.no_grad():
        real = model(x)
    # The next capture auto-rewraps torch and must scope the pre-built hooks.
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    _assert_clean_offload_graph(trace)
    assert torch.equal(trace["output_1"].out, real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True


def test_checkpoint_still_validates_with_hook_scope() -> None:
    """Non-reentrant checkpoint subclasses saved_tensors_hooks; F-2 must hold."""

    class CkptModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.utils.checkpoint.checkpoint(
                lambda t: self.lin(t).relu(), x, use_reentrant=False
            )

    torch.manual_seed(0)
    model = CkptModel()
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    func_names = [getattr(layer, "func_name", None) for layer in trace.layer_list]
    assert "linear" in func_names and "relu" in func_names
    assert func_names.count("linear") == 1, "phantom checkpoint recompute op"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True
