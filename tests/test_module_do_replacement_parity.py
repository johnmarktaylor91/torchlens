"""A post-capture ``do(tl.module(...))`` is recorded like its capture-time twin.

A capture-time module edit (``intervene=tl.when(tl.module(addr), edit)``) lands
in its own ``interventionreplacement`` op after the module's last op: the last
op keeps its computed value, so a submodule whose output IS that op keeps
reporting the pre-edit value, the way a plain forward hook on the outer module
leaves the inner module's hook untouched. ``fork.do(tl.module(addr), edit)``
used to commit the edit in place at the last op, so the inner submodule
reported the edited value (GPT-2's ``mlp.dropout`` under an ``mlp`` ablation).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions, InterventionOptions

_REPLACEMENT = "intervention_replacement"


class _Nested(nn.Module):
    """Linear, then a (Linear, ReLU) block ``mlp`` whose last op is ``mlp.1``'s, then a head."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.mlp = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(torch.relu(self.inp(x))))


def _model_and_input() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Nested().eval(), torch.randn(2, 8)


def _plain_hook_outputs(model: nn.Module, x: torch.Tensor, ablate: str) -> dict[str, torch.Tensor]:
    """Every module's output from one plain forward with a zeroing hook on ``ablate``."""

    records: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _args, out: torch.Tensor) -> torch.Tensor:
            if name == ablate:
                out = torch.zeros_like(out)
            records[name] = out.detach().clone()
            return out

        return hook

    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            records["__output__"] = model(x)
    finally:
        for handle in handles:
            handle.remove()
    return records


def _forked_module_edit(
    model: nn.Module, x: torch.Tensor, address: str
) -> tuple[tl.Trace, tl.Trace]:
    clean = tl.trace(model, x, save="all", capture=CaptureOptions(replayable=True))
    fork = clean.fork("ablated")
    fork.do(
        tl.module(address),
        tl.zero_ablate(),
        intervention=InterventionOptions(confirm_mutation=True),
    )
    return clean, fork


def _module_outputs(trace: tl.Trace) -> dict[str, torch.Tensor]:
    outs = {str(m.address): m.calls[0].outs[0] for m in trace.modules}
    outs["__output__"] = trace[trace.output_layers[0]].out
    return outs


def _replacement_ops(trace: tl.Trace) -> list:
    return [op for op in trace._layer_list if op.func_name == _REPLACEMENT]


def test_forked_module_edit_matches_plain_hooks_at_every_module_call() -> None:
    model, x = _model_and_input()
    plain = _plain_hook_outputs(model, x, "mlp")
    _clean, fork = _forked_module_edit(model, x, "mlp")

    forked = _module_outputs(fork)
    assert set(forked) == set(plain)
    mismatched = {
        name: float((forked[name] - plain[name]).abs().max())
        for name in plain
        if not torch.allclose(forked[name], plain[name])
    }
    assert mismatched == {}
    # The inner ReLU shares the ablated module's last op and keeps the pre-edit value.
    assert torch.count_nonzero(forked["mlp.1"]) > 0
    assert torch.count_nonzero(forked["mlp"]) == 0


def test_forked_module_edit_has_the_capture_time_shape() -> None:
    model, x = _model_and_input()
    clean, fork = _forked_module_edit(model, x, "mlp")
    live = tl.trace(model, x, intervene=tl.when(tl.module("mlp"), tl.zero_ablate()))

    (replacement,) = _replacement_ops(fork)
    (live_replacement,) = _replacement_ops(live)
    last_op = fork[clean.modules["mlp"].calls[0].output_ops[0]]
    assert replacement.parents == (last_op.label,)
    assert replacement.is_edited is True
    assert replacement.func is None
    assert torch.equal(replacement.out, torch.zeros_like(last_op.out))
    assert torch.equal(last_op.out, clean[last_op.label].out), "the last op keeps its value"
    assert last_op.is_edited is False

    # The edited module (and its enclosing calls) report the replacement; the
    # inner submodule keeps the original op, exactly as the capture-time trace.
    assert list(fork.modules["mlp"].calls[0].output_ops) == [replacement.label]
    assert list(fork.modules["mlp.1"].calls[0].output_ops) == [last_op.label]
    assert list(live.modules["mlp"].calls[0].output_ops) == [live_replacement.label]
    assert (
        list(live.modules["mlp.1"].calls[0].output_ops) == live.modules["mlp.1"].calls[0].output_ops
        and live_replacement.label not in live.modules["mlp.1"].calls[0].output_ops
    )
    assert replacement.output_of_module_calls == live_replacement.output_of_module_calls

    # The replacement is reachable by every label spelling and sits after the last op.
    assert fork[replacement.label] is replacement
    assert fork[replacement.layer_label] is replacement
    index = [op.label for op in fork._layer_list]
    assert index.index(replacement.label) == index.index(last_op.label) + 1
    # Consumers of the module's output now read the replacement.
    for child_label in replacement.children:
        assert replacement.label in fork[child_label].parents
        assert last_op.label not in fork[child_label].parents
    # Every other op keeps its capture label: nothing is renumbered post-capture.
    assert [op.label for op in clean._layer_list] == [
        op.label for op in fork._layer_list if op is not replacement
    ]


def test_a_later_upstream_push_passes_through_the_replacement() -> None:
    model, x = _model_and_input()
    _clean, fork = _forked_module_edit(model, x, "mlp")
    fork.do(
        tl.module("inp"),
        tl.zero_ablate(),
        intervention=InterventionOptions(confirm_mutation=True),
    )
    with torch.no_grad():
        inner = torch.relu(model.mlp[0](torch.relu(torch.zeros(2, 16))))
        expected_output = model.head(torch.zeros(2, 16))

    (replacement,) = _replacement_ops(fork)
    assert torch.allclose(fork.modules["mlp.1"].calls[0].outs[0], inner)
    assert torch.count_nonzero(replacement.out) == 0
    assert torch.allclose(fork[fork.output_layers[0]].out, expected_output)


def test_forked_module_edit_validates_against_the_plain_hook_output() -> None:
    model, x = _model_and_input()
    plain = _plain_hook_outputs(model, x, "mlp")
    _clean, fork = _forked_module_edit(model, x, "mlp")
    assert fork.validate_forward_pass(plain["__output__"]) is True


@pytest.mark.parametrize("address", ["mlp", "mlp.1"])
def test_forked_edit_of_the_innermost_owner_stays_in_place(address: str) -> None:
    """Ablating the submodule that OWNS the last op edits that op: no replacement is needed."""

    model, x = _model_and_input()
    plain = _plain_hook_outputs(model, x, address)
    _clean, fork = _forked_module_edit(model, x, address)
    forked = _module_outputs(fork)
    assert all(torch.allclose(forked[name], plain[name]) for name in plain)
    assert len(_replacement_ops(fork)) == (1 if address == "mlp" else 0)
