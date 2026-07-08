"""Regression tests for module root/non-root role swaps (cert10).

A module that is ever prepared as a NON-root submodule permanently kept a
toggle-gated ``forward`` decoration and a root-relative dotted address keyed to
that role. Tracing the SAME module later as its OWN top-level root then crashed
with ``KeyError`` inside ``push_frame`` (the root is never registered in the
per-session module-call dicts), and re-tracing the original container afterward
produced stale addresses for the re-rooted descendant.

These tests exercise both role-swap directions plus the re-trace-again case and
assert that each trace is correct: right layers, right root-relative addresses,
and the root call's ops exactly mirror ``trace.layer_labels`` (no pass-qualified
root ops, so ``validation._check_function_root_module_invariants`` stays armed).
"""

import pytest
import torch
import torch.nn as nn

import torchlens as tl


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


class _Outer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1


def _linear_layer(log: "tl.Trace") -> str:
    matches = [label for label in log.layer_labels if "linear" in label]
    assert matches, f"expected a linear layer in {log.layer_labels}"
    return matches[0]


def _assert_root_call_mirrors_layers(log: "tl.Trace") -> None:
    """The root ``self`` call's ops must equal ``trace.layer_labels`` verbatim.

    This is the module-trace analogue of
    ``validation._check_function_root_module_invariants``: the root op sequence
    is never pass-qualified, so it lines up 1:1 with the trace's layer labels.
    """
    root_call = log["self"]
    assert list(root_call.ops) == list(log.layer_labels), (
        list(root_call.ops),
        list(log.layer_labels),
    )


@pytest.mark.smoke
def test_trace_submodule_as_root_after_container() -> None:
    """(a) outer -> inner-as-root -> outer again all trace correctly."""
    outer = _Outer()
    x = torch.randn(2, 4)

    log_outer = tl.trace(outer, x)
    assert log_outer.layer_labels == [
        "input_1",
        "linear_1_1",
        "relu_1_2",
        "add_1_3",
        "output_1",
    ]
    # inner.lin under outer is addressed relative to the outer root.
    assert log_outer[_linear_layer(log_outer)].modules == ["inner:1", "inner.lin:1"]
    _assert_root_call_mirrors_layers(log_outer)

    # The submodule as its OWN root: previously a KeyError in push_frame.
    log_inner = tl.trace(outer.inner, x)
    assert log_inner.layer_labels == ["input_1", "linear_1_1", "relu_1_2", "output_1"]
    # Now root-relative: the linear op's module is "lin", not "inner.lin".
    assert log_inner[_linear_layer(log_inner)].modules == ["lin:1"]
    _assert_root_call_mirrors_layers(log_inner)

    # Re-tracing the original container must still work AND restore the
    # descendant's root-relative-to-outer address (no leaked "" / "lin" state).
    log_outer_again = tl.trace(outer, x)
    assert log_outer_again.layer_labels == log_outer.layer_labels
    assert log_outer_again[_linear_layer(log_outer_again)].modules == [
        "inner:1",
        "inner.lin:1",
    ]
    _assert_root_call_mirrors_layers(log_outer_again)


@pytest.mark.smoke
def test_trace_submodule_as_root_before_container() -> None:
    """(b) reverse order: inner-as-root first, then the container."""
    outer = _Outer()
    x = torch.randn(2, 4)

    log_inner = tl.trace(outer.inner, x)
    assert log_inner.layer_labels == ["input_1", "linear_1_1", "relu_1_2", "output_1"]
    assert log_inner[_linear_layer(log_inner)].modules == ["lin:1"]
    _assert_root_call_mirrors_layers(log_inner)

    log_outer = tl.trace(outer, x)
    assert log_outer.layer_labels == [
        "input_1",
        "linear_1_1",
        "relu_1_2",
        "add_1_3",
        "output_1",
    ]
    # The submodule's forward must now be (re)decorated so its module boundary is
    # captured -- addresses prefixed by the "inner" container.
    assert log_outer[_linear_layer(log_outer)].modules == ["inner:1", "inner.lin:1"]
    _assert_root_call_mirrors_layers(log_outer)


def test_role_swap_repeated_alternation() -> None:
    """Alternating roles many times stays correct in both directions."""
    outer = _Outer()
    x = torch.randn(2, 4)

    for _ in range(3):
        log_outer = tl.trace(outer, x)
        assert log_outer.layer_labels == [
            "input_1",
            "linear_1_1",
            "relu_1_2",
            "add_1_3",
            "output_1",
        ]
        assert log_outer[_linear_layer(log_outer)].modules == [
            "inner:1",
            "inner.lin:1",
        ]
        _assert_root_call_mirrors_layers(log_outer)

        log_inner = tl.trace(outer.inner, x)
        assert log_inner.layer_labels == [
            "input_1",
            "linear_1_1",
            "relu_1_2",
            "output_1",
        ]
        assert log_inner[_linear_layer(log_inner)].modules == ["lin:1"]
        _assert_root_call_mirrors_layers(log_inner)


def test_independent_models_no_spurious_reprep() -> None:
    """Interleaving two unrelated models must not corrupt either trace.

    Guards the fast path: models whose subtrees never overlap must never mark
    each other stale (a role swap only occurs when a module is re-rooted).
    """
    a = _Outer()
    b = _Outer()
    x = torch.randn(2, 4)

    for _ in range(2):
        log_a = tl.trace(a, x)
        log_b = tl.trace(b, x)
        assert log_a.layer_labels == log_b.layer_labels
        assert log_a[_linear_layer(log_a)].modules == ["inner:1", "inner.lin:1"]
        assert log_b[_linear_layer(log_b)].modules == ["inner:1", "inner.lin:1"]
        _assert_root_call_mirrors_layers(log_a)
        _assert_root_call_mirrors_layers(log_b)
