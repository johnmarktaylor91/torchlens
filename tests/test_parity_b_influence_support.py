"""Shared checks for gradient influence probes on the five non-torch backends.

Each backend's ``tests/test_parity_b_influence_<backend>.py`` builds a :class:`Kit`
around the feature-table subject (a 3x3 conv, a ReLU, a flatten and a dense head
on one 6x6 image) and runs the checks below. The expected values never come from
TorchLens: they are the framework's own gradient of the model, central finite
differences through the model, and the PyTorch probe on a twin model holding the
same weights. This module defines no tests of its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from torchlens.receptive_field._gradient import _backward_event_count, gradient_for_unit
from torchlens.receptive_field._gradient_forward import projective_gradient_for_unit

#: The two output units of the subject's dense head.
OUTPUT_UNITS = ((0, 0), (0, 1))


@dataclass(frozen=True)
class Kit:
    """One backend's subject plus the framework-side oracles the checks compare against."""

    trace: Any
    x: np.ndarray
    center: tuple[int, ...]
    activation: str
    unit_gradient: Callable[[int], np.ndarray]
    head_gradient: Callable[[np.ndarray, int], np.ndarray]
    forward: Callable[[np.ndarray], np.ndarray]
    twin: nn.Module
    rng_state: Callable[[], Any]


class Twin(nn.Module):
    """The subject's architecture in PyTorch, taking the subject's input layout."""

    def __init__(self, conv: tuple[Any, Any], head: tuple[Any, Any], channels_last: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.head = nn.Linear(64, 2)
        self.channels_last = channels_last
        with torch.no_grad():
            self.conv.weight.copy_(torch.as_tensor(np.asarray(conv[0], dtype=np.float32)))
            self.conv.bias.copy_(torch.as_tensor(np.asarray(conv[1], dtype=np.float32)))
            self.head.weight.copy_(torch.as_tensor(np.asarray(head[0], dtype=np.float32)))
            self.head.bias.copy_(torch.as_tensor(np.asarray(head[1], dtype=np.float32)))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv, ReLU, flatten in the subject's memory order, dense head."""

        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        hidden = torch.relu(self.conv(x))
        if self.channels_last:
            hidden = hidden.permute(0, 2, 3, 1)
        return self.head(hidden.flatten(1))


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _output(trace: Any) -> Any:
    return trace[trace.output_layers[0]]


def _input(trace: Any) -> Any:
    return trace[trace.input_layers[0]]


def _activation(trace: Any, func_name: str) -> Any:
    return next(op for op in trace._layer_list if op.func_name == func_name)


def _close(actual: np.ndarray, expected: np.ndarray, what: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5, err_msg=what)


def check_receptive_matches_framework(kit: Kit) -> None:
    """Receptive probe values and support equal the framework's own unit gradient."""

    target, source = _output(kit.trace), _input(kit.trace)
    for unit in OUTPUT_UNITS:
        result = gradient_for_unit(target, unit, source=source.label)
        expected = kit.unit_gradient(unit[1])
        _close(_np(result.signed_grad), expected, f"signed gradient of {unit}")
        _close(_np(result.grad), np.abs(expected), f"gradient magnitude of {unit}")
        assert np.array_equal(_np(result.support_mask), expected != 0), unit
        by_role = gradient_for_unit(target, unit)
        (only,) = by_role.values()
        _close(_np(only.signed_grad), expected, f"input-path gradient of {unit}")


def check_receptive_matches_finite_differences(kit: Kit, samples: int = 8) -> None:
    """Probe values agree with central finite differences through the model."""

    target = _output(kit.trace)
    rng = np.random.default_rng(1)
    flat = rng.choice(kit.x.size, size=samples, replace=False)
    epsilon = 1e-3
    for unit in OUTPUT_UNITS:
        probed = _np(gradient_for_unit(target, unit, source=_input(kit.trace).label).signed_grad)
        for index in flat:
            element = np.unravel_index(int(index), kit.x.shape)
            up, down = kit.x.copy(), kit.x.copy()
            up[element] += epsilon
            down[element] -= epsilon
            slope = (kit.forward(up)[unit] - kit.forward(down)[unit]) / (2 * epsilon)
            assert abs(probed[element] - slope) <= 1e-3 + 1e-2 * abs(slope), (unit, element)


def check_intermediate_source(kit: Kit) -> None:
    """A probe from the activation op equals the framework gradient of the head."""

    target = _output(kit.trace)
    activation = _activation(kit.trace, kit.activation)
    hidden = kit_np(activation.out)
    for unit in OUTPUT_UNITS:
        result = gradient_for_unit(target, unit, source=activation.label)
        expected = kit.head_gradient(hidden, unit[1])
        _close(_np(result.signed_grad), expected, f"activation-source gradient of {unit}")


def kit_np(value: Any) -> np.ndarray:
    """A framework array as NumPy."""

    if isinstance(value, torch.Tensor):
        return _np(value)
    numpy = getattr(value, "numpy", None)
    return np.asarray(numpy() if callable(numpy) else value)


def check_projective_matches_framework(kit: Kit) -> None:
    """The projective column at one input unit equals the framework Jacobian column."""

    target, source = _output(kit.trace), _input(kit.trace)
    expected = np.zeros((1, 2), dtype=np.float32)
    for unit in OUTPUT_UNITS:
        expected[unit] = kit.unit_gradient(unit[1])[kit.center]
    assert np.any(expected != 0), "the chosen input unit must reach the head"
    explicit = projective_gradient_for_unit(source, kit.center, target=target.label)
    _close(_np(explicit.signed_grad), expected, "projective column")
    assert np.array_equal(_np(explicit.support_mask), expected != 0)
    by_target = projective_gradient_for_unit(source, kit.center)
    (only,) = by_target.values()
    _close(_np(only.signed_grad), expected, "projective column over every output")


def check_twin_matches_torch(kit: Kit) -> None:
    """Both directions equal the PyTorch probe on a twin model with the same weights."""

    import torchlens as tl

    capture = tl.options.CaptureOptions(backward_ready=True, save_mode="reference")
    twin = tl.trace(kit.twin, torch.as_tensor(kit.x), capture=capture)
    try:
        target, source = _output(kit.trace), _input(kit.trace)
        twin_target, twin_source = _output(twin), _input(twin)
        for unit in OUTPUT_UNITS:
            ours = gradient_for_unit(target, unit, source=source.label)
            theirs = gradient_for_unit(twin_target, unit, source=twin_source.label)
            _close(_np(ours.signed_grad), _np(theirs.signed_grad), f"receptive {unit} vs torch")
            assert torch.equal(ours.support_mask, theirs.support_mask), unit
            assert ours.batch_support == theirs.batch_support
        ours_column = projective_gradient_for_unit(source, kit.center, target=target.label)
        theirs_column = projective_gradient_for_unit(
            twin_source, kit.center, target=twin_target.label
        )
        _close(_np(ours_column.signed_grad), _np(theirs_column.signed_grad), "projective vs torch")
        assert torch.equal(ours_column.support_mask, theirs_column.support_mask)
    finally:
        twin.cleanup()


def check_selection_methods(kit: Kit) -> None:
    """``.of(method="gradient")`` answers in both directions on the backend's trace."""

    target, source = _output(kit.trace), _input(kit.trace)
    receptive = target.receptive_field.of([OUTPUT_UNITS[0]], method="gradient")
    rows = [entry for entry in receptive if f"{entry.key[0]}:{entry.key[1]}" == source.label]
    assert rows and all(row.provenance.basis == "gradient" for row in rows)
    probe = gradient_for_unit(target, OUTPUT_UNITS[0], source=source.label)
    assert torch.equal(rows[0].mask, probe.support_mask)
    projective = source.projective_field.of([kit.center], method="gradient")
    assert len(projective) > 0
    assert all(entry.provenance.basis == "gradient" for entry in projective)


def check_probe_hygiene(kit: Kit) -> None:
    """Probes append no backward events and leave the framework RNG unchanged."""

    target, source = _output(kit.trace), _input(kit.trace)
    events = _backward_event_count(kit.trace)
    before = kit.rng_state()
    torch_before = torch.random.get_rng_state()
    gradient_for_unit(target, OUTPUT_UNITS[0])
    projective_gradient_for_unit(source, kit.center)
    assert _backward_event_count(kit.trace) == events
    assert _same_state(kit.rng_state(), before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def _same_state(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(_same_state, left, right))
    return bool(np.array_equal(kit_np(left), kit_np(right)))


def check_two_inputs_disconnected(trace: Any, a: np.ndarray, b: np.ndarray) -> None:
    """``f(a, b) = (a * a, b + 1)``: each output's other input gets an all-zero gradient."""

    outputs = [trace[label] for label in trace.output_layers]
    assert len(outputs) == 2
    seen = set()
    for output in outputs:
        unit = tuple(0 for _ in output.shape)
        by_role = gradient_for_unit(output, unit)
        assert len(by_role) == 2
        zero_roles = [role for role, item in by_role.items() if not item.support_mask.any()]
        live_roles = [role for role, item in by_role.items() if item.support_mask.any()]
        assert len(zero_roles) == 1 and len(live_roles) == 1, by_role.keys()
        assert not _np(by_role[zero_roles[0]].signed_grad).any()
        live = _np(by_role[live_roles[0]].signed_grad)
        assert np.count_nonzero(live) == 1 and live[unit] != 0
        value = float(live[unit])
        if np.isclose(value, 1.0):
            seen.add("b")
        else:
            np.testing.assert_allclose(value, 2 * a[unit], rtol=1e-6)
            seen.add("a")
    assert seen == {"a", "b"}
    del b
