"""Round-11 F6 regression: parameterized ``if/else`` models must save runnable.

A conditional arm records a private ``_trace`` back-reference to its owning ``Trace``.
``ConditionalAccessor`` has no portable-state spec, so the bundle scrub returns it
verbatim and the arm's back-reference still points at the LIVE trace -- dragging that
trace's capture-state tensors into the value-free sparse core. A parameter-free
conditional (the only shipped conditional model) never exercised this. The producer now
detaches the runtime-only back-reference from the scrub product (mirroring the top-level
DROP-and-route), leaving the live trace intact and keeping the tensor-payload tripwire
armed for a genuine stray.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens.data_classes.trace import Conditional, ConditionalAccessor, ConditionalArm
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness


class ParamConditionalModel(nn.Module):
    """Parameterized ``if/else`` model (arm bodies see module parameters)."""

    def __init__(self) -> None:
        """Initialize a deterministic linear layer."""

        torch.manual_seed(13)
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Take a data-dependent arm over a parameterized projection."""

        hidden = self.linear(value)
        if hidden.sum() > 0:
            hidden = hidden * 2
        else:
            hidden = hidden - 1
        return hidden


def _capture(model: nn.Module, value: torch.Tensor) -> tl.Trace:
    """Capture an intervention-ready trace with container structure."""

    return tl.trace(
        model,
        value,
        layers_to_save="all",
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


@pytest.mark.smoke
def test_parameterized_conditional_saves_and_verifies(tmp_path: Path) -> None:
    """A param'd if/else model saves runnable and replays VERIFIED on its input."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    model = ParamConditionalModel().eval()
    trace = _capture(model, value)

    path = tmp_path / "param-conditional.tlspec"
    # Previously raised AssertionError: "Sparse core tensor payload at
    # conditionals._list.0.arms.0._trace._runnable_capture_state.linear.weight".
    tl.save(trace, path, level="runnable", include_weights=True)

    # The save must not have mutated the live trace's conditional accessors.
    assert trace.conditionals[0].arms[0]._trace is trace
    assert trace.conditionals[0].arms[0].evaluation_ops

    result = tl.load(path).run(inputs=value, seed=0)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, model(value))


def test_conditional_arm_flip_still_diverges(tmp_path: Path) -> None:
    """A changed input that flips the recorded arm must be caught, not blessed."""

    torch.manual_seed(0)
    value = torch.randn(2, 4)
    model = ParamConditionalModel().eval()
    trace = _capture(model, value)
    path = tmp_path / "flip.tlspec"
    tl.save(trace, path, level="runnable", include_weights=True)
    loaded = tl.load(path)

    captured_then = bool(model.linear(value).sum() > 0)
    flipped = None
    for seed in range(1, 200):
        torch.manual_seed(seed)
        candidate = torch.randn(2, 4) * 5
        if bool(model.linear(candidate).sum() > 0) != captured_then:
            flipped = candidate
            break
    assert flipped is not None, "could not synthesize an arm-flipping input"

    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=flipped, seed=0)


@pytest.mark.smoke
def test_sparse_core_tensor_tripwire_still_fires() -> None:
    """The value-free tripwire must still fire on a genuine stray payload."""

    # A stray tensor in an ordinary field.
    with pytest.raises(AssertionError, match="Sparse core tensor payload"):
        assert_sparse_core_has_no_tensor_payload({"field": {"nested": torch.randn(3)}})

    # A stray tensor smuggled into a conditional arm's NON-``_trace`` field must
    # still be caught: only the runtime-only ``_trace`` back-reference is detached.
    arm = ConditionalArm(kind="then")
    arm.bool_value_at_run = torch.randn(2)  # type: ignore[assignment]
    conditional = Conditional(
        id="cond",
        arms=[arm],
        fired_arm_index=0,
        fired_arm_kind="then",
        source_file=None,
        source_line=None,
    )
    with pytest.raises(AssertionError, match="Sparse core tensor payload"):
        assert_sparse_core_has_no_tensor_payload(
            {"conditionals": ConditionalAccessor([conditional])}
        )
