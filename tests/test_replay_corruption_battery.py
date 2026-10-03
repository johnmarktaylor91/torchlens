"""Planted per-op payload corruption battery (b9 R74/75-2).

The flagship per-op replay comparator had a measured mutation kill margin of
ONE test: with ``matches_saved = True`` planted at the replay callsite, all
of ``test_validation.py`` stayed green and exactly one diagnostics-payload
test failed. Nothing planted a NUMERIC corruption into a saved payload and
asserted the replay walk names the op. This battery is that arming: each
test corrupts exactly one saved activation on a DIFFERENT op family and
asserts (a) the pristine trace validates, (b) the corrupted trace FAILS, and
(c) the recorded ``ValidationFailure`` names the corruption site (the op or
one of its direct relatives on the replay walk).

The battery is the kill set for ``mutation_driver.py``'s M13 (comparator
neuter): every test here goes green-to-red on the comparator, so the margin
is the battery's size, not one incidental diagnostics test.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens.validation.diagnostics import TRACE_FAILURE_ATTR


def _capture(model: nn.Module, x: torch.Tensor):
    """Capture with exhaustive saves plus the ground-truth output.

    Parameters
    ----------
    model:
        Model to trace (switched to eval mode for determinism).
    x:
        Input tensor.

    Returns
    -------
    tuple
        ``(trace, ground_truth_outputs, model)``. The model MUST be kept
        alive by the caller for the trace's lifetime: post-trace parameter
        access holds only a weak reference, so an inline model whose last
        strong owner is this frame dies at the next cyclic-GC pass and
        replay raises ``PostTraceParamUnavailable`` -- an order-dependent
        red under multi-file composition (b9/T14).
    """

    model = model.eval()
    log = trace_fn(
        model,
        x,
        capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True),
    )
    with torch.no_grad():
        ground_truth = model(x)
    outputs = list(ground_truth) if isinstance(ground_truth, (tuple, list)) else [ground_truth]
    return log, outputs, model


def _base_label(label: str) -> str:
    """Return the label without a ``:pass`` qualifier.

    Parameters
    ----------
    label:
        Possibly pass-qualified layer label.

    Returns
    -------
    str
        The unqualified label.
    """

    return label.split(":", 1)[0]


def _neighbors(log, label: str) -> set[str]:
    """Return the op's base label plus its direct parents and children.

    The replay BFS walks backward from the outputs, so the FIRST mismatch a
    single-op corruption produces is reported at the corrupted op or at one
    of its direct graph neighbors (a child replays FROM the corrupted saved
    value before the op itself is replayed).

    Parameters
    ----------
    log:
        Finished trace.
    label:
        Corrupted op's label.

    Returns
    -------
    set[str]
        Base labels at which the first mismatch may legitimately surface.
    """

    op = log[label]
    return {_base_label(item) for item in ({label} | set(op.parents) | set(op.children))}


def _corrupt_and_assert(model: nn.Module, x: torch.Tensor, func_name: str) -> None:
    """Corrupt one saved payload for ``func_name`` and assert the tripwire fires.

    Parameters
    ----------
    model:
        Model to trace.
    x:
        Input tensor.
    func_name:
        Function name of the op whose FIRST occurrence gets corrupted.
    """

    log, outputs, model = _capture(model, x)
    try:
        # Force the cyclic-GC pass that composition pressure used to trigger
        # nondeterministically: with the strong model ref above this is inert,
        # and without it the pristine control below reds 100% of the time.
        gc.collect()
        # (a) pristine control: an already-red baseline would make the
        # corruption verdict meaningless (the b9 lesson).
        assert log.validate_forward_pass(outputs), "pristine trace failed validation"

        target = next(op for op in log.compute_ops if op.func_name == func_name)
        with torch.no_grad():
            corrupted = target.out + 0.25
        target.out = corrupted

        # (b) the corruption must be caught ...
        status = log.validate_forward_pass(outputs)
        assert not status, f"corrupted {target.label} payload validated as clean"

        # (c) ... and the failure record must localize it.
        failure = getattr(log, TRACE_FAILURE_ATTR, None)
        assert failure is not None, "validation failed without recording a failure"
        allowed = _neighbors(log, target.label)
        assert _base_label(failure.op_label or "") in allowed, (
            f"failure named {failure.op_label!r}, expected the corruption site "
            f"{target.label!r} or a direct neighbor {sorted(allowed)}"
        )
    finally:
        log.cleanup()


class _MLP(nn.Module):
    """Two-layer perceptron for the linear/relu plants."""

    def __init__(self) -> None:
        """Build the two linear layers."""

        super().__init__()
        self.fc1 = nn.Linear(6, 5)
        self.fc2 = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run linear -> relu -> linear.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Logits.
        """

        return self.fc2(torch.relu(self.fc1(x)))


class _ConvNet(nn.Module):
    """Small conv/pool stack for the conv2d and pooling plants."""

    def __init__(self) -> None:
        """Build conv and pooling layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv -> relu -> pool.

        Parameters
        ----------
        x:
            Input images.

        Returns
        -------
        torch.Tensor
            Pooled features.
        """

        return self.pool(torch.relu(self.conv(x)))


class _Arithmetic(nn.Module):
    """Elementwise chain for the add/mul plants."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run add -> mul -> tanh.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Squashed result.
        """

        return torch.tanh((x + 1.5) * 0.5)


class _MiniAttention(nn.Module):
    """Matmul/softmax pair for the attention-shaped plants."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run scores = x @ x.T, softmax, weighted sum.

        Parameters
        ----------
        x:
            Token matrix.

        Returns
        -------
        torch.Tensor
            Attention output.
        """

        scores = torch.matmul(x, x.transpose(0, 1))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, x)


def test_corrupted_linear_payload_fails_replay():
    """A corrupted linear activation is caught and localized."""

    _corrupt_and_assert(_MLP(), torch.randn(2, 6), "linear")


def test_corrupted_relu_payload_fails_replay():
    """A corrupted relu activation is caught and localized."""

    _corrupt_and_assert(_MLP(), torch.randn(2, 6), "relu")


def test_corrupted_conv_payload_fails_replay():
    """A corrupted conv2d activation is caught and localized."""

    _corrupt_and_assert(_ConvNet(), torch.randn(1, 1, 6, 6), "conv2d")


def test_corrupted_pool_payload_fails_replay():
    """A corrupted max_pool2d activation is caught and localized."""

    _corrupt_and_assert(_ConvNet(), torch.randn(1, 1, 6, 6), "max_pool2d")


def test_corrupted_add_payload_fails_replay():
    """A corrupted elementwise-add activation is caught and localized."""

    _corrupt_and_assert(_Arithmetic(), torch.randn(3, 4), "__add__")


def test_corrupted_mul_payload_fails_replay():
    """A corrupted elementwise-mul activation is caught and localized."""

    _corrupt_and_assert(_Arithmetic(), torch.randn(3, 4), "__mul__")


def test_corrupted_softmax_payload_fails_replay():
    """A corrupted softmax activation is caught and localized."""

    _corrupt_and_assert(_MiniAttention(), torch.randn(4, 4), "softmax")


def test_corrupted_matmul_payload_fails_replay():
    """A corrupted matmul activation is caught and localized."""

    _corrupt_and_assert(_MiniAttention(), torch.randn(4, 4), "matmul")


def test_corrupted_output_payload_fails_ground_truth():
    """Corrupting the OUTPUT op's saved value trips the ground-truth seed check."""

    log, outputs, model = _capture(_MLP(), torch.randn(2, 6))
    try:
        # Same forced-GC arming as _corrupt_and_assert: `model` is the strong
        # ref that keeps post-trace parameter access alive across this pass.
        gc.collect()
        assert log.validate_forward_pass(outputs), "pristine trace failed validation"
        target = log.output_ops[0]
        # The output op's ``out`` mirrors its producer and is not assignable;
        # in-place mutation corrupts the stored payload itself.
        with torch.no_grad():
            target.out.add_(1.0)
        status = log.validate_forward_pass(outputs)
        assert not status, "corrupted output payload validated as clean"
        failure = getattr(log, TRACE_FAILURE_ATTR, None)
        assert failure is not None, "validation failed without recording a failure"
    finally:
        log.cleanup()
