"""Round-24 runnable regressions for ``__idiv__`` and buffer ``.data`` host writes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness


_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save_load(model: nn.Module, capture_input: torch.Tensor, path: Path) -> Any:
    """Save and reload one runnable trace.

    Parameters
    ----------
    model:
        Model to capture.
    capture_input:
        Input used for capture.
    path:
        Destination ``.tlspec`` path.

    Returns
    -------
    Any
        Loaded runnable trace.
    """

    trace = tl.trace(model, capture_input.clone(), capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path)


def _run_loaded(loaded: Any, x: torch.Tensor) -> Any:
    """Run one loaded runnable trace with poison-return divergence.

    Parameters
    ----------
    loaded:
        Loaded runnable trace.
    x:
        Runtime input.

    Returns
    -------
    Any
        Runnable ``RunResult``.
    """

    return loaded.run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


def _live_output(factory: Callable[[], nn.Module], x: torch.Tensor) -> torch.Tensor:
    """Return a fresh eager output for value-correctness checks.

    Parameters
    ----------
    factory:
        Model factory.
    x:
        Runtime input.

    Returns
    -------
    torch.Tensor
        Fresh eager output.
    """

    with torch.no_grad():
        return factory()(x.clone())


def _path_token(value: str) -> str:
    """Return a filesystem-safe token for a parametrized case.

    Parameters
    ----------
    value:
        Raw case identifier.

    Returns
    -------
    str
        Case identifier safe for a file name.
    """

    return (
        value.replace("/", "div")
        .replace("*", "pow")
        .replace("<", "left")
        .replace(">", "right")
        .replace("&", "and")
        .replace("^", "xor")
        .replace("|", "or")
        .replace("%", "mod")
        .replace("+", "add")
        .replace("-", "sub")
        .replace("=", "eq")
    )


class DataIdivModel(nn.Module):
    """A ``.data`` legacy true-div write that must not false-verify."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        h = x + 4.0
        h.data /= 2.0
        return h * 3.0


class LabelledIdivModel(nn.Module):
    """A labelled legacy true-div write that must replay."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        out = x + 4.0
        out /= 2.0
        return out * 3.0


class AugAssignModel(nn.Module):
    """Model applying one augmented-assignment operation."""

    def __init__(self, op: str, other: int | float) -> None:
        """Store the augmented-assignment operation.

        Parameters
        ----------
        op:
            Operation spelling.
        other:
            Right-hand operand.
        """

        super().__init__()
        self.op = op
        self.other = other

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        out = x.clone()
        if self.op == "+=":
            out += self.other
        elif self.op == "-=":
            out -= self.other
        elif self.op == "*=":
            out *= self.other
        elif self.op == "/=":
            out /= self.other
        elif self.op == "//=":
            out //= self.other
        elif self.op == "%=":
            out %= self.other
        elif self.op == "**=":
            out **= self.other
        elif self.op == "<<=":
            out <<= int(self.other)
        elif self.op == ">>=":
            out >>= int(self.other)
        elif self.op == "&=":
            out &= int(self.other)
        elif self.op == "^=":
            out ^= int(self.other)
        elif self.op == "|=":
            out |= int(self.other)
        else:
            raise AssertionError(f"Unhandled op {self.op!r}")
        return out


class MaskedFillInvertModel(nn.Module):
    """Model using ``~mask`` before a non-in-place masked fill."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        mask = x > 0
        return x.masked_fill(~mask, 0.0)


class GenuineInplaceModel(nn.Module):
    """Model applying one ordinary in-place method."""

    def __init__(self, op: str) -> None:
        """Store the in-place operation.

        Parameters
        ----------
        op:
            In-place method spelling.
        """

        super().__init__()
        self.op = op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        out = x + 1.0
        if self.op == "add_":
            out.add_(2.0)
        elif self.op == "relu_":
            out.relu_()
        elif self.op == "masked_fill_":
            out.masked_fill_(out > 3.0, -1.0)
        else:
            raise AssertionError(f"Unhandled op {self.op!r}")
        return out


class BufferDataAddAfterConsume(nn.Module):
    """A buffer ``.data.add_`` after first consumption must fail closed."""

    def __init__(self) -> None:
        """Initialize the registered buffer."""

        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass."""

        y = x * self.b
        self.b.data.add_(1.0)
        return y + self.b


def test_data_idiv_is_unverifiable_not_false_verified(tmp_path: Path) -> None:
    """A ``.data /=`` legacy ``__idiv__`` write must be UNVERIFIABLE."""

    x = torch.tensor([2.0, 4.0])
    result = _run_loaded(_save_load(DataIdivModel(), x, tmp_path / "data_idiv.tlspec"), x)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


def test_labelled_idiv_replays_verified_and_value_correct(tmp_path: Path) -> None:
    """A labelled ``/=`` legacy ``__idiv__`` write must replay without crashing."""

    x = torch.tensor([2.0, 4.0])
    result = _run_loaded(_save_load(LabelledIdivModel(), x, tmp_path / "idiv.tlspec"), x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, _live_output(LabelledIdivModel, x))


_AUG_ASSIGN_CASES: tuple[tuple[str, int | float, torch.Tensor], ...] = (
    ("+=", 2, torch.tensor([6, 10])),
    ("-=", 1, torch.tensor([6, 10])),
    ("*=", 3, torch.tensor([6, 10])),
    ("/=", 2.0, torch.tensor([6.0, 10.0])),
    ("//=", 2, torch.tensor([6, 10])),
    ("%=", 5, torch.tensor([6, 10])),
    ("**=", 2, torch.tensor([2, 3])),
    ("<<=", 1, torch.tensor([6, 10])),
    (">>=", 1, torch.tensor([6, 10])),
    ("&=", 3, torch.tensor([6, 10])),
    ("^=", 1, torch.tensor([6, 10])),
    ("|=", 2, torch.tensor([6, 10])),
)


@pytest.mark.parametrize(
    "op,other,x", _AUG_ASSIGN_CASES, ids=[case[0] for case in _AUG_ASSIGN_CASES]
)
def test_all_augmented_assignment_dunders_replay_verified(
    op: str,
    other: int | float,
    x: torch.Tensor,
    tmp_path: Path,
) -> None:
    """All in-place augmented-assignment dunders replay and verify."""

    factory: Callable[[], nn.Module] = partial(AugAssignModel, op, other)
    result = _run_loaded(_save_load(factory(), x, tmp_path / f"{_path_token(op)}.tlspec"), x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, _live_output(factory, x))


def test_invert_stays_out_of_place_and_masked_fill_verifies(tmp_path: Path) -> None:
    """``__invert__`` remains out-of-place while ``masked_fill(~mask)`` verifies."""

    x = torch.tensor([-1.0, 2.0])
    trace = tl.trace(MaskedFillInvertModel(), x.clone(), capture=_CAPTURE)
    invert_ops = [op for op in trace.layer_list if op.func_name == "__invert__"]

    assert invert_ops
    assert all(not op.is_inplace for op in invert_ops)

    path = tmp_path / "invert.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = _run_loaded(tl.load(path), x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, _live_output(MaskedFillInvertModel, x))


@pytest.mark.parametrize("op", ("add_", "relu_", "masked_fill_"))
def test_genuine_inplace_methods_stay_inplace_and_verified(op: str, tmp_path: Path) -> None:
    """Ordinary in-place methods remain classified as in-place and replay."""

    x = torch.tensor([-2.0, 4.0])
    factory: Callable[[], nn.Module] = partial(GenuineInplaceModel, op)
    trace = tl.trace(factory(), x.clone(), capture=_CAPTURE)
    matching = [item for item in trace.layer_list if item.func_name == op]

    assert matching
    assert all(item.is_inplace for item in matching)

    path = tmp_path / f"{op}.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = _run_loaded(tl.load(path), x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, _live_output(factory, x))


def test_buffer_data_add_after_consume_is_unverifiable_no_crash(tmp_path: Path) -> None:
    """A surviving buffer ``.data.add_`` host write is UNVERIFIABLE and runnable."""

    x = torch.tensor([2.0, 4.0])
    result = _run_loaded(
        _save_load(BufferDataAddAfterConsume(), x, tmp_path / "buffer_data_add.tlspec"),
        x,
    )

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
