"""``trace.run_transaction(inputs=...)`` binds one input tree the same way on every provider.

The behaviour net found ``run_transaction(inputs=(y, no), fast=True)`` calling a two-argument
forward with the tuple as ONE argument, while ``tl.trace(model, (y, no))``, the
ordinary live run and the loaded provider all spread it. The fast live path now
binds through capture's own ``normalize_input_args``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils.arg_handling import normalize_input_args
from torchlens.options import CaptureOptions


class _TwoArgs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, y: torch.Tensor, no: torch.Tensor) -> torch.Tensor:
        return self.lin(y) * no


class _OnePair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.lin(pair[0]) - pair[1]


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(2, 3), torch.randn(2, 3)


def _spellings(y: torch.Tensor, no: torch.Tensor) -> dict[str, Any]:
    return {"tuple": (y, no), "list": [y, no], "args_kwargs": {"args": [y, no], "kwargs": {}}}


@pytest.mark.parametrize("fast", [False, True], ids=["ordinary", "fast"])
@pytest.mark.parametrize("spelling", ["tuple", "list", "args_kwargs"])
def test_live_run_spreads_a_sequence_for_a_two_argument_forward(spelling: str, fast: bool) -> None:
    torch.manual_seed(0)
    model = _TwoArgs().eval()
    y, no = _pair()
    trace = tl.trace(model, (y, no))
    new_y, new_no = y + 1, no * 2
    result = trace.run_transaction(inputs=_spellings(new_y, new_no)[spelling], fast=fast)
    with torch.no_grad():
        assert torch.equal(result.output, model(new_y, new_no))


def test_loaded_run_spreads_a_tuple_for_a_two_argument_forward(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _TwoArgs().eval()
    y, no = _pair()
    capture = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)
    path = tmp_path / "two.tlspec"
    tl.trace(model, (y, no), capture=capture).save(path, level="runnable", include_weights=True)
    new_y, new_no = y + 1, no * 2
    result = tl.load(path).run_transaction(inputs=(new_y, new_no))
    with torch.no_grad():
        assert torch.allclose(result.output, model(new_y, new_no))


class _OneNested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: dict[str, Any]) -> torch.Tensor:
        return self.lin(x["a"]) + x["b"]["c"]


def _nested() -> dict[str, Any]:
    torch.manual_seed(0)
    return {"a": torch.randn(2, 3), "b": {"c": torch.randn(2, 3)}}


@pytest.mark.parametrize("empty", [[], ()], ids=["list", "tuple"])
def test_empty_positionals_bind_no_argument_on_a_single_parameter_forward(empty: Any) -> None:
    """An empty positional sequence means no positional argument, never ``model(empty, x=...)``.

    The former rule wrapped an empty sequence into ``[[]]`` whenever ``forward`` takes one
    positional parameter, so a keyword-only call raised "got multiple values for argument".
    """
    assert normalize_input_args(empty, _OneNested()) == []


def test_validate_keyword_only_call_on_a_single_parameter_forward() -> None:
    """``tl.validate(model, [], {"x": nested})`` reaches ``forward(x=nested)`` once and passes."""
    torch.manual_seed(0)
    model = _OneNested().eval()
    assert tl.validate(model, [], {"x": _nested()}, scope="forward", seed=0) is True
