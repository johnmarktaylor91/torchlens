"""Validation regressions: float64 default dtype, and a Parameter returned in the output.

Both refused healthy captures before validation looked at a single op:

* ``torch.set_default_dtype(torch.float64)`` in a model's constructor made the
  comparator self-test build float64 sentinels and judge its fp32-derived accept
  pair against the float64 band, so every such model raised "comparator
  self-test failed".
* A model returning one of its own ``nn.Parameter`` objects in the output
  (a learned token in a DETR-style output dict) was counted as a dropped
  output leaf, although capture excludes Parameters from outputs by design.

Each fix is narrow: the poisoned-band and dropped-leaf tripwires still fire.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import _output_walk


@pytest.fixture
def float64_default_dtype():
    """Run a test under a float64 default dtype, restoring the prior one."""

    prior = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prior)


def test_comparator_self_test_passes_under_float64_default_dtype(float64_default_dtype) -> None:
    """A healthy comparator passes whatever the process default dtype is."""

    from torchlens.validation.core import _comparator_self_test

    _comparator_self_test()


def test_comparator_self_test_still_rejects_poisoned_band_under_float64_default(
    float64_default_dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning the sentinel dtype must not blind the band tripwire."""

    from torchlens.utils import tensor_utils as tensor_utils_mod
    from torchlens.validation.core import _comparator_self_test

    monkeypatch.setitem(tensor_utils_mod._DTYPE_FLOAT_TOLERANCES, torch.float32, (0.33, 1e-5))
    with pytest.raises(RuntimeError, match="comparator self-test failed"):
        _comparator_self_test()


class _Float64Linear(nn.Module):
    """A model that switches the process default dtype in its constructor."""

    def __init__(self) -> None:
        super().__init__()
        torch.set_default_dtype(torch.float64)
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x.to(torch.get_default_dtype()))


def test_forward_validation_runs_for_float64_default_model() -> None:
    """End to end: the model that set float64 validates instead of raising."""

    prior = torch.get_default_dtype()
    try:
        torch.manual_seed(0)
        model = _Float64Linear().eval()
        assert bool(tl.validate(model, torch.randn(3, 4), scope="forward"))
    finally:
        torch.set_default_dtype(prior)


class _ReturnsParameter(nn.Module):
    """Return a computed tensor and a raw learned token in one output dict."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.token = nn.Parameter(torch.ones(2, 4))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"y": self.lin(x), "token": self.token}


def test_forward_validation_accepts_parameter_in_output() -> None:
    """A Parameter leaf in the output is model state, not a dropped output."""

    torch.manual_seed(0)
    assert bool(tl.validate(_ReturnsParameter().eval(), torch.randn(2, 4), scope="forward"))


def test_independent_walker_skips_only_parameter_leaves() -> None:
    """The walker drops Parameter leaves and still reports every other tensor."""

    token = nn.Parameter(torch.ones(2))
    plain = torch.zeros(2)
    ids = _output_walk.independent_output_tensor_ids({"token": token, "nested": [plain]})
    assert ids == [id(plain)]
