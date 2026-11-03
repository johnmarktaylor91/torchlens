"""``tl.extract`` and ``tl.pluck`` refuse an unknown layer typed, never with a builtin error.

Behaviour-net finding IO-1 (2026-11-01): a list or mapping request naming a layer the
capture does not record raised a builtin ``ValueError`` ("Layer ... not found. Did you
mean ...?"), a request whose key is a bare module address with no pass qualifier crashed
with ``ValueError: not enough values to unpack``, and the error contract promises
``op_lookup_not_found`` for every lookup that matches nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import InvalidArgumentError, TorchLensError

# The capture still discloses a zero-match ``layers_to_save`` request with a UserWarning
# before the typed refusal (``tests/test_onramp.py`` pins both signals); the package's
# error filter would otherwise surface that warning here instead of the refusal under test.
pytestmark = [
    pytest.mark.filterwarnings(r"default:layers_to_save=.* matched zero layers:UserWarning")
]


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x)).sum(dim=-1)


def _code(exc: BaseException) -> Any:
    return exc.fields.get("code") if isinstance(exc, TorchLensError) else None


@pytest.fixture
def tiny() -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    return _Tiny().eval(), torch.randn(2, 4)


@pytest.mark.parametrize(
    "layers",
    [
        {"mine": "no_such_layer_anywhere"},
        ["lin:3"],
    ],
    ids=["mapping", "module-pass-beyond-count"],
)
def test_extract_unknown_layer_refuses_op_lookup_not_found(
    tiny: tuple[nn.Module, torch.Tensor], layers: Any
) -> None:
    model, x = tiny
    with pytest.raises(InvalidArgumentError) as info:
        tl.extract(model, x, layers)
    assert _code(info.value) in {"op_lookup_not_found", "op_lookup_pass_out_of_range"}, info.value
    assert info.value.fields.get("remedy")
    assert "not enough values to unpack" not in str(info.value)


def test_extract_root_module_address_is_the_output_not_a_crash(
    tiny: tuple[nn.Module, torch.Tensor],
) -> None:
    """``self`` is the root module's address: extract returns its output instead of crashing.

    At the base this key reached ``key.rsplit(":", 1)`` with no colon and raised the builtin
    ``ValueError: not enough values to unpack``.
    """

    model, x = tiny
    out = tl.extract(model, x, ["self"])
    torch.testing.assert_close(out["self"], model(x))


def test_extract_unknown_layer_message_still_suggests_candidates(
    tiny: tuple[nn.Module, torch.Tensor],
) -> None:
    model, x = tiny
    with pytest.raises(InvalidArgumentError) as info:
        tl.extract(model, x, ["linear_1_1_typo"])
    assert _code(info.value) == "op_lookup_not_found"
    assert "  candidates" in str(info.value)
    assert info.value.fields.get("key") == "linear_1_1_typo"


@pytest.mark.smoke
def test_pluck_unknown_layer_refuses_op_lookup_not_found(
    tiny: tuple[nn.Module, torch.Tensor],
) -> None:
    model, x = tiny
    with pytest.raises(InvalidArgumentError) as info:
        tl.pluck(model, x, "no_such_layer_anywhere")
    assert _code(info.value) == "op_lookup_not_found"
