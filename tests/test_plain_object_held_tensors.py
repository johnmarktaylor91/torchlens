"""Tensors held on a plain (non-Module) object attribute of a module are buffer sources.

A ``torch.distributions`` prior or a diffusers scheduler built in ``__init__``
holds tensors that exist before the forward and are neither parameters,
buffers nor inputs. Their reads used to become internal sources with an
unattributed argument, and the ``graph_connectivity`` metadata invariant
raised (Menagerie rung 4: f-topographic-vae ``Normal.scale``, f-mld
``final_alpha_cumprod``). Model preparation now stamps an object's direct
public tensor attributes like module-held tensors (``prior[scale]``).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


class _Schedule:
    """A plain object holding a constructor-time tensor (the diffusers shape)."""

    def __init__(self) -> None:
        self.final_alpha_cumprod = torch.tensor(0.99915)
        self._private = torch.ones(1)


class _ScheduleUser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.schedule = _Schedule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * (1 - self.schedule.final_alpha_cumprod)


class _DistributionPrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.prior = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prior.log_prob(self.lin(x)).sum()


@pytest.mark.smoke
@pytest.mark.parametrize("model_cls", [_ScheduleUser, _DistributionPrior])
def test_plain_object_held_tensor_validates(model_cls: type[nn.Module]) -> None:
    torch.manual_seed(0)
    model = model_cls().eval()
    x = torch.randn(3, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is True, last_validation_failure()


@pytest.mark.smoke
def test_plain_object_held_tensor_is_a_named_buffer_source() -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(_ScheduleUser().eval(), torch.randn(3, 4))
    buffer_addresses = [op.address for op in trace.ops if op.is_buffer]
    assert buffer_addresses == ["schedule[final_alpha_cumprod]"]
