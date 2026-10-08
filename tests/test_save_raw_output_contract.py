"""``save_raw_output`` is a bundle-save policy; ``raw_output`` comes from ``output_transform``.

The glossary contract: ``Trace.raw_output`` is the human-readable model output
after ``output_transform``, and ``save_raw_output`` only decides how that value
is written into portable bundles. Setting ``save_raw_output`` with no
``output_transform`` therefore leaves ``raw_output`` empty, which callers have
read as a silent failure. The capture must say so with a coded warning that
names the remedy, and must stay quiet whenever the policy is not inert.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions

_CODE = "save_raw_output_without_output_transform"


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Identity()).eval()


def _coded(record: list[warnings.WarningMessage]) -> list[TorchLensWarning]:
    return [
        w.message
        for w in record
        if isinstance(w.message, TorchLensWarning) and w.message.fields.get("code") == _CODE
    ]


@pytest.mark.parametrize("policy", [True, "small"])
def test_explicit_policy_without_output_transform_warns(policy: object) -> None:
    """The elicit repro: an explicit policy with no transform warns and names the remedy."""

    with pytest.warns(TorchLensWarning) as record:
        trace = tl.trace(
            _model(),
            torch.ones(1, 4),
            save=tl.module("1"),
            capture=CaptureOptions(inference_only=True, save_raw_output=policy),
        )
    coded = _coded(list(record))
    assert len(coded) == 1
    assert "output_transform" in coded[0].fields["remedy"]
    assert trace.raw_output is None
    assert trace.find_sites(tl.module("1")).first().out.shape == (1, 4)


def test_identity_output_transform_populates_raw_output() -> None:
    """The documented way to keep the model output on the trace."""

    model = _model()
    x = torch.ones(1, 4)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        trace = tl.trace(
            model,
            x,
            capture=CaptureOptions(
                inference_only=True, save_raw_output=True, output_transform=lambda out: out
            ),
        )
    assert not _coded(record)
    assert isinstance(trace.raw_output, torch.Tensor)
    assert torch.equal(trace.raw_output, model(x))


@pytest.mark.parametrize("capture", [None, CaptureOptions(save_raw_output=False)])
def test_default_or_disabled_policy_stays_quiet(capture: CaptureOptions | None) -> None:
    """The default policy and an explicit opt-out are not inert requests."""

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        trace = tl.trace(_model(), torch.ones(1, 4), capture=capture)
    assert not _coded(record)
    assert trace.raw_output is None
