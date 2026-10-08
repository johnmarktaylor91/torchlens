"""Raw-I/O save policies save; the transforms fill ``raw_input`` / ``raw_output``.

The glossary contract: ``Trace.raw_input`` is the original user input before
``transform`` and ``Trace.raw_output`` is the model output after
``output_transform``; ``save_raw_input`` / ``save_raw_output`` only decide how
those values are written into portable bundles. Setting a policy with no
transform therefore leaves the field empty, which callers have read as a silent
failure. The capture must say so with one coded warning per field per capture,
naming the remedy, and must stay quiet whenever the policy is not inert.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions

_OUTPUT_CODE = "save_raw_output_without_output_transform"
_INPUT_CODE = "save_raw_input_without_transform"


def _model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Identity()).eval()


def _codes(record: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(w.message.fields.get("code"))
        for w in record
        if isinstance(w.message, TorchLensWarning)
        and w.message.fields.get("code") in {_OUTPUT_CODE, _INPUT_CODE}
    ]


def _trace_recording(*args: object, **kwargs: object) -> tuple[object, list[str]]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        trace = tl.trace(*args, **kwargs)
    return trace, _codes(record)


@pytest.mark.parametrize("policy", [True, "small"])
def test_explicit_output_policy_without_output_transform_warns(policy: object) -> None:
    """The elicit repro: an explicit policy with no transform warns and names the remedy."""

    with pytest.warns(TorchLensWarning) as record:
        trace = tl.trace(
            _model(),
            torch.ones(1, 4),
            save=tl.module("1"),
            capture=CaptureOptions(inference_only=True, save_raw_output=policy),
        )
    coded = [
        w.message
        for w in record
        if isinstance(w.message, TorchLensWarning) and w.message.fields.get("code") == _OUTPUT_CODE
    ]
    assert len(coded) == 1
    assert "output_transform" in coded[0].fields["remedy"]
    assert trace.raw_output is None
    assert trace.find_sites(tl.module("1")).first().out.shape == (1, 4)


@pytest.mark.parametrize("policy", [True, "small"])
def test_explicit_input_policy_without_transform_warns(policy: object) -> None:
    """A tensor input with no transform= keeps no raw_input, so the policy warns."""

    trace, codes = _trace_recording(
        _model(), torch.ones(1, 4), capture=CaptureOptions(save_raw_input=policy)
    )
    assert codes == [_INPUT_CODE]
    assert trace.raw_input is None


def test_identity_transforms_populate_both_fields_quietly() -> None:
    """The documented way to keep the input and the model output on the trace."""

    model = _model()
    x = torch.ones(1, 4)
    trace, codes = _trace_recording(
        model,
        x,
        capture=CaptureOptions(
            save_raw_input=True,
            transform=lambda value: value,
            save_raw_output=True,
            output_transform=lambda value: value,
        ),
    )
    assert codes == []
    assert torch.equal(trace.raw_input, x)
    assert torch.equal(trace.raw_output, model(x))


def test_auto_coerced_raw_input_satisfies_the_policy() -> None:
    """Auto-coercion keeps raw_input without transform=, so the policy is not inert."""

    class _WithConfig(nn.Module):
        def forward(self, x: torch.Tensor, cfg: dict) -> torch.Tensor:
            return x * 2.0

    trace, codes = _trace_recording(
        _WithConfig(),
        [torch.ones(3), {"mode": "fast"}],
        capture=CaptureOptions(save_raw_input=True),
    )
    assert trace.raw_input is not None
    assert codes == []


def test_each_inert_policy_warns_once_per_capture_even_when_chunked() -> None:
    """One warning per field per capture, never per op or per chunk."""

    capture = CaptureOptions(save_raw_input=True, save_raw_output=True)
    _, plain_codes = _trace_recording(_model(), torch.ones(1, 4), capture=capture)
    _, chunked_codes = _trace_recording(
        _model(), torch.ones(4, 4), capture=capture, save=tl.module("0"), chunk_size=1
    )
    assert sorted(plain_codes) == sorted([_INPUT_CODE, _OUTPUT_CODE])
    assert sorted(chunked_codes) == sorted([_INPUT_CODE, _OUTPUT_CODE])


@pytest.mark.parametrize(
    "capture",
    [None, CaptureOptions(save_raw_output=False, save_raw_input=False)],
)
def test_default_or_disabled_policies_stay_quiet(capture: CaptureOptions | None) -> None:
    """The default policies and explicit opt-outs are not inert requests."""

    trace, codes = _trace_recording(_model(), torch.ones(1, 4), capture=capture)
    assert codes == []
    assert trace.raw_output is None
