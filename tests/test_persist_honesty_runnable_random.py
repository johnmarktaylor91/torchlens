"""Persistence honesty: running a weight-free runnable artifact warns loudly.

WT1 A-IV item 17 (lane A08): a runnable artifact saved with the default
``include_weights=False`` loaded and ``.run()`` executed on RANDOM role-init
weights with zero warning, and its report could legitimately settle
``verified`` (path faithfulness against the random state). The run now emits a
``TorchLensWarning`` naming the slot count and both remedies; the settlement
semantics (state_source disclosure, ``verified`` meaning, ``not_applicable``
numeric attestation) are deliberately unchanged.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions
from torchlens.runnable import StateSource

pytestmark = pytest.mark.smoke


@pytest.fixture()
def runnable_bundle(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    path = tmp_path / "runnable"
    tl.save(trace, path, level="runnable")
    return path, model, x


def test_random_state_run_warns(runnable_bundle):
    path, _, x = runnable_bundle
    loaded = tl.load(path)
    with pytest.warns(TorchLensWarning, match="RANDOM") as record:
        result = loaded.run(inputs=x, seed=0)
    assert result.report.state_source is StateSource.RANDOM_INITIALIZATION
    warnings_seen = [w.message for w in record if issubclass(w.category, TorchLensWarning)]
    # S-18 contract: consumers branch on fields["code"], never message text.
    assert any(
        getattr(w, "fields", {}).get("code") == "runnable_random_init_run" for w in warnings_seen
    )
    messages = [str(w) for w in warnings_seen]
    assert any("include_weights=True" in m and "load_state_dict" in m for m in messages)


def test_embedded_weights_run_does_not_warn(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=CaptureOptions(intervention_ready=True))
    path = tmp_path / "runnable_weights"
    tl.save(trace, path, level="runnable", include_weights=True)
    loaded = tl.load(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = loaded.run(inputs=x)
    assert result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
    random_warnings = [
        w for w in caught if issubclass(w.category, TorchLensWarning) and "RANDOM" in str(w.message)
    ]
    assert random_warnings == []


def test_staged_user_state_run_does_not_warn(runnable_bundle):
    path, model, x = runnable_bundle
    loaded = tl.load(path)
    loaded.load_state_dict(model.state_dict())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = loaded.run(inputs=x)
    assert result.report.state_source is StateSource.USER_STATE_DICT
    random_warnings = [
        w for w in caught if issubclass(w.category, TorchLensWarning) and "RANDOM" in str(w.message)
    ]
    assert random_warnings == []
    # The staged real-weights run reproduces the captured model's output.
    assert torch.allclose(result.output, model(x))
