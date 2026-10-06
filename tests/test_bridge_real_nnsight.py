"""nnsight bridge against the real nnsight package (no sys.modules fakes).

A live ``LanguageModel.trace`` tracer exposes no mapping, ``to_dict()`` or
``nodes``; the bridge must refuse it typed instead of returning an empty
payload, and must carry values the user saved with nnsight through unchanged.
"""

from __future__ import annotations

import os

import pytest
import torch
from support.hf_cache import skip_unless_hf_checkpoint_cached

import torchlens as tl

nnsight = pytest.importorskip("nnsight")

pytestmark = [pytest.mark.optional, pytest.mark.heavy]

_TINY = "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture(scope="module")
def lm():  # noqa: ANN201
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    skip_unless_hf_checkpoint_cached(_TINY)
    return nnsight.LanguageModel(_TINY, device_map="cpu", dispatch=True)


def test_real_tracer_is_refused_not_emptied(lm) -> None:  # noqa: ANN001
    with lm.trace("Hello") as tracer:
        lm.transformer.h[0].output.save()
    with pytest.raises(TypeError, match="live nnsight tracer"):
        tl.bridge.nnsight.from_trace(tracer)


def test_saved_values_pass_through_a_mapping(lm) -> None:  # noqa: ANN001
    with lm.trace("Hello"):
        hidden = lm.transformer.h[0].output.save()
    direct = hidden[0] if isinstance(hidden, tuple) else hidden
    assert isinstance(direct, torch.Tensor)
    payload = tl.bridge.nnsight.from_trace(
        {"nodes": [{"name": "transformer.h.0", "value": direct}], "source": "nnsight"}
    )
    assert payload["schema"] == "torchlens.nnsight_trace.v1"
    assert payload["nodes"][0]["value"] is direct
    assert payload["metadata"] == {"source": "nnsight"}
