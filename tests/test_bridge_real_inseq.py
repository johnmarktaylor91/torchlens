"""inseq bridge against the real inseq package (no sys.modules fakes).

The default method must be a name inseq knows, and ``generated_texts`` must
reach inseq's own ``generated_texts=`` so the attributed target is the
caller's text; the result must equal inseq called directly.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

import torchlens as tl

inseq = pytest.importorskip("inseq")

pytestmark = [pytest.mark.optional, pytest.mark.heavy]

_TINY = "hf-internal-testing/tiny-random-gpt2"
_SOURCE = "Hello world"
_TARGET = "Hello world and more"


def _load(method: str):  # noqa: ANN202
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return inseq.load_model(_TINY, method)
    except OSError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"checkpoint {_TINY} not cached: {exc}")


def test_default_method_is_an_inseq_method() -> None:
    default = inspect.signature(tl.bridge.inseq.attribute).parameters["method"].default
    assert default in inseq.list_feature_attribution_methods()


def test_bridge_keywords_exist_in_inseq() -> None:
    from inseq.models import AttributionModel

    params = inspect.signature(AttributionModel.attribute).parameters
    assert "generated_texts" in params
    assert "attribution_method" in inspect.signature(inseq.load_model).parameters


def test_default_attribution_matches_inseq_direct() -> None:
    torch.manual_seed(0)
    direct = _load("integrated_gradients").attribute(
        _SOURCE, generated_texts=_TARGET, n_steps=4, show_progress=False
    )
    torch.manual_seed(0)
    payload = tl.bridge.inseq.attribute(
        _TINY, _SOURCE, generated_texts=_TARGET, n_steps=4, show_progress=False
    )
    assert payload["method"] == "integrated_gradients"
    bridged = payload["attributions"]
    d_seq, b_seq = direct.sequence_attributions[0], bridged.sequence_attributions[0]
    d_tokens = [t.token for t in d_seq.target]
    assert [t.token for t in b_seq.target] == d_tokens
    # generated_texts reached inseq: the attributed target is the caller's text.
    assert "".join(d_tokens).replace("Ġ", " ") == _TARGET
    torch.testing.assert_close(
        b_seq.target_attributions, d_seq.target_attributions, equal_nan=True, rtol=0, atol=0
    )
