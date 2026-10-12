"""inseq bridge against the real inseq package (walk U22.1).

Forced decoding must return the requested text: the bridge passes
``generated_texts=`` through inseq's own keyword and refuses a result that
attributes any other text. TorchLens method spellings map to inseq's names.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from support.hf_cache import skip_unless_hf_checkpoint_cached

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.bridge import inseq as binseq

inseq = pytest.importorskip("inseq")

pytestmark = [pytest.mark.optional, pytest.mark.heavy]

_TINY = "hf-internal-testing/tiny-random-gpt2"
_SOURCE = "Hello world"
_TARGET = "Hello world and more"


@pytest.fixture(autouse=True)
def _offline_tiny_checkpoint() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    skip_unless_hf_checkpoint_cached(_TINY)


def test_forced_decoding_returns_the_requested_text() -> None:
    payload = tl.bridge.inseq.attribute(
        _TINY, _SOURCE, method="saliency", generated_texts=_TARGET, show_progress=False
    )
    result = payload["attributions"]
    assert result.info["generated_texts"] == [_TARGET]
    tokens = [token.token for token in result.sequence_attributions[0].target]
    assert "".join(tokens).replace("Ġ", " ") == _TARGET


def test_torchlens_method_spelling_maps_to_inseq() -> None:
    payload = tl.bridge.inseq.attribute(
        _TINY, _SOURCE, method="input_x_grad", generated_texts=_TARGET, show_progress=False
    )
    assert payload["method"] == "input_x_gradient"


def test_unknown_method_refuses_before_loading() -> None:
    with pytest.raises(InvalidArgumentError) as info:
        tl.bridge.inseq.attribute(_TINY, _SOURCE, method="integrated_grads")
    assert info.value.fields["code"] == "bridge_inseq_method_unknown"
    assert "integrated_gradients" in info.value.fields["remedy"]


def test_a_result_for_another_text_is_refused() -> None:
    """inseq reports an ignored keyword only in its log; the bridge checks the text."""

    free_generation = SimpleNamespace(info={"generated_texts": ["Hello worldorld R R"]})
    with pytest.raises(InvalidArgumentError) as info:
        binseq._check_forced_decoding(free_generation, _TARGET)
    assert info.value.fields["code"] == "bridge_inseq_forced_decoding_ignored"
    binseq._check_forced_decoding(SimpleNamespace(info={"generated_texts": [_TARGET]}), _TARGET)
