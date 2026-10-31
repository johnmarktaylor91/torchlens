"""U04.4: ``tl.summary("<hub name>", input_shape=...)`` builds on the meta path.

Offline-safe: the config fetch is replaced by a tiny local config and any
weight download fails the test, so no network is touched.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl

transformers = pytest.importorskip("transformers")


def _tiny_gpt2_config() -> Any:
    return transformers.GPT2Config(
        n_layer=2,
        n_embd=32,
        n_head=2,
        n_positions=64,
        vocab_size=128,
        architectures=["GPT2LMHeadModel"],
    )


@pytest.fixture
def offline_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Serve the config locally and refuse every weight download."""

    fetched: list[str] = []

    def fake_config(name: str, *args: Any, **kwargs: Any) -> Any:
        assert "trust_remote_code" not in kwargs
        fetched.append(name)
        return _tiny_gpt2_config()

    def no_weights(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("tl.summary(<hub name>) must not download weights")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(transformers.PreTrainedModel, "from_pretrained", no_weights)
    return fetched


def test_hub_name_summary_builds_on_meta_without_weights(offline_hub: list[str]) -> None:
    report = tl.summary("gpt2", input_shape=(1, 8))

    assert offline_hub == ["gpt2"]
    text = str(report)
    assert "GPT2LMHeadModel" in text
    expected_params = sum(
        parameter.numel()
        for parameter in transformers.GPT2LMHeadModel(_tiny_gpt2_config()).parameters()
    )
    assert f"{expected_params:,}" in text or str(expected_params) in text


def test_hub_name_summary_needs_input_evidence(offline_hub: list[str]) -> None:
    with pytest.raises(Exception) as excinfo:
        tl.summary("gpt2")
    assert getattr(excinfo.value, "fields", {}).get("code") == (
        "weightsfree_facade_input_evidence_required"
    )


def test_hub_meta_model_is_all_meta(offline_hub: list[str]) -> None:
    from torchlens.report._summary_hub import build_hub_meta_model

    model = build_hub_meta_model("gpt2")
    assert type(model).__name__ == "GPT2LMHeadModel"
    assert not model.training
    assert all(parameter.is_meta for parameter in model.parameters())
