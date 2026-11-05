"""The graceful test for the pickled-checkpoint route (``requires_pickled_weight_loading``).

transformers gates ``torch.load`` of pickled weights on torch 2.6 or newer
(CVE-2025-32434): below it, ``from_pretrained`` on a ``pytorch_model.bin``
checkpoint raises a ``ValueError`` naming ``torch.load`` and pointing at
safetensors. Tests whose hub checkpoint ships only pickled weights route on
``transformers_loads_pickled_weights``; this test proves on every torch that
the probe agrees with what transformers really does with such a checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from tests.support.hf_capabilities import transformers_loads_pickled_weights

transformers = pytest.importorskip("transformers")


def test_pickled_checkpoint_loads_exactly_when_the_probe_says_so(tmp_path) -> None:
    """A ``.bin``-only checkpoint loads bit-exactly, or is refused by the CVE-2025-32434 gate."""

    config = transformers.GPT2Config(n_layer=1, n_head=2, n_embd=8, vocab_size=16, n_positions=8)
    model = transformers.GPT2Model(config).eval()
    config.save_pretrained(tmp_path)
    torch.save(model.state_dict(), tmp_path / "pytorch_model.bin")
    assert not list(tmp_path.glob("*.safetensors"))
    if transformers_loads_pickled_weights():
        loaded = transformers.GPT2Model.from_pretrained(tmp_path).eval()
        assert torch.equal(loaded.wte.weight, model.wte.weight)
        return
    with pytest.raises(ValueError, match="torch.load") as refusal:
        transformers.GPT2Model.from_pretrained(tmp_path)
    assert "safetensors" in str(refusal.value)
