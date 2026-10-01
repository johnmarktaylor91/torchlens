"""Lane F44 stage 2: the real-model persistence gate (foldA MEMO s6).

The realism rule -- toy-only validation is disqualifying: SAE-style
injected computation on a REAL GPT-2 MLP site persists, loads
degrade-to-unattested, and replays to attested; a torchvision ResNet
block-boundary splice does the same. Fixed token ids (never a tokenizer)
and random init keep the gate offline and deterministic.
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens.intervention.injection import attest_injected_ops

transformers = pytest.importorskip("transformers")

_LOGGED = tl.options.CaptureOptions(log_injections=True)


def _sae_like(out: torch.Tensor, *, hook) -> torch.Tensor:
    """SAE-style encode/gate/decode on the hooked activation."""

    z = torch.relu(out @ torch.eye(out.shape[-1]))
    return torch.sigmoid(z) * out


@pytest.mark.heavy
def test_gpt2_mlp_sae_injection_persists_and_attests(tmp_path) -> None:
    """A GPT-2 MLP-site SAE injection survives save/load and replays."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=128, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).eval()
    token_ids = torch.tensor([[3, 14, 15, 92, 65, 35]])
    # the MLP's c_fc projection (Conv1D -> addmm) is the classic SAE site
    spec = tl.when(tl.func("addmm") & tl.in_module("transformer.h.0.mlp"), _sae_like)
    logged = tl.trace(model, token_ids, intervene=spec, capture=_LOGGED)
    records = logged.injected_ops
    assert records, "the GPT-2 MLP hook recorded no injected ops"
    assert {record.provenance.spec_rule_id for record in records} == {spec.rules[0].rule_id}
    host_sites = {record.provenance.host_site_key for record in records}
    assert all(isinstance(key, str) and key.startswith("s1|") for key in host_sites)
    path = str(tmp_path / "gpt2_sae.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    restored = loaded.injected_ops
    assert len(restored) == len(records)
    assert {record.attestation for record in restored} == {"unattested"}
    assert [op.label for op in loaded.model_ops] == [op.label for op in logged.model_ops]
    report = attest_injected_ops(loaded)
    assert report.passed
    assert {row.status for row in report.rows} == {"attested"}


@pytest.mark.heavy
def test_torchvision_block_splice_injection_persists_and_attests(tmp_path) -> None:
    """A ResNet block-boundary splice's injected ops persist and replay."""

    torchvision = pytest.importorskip("torchvision")

    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(1, 3, 64, 64)

    def block_splice(out: torch.Tensor, *, hook) -> torch.Tensor:
        """Splice a channel-recombination into the block output."""

        pooled = torch.mean(out, dim=1, keepdim=True)
        return out + torch.tanh(pooled)

    # the block's residual sum is the in-place add at the block boundary
    spec = tl.when(tl.func("iadd") & tl.in_module("layer1.0"), block_splice)
    logged = tl.trace(model, x, intervene=spec, capture=_LOGGED)
    records = logged.injected_ops
    assert records, "the ResNet block splice recorded no injected ops"
    func_names = {record.func_name for record in records}
    assert {"mean", "tanh", "add"} & func_names
    path = str(tmp_path / "resnet_splice.tlspec")
    tl.save(logged, path)
    loaded = tl.load(path)
    assert len(loaded.injected_ops) == len(records)
    report = attest_injected_ops(loaded)
    assert report.passed
    assert {row.status for row in report.rows} == {"attested"}
    # the model op family is untouched by the injected rows after load
    assert not any(
        getattr(op, "injection_provenance", None) is not None for op in loaded.layer_list
    )
