"""Real-model Tier-1 rows (memo sec 8.1 items 3-4, 8; sec 8.2 expectations).

Toy-only validation is disqualifying — the shipped structure-only suite was
green while five defect mechanisms lived on real models. Structure-only
capture is CPU-cheap, so genuine HF rows run every PR (offline: config-built
twins, cached checkpoints only).

The 8.2 expectation table is per-VERSION: green rows assert parity +
discharge on the installed leg; refusal rows assert the typed family. The
declared-band (4.x) legs run only where that transformers is installed
(clean-venv legs are CI work, verification queue).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

import torchlens as tl
from torchlens.options import CaptureOptions

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

_TRANSFORMERS_MAJOR = int(transformers.__version__.split(".")[0])


def _structure(model, ids):
    return tl.trace(model, ids, capture=CaptureOptions(structure_only=True))


def _gate(real_trace, meta_trace) -> tuple[bool, object]:
    digest_equal = tl.hash.trace(real_trace) == tl.hash.trace(meta_trace)
    discharge = meta_trace.discharge_against(real_trace)
    return digest_equal, discharge


def _skip_unless_cached(name: str):
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(name)
    except Exception:  # noqa: BLE001 — any offline-cache miss shape means skip, never fail
        pytest.skip(f"{name} not in the offline HF cache")


def test_distilgpt2_real_weights_twin() -> None:
    """The launch blocker's direct closure: real-weights checkpoint vs meta
    config twin — digest EQUAL, >=800 claims CORROBORATED (memo headline:
    291/291 records, 873 claims on the 5.x leg)."""

    from transformers import AutoModelForCausalLM

    cfg = _skip_unless_cached("distilgpt2")
    real = AutoModelForCausalLM.from_pretrained("distilgpt2")
    real.eval()
    with torch.device("meta"):
        twin = AutoModelForCausalLM.from_config(cfg)
    twin.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids_meta = torch.zeros(1, 8, dtype=torch.long, device="meta")
    real_trace = tl.trace(real, ids)
    meta_trace = _structure(twin, ids_meta)
    digest_equal, discharge = _gate(real_trace, meta_trace)
    assert digest_equal
    assert discharge.verdict.value == "corroborated"
    assert len(discharge.claims) >= 800
    assert len(real_trace.layer_list) == len(meta_trace.layer_list)


def test_bert_config_twin_buffer_parity() -> None:
    """Class-matched in-code config twin: buffer parity, zero fabricated
    writes, no host-escape flag, CORROBORATED (memo acceptance: 307/307
    records, 921 claims on the 5.x leg; 4.x no-mask legs refuse typed)."""

    from transformers import AutoModel

    cfg = _skip_unless_cached("bert-base-uncased")
    torch.manual_seed(0)
    real = AutoModel.from_config(cfg)
    real.eval()
    with torch.device("meta"):
        twin = AutoModel.from_config(cfg)
    twin.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids_meta = torch.zeros(1, 8, dtype=torch.long, device="meta")
    if _TRANSFORMERS_MAJOR < 5:
        # 8.2 row: BERT-no-mask refuses on 4.x via a padding-warning guard
        # that reads token values (guard gone in 5.x).
        with pytest.raises(Exception) as excinfo:
            _structure(twin, ids_meta)
        assert getattr(excinfo.value, "fields", {}).get("code") in (
            "value_dependent_branch_unsupported",
            "structure_only_substrate_mismatch",
            "meta_kernel_unavailable",
        )
        return
    real_trace = tl.trace(real, ids)
    meta_trace = _structure(twin, ids_meta)
    digest_equal, discharge = _gate(real_trace, meta_trace)
    assert digest_equal
    assert discharge.verdict.value == "corroborated"
    from torchlens.backends.torch.completeness_witness import (
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
    )

    assert meta_trace not in _HOST_ESCAPE_MUTABLE_WRITEBACK
    real_writes = [(real_trace[label].buffer_write_kind) for label in real_trace.buffer_write_ops]
    meta_writes = [(meta_trace[label].buffer_write_kind) for label in meta_trace.buffer_write_ops]
    assert sorted(meta_writes) == sorted(real_writes)


@pytest.mark.slow
def test_llama_7b_flagship_structural_anchors() -> None:
    """The 7B flagship row (memo 8.1 item 3): published Llama-2-7B geometry
    in code, meta int64 input_ids — 6,738,415,616 declared parameters
    EXACTLY, 32 decoder layers, 225 linear records (7 per layer x 32 + head),
    COMPLETE, no payload storage, generous wall budget (120 s vs measured
    ~8-15 s). NEVER golden-pin the total op count (config-sensitive)."""

    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        pytest.skip("llama family unavailable in this transformers")

    cfg = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        tie_word_embeddings=False,
    )
    started = time.monotonic()
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.eval()
    ids = torch.zeros(1, 8, dtype=torch.long, device="meta")
    trace = _structure(model, ids)
    elapsed = time.monotonic() - started
    assert trace.outcome.status.value == "complete"
    assert trace.num_params == 6_738_415_616
    linear_records = [layer for layer in trace.layer_list if layer.func_name == "linear"]
    assert len(linear_records) == 225  # 7 per decoder layer x 32 layers + lm_head
    assert elapsed < 120, f"flagship budget blown: {elapsed:.0f}s"
    for layer in trace.layer_list:
        assert getattr(layer, "out", None) is None
    envelope = trace.structure_evidence
    assert envelope is not None and envelope["substrate"] == "meta"


def test_decoder_with_attention_mask_refuses_value_branch() -> None:
    """8.2 row (every leg): any decoder + attention_mask is a typed
    value-branch refusal — the mask, not the architecture and not the
    transformers major, is what refuses."""

    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        pytest.skip("llama family unavailable in this transformers")

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    with torch.device("meta"):
        model = LlamaForCausalLM(cfg)
    model.eval()
    ids = torch.zeros(1, 8, dtype=torch.long, device="meta")
    mask = torch.ones(1, 8, dtype=torch.long, device="meta")
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            model,
            input_kwargs={"input_ids": ids, "attention_mask": mask},
            capture=CaptureOptions(structure_only=True),
        )
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code in (
        "value_dependent_branch_unsupported",
        "meta_kernel_unavailable",
        "structure_only_substrate_mismatch",
    ), f"expected a typed refusal family, got {excinfo.value!r}"


def test_llama_2l_input_ids_only_twin_green() -> None:
    """8.2 flagship-family row on the installed leg: input_ids-only 2-layer
    llama twin passes the full gate (the 4.x legs need W1-AC, measured green
    with the shim; the 5.x leg is green natively)."""

    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError:
        pytest.skip("llama family unavailable in this transformers")

    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    torch.manual_seed(0)
    real = LlamaForCausalLM(cfg)
    real.eval()
    with torch.device("meta"):
        twin = LlamaForCausalLM(cfg)
    twin.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids_meta = torch.zeros(1, 8, dtype=torch.long, device="meta")
    real_trace = tl.trace(real, input_kwargs={"input_ids": ids})
    meta_trace = tl.trace(
        twin,
        input_kwargs={"input_ids": ids_meta},
        capture=CaptureOptions(structure_only=True),
    )
    digest_equal, discharge = _gate(real_trace, meta_trace)
    assert digest_equal
    assert discharge.verdict.value == "corroborated"


@pytest.mark.parametrize(
    "escape",
    ["nonzero", "masked_select", "bool_index", "unique"],
)
def test_value_limitation_matrix_refuses_typed(escape: str) -> None:
    """8.1 item 8: ops with no meta kernel / value-dependent selection refuse
    typed with the user's line named; static topk succeeds (declared shape)."""

    import torch.nn as nn

    class Escaping(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.kind = kind

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.fc(x)
            if self.kind == "nonzero":
                return torch.nonzero(y)
            if self.kind == "masked_select":
                return torch.masked_select(y, y > 0)
            if self.kind == "bool_index":
                return y[y > 0]
            return torch.unique(y)

    with torch.device("meta"):
        model = Escaping(escape)
    model.eval()
    with pytest.raises(Exception) as excinfo:
        tl.trace(
            model, torch.empty(2, 4, device="meta"), capture=CaptureOptions(structure_only=True)
        )
    code = getattr(excinfo.value, "fields", {}).get("code")
    assert code in (
        "meta_kernel_unavailable",
        "value_dependent_branch_unsupported",
    ), f"{escape}: expected the typed teaching family, got {excinfo.value!r}"


def test_static_topk_succeeds_with_declared_shape() -> None:
    import torch.nn as nn

    class TopK(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            values, _indices = torch.topk(self.fc(x), k=3, dim=-1)
            return values

    with torch.device("meta"):
        model = TopK()
    model.eval()
    trace = tl.trace(
        model, torch.empty(2, 8, device="meta"), capture=CaptureOptions(structure_only=True)
    )
    assert trace.outcome.status.value == "complete"
    top_layers = [layer for layer in trace.layer_list if layer.func_name == "topk"]
    assert top_layers and tuple(top_layers[0].shape) in ((2, 3), (2, 3, 2))
