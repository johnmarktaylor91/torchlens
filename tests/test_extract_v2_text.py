"""T-POSITION-shaped text rows: real tokenizer, real architectures (F18).

The LM track at zero network: the REAL vendored GPT-2 tokenizer (byte
identity ledgered in the R0 corpus) drives ``hf_collate`` against a
config-built GPT-2 (learned-absolute decoder — the measured rel 41.6%
defect class) and a config-built BERT (learned-absolute encoder). Left and
right padding at several batch sizes must agree with per-string solo
forwards under the relative bound; the pad-token refusal and EOS opt-in
are exercised on the true no-pad-token tokenizer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torchlens.dataset_extraction import extract_dataset, hf_collate, open_extraction

pytestmark = [pytest.mark.real_model, pytest.mark.heavy]

_PROMPTS = [
    "The quick brown fox jumps over the lazy dog",
    "Hello",
    "Attention is all you need for sequence transduction",
    "Deep nets",
]


@pytest.fixture(scope="module")
def gpt2_and_tokenizer(tmp_path_factory):
    """Config-built GPT-2 + the REAL vendored tokenizer, zero network."""

    transformers = pytest.importorskip("transformers")
    from tests.real_model.r0.families import load_vendored_gpt2_tokenizer

    tokenizer = load_vendored_gpt2_tokenizer(str(tmp_path_factory.mktemp("tok")))
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=2,
        attn_implementation="eager",
    )
    model = transformers.GPT2Model(config).eval()
    return model, tokenizer


def _solo_rows(model, tokenizer, layer: str = "ln_f") -> list[torch.Tensor]:
    """Per-string solo forwards: the position-correctness oracle."""

    rows = []
    with torch.no_grad():
        for prompt in _PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt")
            rows.append(model(**encoded).last_hidden_state[0])
    return rows


@pytest.mark.parametrize("padding_side", ["right", "left"])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_gpt2_padded_batches_match_solo_forwards(
    gpt2_and_tokenizer, padding_side: str, batch_size: int, tmp_path: Path
) -> None:
    """T-POSITION: right AND left padding agree with solo rows per string.

    The left legs are the measured rel 41.6% defect class: they pass ONLY
    because the engine derives mask-based position_ids (disclosed once).
    """

    model, tokenizer = gpt2_and_tokenizer
    tokenizer.padding_side = padding_side
    collate = hf_collate(tokenizer, pad_token="eos")
    kwargs = {}
    solo = _solo_rows(model, tokenizer)
    import contextlib

    disclosure = (
        pytest.warns(UserWarning, match="derived position_ids")
        if padding_side == "left" and batch_size > 1
        else contextlib.nullcontext()
    )
    with disclosure:
        extract_dataset(
            model,
            list(_PROMPTS),
            {"h": "ln_f"},
            batch_size=batch_size,
            output_dir=tmp_path,
            collate=collate,
            ragged="trim",
            progress=False,
            **kwargs,
        )
    reader = open_extraction(tmp_path)
    for index in range(len(_PROMPTS)):
        row = reader.row(index)["h"]
        reference = solo[index]
        assert row.shape == reference.shape, f"row {index} extent"
        rel = (row - reference).abs().max() / reference.abs().max()
        assert rel < 1e-4, f"row {index} rel {rel:.3e} under {padding_side} padding"


def test_gpt2_no_pad_token_refuses_then_eos_opt_in_recorded(
    gpt2_and_tokenizer, tmp_path: Path
) -> None:
    """The REAL GPT-2 tokenizer has no pad token: refusal, then the opt-in."""

    from torchlens._errors import InvalidArgumentError

    model, tokenizer = gpt2_and_tokenizer
    tokenizer.pad_token = None
    tokenizer.padding_side = "right"
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            list(_PROMPTS),
            {"h": "ln_f"},
            batch_size=2,
            collate=hf_collate(tokenizer),
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_pad_token_missing"
    extract_dataset(
        model,
        list(_PROMPTS),
        {"h": "ln_f"},
        batch_size=2,
        output_dir=tmp_path,
        collate=hf_collate(tokenizer, pad_token="eos"),
        pool="last_token",
        progress=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    observed = manifest["run"]["collate_observed"]
    assert observed["pad_token_source"] == "eos_opt_in"
    assert manifest["signature"]["collate"]["kind"] == "hf_tokenizer"


def test_tokenizer_resolves_once_per_run(gpt2_and_tokenizer, tmp_path: Path) -> None:
    """D9: tokenizer resolution happens ONCE, attached-tokenizer precedence."""

    model, tokenizer = gpt2_and_tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    calls = {"n": 0}
    real_call = tokenizer.__class__.__call__

    class _CountingTokenizer:
        """Attached tokenizer proxy counting resolutions via __call__."""

        def __init__(self, inner):
            """Wrap the inner tokenizer."""

            self._inner = inner

        def __call__(self, *args, **kwargs):
            """Delegate and count."""

            calls["n"] += 1
            return real_call(self._inner, *args, **kwargs)

        def __getattr__(self, name):
            """Delegate attribute reads."""

            return getattr(self._inner, name)

    model.tokenizer = _CountingTokenizer(tokenizer)
    try:
        extract_dataset(
            model,
            list(_PROMPTS),
            {"h": "ln_f"},
            batch_size=2,
            output_dir=tmp_path,
            pool="last_token",
            progress=False,
        )
    finally:
        del model.tokenizer
    assert calls["n"] == 2, "one tokenization per batch through the ONE resolved tokenizer"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run"]["collate_observed"]["tokenizer_source"] == "attached"


def test_bert_class_left_padding_via_engine_matches_solo(tmp_path: Path) -> None:
    """The encoder leg (measured rel 25.6% class) through the full engine."""

    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    model = transformers.BertModel(config).eval()

    rows = [[5, 6, 7, 8], [9, 10], [11, 12, 13]]

    def collate_left(items):
        """Left-pad integer token rows."""

        width = max(len(row) for row in items)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros(len(items), width, dtype=torch.long)
        for i, row in enumerate(items):
            ids[i, width - len(row) :] = torch.tensor(row)
            mask[i, width - len(row) :] = 1
        return {"input_ids": ids, "attention_mask": mask}

    with pytest.warns(UserWarning, match="derived position_ids"):
        extract_dataset(
            model,
            rows,
            {"h": "encoder.layer.0"},
            batch_size=3,
            output_dir=tmp_path,
            collate=collate_left,
            ragged="trim",
            progress=False,
        )
    reader = open_extraction(tmp_path)
    with torch.no_grad():
        for index, row in enumerate(rows):
            ids = torch.tensor([row])
            solo = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
            ).hidden_states[1][0]
            stored = reader.row(index)["h"]
            rel = (stored - solo).abs().max() / solo.abs().max()
            assert rel < 1e-4, f"row {index} rel {rel:.3e}"
