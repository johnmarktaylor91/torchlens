"""LIT blocks-preset architecture matrix on real checkpoints (lane F31; T19/T24).

The memo's six-checkpoint matrix, at PR-leg scale: the four real NLP
architectures (DistilBERT / BERT / RoBERTa / GPT-2, tiny random-weight
checkpoints) prove the preset discovers exactly the config-declared stack,
that pinned ``site_key`` quads re-resolve under a ragged padded re-batch, and
that served block values are BIT-EXACT against HF ``output_hidden_states``
(with the GPT-2 ``ln_f`` closure for the final block -- the negative that
stops a future "simplification" to reading ``hidden_states`` directly).
The ViT / ResNet18 rows are the memo's deferred-vision rows (M(lit) section
8) and are deliberately NOT claimed here. Needs only transformers + cached
checkpoints -- lit-nlp itself is not imported, so this matrix also guards
environments where the LIT extra is absent.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.bridge.lit import _sites

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.heavy

# (checkpoint, task, block-stack address prefix, config depth attribute)
_MATRIX: tuple[tuple[str, str, str, str], ...] = (
    (
        "hf-internal-testing/tiny-random-distilbert",
        "seq_cls",
        "distilbert.transformer.layer",
        "num_hidden_layers",
    ),
    ("hf-internal-testing/tiny-random-bert", "seq_cls", "bert.encoder.layer", "num_hidden_layers"),
    (
        "hf-internal-testing/tiny-random-roberta",
        "seq_cls",
        "roberta.encoder.layer",
        "num_hidden_layers",
    ),
    ("sshleifer/tiny-gpt2", "causal_lm", "transformer.h", "n_layer"),
)
_NAMES = tuple(row[0] for row in _MATRIX)
_RAGGED = ("a", "a much longer ragged probe example sentence", "mid size probe")


@pytest.fixture(scope="module")
def matrix() -> dict[str, dict[str, Any]]:
    """Load EVERY matrix checkpoint before the first trace in this module.

    All constructions happen up front (model construction after capture has
    begun is the known-fragile ordering); a checkpoint missing from the local
    cache maps to ``None`` so its rows skip individually instead of killing
    the whole module.

    Returns
    -------
    dict[str, dict[str, Any]]
        ``checkpoint name -> {"net": ..., "tok": ..., "task": ...}``.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    loaded: dict[str, dict[str, Any]] = {}
    for name, task, _prefix, _depth_attr in _MATRIX:
        cls = (
            transformers.AutoModelForCausalLM
            if task == "causal_lm"
            else transformers.AutoModelForSequenceClassification
        )
        try:
            tok = transformers.AutoTokenizer.from_pretrained(name)
            net = cls.from_pretrained(name).eval()
        except (OSError, TypeError):  # pragma: no cover - environment-dependent
            loaded[name] = {"net": None, "tok": None, "task": task}
            continue
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        loaded[name] = {"net": net, "tok": tok, "task": task}
    return loaded


def _entry(matrix: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    """Return one loaded matrix entry, skipping when the checkpoint is absent.

    Parameters
    ----------
    matrix:
        The module fixture value.
    name:
        Checkpoint name.

    Returns
    -------
    dict[str, Any]
        The entry with a live ``net``.
    """

    entry = matrix[name]
    if entry["net"] is None:
        pytest.skip(f"checkpoint {name} not cached")
    return entry


def _trace(entry: dict[str, Any], texts: list[str]) -> Any:
    """Trace one right-padded batch the way the adapters do (memo D9).

    Parameters
    ----------
    entry:
        A matrix entry.
    texts:
        Prompt strings.

    Returns
    -------
    Any
        The finished ``Trace``.
    """

    enc = entry["tok"](texts, return_tensors="pt", padding=len(texts) > 1)
    kwargs = dict(enc)
    if entry["task"] == "causal_lm":
        kwargs["use_cache"] = False
    return tl.trace(entry["net"], [], input_kwargs=kwargs)


@pytest.mark.parametrize("name", _NAMES)
def test_blocks_preset_matches_config_depth(matrix: dict[str, dict[str, Any]], name: str) -> None:
    """The preset discovers exactly the config-declared consecutive stack."""

    row = next(r for r in _MATRIX if r[0] == name)
    entry = _entry(matrix, name)
    depth = int(getattr(entry["net"].config, row[3]))
    log = _trace(entry, ["a short probe"])
    stack = _sites.discover_block_stack(log)
    assert stack == tuple(f"{row[2]}.{i}" for i in range(depth))
    specs = _sites.pin_blocks(log)
    assert [spec.field_name for spec in specs] == [f"tl_block_{i}" for i in range(depth)]


@pytest.mark.parametrize("name", _NAMES)
def test_site_keys_survive_ragged_rebatch(matrix: dict[str, dict[str, Any]], name: str) -> None:
    """Pins from an unpadded probe re-resolve on a ragged padded batch (D4/D5)."""

    entry = _entry(matrix, name)
    specs = _sites.pin_blocks(_trace(entry, ["a short probe"]))
    resolved = _sites.resolve_pinned(_trace(entry, list(_RAGGED)), specs)
    assert set(resolved) == {spec.field_name for spec in specs}


@pytest.mark.parametrize("name", _NAMES[:3])
def test_encoder_blocks_bit_exact_vs_hidden_states(
    matrix: dict[str, dict[str, Any]], name: str
) -> None:
    """Every encoder block value equals HF ``output_hidden_states`` EXACTLY."""

    entry = _entry(matrix, name)
    enc = entry["tok"]("a short parity probe", return_tensors="pt")
    log = tl.trace(entry["net"], [], input_kwargs=dict(enc))
    resolved = _sites.resolve_pinned(log, _sites.pin_blocks(log))
    with torch.no_grad():
        hidden = entry["net"](**enc, output_hidden_states=True).hidden_states
    for i, field in enumerate(sorted(resolved, key=lambda f: int(f.rsplit("_", 1)[1]))):
        assert torch.equal(resolved[field].out, hidden[i + 1]), field


def test_gpt2_last_block_closes_under_ln_f(matrix: dict[str, dict[str, Any]]) -> None:
    """GPT-2 blocks are exact vs hidden states; the LAST needs ``ln_f`` (T24).

    ``hidden_states[-1]`` is POST-``ln_f`` in HF GPT-2, so the raw final block
    output must NOT equal it while ``ln_f(out)`` must equal it bit-exactly --
    the documented negative that stops replacing traced reads with
    ``output_hidden_states``.
    """

    entry = _entry(matrix, "sshleifer/tiny-gpt2")
    enc = entry["tok"]("The quick brown fox", return_tensors="pt")
    log = tl.trace(entry["net"], [], input_kwargs={**dict(enc), "use_cache": False})
    specs = _sites.pin_blocks(log)
    resolved = _sites.resolve_pinned(log, specs)
    with torch.no_grad():
        hidden = entry["net"](**enc, output_hidden_states=True, use_cache=False).hidden_states
        final_ln = entry["net"].transformer.ln_f(resolved[specs[-1].field_name].out)
    for i, spec in enumerate(specs[:-1]):
        assert torch.equal(resolved[spec.field_name].out, hidden[i + 1]), spec.field_name
    assert not torch.equal(resolved[specs[-1].field_name].out, hidden[-1])
    assert torch.equal(final_ln, hidden[-1])
