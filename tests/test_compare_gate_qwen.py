"""A-GATE real-model leg: two real Qwen2.5-0.5B-Instruct turns (foldB s5 row 5).

Toy-only validation is disqualifying for chain territory: two successive
turns of one real chat model are exactly the Path-1 shape (one live model
object, different inputs, ``same_object`` relationship), and the topology
refusal is the exchange-side requirement this lane precedes chain reads for.
Composition per the real-model matrix row: topology refusal x save/load x
rank audit, every refusal asserting its stable code (test law 3).

Offline by design: the checkout's pinned revision must already sit in the
local HF cache (it is a cached matrix row); the test skips, never downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import BundleRelationshipError
from torchlens.intervention.types import Relationship

pytestmark = pytest.mark.slow

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# The public HF git revision pinned by tests/real_model/artifact_registry.jsonl.
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"  # pragma: allowlist secret


def _chat_ids(tokenizer: object, messages: list[dict[str, str]]) -> torch.Tensor:
    """Return chat-templated input ids for one conversation state.

    Parameters
    ----------
    tokenizer:
        Qwen tokenizer.
    messages:
        Chat messages in transformers chat format.

    Returns
    -------
    torch.Tensor
        ``input_ids`` tensor with the generation prompt appended.
    """

    encoded = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    return encoded["input_ids"]


def test_two_real_turns_topology_refusal_and_rank_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two real conversation turns refuse comparison for the right reasons.

    One model load carries every leg: (1) the live pair derives the identity
    relationship yet refuses on input VALUES (Path 1 on a real model);
    (2) with the ``successor_of`` row the refusal keys on TOPOLOGY, taking
    precedence over input evidence; (3) both survive the Bundle artifact
    round trip; (4) the loaded pair's relationship rank only ever goes DOWN
    (the save/load rank audit on a real model).
    """

    transformers = pytest.importorskip("transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL_ID, revision=_REVISION)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, revision=_REVISION, dtype=torch.float32
        )
    except Exception as exc:  # noqa: BLE001 - cache miss skips, never downloads
        pytest.skip(f"pinned {_MODEL_ID}@{_REVISION[:8]} not in the local HF cache: {exc}")
    model.eval()

    turn1_ids = _chat_ids(tokenizer, [{"role": "user", "content": "Name one prime number."}])
    turn2_ids = _chat_ids(
        tokenizer,
        [
            {"role": "user", "content": "Name one prime number."},
            {"role": "assistant", "content": "7 is a prime number."},
            {"role": "user", "content": "Name another one."},
        ],
    )
    assert turn1_ids.shape != turn2_ids.shape

    turn1 = tl.trace(model, turn1_ids)
    turn2 = tl.trace(model, turn2_ids)
    bundle = tl.bundle({"turn1": turn1, "turn2": turn2})

    # Path-1 shape on a real chat model: the derived relationship is the
    # identity rank, exactly the evidence the old gate accepted as proof of
    # input equality.
    assert bundle.relationship("turn1", "turn2") is Relationship.SAME_OBJECT

    # WITHOUT the ordering row, the refusal is the input predicate's own
    # (right reason: the turns captured different input values).
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at("model.norm")
    assert excinfo.value.fields["code"] == "bundle_gate_input_values_differ"

    # WITH the ordering row, topology takes precedence over input evidence.
    bundle.relate({"kind": "successor_of", "from": "turn2", "to": "turn1", "params": {}})
    with pytest.raises(BundleRelationshipError) as excinfo:
        bundle.compare_at("model.norm")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]

    # Everything repeated after the Bundle artifact round trip (test law 2).
    path = tmp_path / "qwen_chain.tlspec"
    bundle.save(str(path))
    loaded = tl.load(str(path))

    loaded_rel = loaded.relationship("turn1", "turn2")
    # Rank audit: identity evidence does not survive serialization; the
    # loaded pair settles to the graph/input-signature evidence that
    # round-trips (different lengths -> different input signature).
    assert loaded_rel is Relationship.SHARED_GRAPH_DIFFERENT_INPUT

    with pytest.raises(BundleRelationshipError) as excinfo:
        loaded.compare_at("model.norm")
    assert excinfo.value.fields["code"] == "bundle_gate_ordering_topology"
    assert excinfo.value.fields["ordering_path"] == ["successor_of"]
