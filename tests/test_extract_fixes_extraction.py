"""Lane A11 extraction fails-open fixes (extract MEMO items 0 and 3).

Pins, per the trilab extract MEMO:

- ``load_extraction`` reads shards with ``mmap=True`` + ``weights_only=True``
  (item 0; measured 13.4x on selective reads, retroactive on existing
  artifacts) and returns byte-identical activations.
- ``stimulus_ids=`` refuses typed in in-memory mode (item 0 / D2: a validated
  argument that can neither affect nor accompany the result is a false
  affordance).
- D5 left padding is corrected or refused, never silent: non-right-aligned
  attention masks get mask-derived ``position_ids`` when the model's forward
  declares that parameter (disclosed once per run), and a typed refusal when
  it does not. Measured defect: BERT rel 25.6% / GPT-2 rel 41.6% wrong.
- Every forward runs under ``no_grad`` + ``eval`` with every submodule's exact
  ``training`` flag restored in a ``finally`` block (item 3 / D3), and a
  completed compatible resume is a true no-op that never touches the model.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._errors import InvalidArgumentError
from torchlens.dataset_extraction import extract_dataset, load_extraction
from torchlens.errors._base import TorchLensWarning
from torchlens.utils._torch_compat import TorchCapabilityWarning


class _AbsPosModel(nn.Module):
    """Learned-absolute toy: wrong under left padding unless positions given.

    The forward signature mirrors the HF encoder layout
    ``(input_ids, attention_mask, token_type_ids, position_ids)`` so the
    positional-carrier detection and injection paths are both exercised.
    """

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(50, 8)
        self.pos = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)
        hidden = self.proj(self.emb(input_ids) + self.pos(position_ids))
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1)
        return hidden


class _NoPosModel(nn.Module):
    """Mask-consuming toy whose forward declares no position_ids parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(50, 8)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids) * attention_mask.unsqueeze(-1)


def _left_padded_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, attention_mask) with left-padded rows."""

    ids = torch.tensor([[0, 0, 7, 8, 9], [0, 1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    return ids, mask


def _tuple_stimuli(ids: torch.Tensor, mask: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
    """Per-stimulus (ids_row, mask_row) tuples for extract_dataset."""

    return [(ids[i], mask[i]) for i in range(ids.shape[0])]


@pytest.mark.smoke
def test_load_extraction_passes_mmap_and_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader reads shards with mmap=True + weights_only=True, values equal."""

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    stimuli = torch.randn(6, 3)
    out_dir = tmp_path / "artifact"
    # shard_format="pt": the mmap + weights_only contract is the .pt codec's
    # (v2 defaults to safetensors; .pt stays writable by explicit opt-out).
    extract_dataset(
        model,
        stimuli,
        ["relu"],
        batch_size=2,
        output_dir=out_dir,
        progress=False,
        shard_format="pt",
    )

    seen_kwargs: list[dict[str, object]] = []
    real_torch_load = torch.load

    def _spy_load(*args: object, **kwargs: object) -> object:
        seen_kwargs.append(dict(kwargs))
        return real_torch_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch, "load", _spy_load)
    loaded = load_extraction(out_dir)
    assert seen_kwargs, "load_extraction did not go through torch.load"
    for kwargs in seen_kwargs:
        assert kwargs.get("mmap") is True
        assert kwargs.get("weights_only") is True

    monkeypatch.undo()
    eager = OrderedDict()
    for path in loaded.batch_paths:
        for key, tensor in torch.load(path, weights_only=True).items():
            eager.setdefault(key, []).append(tensor)
    for key, tensors in eager.items():
        assert torch.equal(loaded.activations[key], torch.cat(tensors, dim=0))


@pytest.mark.smoke
def test_stimulus_ids_in_memory_refuses_typed() -> None:
    """In-memory mode refuses stimulus_ids= with a teaching typed error (D2)."""

    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            model,
            torch.randn(4, 3),
            ["relu"],
            batch_size=2,
            stimulus_ids=["a", "b", "c", "d"],
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_stimulus_ids_in_memory_unsupported"
    assert "output_dir" in str(excinfo.value)


@pytest.mark.smoke
def test_left_padded_batch_derives_position_ids_and_matches_reference() -> None:
    """Non-right-aligned masks get derived position_ids: bit-exact fix (D5.1)."""

    torch.manual_seed(0)
    model = _AbsPosModel().eval()
    ids, mask = _left_padded_pair()

    with pytest.warns(TorchLensWarning, match="derived position_ids") as record:
        out = extract_dataset(
            model, _tuple_stimuli(ids, mask), ["proj"], batch_size=2, progress=False
        )
    # Key on the disclosure code, not bare TorchLensWarning membership: a
    # floor-torch install may also fire a one-time TorchCapabilityWarning
    # (itself a TorchLensWarning) from an unrelated capability probe during
    # this capture, which must not be mistaken for the position-ids notice.
    disclosures = [
        entry.message
        for entry in record
        if isinstance(entry.message, TorchLensWarning)
        and entry.message.fields.get("code") == "extraction_position_ids_derived"
    ]
    assert len(disclosures) == 1
    assert disclosures[0].fields["remedy"].startswith("right-pad the batch")

    derived = (mask.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        reference = model.proj(model.emb(ids) + model.pos(derived))
        uncorrected = model.proj(
            model.emb(ids) + model.pos(torch.arange(ids.shape[1]).unsqueeze(0).expand_as(ids))
        )
    got = next(iter(out.values()))
    assert torch.equal(got, reference)
    assert not torch.allclose(uncorrected, reference)


@pytest.mark.smoke
def test_left_padded_batch_without_position_ids_refuses_typed() -> None:
    """A model whose forward has no position_ids parameter refuses typed (D5.2)."""

    model = _NoPosModel().eval()
    ids, mask = _left_padded_pair()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(model, _tuple_stimuli(ids, mask), ["mul"], batch_size=2, progress=False)
    assert excinfo.value.fields["code"] == "extraction_left_padding_unsupported"
    message = str(excinfo.value)
    assert "padding_side" in message
    assert "batch_size=1" in message
    # Disclosed conservatism: relative-position models are value-correct under
    # left padding; the refusal is deliberately fail-closed for them.
    assert "relative position" in message


@pytest.mark.smoke
def test_right_aligned_batches_are_untouched() -> None:
    """Right-padded masks trigger neither derivation nor a disclosure warning."""

    torch.manual_seed(0)
    model = _AbsPosModel().eval()
    ids = torch.tensor([[7, 8, 9, 0, 0], [1, 2, 3, 4, 0]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        # A floor-torch install may fire a one-time TorchCapabilityWarning
        # (an unrelated capability probe tripped by this capture) ahead of
        # the business-logic check this test guards; tolerate that category
        # specifically without loosening the "no position-ids disclosure"
        # guarantee.
        warnings.simplefilter("ignore", TorchCapabilityWarning)
        out = extract_dataset(
            model, _tuple_stimuli(ids, mask), ["proj"], batch_size=2, progress=False
        )
    with torch.no_grad():
        default_positions = model.proj(
            model.emb(ids) + model.pos(torch.arange(ids.shape[1]).unsqueeze(0).expand_as(ids))
        )
    assert torch.equal(next(iter(out.values())), default_positions)


@pytest.mark.smoke
def test_caller_supplied_position_ids_are_trusted() -> None:
    """A batch already carrying position_ids is passed through unchanged."""

    torch.manual_seed(0)
    model = _AbsPosModel().eval()
    ids, mask = _left_padded_pair()
    caller_positions = torch.ones_like(ids)
    stimuli = [(ids[i], mask[i], None, caller_positions[i]) for i in range(2)]

    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        # See test_right_aligned_batches_are_untouched: tolerate an unrelated
        # one-time floor-torch capability notice without loosening the
        # "no position-ids disclosure" guarantee this test checks.
        warnings.simplefilter("ignore", TorchCapabilityWarning)
        out = extract_dataset(model, stimuli, ["proj"], batch_size=2, progress=False)
    with torch.no_grad():
        reference = model.proj(model.emb(ids) + model.pos(caller_positions))
    assert torch.equal(next(iter(out.values())), reference)


@pytest.mark.smoke
def test_inference_guard_restores_mixed_train_eval_flags_and_buffers() -> None:
    """Forwards run in eval/no_grad; exact per-submodule flags restore (D3)."""

    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU())
    model.train()
    model[2].training = False  # mixed tree: ReLU deliberately eval
    flags_before = [module.training for module in model.modules()]
    buffers_before = {name: buffer.clone() for name, buffer in model.named_buffers()}

    out = extract_dataset(model, torch.randn(6, 3), ["relu"], batch_size=3, progress=False)

    assert [module.training for module in model.modules()] == flags_before
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, buffers_before[name]), f"extraction mutated live buffer {name!r}"
    assert not any(tensor.requires_grad for tensor in out.values())


@pytest.mark.smoke
def test_inference_guard_restores_flags_on_refusal_mid_run() -> None:
    """The exact-restore covers exception paths, including typed refusals."""

    model = _NoPosModel()
    model.train()
    ids, mask = _left_padded_pair()
    with pytest.raises(InvalidArgumentError):
        extract_dataset(model, _tuple_stimuli(ids, mask), ["mul"], batch_size=2, progress=False)
    assert model.training is True
    assert model.emb.training is True


@pytest.mark.smoke
def test_completed_resume_is_a_true_noop(tmp_path: Path) -> None:
    """A completed compatible resume touches neither model mode nor device (D3)."""

    forward_calls: list[int] = []
    to_calls: list[object] = []

    class _Spy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            forward_calls.append(1)
            return torch.relu(self.linear(x))

        def to(self, *args: object, **kwargs: object):  # type: ignore[override]
            to_calls.append(args)
            return super().to(*args, **kwargs)  # type: ignore[arg-type]

    model = _Spy().eval()
    stimuli = torch.randn(4, 3)
    out_dir = tmp_path / "artifact"
    extract_dataset(
        model, stimuli, ["relu"], batch_size=2, output_dir=out_dir, device="cpu", progress=False
    )
    n_forwards = len(forward_calls)
    n_moves = len(to_calls)
    assert n_forwards == 2 and n_moves == 1

    model.train()  # a later mode change must survive the no-op resume
    paths = extract_dataset(
        model,
        stimuli,
        ["relu"],
        batch_size=2,
        output_dir=out_dir,
        device="cpu",
        resume=True,
        progress=False,
    )
    assert len(paths) == 2
    assert len(forward_calls) == n_forwards, "completed resume ran the model"
    assert len(to_calls) == n_moves, "completed resume moved the model"
    assert model.training is True, "completed resume flipped model mode"


@pytest.mark.real_model
@pytest.mark.heavy
def test_bert_class_left_padded_extraction_matches_right_padded() -> None:
    """R0 BERT row: left-padded extraction equals right-padded extraction.

    Uses the real BertModel class at a small random-init config (R0 band).
    Without derived position_ids the learned-absolute encoder was measured
    rel 25.6% wrong under left padding while every other check passed.
    """

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

    lengths = [3, 5, 4]
    seq_len = 6
    rows = [torch.randint(1, 64, (length,)) for length in lengths]

    def _padded(left: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
        stimuli = []
        for row in rows:
            pad = torch.zeros(seq_len - row.shape[0], dtype=row.dtype)
            keep = torch.ones(row.shape[0], dtype=torch.long)
            gap = torch.zeros(seq_len - row.shape[0], dtype=torch.long)
            if left:
                stimuli.append((torch.cat([pad, row]), torch.cat([gap, keep])))
            else:
                stimuli.append((torch.cat([row, pad]), torch.cat([keep, gap])))
        return stimuli

    # layernorm_5 = encoder.layer.1.output.LayerNorm on this 2-layer config
    # (embeddings + 2x(attention.output + output)): the final hidden state.
    site = ["layernorm_5"]
    with pytest.warns(TorchLensWarning, match="derived position_ids"):
        left_out = extract_dataset(model, _padded(left=True), site, batch_size=3, progress=False)
    right_out = extract_dataset(model, _padded(left=False), site, batch_size=3, progress=False)

    left_tensor = next(iter(left_out.values()))
    right_tensor = next(iter(right_out.values()))
    for index, length in enumerate(lengths):
        offset = seq_len - length
        left_tokens = left_tensor[index, offset:]
        right_tokens = right_tensor[index, :length]
        assert torch.allclose(left_tokens, right_tokens, atol=1e-4), (
            f"stimulus {index}: left-padded activations diverge from right-padded"
        )


@pytest.mark.real_model
@pytest.mark.heavy
def test_gpt2_class_mapping_batch_derivation_is_exact() -> None:
    """R0 GPT-2 row: the mapping-carrier derivation fixes the decoder exactly.

    GPT-2's forward has ``past_key_values`` before ``attention_mask``, so a
    two-element positional batch cannot name a mask; the mapping carrier is
    where detection applies (end-to-end mapping routing lands with the C04
    kwargs envelope). Measured defect: rel 41.6% wrong under left padding.
    """

    transformers = pytest.importorskip("transformers")
    from torchlens._extraction.engine import _correct_envelope_positions
    from torchlens.dataset_extraction import BatchEnvelope

    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        attn_implementation="eager",
    )
    model = transformers.GPT2Model(config).eval()

    ids = torch.tensor([[0, 0, 7, 8, 9], [0, 1, 2, 3, 4]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    envelope = BatchEnvelope(
        args=(),
        kwargs={"input_ids": ids, "attention_mask": mask},
        row_count=2,
        mask=mask,
        disclosure={"kind": "test"},
    )

    with pytest.warns(TorchLensWarning, match="derived position_ids"):
        corrected_envelope, source = _correct_envelope_positions(model, envelope, 0, {})
    assert source == "derived"
    corrected = dict(corrected_envelope.kwargs)
    assert torch.equal(corrected["position_ids"], (mask.long().cumsum(-1) - 1).clamp(min=0))

    with torch.no_grad():
        fixed = model(**corrected).last_hidden_state
        uncorrected = model(input_ids=ids, attention_mask=mask).last_hidden_state
        per_row = []
        for index in range(ids.shape[0]):
            length = int(mask[index].sum())
            row_ids = ids[index, -length:].unsqueeze(0)
            per_row.append(model(input_ids=row_ids).last_hidden_state[0])

    for index in range(ids.shape[0]):
        length = int(mask[index].sum())
        assert torch.allclose(fixed[index, -length:], per_row[index], atol=1e-4), (
            f"stimulus {index}: corrected activations diverge from single-sequence forward"
        )
    assert not torch.allclose(uncorrected[0, -3:], per_row[0], atol=1e-2), (
        "left padding no longer wrong without correction -- re-measure the defect"
    )


@pytest.mark.real_model
@pytest.mark.heavy
def test_resnet18_class_extraction_leaves_batchnorm_state_untouched() -> None:
    """R0 ResNet row: a train-mode resnet18 harvest mutates zero buffers (D3)."""

    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None)
    model.train()
    buffers_before = {name: buffer.clone() for name, buffer in model.named_buffers()}
    flags_before = [module.training for module in model.modules()]

    stimuli = torch.randn(4, 3, 64, 64)
    out = extract_dataset(model, stimuli, ["layer4"], batch_size=2, progress=False)

    assert [module.training for module in model.modules()] == flags_before
    mutated = [
        name
        for name, buffer in model.named_buffers()
        if not torch.equal(buffer, buffers_before[name])
    ]
    assert mutated == [], f"extraction mutated {len(mutated)} live buffers: {mutated[:5]}"
    assert all(not tensor.requires_grad for tensor in out.values())
