"""``patch_from`` source-site to target-site mapping and a Patchscopes recipe (U43.5).

The mapping reads a hidden state at one layer and position of a SOURCE prompt and
writes it at another layer and position of a TARGET prompt of a different length.
Every TorchLens result is checked against the same patch written with plain
PyTorch forward hooks. ``TestIdentityPatchscope`` reproduces the identity
Patchscope (Ghandeharioun et al., 2024) on pretrained gpt2 in the offline venue:
the source token's early-layer hidden state, placed at the ``?`` of the few-shot
identity prompt, decodes back to that token.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from support.r1_venue import IN_OFFLINE_VENUE

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.intervention.errors import HookValueError

transformers = pytest.importorskip("transformers")


def _hidden(out: Any) -> torch.Tensor:
    """Return the hidden-state tensor of a GPT-2 block output."""

    return out[0] if isinstance(out, tuple) else out


def _logits(out: Any) -> torch.Tensor:
    """Return the logits of a model output or a trace output value."""

    return out.logits if hasattr(out, "logits") else out


def _torch_patch(
    model: torch.nn.Module,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    pairs: list[tuple[int, int]],
) -> torch.Tensor:
    """Plain-PyTorch oracle: patch block outputs position by position, return logits."""

    blocks = model.transformer.h
    store: dict[str, torch.Tensor] = {}

    def _grab(module: Any, args: Any, out: Any) -> None:
        """Keep the source block output."""

        del module, args
        store["hidden"] = _hidden(out).detach().clone()

    def _put(module: Any, args: Any, out: Any) -> Any:
        """Write the mapped source positions into the target block output."""

        del module, args
        hidden = _hidden(out).clone()
        for source_position, target_position in pairs:
            hidden[:, target_position] = store["hidden"][:, source_position]
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

    handle = blocks[source_layer].register_forward_hook(_grab)
    with torch.no_grad():
        model(source_ids)
    handle.remove()
    handle = blocks[target_layer].register_forward_hook(_put)
    with torch.no_grad():
        logits = _logits(model(target_ids))
    handle.remove()
    return logits


def _torchlens_patch(
    model: torch.nn.Module,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    source_layer: int,
    target_layer: int,
    positions: Any,
) -> torch.Tensor:
    """The TorchLens spelling of the same patch, returning the target logits."""

    source = tl.trace(model, source_ids)
    edit = tl.patch_from(
        source,
        source_site=tl.module(f"transformer.h.{source_layer}"),
        positions=positions,
    )
    target = tl.trace(
        model,
        target_ids,
        intervene=tl.when(tl.module(f"transformer.h.{target_layer}"), edit),
    )
    return _logits(target[target.output_layers[0]].out).detach()


@pytest.fixture(scope="module")
def tiny_gpt2() -> torch.nn.Module:
    """Two-block random GPT-2."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=16, vocab_size=64)
    return transformers.GPT2LMHeadModel(config).eval()


_SOURCE = torch.tensor([[7, 8, 9, 10]])
_TARGET = torch.tensor([[1, 2, 3, 4, 5, 6]])


@pytest.mark.parametrize(
    ("source_layer", "target_layer", "positions", "pairs"),
    [
        (0, 1, {-1: 3}, [(-1, 3)]),
        (1, 0, [(0, 5), (2, 1)], [(0, 5), (2, 1)]),
        (1, 1, 2, [(2, 2)]),
    ],
)
def test_mapping_matches_plain_torch(
    tiny_gpt2: torch.nn.Module,
    source_layer: int,
    target_layer: int,
    positions: Any,
    pairs: list[tuple[int, int]],
) -> None:
    """Cross-layer, cross-position, different-length patches equal the hook oracle."""

    got = _torchlens_patch(
        tiny_gpt2,
        _SOURCE,
        _TARGET,
        source_layer=source_layer,
        target_layer=target_layer,
        positions=positions,
    )
    want = _torch_patch(
        tiny_gpt2,
        _SOURCE,
        _TARGET,
        source_layer=source_layer,
        target_layer=target_layer,
        pairs=pairs,
    )
    with torch.no_grad():
        clean = _logits(tiny_gpt2(_TARGET))
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)
    assert not torch.allclose(got, clean)


def test_donor_batch_of_one_broadcasts(tiny_gpt2: torch.nn.Module) -> None:
    """A one-row source patches every row of a batched target."""

    target = torch.cat([_TARGET, _TARGET.flip(1)])
    got = _torchlens_patch(
        tiny_gpt2, _SOURCE, target, source_layer=0, target_layer=1, positions={-1: 2}
    )
    want = _torch_patch(tiny_gpt2, _SOURCE, target, source_layer=0, target_layer=1, pairs=[(-1, 2)])
    torch.testing.assert_close(got, want, rtol=0.0, atol=1e-6)


def test_source_site_by_label(tiny_gpt2: torch.nn.Module) -> None:
    """A pass-qualified label names the donor op as well as a selector does."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    (donor_label,) = source[tl.module("transformer.h.0")].labels
    edit = tl.patch_from(source, source_site=donor_label, positions={-1: 3})
    target = tl.trace(tiny_gpt2, _TARGET, intervene=tl.when(tl.module("transformer.h.1"), edit))
    want = _torch_patch(
        tiny_gpt2, _SOURCE, _TARGET, source_layer=0, target_layer=1, pairs=[(-1, 3)]
    )
    torch.testing.assert_close(
        _logits(target[target.output_layers[0]].out).detach(), want, rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("positions", [[], [(0, "a")], [(0, 1, 2)], 1.5])
def test_invalid_positions_refuse_at_construction(
    tiny_gpt2: torch.nn.Module, positions: Any
) -> None:
    """Malformed position maps refuse typed when the edit is built."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    with pytest.raises(InvalidArgumentError) as info:
        tl.patch_from(source, positions=positions)
    assert info.value.fields["code"] == "patch_positions_invalid"


def test_out_of_range_position_refuses_typed(tiny_gpt2: torch.nn.Module) -> None:
    """A source position past the source prompt's end refuses typed at fire time."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    edit = tl.patch_from(source, source_site=tl.module("transformer.h.0"), positions={9: 0})
    with pytest.raises(HookValueError) as info:
        tl.trace(tiny_gpt2, _TARGET, intervene=tl.when(tl.module("transformer.h.1"), edit))
    assert info.value.fields["code"] == "patch_positions_out_of_range"


def test_mismatched_hidden_width_refuses_typed(tiny_gpt2: torch.nn.Module) -> None:
    """A donor slice of another width refuses ``patch_positions_shape_mismatch``."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    edit = tl.patch_from(
        source, source_site=tl.module("transformer.h.0.mlp.c_fc"), positions={-1: 0}
    )
    with pytest.raises(HookValueError) as info:
        tl.trace(tiny_gpt2, _TARGET, intervene=tl.when(tl.module("transformer.h.1"), edit))
    assert info.value.fields["code"] == "patch_positions_shape_mismatch"


def test_ambiguous_source_site_refuses_typed(tiny_gpt2: torch.nn.Module) -> None:
    """A source site naming more than one op never guesses a donor."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    edit = tl.patch_from(source, source_site=tl.func("layer_norm"), positions=0)
    with pytest.raises(HookValueError) as info:
        tl.trace(tiny_gpt2, _TARGET, intervene=tl.when(tl.module("transformer.h.1"), edit))
    assert info.value.fields["code"] == "patch_source_site_unresolved"


def test_mapping_is_disclosed_in_the_edit_identity(tiny_gpt2: torch.nn.Module) -> None:
    """The audit identity of the edit carries the mapping, never tensors."""

    source = tl.trace(tiny_gpt2, _SOURCE)
    edit = tl.patch_from(source, source_site="transformer.h.0:1", positions={-1: 3})
    kwargs = dict(edit.kwargs)
    assert kwargs["source_site"] == repr("transformer.h.0:1")
    assert kwargs["positions"] == repr(((-1, 3),))
    assert kwargs["position_axis"] == "1"
    assert repr(edit).startswith("tl.Edit('patch_from'")


requires_offline_venue = pytest.mark.skipif(
    not IN_OFFLINE_VENUE,
    reason=(
        "GATE R1_OFFLINE_VENUE: not in the offline preflighted venue; export "
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 with a warmed cache. In venue "
        "this suite can NEVER skip."
    ),
)


#: Few-shot identity prompt ``cat -> cat; 1135 -> 1135; hello -> hello; ? ->`` as gpt2 ids.
_IDENTITY_PROMPT = torch.tensor(
    [[9246, 4613, 3797, 26, 1367, 2327, 4613, 1367, 2327, 26, 23748, 4613, 23748, 26, 5633, 4613]]
)
_PLACEHOLDER = 14  # the " ?" token: the patched position
_READ = 15  # the final " ->": the position whose next-token prediction is decoded
_HELLO = 23748
#: Source prompts ending in one single-token word ("The capital of France is Paris", ...).
_SOURCES = (
    (464, 3139, 286, 4881, 318, 6342),
    (3347, 15063, 281, 17180),
    (2990, 1509, 321, 1973, 262, 7850),
    (1544, 2826, 262, 10047),
    (40, 1816, 284, 766, 262, 6253),
    (1026, 6729, 276, 477, 7374),
    (464, 5822, 5615, 287, 257, 16669),
    (9360, 4004, 3124, 318, 10912),
)


@pytest.fixture(scope="module")
def pretrained_gpt2() -> torch.nn.Module:
    """Pretrained gpt2 from the local cache."""

    return transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()


@requires_offline_venue
@pytest.mark.slow
@pytest.mark.real_model
class TestIdentityPatchscope:
    """Identity Patchscope recipe on pretrained gpt2, layer ``l`` -> layer 0."""

    def test_identity_patchscope_decodes_early_layers(
        self, pretrained_gpt2: torch.nn.Module
    ) -> None:
        """Early-layer states decode to their token; the last layer's no longer do.

        Measured with the plain-torch oracle before this test was written: source
        layers 0-3 patched into target layer 0 decode 8/8 words, source layer 11
        decodes 0/8, and the unpatched prompt predicts " hello".
        """

        model = copy.deepcopy(pretrained_gpt2)
        with torch.no_grad():
            unpatched = _logits(model(_IDENTITY_PROMPT))[0, _READ].argmax()
        assert int(unpatched) == _HELLO
        decoded: dict[int, list[int]] = {1: [], 11: []}
        for words in _SOURCES:
            source_ids = torch.tensor([words])
            source = tl.trace(model, source_ids)
            for layer, hits in decoded.items():
                edit = tl.patch_from(
                    source,
                    source_site=tl.module(f"transformer.h.{layer}"),
                    positions={-1: _PLACEHOLDER},
                )
                target = tl.trace(
                    model,
                    _IDENTITY_PROMPT,
                    intervene=tl.when(tl.module("transformer.h.0"), edit),
                )
                got = _logits(target[target.output_layers[0]].out).detach()
                want = _torch_patch(
                    model,
                    source_ids,
                    _IDENTITY_PROMPT,
                    source_layer=layer,
                    target_layer=0,
                    pairs=[(-1, _PLACEHOLDER)],
                )
                torch.testing.assert_close(got, want, rtol=0.0, atol=1e-5)
                hits.append(int(int(got[0, _READ].argmax()) == words[-1]))
        assert sum(decoded[1]) == len(_SOURCES)
        assert sum(decoded[11]) == 0
