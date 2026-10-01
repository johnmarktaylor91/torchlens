"""Declared episode derivation on a REAL model (lane F40b; foldA s6 F40).

The realism arms the D8 rebuild must flip from refused to supported, on the
R0 roster's REAL distilgpt2 class (vendored config, zero network): a root
returning what real ``generate()`` returns (prompt+completion); the
KV-cache last-token feed (the exact shape the deleted arithmetic
``cache_len`` guessed wrong); a ``digest`` kind reading real float logits;
and a dict/ModelOutput-shaped root resolved through the declared slot."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import EpisodeSpec

pytest.importorskip("transformers")

from tests.real_model.r0.families import SEED, VOCAB, build_distilgpt2

# heavy, not smoke: real 2-layer distilgpt2 episodes measure over the 5 s
# smoke budget on the dev box (tiered honestly, like the F40a real arm).
pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

N_STEPS = 3


class _AppendAllPromptPlusCompletion(nn.Module):
    """Append-all greedy feed returning FULL ids: real generate()'s shape."""

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        for _ in range(self.n_steps):
            logits = self.model(input_ids=ids).logits
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
        return ids


class _KVCacheLastTokenFeed(nn.Module):
    """KV-cache greedy feed: prefill once, then last-token-only steps.

    Returns prompt+completion. This is the feed whose per-step input shape
    the deleted ``cache_len`` arithmetic (prompt length + step) guessed
    wrong; the rebuilt derivation reads the root output tail and never
    guesses a cache fact.
    """

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=ids, use_cache=True)
        for _ in range(self.n_steps - 1):
            next_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            outputs = self.model(
                input_ids=next_token,
                past_key_values=outputs.past_key_values,
                use_cache=True,
            )
        final_token = outputs.logits[:, -1, :].argmax(-1, keepdim=True)
        return torch.cat([ids, final_token], dim=1)


class _LogitsRoot(nn.Module):
    """Float root: the per-step logits row (the hidden-state/digest shape)."""

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        rows = []
        for _ in range(self.n_steps):
            logits = self.model(input_ids=ids).logits
            rows.append(logits[:, -1, :])
            ids = torch.cat([ids, logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        return torch.stack(rows, dim=1)


class _DictRoot(nn.Module):
    """ModelOutput-shaped root: full ids under a named slot plus a float."""

    def __init__(self, model: nn.Module, n_steps: int) -> None:
        super().__init__()
        self.model = model
        self.n_steps = n_steps

    def forward(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        last_logits = None
        for _ in range(self.n_steps):
            last_logits = self.model(input_ids=ids).logits
            ids = torch.cat([ids, last_logits[:, -1, :].argmax(-1, keepdim=True)], dim=1)
        return {"sequences": ids, "scores": last_logits[:, -1, :]}


def _prompt() -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, VOCAB, (1, 8), generator=generator)


def _column(trace: tl.Trace) -> list:
    ledger = trace.episode
    assert ledger is not None
    assert [row.status for row in ledger.rows] == ["complete"] * N_STEPS
    return [row.step_output for row in ledger.rows]


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_real_generate_shape_and_kv_cache_feed_agree():
    """R13-C1 on the real class: prompt+completion roots STOP refusing, and
    the KV-cache last-token feed derives the SAME evidence column as the
    append-all feed (identical weights, identical greedy episode)."""

    prompt = _prompt()
    append_all = _AppendAllPromptPlusCompletion(build_distilgpt2("eager"), N_STEPS)
    kv_cache = _KVCacheLastTokenFeed(build_distilgpt2("eager"), N_STEPS)

    append_trace = tl.trace(
        append_all,
        prompt,
        episode=EpisodeSpec(stepped_module=append_all.model, n_steps=N_STEPS),
    )
    kv_trace = tl.trace(
        kv_cache,
        prompt,
        episode=EpisodeSpec(stepped_module=kv_cache.model, n_steps=N_STEPS),
    )

    append_column = _column(append_trace)
    kv_column = _column(kv_trace)
    assert append_column == kv_column
    assert all(isinstance(entry, tuple) and len(entry) == 1 for entry in append_column)
    # The evidence column is exactly the completion: the prompt prefix stays
    # out (tail alignment), matching the root output's own tail.
    root_out = append_trace.output_ops[0].out
    assert root_out.shape[-1] == prompt.shape[-1] + N_STEPS
    assert [entry[0] for entry in append_column] == root_out[0, -N_STEPS:].tolist()
    header = append_trace.episode.header
    assert header.step_output_kind == "tokens"
    assert header.step_output_from == "output"
    assert isinstance(header.capture_digest, str) and len(header.capture_digest) == 64


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_real_float_logits_root_digest_kind():
    """R13-C2 on the real class: a float logits root captures under
    kind='digest' with per-step content digests along the declared axis."""

    model = _LogitsRoot(build_distilgpt2("eager"), N_STEPS)
    trace = tl.trace(
        model,
        _prompt(),
        episode=EpisodeSpec(
            stepped_module=model.model,
            n_steps=N_STEPS,
            step_output_kind="digest",
            step_axis=1,
        ),
    )
    column = _column(trace)
    assert all(isinstance(entry, str) and entry.startswith("sha256:") for entry in column)
    assert len(set(column)) == N_STEPS  # distinct per-step content
    assert trace.episode.header.step_axis == 1


@pytest.mark.filterwarnings("default:Tensor shape changed for.*:UserWarning")
def test_real_dict_root_declared_slot():
    """A dict/ModelOutput-shaped real root resolves the declared slot."""

    prompt = _prompt()
    model = _DictRoot(build_distilgpt2("eager"), N_STEPS)
    trace = tl.trace(
        model,
        prompt,
        episode=EpisodeSpec(
            stepped_module=model.model,
            n_steps=N_STEPS,
            step_output_from="sequences",
        ),
    )
    column = _column(trace)
    assert all(isinstance(entry, tuple) for entry in column)
    assert trace.episode.header.step_output_from == "sequences"

    # The greedy reference: the same weights' append-all emitted tokens.
    reference = _AppendAllPromptPlusCompletion(build_distilgpt2("eager"), N_STEPS)
    expected = reference(prompt)[0, -N_STEPS:].tolist()
    assert [entry[0] for entry in column] == expected
