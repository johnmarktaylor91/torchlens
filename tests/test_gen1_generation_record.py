"""The generation recorder: ``tl.record(..., generate_kwargs=...)``.

One real Hugging Face ``generate()`` runs with its KV cache while one fastlog
Recorder holds every forward it issues. The oracle for each decode step is an
independent full forward over that step's whole prefix with the cache off,
read by a plain PyTorch forward hook; the tolerance was registered before the
first acceptance run (float32, ``rtol=1e-4``, ``atol=1e-5``).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import build_distilgpt2, build_llama, build_t5  # noqa: E402
from torchlens.fastlog._generation import GenerationRecording  # noqa: E402

RTOL, ATOL = 1e-4, 1e-5
PROMPT = torch.tensor([[454, 137, 139, 386]])
FAMILIES: dict[str, tuple[Callable[[str], nn.Module], tuple[str, ...]]] = {
    "distilgpt2": (
        build_distilgpt2,
        ("transformer.h.0.mlp", "transformer.h.1.mlp", "transformer.ln_f"),
    ),
    "llama": (build_llama, ("model.layers.0.mlp", "model.layers.1.mlp", "model.norm")),
}


def _greedy(n: int) -> dict[str, object]:
    return {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}


def _save(sites: tuple[str, ...]):
    selector = tl.module(sites[0])
    for site in sites[1:]:
        selector = selector | tl.module(site)
    return selector


def _site_payload(records, site: str) -> torch.Tensor:
    matches = [
        record.ram_payload
        for record in records
        if record.ctx.address == site or str(record.ctx.address).startswith(site + ".")
    ]
    assert len(matches) == 1, (site, [record.ctx.address for record in records])
    return matches[0]


def _no_cache_reference(model: nn.Module, prefix: torch.Tensor, site: str) -> torch.Tensor:
    seen: dict[str, torch.Tensor] = {}
    handle = model.get_submodule(site).register_forward_hook(
        lambda module, args, out: seen.__setitem__("out", out.detach().clone())
    )
    try:
        with torch.no_grad():
            model(prefix, use_cache=False)
    finally:
        handle.remove()
    return seen["out"]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_decode_step_matches_a_full_no_cache_forward(family: str) -> None:
    build, sites = FAMILIES[family]
    model = build("eager")
    n_new = 8
    plain = model.generate(PROMPT, **_greedy(n_new))

    tokens, generation = tl.record(
        model, PROMPT, save=_save(sites), generate_kwargs=_greedy(n_new), return_output=True
    )

    assert isinstance(generation, GenerationRecording)
    assert torch.equal(tokens, plain)
    assert torch.equal(generation.sequences, plain)
    assert "forward" not in model.__dict__
    prompt = PROMPT.shape[1]
    assert generation.n_steps == n_new  # the prefill plus n_new - 1 decode forwards
    for entry in generation.steps:
        records = generation.records_at(entry.step)
        prefix = plain[:, : entry.positions.stop]
        if entry.step == 0:
            assert entry.kind == "prefill" and entry.positions == range(0, prompt)
        else:
            assert entry.kind == "decode"
            assert entry.positions == range(prompt + entry.step - 1, prompt + entry.step)
        for site in sites:
            recorded = _site_payload(records, site)
            reference = _no_cache_reference(model, prefix, site)
            assert recorded.shape[-2] == len(entry.positions)
            torch.testing.assert_close(
                recorded, reference[:, entry.positions.start :], rtol=RTOL, atol=ATOL
            )


@pytest.mark.heavy
def test_cost_is_linear_in_generated_tokens() -> None:
    model = build_distilgpt2("eager")
    save = _save(FAMILIES["distilgpt2"][1])
    runs = {n: tl.record(model, PROMPT, save=save, generate_kwargs=_greedy(n)) for n in (12, 48)}
    totals = {}
    for n, generation in runs.items():
        decode = generation.steps[1:]
        assert len(decode) == n - 1
        for counter in ("n_ops", "n_records", "saved_bytes"):
            values = {getattr(entry, counter) for entry in decode}
            assert len(values) == 1, (n, counter, sorted(values))
        totals[n] = {
            counter: sum(getattr(entry, counter) for entry in generation.steps)
            for counter in ("n_ops", "n_records", "saved_bytes")
        }
        assert generation.recording.n_records == totals[n]["n_records"]
    short, long = runs[12].steps, runs[48].steps
    for counter in ("n_ops", "n_records", "saved_bytes"):
        prefill = getattr(short[0], counter)
        per_step = getattr(short[1], counter)
        assert getattr(long[0], counter) == prefill
        assert getattr(long[1], counter) == per_step
        assert totals[12][counter] == prefill + 11 * per_step
        assert totals[48][counter] == prefill + 47 * per_step
        ratio = (totals[48][counter] - prefill) / (totals[12][counter] - prefill)
        assert abs(ratio - 47 / 11) <= 0.01


def test_sampled_generation_records_its_seed_and_reproduces() -> None:
    model = build_distilgpt2("eager")
    settings = {"max_new_tokens": 4, "do_sample": True, "top_k": 20, "use_cache": True}
    torch.manual_seed(7)
    plain = model.generate(PROMPT, **settings)

    generation = tl.record(
        model, PROMPT, save=tl.module("transformer.ln_f"), generate_kwargs=settings, seed=7
    )

    assert torch.equal(generation.sequences, plain)
    assert generation.settings["do_sample"] is True
    assert generation.settings["top_k"] == 20
    assert generation.settings["random_seed"] == 7
    assert len(generation.settings["torch_rng_state_sha256"]) == 16
    assert generation.not_recorded


def test_steps_answer_positions_and_refuse_missing_ones() -> None:
    model = build_distilgpt2("eager")
    generation = tl.record(
        model, PROMPT, save=tl.module("transformer.ln_f"), generate_kwargs=_greedy(3)
    )

    assert generation.step_of_position(0).step == 0
    assert generation.step_of_position(4).step == 1
    assert generation.step_of_position(5).step == 2
    with pytest.raises(tl.errors.InvalidArgumentError) as missing_position:
        generation.step_of_position(6)  # the last token is never fed back
    assert missing_position.value.fields["code"] == "record_generation_step_missing"
    with pytest.raises(tl.errors.InvalidArgumentError) as missing_step:
        generation.records_at(3)
    assert missing_step.value.fields["code"] == "record_generation_step_missing"
    assert len(generation.records_at(2)) == 1


def _refusal(call: Callable[[], object]) -> tl.errors.InvalidArgumentError:
    with pytest.raises(tl.errors.InvalidArgumentError) as excinfo:
        call()
    assert excinfo.value.fields["code"] == "record_generation_unsupported"
    assert excinfo.value.fields["remedy"]
    return excinfo.value


def test_a_model_without_generate_refuses_and_its_remedy_runs() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    x = torch.randn(1, 4)

    error = _refusal(lambda: tl.record(model, x, save=tl.func("relu"), generate_kwargs={}))

    assert "no callable generate()" in str(error)
    assert tl.record(model, x, save=tl.func("relu")).n_records == 1


class _NoCacheGenerator(nn.Module):
    """A toy LM whose generate() recomputes the whole prefix every step."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 2) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(input_ids=input_ids)
            input_ids = torch.cat([input_ids, logits[:, -1:].argmax(-1)], dim=-1)
        return input_ids


def test_a_generate_without_a_cache_path_refuses() -> None:
    torch.manual_seed(0)
    model = _NoCacheGenerator().eval()

    error = _refusal(
        lambda: tl.record(model, torch.tensor([[1, 2]]), save=tl.func("linear"), generate_kwargs={})
    )

    assert "without a KV cache" in str(error)
    assert "forward" not in model.__dict__


@pytest.mark.parametrize(
    ("prompt_rows", "extra", "fragment"),
    [
        (1, {"num_beams": 2}, "beam search"),
        (1, {"num_return_sequences": 2, "do_sample": True}, "more than one return sequence"),
        (2, {}, "a batch of 2 prompts"),
        (1, {"use_cache": False}, "use_cache=False"),
    ],
)
def test_unsupported_decoding_refuses_with_its_reason(
    prompt_rows: int, extra: dict[str, object], fragment: str
) -> None:
    model = build_distilgpt2("eager")
    prompt = PROMPT.repeat(prompt_rows, 1)

    error = _refusal(
        lambda: tl.record(
            model,
            prompt,
            save=tl.module("transformer.ln_f"),
            generate_kwargs={**_greedy(2), **extra},
        )
    )

    assert fragment in str(error)
    assert fragment in error.fields["reason"]


def test_batched_prompt_remedy_runs() -> None:
    model = build_distilgpt2("eager")
    prompt = PROMPT.repeat(2, 1)

    generation = tl.record(
        model, prompt[0:1], save=tl.module("transformer.ln_f"), generate_kwargs=_greedy(2)
    )

    assert generation.n_steps == 2


def test_intervene_and_encoder_decoder_refuse() -> None:
    model = build_distilgpt2("eager")
    spec = tl.when(tl.module("transformer.ln_f"), tl.zero_ablate())
    _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            save=tl.module("transformer.ln_f"),
            intervene=spec,
            generate_kwargs=_greedy(2),
        )
    )
    t5 = build_t5("eager")
    error = _refusal(
        lambda: tl.record(t5, PROMPT, save=tl.func("linear"), generate_kwargs=_greedy(2))
    )
    assert "encoder-decoder" in str(error)


def test_session_end_walk_does_not_grow_with_the_step_count(monkeypatch) -> None:
    """Each pass's model-graph cleanup walk visits the same objects at every decode step."""

    import torchlens.backends.torch.model_prep as model_prep
    from torchlens.fastlog._recorder import Recorder

    visits = [0]
    walk = model_prep._clear_session_tensor_metadata

    def counting_walk(*args, **kwargs):
        visits[0] += 1
        return walk(*args, **kwargs)

    per_pass: list[int] = []
    log = Recorder.log

    def counting_log(self, *args, **kwargs):
        before = visits[0]
        try:
            return log(self, *args, **kwargs)
        finally:
            per_pass.append(visits[0] - before)

    monkeypatch.setattr(model_prep, "_clear_session_tensor_metadata", counting_walk)
    monkeypatch.setattr(Recorder, "log", counting_log)
    model = build_distilgpt2("eager")
    tl.record(model, PROMPT, save=tl.module("transformer.ln_f"), generate_kwargs=_greedy(16))

    assert len(per_pass) == 16
    assert len(set(per_pass[1:])) == 1, per_pass


@pytest.mark.parametrize(
    ("record_kwargs", "extra", "fragment"),
    [
        ({"halt": lambda ctx: False}, {}, "halted forward"),
        ({"on_forward_error": "return_partial"}, {}, "partial forward-error modes"),
        ({}, {"prompt_lookup_num_tokens": 2}, "assisted decoding"),
    ],
)
def test_halt_partial_modes_and_assisted_decoding_refuse(
    record_kwargs: dict[str, object], extra: dict[str, object], fragment: str
) -> None:
    model = build_distilgpt2("eager")

    error = _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            save=tl.module("transformer.ln_f"),
            generate_kwargs={**_greedy(2), **extra},
            **record_kwargs,
        )
    )

    assert fragment in error.fields["reason"]
