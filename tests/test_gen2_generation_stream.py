"""The streaming generation recorder: ``tl.record(..., generate_kwargs=..., generation=...)``.

Per-token reducers, KV-cache slices as sites, the ring buffer with disk spill
and the retained-byte guard, on the same GPT-2-shaped model as the GEN-1 tests.
The oracle for every reduced value is the reducer applied to an independent
full no-cache forward over that step's prefix (float32, ``rtol=1e-4``,
``atol=1e-5``, the tolerance GEN-1 registered).
"""

from __future__ import annotations

import collections
import gc
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("transformers")

from tests.real_model.r0.families import build_distilgpt2  # noqa: E402
from torchlens.recording import GenerationOptions, TokenSeries, reducers  # noqa: E402
from torchlens.recording._generation import GenerationRecording  # noqa: E402

RTOL, ATOL = 1e-4, 1e-5
PROMPT = torch.tensor([[454, 137, 139, 386]])
MLP0, MLP1, LN_F = "transformer.h.0.mlp", "transformer.h.1.mlp", "transformer.ln_f"
WIDTH = 64


def _greedy(n: int) -> dict[str, object]:
    return {"max_new_tokens": n, "min_new_tokens": n, "do_sample": False, "use_cache": True}


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


def _probe_terms() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(1, WIDTH, generator=generator), torch.randn(1, generator=generator)


def _refusal(call: Callable[[], object], code: str) -> tl.errors.InvalidArgumentError:
    with pytest.raises(tl.errors.InvalidArgumentError) as excinfo:
        call()
    assert excinfo.value.fields["code"] == code
    assert excinfo.value.fields["remedy"]
    return excinfo.value


def test_reducers_keep_only_reduced_values_and_match_the_oracle() -> None:
    model = build_distilgpt2("eager")
    n_new = 8
    weight, bias = _probe_terms()
    declared = {
        MLP0: reducers.norm(),
        MLP1: reducers.top_k(3, dim=-1, position_dim=1),
        LN_F: reducers.probe((weight, bias)),
    }
    plain = model.generate(PROMPT, **_greedy(n_new))

    generation = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(reducers=declared, max_saved_bytes=None),
    )

    assert isinstance(generation, GenerationRecording)
    assert torch.equal(generation.sequences, plain)
    assert generation.series_names == (MLP0, MLP1, LN_F)
    assert generation.evicted_steps == {}
    recorded_positions = plain.shape[1] - 1
    for site, reducer in declared.items():
        series = generation.series(site)
        assert isinstance(series, TokenSeries)
        assert series.positions == range(0, recorded_positions)
        assert series.values.shape[0] == recorded_positions == len(series)
        for entry in generation.steps:
            prefix = plain[:, : entry.positions.stop]
            reference = reducer(_no_cache_reference(model, prefix, site))
            reference = reference[entry.positions.start :]
            got = series.values[entry.positions.start : entry.positions.stop]
            if got.dtype.is_floating_point:
                torch.testing.assert_close(got, reference, rtol=RTOL, atol=ATOL)
            else:
                assert torch.equal(got, reference)
    # Every retained payload is the reduced value, never the activation.
    for entry in generation.steps:
        records = generation.records_at(entry.step)
        assert len(records) == 3
        for record in records:
            assert record.ram_payload is not None
            assert record.ram_payload.shape[0] == len(entry.positions)
            assert record.ram_payload.numel() <= 3 * len(entry.positions)
            assert record.metadata["reducer"].startswith(("norm", "top_k", "probe"))
            assert record.metadata["reduced_from_shape"][-1] == WIDTH
    decode_bytes = {entry.saved_bytes for entry in generation.steps[1:]}
    assert decode_bytes == {4 + 3 * 8 + 4}  # one norm, three int64 indices, one score
    assert generation.peak_saved_bytes == sum(entry.saved_bytes for entry in generation.steps)


def test_a_save_selector_and_reducers_compose() -> None:
    model = build_distilgpt2("eager")

    generation = tl.record(
        model,
        PROMPT,
        save=tl.module(MLP1),
        generate_kwargs=_greedy(3),
        generation=GenerationOptions(reducers={MLP0: reducers.norm()}),
    )

    for entry in generation.steps:
        by_site = {
            call.rsplit(":", 1)[0]: record
            for record in generation.records_at(entry.step)
            for call in record.ctx.output_of_module_calls
        }
        assert set(by_site) == {MLP0, MLP1}
        assert by_site[MLP0].ram_payload is not None
        assert by_site[MLP0].ram_payload.shape == (len(entry.positions),)
        assert by_site[MLP1].ram_payload is not None
        assert by_site[MLP1].ram_payload.shape == (1, len(entry.positions), WIDTH)
        assert "reducer" not in by_site[MLP1].metadata
    assert generation.series_names == (MLP0,)


class _Fitted:
    """A fitted scikit-learn linear estimator's duck type."""

    def __init__(self, coef: torch.Tensor, intercept: torch.Tensor) -> None:
        self.coef_ = coef.numpy()
        self.intercept_ = intercept.numpy()


@pytest.mark.parametrize("outputs", [1, 3])
def test_probe_accepts_fitted_estimators_linear_layers_and_pairs(outputs: int) -> None:
    generator = torch.Generator().manual_seed(outputs)
    weight = torch.randn(outputs, WIDTH, generator=generator)
    bias = torch.randn(outputs, generator=generator)
    linear = nn.Linear(WIDTH, outputs)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    payload = torch.randn(1, 5, WIDTH, generator=generator)
    expected = payload[0] @ weight.T + bias
    if outputs == 1:
        expected = expected[:, 0]

    for fitted in (_Fitted(weight, bias), linear, (weight, bias)):
        scores = reducers.probe(fitted)(payload)
        assert scores.shape == expected.shape
        torch.testing.assert_close(scores, expected, rtol=RTOL, atol=ATOL)

    assert repr(reducers.probe(linear)) == f"Reducer('probe(nn.Linear, outputs={outputs})')"
    _refusal(lambda: reducers.probe(object()), "record_generation_reducer_invalid")
    wrong_width = reducers.probe((weight[:, :8], bias))
    _refusal(lambda: wrong_width(payload), "record_generation_reducer_invalid")


def _kv_oracle(model: nn.Module, sequence: torch.Tensor, layer: int) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = model(sequence, use_cache=True)
    cache = out.past_key_values
    entry = cache.layers[layer] if hasattr(cache, "layers") else None
    keys = entry.keys if entry is not None else cache.key_cache[layer]
    values = entry.values if entry is not None else cache.value_cache[layer]
    return {
        f"kv.{layer}.key": torch.movedim(keys[0], 1, 0),
        f"kv.{layer}.value": torch.movedim(values[0], 1, 0),
    }


def test_kv_sites_match_the_cache_of_a_plain_forward() -> None:
    model = build_distilgpt2("eager")
    n_new = 6
    plain = model.generate(PROMPT, **_greedy(n_new))

    generation = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(kv_sites=(1,), reducers={LN_F: reducers.norm()}),
    )

    assert generation.series_names == (LN_F, "kv.1.key", "kv.1.value")
    oracle = _kv_oracle(model, plain[:, :-1], layer=1)
    for name, expected in oracle.items():
        series = generation.series(name)
        assert series.positions == range(0, plain.shape[1] - 1)
        assert series.values.shape == expected.shape  # [positions, heads, head_dim]
        torch.testing.assert_close(series.values, expected, rtol=RTOL, atol=ATOL)
    every = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(2),
        generation=GenerationOptions(kv_sites=True),
    )
    assert every.series_names == ("kv.0.key", "kv.0.value", "kv.1.key", "kv.1.value")
    assert "kv_sites=" in every.not_recorded[0]
    error = _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            generate_kwargs=_greedy(2),
            generation=GenerationOptions(kv_sites=(5,)),
        ),
        "record_generation_reducer_invalid",
    )
    assert "layer 5" in str(error)


def test_ring_buffer_drops_or_spills_older_forwards(tmp_path: Path) -> None:
    model = build_distilgpt2("eager")
    n_new = 8
    declared = {MLP0: reducers.norm(), LN_F: reducers.norm()}
    unbounded = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(reducers=declared),
    )
    full = unbounded.series(MLP0)

    dropped = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(reducers=declared, keep_last=3),
    )
    spill = tmp_path / "spill"
    spilled = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(reducers=declared, keep_last=3, spill=spill),
    )

    n_steps = unbounded.n_steps
    assert dropped.evicted_steps == dict.fromkeys(range(n_steps - 3), "dropped")
    assert spilled.evicted_steps == dict.fromkeys(range(n_steps - 3), "spilled")
    kept = dropped.series(MLP0)
    assert kept.positions == range(full.positions.stop - 3, full.positions.stop)
    torch.testing.assert_close(kept.values, full.values[-3:])
    read_back = spilled.series(MLP0)
    assert read_back.positions == full.positions
    torch.testing.assert_close(read_back.values, full.values)
    assert sorted(path.name for path in spill.iterdir()) == [
        f"step-{step:06d}.pt" for step in range(n_steps - 3)
    ]
    for record in dropped.records_at(0):
        assert record.ram_payload is None and record.metadata["payload"] == "dropped"
    for record in spilled.records_at(0):
        assert record.ram_payload is None and record.metadata["payload"] == "spilled"
        assert Path(record.metadata["spill_file"]).exists()
    assert dropped.records_at(n_steps - 1)[0].ram_payload is not None
    assert dropped.peak_saved_bytes < unbounded.peak_saved_bytes
    assert spilled.peak_saved_bytes == dropped.peak_saved_bytes
    decode = unbounded.steps[1].saved_bytes
    assert dropped.peak_saved_bytes <= unbounded.steps[0].saved_bytes + 3 * decode


def test_budget_guard_refuses_early_with_its_remedy() -> None:
    model = build_distilgpt2("eager")
    forwards: list[int] = []
    # The recorder calls the model once more inside each routed forward, so the
    # root module's hook fires twice per step; the decoder stack runs once.
    handle = model.transformer.register_forward_hook(lambda module, args, out: forwards.append(1))
    try:
        projected = _refusal(
            lambda: tl.record(
                model,
                PROMPT,
                save=tl.module(MLP0),
                generate_kwargs=_greedy(2000),
                generation=GenerationOptions(max_saved_bytes=4096),
            ),
            "record_generation_budget_exceeded",
        )
    finally:
        handle.remove()

    assert len(forwards) == 2  # the prefill and the first decode step, then the projection
    assert "2,000 new tokens would retain" in projected.fields["reason"]
    assert "keep_last=" in projected.fields["remedy"]
    running = _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            save=tl.module(MLP0),
            generate_kwargs=_greedy(4),
            generation=GenerationOptions(keep_last=2, max_saved_bytes=256),
        ),
        "record_generation_budget_exceeded",
    )
    assert "forward 0 brought" in running.fields["reason"]
    within = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(4),
        generation=GenerationOptions(reducers={MLP0: reducers.norm()}, max_saved_bytes=4096),
    )
    assert within.n_steps == 4


def test_reducer_contract_and_option_refusals(tmp_path: Path) -> None:
    model = build_distilgpt2("eager")
    gen = _greedy(2)

    def wrong_shape(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum()

    error = _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            generate_kwargs=gen,
            generation=GenerationOptions(reducers={MLP0: wrong_shape}),
        ),
        "record_generation_reducer_invalid",
    )
    assert "first axis runs over the forward's positions" in str(error)
    missing = _refusal(
        lambda: tl.record(
            model,
            PROMPT,
            generate_kwargs=gen,
            generation=GenerationOptions(reducers={"transformer.h.9.mlp": reducers.norm()}),
        ),
        "record_generation_reducer_invalid",
    )
    assert "recorded no output" in str(missing)
    _refusal(lambda: GenerationOptions(spill=tmp_path), "record_generation_options_invalid")
    _refusal(lambda: GenerationOptions(keep_last=0), "record_generation_options_invalid")
    _refusal(lambda: GenerationOptions(kv_sites=(-1,)), "record_generation_options_invalid")
    _refusal(lambda: GenerationOptions(max_saved_bytes=0), "record_generation_options_invalid")
    _refusal(
        lambda: GenerationOptions(reducers={MLP0: "norm"}), "record_generation_options_invalid"
    )
    _refusal(lambda: reducers.top_k(0), "record_generation_reducer_invalid")
    without_generate = _refusal(
        lambda: tl.record(model, PROMPT, save=tl.module(MLP0), generation=GenerationOptions()),
        "record_generation_options_invalid",
    )
    assert "needs generate_kwargs=" in str(without_generate)
    plain = tl.record(model, PROMPT, save=tl.module(MLP0), generate_kwargs=gen)
    assert plain.series_names == () and plain.options is None
    _refusal(lambda: plain.series(MLP0), "record_generation_step_missing")
    widened = tl.record(
        model,
        PROMPT,
        generate_kwargs=gen,
        generation=GenerationOptions(reducers={MLP0: reducers.norm()}),
    )
    _refusal(lambda: widened.series(LN_F), "record_generation_step_missing")
    assert repr(widened.options.reducers[MLP0]) == "Reducer('norm(ord=2.0)')"


def test_past_forwards_release_their_inputs_and_logits() -> None:
    """Only the declared sites and reducers survive a step: the KV cache a decode step
    took as input and the logits it produced are freed, so memory grows with the number
    of tokens, not with its square (the 2,000-token acceptance found the cache held)."""
    model = build_distilgpt2("eager")
    n_new = 6
    generation = tl.record(
        model,
        PROMPT,
        generate_kwargs=_greedy(n_new),
        generation=GenerationOptions(reducers={LN_F: reducers.norm()}, max_saved_bytes=None),
    )
    assert generation.n_steps == n_new
    assert generation.series(LN_F).values.shape[0] == PROMPT.shape[1] + n_new - 1

    gc.collect()
    live = collections.Counter(
        tuple(o.shape) for o in gc.get_objects() if isinstance(o, torch.Tensor)
    )
    heads, head_dim = model.config.n_head, WIDTH // model.config.n_head
    # Decode step s takes the cache of prompt + s - 1 positions as its input; the recording
    # keeps none of them (the newest pass's inputs stay with the recording's live state, as
    # in a plain recording), and generate() returns none, so every older length is gone.
    cache_shapes = [(1, heads, PROMPT.shape[1] + s, head_dim) for s in range(n_new - 1)]
    assert {shape: live[shape] for shape in cache_shapes if live[shape]} == {}
    # The logits of every pass but the newest are released too.
    assert live[(1, 1, model.config.vocab_size)] <= 1
