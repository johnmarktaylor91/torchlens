"""Decode-step selector on the capture-free bound generation path (R2-25, N-IV-12).

``torchlens.at_step(k)`` on a bound ``generate`` fires only during
the k-th forward of the generation (step 0 is the prefill), capture-free. Each
row is checked against plain PyTorch forward hooks that count the model's
forward calls and apply the same edit at the same step, comparing the per-step
scores ``generate`` returns.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import at_step
from torchlens.intervention.errors import SelectorCapabilityError

transformers = pytest.importorskip("transformers")

_GENERATE = {
    "max_new_tokens": 4,
    "do_sample": False,
    "output_scores": True,
    "return_dict_in_generate": True,
    "pad_token_id": 0,
}
_IDS = torch.tensor([[5, 17, 3, 42, 9]])


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    """Two-block random GPT-2 with a real ``generate``."""

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=64)
    return transformers.GPT2LMHeadModel(config).eval()


def _oracle_scores(
    base: torch.nn.Module, module_names: tuple[str, ...], steps: tuple[int, ...], fn: Any
) -> torch.Tensor:
    """Generate with ``fn`` applied to each named module's output at the given decode steps."""

    oracle = copy.deepcopy(base)
    calls = {"root": -1}

    def _count(module: Any, args: Any, kwargs: Any) -> None:
        """Count root forwards (step 0 is the prefill)."""

        del module, args, kwargs
        calls["root"] += 1

    def _edit(module: Any, args: Any, out: Any) -> Any:
        """Apply ``fn`` to the module output during the chosen steps only."""

        del module, args
        if calls["root"] not in steps:
            return None
        if isinstance(out, tuple):
            return (fn(out[0]), *out[1:])
        return fn(out)

    oracle.register_forward_pre_hook(_count, with_kwargs=True)
    for name in module_names:
        oracle.get_submodule(name).register_forward_hook(_edit)
    with torch.no_grad():
        result = oracle.generate(_IDS, **_GENERATE)
    return torch.stack(result.scores)


def _scores(result: Any) -> torch.Tensor:
    """Stack the per-step scores of a generate result."""

    return torch.stack(result.scores)


def test_module_boundary_rule_fires_at_one_decode_step(model: torch.nn.Module) -> None:
    """``tl.module(...) & at_step(2)`` zero-ablates block 0 only during decode step 2."""

    spec = tl.when(tl.module("transformer.h.0") & at_step(2), tl.zero_ablate())
    bound = spec.bind(copy.deepcopy(model))
    with torch.no_grad():
        result = bound.generate(_IDS, **_GENERATE)
    report = bound.last_report
    assert report is not None and report.trace is None
    assert report.num_fires == 1
    want = _oracle_scores(model, ("transformer.h.0",), (2,), torch.zeros_like)
    torch.testing.assert_close(_scores(result), want, rtol=0.0, atol=1e-6)
    with torch.no_grad():
        clean = _scores(copy.deepcopy(model).generate(_IDS, **_GENERATE))
    torch.testing.assert_close(_scores(result)[:2], clean[:2], rtol=0.0, atol=0.0)
    assert not torch.allclose(_scores(result)[2], clean[2])


def test_op_level_rule_fires_at_one_decode_step(model: torch.nn.Module) -> None:
    """An op-level rule qualified by ``at_step(1)`` fires only during decode step 1."""

    spec = tl.when(tl.func("layer_norm") & at_step(1), tl.scale(0.5))
    bound = spec.bind(copy.deepcopy(model))
    with torch.no_grad():
        result = bound.generate(_IDS, **_GENERATE)
    report = bound.last_report
    assert report is not None
    layer_norms = ("transformer.h.0.ln_1", "transformer.h.0.ln_2")
    layer_norms += ("transformer.h.1.ln_1", "transformer.h.1.ln_2", "transformer.ln_f")
    assert report.num_fires == len(layer_norms)
    want = _oracle_scores(model, layer_norms, (1,), lambda out: out * 0.5)
    torch.testing.assert_close(_scores(result), want, rtol=0.0, atol=1e-6)


def test_bound_steering_honours_the_decode_step(model: torch.nn.Module) -> None:
    """Bound generation applies a steering direction at the qualified steps."""

    direction = torch.ones(16)
    spec = tl.when(tl.module("transformer.h.1") & at_step(0, 3), tl.steer(direction, 2.0, dim=-1))
    bound = spec.bind(copy.deepcopy(model))
    steered = bound.generate(_IDS, **_GENERATE)
    report = bound.last_report
    assert report is not None
    assert report.num_fires == 2
    assert report.trace is None

    def _add(out: torch.Tensor) -> torch.Tensor:
        """The same steering edit, written by hand."""

        return out + 2.0 * direction

    want = _oracle_scores(model, ("transformer.h.1",), (0, 3), _add)
    torch.testing.assert_close(_scores(steered), want, rtol=0.0, atol=1e-6)


def test_step_qualifier_without_binding_or_episode_still_refuses(
    model: torch.nn.Module,
) -> None:
    """Episode-capture semantics are unchanged: a plain capture refuses the qualifier."""

    with pytest.raises(Exception) as info:
        tl.trace(
            copy.deepcopy(model),
            _IDS,
            intervene=tl.when(tl.func("layer_norm") & at_step(0), tl.scale(0.5)),
        )
    chain: list[BaseException] = []
    current: BaseException | None = info.value
    while current is not None:
        chain.append(current)
        current = current.__cause__ or current.__context__
    assert any(isinstance(error, SelectorCapabilityError) for error in chain)
