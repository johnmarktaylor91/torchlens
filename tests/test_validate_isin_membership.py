"""Validation of ``torch.isin`` edges, and the verbose report of a False verdict.

A generate loop stops on ``isin(input_ids[:, -1], eos_ids)``. The output of a
membership test moves only when an element starts or stops matching a test
value, and the generic integer perturbation (the saved range, or ``0..10`` for
a one-valued tensor) almost never lands on a token id such as 50256, so
validation reported a real edge as perturbation-insensitive and GPT-2
``generate`` validated False. These tests pin that a real ``isin`` edge
validates, that the membership-toggling perturbation flips every element's
membership, that an ``isin`` whose replay ignores its recorded operands still
fails, and that ``verbose=True`` names the failing check and op.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.validation import last_validation_failure
from torchlens.validation._membership_perturbation import isin_membership_toggle_values

EOS = 50256


class _EosStop(nn.Module):
    """A stopping check shaped like Hugging Face's end-of-sequence criterion."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("eos", torch.tensor([EOS]))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        done = torch.isin(ids[:, -1], self.eos)
        return ids.float() * (~done).float().unsqueeze(-1)


def _ids() -> torch.Tensor:
    return torch.tensor([[464, 3139, 286, 4881, 318]])


def _layer(args: tuple[Any, ...], slot_labels: dict[int, str], **kwargs: Any) -> Any:
    return SimpleNamespace(
        func_name="isin",
        saved_args=args,
        saved_kwargs=kwargs,
        parent_arg_positions={"args": slot_labels, "kwargs": {}},
    )


def test_eos_stopping_check_validates() -> None:
    assert tl.validate(_EosStop().eval(), _ids(), scope="forward", seed=0) is True


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_elements_toggle_flips_every_membership(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    test = torch.tensor([EOS, 7, 11])
    elements = torch.randint(0, 20, (4, 6), generator=generator)
    elements[0, :3] = torch.tensor([EOS, 7, 11])
    layer = _layer((elements, test), {0: "elements_parent"})

    toggled = isin_membership_toggle_values(layer, "elements_parent", elements)

    assert toggled is not None
    assert toggled.dtype == elements.dtype and toggled.shape == elements.shape
    assert bool((torch.isin(toggled, test) != torch.isin(elements, test)).all())


@pytest.mark.parametrize("has_member", [False, True])
def test_test_elements_toggle_changes_a_membership(has_member: bool) -> None:
    elements = torch.tensor([3, 9, EOS if has_member else 12])
    test = torch.tensor([EOS, EOS + 1])
    layer = _layer((elements, test), {1: "test_parent"})

    toggled = isin_membership_toggle_values(layer, "test_parent", test)

    assert toggled is not None and toggled.shape == test.shape
    assert not torch.equal(torch.isin(elements, toggled), torch.isin(elements, test))


def test_toggle_declines_assume_unique_and_unknown_slots() -> None:
    elements, test = torch.tensor([1, 2]), torch.tensor([2])
    assert (
        isin_membership_toggle_values(
            _layer((elements, test), {0: "p"}, assume_unique=True), "p", elements
        )
        is None
    )
    assert isin_membership_toggle_values(_layer((elements, test), {0: "p"}), "q", elements) is None


def _isin_ops(trace: Trace) -> Iterator[Any]:
    return (op for op in trace._layer_list if op.func_name == "isin")


def _plant_isin_func(
    monkeypatch: pytest.MonkeyPatch, planted: Callable[..., torch.Tensor]
) -> list[str]:
    """Replace every captured ``isin`` callable before replay; return the labels planted."""

    planted_labels: list[str] = []
    original = Trace.validate_forward_pass

    def validate_with_planted_isin(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        for op in _isin_ops(trace):
            op.func = planted
            planted_labels.append(op.label)
        return original(trace, *args, **kwargs)

    monkeypatch.setattr(Trace, "validate_forward_pass", validate_with_planted_isin)
    return planted_labels


def _ignores_elements(elements: torch.Tensor, test: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """A capture bug: the replayed isin does not read its recorded elements."""

    return torch.isin(torch.zeros_like(elements), test, **kwargs)


def _wrong_test_elements(elements: torch.Tensor, test: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """A capture bug: the replayed isin tests against a value it was never given."""

    return torch.isin(elements, torch.full_like(test, EOS + 1), **kwargs)


@pytest.mark.parametrize("planted", [_ignores_elements, _wrong_test_elements])
def test_planted_isin_capture_bug_still_fails(
    monkeypatch: pytest.MonkeyPatch, planted: Callable[..., torch.Tensor]
) -> None:
    planted_labels = _plant_isin_func(monkeypatch, planted)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(_EosStop().eval(), _ids(), scope="forward", seed=0)

    failure = last_validation_failure()
    assert verdict is False and planted_labels
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == "isin"


def test_verbose_false_verdict_names_check_and_op(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _plant_isin_func(monkeypatch, _ignores_elements)

    with pytest.warns(tl.errors.TorchLensWarning):
        verdict = tl.validate(_EosStop().eval(), _ids(), scope="forward", seed=0, verbose=True)

    printed = capsys.readouterr().out
    failure = last_validation_failure()
    assert verdict is False and failure is not None and failure.op_label is not None
    assert "tl.validate FAILED" in printed
    assert "perturbation" in printed and failure.op_label in printed and "isin" in printed


def test_verbose_passing_verdict_prints_no_failure(capsys: pytest.CaptureFixture[str]) -> None:
    assert tl.validate(_EosStop().eval(), _ids(), scope="forward", seed=0, verbose=True) is True
    assert "FAILED" not in capsys.readouterr().out


@pytest.mark.real_model
@pytest.mark.slow
def test_gpt2_generate_validates() -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    kwargs = {"max_new_tokens": 4}

    assert tl.validate(model.generate, ids, input_kwargs=kwargs, scope="forward", seed=0) is True


@pytest.mark.real_model
@pytest.mark.slow
def test_gpt2_generate_planted_isin_bug_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    planted_labels = _plant_isin_func(monkeypatch, _ignores_elements)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(
            model.generate, ids, input_kwargs={"max_new_tokens": 4}, scope="forward", seed=0
        )

    failure = last_validation_failure()
    assert verdict is False and planted_labels
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == "isin"
