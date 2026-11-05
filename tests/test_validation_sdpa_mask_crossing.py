"""Perturbation evidence for an additive sdpa mask over saturated attention scores.

With large queries and keys every softmax row is one-hot, so mask noise sized to the
output (the generic draw) never changes which key wins and a live mask edge looked
dead. The mask now gets a margin sized to the recomputed scores at each row's winning
key. Each live mask must validate; a replay that ignores its recorded mask (a planted
capture bug) must still fail the perturbation check.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
from torchlens._data_classes.trace import Trace
from torchlens.validation import last_validation_failure
from torchlens.validation._crossing_perturbations import sdpa_mask_crossing_values

SDPA = "scaled_dot_product_attention"


class _SaturatedMaskedAttention(nn.Module):
    """Key-padding attention whose scores (~1e6) pin every softmax row to one key."""

    def __init__(self, form: str) -> None:
        super().__init__()
        self.form = form

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q = (x * 1e3).unsqueeze(1)
        k = (x * 1e3).unsqueeze(1)
        v = x.unsqueeze(1)
        padding = x.abs().sum(-1) < 0  # no padded key: an all-zero additive mask
        mask = padding.to(x.dtype).mul(-1e4).reshape(batch, 1, 1, length)
        if self.form == "full":
            mask = mask.expand(batch, 1, length, length) * 1.0
        if self.form == "keyword":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return F.scaled_dot_product_attention(q, k, v, mask)


_FORMS = ["broadcast", "full", "keyword"]


def _input() -> torch.Tensor:
    # In each batch one key out-scores the others for every query row.
    return torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]], [[2.0, -1.0], [-4.0, -4.0], [0.5, 0.0]]]
    )


@pytest.mark.parametrize("form", _FORMS)
def test_mask_over_saturated_scores_validates(form: str) -> None:
    model = _SaturatedMaskedAttention(form).eval()

    assert tl.validate(model, _input(), scope="forward", seed=0) is True


def _plant_frozen_mask(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every sdpa op replay with its recorded mask, whatever mask it is given."""

    planted_labels: list[str] = []
    original = Trace.validate_forward_pass

    def validate_with_plant(trace: Trace, *args: Any, **kwargs: Any) -> Any:
        for op in trace._layer_list:
            if op.func_name != SDPA:
                continue
            saved_kwargs = dict(op.saved_kwargs or {})
            recorded = (
                saved_kwargs["attn_mask"] if "attn_mask" in saved_kwargs else op.saved_args[3]
            )
            frozen_mask = recorded.clone()
            func = op.func

            def planted(*a: Any, _func: Any = func, _mask: Any = frozen_mask, **k: Any) -> Any:
                if "attn_mask" in k:
                    k["attn_mask"] = _mask
                else:
                    a = (*a[:3], _mask, *a[4:])
                return _func(*a, **k)

            op.func = planted
            planted_labels.append(op.label)
        return original(trace, *args, **kwargs)

    monkeypatch.setattr(Trace, "validate_forward_pass", validate_with_plant)
    return planted_labels


@pytest.mark.parametrize("form", _FORMS)
def test_replay_ignoring_its_mask_fails(monkeypatch: pytest.MonkeyPatch, form: str) -> None:
    planted_labels = _plant_frozen_mask(monkeypatch)

    with pytest.warns(tl.errors.TorchLensWarning, match="tl.validate FAILED"):
        verdict = tl.validate(
            _SaturatedMaskedAttention(form).eval(), _input(), scope="forward", seed=0
        )

    failure = last_validation_failure()
    assert planted_labels, "no op was planted; the test proves nothing"
    assert verdict is False
    assert failure is not None and failure.check_name == "perturbation"
    assert failure.func_name == SDPA


def _layer(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> Any:
    return SimpleNamespace(
        func_name=SDPA,
        saved_args=(q, k, k, mask),
        saved_kwargs={},
        parent_arg_positions={"args": {3: "m"}, "kwargs": {}},
    )


def test_moved_mask_changes_each_rows_winner() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) * 1e3
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]]]) * 1e3
    mask = torch.zeros(1, 2, 3)

    moved = sdpa_mask_crossing_values(_layer(q, k, mask), "m", mask)

    assert moved is not None and bool(torch.isfinite(moved).all())
    before = (q @ k.transpose(-2, -1) + mask).argmax(-1)
    after = (q @ k.transpose(-2, -1) + moved).argmax(-1)
    assert bool((before != after).all())


def test_single_key_or_all_keys_winning_is_left_to_the_generic_draw() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) * 1e3
    one_key = torch.tensor([[[1.0, 1.0]]])
    assert (
        sdpa_mask_crossing_values(
            _layer(q, one_key, torch.zeros(1, 1, 1)), "m", torch.zeros(1, 1, 1)
        )
        is None
    )

    two_keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]) * 1e3
    shared_mask = torch.zeros(1, 1, 2)  # each key wins one row, both share the mask row
    layer = _layer(q, two_keys, shared_mask)
    assert sdpa_mask_crossing_values(layer, "m", shared_mask) is None
