"""The direct payload path records exactly what the general save path records.

``backends/torch/_ops_fresh_payload.py`` saves a fresh plain output under the
default options without the general path's wrappers. Each test captures the same
model twice, once as shipped and once with the direct path switched off, and
requires the saved payloads, their fields, the dedup annotations, the payload
aliasing and the save-budget ledger (or the refusal) to be identical.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._save_budget import SaveBudgetExceededError
from torchlens.backends.torch import _ops_activations, _ops_fresh_payload
from torchlens.options import CaptureOptions


class _SameObjectReturns(nn.Module):
    """Outputs that are not fresh: ``contiguous`` returns its receiver, a view shares storage."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix fresh outputs, a same-object return and a whole-storage view."""

        y = torch.relu(x * 2)
        z = y.contiguous()
        v = (z + 1).view(-1)
        return v * 3 + z.sum()


def _chain() -> nn.Module:
    """Return a small parameterised chain whose outputs require gradients."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Tanh())


def _record(trace: tl.Trace) -> dict[str, Any]:
    """Return every saved-payload fact the two paths must agree on."""

    ops = list(trace._layer_list)
    pointers: dict[int, int] = {}
    shares: list[int | None] = []
    rows: list[tuple[Any, ...]] = []
    for index, op in enumerate(ops):
        out = op.out if isinstance(op.out, torch.Tensor) else None
        annotations = dict(getattr(op, "annotations", None) or {})
        if out is not None:
            pointer = out.untyped_storage().data_ptr()
            shares.append(pointers.setdefault(pointer, index))
        else:
            shares.append(None)
        rows.append(
            (
                op.label,
                op.has_saved_activation,
                op.shape,
                op.dtype,
                op.cost.activation_memory,
                None if out is None else out.detach().clone(),
                None if out is None else (out.requires_grad, type(out.grad_fn).__name__),
                annotations.get("dedup_source_id"),
                annotations.get("dedup_source_version"),
                annotations.get("dedup_reference_label"),
            )
        )
    budget = getattr(trace, "_save_budget_accountant", None)
    ledgers = {}
    if budget is not None:
        for key, ledger in budget.ledgers.items():
            entries = sorted(
                (e.physical_bytes, e.live_refs) for e in ledger.retained_storage.values()
            )
            ledgers[key] = (ledger.committed_bytes, ledger.num_saved, entries)
    return {"rows": rows, "shares": shares, "ledgers": ledgers}


def _same(left: dict[str, Any], right: dict[str, Any]) -> None:
    """Assert two records are equal, comparing tensors by value."""

    assert left["shares"] == right["shares"]
    assert left["ledgers"] == right["ledgers"]
    assert len(left["rows"]) == len(right["rows"])
    for a, b in zip(left["rows"], right["rows"], strict=True):
        assert a[:5] == b[:5]
        assert (a[5] is None) == (b[5] is None)
        if a[5] is not None:
            assert torch.equal(a[5], b[5])
        assert a[6:] == b[6:]


def _both(
    monkeypatch: pytest.MonkeyPatch, capture: Callable[[], tl.Trace]
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Capture with the direct path on, then off; return both records and direct-path calls."""

    calls = [0]
    real = _ops_activations.save_fresh_plain_activation

    def counting(*args: Any) -> None:
        calls[0] += 1
        real(*args)

    with monkeypatch.context() as patch:
        patch.setattr(_ops_activations, "save_fresh_plain_activation", counting)
        direct = _record(capture())
    with monkeypatch.context() as patch:
        patch.setattr(_ops_activations, "fresh_plain_payload_eligible", lambda *a: None)
        general = _record(capture())
    return direct, general, calls[0]


@pytest.mark.parametrize("requires_grad", [False, True])
def test_direct_path_matches_general_path_on_a_chain(
    monkeypatch: pytest.MonkeyPatch, requires_grad: bool
) -> None:
    """Default options: same payloads, fields, aliasing and ledger; the direct path ran."""

    model = _chain()
    x = torch.randn(4, 16, requires_grad=requires_grad)
    direct, general, calls = _both(monkeypatch, lambda: tl.trace(model, x))
    assert calls > 0
    _same(direct, general)


def test_direct_path_matches_with_detached_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """``detach_saved_activations=True`` takes the direct path with the same results."""

    model = _chain()
    x = torch.randn(4, 16)
    options = CaptureOptions(detach_saved_activations=True)
    direct, general, calls = _both(monkeypatch, lambda: tl.trace(model, x, capture=options))
    assert calls > 0
    _same(direct, general)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_same_object_returns_and_views_keep_dedup_and_ledger(
    monkeypatch: pytest.MonkeyPatch, requires_grad: bool
) -> None:
    """Same-object returns and whole-storage views record the same payloads and ledger."""

    x = torch.randn(3, 5, requires_grad=requires_grad)
    direct, general, calls = _both(monkeypatch, lambda: tl.trace(_SameObjectReturns(), x))
    assert calls > 0
    _same(direct, general)


def _fields(label: str) -> dict[str, Any]:
    """Return the field subset ``_save_activation_fields`` reads for a direct call."""

    return {
        "_label_raw": f"{label}_raw",
        "_layer_label_raw": label,
        "annotations": {},
        "backward_ready": False,
        "detach_saved_activations": False,
        "func_name": "manual_test",
        "output_device": "same",
    }


def _save_twice(direct: bool) -> list[dict[str, Any]]:
    """Save one tensor twice and another twice; return the four field dicts."""

    torch.manual_seed(0)
    trace = tl.trace(nn.Identity(), torch.randn(1, 2))
    first, second = torch.randn(2, 2), torch.randn(2, 2)
    fields = [_fields(f"manual_{i}") for i in range(4)]
    save = _ops_activations._save_activation_fields
    for entry, tensor in zip(fields, (first, first, second, second), strict=True):
        if direct:
            save(trace, entry, tensor, (), {}, None)
        else:
            original = _ops_activations.fresh_plain_payload_eligible
            _ops_activations.fresh_plain_payload_eligible = lambda *a: None
            try:
                save(trace, entry, tensor, (), {}, None)
            finally:
                _ops_activations.fresh_plain_payload_eligible = original
    return fields


def test_repeat_save_after_a_direct_insert_hits_the_identity_cache() -> None:
    """A tensor saved again hits the entry the direct path inserted, with the same ordinal."""

    direct, general = _save_twice(True), _save_twice(False)
    for paths in (direct, general):
        assert paths[1]["out"] is paths[0]["out"]
        assert paths[3]["out"] is paths[2]["out"]
    for a, b in zip(direct, general, strict=True):
        assert a["annotations"] == b["annotations"]
        assert {k: v for k, v in a.items() if k not in {"out", "annotations"}} == {
            k: v for k, v in b.items() if k not in {"out", "annotations"}
        }
        assert torch.equal(a["out"], b["out"])
    assert direct[1]["annotations"]["dedup_source_id"] == 1
    assert direct[3]["annotations"]["dedup_source_id"] == 2


def test_refusals_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tight absolute budget refuses at the same op, phase and figures on both paths."""

    model = _chain()
    x = torch.randn(4, 16)
    options = CaptureOptions(save_budget=700)
    fields = []
    for eligible in (_ops_fresh_payload.fresh_plain_payload_eligible, lambda *a: None):
        with monkeypatch.context() as patch:
            patch.setattr(_ops_activations, "fresh_plain_payload_eligible", eligible)
            with pytest.raises(SaveBudgetExceededError) as excinfo:
                tl.trace(model, x, capture=options)
        fields.append(dict(excinfo.value.fields))
    assert fields[0] == fields[1]
    assert fields[0]["accounting_phase"] == "pre_allocation_admission"


@pytest.mark.parametrize(
    "options",
    [
        {"capture": CaptureOptions(save_args=True)},
        {"capture": CaptureOptions(activation_transform=lambda tensor: tensor * 2)},
        {"capture": CaptureOptions(output_device="cpu")},
    ],
    ids=["save_args", "transform", "output_device"],
)
def test_other_modes_keep_the_general_path(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    """Every non-default mode named here never reaches the direct path."""

    calls = [0]

    def counting(*args: Any) -> None:
        calls[0] += 1

    monkeypatch.setattr(_ops_activations, "save_fresh_plain_activation", counting)
    tl.trace(_chain(), torch.randn(4, 16), **options)
    assert calls[0] == 0
