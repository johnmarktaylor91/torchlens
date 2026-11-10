"""The direct path's alias and budget steps decide exactly as the general path's do.

``backends/torch/_ops_fresh_alias.py`` registers the deferred payload alias for a
fresh plain output without the checks the direct path's eligibility already
settled, and ``_ops_fresh_budget.py`` admits and reconciles its bytes without the
reservation record in the common case. These tests compare both with the
general-path functions they stand in for: the alias decision and registry entry
against ``tensor_utils._try_defer_payload_alias`` over the case each check exists
for, and the ledgers after release and the refusals against a capture with the
direct path switched off.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._save_budget import SaveBudgetExceededError
from torchlens._state import pause_logging
from torchlens._utils import tensor_utils
from torchlens.backends.torch import _ops_activations, _ops_fresh_alias
from torchlens.options import CaptureOptions

_STATE = torch.randn(4, 4)


def _cases() -> dict[str, Callable[[], torch.Tensor]]:
    """Return one tensor factory per alias check (all exact strided ``torch.Tensor``)."""

    def inference() -> torch.Tensor:
        with torch.inference_mode():
            return torch.randn(3, 4)

    return {
        "contiguous": lambda: torch.randn(3, 4),
        "transposed": lambda: torch.randn(3, 4).t(),
        "channels_last": lambda: torch.randn(2, 3, 4, 5).to(memory_format=torch.channels_last),
        "expanded_whole": lambda: torch.randn(1, 4).expand(3, 4),
        "offset_slice": lambda: torch.randn(6, 4)[2:],
        "prefix_slice": lambda: torch.randn(6, 4)[:3],
        "stepped": lambda: torch.randn(8)[::2],
        "scalar": lambda: torch.tensor(2.0),
        "empty": lambda: torch.randn(0, 4),
        "conj": lambda: torch.randn(3, dtype=torch.cfloat).conj(),
        "neg": lambda: torch.randn(3, dtype=torch.cfloat).conj().imag,
        "inference": inference,
        "state_storage": lambda: _STATE,
        "state_view": lambda: _STATE.view(16),
        "quantized": lambda: torch.quantize_per_tensor(torch.randn(4), 0.1, 0, torch.qint8),
    }


def _registry() -> dict[tuple[Any, int], tuple[tuple[int, int, str], int]]:
    """Return every live registry entry, keyed by its key and the entry's identity."""

    return {
        (key, id(entry)): (key, entry.version)
        for key, entries in tensor_utils._DEFER_PENDING.items()
        for entry in entries
        if entry.ref() is not None
    }


@pytest.mark.parametrize("graph_connected", [False, True])
@pytest.mark.parametrize("case", sorted(_cases()))
def test_alias_decision_matches_the_general_path(
    monkeypatch: pytest.MonkeyPatch, case: str, graph_connected: bool
) -> None:
    """Same alias-or-copy decision, registry entry, gradient edge and storage identity."""

    monkeypatch.setattr(
        tensor_utils, "_DEFER_STATE_PTRS", frozenset({_STATE.untyped_storage().data_ptr()})
    )
    source = _cases()[case]()
    if graph_connected and source.is_floating_point() and not source.is_inference():
        source = source.detach().requires_grad_(True)
    before = _registry()
    with pause_logging():
        general = tensor_utils._try_defer_payload_alias(source, graph_connected=graph_connected)
    middle = _registry()
    with pause_logging():
        lean = _ops_fresh_alias._defer_fresh_alias(source, graph_connected)
    after = _registry()
    general_added = [entry for key, entry in middle.items() if key not in before]
    lean_added = [entry for key, entry in after.items() if key not in middle]
    assert (lean is None) == (general is None), case
    assert lean_added == general_added
    if lean is None:
        assert lean_added == []
        return
    alias, identity = lean
    storage = alias.untyped_storage()
    assert identity == (str(alias.device), storage.data_ptr(), storage.nbytes())
    assert alias.data_ptr() == general.data_ptr() == source.data_ptr()
    assert (alias.shape, alias.stride(), alias.dtype) == (
        general.shape,
        general.stride(),
        general.dtype,
    )
    assert alias.requires_grad == general.requires_grad
    assert type(alias.grad_fn) is type(general.grad_fn)


def _chain() -> nn.Module:
    """Return a small parameterised chain."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.Tanh())


def _ledgers(budget: Any) -> dict[str, tuple[int, int, list[tuple[int, int]]]]:
    """Return each ledger's committed bytes, saved count and retained entries."""

    return {
        key: (
            ledger.committed_bytes,
            ledger.num_saved,
            sorted((e.physical_bytes, e.live_refs) for e in ledger.retained_storage.values()),
        )
        for key, ledger in budget.ledgers.items()
    }


def _capture_and_release(direct: bool, requires_grad: bool) -> tuple[Any, Any]:
    """Capture the chain; return the ledgers while the trace lives and after it is freed."""

    x = torch.randn(4, 16, requires_grad=requires_grad)
    original = _ops_activations.fresh_plain_payload_eligible
    if not direct:
        _ops_activations.fresh_plain_payload_eligible = lambda *a: None
    try:
        trace = tl.trace(_chain(), x)
    finally:
        _ops_activations.fresh_plain_payload_eligible = original
    budget = trace._save_budget_accountant
    assert budget is not None
    live = _ledgers(budget)
    del trace
    gc.collect()
    return live, _ledgers(budget)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_release_credits_match_the_general_path(requires_grad: bool) -> None:
    """The ledgers match while the trace lives and after its payloads are released."""

    direct_live, direct_freed = _capture_and_release(True, requires_grad)
    general_live, general_freed = _capture_and_release(False, requires_grad)
    assert direct_live == general_live
    assert direct_freed == general_freed
    assert any(entry[0] > 0 for entry in direct_live.values())


@pytest.mark.parametrize("budget_bytes", [100, 700, 1100])
def test_refusal_on_a_first_or_a_later_save_matches(
    monkeypatch: pytest.MonkeyPatch, budget_bytes: int
) -> None:
    """A refusal on the ledger's first save or a later one has the same fields on both paths."""

    fields = []
    for eligible in (_ops_activations.fresh_plain_payload_eligible, lambda *a: None):
        with monkeypatch.context() as patch:
            patch.setattr(_ops_activations, "fresh_plain_payload_eligible", eligible)
            with pytest.raises(SaveBudgetExceededError) as excinfo:
                tl.trace(
                    _chain(), torch.randn(4, 16), capture=CaptureOptions(save_budget=budget_bytes)
                )
        fields.append(dict(excinfo.value.fields))
    assert fields[0] == fields[1]
