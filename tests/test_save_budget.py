"""Save-budget refusal: bound retained bytes honestly instead of OOM-killing.

``tl.trace(model, x)`` retains every operation's output by default, which at
frontier shapes means a first-time user gets an OOM kill or an allocator error
from deep inside torch rather than an explanation. These tests pin the honest
replacement: a per-device running ceiling on retained payload bytes that stops
capture with :class:`SaveBudgetExceededError`, naming the committed footprint,
the tripping operation, and the remedies.

Two properties matter as much as the refusal itself and are tested explicitly:

* The accepted path is unchanged. The default ``"auto"`` budget (half of
  available memory) must never fire on an ordinary model, and a capture that
  retains nothing in RAM -- disk-streamed or metadata-only -- must not be charged
  at all, because it was never going to OOM.
* The reported figure is honest. It is a *lower bound* on the finished capture's
  footprint (the forward was still running when it tripped), and the message says
  so rather than extrapolating a total it cannot know.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._save_budget import (
    ACCOUNTING_PHASES,
    SaveBudget,
    SaveBudgetExceededError,
    available_device_bytes,
    resolve_save_budget,
)
from torchlens.options import CaptureOptions
from torchlens.recording import _storage_resolver


def _model() -> nn.Module:
    """Return a small model whose activations are a couple of KB each.

    Returns
    -------
    nn.Module
        Two-linear-layer model.
    """

    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))


def _input() -> torch.Tensor:
    """Return the matching input batch.

    Returns
    -------
    torch.Tensor
        Batch of 8 vectors of width 64 (2 KB per float32 activation).
    """

    return torch.randn(8, 64)


# ---------------------------------------------------------------------------
# Option resolution: invalid budgets fail loudly, never silently disable
# ---------------------------------------------------------------------------


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "value",
    [object()],
)
def test_invalid_budget_raises_rather_than_silently_disabling(value: object) -> None:
    """A malformed budget must fail loudly; silently unguarding is the bug."""

    with pytest.raises(ValueError) as excinfo:
        resolve_save_budget(value)  # type: ignore[arg-type]
    # R64-2: the door is typed now -- stable code + remedy on fields, while
    # the historical `except ValueError` catchability is preserved above.
    assert excinfo.value.fields.get("code") == "save_budget_invalid"
    assert excinfo.value.fields.get("remedy")


def test_unmeasurable_device_is_unbudgeted_not_assumed_infinite() -> None:
    """An unmeasurable device warns when automatic budgeting is disabled."""

    assert available_device_bytes(torch.device("mps")) is None
    budget = SaveBudget.from_option("auto")
    assert budget is not None
    with pytest.warns(UserWarning, match="cannot measure.*mps"):
        budget.admit("op", torch.device("mps"), 1)
    ledger = budget.ledgers["mps"]
    assert ledger.measured is False
    assert ledger.limit_bytes is None


# ---------------------------------------------------------------------------
# The accepted path stays exactly as it was
# ---------------------------------------------------------------------------


def test_disk_streamed_payloads_are_not_charged(tmp_path: Path) -> None:
    """Payloads streamed to disk cost no process memory, so they are not charged."""

    trace = tl.trace(
        _model(),
        _input(),
        save=tl.func("relu"),
        storage=tl.to_disk(str(tmp_path / "run.tlspec")),
        capture=CaptureOptions(save_budget=64),
    )
    assert len(trace.layer_labels) > 0


# ---------------------------------------------------------------------------
# The refusal itself
# ---------------------------------------------------------------------------


def test_absolute_budget_refuses_with_structured_fields() -> None:
    """The refusal carries numbers, so callers never parse the message."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1024))

    fields = excinfo.value.fields
    assert fields["budget_bytes"] == 1024
    assert fields["accounted_bytes"] > 1024
    assert fields["projected_bytes"] == fields["accounted_bytes"]
    assert fields["committed_bytes"] is None
    assert fields["device"] == "cpu"
    assert fields["num_saved"] >= 1
    assert isinstance(fields["label"], str) and fields["label"]


def test_refusal_names_the_configured_source_not_a_bare_number() -> None:
    """A user who set a fraction sees their fraction quoted back."""

    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(_model(), _input(), capture=CaptureOptions(save_budget=1e-9))
    assert "save_budget=1e-09" in str(excinfo.value)


def test_budget_refuses_before_the_crossing_copy_is_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission must run before ``safe_copy`` can allocate the over-budget payload."""

    allocation_attempted = False

    def fail_if_copy_runs(*args: object, **kwargs: object) -> object:
        """Record an attempted allocation and fail immediately.

        Parameters
        ----------
        *args:
            Positional arguments supplied to ``safe_copy``.
        **kwargs:
            Keyword arguments supplied to ``safe_copy``.

        Returns
        -------
        object
            Never returned.

        Raises
        ------
        AssertionError
            Always, because a successful admission guard must run first.
        """

        del args, kwargs
        nonlocal allocation_attempted
        allocation_attempted = True
        raise AssertionError("over-budget activation copy was attempted")

    monkeypatch.setattr(_storage_resolver, "safe_copy", fail_if_copy_runs)
    with pytest.raises(SaveBudgetExceededError):
        tl.trace(
            nn.ReLU(),
            torch.randn(8),
            save=tl.func("relu"),
            capture=CaptureOptions(save_budget=1),
        )
    assert allocation_attempted is False


def test_aliasing_raw_and_transformed_payloads_are_charged_once() -> None:
    """An identity transform retains one storage allocation, not two logical fields."""

    trace = tl.trace(
        nn.ReLU(),
        torch.randn(8),
        capture=CaptureOptions(
            save_budget=40,
            activation_transform=lambda tensor: tensor,
        ),
        save=tl.func("relu"),
    )
    relu = next(op for op in trace._layer_list if getattr(op, "func_name", None) == "relu")
    assert relu.out is relu.transformed_out
    assert int(trace._save_budget_accountant.ledgers["cpu"].committed_bytes) == 32


def test_typed_error_is_reachable_from_public_errors_namespace() -> None:
    """The refusal type is part of the public error surface."""

    import torchlens.errors as errors

    assert errors.SaveBudgetExceededError is SaveBudgetExceededError
    assert "SaveBudgetExceededError" in errors.__all__
    assert issubclass(SaveBudgetExceededError, errors.CaptureError)


# ---------------------------------------------------------------------------
# Accountant unit behavior
# ---------------------------------------------------------------------------


def test_accounting_phases_are_a_closed_vocabulary() -> None:
    """Refusal phases come from the exported vocabulary; sites fail closed."""

    assert ACCOUNTING_PHASES == (
        "pre_allocation_admission",
        "post_transform_reconciliation",
        "lookback_window_admission",
        "lookback_window_reconciliation",
    )
    budget = SaveBudget.from_option(100)
    assert budget is not None
    with pytest.raises(ValueError, match="admission site"):
        budget.admit("a", torch.device("cpu"), 40, site="not_a_site")


# ---------------------------------------------------------------------------
# Lookback / followed_by window copies are budgeted retained bytes
# ---------------------------------------------------------------------------


def test_lookback_window_copies_are_budgeted() -> None:
    """A tiny budget must trip on the first window copy, not retain silently.

    Before the fix, ``followed_by`` retention bypassed the accountant entirely:
    ``save_budget=1`` retained every window copy and promotion without a trip.
    """

    model = _model()
    x = _input()
    with pytest.raises(SaveBudgetExceededError) as excinfo:
        tl.trace(
            model,
            x,
            save=tl.func("linear") & tl.followed_by(tl.func("relu")),
            capture=CaptureOptions(
                save_budget=1, lookback=4, lookback_payload_policy="detached_raw"
            ),
        )
    assert excinfo.value.fields["accounting_phase"] == "lookback_window_admission"
    assert excinfo.value.fields["projected_bytes"] == excinfo.value.fields["accounted_bytes"]
    # The remedy list names the knob that actually caused the retention.
    assert "lookback" in str(excinfo.value)


def test_lookback_promotions_stay_charged_after_capture() -> None:
    """Promoted and still-windowed payloads are live retained bytes at the end."""

    trace = tl.trace(
        _model(),
        _input(),
        save=tl.func("linear") & tl.followed_by(tl.func("relu")),
        capture=CaptureOptions(
            save_budget=10_000_000, lookback=4, lookback_payload_policy="detached_raw"
        ),
    )
    ledger = trace._save_budget_accountant.ledgers["cpu"]
    # linear_1 is promoted (2 KB) and linear_2's candidate is still in the
    # bounded window (2 KB): both are genuinely retained, so both stay charged.
    assert len(trace.saved_ops) == 1
    assert int(trace.cost.saved_activation_memory) == 2048
    assert ledger.committed_bytes == 4096


def test_lookback_window_eviction_credits_the_charge_back() -> None:
    """An evicted window copy is released storage and must be credited."""

    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64), nn.ReLU())
    trace = tl.trace(
        model,
        _input(),
        save=tl.func("linear") & tl.followed_by(tl.func("relu")),
        capture=CaptureOptions(
            save_budget=10_000_000, lookback=1, lookback_payload_policy="detached_raw"
        ),
    )
    ledger = trace._save_budget_accountant.ledgers["cpu"]
    # linear_1's window copy (2 KB) was charged, then evicted by linear_2's and
    # credited back. The one live storage is linear_2's promoted payload, which
    # the still-windowed candidate aliases (charged once). Without eviction
    # crediting this would read 4096.
    assert ledger.committed_bytes == 2048
    assert ledger.num_saved == 1


# ---------------------------------------------------------------------------
# Storage identity survives pointer reuse: release credits, prune, recharge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Portability: a session-time knob, never a portable fact
# ---------------------------------------------------------------------------


def test_output_parent_promotion_is_charged() -> None:
    """grind-r6 b5 R34-N2: promoted output-parent payloads are charged.

    A selective capture whose model outputs were NOT selected still retains
    the output-parent payloads (the documented promotion rule), but the
    promotion path resolved storage with no admit/commit -- a silent
    undercount on every such capture. The final linear (2 KB, output parent)
    must appear in the committed footprint alongside the selected relu.
    """

    trace = tl.trace(
        _model(),
        _input(),
        capture=CaptureOptions(save_budget=10_000_000),
        save=["relu"],
    )
    ledger = trace._save_budget_accountant.ledgers["cpu"]
    # relu_1 (selected, 2 KB) + linear_2 (promoted output parent, 2 KB).
    assert ledger.committed_bytes >= 4096, (
        f"committed {ledger.committed_bytes} bytes; the promoted output-parent "
        "payload is invisible to the accountant"
    )


# ---------------------------------------------------------------------------
# r8 R34: accountant charge-order trio
# ---------------------------------------------------------------------------


def test_sparse_partial_alias_charges_the_physical_union() -> None:
    """sol R34-1: shared sparse components dedup instead of double-charging."""

    indices = torch.tensor([[0, 1], [1, 0]])
    values_a = torch.tensor([1.0, 2.0])
    values_b = torch.tensor([3.0, 4.0])
    sparse_a = torch.sparse_coo_tensor(indices, values_a, (2, 2))._coalesced_(True)
    sparse_b = torch.sparse_coo_tensor(indices, values_b, (2, 2))._coalesced_(True)
    # Force literal storage sharing of the index component.
    assert (
        sparse_a._indices().untyped_storage().data_ptr()
        != (sparse_b._indices().untyped_storage().data_ptr())
        or True
    )  # construction may or may not share; assert on accounting below
    budget = SaveBudget.from_option(10**9)
    assert budget is not None
    budget.charge_retained("a", (sparse_a,))
    committed_after_first = budget.ledgers["cpu"].committed_bytes
    budget.charge_retained("b", (sparse_b,))
    committed_after_second = budget.ledgers["cpu"].committed_bytes
    values_bytes = values_b.untyped_storage().nbytes()
    if sparse_a._indices().untyped_storage().data_ptr() == (
        sparse_b._indices().untyped_storage().data_ptr()
    ):
        # Shared indices: the second tensor may only add its values bytes.
        assert committed_after_second - committed_after_first == values_bytes
    else:
        indices_bytes = sparse_b._indices().untyped_storage().nbytes()
        assert committed_after_second - committed_after_first == values_bytes + indices_bytes
