"""W15 F1 regressions: batched per-consumption state TOCTOU sampling stays coverage-identical.

The per-consumption param/buffer TOCTOU scans batch their storage-pointer reads through the
true-original C accessors (``_split_consumed_state_items``) so the armed numpy-RNG setprofile
classifier no longer pays a per-registered-tensor toll on every dispatched op. These tests pin
the coverage properties that make the batch legal:

* the scan stays a FULL per-op scan over CURRENT pointers -- a mid-forward ``p.data = other``
  rebind is still attributed at consumption (a registration-time ptr index would go stale and
  falsely VERIFY a byte-restored transient write);
* exotic tensor subclasses keep the verbatim wrapped per-item path;
* the split helper is decision-identical to the original per-item loop, raw ``0`` pointers
  (meta/storageless) included.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import completeness_witness as cw
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save_and_run(model: nn.Module, x: torch.Tensor, path: Path) -> tl.RunResult:
    """Capture, save runnable with weights, load, and run on the original input."""

    trace = tl.trace(model, x, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


class ParamRebindMutateRestore(nn.Module):
    """Rebind a param's storage mid-forward, consume it, then restore the original storage.

    ``self.w.data = evil`` swaps the parameter's underlying storage WITHOUT bumping its
    version and WITHOUT any aten dispatch, so only the per-consumption scan's CURRENT
    pointer read can attribute the consuming op to the watched param. After the byte-exact
    restore, the forward-end ``_reconcile_params`` sweep sees baseline bytes and passes --
    this test fails on any implementation that matches against registration-time pointers.
    """

    def __init__(self) -> None:
        """Initialize the module with one parameter."""

        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a value computed from a transiently rebound parameter storage.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output using the transient (rebound) parameter value.
        """

        original = self.w.data
        self.w.data = torch.tensor([12.0, 13.0])
        out = x * self.w
        self.w.data = original
        return out


class _ParamSubclass(nn.Parameter):
    """A ``nn.Parameter`` subclass: exact-class gating must route it to the verbatim path."""


class SubclassParamMutateRestore(nn.Module):
    """Transiently mutate a subclass-typed parameter consumed by a traced op."""

    def __init__(self) -> None:
        """Initialize the module with one subclass-typed parameter."""

        super().__init__()
        self.w = _ParamSubclass(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a value that depends on a transient subclass-param mutation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output using the transient parameter value.
        """

        with torch.no_grad():
            self.w.add_(10.0)
        out = x * self.w
        with torch.no_grad():
            self.w.sub_(10.0)
        return out


def test_param_storage_rebind_mutate_consume_restore_is_unverifiable(tmp_path: Path) -> None:
    """A mid-forward ``p.data`` storage rebind consumed by a traced op must fail closed."""

    capture_x = torch.tensor([2.0, 4.0])
    result = _save_and_run(ParamRebindMutateRestore(), capture_x, tmp_path / "rebind.tlspec")

    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def test_subclass_param_transient_mutation_is_unverifiable(tmp_path: Path) -> None:
    """A transient mutation of a subclass-typed param must still fail closed (verbatim path)."""

    capture_x = torch.tensor([2.0, 4.0])
    result = _save_and_run(SubclassParamMutateRestore(), capture_x, tmp_path / "subclass.tlspec")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


def _reference_matches(items: tuple[tuple[str, object], ...], consumed_ptrs: set[int]) -> list[str]:
    """Original per-item matching semantics: wrapped current-pointer membership test."""

    matched: list[str] = []
    for address, source in items:
        if not isinstance(source, torch.Tensor):
            continue
        try:
            if source.untyped_storage().data_ptr() not in consumed_ptrs:
                continue
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        matched.append(address)
    return matched


def _split_matches(items: tuple[tuple[str, object], ...], consumed_ptrs: set[int]) -> list[str]:
    """Matching decisions produced by the batched split plus its verbatim leftover path."""

    hits, leftovers = cw._split_consumed_state_items(items, consumed_ptrs)
    matched = [address for address, _ in hits]
    matched.extend(_reference_matches(tuple(leftovers), consumed_ptrs))
    return matched


def test_split_consumed_state_items_is_decision_identical() -> None:
    """The batched split matches the original loop on plain, subclass, meta, and junk items."""

    plain = torch.randn(3)
    param = nn.Parameter(torch.randn(3))
    other = torch.randn(3)
    meta = torch.empty(3, device="meta")
    sub = torch.Tensor._make_subclass(_ParamSubclass, torch.randn(3))
    items: tuple[tuple[str, object], ...] = (
        ("plain", plain),
        ("param", param),
        ("other", other),
        ("meta", meta),
        ("sub", sub),
        ("junk", "not-a-tensor"),
    )

    consumed = {
        plain.untyped_storage().data_ptr(),
        sub.untyped_storage().data_ptr(),
        0,  # meta/storageless raw pointer stays 0 and must still match, as verbatim
    }
    assert sorted(_split_matches(items, consumed)) == sorted(_reference_matches(items, consumed))
    assert set(_split_matches(items, consumed)) == {"plain", "meta", "sub"}

    consumed_none: set[int] = {other.untyped_storage().data_ptr() + 1}
    assert _split_matches(items, consumed_none) == _reference_matches(items, consumed_none)
    assert _split_matches(items, consumed_none) == []

    # Exotic classes must be routed to the verbatim leftover path, never raw-read.
    hits, leftovers = cw._split_consumed_state_items(items, consumed)
    assert {address for address, _ in leftovers} == {"sub", "junk"}
    assert {address for address, _ in hits} == {"plain", "meta"}
