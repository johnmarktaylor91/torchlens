"""grind-r8 cluster 4.0.7: the lifetime/eviction pair (R37 + R50).

R37 (sol repro): the last-owner payload eviction replaced only DIRECT tensor
cells, so a tensor nested inside a saved-args container (``torch.stack``'s
list argument, ``input_activations`` tuples) survived Trace death through any
retained ``Op`` facade -- pinning the very payloads the F4 fix exists to
release. The eviction now walks builtin container cells (bounded depth,
cycle-safe); tensors inside non-builtin custom objects remain the disclosed
residual.

R50 (sol, verify-first): ``Trace.cleanup()`` left ``_save_budget_accountant``
alive on the husked trace (the sibling ``_predicate_lookback_candidates``
claim was verified already-clean -- the field is deleted by the existing
husk).
"""

from __future__ import annotations

import gc
import weakref

import torch
from torch import nn

import torchlens as tl


class _StackNet(nn.Module):
    """Forward whose stack call saves a CONTAINER of tensors as an argument."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([x + 1, x + 2]).sum(dim=0)


def _nested_arg_tensor_refs(trace) -> list[weakref.ref]:
    """Weakrefs to every tensor nested in retained arg containers."""

    refs: list[weakref.ref] = []

    def _collect(value) -> None:
        if isinstance(value, torch.Tensor):
            refs.append(weakref.ref(value))
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)

    for op in trace.ops.values():
        for field_name in ("saved_args", "input_activations"):
            _collect(getattr(op, field_name, None))
    return refs


def test_nested_container_payloads_evict_with_last_owner() -> None:
    """Tensors inside saved-args containers must not survive Trace death."""

    log = tl.trace(
        _StackNet(),
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(save_arg_values=True),
    )
    refs = _nested_arg_tensor_refs(log)
    assert refs, "expected retained arg containers holding tensors"
    retained_op = log["stack"]  # keep ONE facade alive past the trace
    del log
    gc.collect()
    survivors = [ref for ref in refs if ref() is not None]
    assert not survivors, (
        f"{len(survivors)}/{len(refs)} nested arg tensors survived Trace death "
        "through the retained Op facade"
    )
    # Metadata stays readable on the retained facade; evicted slots read None.
    saved = retained_op.saved_args
    assert saved is not None
    assert not any(
        isinstance(item, torch.Tensor)
        for item in (saved if isinstance(saved, (list, tuple)) else [])
    )


def test_cleanup_husks_the_save_budget_accountant() -> None:
    """cleanup() must delete the accountant with the rest of the session state."""

    log = tl.trace(nn.Linear(4, 4), torch.randn(1, 4))
    assert "_save_budget_accountant" in log.__dict__
    log.cleanup()
    assert "_save_budget_accountant" not in log.__dict__
    assert "_predicate_lookback_candidates" not in log.__dict__
