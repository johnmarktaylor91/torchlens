"""Raw hooks returning ALREADY-TRACED tensors must not bless functionless ops.

grind-p3-fixplan T3 (fix-patching lane, tripwire-adjacent): when a raw
``register_forward_hook`` returned a traced tensor other than the module's own
output, capture stamped the PRODUCING op ``intervention_replaced=True`` (a
correct intervened-capture disclosure -- the runnable-save refusal keys on it)
AND minted a trace-level replacement-event ledger entry for that op's label.
The ledger entry is exactly what corroborates the FUNCTIONLESS
``intervention_replacement`` validation exemption, so any functionless op a
hook happened to return -- including a lost-func plain-capture gap -- was
blessed instead of failing, defeating the locked 2026-06-02 rule that a
functionless op appearing where a real function should be must STILL fail.

The mint is now confined to genuine opaque replacements (fresh untraced
tensors synthesized as boundary ops); a traced-tensor return keeps only the
per-op disclosure stamp.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation.invariants import (
    MetadataInvariantError,
    check_metadata_invariants,
)


class TwoLinear(nn.Module):
    """Two-linear model whose second module can be raw-hooked."""

    def __init__(self) -> None:
        """Initialize the two linear maps."""

        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1 then fc2."""

        return self.fc2(self.fc1(x))


def _traced_return_capture() -> tl.Trace:
    """Capture with a raw hook substituting fc2's output with fc1's output."""

    model = TwoLinear()
    handle = model.fc2.register_forward_hook(lambda _module, args, _out: args[0])
    try:
        return tl.trace(
            model,
            torch.randn(2, 4),
            capture=CaptureOptions(layers_to_save="all", save_arg_values=True),
        )
    finally:
        handle.remove()


def test_traced_return_keeps_disclosure_stamp_and_validates() -> None:
    """The honest capture keeps the intervened disclosure and still validates."""

    trace = _traced_return_capture()
    producing = next(op for op in trace.layer_list if op.func_name == "linear")

    # The capture IS intervened: the durable per-op disclosure must survive
    # (the runnable-save user_intervention_not_replayable refusal keys on it).
    assert producing.intervention_replaced is True
    # The op still carries its real replayable function and full metadata.
    assert callable(producing.func)
    check_metadata_invariants(trace)


def test_traced_return_does_not_bless_functionless_op() -> None:
    """A functionless op whose only credential is 'a hook returned it' FAILS.

    Pre-fix, the traced-return path minted a causally-bound replacement-event
    ledger entry for the producing op's label, so forging the lost-func gap on
    that op (func nulled, placeholder func_name) passed every invariant -- the
    ledger blessed it. The mint is gone; the forged gap must fail closed.
    """

    trace = _traced_return_capture()
    producing = next(op for op in trace.layer_list if op.func_name == "linear")
    assert producing.intervention_replaced is True

    producing.func = None
    producing.func_name = "intervention_replacement"

    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(trace)
