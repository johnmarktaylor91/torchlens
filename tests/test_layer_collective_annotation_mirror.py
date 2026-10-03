"""Layer.annotations: independent per-layer dict + reserved collective mirror.

Regression pins reconciling two requirements that briefly conflicted:

- 1a255943 ("stop Layer.annotations aliasing ops[0]'s user dict"): the M8a
  mirror made ``layer.annotations`` ALIAS the first pass op's dict, leaking
  op annotations into the layer (and layer writes into the op) and
  corrupting bundle round-trips. The layer must own an independent dict.
- R18-7: a collective boundary layer must still SURFACE its first pass's
  portable ``collective_boundary_v1`` payload through
  ``layer.annotations["collective"]``.

The reconciliation seeds the layer's OWN dict with a deep copy of ONLY the
reserved ``"collective"`` key at construction; every other op annotation key
(user keys, ``save_mode``/``saved_out_version``) stays op-only.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.layer import Layer


class _Tiny(nn.Module):
    """Minimal model with one relu site."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def _collective_payload() -> dict:
    """A plain-data stand-in for a collective_boundary_v1 payload."""

    return {
        "schema": "collective_boundary_v1",
        "kind": "all_reduce",
        "correlation": {
            "membership_digest": "digest",
            "lifetime_ordinal": 0,
            "channel": "coll",
            "seq": 0,
        },
    }


def test_layer_annotations_do_not_alias_op_annotations() -> None:
    """The layer dict is independent: writes never cross either boundary."""

    log = tl.trace(_Tiny(), torch.randn(2, 4))
    op = [o for o in log.ops if o.type == "relu"][0]
    layer = log[op.layer_label]

    assert layer.annotations is not op.annotations
    op.annotations["op_user_key"] = 1
    assert "op_user_key" not in layer.annotations
    layer.annotations["layer_user_key"] = 2
    assert "layer_user_key" not in op.annotations


def test_reserved_collective_key_mirrors_into_independent_layer_dict() -> None:
    """A first-pass collective payload seeds the layer's own dict by value."""

    log = tl.trace(_Tiny(), torch.randn(2, 4))
    op = [o for o in log.ops if o.type == "relu"][0]
    payload = _collective_payload()
    op.annotations = {"collective": payload, "user_key": "op_only"}

    layer = Layer(first_pass=op)

    # The reserved namespace mirrors by VALUE...
    assert layer.annotations["collective"] == payload
    # ...into an independent dict holding an independent deep copy.
    assert layer.annotations is not op.annotations
    assert layer.annotations["collective"] is not payload
    layer.annotations["collective"]["correlation"]["seq"] = 99
    assert op.annotations["collective"]["correlation"]["seq"] == 0
    # No other op annotation key leaks into the layer.
    assert "user_key" not in layer.annotations


def test_layer_without_collective_payload_starts_empty() -> None:
    """Plain layers keep the dict-era fresh empty annotations dict."""

    log = tl.trace(_Tiny(), torch.randn(2, 4))
    op = [o for o in log.ops if o.type == "relu"][0]
    layer = log[op.layer_label]
    assert layer.annotations == {}
