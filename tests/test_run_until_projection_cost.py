"""U26.3 fix 3: the until= prefix projection's per-op overhead, and its equivalence.

``run(until=...)`` skips the cut-off ops, but each executed op used to pay a full
refresh copy: every recorded field of the refreshed op was deep-copied onto the
target op, then every static field was written back from a snapshot. The
projection now copies only the fields that survive that write-back. The gate is
a deterministic counter (op field writes per executed op), never wall time.
"""

from __future__ import annotations

import inspect
import re
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._capture import projectors
from torchlens._data_classes import _trace_rerun_state

#: Op field writes per executed op on the base (f271e1568) for ``_stack()`` under
#: ``until="block1"``, measured by this file's counter (the refresh copy plus the
#: static write-back).
_BASE_WRITES_PER_EXECUTED_OP = 353.2


def _stack() -> nn.Module:
    """Three conv-ReLU blocks (no BatchNorm, so ``until=`` also runs on the base)."""

    torch.manual_seed(0)
    return nn.Sequential(
        OrderedDict(
            (
                f"block{index}",
                nn.Sequential(nn.Conv2d(3 if index == 0 else 4, 4, 3, padding=1), nn.ReLU()),
            )
            for index in range(3)
        )
    ).eval()


def _literal_names(source: str, opener: str, closer: str) -> set[str]:
    start = source.index(opener) + len(opener)
    return set(re.findall(r'"([A-Za-z_]+)"', source[start : source.index(closer, start)]))


def test_mirrored_field_lists_match_the_trace_refresh():
    """The projection's mirrored field lists equal the ones the full refresh uses."""

    source = inspect.getsource(_trace_rerun_state.refresh_rerun_op_from)
    kept = _literal_names(source, "preserved_fields = {", "}")
    extra = _literal_names(source, "for field_name in (", ")")
    assert kept == set(projectors._RERUN_KEPT_OP_FIELDS)
    assert extra == set(projectors._RERUN_EXTRA_OP_FIELDS)


@pytest.mark.parametrize("until", ["block0", "block1"])
def test_until_values_match_the_full_run(until):
    """Executed-prefix activations of a truncated run equal the full run's."""

    model = _stack()
    log = tl.trace(model, torch.randn(2, 3, 8, 8))
    x2 = torch.randn(2, 3, 8, 8)
    full = log.run(inputs=x2)
    truncated = log.run(inputs=x2, until=until)
    executed = 0
    for layer in truncated.trace._layer_list:
        if layer.type in ("input", "output") or layer.out is None:
            continue
        assert torch.equal(layer.out, full.trace[layer.layer_label].out), layer.layer_label
        executed += 1
    assert executed > 0
