"""The public label property names survive the record refactor (record design: labels keep their names).

Every name below is a glossary spelling. A rename anywhere in the record fold would fail here
before it reached a user.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _labels
from torchlens._data_classes.trace import Trace

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).

OP_LABEL_NAMES = (
    "label",
    "layer_label",
    "short_label",
    "short_layer_label",
    "raw_label",
    "fx_label",
    "grad_fn_label",
    "atomic_module_call_label",
)
LAYER_LABEL_NAMES = (
    "label",
    "layer_label",
    "short_label",
    "short_layer_label",
    "fx_label",
    "op_labels",
    "call_labels",
)
MODULE_LABEL_NAMES = ("layer_labels", "call_labels")
TRACE_LABEL_NAMES = ("op_labels", "layer_labels")


def _trace() -> Trace:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.trace(model, torch.randn(2, 3))


def test_trace_label_names_and_validate_forward_pass_resolve() -> None:
    trace = _trace()
    for name in TRACE_LABEL_NAMES:
        assert hasattr(trace, name), name
    # op_labels and layer_labels also carry the input and output boundary nodes.
    assert {op.label for op in trace.ops} <= set(trace.op_labels)
    assert all(_labels.is_pass_qualified(label) for label in trace.op_labels)
    assert {layer.layer_label for layer in trace.layers} <= set(trace.layer_labels)
    assert not any(_labels.is_pass_qualified(label) for label in trace.layer_labels)
    assert callable(Trace.validate_forward_pass)
    assert callable(tl.validation.validate_forward_pass)
