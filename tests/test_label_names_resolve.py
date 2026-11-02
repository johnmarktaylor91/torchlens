"""The public label property names survive the record refactor (record design: labels keep their names).

Every name below is a glossary spelling. A rename anywhere in the record fold would fail here
before it reached a user.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _labels
from torchlens._data_classes.layer import Layer
from torchlens._data_classes.module import Module
from torchlens._data_classes.op import Op
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
MODULE_LABEL_NAMES = ("label", "layer_labels", "call_labels")
TRACE_LABEL_NAMES = ("op_labels", "layer_labels")


def _trace() -> Trace:
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    return tl.trace(model, torch.randn(2, 3))


def test_op_label_names_resolve_on_every_op() -> None:
    trace = _trace()
    for op in trace.ops:
        missing = [name for name in OP_LABEL_NAMES if not hasattr(op, name)]
        assert not missing, (op.label, missing)
        assert _labels.is_pass_qualified(op.label)
        assert not _labels.is_pass_qualified(op.layer_label)


def test_layer_and_module_label_names_resolve() -> None:
    trace = _trace()
    for layer in trace.layers:
        missing = [name for name in LAYER_LABEL_NAMES if not hasattr(layer, name)]
        assert not missing, (layer.layer_label, missing)
    for module in trace.modules:
        missing = [name for name in MODULE_LABEL_NAMES if not hasattr(module, name)]
        assert not missing, (module.label, missing)
    assert isinstance(Layer, type) and isinstance(Module, type) and isinstance(Op, type)


def test_trace_label_names_and_validate_forward_pass_resolve() -> None:
    trace = _trace()
    for name in TRACE_LABEL_NAMES:
        assert hasattr(trace, name), name
    assert list(trace.op_labels) == [op.label for op in trace.ops]
    assert list(trace.layer_labels) == [layer.layer_label for layer in trace.layers]
    assert callable(Trace.validate_forward_pass)
    assert callable(tl.validation.validate_forward_pass)
