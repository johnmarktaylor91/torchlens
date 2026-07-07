"""Anti-recurrence schema-coverage tests for the FIELD_ORDER / to_pandas() desync class.

Two independent bug classes were found and fixed in the field-order-sync sprint
(cert3/cert4/cert5), and both must be gated so neither can silently reappear:

1. **FIELD_ORDER incompleteness** -- a field is genuinely set at construction and
   marked ``FieldPolicy.KEEP`` in a record class's own ``PORTABLE_STATE_SPEC``,
   yet the class's public ``*_FIELD_ORDER`` constant in ``constants.py`` never
   picked it up (e.g. ``Layer.is_in_conditional_body``, ``Buffer.module_address``
   before this fix). ``build_record_field_policy_table`` marks any such field
   ``user_facing=False`` even though its portable policy is ``KEEP`` -- that
   combination is the precise signature of this bug class.
2. **to_pandas() bypass** -- the ``*_FIELD_ORDER`` constant is complete, but the
   ``to_pandas()`` method hand-rolls a small hardcoded column subset instead of
   deriving columns from the constant (e.g. ``LayerAccessor``/``Module``/
   ``ModuleAccessor``/``BackwardPass`` before this fix), silently dropping
   populated fields from the natural bulk-export entry point.

This file covers the six record surfaces from the cert4 MAJOR-3 finding that
did NOT already have an equivalent regression test: ``Layer``, ``LayerAccessor``
(``trace.layers.to_pandas()``), ``Module``, ``ModuleAccessor``
(``trace.modules.to_pandas()``), ``BackwardPass``, and ``Op``. ``Buffer``,
``Param``, ``ModuleCall``, and ``Trace`` already have equivalent coverage in
``tests/test_io_pandas.py`` / ``tests/test_record_export.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

from torchlens import trace as trace_fn  # noqa: E402
from torchlens._io import FieldPolicy  # noqa: E402
from torchlens.constants import (  # noqa: E402
    BACKWARD_PASS_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
)
from torchlens.data_classes.backward_pass import (  # noqa: E402
    BackwardPass,
    _TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS,
)
from torchlens.data_classes.layer import Layer  # noqa: E402
from torchlens.data_classes.module import Module, _TO_PANDAS_EXCLUDED_MODULE_FIELDS  # noqa: E402
from torchlens.data_classes.op import Op  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402


class _FieldCoverageModel(nn.Module):
    """Model exercising a conditional branch, a submodule, and a scalar output.

    The ``if``/``else`` on ``y.mean()`` gives at least one Layer/Op with
    ``is_in_conditional_body=True`` (the cert4 MAJOR-1 repro field), and the
    scalar sum output supports a real ``.backward()`` call for BackwardPass
    field coverage.
    """

    def __init__(self) -> None:
        """Initialize a single linear submodule."""

        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a conditional branch on the linear output, then reduce to scalar.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar loss-like output.
        """

        y = self.linear(x)
        if y.mean() > 0:
            y = torch.relu(y)
        else:
            y = torch.sigmoid(y)
        return y.sum()


@pytest.fixture
def coverage_trace() -> Any:
    """Return a trace with a conditional branch, module calls, and a backward pass."""

    torch.manual_seed(0)
    model = _FieldCoverageModel()
    x = torch.ones(2, 2, requires_grad=True)  # forces mean() > 0 -> the relu (then) arm
    trace = trace_fn(
        model,
        x,
        capture=CaptureOptions(save_code_context=True),
        save_grads="all",
    )
    trace.log_backward(trace[trace.output_layers[0]].out)
    try:
        yield trace
    finally:
        trace.cleanup()


def _field_policy_desync(cls: type[Any]) -> list[str]:
    """Return FIELD_POLICY entries marked KEEP but absent from FIELD_ORDER.

    Parameters
    ----------
    cls:
        Record class exposing a ``FIELD_POLICY`` table built by
        ``build_record_field_policy_table``.

    Returns
    -------
    list[str]
        Field names with ``portable_policy is FieldPolicy.KEEP`` but
        ``user_facing is False`` -- i.e. genuinely portable fields the class's
        FIELD_ORDER constant never picked up. Excludes underscore-prefixed
        private backing fields for a public ``@property`` of the same name
        that IS already a user-facing FIELD_ORDER entry (e.g.
        ``_is_in_conditional_body`` backing the public
        ``is_in_conditional_body`` property) -- a known false-positive
        category, not a real desync, since the public name already covers it.
    """

    table = cls.FIELD_POLICY
    desynced = []
    for name, policy in table.items():
        if policy.user_facing or policy.portable_policy is not FieldPolicy.KEEP:
            continue
        public_name = name.lstrip("_")
        if public_name != name and public_name in table and table[public_name].user_facing:
            continue
        desynced.append(name)
    return desynced


@pytest.mark.parametrize("cls", [Layer, Module, BackwardPass, Op])
def test_field_order_has_no_keep_field_desync(cls: type[Any]) -> None:
    """No record class should have a KEEP-policy field missing from FIELD_ORDER.

    Regression gate for the cert3/cert4 desync class: a field is set at
    construction and marked ``FieldPolicy.KEEP`` in the class's
    PORTABLE_STATE_SPEC, yet the FIELD_ORDER constant in ``constants.py``
    never picked it up, so it silently vanished from every ``to_pandas()``
    export built on that constant (e.g. ``Layer.is_in_conditional_body``,
    ``Buffer.module_address`` -- fixed by this same change).
    """

    desynced = _field_policy_desync(cls)
    assert desynced == [], (
        f"{cls.__name__}.FIELD_POLICY marks {desynced} as FieldPolicy.KEEP but "
        f"they are absent from {cls.__name__}'s FIELD_ORDER constant in "
        "constants.py. Add them to the FIELD_ORDER list next to their sibling "
        "fields so to_pandas() picks them up."
    )


def test_layer_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """Layer.to_pandas() must export every LAYER_LOG_FIELD_ORDER field."""

    conditional_layer = next(
        layer for layer in coverage_trace.layers if layer.is_in_conditional_body
    )
    df = conditional_layer.to_pandas()

    assert list(df.columns) == LAYER_LOG_FIELD_ORDER
    assert len(df.columns) == len(set(df.columns))
    assert bool(df.loc[0, "is_in_conditional_body"]) is True


def test_layer_accessor_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """``trace.layers.to_pandas()`` must export every LAYER_LOG_FIELD_ORDER field.

    Regression gate for cert4 MAJOR-3.1: this accessor used to hand-roll a
    12-field subset instead of delegating to ``Layer.to_pandas()`` /
    ``LAYER_LOG_FIELD_ORDER``.
    """

    df = coverage_trace.layers.to_pandas()

    assert list(df.columns) == LAYER_LOG_FIELD_ORDER
    assert len(df.columns) == len(set(df.columns))
    assert len(df) == len(coverage_trace.layers)
    assert bool(df["is_in_conditional_body"].any())


def test_module_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """``Module.to_pandas()`` (this module's layers) must cover LAYER_LOG_FIELD_ORDER.

    Regression gate for cert4 MAJOR-3.2: this method used to hand-roll a
    6-field subset instead of delegating to each layer's
    ``LAYER_LOG_FIELD_ORDER``.
    """

    module = coverage_trace.modules["linear"]
    df = module.to_pandas()

    assert set(LAYER_LOG_FIELD_ORDER) <= set(df.columns)
    assert len(df.columns) == len(set(df.columns))
    assert len(df) == len(module.layer_labels)


def test_module_accessor_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """``trace.modules.to_pandas()`` must cover every non-excluded MODULE_LOG_FIELD_ORDER field.

    Regression gate for cert4 MAJOR-3.3: this accessor used to hand-roll a
    7-field subset instead of deriving columns from ``MODULE_LOG_FIELD_ORDER``.
    """

    df = coverage_trace.modules.to_pandas()
    columns = list(df.columns)

    assert len(columns) == len(set(columns))
    unaccounted = [
        field_name
        for field_name in MODULE_LOG_FIELD_ORDER
        if field_name not in columns and field_name not in _TO_PANDAS_EXCLUDED_MODULE_FIELDS
    ]
    assert unaccounted == []
    assert not (_TO_PANDAS_EXCLUDED_MODULE_FIELDS & set(columns))
    # Excluded fields are real Module fields (no stale exclusion entries).
    assert _TO_PANDAS_EXCLUDED_MODULE_FIELDS <= set(MODULE_LOG_FIELD_ORDER)
    assert len(df) == len(coverage_trace.modules)


def test_backward_pass_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """``BackwardPass.to_pandas()`` must cover every non-excluded BACKWARD_PASS_FIELD_ORDER field.

    Regression gate for cert4 MAJOR-3.4: this method used to hand-roll a
    12-field subset that silently dropped ``root_grad_fn_ids``/``root_meta``/
    ``root_grad_arguments``/``inputs_subset``/``engine_flags``/
    ``save_grads_policy``.
    """

    bp = coverage_trace.backward_passes[0]
    df = bp.to_pandas()
    columns = list(df.columns)

    assert len(columns) == len(set(columns))
    unaccounted = [
        field_name
        for field_name in BACKWARD_PASS_FIELD_ORDER
        if field_name not in columns and field_name not in _TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS
    ]
    assert unaccounted == []
    assert not (_TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS & set(columns))
    assert _TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS <= set(BACKWARD_PASS_FIELD_ORDER)

    # Fields the old hand-rolled subset silently dropped are now real, present,
    # non-default values -- not just present-but-empty columns.
    assert df.loc[0, "root_grad_fn_ids"]
    assert isinstance(df.loc[0, "engine_flags"], dict)
    assert df.loc[0, "save_grads_policy"] is not None

    # trace.backward_passes.to_pandas() delegates through the base Accessor,
    # which concats each item's to_pandas() -- so it inherits the same fix.
    accessor_df = coverage_trace.backward_passes.to_pandas()
    assert list(accessor_df.columns) == columns


def test_op_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """Op.to_pandas() must export every LAYER_PASS_LOG_FIELD_ORDER field."""

    conditional_op = next(op for op in coverage_trace.ops if op.is_in_conditional_body)
    df = conditional_op.to_pandas()

    assert list(df.columns) == LAYER_PASS_LOG_FIELD_ORDER
    assert len(df.columns) == len(set(df.columns))
    assert bool(df.loc[0, "is_in_conditional_body"]) is True
