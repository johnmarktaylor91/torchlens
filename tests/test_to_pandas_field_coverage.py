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
(``trace.modules.to_pandas()``), ``BackwardPass``, and ``Op``. ``Param``,
``ModuleCall``, and ``Trace`` already have equivalent coverage in
``tests/test_io_pandas.py`` / ``tests/test_record_export.py``.

``Buffer`` is ALSO covered here (cert6/cert7 MAJOR-1): ``Buffer.initial_value``
-- a real, populated field backed by a public ``@property`` and marked
``FieldPolicy.KEEP`` -- was silently absent from ``BUFFER_LOG_FIELD_ORDER``, so
it never appeared in ``Buffer.to_pandas()`` / ``BufferAccessor.to_pandas()``.
``tests/test_io_pandas.py::test_buffer_accessor_to_pandas_schema`` only checked
that the exported columns matched ``BUFFER_LOG_FIELD_ORDER`` -- a
self-consistency check that can never catch a field missing from
``BUFFER_LOG_FIELD_ORDER`` itself, since both sides of that comparison share
the same blind spot. ``Buffer`` had also never been added to
``test_field_order_has_no_keep_field_desync``'s parametrize list below (the
one check in this file that DOES compare against the ground-truth
``PORTABLE_STATE_SPEC``), so the class was never actually exercised despite the
docstring here previously claiming it already had "equivalent coverage".

``Trace`` is ALSO covered here (cert8 BLOCKER-1): ``Trace._grad_fn_param_refs``
-- a real, populated dict written directly (no ``getattr``/``setdefault``
guard) by ``backends/torch/backward.py`` -- was silently absent from both
``MODEL_LOG_FIELD_ORDER`` and (as a direct consequence) ``_MODEL_LOG_DEFAULT_FILL``,
so any ``Trace`` reconstructed from a state dict that predates the field (a
supported backward-compat path per ``read_tlspec_version``'s pre-versioning
mode) crashed with ``AttributeError`` mid-way through ``log_backward()``.
``Trace`` was never added to ``test_field_order_has_no_keep_field_desync``'s
parametrize list either, even though ``Trace.to_pandas()`` genuinely has no
per-record projection built from ``MODEL_LOG_FIELD_ORDER`` (so the to_pandas()
bypass variant of this bug class does not apply to it) -- the FIELD_ORDER
constant still drives the separate, more consequential default-fill/state-
restore surface, and that surface has the exact same "field missing from the
FIELD_ORDER constant" blind spot as to_pandas() does for the other classes
here. ``_TRACE_KEEP_FIELD_DESYNC_EXCLUSIONS`` documents the one legitimate,
non-bug exception: ``ops_with_params`` is a computed ``@property`` that is
never stored in ``self.__dict__`` (cert7/cert8 MINOR-2), so it has no state to
desync in the first place.

The underlying ``_field_policy_desync`` false-positive filter only recognized
one alias shape -- an underscore-prefixed private field backing a same-named
public property (e.g. ``_is_in_conditional_body`` / ``is_in_conditional_body``).
That is narrower than the docstring here previously claimed: it does not
recognize a KEEP field that is a real, populated value but whose FIELD_ORDER
counterpart lives at a *different* name (e.g. ``Param._grad_memory`` backs the
differently-named public ``gradient_memory`` property) or is deliberately
excluded from FIELD_ORDER by design (e.g. ``Param._derived_grad_payload``
backs the live-handle ``.grad`` property, which the metadata-only
``to_pandas()`` contract excludes everywhere). Both shapes are real,
documented exceptions, not FIELD_ORDER gaps -- but the narrow heuristic would
have reported them as gaps regardless. ``test_field_order_has_no_keep_field_desync``
is scoped to the five surfaces above plus ``Trace``; ``Param`` and
``GradFnCall`` have their own real, currently-unresolved desync findings
(cert7/cert8 MINOR-1, and the ``label``/``call_label`` policy-alias shape)
that need a dedicated alias-exclusion design before they can be added here
without either masking a genuine gap or failing on one out of scope for this
change -- tracked separately, not silently swept into this file's exclusion
set.
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
    BUFFER_LOG_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
)
from torchlens.data_classes.backward_pass import (  # noqa: E402
    BackwardPass,
    _TO_PANDAS_EXCLUDED_BACKWARD_PASS_FIELDS,
)
from torchlens.data_classes.buffer import (  # noqa: E402
    Buffer,
    _TO_PANDAS_EXCLUDED_BUFFER_FIELDS,
)
from torchlens.data_classes.layer import Layer  # noqa: E402
from torchlens.data_classes.module import Module, _TO_PANDAS_EXCLUDED_MODULE_FIELDS  # noqa: E402
from torchlens.data_classes.op import Op  # noqa: E402
from torchlens.data_classes.trace import Trace  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402


# Per-class documented exclusions for `test_field_order_has_no_keep_field_desync`.
#
# Every entry here MUST be a verified, non-bug exception -- never a mask for a
# real FIELD_ORDER gap. See the ``Trace`` paragraph in this module's docstring
# for the full audit trail (cert7/cert8 MINOR-2).
_KEEP_FIELD_DESYNC_EXCLUSIONS: dict[type[Any], frozenset[str]] = {
    Trace: frozenset(
        {
            # Computed @property (data_classes/_trace_stats.py); never stored
            # in self.__dict__, so there is no state for FIELD_ORDER to lose.
            # Its sibling `num_ops_with_params` (also computed) correctly has
            # no PORTABLE_STATE_SPEC entry at all -- this KEEP declaration is
            # confirmed-vestigial, not a live desync (cert7/cert8 MINOR-2).
            "ops_with_params",
        }
    ),
}


class _FieldCoverageModel(nn.Module):
    """Model exercising a conditional branch, a submodule, and a scalar output.

    The ``if``/``else`` on ``y.mean()`` gives at least one Layer/Op with
    ``is_in_conditional_body=True`` (the cert4 MAJOR-1 repro field), and the
    scalar sum output supports a real ``.backward()`` call for BackwardPass
    field coverage. ``offset`` is a registered buffer that is read but never
    overwritten in ``forward``, giving ``Buffer.initial_value`` (the cert6/
    cert7 MAJOR-1 repro field) a real, populated value.
    """

    def __init__(self) -> None:
        """Initialize a single linear submodule and a static buffer."""

        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer("offset", torch.tensor([0.25, -0.25]))

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

        y = self.linear(x) + self.offset
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
        Does NOT apply ``_KEEP_FIELD_DESYNC_EXCLUSIONS`` -- callers that need
        those documented, per-class exceptions filtered out must do so
        themselves (see ``test_field_order_has_no_keep_field_desync``), so
        this raw scan stays a faithful, un-narrowed ground-truth signal that
        can be inspected on its own for auditing.
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


@pytest.mark.parametrize("cls", [Layer, Module, BackwardPass, Op, Buffer, Trace])
def test_field_order_has_no_keep_field_desync(cls: type[Any]) -> None:
    """No record class should have a KEEP-policy field missing from FIELD_ORDER.

    Regression gate for the cert3/cert4/cert6/cert8 desync class: a field is
    set at construction and marked ``FieldPolicy.KEEP`` in the class's
    PORTABLE_STATE_SPEC, yet the FIELD_ORDER constant in ``constants.py``
    never picked it up, so it silently vanished from every ``to_pandas()``
    export built on that constant (e.g. ``Layer.is_in_conditional_body``,
    ``Buffer.module_address``, ``Buffer.initial_value`` -- all fixed by this
    same change). ``Buffer`` is included here specifically because it was
    the cert6/cert7 MAJOR-1 gap: the class was never added to this
    parametrize list even though its own ``PORTABLE_STATE_SPEC`` had a real
    KEEP-and-missing field (``_initial_value``) the whole time.

    ``Trace`` is included here for the cert8 BLOCKER-1 gap: unlike the other
    classes, ``Trace`` has no per-record ``to_pandas()`` built from
    ``MODEL_LOG_FIELD_ORDER`` -- but that constant also drives
    ``_MODEL_LOG_DEFAULT_FILL`` (the backward-compat default-fill table used
    by ``Trace.__setstate__``), a strictly more consequential surface than
    to_pandas(): a field missing there does not just drop a column, it leaves
    the restored ``Trace`` instance without the attribute entirely, crashing
    any code that writes to it unconditionally (``_grad_fn_param_refs`` did,
    in ``backends/torch/backward.py``). ``_KEEP_FIELD_DESYNC_EXCLUSIONS``
    filters out the one verified non-bug exception for ``Trace``
    (``ops_with_params`` -- a computed property with no stored state, see the
    module docstring); every other class defaults to an empty exclusion set,
    so this test is not weakened for them.
    """

    desynced = [
        name
        for name in _field_policy_desync(cls)
        if name not in _KEEP_FIELD_DESYNC_EXCLUSIONS.get(cls, frozenset())
    ]
    assert desynced == [], (
        f"{cls.__name__}.FIELD_POLICY marks {desynced} as FieldPolicy.KEEP but "
        f"they are absent from {cls.__name__}'s FIELD_ORDER constant in "
        "constants.py. Add them to the FIELD_ORDER list next to their sibling "
        "fields so to_pandas() picks them up."
    )


def test_buffer_to_pandas_covers_field_order(coverage_trace: Any) -> None:
    """Buffer.to_pandas() must export every non-excluded BUFFER_LOG_FIELD_ORDER field.

    Regression gate for cert6/cert7 MAJOR-1: ``initial_value`` -- a real,
    populated field backed by a public ``@property`` and marked
    ``FieldPolicy.KEEP`` -- used to be silently absent from
    ``BUFFER_LOG_FIELD_ORDER``, so it never reached ``Buffer.to_pandas()`` /
    ``BufferAccessor.to_pandas()`` even though the underlying data was real.
    """

    buffer = coverage_trace.buffers["offset"]
    df = buffer.to_pandas()
    columns = list(df.columns)

    assert len(columns) == len(set(columns))
    unaccounted = [
        field_name
        for field_name in BUFFER_LOG_FIELD_ORDER
        if field_name not in columns and field_name not in _TO_PANDAS_EXCLUDED_BUFFER_FIELDS
    ]
    assert unaccounted == []
    assert not (_TO_PANDAS_EXCLUDED_BUFFER_FIELDS & set(columns))
    # Excluded fields are real Buffer fields (no stale exclusion entries).
    assert _TO_PANDAS_EXCLUDED_BUFFER_FIELDS <= set(BUFFER_LOG_FIELD_ORDER)

    # initial_value is a real, non-None value -- not just a present-but-empty
    # column (the field the old hand-rolled/desynced FIELD_ORDER dropped).
    assert "initial_value" in columns
    assert df.loc[0, "initial_value"] is not None

    # trace.buffers.to_pandas() delegates through BufferAccessor, which must
    # inherit the same fix.
    accessor_df = coverage_trace.buffers.to_pandas()
    assert list(accessor_df.columns) == columns
    assert accessor_df.loc[0, "initial_value"] is not None


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
