"""Live-record lockstep ENFORCEMENT (grind r2 row 20 / matrix R11).

The repository-policy half of this lockstep (field catalogs vs their owning
``FIELD_POLICY``, the private-ordered-field ledgers, generated artifacts vs
their generators, the persistence version pins) reads repository files and
generator output and lives in ``scripts/checks/schema_lockstep.py``. What stays
here captures a model and inspects live records:

1. **Runtime == declared.** Every attribute a live captured record actually
   carries must be declared in that record's ``FIELD_POLICY``. This derivation
   is SOURCE-FREE (no ``inspect.getsource``), so unlike the older
   ``test_internals.py::TestFieldOrderSync`` checks it cannot go red for
   environment reasons (stale bytecode, zipped/installed source, missing
   ``.py`` files) while the declarations are actually fine.
2. **Red-capability.** The checker is a plain function over its two sides, and
   ``TestMechanismIsRedCapable`` plants drift into it and proves it reports it.
   A lockstep gate nobody has proved can fail is not a gate.

Nearly everything here is smoke-tier: declaration arithmetic plus one small
capture. The module applies ``smoke`` per test rather than module-wide
(markers are additive; a module-level ``smoke`` could not be removed from a
heavy test and would trip ``tests/test_marker_lint.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.field_policy import RecordFieldPolicy
from torchlens._data_classes.op import Op
from torchlens._data_classes.trace import Trace

# ---------------------------------------------------------------------------
# Authority 4: live record attributes vs their declared policy.
# ---------------------------------------------------------------------------

#: Row-facade plumbing every kind-table-backed record carries. These are not
#: fields: they are the two-word ``(core, row)`` handle the facade reads
#: through (see ``torchlens/_trace_core/record_rows.py``).
FACADE_PLUMBING_ATTRS = frozenset({"_tl_core", "_tl_row"})


class _LockstepModel(nn.Module):
    """Small model populating params, buffers, modules, and grad_fns."""

    def __init__(self) -> None:
        """Initialize the covered layer families."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar so one backward pass covers the grad families.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar output.
        """

        return torch.relu(self.bn(self.linear(x))).sum()


@pytest.fixture(scope="module")
def lockstep_trace() -> Iterator[Trace]:
    """Yield one populated trace shared by the runtime-declaration checks.

    Yields
    ------
    Trace
        Trace with every record family populated and one backward pass.
    """

    model = _LockstepModel().eval()
    trace = tl.trace(
        model,
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(save_grads="all"),
    )
    trace.log_backward(trace[trace.output_layers[0]].out)
    try:
        yield trace
    finally:
        trace.cleanup()


def _live_records(trace: Trace) -> dict[str, Any]:
    """Return one representative live instance per record family.

    Parameters
    ----------
    trace:
        Populated trace.

    Returns
    -------
    dict[str, Any]
        Record class name -> representative instance.
    """

    grad_fn = next(record for record in trace.autograd.grad_fns if record.calls)
    return {
        "Trace": trace,
        "Op": next(iter(trace.ops)),
        "Layer": next(iter(trace.layers)),
        "Param": next(iter(trace.params)),
        "Buffer": next(iter(trace.buffers)),
        "GradFn": grad_fn,
        "GradFnCall": next(iter(grad_fn.calls.values())),
        "ModuleCall": next(iter(trace.module_calls)),
        "Module": next(iter(trace.modules)),
        "BackwardPass": next(iter(trace.autograd.backward_passes)),
    }


def all_live_records(trace: Trace) -> dict[str, list[Any]]:
    """Return EVERY live instance per record family on one trace.

    The B1-17 counterpart to :func:`_live_records`: that helper samples ONE
    representative per family via ``next(iter(...))``, which cannot see an
    attribute only some instances carry (a leak on the intervened op, the
    buffer that came from an input, the second pass of a recurrent layer).

    Parameters
    ----------
    trace:
        Populated trace.

    Returns
    -------
    dict[str, list[Any]]
        Record class name -> every live instance of that family.
    """

    families: dict[str, list[Any]] = {
        "Trace": [trace],
        "Op": list(trace.ops),
        "Layer": list(trace.layers),
        "Param": list(trace.params),
        "Buffer": list(trace.buffers),
        "ModuleCall": list(trace.module_calls),
        "Module": list(trace.modules),
    }
    grad_fns = list(trace.autograd.grad_fns)
    families["GradFn"] = grad_fns
    families["GradFnCall"] = [call for record in grad_fns for call in record.calls.values()]
    families["BackwardPass"] = list(trace.autograd.backward_passes)
    return families


def runtime_attribute_names(record: Any) -> set[str]:
    """Return every attribute a live record carries at runtime.

    ``__dict__``-backed records contribute their instance dict; columnar /
    slotted records (``Op`` carries only ``(_core, _row)`` plus generated
    field descriptors) contribute their stored-field universe through the
    ``state_items`` adapter, which is also what ``__getstate__`` persists.
    """

    from torchlens._data_classes._state_adapter import state_items

    names = set(vars(record)) if hasattr(record, "__dict__") else set()
    names |= {name for name, _value in state_items(record)}
    return names


def undeclared_runtime_attributes(
    instance_attrs: set[str],
    policy: dict[str, RecordFieldPolicy],
    allowed: frozenset[str] = FACADE_PLUMBING_ATTRS,
) -> set[str]:
    """Return attributes a live record carries but never declared.

    Parameters
    ----------
    instance_attrs:
        Attribute names present in the instance ``__dict__``.
    policy:
        The record class's declared ``FIELD_POLICY`` table.
    allowed:
        Facade plumbing that legitimately owns no declared field.

    Returns
    -------
    set[str]
        Undeclared attribute names.
    """

    return instance_attrs - set(policy) - allowed


_RECORD_NAMES = (
    "Trace",
    "Op",
    "Layer",
    "Param",
    "Buffer",
    "GradFn",
    "GradFnCall",
    "ModuleCall",
    "Module",
    "BackwardPass",
)


@pytest.mark.smoke_cells(
    "test_live_record_attributes_are_all_declared[Layer]",
    "test_live_record_attributes_are_all_declared[Param]",
)
@pytest.mark.parametrize("record_name", _RECORD_NAMES)
def test_live_record_attributes_are_all_declared(lockstep_trace: Trace, record_name: str) -> None:
    """Every attribute a captured record carries is declared in FIELD_POLICY.

    The derivation is runtime-only -- no source reading -- so this check is
    immune to the environment failure modes (stale bytecode, source-less
    installs) that make the source-introspecting field-order tests
    environment-sensitive.
    """

    record = _live_records(lockstep_trace)[record_name]
    runtime_names = runtime_attribute_names(record)
    # AUD-CODE 3.0c: the derivation must be NON-VACUOUS. ``Op`` is a columnar
    # slotted record with no ``__dict__``; the former ``vars(record)`` read
    # yielded an empty set there, so an undeclared dynamic slot could never
    # have been caught. Every record family carries fields at runtime.
    assert runtime_names, f"{record_name}: runtime attribute derivation is vacuous"
    undeclared = undeclared_runtime_attributes(runtime_names, type(record).FIELD_POLICY)
    assert not undeclared, (
        f"{record_name} carries undeclared attributes at runtime "
        f"(add them to FIELD_POLICY): {sorted(undeclared)}"
    )


def test_runtime_attribute_derivation_catches_an_undeclared_op_slot(lockstep_trace: Trace) -> None:
    """The AUD-CODE 3.0c non-vacuity pin: an undeclared Op column IS reported."""

    op = _live_records(lockstep_trace)["Op"]
    names = runtime_attribute_names(op)
    assert len(names) > 100, "Op's stored-field universe must be enumerated, not vars()"
    assert "_source_trace_ref" in names, "dynamic slots are part of the runtime universe"
    flagged = undeclared_runtime_attributes(names | {"a_new_dynamic_slot"}, Op.FIELD_POLICY)
    assert flagged == {"a_new_dynamic_slot"}


def _postprocess_axis_names() -> list[str]:
    """Return the postprocess matrix axis names, or [] when unavailable."""

    from support.postprocess_axes import iter_axes

    return [name for name, _ in iter_axes()]


@pytest.mark.smoke_cells(
    "test_live_record_attributes_are_declared_on_every_capture_axis[cooked_recording_halted]",
    "test_live_record_attributes_are_declared_on_every_capture_axis[transform]",
)
@pytest.mark.parametrize("axis_name", _postprocess_axis_names())
def test_live_record_attributes_are_declared_on_every_capture_axis(axis_name: str) -> None:
    """The runtime-declaration gate runs on EVERY capture axis (B1-17).

    The gate above samples ONE plain-capture fixture and ONE representative
    instance per family, so it was blind on two counts at once: a field only a
    non-plain capture writes, and a field only some instances of a family
    carry. Both blind spots were real -- the halted axis carried four
    undeclared Trace attrs (B1-02: two of them live user objects, one an HF
    tokenizer that plain pickle then baked into the artifact) and the
    intervened axis two more.

    The axis list is the postprocess enforcement matrix
    (``tests/support/postprocess_axes.py``), which is already the shared input
    of the declaration-seeding sweep and the read-enforcement CI leg, so a new
    capture configuration is covered here the moment it is added there. The
    honest residual is exactly the axes NOT in that matrix.
    """

    from support.postprocess_axes import iter_axes

    axis = dict(iter_axes())[axis_name]
    trace = axis()
    if trace is None:
        # Axes that clean up internally (refresh) expose no product to sweep.
        pytest.skip(f"axis {axis_name!r} returns no trace to inspect")
    try:
        offenders: dict[str, set[str]] = {}
        for family, instances in all_live_records(trace).items():
            for instance in instances:
                if not hasattr(instance, "__dict__"):
                    continue
                undeclared = undeclared_runtime_attributes(
                    set(vars(instance)),
                    type(instance).FIELD_POLICY,
                )
                if undeclared:
                    offenders.setdefault(family, set()).update(undeclared)
        assert not offenders, (
            f"capture axis {axis_name!r} carries undeclared attributes at "
            f"runtime (add them to FIELD_POLICY): "
            f"{ {family: sorted(names) for family, names in sorted(offenders.items())} }"
        )
    finally:
        trace.cleanup()


def test_the_axis_sweep_is_not_vacuous() -> None:
    """The widened gate really covers the axes that carried the leaks.

    A sweep that silently enumerated nothing would pass forever, so the axis
    list is pinned to contain the two configurations B1-02/B1-17 found leaks
    on, plus a materially larger set than the single plain fixture.
    """

    names = _postprocess_axis_names()
    assert "halted" in names
    assert "intervention" in names
    assert len(names) >= 20


def test_all_live_records_sweeps_more_than_one_instance_per_family(
    lockstep_trace: Trace,
) -> None:
    """``all_live_records`` is a real widening over ``_live_records``.

    If it collapsed to one instance per family it would re-introduce exactly
    the blind spot it exists to close.
    """

    families = all_live_records(lockstep_trace)
    assert set(families) == set(_RECORD_NAMES)
    assert any(len(instances) > 1 for instances in families.values())
    # Every representative the narrow helper picks is inside the wide sweep.
    for family, representative in _live_records(lockstep_trace).items():
        assert any(instance is representative for instance in families[family]), family


def test_facade_plumbing_allowance_stays_minimal() -> None:
    """The undeclared-attribute allowance stays the two facade handles.

    A growing allowance is how a real drift gets excused, so the allowance
    itself is pinned.
    """

    assert {"_tl_core", "_tl_row"} == FACADE_PLUMBING_ATTRS


# ---------------------------------------------------------------------------
# The mechanism must be able to go RED. The checker gets a planted drift.
# ---------------------------------------------------------------------------


class TestMechanismIsRedCapable:
    """Plant drift into the runtime-declaration checker and prove it is reported.

    Without these, a lockstep gate could be silently vacuous -- a registry that
    matches itself, a diff that compares a value with itself. Each test below
    is the negative control for one checker above.
    """

    def test_runtime_declaration_checker_detects_an_undeclared_attribute(self) -> None:
        """An attribute outside the policy and the allowance is reported."""

        undeclared = undeclared_runtime_attributes(
            {"label", "_tl_core", "smuggled_field"}, dict(Op.FIELD_POLICY)
        )
        assert undeclared == {"smuggled_field"}

    def test_runtime_declaration_checker_honors_only_the_named_allowance(self) -> None:
        """The allowance excuses exactly the facade handles, nothing more."""

        assert undeclared_runtime_attributes({"_tl_row"}, {}) == set()
        assert undeclared_runtime_attributes({"_tl_rows"}, {}) == {"_tl_rows"}


@pytest.mark.smoke
def test_schema_generator_renders_unions_identically_on_every_python() -> None:
    """Evaluated union annotations render the same text on 3.10 and 3.11+.

    Regression: the generator used ``repr``, which reads
    ``typing.Optional[typing.Any]`` on 3.10 and ``typing.Any | None`` on 3.11+,
    so the generated bindings were stale on every other interpreter.
    """

    from fractions import Fraction
    from typing import Any, Literal, Optional, Union

    from tools.generate_record_schema import _render_annotation

    assert _render_annotation(Optional[Any]) == "typing.Any | None"  # noqa: UP045
    assert _render_annotation(Union[int, None]) == "int | None"  # noqa: UP007
    assert _render_annotation(int | None) == "int | None"
    assert _render_annotation(Any | None) == "typing.Any | None"
    assert _render_annotation("Any | None") == "Any | None"
    assert _render_annotation(Fraction | None) == "fractions.Fraction | None"
    assert _render_annotation(Union[Literal["a"], str]) == "typing.Literal['a'] | str"  # noqa: UP007
    assert _render_annotation(list[int] | None) == "list[int] | None"
