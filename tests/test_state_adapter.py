"""Tests for class-agnostic TorchLens state adapter helpers."""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from torchlens._io import FieldPolicy, TorchLensIOError
from torchlens._io.runnable import assert_sparse_core_has_no_tensor_payload
from torchlens._io.scrub import _scrub_value, _ScrubOptions
from torchlens.data_classes._state_adapter import state_items, state_new, state_restore
from torchlens.errors.runnable import SparseCorePayloadError


class _DictBackedState:
    """Small dict-backed object for adapter round-trip checks."""

    def __init__(self) -> None:
        """Populate deterministic live state."""

        self.alpha = 1
        self.beta = {"nested": [2, 3]}


class _SlottedState:
    """Small slotted object for adapter enumeration checks."""

    __slots__ = ("alpha", "beta", "unset")

    def __init__(self) -> None:
        """Populate two of three declared slots."""

        self.alpha = 1
        self.beta = 2


class _MixedState:
    """Small object with independently populated dict and slotted fields."""

    __slots__ = ("slot_value", "__dict__")

    def __init__(self, slot_value: object, dict_value: object) -> None:
        """Populate one slot and one dict-backed field.

        Parameters
        ----------
        slot_value:
            Value stored in the slotted field.
        dict_value:
            Value stored in the dict-backed field.
        """

        self.slot_value = slot_value
        self.dict_value = dict_value


class _IncompletePortableState:
    """Portable-state object with an intentionally missing field policy."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"covered": FieldPolicy.KEEP}

    def __init__(self) -> None:
        """Populate a missing field to exercise the scrub tripwire."""

        self.covered = "ok"
        self.missing = "tripwire"


def test_state_items_enumerates_every_set_dict_field() -> None:
    """The adapter enumerates all live fields on dict-backed objects."""

    obj = _DictBackedState()

    assert list(state_items(obj)) == list(vars(obj).items())


def test_state_items_enumerates_every_set_slot_field() -> None:
    """The adapter enumerates set slots and skips unset slots."""

    obj = _SlottedState()

    assert list(state_items(obj)) == [("alpha", 1), ("beta", 2)]


def test_state_items_enumerates_dict_and_slot_fields_for_mixed_object() -> None:
    """Mixed-shape objects expose every dict and slotted state field once."""

    obj = _MixedState(slot_value="slot", dict_value="dict")

    assert list(state_items(obj)) == [("dict_value", "dict"), ("slot_value", "slot")]


@pytest.mark.parametrize(
    ("slot_value", "dict_value", "expected_field"),
    (
        (torch.ones(1), None, "slot_value"),
        (None, torch.ones(1), "dict_value"),
    ),
)
def test_sparse_core_backstop_inspects_mixed_dict_and_slot_fields(
    slot_value: object,
    dict_value: object,
    expected_field: str,
) -> None:
    """Sparse-core traversal catches tensor payloads in either mixed field store."""

    with pytest.raises(SparseCorePayloadError, match=expected_field) as caught:
        assert_sparse_core_has_no_tensor_payload(_MixedState(slot_value, dict_value))
    assert caught.value.fields["code"] == "sparse_core_tensor_payload"
    # The typed tripwire stays catchable as AssertionError for historical callers.
    assert isinstance(caught.value, AssertionError)


def test_state_new_restore_round_trips_dict_backed_state() -> None:
    """Uninitialized objects can be restored from adapter-enumerated state."""

    obj = _DictBackedState()
    restored = state_restore(state_new(type(obj)), dict(state_items(obj)))

    assert type(restored) is type(obj)
    assert vars(restored) == vars(obj)
    assert restored is not obj


def test_state_new_restore_round_trips_slotted_state() -> None:
    """Uninitialized slotted objects can be restored slot by slot."""

    obj = _SlottedState()
    restored = state_restore(state_new(type(obj)), dict(state_items(obj)))

    assert type(restored) is type(obj)
    assert list(state_items(restored)) == list(state_items(obj))
    assert restored is not obj


def test_scrub_completeness_tripwire_uses_adapter_enumeration() -> None:
    """Scrub still raises when adapter-enumerated state lacks a field policy."""

    options = _ScrubOptions(
        include_outs=True,
        include_grads=True,
        include_saved_args=True,
        include_rng_states=True,
    )

    with pytest.raises(TorchLensIOError, match="missing from PORTABLE_STATE_SPEC"):
        _scrub_value(
            _IncompletePortableState(),
            options,
            memo={},
            blob_specs=[],
            blob_counter=[0],
        )


class _ColumnarBackedState:
    """Facade-shaped object whose state lives in an external column store."""

    __slots__ = ("_core", "_row_id")

    def __init__(self, core: dict[str, list[object]], row_id: int) -> None:
        """Bind one row of a struct-of-arrays core.

        Parameters
        ----------
        core:
            Column-name-to-column mapping.
        row_id:
            Row index of this facade.
        """

        self._core = core
        self._row_id = row_id

    def __tl_state_items__(self) -> list[tuple[str, object]]:
        """Materialize the full row as adapter state."""

        return [
            (column_name, column[self._row_id])
            for column_name, column in sorted(self._core.items())
        ]

    def __tl_state_restore__(self, mapping: dict[str, object]) -> None:
        """Install restored state into a fresh single-row core."""

        self._core = {name: [value] for name, value in mapping.items()}
        self._row_id = 0


def test_state_items_prefers_columnar_opt_in_hook_over_slots() -> None:
    """``__tl_state_items__`` wins over dict/slot introspection."""

    core = {"alpha": [10, 11], "beta": ["x", "y"]}
    obj = _ColumnarBackedState(core, row_id=1)

    assert list(state_items(obj)) == [("alpha", 11), ("beta", "y")]


def test_state_restore_prefers_columnar_opt_in_hook() -> None:
    """``__tl_state_restore__`` receives the mapping instead of slot writes."""

    core = {"alpha": [10, 11], "beta": ["x", "y"]}
    obj = _ColumnarBackedState(core, row_id=1)
    restored = state_restore(state_new(_ColumnarBackedState), dict(state_items(obj)))

    assert list(state_items(restored)) == [("alpha", 11), ("beta", "y")]
    assert restored is not obj


def test_field_order_state_coverage_on_live_records() -> None:
    """Coupling-A tripwire: declared fields must flow through ``state_items``.

    Save/load, pickle, scrub, and fork all enumerate object state through
    ``state_items``. A storage re-plumbing that turns a declared field into a
    computed property without opting into ``__tl_state_items__`` would
    silently drop it from every persistence path; this test fails first.
    """

    from torch import nn

    import torchlens as tl
    from torchlens import constants as tl_constants
    from torchlens.data_classes.layer import Layer
    from torchlens.data_classes.module import Module, ModuleCall
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.param import Param
    from torchlens.data_classes.trace import Trace

    class _CouplingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
            self.head = nn.Linear(2, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = torch.relu(self.conv(x))
            return self.head(y.mean(dim=(2, 3)))

    torch.manual_seed(0)
    trace = tl.trace(_CouplingModel(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))
    cases: list[tuple[object, tuple[str, ...]]] = [
        (trace, tuple(tl_constants.MODEL_LOG_FIELD_ORDER)),
        (
            next(iter(trace.ops.values())),
            tuple(tl_constants.LAYER_PASS_LOG_FIELD_ORDER),
        ),
        (
            next(iter(trace.layers.values())),
            tuple(tl_constants.LAYER_LOG_FIELD_ORDER),
        ),
        (
            next(iter(trace.modules.values())),
            tuple(tl_constants.MODULE_LOG_FIELD_ORDER),
        ),
        (
            next(iter(trace.module_calls.values())),
            tuple(tl_constants.MODULE_PASS_LOG_FIELD_ORDER),
        ),
        (
            next(iter(trace.params.values())),
            tuple(tl_constants.PARAM_LOG_FIELD_ORDER),
        ),
    ]
    assert isinstance(cases[0][0], Trace)
    assert isinstance(cases[1][0], Op)
    assert isinstance(cases[2][0], Layer)
    assert isinstance(cases[3][0], Module)
    assert isinstance(cases[4][0], ModuleCall)
    assert isinstance(cases[5][0], Param)
    for record, field_order in cases:
        state_keys = {name for name, _ in state_items(record)}
        dropped: list[str] = []
        for field_name in field_order:
            descriptor = next(
                (
                    mro_cls.__dict__[field_name]
                    for mro_cls in type(record).__mro__
                    if field_name in mro_cls.__dict__
                ),
                None,
            )
            if isinstance(descriptor, property):
                continue
            try:
                getattr(record, field_name)
            except AttributeError:
                continue
            if field_name not in state_keys:
                dropped.append(field_name)
        assert not dropped, (
            f"{type(record).__name__} declared fields invisible to "
            f"state_items (would silently drop from save/pickle/scrub): "
            f"{dropped}"
        )


def test_every_record_class_owns_its_explicit_state_protocol() -> None:
    """Core record classes never take the generic introspection fallback.

    M11 disposition of the "generic core-record state walker": the generic
    ``__dict__``/slots branch of ``state_items``/``state_restore`` stays for
    arbitrary nested values (scrub/rehydrate walk user objects), but every
    core record class must define BOTH explicit protocol hooks so the
    generic branch provably never fires for a record. A record class losing
    its hook would silently fall back to storage introspection — exactly
    the state-adapter blindness the M2 protocol closed.
    """

    from torchlens.data_classes.backward_pass import BackwardPass
    from torchlens.data_classes.buffer import Buffer
    from torchlens.data_classes.func_call_location import FuncCallLocation
    from torchlens.data_classes.grad_fn import GradFn
    from torchlens.data_classes.grad_fn_call import GradFnCall
    from torchlens.data_classes.layer import Layer
    from torchlens.data_classes.module import Module, ModuleCall
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.param import Param

    record_classes = (
        Op,
        Layer,
        Module,
        ModuleCall,
        Param,
        Buffer,
        FuncCallLocation,
        GradFn,
        GradFnCall,
        BackwardPass,
    )
    missing = [
        cls.__name__
        for cls in record_classes
        if getattr(cls, "__tl_state_items__", None) is None
        or getattr(cls, "__tl_state_restore__", None) is None
    ]
    assert not missing, (
        f"record classes without an explicit state protocol (would take the "
        f"generic walker): {missing}"
    )
