"""Debugger/IDE DX contract for record objects (god-object facade gate).

The columnar facade work must keep every declared field visible to the exact
enumeration protocol IDE debuggers use (pydevd/debugpy: ``dir(obj)`` then
``getattr``), backed by a real inspectable descriptor (slot, property, or
instance-dict entry) — never a bare ``__getattr__``. These tests freeze that
contract at the pre-columnar baseline so any storage re-plumbing that would
break IDE field preview fails here first.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import constants as tl_constants
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace


class _DxModel(nn.Module):
    """Tiny deterministic model exercising conv, relu, and linear records."""

    def __init__(self) -> None:
        """Initialize fixed-shape layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv-relu-pool-linear."""

        y = torch.relu(self.conv(x))
        return self.head(y.mean(dim=(2, 3)))


@pytest.fixture(scope="module")
def dx_trace() -> Iterator[Trace]:
    """Capture one deterministic trace for DX assertions."""

    torch.manual_seed(0)
    model = _DxModel()
    trace = tl.trace(model, torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _debugger_visible_names(obj: object) -> set[str]:
    """Return names an IDE debugger enumerates: dir() then getattr()."""

    visible: set[str] = set()
    for name in dir(obj):
        try:
            getattr(obj, name)
        except Exception:  # noqa: BLE001 - debugger swallows and moves on
            continue
        visible.add(name)
    return visible


def _has_inspectable_descriptor(obj: object, name: str) -> bool:
    """Return whether a field is backed by a real inspectable descriptor.

    A slot descriptor, ``property``, or instance-``__dict__`` entry is
    inspectable; resolution purely through ``__getattr__`` is not.
    """

    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None and name in instance_dict:
        return True
    for mro_cls in type(obj).__mro__:
        descriptor = mro_cls.__dict__.get(name)
        if descriptor is None:
            continue
        if isinstance(descriptor, property):
            return True
        if hasattr(descriptor, "__get__"):
            return True
    return False


_CASES = (
    ("trace", Trace, tuple(tl_constants.MODEL_LOG_FIELD_ORDER)),
    ("op", Op, tuple(tl_constants.LAYER_PASS_LOG_FIELD_ORDER)),
    ("layer", Layer, tuple(tl_constants.LAYER_LOG_FIELD_ORDER)),
    ("module", Module, tuple(tl_constants.MODULE_LOG_FIELD_ORDER)),
    ("module_call", ModuleCall, tuple(tl_constants.MODULE_PASS_LOG_FIELD_ORDER)),
    ("param", Param, tuple(tl_constants.PARAM_LOG_FIELD_ORDER)),
)

#: Declared schema fields that are legitimately ABSENT (AttributeError) on the
#: fixture's capture path today. Frozen exactly: a storage re-plumbing that
#: makes any other field vanish — or silently materializes one of these —
#: must update this table consciously.
_EXPECTED_ABSENT: dict[str, frozenset[str]] = {
    "trace": frozenset({"input_structure", "_containers", "_buffer_persistence"}),
    "op": frozenset(),
    "layer": frozenset(),
    "module": frozenset(),
    "module_call": frozenset(),
    "param": frozenset(),
}


def _get_instance(dx_trace: Trace, kind: str) -> object:
    """Return one live record instance of the requested kind."""

    if kind == "trace":
        return dx_trace
    if kind == "op":
        return next(iter(dx_trace.ops.values()))
    if kind == "layer":
        return next(iter(dx_trace.layers.values()))
    if kind == "module":
        return next(iter(dx_trace.modules.values()))
    if kind == "module_call":
        return next(iter(dx_trace.module_calls.values()))
    if kind == "param":
        return next(iter(dx_trace.params.values()))
    raise KeyError(kind)


@pytest.mark.smoke_cells(
    "test_all_declared_fields_are_debugger_visible[layer-Layer-field_order2]",
    "test_all_declared_fields_are_debugger_visible[module-Module-field_order3]",
)
@pytest.mark.parametrize(("kind", "cls", "field_order"), _CASES)
def test_all_declared_fields_are_debugger_visible(
    dx_trace: Trace, kind: str, cls: type, field_order: tuple[str, ...]
) -> None:
    """Every FIELD_ORDER field survives the pydevd dir-then-getattr protocol."""

    instance = _get_instance(dx_trace, kind)
    assert isinstance(instance, cls)
    visible = _debugger_visible_names(instance)
    missing = {name for name in field_order if name not in visible}
    assert missing == _EXPECTED_ABSENT[kind], (
        f"{cls.__name__} debugger-invisible field set changed: "
        f"unexpected={sorted(missing - _EXPECTED_ABSENT[kind])} "
        f"newly-visible={sorted(_EXPECTED_ABSENT[kind] - missing)}"
    )


@pytest.mark.parametrize(("kind", "cls", "field_order"), _CASES)
def test_all_declared_fields_have_inspectable_descriptors(
    dx_trace: Trace, kind: str, cls: type, field_order: tuple[str, ...]
) -> None:
    """No FIELD_ORDER field may resolve purely through ``__getattr__``."""

    instance = _get_instance(dx_trace, kind)
    uninspectable = {
        name for name in field_order if not _has_inspectable_descriptor(instance, name)
    }
    assert uninspectable == _EXPECTED_ABSENT[kind], (
        f"{cls.__name__} fields without slot/property/dict backing changed: "
        f"unexpected={sorted(uninspectable - _EXPECTED_ABSENT[kind])} "
        f"newly-backed={sorted(_EXPECTED_ABSENT[kind] - uninspectable)}"
    )


def test_dir_is_sorted_and_stable(dx_trace: Trace) -> None:
    """``dir()`` on records stays usable: no duplicates, deterministic."""

    op = next(iter(dx_trace.ops.values()))
    for target in (dx_trace, op):
        listing = dir(target)
        assert len(listing) == len(set(listing))
        assert listing == dir(target)
