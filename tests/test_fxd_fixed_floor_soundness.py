"""Soundness of the FXD fixed-floor cuts: each cut gives the old result after the state it reads changes.

Cuts covered:

* the module-namespace container descent decides its inert-type and class tests once
  per distinct item type (``_session_cleanup._clear_container_tree_tensor_metadata``),
  and the namespace scan reads names and values as two flat snapshots;
* the capture door's state copies use ``replace_fields`` (``dataclasses.replace``
  without the per-call field scan);
* the legacy constructor install resolves its roster from precomputed rows and skips
  live patches inline;
* the per-backend seeding takes already-imported accelerator modules from
  ``sys.modules``.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from collections import OrderedDict, deque
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._utils import _torch_seed_backends as seed_backends
from torchlens._utils._dataclass_replace import replace_fields
from torchlens._utils._tensor_label import TorchLensMeta
from torchlens.backends.torch import _session_cleanup as sc, legacy_ctors

_CONTAINERS = (dict, list, tuple, set, frozenset, deque)


# --- container descent ---------------------------------------------------------------


def _reference_descent(value: Any, seen: set[int], depth: int, visit: Any) -> None:
    """The container descent with the per-item filter, as it was before the cut."""

    if type(value).__name__ == "_reduce_op" and sc._is_isinstance_hostile_deprecation_shim(value):
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, nn.Parameter):
            visit(value)
        return
    if not isinstance(value, _CONTAINERS):
        return
    if id(value) in seen or depth >= 12:
        return
    seen.add(id(value))
    items = (
        [item for pair in value.items() for item in pair]
        if isinstance(value, dict)
        else list(value)
    )
    for item in items:
        item_type = type(item)
        if item_type in sc._NAMESPACE_INERT_TYPES or issubclass(item_type, type):
            continue
        _reference_descent(item, seen, depth + 1, visit)


class _StrSub(str):
    """A ``str`` subclass: not an exact inert type, so the descent must visit it."""


class _Holder:
    """A plain object: the descent never enters it."""

    def __init__(self) -> None:
        self.t = torch.ones(1)


def _tree() -> Any:
    """A container tree mixing every kind of member the descent meets."""

    shared = torch.zeros(2)
    inner = [torch.ones(1), "s", 3, (torch.ones(2), shared), {"k": torch.ones(3)}]
    return {
        "names": ["a", "b", "c"] * 50,
        "mixed": [1, 2.0, "x", _StrSub("y"), int, torch.float32, None, shared, inner],
        "nested": deque([inner, (inner, [shared]), frozenset({1, 2})]),
        "param": [nn.Parameter(torch.ones(1)), torch.ones(4)],
        "objects": [_Holder(), types.SimpleNamespace(t=torch.ones(1))],
        torch.ones(5): "tensor key",
        "ordered": OrderedDict(a=torch.ones(6), b=[torch.ones(7)]),
        "classes": {str, int, nn.Linear},
        "empty": [],
        "deep": [[[[[[[[[[[[[torch.ones(8)]]]]]]]]]]]]],
    }


@pytest.mark.parametrize("depth", [0, 5, 11, 12])
def test_container_descent_visits_what_the_per_item_filter_visits(depth: int) -> None:
    """The cut descent visits the same tensors, in the same order, from any depth."""

    tree = _tree()
    expected: list[Any] = []
    _reference_descent(tree, set(), depth, expected.append)
    got: list[Any] = []
    sc._clear_container_tree_tensor_metadata(tree, set(), depth, got.append)
    assert [id(t) for t in got] == [id(t) for t in expected]


def test_container_descent_sees_a_list_member_changed_between_walks() -> None:
    """A container that gains, loses or swaps a member is walked as it is now (no memo)."""

    names = ["a"] * 20
    for mutate in (
        lambda: names.append(torch.ones(1)),
        lambda: names.__setitem__(3, torch.ones(2)),
        lambda: names.__setitem__(3, "b"),
        lambda: names.pop(),
        lambda: names.insert(0, [torch.ones(3)]),
    ):
        mutate()
        expected: list[Any] = []
        _reference_descent(names, set(), 0, expected.append)
        got: list[Any] = []
        sc._clear_container_tree_tensor_metadata(names, set(), 0, got.append)
        assert [id(t) for t in got] == [id(t) for t in expected]


def test_container_descent_shares_the_seen_set_with_the_reference() -> None:
    """Objects already seen are skipped exactly as before, and the seen set ends equal."""

    tree = _tree()
    seen_ref = {id(tree["nested"])}
    seen_cut = set(seen_ref)
    expected: list[Any] = []
    got: list[Any] = []
    _reference_descent(tree, seen_ref, 0, expected.append)
    sc._clear_container_tree_tensor_metadata(tree, seen_cut, 0, got.append)
    assert [id(t) for t in got] == [id(t) for t in expected]
    assert seen_cut == seen_ref


# --- module namespace scan -----------------------------------------------------------


def _reference_namespace(value: types.ModuleType, seen: set[int], depth: int, visit: Any) -> None:
    """The module-namespace scan reading one ``(name, value)`` pair per item, as before."""

    if id(value) in seen or depth >= 12:
        return
    seen.add(id(value))
    items = tuple(vars(value).items())
    kinds = {type(item) for _, item in items} - sc._NAMESPACE_INERT_TYPES
    live = {kind for kind in kinds if not issubclass(kind, type)}
    for name, item in items:
        if type(item) not in live or sc._is_isinstance_hostile_deprecation_shim(item):
            continue
        if not isinstance(item, (torch.Tensor, *_CONTAINERS)):
            continue
        item = vars(value).get(name)
        if isinstance(item, torch.Tensor) and not isinstance(item, nn.Parameter):
            visit(item)
        elif isinstance(item, _CONTAINERS):
            sc._clear_container_tree_tensor_metadata(item, seen, depth + 1, visit)


def _namespace_visits(walk: Any, module: types.ModuleType, depth: int = 0) -> tuple[Any, ...]:
    """Return the tensor ids ``walk`` visits in ``module`` and the seen set it leaves."""

    seen: set[int] = set()
    got: list[Any] = []
    if walk is _reference_namespace:
        walk(module, seen, depth, got.append)
    else:
        walk(module, seen, depth, got.append, got.append)
    return [id(t) for t in got], seen


def test_namespace_scan_matches_the_paired_scan_after_every_kind_of_change() -> None:
    """Replacing a value at constant size, adding and deleting names: same visits, same order."""

    mod = types.ModuleType("fxd_namespace_probe")
    mod.stash = 5
    mod.names = ["a"] * 30
    mod.cls, mod.fn, mod.param = nn.Linear, len, nn.Parameter(torch.ones(1))
    mod.sub, mod.obj = _StrSub("s"), _Holder()
    walk = sc._clear_module_namespace_tensor_metadata
    for mutate in (
        lambda: None,
        lambda: setattr(mod, "stash", torch.ones(1)),  # constant size, value replaced
        lambda: setattr(mod, "stash", [torch.ones(2)]),
        lambda: setattr(mod, "late", (torch.ones(3), {"k": torch.ones(4)})),
        lambda: delattr(mod, "names"),
        lambda: setattr(mod, "stash", 7),
    ):
        mutate()
        assert _namespace_visits(walk, mod) == _namespace_visits(_reference_namespace, mod)
    assert _namespace_visits(walk, mod, 12) == _namespace_visits(_reference_namespace, mod, 12)
    empty = types.ModuleType("fxd_empty_probe")
    empty.__dict__.clear()
    assert _namespace_visits(walk, empty) == ([], {id(empty)})


def test_namespace_scan_matches_the_paired_scan_on_real_modules() -> None:
    """``torch`` and ``math``: the namespaces a forward really names."""

    import math

    walk = sc._clear_module_namespace_tensor_metadata
    for module in (torch, math, sc):
        assert _namespace_visits(walk, module) == _namespace_visits(_reference_namespace, module)


_STASH_MODULE = types.ModuleType("fxd_stash_module")
_STASH_MODULE.cache = ["warm"] * 10
sys.modules[_STASH_MODULE.__name__] = _STASH_MODULE


class _StashesInModuleList(nn.Module):
    """A forward that appends an intermediate to a list held in a module namespace."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.lin(x))
        fxd_stash_module.cache.append(y * 2)  # type: ignore[name-defined]  # noqa: F821
        return y + 1


_StashesInModuleList.forward.__globals__["fxd_stash_module"] = _STASH_MODULE


def _stamped(t: torch.Tensor) -> bool:
    return isinstance(getattr(t, "tl_meta", None), TorchLensMeta) and bool(
        getattr(t.tl_meta, "tensor_label_raw", None)
    )


def test_tensor_appended_to_a_module_list_is_cleared_across_repeated_captures() -> None:
    """Each capture's stash, appended to a namespace list of strings, ends unstamped."""

    model = _StashesInModuleList()
    x = torch.randn(2, 3)
    labels = None
    for _ in range(3):
        trace = tl.trace(model, x)
        got = list(trace.layer_labels)
        assert labels is None or got == labels
        labels = got
        trace.cleanup()
        stashes = [v for v in _STASH_MODULE.cache if isinstance(v, torch.Tensor)]
        assert stashes and not any(_stamped(t) for t in stashes)


def test_capture_after_a_raising_capture_with_a_namespace_stash_is_clean() -> None:
    """A capture that raised mid-forward leaves no stamp, and the next one gives the same labels."""

    model = _StashesInModuleList()
    x = torch.randn(2, 3)
    reference = tl.trace(model, x)
    labels = list(reference.layer_labels)
    reference.cleanup()

    class _Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.inner(x)
            raise RuntimeError("boom")

    with pytest.raises(Exception):  # noqa: B017 - torchlens may wrap the forward's error
        tl.trace(_Boom(), x)
    assert not any(_stamped(v) for v in _STASH_MODULE.cache if isinstance(v, torch.Tensor))
    again = tl.trace(model, x)
    assert list(again.layer_labels) == labels
    again.cleanup()


# --- replace_fields ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenSlots:
    a: int
    b: Any = None
    c: tuple[int, ...] = ()
    d: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Plain:
    a: int
    b: Any = None
    calls: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.calls = [*self.calls, "post_init"]


@dataclasses.dataclass
class _WithInitVar:
    a: int
    scale: dataclasses.InitVar[int] = 1

    def __post_init__(self, scale: int) -> None:
        self.a = self.a * scale


@dataclasses.dataclass
class _WithNoInit:
    a: int
    derived: int = dataclasses.field(init=False, default=0)


@dataclasses.dataclass
class _Single:
    a: int


def _same(x: Any, y: Any) -> None:
    assert type(x) is type(y)
    assert dataclasses.asdict(x) == dataclasses.asdict(y)
    for f in dataclasses.fields(x):
        assert getattr(x, f.name) is getattr(y, f.name) or getattr(x, f.name) == getattr(y, f.name)


@pytest.mark.parametrize(
    ("obj", "changes"),
    [
        (_FrozenSlots(1, "x", (1,), [2]), {"b": "y"}),
        (_FrozenSlots(1), {}),
        (_FrozenSlots(1), {"a": 2, "b": 3, "c": (4,), "d": [5]}),
        (_Plain(1, "x"), {"a": 7}),
        (_Single(1), {"a": 2}),
        (_Single(1), {}),
        (_WithInitVar(2, 3), {"scale": 5}),
        (_WithNoInit(1), {"a": 4}),
    ],
)
def test_replace_fields_equals_dataclasses_replace(obj: Any, changes: dict[str, Any]) -> None:
    """Every field, the ``__post_init__`` effect and the field objects match ``replace``."""

    _same(replace_fields(obj, **changes), dataclasses.replace(obj, **changes))


@pytest.mark.parametrize(
    ("obj", "changes"),
    [
        (_FrozenSlots(1), {"zz": 1}),
        (_WithNoInit(1), {"derived": 3}),
        (_WithInitVar(1), {"zz": 1}),
    ],
)
def test_replace_fields_raises_what_replace_raises(obj: Any, changes: dict[str, Any]) -> None:
    """A bad call raises the same exception type as ``dataclasses.replace``."""

    with pytest.raises(Exception) as expected:
        dataclasses.replace(obj, **changes)
    with pytest.raises(expected.type):
        replace_fields(obj, **changes)


def test_replace_fields_shares_unchanged_values_by_identity() -> None:
    """Unchanged field values are the same objects, as ``replace`` passes them."""

    payload = [1, 2]
    copy = replace_fields(_FrozenSlots(1, payload), a=2)
    assert copy.b is payload and copy.a == 2


def test_replace_fields_follows_a_replaced_field_table() -> None:
    """A class whose field table object changes is re-planned from the new table."""

    @dataclasses.dataclass
    class Local:
        a: int
        b: int = 0

    replace_fields(Local(1), b=2)
    old_table = Local.__dataclass_fields__
    Local.__dataclass_fields__ = dict(old_table)  # type: ignore[misc]
    try:
        _same(replace_fields(Local(1, 2), a=3), dataclasses.replace(Local(1, 2), a=3))
    finally:
        Local.__dataclass_fields__ = old_table  # type: ignore[misc]


def test_replace_fields_on_an_unset_slot_falls_back_to_replace() -> None:
    """An instance with an unset slot gets ``replace``'s own result or error."""

    obj = object.__new__(_FrozenSlots)
    object.__setattr__(obj, "a", 1)
    with pytest.raises(AttributeError):
        dataclasses.replace(obj)
    with pytest.raises(AttributeError):
        replace_fields(obj)
    full: dict[str, Any] = {"a": 1, "b": 2, "c": (), "d": []}
    _same(replace_fields(obj, **full), dataclasses.replace(obj, **full))


def test_door_state_copies_match_replace() -> None:
    """The door's own state class copies equal ``dataclasses.replace`` field by field."""

    from torchlens._capture.prepare.request import _DoorState

    state = object.__new__(_DoorState)
    values = {f.name: object() for f in dataclasses.fields(_DoorState)}
    for name, value in values.items():
        object.__setattr__(state, name, value)
    copy = replace_fields(state, log_name="x", cache_key="k")
    ref = dataclasses.replace(state, log_name="x", cache_key="k")
    for f in dataclasses.fields(_DoorState):
        assert getattr(copy, f.name) is getattr(ref, f.name)


def test_repeated_captures_with_changing_options_match_fresh_captures() -> None:
    """Door state built through the copies gives the same trace as before, option by option."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(3, 4)
    for kwargs in ({}, {"save": "all"}, {"save": None}, {}):
        a = tl.trace(model, x, **kwargs)
        b = tl.trace(model, x, **kwargs)
        assert list(a.layer_labels) == list(b.layer_labels)
        a.cleanup()
        b.cleanup()


# --- legacy constructor install ------------------------------------------------------


def _reference_roster() -> list[tuple[str, type]]:
    """Resolve the roster row by row, as it was before the cut."""

    from torchlens._legacy_ctor_sites import LEGACY_TENSOR_CONSTRUCTOR_SITES
    from torchlens._utils import _torch_compat

    resolved = []
    for namespace_name, class_name in LEGACY_TENSOR_CONSTRUCTOR_SITES:
        namespace = _torch_compat.get_optional_torch_namespace(namespace_name)
        cls = legacy_ctors._class_in(namespace, class_name)
        if cls is not None:
            resolved.append((legacy_ctors._legacy_func_name(namespace_name, class_name), cls))
    return resolved


def test_roster_rows_are_grouped_in_roster_order() -> None:
    """The precomputed grouping walks the roster in its own order."""

    assert legacy_ctors._ROSTER_ROWS_GROUPED_IN_ORDER


def test_resolved_roster_equals_row_by_row_resolution() -> None:
    """Names and classes, by identity, equal the row-by-row resolution."""

    got = legacy_ctors._resolve_roster()
    expected = _reference_roster()
    assert [n for n, _ in got] == [n for n, _ in expected]
    assert all(a is b for (_, a), (_, b) in zip(got, expected, strict=True))


def test_resolved_roster_sees_a_rebound_or_removed_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """A class rebound or deleted in its namespace is resolved as it is now."""

    class Stand(torch.Tensor):
        pass

    monkeypatch.setattr(torch, "FloatTensor", Stand)
    got = dict(legacy_ctors._resolve_roster())
    assert got["FloatTensor"] is Stand
    assert dict(_reference_roster())["FloatTensor"] is Stand
    monkeypatch.setattr(torch, "FloatTensor", 3)
    got_names = [n for n, _ in legacy_ctors._resolve_roster()]
    assert "FloatTensor" not in got_names
    assert got_names == [n for n, _ in _reference_roster()]


class _LegacyCtorModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.FloatTensor(2).fill_(1.0).sum()


def _legacy_logged(trace: Any) -> bool:
    return "FloatTensor" in [op.func_name for op in trace._layer_list]


def test_legacy_constructor_is_logged_after_unwrap_and_rewrap() -> None:
    """Unwrapping torch between captures restores the class; the next capture re-patches it."""

    from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch

    x = torch.ones(2)
    first = tl.trace(_LegacyCtorModel(), x)
    assert _legacy_logged(first)
    first.cleanup()
    unwrap_torch()
    assert not legacy_ctors.installed_legacy_constructor_classes()
    assert "__new__" not in torch.FloatTensor.__dict__ or not any(
        torch.FloatTensor.__dict__["__new__"] is r.patch.installed_new
        for r in legacy_ctors._INSTALLED.values()
    )
    torch.FloatTensor(1)  # the eager constructor still works unwrapped
    wrap_torch()
    second = tl.trace(_LegacyCtorModel(), x)
    assert _legacy_logged(second)
    assert "FloatTensor" in legacy_ctors.installed_legacy_constructor_classes()
    second.cleanup()


def test_install_is_idempotent_and_repairs_a_removed_patch() -> None:
    """A repeated install changes nothing; one after a patch was undone re-installs it."""

    legacy_ctors.install_legacy_constructor_wrappers()
    before = {k: r.patch.installed_new for k, r in legacy_ctors._INSTALLED.items()}
    legacy_ctors.install_legacy_constructor_wrappers()
    after = {k: r.patch.installed_new for k, r in legacy_ctors._INSTALLED.items()}
    assert before == after and before
    legacy_ctors.uninstall_legacy_constructor_wrappers()
    assert not legacy_ctors._INSTALLED
    legacy_ctors.install_legacy_constructor_wrappers()
    assert set(legacy_ctors._INSTALLED) == set(before)
    trace = tl.trace(_LegacyCtorModel(), torch.ones(2))
    assert _legacy_logged(trace)
    trace.cleanup()


# --- seeding ------------------------------------------------------------------------


def test_loaded_module_matches_import_module_for_loaded_modules() -> None:
    """An imported module is returned as ``import_module`` returns it."""

    for name in ("torch.cuda", "torch.mps", "json"):
        importlib.import_module(name)
        assert seed_backends._loaded_module(name) is importlib.import_module(name)


def test_loaded_module_defers_to_import_module_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing, ``None`` or still-initializing entry is left to ``import_module``."""

    assert seed_backends._loaded_module("fxd_no_such_module") is None
    monkeypatch.setitem(sys.modules, "fxd_none_entry", None)
    assert seed_backends._loaded_module("fxd_none_entry") is None
    half = types.ModuleType("fxd_half")
    half.__spec__ = importlib.machinery.ModuleSpec("fxd_half", None)
    half.__spec__._initializing = True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fxd_half", half)
    assert seed_backends._loaded_module("fxd_half") is None
    half.__spec__._initializing = False  # type: ignore[attr-defined]
    assert seed_backends._loaded_module("fxd_half") is half


def test_seeding_matches_torch_manual_seed() -> None:
    """The per-backend seeding leaves the CPU generator exactly as ``torch.manual_seed`` does."""

    for seed in (0, 7, 2**31 - 1):
        torch.manual_seed(seed)
        expected = torch.random.get_rng_state()
        torch.manual_seed(seed + 1)
        seed_backends.manual_seed_compiled_backends(seed)
        assert torch.equal(torch.random.get_rng_state(), expected)


def test_seeded_captures_repeat_after_an_accelerator_module_is_reloaded_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing an accelerator's ``sys.modules`` entry between seeds re-imports it."""

    name = "torch.mps"
    module = importlib.import_module(name)
    # The re-import rebinds ``torch.mps`` too; restore both so later tests see one module.
    monkeypatch.setattr(torch, "mps", module)
    monkeypatch.delitem(sys.modules, name)
    torch.manual_seed(3)
    expected = torch.random.get_rng_state()
    seed_backends.manual_seed_compiled_backends(3)
    assert torch.equal(torch.random.get_rng_state(), expected)
    assert sys.modules.get(name) is not None
