"""Cheaper per-trace preparation observes exactly what the longer path observed.

Session preparation builds one parameter record per parameter, snapshots the
tensors a forward can reach, scans for compiled callables, and releases the
parameter stamps at session end, on every trace. Each of those was made
cheaper without remembering anything between traces. These tests hold the
cheaper paths to the longer ones: a parameter record written in one row equals
the record written field by field; the forward-callable sweep with its
per-sweep memo visits the tensors the sweep without it visits; the
compiled-callable scan lists what the unskipped scan lists; the single-pass
release clears every stamp; and a model changed between two traces is seen as
changed by the second.
"""

from __future__ import annotations

import functools
import sys
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture_state_helpers import bypass_compiled_plain_callables
from torchlens._data_classes._param_row import PARAM_ROW_FIELDS, init_param_row
from torchlens._data_classes.param import Param
from torchlens._ir.refs import DtypeRef
from torchlens._plain_attr_fidelity import _PLAIN_ATTR_IGNORED_NAMES
from torchlens._quantities import Bytes
from torchlens._trace_core.record_rows import CORE_KEY, ROW_KEY, bind_record_row
from torchlens._utils._tensor_label import get_param_meta, set_param_meta
from torchlens._utils._torch_compat import is_dynamo_compiled_callable
from torchlens.backends.torch._compiled_callable_sites import (
    cannot_carry_compile_marker,
    compiled_plain_callable_sites,
)
from torchlens.backends.torch._held_plain_holders import (
    DICTLESS_LEAF_TYPES,
    plain_holder_tensor_attrs,
)
from torchlens.backends.torch._held_tensor_scan import iter_module_held_plain_tensors
from torchlens.backends.torch._module_boundary_adoption import collect_pre_forward_tensor_ids
from torchlens.backends.torch._session_cleanup import (
    _clear_module_callables_tensor_metadata,
    method_index,
)
from torchlens.backends.torch._session_param_release import restore_session_param_state

# ---------------------------------------------------------------------------
# Parameter records: one row write equals the field-by-field write
# ---------------------------------------------------------------------------


def _field_by_field_param(**given: Any) -> Param:
    """Build a ``Param`` the way its constructor did before the one-row write."""

    record = Param.__new__(Param)
    address, module_address, dtype = given["address"], given["module_address"], given["dtype"]
    fields: dict[str, Any] = {
        "address": address,
        "name": given["name"],
        "shape": given["shape"],
        "dtype": dtype,
        "dtype_ref": DtypeRef.from_value(dtype),
        "device_ref": None,
        "backend_address": address,
        "resolver_status": "resolved",
        "num_params": given["num_params"],
        "param_memory": Bytes(given["param_memory"]),
        "is_trainable": given["trainable"],
        "module_address": module_address,
        "all_addresses": [address],
        "all_module_addresses": [module_address],
        "barcode": given["barcode"],
        "has_optimizer": given["has_optimizer"],
        "_param_ref": None,
        "_param_ref_released": False,
        "_source_trace_ref": None,
        "num_calls": 1,
        "used_by_ops": [],
        "used_by_layers": [],
        "co_parent_params": [],
        "_has_grad": False,
        "_grad_shape": None,
        "_grad_dtype": None,
        "_grad_memory": Bytes(0),
        "_grad_records": [],
        "_derived_grad_payload": None,
        "_derived_grad_record_path": None,
    }
    assert set(fields) == set(PARAM_ROW_FIELDS)
    for name, value in fields.items():
        setattr(record, name, value)
    return record


def _cells(record: Param) -> dict[str, Any]:
    """Return every stored cell of a record, by field name, read from its row."""

    store, row = record.__dict__[CORE_KEY], record.__dict__[ROW_KEY]
    layout = type(record)._TL_LAYOUT  # type: ignore[attr-defined]
    return {name: store.cell_get(row, layout.fid_by_name[name]) for name in layout.names}


def _same_cell(left: Any, right: Any) -> bool:
    """Whether two cells have the same type and the same value."""

    return type(left) is type(right) and (left == right or repr(left) == repr(right))


_GIVEN: dict[str, Any] = {
    "module_address": "fc",
    "name": "weight",
    "shape": (4, 4),
    "dtype": torch.float32,
    "num_params": 16,
    "param_memory": 64,
    "trainable": True,
    "address": "fc.weight",
    "barcode": "abc123",
    "has_optimizer": None,
}


@pytest.mark.parametrize(
    "changes",
    [{}, {"trainable": False, "has_optimizer": True, "dtype": torch.float64, "shape": ()}],
    ids=["plain", "frozen"],
)
def test_one_row_param_equals_the_field_by_field_param(changes: dict[str, Any]) -> None:
    """Every stored field of a constructed ``Param`` equals the field-by-field build."""

    given = {**_GIVEN, **changes}
    built, reference = _cells(Param(**given)), _cells(_field_by_field_param(**given))
    assert list(built) == list(reference)
    assert set(built) == set(PARAM_ROW_FIELDS)
    for name in reference:
        assert _same_cell(built[name], reference[name]), name


def test_one_row_params_share_no_mutable_cell() -> None:
    """Two records built from the same arguments hold separate lists."""

    first, second = Param(**_GIVEN), Param(**_GIVEN)
    for name in ("all_addresses", "all_module_addresses", "used_by_ops", "used_by_layers"):
        assert getattr(first, name) is not getattr(second, name), name
    first.used_by_ops.append("linear_1")
    assert second.used_by_ops == []


def test_row_write_on_a_bound_record_overwrites_every_field_and_checks_width() -> None:
    """A record that already owns a row is rewritten in place; a short row is refused."""

    record = Param(**_GIVEN)
    store = record.__dict__[CORE_KEY]
    dtype_ref = DtypeRef.from_value(torch.float64)
    given = ("head", "bias", (2,), torch.float64, 2, 16, False, "head.bias", "zz", True)
    init_param_row(record, dtype_ref, *given)
    with pytest.raises(ValueError):
        init_param_row(record, dtype_ref, *given[:-1])
    assert record.__dict__[CORE_KEY] is store
    reference = _cells(
        _field_by_field_param(
            module_address="head",
            name="bias",
            shape=(2,),
            dtype=torch.float64,
            num_params=2,
            param_memory=16,
            trainable=False,
            address="head.bias",
            barcode="zz",
            has_optimizer=True,
        )
    )
    rewritten = _cells(record)
    for name in reference:
        assert _same_cell(rewritten[name], reference[name]), name
    with pytest.raises(ValueError):
        bind_record_row(record, [None])
    with pytest.raises(ValueError):
        bind_record_row(Param.__new__(Param), [None])


# ---------------------------------------------------------------------------
# Parameter records and stamps follow the model between traces
# ---------------------------------------------------------------------------


class _TwoLayer(nn.Module):
    """Linear, activation, linear."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layers in order."""

        return self.head(torch.relu(self.fc(x)))


def test_parameter_changes_between_traces_reach_the_next_trace_records() -> None:
    """A value, trainability, shape or membership change is in the second trace's records."""

    torch.manual_seed(0)
    model = _TwoLayer().eval()
    x = torch.randn(3, 4)
    first = tl.trace(model, x)
    assert first.params["fc.bias"].is_trainable is True
    assert tuple(first.params["head.weight"].shape) == (2, 4)

    with torch.no_grad():
        model.fc.weight.zero_()
    model.fc.bias.requires_grad_(False)
    model.head.weight = nn.Parameter(torch.zeros(3, 4))
    model.head.bias = nn.Parameter(torch.ones(3))
    model.fc.register_parameter("extra", nn.Parameter(torch.ones(1)))
    second = tl.trace(model, x)

    assert torch.count_nonzero(second.params["fc.weight"].value) == 0
    assert second.params["fc.bias"].is_trainable is False
    assert tuple(second.params["head.weight"].shape) == (3, 4)
    assert second.params["head.weight"].num_params == 12
    assert "fc.extra" in {param.address for param in second.params}
    (output_op,) = second.output_ops
    assert torch.equal(output_op.out, model(x))
    # The first trace's records are its own: nothing of the second leaked back.
    assert first.params["fc.bias"].is_trainable is True
    assert tuple(first.params["head.weight"].shape) == (2, 4)
    assert first.params["fc.weight"] is not second.params["fc.weight"]

    del model.fc._parameters["extra"]
    third = tl.trace(model, x)
    assert "fc.extra" not in {param.address for param in third.params}
    # Session end restored every flag and left no stamp behind.
    assert model.fc.bias.requires_grad is False
    assert model.fc.weight.requires_grad is True
    assert all(get_param_meta(param) is None for param in model.parameters())


def test_single_pass_release_clears_recorded_and_live_parameters() -> None:
    """Stamps recorded at prep, stamps only on the live tree, and both, are all released."""

    model = _TwoLayer()
    recorded_and_live = model.fc.weight
    live_only = model.head.bias
    popped = nn.Parameter(torch.ones(2))  # recorded at prep, gone from the tree at session end
    for index, param in enumerate((recorded_and_live, live_only, popped)):
        set_param_meta(param, barcode=f"b{index}", address=f"p{index}", requires_grad_before=False)
        param.requires_grad_(True)
    trace = SimpleNamespace(_session_param_inventory=[recorded_and_live, popped])

    restore_session_param_state(cast(Any, trace), model)

    for param in (recorded_and_live, live_only, popped):
        assert get_param_meta(param) is None
        assert param.requires_grad is False
    assert trace._session_param_inventory == []
    # Unstamped parameters are untouched, and a second release changes nothing.
    assert model.fc.bias.requires_grad is True and model.head.weight.requires_grad is True
    restore_session_param_state(cast(Any, trace), model)
    assert recorded_and_live.requires_grad is False


# ---------------------------------------------------------------------------
# Forward-callable sweep: the per-sweep memo visits what the full sweep visits
# ---------------------------------------------------------------------------

_BASE_TENSOR = torch.zeros(1)
_SUB_TENSOR = torch.ones(1)
_WRAPPED_TENSOR = torch.full((1,), 2.0)
_LATE: torch.Tensor | None = None


class _Base(nn.Module):
    """A forward shared by subclasses, reaching a module-level tensor through a helper."""

    def helper(self, x: torch.Tensor) -> torch.Tensor:
        """Add the base tensor."""

        return x + _BASE_TENSOR

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the helper, and add the late tensor once one exists."""

        y = self.helper(x)
        return y if _LATE is None else y + _LATE


class _Sub(_Base):
    """Shares ``_Base.forward`` and overrides the helper it calls."""

    def helper(self, x: torch.Tensor) -> torch.Tensor:
        """Add the subclass tensor."""

        return x + _SUB_TENSOR


def _passthrough(func: Any) -> Any:
    """A ``functools.wraps`` decorator that changes nothing."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


class _Decorated(nn.Module):
    """A forward behind a user decorator."""

    @_passthrough
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the wrapped tensor."""

        return x + _WRAPPED_TENSOR


def _closing_class(held: torch.Tensor) -> type[nn.Module]:
    """Return a module class whose forward closes over ``held`` (same code, new function)."""

    class _Closing(nn.Module):
        """A forward that closes over one tensor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add the closed-over tensor."""

            return x + held

    return _Closing


def _shared_forward_model(sub_first: bool) -> tuple[nn.Sequential, list[torch.Tensor]]:
    """Return a model whose modules share forwards, and the tensors those forwards reach."""

    closed = [torch.full((1,), 3.0), torch.full((1,), 4.0)]
    blocks: list[nn.Module] = [_Base(), _Base(), _Sub(), _Base()]
    if sub_first:
        blocks.reverse()
    blocks += [_Decorated(), _Decorated()]
    blocks += [_closing_class(closed[0])(), _closing_class(closed[1])()]
    return nn.Sequential(*blocks), [_BASE_TENSOR, _SUB_TENSOR, _WRAPPED_TENSOR, *closed]


def _callable_sweep(model: nn.Module, memo: bool) -> set[int]:
    """Return the ids of the tensors one forward-callable sweep of ``model`` visits."""

    visited: list[int] = []
    seen: set[int] = set()
    methods = method_index(type(module) for module in model.modules())
    swept: dict[int, Any] | None = {} if memo else None
    for module in model.modules():
        _clear_module_callables_tensor_metadata(
            module, seen, visit=lambda t: visited.append(id(t)), methods=methods, swept=swept
        )
    return set(visited)


@pytest.mark.parametrize("sub_first", [False, True], ids=["base_first", "sub_first"])
def test_memoized_callable_sweep_visits_what_the_full_sweep_visits(sub_first: bool) -> None:
    """Shared forwards walked once reach every tensor the per-module walk reaches."""

    model, reachable = _shared_forward_model(sub_first)
    full = _callable_sweep(model, memo=False)
    assert _callable_sweep(model, memo=True) == full
    assert {id(tensor) for tensor in reachable} <= full
    owned, callable_only = collect_pre_forward_tensor_ids(model)
    assert {id(tensor) for tensor in reachable} <= set(owned)
    assert {id(tensor) for tensor in reachable} <= callable_only


def test_tensor_added_to_a_forward_global_between_sweeps_is_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing of one sweep carries into the next: a new global tensor is in the next snapshot."""

    model, _reachable = _shared_forward_model(sub_first=False)
    late = torch.full((1,), 5.0)
    owned_before, _ = collect_pre_forward_tensor_ids(model)
    assert id(late) not in owned_before
    monkeypatch.setattr(sys.modules[__name__], "_LATE", late)
    owned_after, callable_only = collect_pre_forward_tensor_ids(model)
    assert owned_after[id(late)] is late
    assert id(late) in callable_only
    assert _callable_sweep(model, memo=True) == _callable_sweep(model, memo=False)
    monkeypatch.setattr(sys.modules[__name__], "_LATE", None)
    owned_last, _ = collect_pre_forward_tensor_ids(model)
    assert id(late) not in owned_last


def test_module_attribute_tensor_added_between_sweeps_is_seen() -> None:
    """A tensor stored on a module between two snapshots is in the second snapshot."""

    model, _reachable = _shared_forward_model(sub_first=False)
    held = torch.full((1,), 6.0)
    owned_before, _ = collect_pre_forward_tensor_ids(model)
    assert id(held) not in owned_before
    cast(Any, model[2]).stash = [held]
    owned_after, callable_only = collect_pre_forward_tensor_ids(model)
    assert owned_after[id(held)] is held
    assert id(held) not in callable_only


# ---------------------------------------------------------------------------
# Compiled-callable scan: the type skip lists what the unskipped scan lists
# ---------------------------------------------------------------------------


def _marked_callable() -> Any:
    """Return a plain function carrying Dynamo's original-callable marker."""

    def compiled(x: torch.Tensor) -> torch.Tensor:
        return x

    compiled._torchdynamo_orig_callable = lambda x: x  # type: ignore[attr-defined]
    return compiled


def _unskipped_sites(model: nn.Module) -> tuple[str, ...]:
    """The scan with no type skip: the name, module and marker tests on every attribute."""

    sites: list[str] = []
    for module_name, module in model.named_modules():
        for attr_name, value in vars(module).items():
            if attr_name in _PLAIN_ATTR_IGNORED_NAMES or isinstance(value, nn.Module):
                continue
            if is_dynamo_compiled_callable(value):
                sites.append(f"{module_name or '<root>'}.{attr_name}")
    return tuple(dict.fromkeys(sites))


class _MarkedDict(OrderedDict[str, int]):
    """An ``OrderedDict`` subclass; never skipped on its type."""


def _attribute_zoo() -> Any:
    """Return a model whose instance dictionaries hold every skipped type and some marked values."""

    model: Any = _TwoLayer()
    marked_ordered: OrderedDict[str, int] = OrderedDict(a=1)
    marked_ordered._torchdynamo_orig_callable = len  # type: ignore[attr-defined]
    marked_subclass = _MarkedDict()
    marked_subclass._torchdynamo_orig_callable = len  # type: ignore[attr-defined]
    plain: dict[str, Any] = {
        "none": None,
        "flag": True,
        "count": 3,
        "ratio": 0.5,
        "z": 1j,
        "text": "x",
        "raw": b"x",
        "items": [1],
        "pair": (1, 2),
        "table": {"a": 1},
        "group": {1},
        "frozen": frozenset({1}),
        "ordered": OrderedDict(a=1),
        "function": len,
        "tensor": torch.ones(1),
    }
    for name, value in plain.items():
        setattr(model.fc, name, value)
    model.head.marked_ordered = marked_ordered
    model.head.marked_subclass = marked_subclass
    model.fast = _marked_callable()
    return model


def test_type_skip_never_hides_a_module_or_a_marked_value() -> None:
    """A skipped value is never a module and never carries the compile marker."""

    model = _attribute_zoo()
    skipped = 0
    for module in model.modules():
        for value in vars(module).values():
            if cannot_carry_compile_marker(value):
                skipped += 1
                assert not isinstance(value, nn.Module)
                assert not is_dynamo_compiled_callable(value)
    assert skipped > 20
    assert not cannot_carry_compile_marker(model.head.marked_ordered)
    assert not cannot_carry_compile_marker(model.head.marked_subclass)
    assert not cannot_carry_compile_marker(model.fast)
    assert cannot_carry_compile_marker(model.fc.ordered)


def test_compiled_site_scan_equals_the_unskipped_scan_and_follows_changes() -> None:
    """The scan lists what the unskipped scan lists, before and after the model changes."""

    model = _attribute_zoo()
    supported = is_dynamo_compiled_callable(model.fast)
    assert compiled_plain_callable_sites(model) == _unskipped_sites(model)
    if supported:
        assert set(compiled_plain_callable_sites(model)) == {
            "<root>.fast",
            "head.marked_ordered",
            "head.marked_subclass",
        }

    del model.fast
    del model.head.marked_ordered
    assert compiled_plain_callable_sites(model) == _unskipped_sites(model)
    assert "<root>.fast" not in compiled_plain_callable_sites(model)

    model.fc.late = _marked_callable()
    assert compiled_plain_callable_sites(model) == _unskipped_sites(model)
    if supported:
        assert "fc.late" in compiled_plain_callable_sites(model)


def test_bypass_swaps_a_marked_callable_added_between_uses_and_restores_it() -> None:
    """The bypass finds a compiled attribute added after an earlier pass, then puts it back."""

    model = _attribute_zoo()
    del model.fast
    with bypass_compiled_plain_callables(model):
        assert not hasattr(model, "fast")
    added = _marked_callable()
    model.fast = added
    if not is_dynamo_compiled_callable(added):
        pytest.skip("this torch publishes no Dynamo original-callable marker")
    with bypass_compiled_plain_callables(model):
        assert model.__dict__["fast"] is not added
        assert model.fc.__dict__["ordered"] == OrderedDict(a=1)
    assert model.__dict__["fast"] is added


# ---------------------------------------------------------------------------
# Held plain tensor scan: the type skip drops only values that hold no tensor
# ---------------------------------------------------------------------------


class _Prior:
    """A plain object holding one public tensor."""

    def __init__(self) -> None:
        """Hold the tensor."""

        self.loc = torch.ones(2)


def test_held_scan_type_skip_drops_only_values_that_hold_no_tensor() -> None:
    """Every skipped type yields nothing on the unskipped path; held tensors are still found."""

    leaves: list[Any] = [None, True, 3, 0.5, 1j, "x", b"x"]
    assert {type(leaf) for leaf in leaves} == DICTLESS_LEAF_TYPES
    for leaf in leaves:
        assert plain_holder_tensor_attrs(leaf) == []

    module: Any = nn.Linear(2, 2)
    module.scale = torch.full((2,), 2.0)
    module.count = 3
    module.label = "x"
    module.nothing = None
    module.prior = _Prior()
    module.items = [torch.zeros(1)]
    found = dict(iter_module_held_plain_tensors(module))
    assert set(found) == {"scale", "prior[loc]", "items.0"}
    assert found["scale"] is module.scale
    assert found["prior[loc]"] is module.prior.loc

    # A skipped attribute that becomes a tensor between two scans is found by the next.
    module.count = torch.ones(1)
    del module.scale
    found_after = dict(iter_module_held_plain_tensors(module))
    assert set(found_after) == {"count", "prior[loc]", "items.0"}
    assert found_after["count"] is module.count
