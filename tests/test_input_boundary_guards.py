"""r-b4 R27-1: capture-entry input walkers are depth-bounded and cycle-guarded.

The four input-boundary walkers (``walk_input_boundary``, ``snapshot_input_boundary``,
``backends.default_specs._simple_leaves``, ``utils.arg_handling.copy_arg_tree``) share
ONE nesting ceiling (``INPUT_TREE_MAX_DEPTH``) and refuse deep or self-referential
input trees TYPED (``input_tree_depth_exceeded`` / ``input_tree_cycle``) instead of
dying in a raw ``RecursionError`` (probe: ~350 user levels crossed the interpreter
limit; a self-referential list crashed every walker without a cycle guard).

DAG-shaped (shared, acyclic) inputs remain fully walked: the cycle guard is
path-scoped, never global, because every occurrence of a shared container must be
witnessed under its own path.
"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import typing
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._input_walk import (
    INPUT_TREE_MAX_DEPTH,
    raw_mapping_key_component,
    snapshot_input_boundary,
    walk_input_boundary,
)
from torchlens.backends.default_specs import _simple_leaves
from torchlens.utils.arg_handling import copy_arg_tree


def _deep_list(depth: int, leaf: object) -> object:
    """Build one ``depth``-level nested list around ``leaf``."""

    value = leaf
    for _ in range(depth):
        value = [value]
    return value


def _cyclic_list() -> list[object]:
    """Build one self-referential list holding a tensor leaf."""

    value: list[object] = [torch.ones(1)]
    value.append(value)
    return value


def test_trace_refuses_overdeep_input_typed() -> None:
    """A too-deep input tree refuses typed at capture entry, never RecursionError."""

    deep = _deep_list(INPUT_TREE_MAX_DEPTH + 50, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(nn.Identity(), deep)
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"


def test_trace_refuses_cyclic_input_typed() -> None:
    """A self-referential input container refuses typed at capture entry."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(nn.Identity(), _cyclic_list())
    assert excinfo.value.fields["code"] == "input_tree_cycle"


@pytest.mark.smoke
def test_trace_recovers_and_accepts_shared_substructure_after_refusal() -> None:
    """State recovers after a refusal, and DAG-shaped inputs still trace."""

    with pytest.raises(InvalidArgumentError):
        tl.trace(nn.Identity(), _cyclic_list())
    shared = [torch.ones(1), torch.ones(1)]
    log = tl.trace(nn.Sequential(nn.Identity()), [shared, shared])
    assert len(log) > 0


def test_walk_input_boundary_depth_and_cycle_ceiling_the_subtree() -> None:
    """The normative walker ceilings over-deep and cyclic subtrees, never crashes.

    Post d95ca11f (input-walk union) the WALKER routes a depth/cycle violation
    to ``on_opaque_key_subtree`` and skips the subtree — the typed refusal
    guarantee lives at capture entry (``test_trace_refuses_*_typed`` above) and
    in the snapshot refusals ledger, not here.
    """

    deep = _deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1))
    deep_opaque: list[tuple[Any, ...]] = []
    walk_input_boundary(
        deep,
        key_component=raw_mapping_key_component,
        on_opaque_key_subtree=lambda _child, path: deep_opaque.append(path),
    )
    assert len(deep_opaque) == 1, "over-deep subtree must ceiling exactly once"

    cyclic_opaque: list[tuple[Any, ...]] = []
    walk_input_boundary(
        _cyclic_list(),
        key_component=raw_mapping_key_component,
        on_opaque_key_subtree=lambda _child, path: cyclic_opaque.append(path),
    )
    assert len(cyclic_opaque) == 1, "cyclic subtree must ceiling exactly once"


def test_walk_input_boundary_walks_every_shared_occurrence() -> None:
    """The cycle guard is PATH-scoped: a shared container is walked per occurrence."""

    leaf = torch.ones(1)
    shared = [leaf]
    seen: list[tuple[object, ...]] = []
    walk_input_boundary(
        [shared, shared],
        key_component=raw_mapping_key_component,
        on_tensor=lambda _tensor, path: seen.append(path),
    )
    assert seen == [(0, 0), (1, 0)]


def test_snapshot_input_boundary_is_total_and_refuses_in_ledger() -> None:
    """The runnable structure snapshot stays TOTAL: violations join the refusals."""

    cyclic_snapshot = snapshot_input_boundary(_cyclic_list())
    assert "input_container_cycle" in {r["reason"] for r in cyclic_snapshot["refusals"]}

    deep_snapshot = snapshot_input_boundary(_deep_list(INPUT_TREE_MAX_DEPTH + 10, 1))
    assert "input_container_too_deep" in {r["reason"] for r in deep_snapshot["refusals"]}


def test_snapshot_input_boundary_clean_input_has_no_guard_refusals() -> None:
    """Ordinary nested inputs snapshot with zero guard refusals (no false refusal)."""

    snapshot = snapshot_input_boundary({"a": [torch.ones(1), {"b": (1, 2.5)}]})
    reasons = {r["reason"] for r in snapshot["refusals"]}
    assert "input_tree_cycle" not in reasons
    assert "input_tree_depth_exceeded" not in reasons


def test_simple_leaves_depth_and_cycle_refuse_typed() -> None:
    """Backend-resolution leaf sniffing shares the same typed guards."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        _simple_leaves(_deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1)))
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"

    with pytest.raises(InvalidArgumentError) as excinfo:
        _simple_leaves(_cyclic_list())
    assert excinfo.value.fields["code"] == "input_tree_cycle"

    shared = [torch.ones(1)]
    assert len(_simple_leaves([shared, shared])) == 2


def test_copy_arg_tree_depth_refuses_typed_and_cycles_still_reproduce() -> None:
    """The canonical input copier is depth-bounded; cycle reproduction is unchanged."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        copy_arg_tree(_deep_list(INPUT_TREE_MAX_DEPTH + 10, torch.ones(1)))
    assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"

    cyclic = _cyclic_list()
    copied = copy_arg_tree(cyclic)
    assert copied[1] is copied  # the cycle is reproduced in the copy
    assert torch.equal(copied[0], cyclic[0])


def test_moderate_nesting_still_traces() -> None:
    """Inputs well under the ceiling keep working end to end."""

    log = tl.trace(nn.Identity(), _deep_list(30, torch.ones(1)))
    assert len(log) > 0


def test_copy_arg_tree_dag_is_linear_and_preserves_aliasing() -> None:
    """r-b4 R29-3: a DAG-shaped input copies O(nodes), not O(paths).

    The historical path-scoped memo copied a shared sub-container once per PATH
    (x2 per shared-substructure level; depth 25 hung capture entry ~4 minutes).
    The call-scoped memo copies it once and PRESERVES the aliasing topology the
    model itself would have seen.
    """

    import time

    node: object = [torch.ones(1)]
    for _ in range(60):  # 2**60 paths under the old memo: only a linear memo finishes
        node = [node, node]
    start = time.perf_counter()
    copied = copy_arg_tree(node)
    assert time.perf_counter() - start < 5.0
    assert copied[0] is copied[1]  # shared substructure stays aliased in the copy
    assert copied[0] is not node[0]  # ...but is a genuine copy


def test_copy_arg_tree_distinct_containers_stay_distinct() -> None:
    """Equal-valued but DISTINCT containers still copy to distinct objects."""

    left = [torch.ones(1)]
    right = [torch.ones(1)]
    copied = copy_arg_tree([left, right])
    assert copied[0] is not copied[1]


# --- grind-p3 T11.4: one ceiling, and stack exhaustion refuses typed ------------------


def _tight_stack(fn: Any, headroom: int = 130) -> Any:
    """Run ``fn`` with just enough recursion headroom that a ~100-level walk dies.

    Parameters
    ----------
    fn:
        Zero-argument callable to run.
    headroom:
        Python frames granted above the CURRENT stack depth: enough for the
        guard machinery, far less than the walkers' ~2-3 frames per level over
        100 levels.
    """

    import sys

    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(depth + headroom)
    try:
        return fn()
    finally:
        sys.setrecursionlimit(old_limit)


def test_input_search_ceiling_locksteps_the_shared_ceiling() -> None:
    """The tensor-extraction walker's ceiling covers the boundary contract.

    ``INPUT_SEARCH_DEPTH_LIMIT`` was a private 64 while the boundary ceiling
    is 200, so a legal depth-65..200 input passed every walker and then
    silently dropped its tensor leaves into a traversal gap.
    """

    from torchlens.utils.introspection import INPUT_SEARCH_DEPTH_LIMIT

    assert INPUT_SEARCH_DEPTH_LIMIT > INPUT_TREE_MAX_DEPTH


def test_input_search_walker_reaches_ceiling_depth_tensor_leaves() -> None:
    """A tensor leaf at legal depth (>64, <=ceiling) is enumerated, gap-free."""

    from torchlens.utils.introspection import (
        INPUT_SEARCH_DEPTH_LIMIT,
        get_vars_of_type_from_obj,
    )

    for depth in (100, INPUT_TREE_MAX_DEPTH):
        leaf = torch.ones(1)
        gaps: list[str] = []
        found = get_vars_of_type_from_obj(
            _deep_list(depth, leaf),
            torch.Tensor,
            search_depth=INPUT_SEARCH_DEPTH_LIMIT,
            depth_exceeded_paths=gaps,
        )
        assert any(item is leaf for item in found), f"leaf at depth {depth} dropped"
        assert gaps == [], f"legal depth {depth} recorded a traversal gap"


def test_walk_input_boundary_stack_exhaustion_refuses_typed() -> None:
    """A LEGAL 100-deep input under a consumed stack refuses typed, never raw."""

    deep = _deep_list(100, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _tight_stack(
            lambda: walk_input_boundary(
                deep, key_component=raw_mapping_key_component, on_tensor=lambda t, p: None
            )
        )
    assert excinfo.value.fields["code"] == "input_tree_stack_exhausted"


def test_snapshot_input_boundary_stack_exhaustion_refuses_typed() -> None:
    """The snapshot spine shares the typed stack-budget refusal."""

    deep = _deep_list(100, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _tight_stack(lambda: snapshot_input_boundary(deep))
    assert excinfo.value.fields["code"] == "input_tree_stack_exhausted"


def test_copy_arg_tree_stack_exhaustion_refuses_typed() -> None:
    """The capture-entry input copier shares the typed stack-budget refusal.

    The copier burns ~1 frame per level (fewer than the walkers), so this
    uses the full legal ceiling depth to exceed the tightened headroom.
    """

    deep = _deep_list(INPUT_TREE_MAX_DEPTH, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _tight_stack(lambda: copy_arg_tree(deep))
    assert excinfo.value.fields["code"] == "input_tree_stack_exhausted"


def test_simple_leaves_stack_exhaustion_refuses_typed() -> None:
    """Backend-resolution leaf sniffing shares the typed stack-budget refusal."""

    deep = _deep_list(100, torch.ones(1))
    with pytest.raises(InvalidArgumentError) as excinfo:
        _tight_stack(lambda: _simple_leaves(deep))
    assert excinfo.value.fields["code"] == "input_tree_stack_exhausted"


def test_walkers_still_complete_legal_depths_on_a_healthy_stack() -> None:
    """No false stack refusal: the same legal input walks fine untightened."""

    deep = _deep_list(100, torch.ones(1))
    seen: list[tuple[Any, ...]] = []
    walk_input_boundary(
        deep, key_component=raw_mapping_key_component, on_tensor=lambda t, p: seen.append(p)
    )
    assert len(seen) == 1
    assert snapshot_input_boundary(deep)["refusals"] == []
    assert isinstance(copy_arg_tree(deep), list)
    assert len(_simple_leaves(deep)) == 1


class _CycleModel(nn.Module):
    """Parameterized model whose forward reads one attribute/key of the input."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, box: Any) -> torch.Tensor:
        """Consume the boxed tensor."""

        tensor = box["x"] if isinstance(box, collections.abc.Mapping) else box.x
        return self.lin(tensor)


@dataclasses.dataclass
class _PeerBox:
    """Dataclass input that can close a reference cycle through ``peer``."""

    x: torch.Tensor
    peer: Any = None


@pytest.mark.smoke
def test_device_move_walker_refuses_cycles_and_depth_typed() -> None:
    """The device-move walker carries the shared cycle/depth guards.

    It runs FIRST for dataclass and non-``dict``-Mapping trees (the
    ``_simple_leaves`` entry gate descends only ``dict|tuple|list``), and it
    fires for every model with at least one parameter, so a cycle closed
    through a dataclass or a ``UserDict`` killed plain ``tl.trace`` with a
    raw ``RecursionError`` from library internals (unswept sibling of the
    95926175 walker-guard fix).
    """

    from torchlens._errors import InvalidArgumentError

    cyclic = _PeerBox(x=torch.ones(2))
    cyclic.peer = cyclic
    with pytest.raises(InvalidArgumentError):
        tl.trace(_CycleModel(), cyclic)

    user_dict = collections.UserDict()
    user_dict["x"] = torch.ones(2)
    user_dict["self"] = user_dict
    with pytest.raises(InvalidArgumentError):
        tl.trace(_CycleModel(), user_dict)

    deep = _PeerBox(x=torch.ones(2))
    for _ in range(INPUT_TREE_MAX_DEPTH + 10):
        deep = _PeerBox(x=torch.ones(2), peer=deep)
    with pytest.raises(InvalidArgumentError):
        tl.trace(_CycleModel(), deep)


@pytest.mark.smoke
def test_device_move_dataclass_rebuild_never_reruns_user_init() -> None:
    """The dataclass device-move rebuild is inert (no ``__post_init__`` re-run).

    Rebuilding by calling the user's constructor executed ``__init__``/
    ``__post_init__`` a SECOND time on already-initialized values, so the
    forward received VALUE-mutated fields (incremented counters, re-drawn
    RNG, recomputed derived tensors) that the post-move witness then honestly
    recorded -- a capture of a different program.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    @dataclasses.dataclass
    class CountingBox:
        """Dataclass whose ``__post_init__`` observably mutates a field."""

        x: torch.Tensor
        n: int = 0

        def __post_init__(self) -> None:
            """Increment the counter (a second run is detectable)."""

            self.n += 1

    original = CountingBox(x=torch.ones(2))
    assert original.n == 1
    moved = _move_tensors_to_device(original, "meta")
    assert moved is not original
    assert moved.x.device.type == "meta"
    assert moved.n == 1, "user __init__/__post_init__ ran a second time during the move"


def test_device_move_walker_descends_registered_containers() -> None:
    """Registered containers move through their own flatten/unflatten hooks.

    The walker's docstring claimed registered-container coverage while the
    implementation fell through every branch and returned the original
    unmoved -- indistinguishable from "nothing moved", surfacing later as a
    device-mismatch crash blamed on the user's model.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class WrappedBatch:
        """Minimal user container holding one tensor."""

        def __init__(self, tensor: torch.Tensor) -> None:
            self.tensor = tensor

    tl.register_container(
        WrappedBatch,
        lambda wrap: ([wrap.tensor], None),
        lambda aux, children: WrappedBatch(children[0]),
        state_complete=True,
    )
    wrapped = WrappedBatch(torch.ones(2))
    moved = _move_tensors_to_device(wrapped, "meta")
    assert moved is not wrapped
    assert moved.tensor.device.type == "meta"


# --- grind-p5 b3-opus-R12-2: mapping/list-subclass device-move rebuilds are INERT ------


def test_device_move_dict_subclass_rebuild_never_reruns_ctor_and_keeps_state() -> None:
    """The dict-subclass device-move rebuild is inert (no user ctor, state kept).

    The 66263de1 inert-rebuild fix landed only on the dataclass arm: the Mapping
    arm still called ``type(obj)(moved_mapping)``, re-running the user's
    ``__init__`` and RESETTING same-class instance state -- which the
    instance-state witness then recorded with zero refusals.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class ModeBox(dict):
        """Dict subclass whose ctor observably resets a mode attribute."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mode = "ctor-default"
            self.ctor_runs = getattr(self, "ctor_runs", 0) + 1

    box = ModeBox({"x": torch.ones(2)})
    box.mode = "user-set"
    moved = _move_tensors_to_device(box, "meta")
    assert moved is not box
    assert type(moved) is ModeBox
    assert moved["x"].device.type == "meta"
    assert moved.mode == "user-set", "user ctor re-ran and RESET instance state"
    assert moved.ctor_runs == 1


@pytest.mark.smoke
def test_device_move_list_subclass_rebuild_never_reruns_ctor_and_keeps_state() -> None:
    """The list-subclass device-move rebuild is inert (no user ctor, state kept)."""

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class TaggedList(list):
        """List subclass whose ctor observably resets a tag attribute."""

        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            self.tag = "ctor-default"

    tagged = TaggedList([torch.ones(2)])
    tagged.tag = "user-set"
    moved = _move_tensors_to_device(tagged, "meta")
    assert moved is not tagged
    assert type(moved) is TaggedList
    assert moved[0].device.type == "meta"
    assert moved.tag == "user-set", "user ctor re-ran and RESET instance state"


def test_device_move_defaultdict_moves_and_keeps_factory() -> None:
    """``defaultdict`` inputs actually MOVE (the ctor TypeError was swallowed).

    ``type(obj)(moved_mapping)`` on a defaultdict put the mapping in the
    ``default_factory`` slot, raised TypeError, and the except arm returned
    ``_UNMOVED`` -- so defaultdict inputs were silently never device-moved.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    source: collections.defaultdict[str, Any] = collections.defaultdict(list)
    source["x"] = torch.ones(2)
    moved = _move_tensors_to_device(source, "meta")
    assert moved is not source
    assert type(moved) is collections.defaultdict
    assert moved["x"].device.type == "meta"
    assert moved.default_factory is list


@pytest.mark.smoke
def test_device_move_ordereddict_subclass_keeps_order_and_state() -> None:
    """OrderedDict subclasses rebuild through the od physical channel, in order."""

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class OdBox(collections.OrderedDict):
        """OrderedDict subclass carrying one extra instance attribute."""

    box = OdBox([("b", torch.ones(2)), ("a", torch.zeros(2))])
    box.note = "kept"
    moved = _move_tensors_to_device(box, "meta")
    assert moved is not box
    assert type(moved) is OdBox
    assert list(moved.keys()) == ["b", "a"]
    assert moved["b"].device.type == "meta"
    assert moved.note == "kept"
    moved.move_to_end("b")
    assert list(moved.keys()) == ["a", "b"]  # od linked list is coherent


def test_device_move_dict_subclass_lying_keys_reads_physical_storage() -> None:
    """A ``keys()``/``__iter__`` override cannot hide children from the move.

    The mapping arm iterated ``obj.keys()`` -- overridable user code -- so a
    lying ``keys()`` shrank the rebuilt container refusal-free and hid tensor
    children from the device move.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class HidingDict(dict):
        """Dict subclass whose ``keys()``/``__iter__`` hide one key."""

        def keys(self) -> Any:  # type: ignore[override]
            return [key for key in dict.keys(self) if key != "hidden"]

        def __iter__(self) -> Any:
            return iter(self.keys())

    hiding = HidingDict({"seen": torch.ones(2), "hidden": torch.ones(2)})
    moved = _move_tensors_to_device(hiding, "meta")
    assert moved is not hiding
    assert dict.__len__(moved) == 2, "lying keys() shrank the rebuilt container"
    assert dict.__getitem__(moved, "hidden").device.type == "meta"


def test_device_move_list_subclass_lying_iter_reads_physical_storage() -> None:
    """A list-subclass ``__iter__`` override cannot hide children from the move."""

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class HidingList(list):
        """List subclass whose ``__iter__`` truncates to the first element."""

        def __iter__(self) -> Any:
            return iter([list.__getitem__(self, 0)])

    hiding = HidingList([torch.ones(2), torch.ones(3)])
    moved = _move_tensors_to_device(hiding, "meta")
    assert moved is not hiding
    assert list.__len__(moved) == 2, "lying __iter__ shrank the rebuilt container"
    assert list.__getitem__(moved, 1).device.type == "meta"


@pytest.mark.smoke
def test_device_move_userdict_moves_via_instance_state_without_protocol() -> None:
    """Non-dict Mappings move through their INSTANCE STATE, dataclass-style.

    The historical arm ran the user's constructor over a protocol read; the
    inert arm descends the enumerable instance state (``UserDict.data``),
    rebuilds via allocation + verbatim state, and never runs user ctor code.
    """

    from torchlens._capture_state_helpers import _move_tensors_to_device

    class Batch(collections.UserDict):
        """UserDict subclass whose ctor observably resets a mode attribute."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mode = "ctor-default"

    batch = Batch({"x": torch.ones(2)})
    batch.mode = "user-set"
    moved = _move_tensors_to_device(batch, "meta")
    assert moved is not batch
    assert type(moved) is Batch
    assert moved["x"].device.type == "meta"
    assert moved.mode == "user-set", "user ctor re-ran and RESET instance state"


def test_copy_arg_tree_dict_subclass_inert_rebuild_keeps_state_no_ctor() -> None:
    """``copy_arg_tree`` rebuilds dict subclasses inertly (sibling of the mover fix).

    The dict arm called ``type(arg)()`` -- the user's constructor -- and read
    children through the overridable ``items()`` protocol; the defaultdict arm
    additionally SUBSTITUTED exact ``defaultdict`` for any subclass.
    """

    class ModeBox(dict):
        """Dict subclass whose ctor observably resets a mode attribute."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mode = "ctor-default"

    box = ModeBox({"x": torch.ones(2)})
    box.mode = "user-set"
    copied = copy_arg_tree(box)
    assert copied is not box
    assert type(copied) is ModeBox
    assert copied.mode == "user-set", "user ctor re-ran and RESET instance state"
    assert torch.equal(copied["x"], box["x"]) and copied["x"] is not box["x"]


def test_copy_arg_tree_defaultdict_subclass_keeps_type_and_factory() -> None:
    """A defaultdict SUBCLASS copies to the same class, factory preserved."""

    class TrackingDefaults(collections.defaultdict):
        """Defaultdict subclass (the historical arm substituted exact defaultdict)."""

    source = TrackingDefaults(list)
    source["x"] = torch.ones(2)
    copied = copy_arg_tree(source)
    assert type(copied) is TrackingDefaults
    assert copied.default_factory is list
    assert torch.equal(copied["x"], source["x"])


@pytest.mark.smoke
def test_copy_arg_tree_lying_iteration_reads_physical_storage() -> None:
    """Lying ``keys()``/``__iter__`` overrides cannot shrink the copy."""

    class HidingDict(dict):
        """Dict subclass whose ``keys()``/``items()`` hide one key."""

        def keys(self) -> Any:  # type: ignore[override]
            return [key for key in dict.keys(self) if key != "hidden"]

        def items(self) -> Any:  # type: ignore[override]
            return [(key, dict.__getitem__(self, key)) for key in self.keys()]

        def __iter__(self) -> Any:
            return iter(self.keys())

    class HidingList(list):
        """List subclass whose ``__iter__`` truncates to the first element."""

        def __iter__(self) -> Any:
            return iter([list.__getitem__(self, 0)])

    hiding_dict = HidingDict({"seen": torch.ones(2), "hidden": torch.ones(2)})
    copied_dict = copy_arg_tree(hiding_dict)
    assert dict.__len__(copied_dict) == 2, "lying keys() shrank the copy"

    hiding_list = HidingList([torch.ones(2), torch.ones(3)])
    copied_list = copy_arg_tree(hiding_list)
    assert list.__len__(copied_list) == 2, "lying __iter__ shrank the copy"


# --- grind-p5 b3-opus-R12-1: the input-copy walker never runs user copy protocols ------


@pytest.mark.smoke
def test_safe_copy_input_tree_ignores_user_deepcopy_substitution() -> None:
    """A user ``__deepcopy__`` cannot substitute the captured program.

    The FIRST walker over the user's input tree was ``copy.deepcopy``: a
    container subclass's ``__deepcopy__``/``__reduce_ex__`` ran user code that
    could return a DIFFERENT tree (executed: forward captured over [5.0, 5.0]
    instead of [-3, -4]), and every witness honestly described the SUBSTITUTED
    tree -- zero refusals.
    """

    from torchlens.utils.arg_handling import safe_copy_input_tree

    protocol_calls: list[str] = []

    class SwappingList(list):
        """List subclass whose deepcopy protocol substitutes the payload."""

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            protocol_calls.append("deepcopy")
            return SwappingList([torch.tensor([5.0, 5.0])])

        def __reduce_ex__(self, protocol: int) -> Any:
            protocol_calls.append("reduce")
            return (SwappingList, ([torch.tensor([5.0, 5.0])],))

    original = SwappingList([torch.tensor([-3.0, -4.0])])
    copied_args, _, gaps = safe_copy_input_tree([original], {})
    copied = copied_args[0]
    assert protocol_calls == [], "user copy protocol ran during input copy"
    assert type(copied) is SwappingList
    assert torch.equal(list.__getitem__(copied, 0), torch.tensor([-3.0, -4.0])), (
        "user __deepcopy__ SUBSTITUTED the captured input tree"
    )


def test_safe_copy_input_tree_dict_subclass_state_kept_no_ctor() -> None:
    """Dict-subclass inputs copy inertly: no ctor re-run, state preserved."""

    from torchlens.utils.arg_handling import safe_copy_input_tree

    class ModeBox(dict):
        """Dict subclass whose ctor observably resets a mode attribute."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.mode = "ctor-default"

    box = ModeBox({"x": torch.ones(2)})
    box.mode = "user-set"
    copied_args, _, _ = safe_copy_input_tree([box], {})
    copied = copied_args[0]
    assert type(copied) is ModeBox
    assert copied is not box
    assert copied.mode == "user-set"
    assert torch.equal(copied["x"], box["x"]) and copied["x"] is not box["x"]


def test_trace_captures_original_values_despite_hostile_deepcopy() -> None:
    """End to end: the forward runs over the ORIGINAL values, not a substitute.

    Executed repro of the b3-opus-R12-1 headline: on the deepcopy-era walker
    this capture ran the forward over ``[5.0, 5.0]`` instead of
    ``[-3.0, -4.0]`` and every witness honestly described the substituted
    tree. The container is a namedtuple subclass (a fully supported input
    kind); its ``__deepcopy__`` must never be consulted.
    """

    class Box(typing.NamedTuple):
        """Declared one-field schema carrying the payload."""

        x: torch.Tensor

    class SwappingBox(Box):
        """Namedtuple subclass whose deepcopy protocol substitutes the payload."""

        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            return SwappingBox(torch.tensor([5.0, 5.0]))

    class FirstElement(nn.Module):
        """Model reading the wrapped payload physically."""

        def forward(self, wrapped: Any) -> torch.Tensor:
            return tuple.__getitem__(wrapped, 0) * 1.0

    log = tl.trace(FirstElement(), [SwappingBox(torch.tensor([-3.0, -4.0]))])
    assert torch.equal(log[-1].out, torch.tensor([-3.0, -4.0])), (
        "capture ran the forward over the deepcopy-SUBSTITUTED input tree"
    )


def test_safe_copy_input_tree_preserves_view_topology_and_grad_paths() -> None:
    """The inert rewrite keeps the deepcopy-era guarantees it replaced.

    Contract tensors (leaf, no grad, exact Tensor) keep CROSS-TENSOR storage
    sharing through the torch deepcopy protocol; grad-tracked tensors keep the
    clone path plus the metadata gap ledger; repeated tensor identity stays one
    object; unknown wrappers pass by reference.
    """

    from torchlens.utils.arg_handling import safe_copy_input_tree

    base = torch.arange(6.0)
    left, right = base[0:3], base[3:6]
    shared = torch.ones(2)
    grad_leaf = torch.ones(3, requires_grad=True)

    class OpaqueWrapper:
        """Non-container wrapper: passes by reference."""

    wrapper = OpaqueWrapper()
    copied_args, copied_kwargs, gaps = safe_copy_input_tree(
        [left, right, {"a": shared}, wrapper], {"k": (shared, grad_leaf)}
    )
    copied_left, copied_right, copied_map, copied_wrapper = copied_args
    assert copied_left.untyped_storage().data_ptr() == copied_right.untyped_storage().data_ptr()
    assert copied_left.untyped_storage().data_ptr() != left.untyped_storage().data_ptr()
    assert copied_map["a"] is copied_kwargs["k"][0]  # repeated identity stays one object
    assert copied_wrapper is wrapper
    copied_grad = copied_kwargs["k"][1]
    assert copied_grad.requires_grad and copied_grad is not grad_leaf


def test_snapshot_mapping_protocol_totality_refusal() -> None:
    """A dict-backed mapping whose protocol hides physical entries refuses.

    grind-p5 rollup: the mapping arm derived its ordered-key fact and child
    descent purely from the instance ``items()`` protocol and recorded no
    physical-arity fact -- a lying ``items()``/``keys()`` shrank the witnessed
    structure identically on the capture and runtime snapshots (a
    false-VERIFIED shape, the exact class ``physical_sequence_len`` guards on
    sequences and namedtuples). Dict-backed mappings now refuse when the
    protocol view is not total over the physical storage.
    """

    class HidingDict(dict):
        """Dict subclass whose protocol hides one physical entry."""

        def items(self) -> Any:  # type: ignore[override]
            return [(k, dict.__getitem__(self, k)) for k in dict.keys(self) if k != "hidden"]

    hiding = HidingDict({"seen": 1.0, "hidden": torch.ones(2)})
    snapshot = snapshot_input_boundary({"box": hiding})
    reasons = {refusal["reason"] for refusal in snapshot["refusals"]}
    assert "mapping_protocol_not_total" in reasons

    honest = snapshot_input_boundary({"box": {"seen": 1.0, "hidden": torch.ones(2)}})
    honest_reasons = {refusal["reason"] for refusal in honest["refusals"]}
    assert "mapping_protocol_not_total" not in honest_reasons


def _sneaky_substitution_dict(hidden: torch.Tensor) -> dict:
    """Dict subclass forging a count-PRESERVING protocol substitution (R12)."""

    class SneakySub(dict):
        """Physically holds ``hidden``; protocol presents an equal-count decoy."""

        def items(self) -> Any:  # type: ignore[override]
            return [
                ("visible", dict.__getitem__(self, "visible")),
                ("decoy", torch.zeros(2)),
            ]

    return SneakySub({"visible": torch.ones(2), "hidden": hidden})


def test_snapshot_mapping_count_preserving_substitution_refuses() -> None:
    """A count-preserving ``items()`` substitution refuses; forged snapshots never bless.

    R12 (4th-round carry): the totality fact compared only COUNT, so a lying
    ``items()`` presenting ``[("visible", t), ("decoy", zeros)]`` over physical
    storage ``{"visible": t, "hidden": t_h}`` passed with EMPTY refusals, the
    hidden tensor appeared in no snapshot node, and two instances with
    DIFFERENT hidden payloads snapshot EQUAL while concrete
    ``dict.__getitem__`` reads steered the forward -- a false-VERIFIED lane
    (the shrink/pad and namedtuple siblings are already refused).
    """

    snapshot = snapshot_input_boundary({"box": _sneaky_substitution_dict(torch.ones(3))})
    reasons = {refusal["reason"] for refusal in snapshot["refusals"]}
    assert "mapping_protocol_not_total" in reasons

    other = snapshot_input_boundary({"box": _sneaky_substitution_dict(torch.full((7,), 5.0))})
    assert {refusal["reason"] for refusal in other["refusals"]} >= {"mapping_protocol_not_total"}


def test_walker_routes_substituted_mapping_to_opaque_channel() -> None:
    """``walk_input_boundary`` never walks a forged protocol view (R12).

    The forged node routes WHOLE to the opaque channel (the fail-closed lane
    cycles and over-deep nests use) instead of witnessing the decoy entries,
    while an order-only divergence (a sorted-view subclass presenting the same
    entries) keeps walking normally -- the ordered-key witness follows the
    protocol traversal the model actually iterates.
    """

    tensors: list[tuple[Any, ...]] = []
    opaque: list[Any] = []
    walk_input_boundary(
        {"box": _sneaky_substitution_dict(torch.ones(3))},
        key_component=raw_mapping_key_component,
        on_tensor=lambda value, path: tensors.append(path),
        on_opaque_key_subtree=lambda value, path: opaque.append(value),
    )
    assert len(opaque) == 1 and isinstance(opaque[0], dict)
    assert not any("decoy" in path for path in tensors)

    class SortedView(dict):
        """Same entries, different order: legal protocol divergence."""

        def items(self) -> Any:  # type: ignore[override]
            return sorted(dict.items(self), key=lambda pair: pair[0], reverse=True)

    honest_tensors: list[tuple[Any, ...]] = []
    honest_opaque: list[Any] = []
    walk_input_boundary(
        {"box": SortedView({"a": torch.ones(2), "b": torch.ones(2)})},
        key_component=raw_mapping_key_component,
        on_tensor=lambda value, path: honest_tensors.append(path),
        on_opaque_key_subtree=lambda value, path: honest_opaque.append(value),
    )
    assert not honest_opaque
    assert len(honest_tensors) == 2
    sorted_snapshot = snapshot_input_boundary({"box": SortedView({"a": 1.0, "b": 2.0})})
    assert "mapping_protocol_not_total" not in {
        refusal["reason"] for refusal in sorted_snapshot["refusals"]
    }
