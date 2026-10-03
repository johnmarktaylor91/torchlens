"""Input-boundary walking: one inert authority, witnessed arity, bounded descent (B3 L4).

Every case below was a live defect on main:

* classification probed ``_fields`` through the LIVE instance (``hasattr``), so a hostile
  container's ``__getattribute__`` executed during ``snapshot_input_boundary`` -- the exact
  thing the r71 C contract promises never happens -- and could STEER the kind: a genuine
  namedtuple subclass whose hook raised ``AttributeError`` for ``"_fields"`` classified as
  a plain ``sequence``, so no declared-schema gate was ever consulted and its hidden state
  escaped judgment;
* the capture walker resolved fields by raw MRO while the runtime binding walker used a
  live ``getattr``, so the same container was field-addressable to one and zero-field to
  the other (untyped ``AttributeError`` at runtime, or two different path keyings);
* physical namedtuple arity was never witnessed: snapshots of instances with different
  hidden positional payloads compared EQUAL, and a zero-field namedtuple silently dropped
  its children INCLUDING TENSORS with no refusal and no witness gap;
* neither walker had a cycle guard or depth bound, so a self-referential container raised
  ``RecursionError`` from internals (not a diagnosable refusal);
* an unset ``init=False`` dataclass field killed every intervention-ready capture with an
  untyped ``AttributeError``;
* registered-container ``aux`` was compared type-blind, so a ``mode=True`` capture re-run
  with the ``mode=1`` twin reported VERIFIED end-to-end.
"""

from __future__ import annotations

import collections
import dataclasses
import enum
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._input_walk import (
    classify_input_container,
    declares_namedtuple_fields,
    empty_input_container_kind,
    namedtuple_arity_mismatch,
    snapshot_input_boundary,
    unset_declared_fields,
    walk_input_boundary,
)
from torchlens._runnable_witness_contracts import _container_field_names

_OneField = collections.namedtuple("_OneField", "x")
_ZeroField = collections.namedtuple("_ZeroField", "")


class _PropFields(tuple):
    """Tuple subclass whose ``_fields`` is a PROPERTY (a live hook, not a schema)."""

    probes: list[str] = []

    @property
    def _fields(self) -> tuple[str, ...]:
        """Record the probe and hand back a forged schema."""

        _PropFields.probes.append("_fields")
        return ("a", "b")


class _ListFields(tuple):
    """Tuple subclass whose ``_fields`` is a list, not a tuple of names."""

    _fields = ["a", "b"]


class _HidingNamedtuple(_OneField):
    """Real namedtuple subclass whose hook hides ``_fields`` from a live read."""

    def __getattribute__(self, name: str) -> Any:
        """Hide the declared schema from any live probe."""

        if name == "_fields":
            raise AttributeError(name)
        return object.__getattribute__(self, name)


def _reasons(snapshot: dict[str, Any]) -> list[str]:
    """Refusal reasons of one boundary snapshot, in order."""

    return [refusal["reason"] for refusal in snapshot["refusals"]]


def test_classification_never_touches_the_instance():
    """A ``_fields`` property must not run during classify or snapshot."""

    value = _PropFields((torch.zeros(1), torch.zeros(1)))
    _PropFields.probes.clear()
    assert classify_input_container(value) == "namedtuple"
    snapshot_input_boundary(value)
    assert _PropFields.probes == []


def test_a_hiding_hook_cannot_steer_classification():
    """A namedtuple whose hook hides ``_fields`` still classifies as a namedtuple."""

    value = tuple.__new__(_HidingNamedtuple, (torch.zeros(1),))
    assert declares_namedtuple_fields(value)
    assert classify_input_container(value) == "namedtuple"


def test_malformed_fields_refuse_instead_of_reading_as_zero_field():
    """A list-valued ``_fields`` is a non-total schema, not an empty one."""

    value = _ListFields((torch.zeros(1), torch.zeros(1)))
    assert namedtuple_arity_mismatch(value)
    assert "namedtuple_schema_not_total" in _reasons(snapshot_input_boundary(value))


def test_hidden_positional_payload_refuses_typed():
    """Physical arity beyond the declared schema must refuse, never pass silently."""

    value = tuple.__new__(_OneField, (torch.zeros(1), "capture_flag"))
    snapshot = snapshot_input_boundary(value)
    assert "namedtuple_schema_not_total" in _reasons(snapshot)
    root = snapshot["nodes"][0]
    assert root["kind"] == "namedtuple"
    assert root["size"] == 2  # PHYSICAL arity, not len(_fields)
    assert root["fields"] == ["x"]


def test_zero_field_namedtuple_carrying_tensors_is_not_empty():
    """A zero-field namedtuple with children must not classify as an EMPTY container."""

    value = tuple.__new__(_ZeroField, (torch.zeros(1), "steer"))
    assert empty_input_container_kind(value) is None
    assert classify_input_container(value) == "namedtuple"
    assert "namedtuple_schema_not_total" in _reasons(snapshot_input_boundary(value))
    # A GENUINELY empty namedtuple still reads as empty.
    assert empty_input_container_kind(_ZeroField()) == "namedtuple"


def test_empty_container_kind_alias_delegates_to_the_one_authority():
    """The ``_io.runnable`` spelling must be the inert implementation, not a twin."""

    from torchlens._io.runnable import empty_container_kind

    value = tuple.__new__(_ZeroField, (torch.zeros(1),))
    assert empty_container_kind(value) is empty_input_container_kind(value)
    assert empty_container_kind({}) == "mapping"
    assert empty_container_kind([]) == "sequence"


def test_both_walkers_resolve_fields_through_one_authority():
    """The runtime binding walker must agree with the capture walker, and never crash."""

    prop = _PropFields((torch.zeros(1), torch.zeros(1)))
    listed = _ListFields((torch.zeros(1), torch.zeros(1)))
    assert _container_field_names(prop) == ()
    assert _container_field_names(listed) == ()
    assert _container_field_names(_OneField(torch.zeros(1))) == ("x",)


def test_self_referential_container_refuses_instead_of_recursing():
    """A cycle must become a typed refusal, never a ``RecursionError``."""

    tree: dict[str, Any] = {"x": torch.zeros(1)}
    tree["self"] = tree
    snapshot = snapshot_input_boundary(tree)
    assert "input_container_cycle" in _reasons(snapshot)


def test_over_deep_container_refuses_instead_of_recursing():
    """A nest past the declared bound must become a typed refusal."""

    value: Any = torch.zeros(1)
    for _ in range(3000):
        value = [value]
    snapshot = snapshot_input_boundary(value)
    assert "input_container_too_deep" in _reasons(snapshot)


def test_cycle_ceilings_the_subtree_in_the_value_walker():
    """The literal/site walker must ceiling a cycle through its opaque channel."""

    tree: dict[str, Any] = {"x": torch.zeros(1)}
    tree["self"] = tree
    opaque: list[tuple[Any, ...]] = []
    tensors: list[tuple[Any, ...]] = []
    walk_input_boundary(
        tree,
        key_component=lambda key: key,
        on_tensor=lambda _value, path: tensors.append(path),
        on_opaque_key_subtree=lambda _value, path: opaque.append(path),
    )
    assert tensors == [("x",)]
    assert opaque == [("self",)]


def test_repeated_sibling_container_is_still_witnessed_twice():
    """The fence tracks ANCESTORS only: a shared container must not be dropped."""

    shared = {"t": torch.zeros(1)}
    tensors: list[tuple[Any, ...]] = []
    walk_input_boundary(
        {"a": shared, "b": shared},
        key_component=lambda key: key,
        on_tensor=lambda _value, path: tensors.append(path),
    )
    assert sorted(tensors) == [("a", "t"), ("b", "t")]


def test_unset_init_false_dataclass_field_no_longer_kills_the_capture():
    """A legal lazily-populated dataclass field must not raise from internals."""

    @dataclasses.dataclass
    class _Box:
        """Dataclass with a declared-but-unset scratch field."""

        x: Any
        cache: Any = dataclasses.field(init=False)

    class _Model(nn.Module):
        """Reads only the set field."""

        def forward(self, box: Any) -> torch.Tensor:
            """Double the tensor field."""

            return box.x * 2

    box = _Box(torch.randn(1, 4))
    assert unset_declared_fields(box) == ("cache",)
    snapshot = snapshot_input_boundary(box)
    assert "unset_declared_field" in _reasons(snapshot)
    assert snapshot["nodes"][0]["unset_fields"] == ["cache"]
    # The whole point: an intervention-ready capture completes instead of raising.
    trace = tl.trace(_Model(), box, capture=tl.options.CaptureOptions(intervention_ready=True))
    assert trace.num_ops >= 1


def test_registered_aux_is_type_strict():
    """``True``/``1`` aux twins must not compare equal; a semantic aux must refuse."""

    @dataclasses.dataclass
    class _Wrap:
        """Registered container whose aux carries a mode flag."""

        value: Any
        mode: Any

    tl.register_container(
        _Wrap,
        lambda wrap: ((wrap.value,), wrap.mode),
        lambda children, aux: _Wrap(children[0], aux),
        state_complete=True,
    )
    tensor = torch.zeros(1)
    bool_aux = snapshot_input_boundary(_Wrap(tensor, True))
    int_aux = snapshot_input_boundary(_Wrap(tensor, 1))
    assert bool_aux["nodes"] != int_aux["nodes"]
    assert not _reasons(bool_aux)

    # A NaN aux must compare equal to itself instead of false-diverging.
    nan_a = snapshot_input_boundary(_Wrap(tensor, float("nan")))
    nan_b = snapshot_input_boundary(_Wrap(tensor, float("nan")))
    assert nan_a["nodes"] == nan_b["nodes"]

    class _Mode(enum.IntEnum):
        """Semantic aux the declared schema cannot carry."""

        FAST = 1

    assert "registered_aux_unsafe" in _reasons(snapshot_input_boundary(_Wrap(tensor, _Mode.FAST)))


def test_foreign_grafted_getattribute_is_uninspectable():
    """A grafted foreign C slot wrapper must fail closed, not pass the inertness gate."""

    from torchlens._input_walk import _declared_schema_uninspectable

    @dataclasses.dataclass
    class _Sneak:
        """Dataclass with ``type.__getattribute__`` grafted on."""

        x: Any

    _Sneak.__getattribute__ = type.__getattribute__  # type: ignore[method-assign]
    assert _declared_schema_uninspectable(_Sneak.__new__(_Sneak)) is True


def test_ordinary_containers_stay_inspectable():
    """The tightened gate must not reject the containers real users pass."""

    from torchlens._input_walk import _declared_schema_uninspectable

    @dataclasses.dataclass
    class _Cfg:
        """Plain dataclass input."""

        x: Any

    assert _declared_schema_uninspectable(_Cfg(torch.zeros(1))) is False
    assert _declared_schema_uninspectable(_OneField(torch.zeros(1))) is False
    assert _declared_schema_uninspectable({}) is False
    assert _declared_schema_uninspectable([]) is False


def test_mapping_ordered_key_fact_comes_from_the_child_traversal():
    """The persisted key order must be the order the model actually iterates."""

    class _Skew(dict):
        """Mapping whose ``keys()`` order contradicts ``__iter__``/``items()``."""

        def keys(self):  # type: ignore[override]
            """Return the keys in reversed order."""

            return reversed(list(dict.keys(self)))

    value = _Skew({"a": torch.zeros(1), "b": torch.zeros(1)})
    root = snapshot_input_boundary({"cfg": value})["nodes"][1]
    assert root["kind"] == "mapping"
    assert root["keys"] == ["a", "b"]


class _EvilZeroField(tuple):
    """Zero-field namedtuple subclass whose ``__len__`` hides a physical payload."""

    _fields: tuple[str, ...] = ()

    def __len__(self) -> int:  # noqa: D105 - hostile probe
        return 0


class _EvilOneField(collections.namedtuple("_EvilOneFieldBase", ["x"])):
    """One-field namedtuple subclass whose ``__len__`` forges the recorded arity."""

    __slots__ = ()

    def __len__(self) -> int:  # noqa: D105 - hostile probe
        return 1


class _LyingLenList(list):
    """List subclass whose ``__len__`` claims emptiness over real children."""

    def __len__(self) -> int:  # noqa: D105 - hostile probe
        return 0


def test_hostile_len_cannot_steer_container_kind_or_forge_arity():
    """Arity facts read the concrete builtin slot, never the instance ``__len__``.

    A zero-field namedtuple subclass with ``__len__() == 0`` physically carrying
    ``(tensor, "steer")`` classified ``empty`` and every walker dropped its
    children (tensors included) with no refusal -- and the runtime snapshot
    computed the SAME wrong value, so the structure tripwire passed on both
    ends (false VERIFIED). A one-field variant physically carrying two elements
    recorded ``size == 1`` and snapshots of different hidden payloads compared
    equal.
    """

    from torchlens._input_walk import snapshot_input_boundary

    hidden_tensor = torch.ones(2)
    evil_zero = tuple.__new__(_EvilZeroField, (hidden_tensor, "steer"))
    assert classify_input_container(evil_zero) == "namedtuple"
    assert empty_input_container_kind(evil_zero) is None
    snapshot = snapshot_input_boundary(evil_zero)
    reasons = [refusal["reason"] for refusal in snapshot.get("refusals", [])]
    assert "namedtuple_schema_not_total" in reasons

    evil_one = tuple.__new__(_EvilOneField, (torch.ones(2), "hidden"))
    one_snapshot = snapshot_input_boundary(evil_one)
    one_reasons = [refusal["reason"] for refusal in one_snapshot.get("refusals", [])]
    assert "namedtuple_schema_not_total" in one_reasons
    named_nodes = [node for node in one_snapshot["nodes"] if node.get("kind") == "namedtuple"]
    assert named_nodes and named_nodes[0]["size"] == 2


def test_hostile_len_sequence_still_walks_physical_children():
    """A lying sequence ``__len__`` cannot classify real children away."""

    from torchlens._input_walk import (
        raw_mapping_key_component,
        snapshot_input_boundary,
        walk_input_boundary,
    )

    lying = _LyingLenList([torch.ones(2), torch.zeros(2)])
    assert classify_input_container(lying) == "sequence"
    snapshot = snapshot_input_boundary(lying)
    sizes = [node["size"] for node in snapshot["nodes"] if "size" in node]
    assert sizes == [2]

    seen: list[tuple[Any, ...]] = []
    walk_input_boundary(
        lying,
        key_component=raw_mapping_key_component,
        on_tensor=lambda _tensor, path: seen.append(tuple(path)),
    )
    assert len(seen) == 2


class _ModeBox(dict):
    """Mapping subclass whose non-protocol attribute steers forward control flow."""


class _SideChannelList(list):
    """List subclass carrying a literal side field."""


class _BackingStoreMapping(dict):
    """Well-behaved custom mapping keeping an opaque backing attribute."""

    def __init__(self, data: dict) -> None:
        super().__init__(data)
        self.extra_store = dict(data)


def test_same_class_instance_state_is_witnessed_on_protocol_subclasses():
    """Changed same-class instance state diverges the structure snapshot.

    The exact-type node fact catches a class swap but not changed fields on
    another instance of the SAME class: a ``ModeBox(dict)`` whose ``mode``
    flipped between capture and replay walked to the same tensor-leaf
    structure, so a numerically wrong replay reported VERIFIED.
    """

    from torchlens._input_walk import snapshot_input_boundary

    box_a = _ModeBox({"x": torch.ones(2)})
    box_a.mode = "a"
    box_b = _ModeBox({"x": torch.ones(2)})
    box_b.mode = "b"
    snap_a = snapshot_input_boundary(box_a)
    snap_b = snapshot_input_boundary(box_b)
    assert snap_a != snap_b, "same-class changed literal field must change the snapshot"
    assert not snap_a.get("refusals")

    seq_a = _SideChannelList([torch.ones(2)])
    seq_a.flag = 1
    seq_b = _SideChannelList([torch.ones(2)])
    seq_b.flag = 2
    assert snapshot_input_boundary(seq_a) != snapshot_input_boundary(seq_b)

    # A well-behaved custom mapping's opaque backing store witnesses as a
    # stable type-identity token: two same-shape instances stay comparable.
    store_a = _BackingStoreMapping({"x": torch.ones(2)})
    store_b = _BackingStoreMapping({"x": torch.ones(2)})
    assert snapshot_input_boundary(store_a) == snapshot_input_boundary(store_b)
    assert not snapshot_input_boundary(store_a).get("refusals")


def test_mode_box_changed_field_diverges_runnable_replay(tmp_path):
    """The r66-R1/ModeBox end-to-end repro: same class, changed field, typed refusal."""

    from torchlens.errors import PathDivergenceError
    from torchlens.runnable import PathFaithfulness

    class ModeRoutedModel(nn.Module):
        """Add a mode-selected constant to the boxed tensor."""

        def forward(self, box: _ModeBox) -> torch.Tensor:
            """Route on the box's non-protocol attribute."""

            return box["x"] + (1 if box.mode == "a" else 2)

    model = ModeRoutedModel().eval()
    tensor = torch.arange(2.0)
    box_a = _ModeBox({"x": tensor})
    box_a.mode = "a"
    captured = tl.trace(
        model,
        box_a,
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    path = tmp_path / "m.tlspec"
    tl.save(captured, path, level="runnable", include_weights=True)

    changed = _ModeBox({"x": tensor})
    changed.mode = "b"
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=changed)

    same = _ModeBox({"x": tensor})
    same.mode = "a"
    result = tl.load(path).run(inputs=same)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_nontotal_namedtuple_input_refuses_capture_typed():
    """A malformed-``_fields`` namedtuple INPUT refuses capture entry typed (B3R4-R12-1).

    Before the fix the capture-side walkers descended the (empty) declared
    field list, so every tensor leaf under the container vanished: the trace
    had NO input node, the consuming op had no parents, and the gap was
    misattributed to a stale-reference "escape" -- while the settled outcome
    still read COMPLETE.
    """

    from torchlens._errors import InvalidArgumentError

    model = nn.Identity()

    class _Consume(nn.Module):
        def forward(self, box):  # noqa: D102 - test fixture
            return box[0] * 2

    model = _Consume()
    tensor = torch.zeros(3)

    # Control: the plain-tuple spelling captures with a real input node.
    control = tl.trace(model, (tensor,))
    assert len(control.input_ops) == 1

    malformed = _ListFields((tensor,))
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.trace(model, malformed)
    assert excinfo.value.fields.get("code") == "input_namedtuple_schema_not_total"

    # Arity-mismatched schema (declared 1 field, physically 2 elements). The
    # arg-copy ladder independently warns `input_copy_semantics_unverifiable`
    # (it cannot rebuild the subclass faithfully) before the typed refusal;
    # tolerate that pre-existing disclosure here.
    import warnings as _warnings

    from torchlens._errors import TorchLensCaptureGapWarning

    hidden = tuple.__new__(_OneField, (tensor, torch.ones(1)))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", TorchLensCaptureGapWarning)
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.trace(model, hidden)
    assert excinfo.value.fields.get("code") == "input_namedtuple_schema_not_total"

    # A well-formed namedtuple input keeps capturing.
    fine = _OneField(x=tensor)
    ok = tl.trace(model, fine)
    assert len(ok.input_ops) == 1


def test_walk_input_boundary_refuses_nontotal_namedtuple():
    """The shared traversal itself fails closed on a non-total schema."""

    from torchlens._errors import InvalidArgumentError
    from torchlens._input_walk import raw_mapping_key_component

    seen: list[Any] = []
    with pytest.raises(InvalidArgumentError) as excinfo:
        walk_input_boundary(
            {"box": _ListFields((torch.zeros(1),))},
            (),
            key_component=raw_mapping_key_component,
            on_tensor=lambda tensor, path: seen.append(path),
        )
    assert excinfo.value.fields.get("code") == "input_namedtuple_schema_not_total"
    assert seen == []
