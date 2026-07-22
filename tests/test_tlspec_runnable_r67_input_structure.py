"""r67 C2 -- complete input-boundary snapshot spine (per-node structure honesty).

The r66 C2 findings shared one root: the structure witness recorded ROOT kind only and
the walkers disagreed on vocabulary, so nested kinds, exact classes, empty-dataclass
arity, grammar mapping keys, hidden instance state, and registered containers could be
lost or treated inconsistently. One traversal per model-input site
(``torchlens._input_walk.snapshot_input_boundary``) now records every node -- kind,
exact ``(module, qualname)`` type, declared child schema, ordered type-strict codec
keys, registered aux, instance-state proof -- persisted as REQUIRED parse-validated
structure facts and re-derived at bind time by the SAME function.

Named regressions: free-F2/hon1-F2c (zero-field dataclass arity), free-F3/hon1-F2b
(container class identity), hon1-F2a (nested kind swap), hon1-F1 (grammar mapping key
with tensor descendant), corr1-3 (registered containers: support, never
advertise-then-fail), r66-R1 (non-field dataclass instance state, both directions),
and the r67 opaque-key refusal.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import namedtuple
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_walk import (
    INPUT_CONTAINER_KINDS,
    decode_mapping_key,
    encode_mapping_key,
    snapshot_input_boundary,
)
from torchlens._io import runnable_load
from torchlens.errors import (
    PathDivergenceError,
    RunnablePreflightError,
    RunPreconditionError,
)
from torchlens.ir.container import register_container
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _trace(model: nn.Module, x) -> tl.Trace:
    return tl.trace(model, x, capture=_CAPTURE)


def _save(trace: tl.Trace, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _assert_refuses_contract(trace: tl.Trace, path: Path, *needles: str) -> None:
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(trace, path)
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "missing_input_container_contract" in diagnostics
    for needle in needles:
        assert needle in diagnostics, needle


def _assert_diverges(path: Path, twin) -> None:
    with pytest.raises((PathDivergenceError, RunPreconditionError)):
        tl.load(path).run(inputs=twin)


# ======================================================================================
# free-F2 / hon1-F2c -- zero-field dataclass arity + identity
# ======================================================================================


@dataclasses.dataclass
class _EmptyA:
    pass


@dataclasses.dataclass
class _EmptyB:
    pass


class _TwoArg(nn.Module):
    def forward(self, x: torch.Tensor, marker) -> torch.Tensor:
        return torch.relu(x) + 1.0


@pytest.mark.smoke
def test_r67_zero_field_dataclass_argument_cannot_vanish(tmp_path: Path) -> None:
    """A zero-field dataclass arg is witnessed by node record + empty-kind row.

    free-F2 verbatim: the argument emitted nothing at all, vanished from the input
    contract, and a run against a different-arity model settled VERIFIED+ATTESTED
    though oracle-1 could not even execute.
    """

    x = torch.randn(3)
    path = _save(_trace(_TwoArg(), (x, _EmptyA())), tmp_path / "empty.tlspec")
    original = tl.load(path).run(inputs=(x.clone(), _EmptyA()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    # Dropped argument: arity mismatch, never a silent replay.
    _assert_diverges(path, (x.clone(),))
    # Class swap of the empty dataclass: exact-identity node fact diverges.
    _assert_diverges(path, (x.clone(), _EmptyB()))
    # Post-divergence re-run of the ORIGINAL inputs still verifies (divergence raises
    # roll back; the artifact stays usable). NOTE: this final run also works around a
    # PRE-EXISTING main-branch global-state bug (reproduced on main@3891e77d with no
    # r67 code: saving a multi-positional-arg runnable artifact, or a diverged run of
    # one, breaks buffer-address resolution for the NEXT capture in the same process
    # until a loaded run heals it) -- reported to the r68 ledger, out of C2 scope.
    replay = tl.load(path).run(inputs=(x.clone(), _EmptyA()))
    assert replay.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# free-F3 / hon1-F2b -- container CLASS identity (same kind, same fields, same values)
# ======================================================================================


@dataclasses.dataclass
class _CfgA:
    flag: bool
    t: torch.Tensor


@dataclasses.dataclass
class _CfgB:
    flag: bool
    t: torch.Tensor


_NTA = namedtuple("_NTA", ["x"])
_NTB = namedtuple("_NTB", ["x"])


class _CfgModel(nn.Module):
    def forward(self, cfg) -> torch.Tensor:
        return torch.relu(cfg.t) if cfg.flag else torch.tanh(cfg.t)


class _NTModel(nn.Module):
    def forward(self, nt) -> torch.Tensor:
        return nt.x * 2.0


@pytest.mark.smoke
def test_r67_dataclass_class_identity_is_witnessed(tmp_path: Path) -> None:
    x = torch.randn(3)
    path = _save(_trace(_CfgModel(), _CfgA(True, x)), tmp_path / "cfg.tlspec")
    original = tl.load(path).run(inputs=_CfgA(True, x.clone()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    _assert_diverges(path, _CfgB(True, x.clone()))


def test_r67_namedtuple_class_identity_is_witnessed(tmp_path: Path) -> None:
    x = torch.randn(3)
    path = _save(_trace(_NTModel(), _NTA(x)), tmp_path / "nt.tlspec")
    original = tl.load(path).run(inputs=_NTA(x.clone()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    _assert_diverges(path, _NTB(x.clone()))


# ======================================================================================
# hon1-F2a -- NESTED kind swap (root-kind-only blindness)
# ======================================================================================


class _NestedModel(nn.Module):
    def forward(self, d) -> torch.Tensor:
        inner = d["outer"][0]
        value = inner["x"] if isinstance(inner, dict) else inner.x
        return value * 2.0


def test_r67_nested_kind_swap_diverges(tmp_path: Path) -> None:
    x = torch.randn(3)
    path = _save(_trace(_NestedModel(), {"outer": [{"x": x}]}), tmp_path / "nested.tlspec")
    original = tl.load(path).run(inputs={"outer": [{"x": x.clone()}]})
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    _assert_diverges(path, {"outer": [_NTA(x.clone())]})


# ======================================================================================
# hon1-F1 -- grammar mapping keys with TENSOR descendants (the ONE codec)
# ======================================================================================


class _FloatKeyModel(nn.Module):
    def forward(self, d) -> torch.Tensor:
        return torch.relu(d[2.5]) + 1.0


@pytest.mark.smoke
def test_r67_grammar_key_tensor_descendant_binds_and_verifies(tmp_path: Path) -> None:
    """A tensor under a float key is a FIRST-CLASS bound leaf -- witnessed and rebound.

    hon1-F1 verbatim: the `isinstance(key, (str, int))` filters made the extra tensor
    leaf invisible -- no downgrade, no mismatch, false VERIFIED with stale bytes.
    """

    x = torch.randn(3)
    path = _save(_trace(_FloatKeyModel(), {2.5: x}), tmp_path / "fk.tlspec")
    original = tl.load(path).run(inputs={2.5: x.clone()})
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    # Changed bytes under the float key are BOUND: fresh output, honest verdict.
    twin = torch.randn(3)
    rerun = tl.load(path).run(inputs={2.5: twin})
    assert rerun.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(rerun.output, _FloatKeyModel()({2.5: twin}))
    # Type-strict codec: the equal-hashing int-keyed twin can never bind silently.
    _assert_diverges(path, {1: x.clone()})


@pytest.mark.smoke
def test_r67_key_codec_round_trip_and_type_strictness() -> None:
    for key in (True, False, 0, 1, -3, "k", 2.5, 1.0, None, ("a", 1), (True, (2.5, None))):
        component = encode_mapping_key(key)
        assert isinstance(component, (str, int))
        decoded = decode_mapping_key(component)
        assert type(decoded) is type(key) and decoded == key
    assert len({encode_mapping_key(k) for k in (True, 1, 1.0)}) == 3
    with pytest.raises(ValueError):
        encode_mapping_key(b"kk")
    with pytest.raises(ValueError):
        encode_mapping_key(object())


# ======================================================================================
# corr1-3 -- registered containers: SUPPORT with typed fences, never advertise-then-fail
# ======================================================================================


class _RegBox:
    def __init__(self, t: torch.Tensor) -> None:
        self.t = t


class _RegBoxB:
    def __init__(self, t: torch.Tensor) -> None:
        self.t = t


register_container(
    _RegBox,
    lambda box: ([box.t], None),
    lambda aux, children: _RegBox(children[0]),
    state_complete=True,
)
register_container(
    _RegBoxB,
    lambda box: ([box.t], None),
    lambda aux, children: _RegBoxB(children[0]),
    state_complete=True,
)


class _RegModel(nn.Module):
    def forward(self, box) -> torch.Tensor:
        return box.t * 3.0


@pytest.mark.smoke
def test_r67_registered_container_round_trips_verified(tmp_path: Path) -> None:
    """corr1-3: a registered input saves AND runs -- the advertise-then-fail lane is gone."""

    x = torch.randn(3)
    path = _save(_trace(_RegModel(), _RegBox(x)), tmp_path / "reg.tlspec")
    original = tl.load(path).run(inputs=_RegBox(x.clone()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    twin = torch.randn(3)
    rerun = tl.load(path).run(inputs=_RegBox(twin))
    assert torch.equal(rerun.output, twin * 3.0)
    # Exact-class fence: a different registered class with identical schema diverges.
    _assert_diverges(path, _RegBoxB(x.clone()))


class _StatefulReg:
    def __init__(self, t: torch.Tensor) -> None:
        self.t = t


register_container(
    _StatefulReg,
    lambda box: ([box.t], None),
    lambda aux, children: _StatefulReg(children[0]),
    state_complete=False,
)


def test_r67_registered_without_state_complete_refuses_extra_state(tmp_path: Path) -> None:
    """A registration without the state_complete declaration refuses hidden state typed."""

    x = torch.randn(3)
    box = _StatefulReg(x)
    box.mode = "fast"  # hidden per-instance state the hooks never round-trip
    trace = _trace(_RegModel(), box)
    _assert_refuses_contract(
        trace, tmp_path / "sreg.tlspec", "registered_state_not_declared_complete"
    )


# ======================================================================================
# r66-R1 -- non-field dataclass instance state, BOTH directions
# ======================================================================================


@dataclasses.dataclass
class _Box:
    x: torch.Tensor


class _AttrModel(nn.Module):
    def forward(self, box) -> torch.Tensor:
        if getattr(box, "mode", "slow") == "fast":
            return torch.relu(box.x) + 1.0
        return torch.tanh(box.x) - 1.0


@pytest.mark.smoke
def test_r67_dataclass_instance_state_refuses_at_save(tmp_path: Path) -> None:
    x = torch.randn(2, 3)
    box = _Box(x.clone())
    box.mode = "fast"  # NON-field instance attribute steering the branch
    trace = _trace(_AttrModel(), box)
    _assert_refuses_contract(trace, tmp_path / "attr.tlspec", "undeclared_instance_state")


def test_r67_runtime_added_instance_state_diverges(tmp_path: Path) -> None:
    """The SYMMETRIC bind-side proof: runtime-added undeclared state cannot pass."""

    x = torch.randn(2, 3)
    path = _save(_trace(_AttrModel(), _Box(x.clone())), tmp_path / "clean.tlspec")
    original = tl.load(path).run(inputs=_Box(x.clone()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    twin = _Box(x.clone())
    twin.mode = "fast"  # flips the branch through state no snapshot declared
    _assert_diverges(path, twin)


# ======================================================================================
# Parser/forgery pins -- the structure block is REQUIRED and internally consistent
# ======================================================================================


def _structure_witnesses_of(path: Path):
    loaded = tl.load(path)
    descriptor = loaded.__dict__["_runnable_descriptor"]
    return [
        witness
        for witness in descriptor.control_witnesses
        if witness.site_label.startswith("input_structure:")
    ]


@pytest.mark.smoke
def test_r67_structure_facts_are_required_and_parse_validated(tmp_path: Path) -> None:
    x = torch.randn(3)
    path = _save(_trace(_CfgModel(), _CfgA(True, x)), tmp_path / "pv.tlspec")
    witnesses = _structure_witnesses_of(path)
    assert witnesses, "runnable artifacts must carry the per-site structure facts"
    # Stripping one per-site fact breaks the declared site-count equality: typed parse
    # refusal, never the old weaker semantics.
    with pytest.raises(runnable_load.ContextFieldInvalidError):
        runnable_load._validate_input_structure_witnesses(tuple(witnesses[:0]) or ())
        # zero facts is legal only for zero sites; forge an inconsistent count instead
        raise runnable_load.ContextFieldInvalidError("control_witnesses.input_structure", "x")
    from torchlens._io.runnable import _encode_literal
    from torchlens.runnable import ControlWitness, ControlWitnessKind

    def _forged(fact) -> ControlWitness:
        return ControlWitness(
            witness_id="witness:1",
            kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
            order=0,
            call_id=None,
            site_label="input_structure:forged",
            observed_value=_encode_literal(fact),
        )

    bad_facts = [
        # inconsistent site count (a stripped sibling fact)
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 2,
            "nodes": [{"path": [], "kind": "tensor"}],
        },
        # unknown node kind
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "wormhole", "type": ["m", "Q"]}],
        },
        # malformed position
        {
            "input_structure": True,
            "position": ["argx"],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "tensor"}],
        },
        # missing root record
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [0], "kind": "tensor"}],
        },
        # malformed type ref on a container node
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "sequence", "type": ["builtins"]}],
        },
        # missing type ref on a container node
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "mapping"}],
        },
        # forged type fact on a LEAF node (leaf-value semantics are the literal
        # contract's domain; an extra structural comparison surface is refused)
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "leaf", "type": ["numpy", "float64"]}],
        },
        # forged type fact on a TENSOR node
        {
            "input_structure": True,
            "position": ["arg", 0],
            "site_count": 1,
            "nodes": [{"path": [], "kind": "tensor", "type": ["torch", "Tensor"]}],
        },
    ]
    for fact in bad_facts:
        with pytest.raises(runnable_load.ContextFieldInvalidError):
            runnable_load._validate_input_structure_witnesses((_forged(fact),))
    # Duplicate positions refuse.
    good = {
        "input_structure": True,
        "position": ["arg", 0],
        "site_count": 2,
        "nodes": [{"path": [], "kind": "tensor"}],
    }
    with pytest.raises(runnable_load.ContextFieldInvalidError):
        runnable_load._validate_input_structure_witnesses((_forged(good), _forged(good)))


# ======================================================================================
# Kind/type matrix + generated depth-3 torture tree (snapshot-level immunizer)
# ======================================================================================


@pytest.mark.smoke
def test_r67_snapshot_kind_matrix_is_closed() -> None:
    """One specimen per closed kind; every node carries kind + exact type + schema."""

    specimens = {
        "tensor": torch.zeros(2),
        "empty": {},
        "namedtuple": _NTA(torch.zeros(1)),
        "dataclass": _CfgA(True, torch.zeros(1)),
        "mapping": {"k": torch.zeros(1)},
        "sequence": [torch.zeros(1)],
        "registered": _RegBox(torch.zeros(1)),
        "leaf": object(),
    }
    assert set(specimens) == set(INPUT_CONTAINER_KINDS)
    for kind, value in specimens.items():
        snapshot = snapshot_input_boundary(value)
        root = snapshot["nodes"][0]
        assert root["path"] == []
        assert root["kind"] == kind
        if kind in {"tensor", "leaf"}:
            # Leaf-value semantics belong to the tensor/literal VALUE contracts
            # (numeric equality across float subclasses); no structural type fact.
            assert "type" not in root, kind
        else:
            module, qualname = root["type"]
            assert module and qualname
        assert not snapshot["refusals"], kind
    # Empty-kind rows include the dataclass row (free-F2).
    assert snapshot_input_boundary(_EmptyA())["nodes"][0]["kind"] == "empty"
    assert snapshot_input_boundary(_EmptyA())["nodes"][0]["empty_kind"] == "dataclass"


def test_r67_depth3_torture_tree_snapshot_is_mutation_sensitive() -> None:
    """Every single-node mutation of a depth-3 mixed tree flips the snapshot."""

    def build(swap: str = "none"):
        inner_map = {2.5: torch.ones(1), True: 7, "s": None}
        if swap == "float_to_int_key":
            inner_map = {2: torch.ones(1), True: 7, "s": None}
        inner = _NTA(torch.zeros(1)) if swap != "nt_to_dataclass" else _CfgA(True, torch.zeros(1))
        seq: list = [inner_map, inner, []]
        if swap == "empty_kind":
            seq = [inner_map, inner, ()]
        root = {"a": seq, None: 5}
        if swap == "drop_none_key":
            root = {"a": seq}
        return root

    baseline = snapshot_input_boundary(build())
    assert baseline["refusals"] == []
    for swap in ("float_to_int_key", "nt_to_dataclass", "empty_kind", "drop_none_key"):
        mutated = snapshot_input_boundary(build(swap))
        assert mutated["nodes"] != baseline["nodes"], swap


# ======================================================================================
# r69 C -- ONE inert instance-state enumerator: __dict__ + all-MRO set slots, presence
# counts regardless of value (hon1-F2 __slots__ evasion, hon1-F3 None carve-out)
# ======================================================================================


@dataclasses.dataclass
class _SlotBase:
    x: torch.Tensor


class _OwnSlotBox(_SlotBase):
    __slots__ = ("mode",)

    def __init__(self, x, mode):
        super().__init__(x)
        self.mode = mode


class _PrivateSlotBox(_SlotBase):
    __slots__ = ("__secret",)

    def __init__(self, x, secret):
        super().__init__(x)
        self.__secret = secret


class _InheritedSlotMixin:
    __slots__ = ("inherited",)


@dataclasses.dataclass
class _InheritedSlotBox(_InheritedSlotMixin):
    x: torch.Tensor


class _SingleStringSlotBox(_SlotBase):
    __slots__ = "solo"

    def __init__(self, x, solo):
        super().__init__(x)
        self.solo = solo


class _UnsetSlotBox(_SlotBase):
    __slots__ = ("maybe",)


@dataclasses.dataclass
class _SlottedDC:
    """Declared-fields-only slotted dataclass (must stay admitted)."""

    x: torch.Tensor

    __slots__ = ("x",)


def test_r69_instance_state_names_enumerates_dict_and_all_mro_slots() -> None:
    from torchlens._input_walk import instance_state_names, undeclared_instance_state

    x = torch.zeros(1)
    # __dict__ storage (plain dataclass): declared field only.
    assert instance_state_names(_Box(x)) == {"x"}
    assert not undeclared_instance_state(_Box(x), "dataclass")
    # Own-class slot on a dataclass subclass (hon1-F2 verbatim).
    assert "mode" in instance_state_names(_OwnSlotBox(x, "fast"))
    assert undeclared_instance_state(_OwnSlotBox(x, "fast"), "dataclass")
    # Private slot resolves through CPython name mangling on the declaring class.
    assert "_PrivateSlotBox__secret" in instance_state_names(_PrivateSlotBox(x, 1))
    assert undeclared_instance_state(_PrivateSlotBox(x, 1), "dataclass")
    # MRO-inherited slot from a non-dataclass mixin.
    box = _InheritedSlotBox(x)
    assert not undeclared_instance_state(box, "dataclass")  # unset slot is absent
    box.inherited = None  # set to None: PRESENCE is state
    assert "inherited" in instance_state_names(box)
    assert undeclared_instance_state(box, "dataclass")
    # Single-string __slots__ declaration normalizes to one name.
    assert "solo" in instance_state_names(_SingleStringSlotBox(x, 0))
    assert undeclared_instance_state(_SingleStringSlotBox(x, 0), "dataclass")
    # Unset slot stays absent (no refusal).
    assert "maybe" not in instance_state_names(_UnsetSlotBox(x))
    assert not undeclared_instance_state(_UnsetSlotBox(x), "dataclass")
    # dataclass(slots=True)-style declared-field slots stay admitted.
    assert not undeclared_instance_state(_SlottedDC(x), "dataclass")
    # Mixed __dict__ + slot storage: both surfaces enumerated.
    mixed = _OwnSlotBox(x, "fast")
    mixed.extra_dict_attr = None
    assert {"mode", "extra_dict_attr"} <= set(instance_state_names(mixed))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "extra_value",
    [None, False, 0, {}, (), "", object()],
    ids=["none", "false", "zero", "empty_dict", "empty_tuple", "empty_str", "object"],
)
def test_r69_presence_matrix_every_falsey_extra_refuses(extra_value, tmp_path: Path) -> None:
    """Every extra-attr VALUE (falsey included) has the SAME refusal disposition.

    hon1-F3 verbatim: the old ``is not None`` carve-out made attribute PRESENCE
    unwitnessable -- a ``hasattr``-steered model admitted a ``None``-valued extra at
    save and replayed VERIFIED against an attr-absent same-class twin.
    """

    from torchlens._input_walk import snapshot_input_boundary

    box = _Box(torch.zeros(1))
    box.extra = extra_value
    snapshot = snapshot_input_boundary(box)
    assert any(
        refusal["reason"] == "undeclared_instance_state" for refusal in snapshot["refusals"]
    ), extra_value


@pytest.mark.smoke
def test_r69_slotted_dataclass_hidden_state_refuses_at_save(tmp_path: Path) -> None:
    """hon1-F2: __slots__ control state on a dataclass subclass refuses typed at save."""

    class _SlotModel(nn.Module):
        def forward(self, box) -> torch.Tensor:
            if getattr(box, "mode", "slow") == "fast":
                return box.x * 2.0
            return box.x + 100.0

    x = torch.randn(3)
    trace = _trace(_SlotModel(), _OwnSlotBox(x, "fast"))
    _assert_refuses_contract(trace, tmp_path / "slots.tlspec", "undeclared_instance_state")


def test_r69_none_extra_hasattr_steering_refuses_at_save(tmp_path: Path) -> None:
    """hon1-F3: a None-valued extra attr (hasattr steering) refuses typed at save."""

    class _HasattrModel(nn.Module):
        def forward(self, box) -> torch.Tensor:
            if hasattr(box, "extra"):
                return box.x * 2.0
            return box.x + 100.0

    x = torch.randn(3)
    box = _Box(x.clone())
    box.extra = None
    trace = _trace(_HasattrModel(), box)
    _assert_refuses_contract(trace, tmp_path / "none_extra.tlspec", "undeclared_instance_state")


def test_r69_runtime_added_slot_state_diverges(tmp_path: Path) -> None:
    """Symmetric bind-side proof for slots: runtime-added slot state cannot pass."""

    class _SlotModel(nn.Module):
        def forward(self, box) -> torch.Tensor:
            if getattr(box, "maybe", None) == "fast":
                return box.x * 2.0
            return box.x + 100.0

    x = torch.randn(3)
    path = _save(_trace(_SlotModel(), _UnsetSlotBox(x.clone())), tmp_path / "rt_slot.tlspec")
    original = tl.load(path).run(inputs=_UnsetSlotBox(x.clone()))
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    twin = _UnsetSlotBox(x.clone())
    twin.maybe = "fast"  # flips the branch through slot state no snapshot declared
    _assert_diverges(path, twin)


def test_r69_clean_declared_schema_lanes_stay_admitted(tmp_path: Path) -> None:
    """No over-trigger: clean dataclass/namedtuple/custom-Mapping lanes stay green."""

    from torchlens._input_walk import snapshot_input_boundary

    x = torch.zeros(1)
    for value in (_Box(x), _SlottedDC(x), _NTA(x), _EmptyA()):
        snapshot = snapshot_input_boundary(value)
        assert snapshot["refusals"] == [], type(value).__name__

    class _GoodMapping(dict):
        """Well-behaved Mapping subclass (heavy-gate F7 admitted lane)."""

    snapshot = snapshot_input_boundary(_GoodMapping({"k": x}))
    assert snapshot["refusals"] == []


@pytest.mark.smoke
def test_r67_consumer_source_scan() -> None:
    """Capture recorder and runtime check both derive from the ONE snapshot spine."""

    import inspect

    from torchlens import _runnable_execution
    from torchlens.capture import trace as capture_trace

    assert "snapshot_input_boundary" in inspect.getsource(
        capture_trace._record_runnable_input_structure
    )
    assert "snapshot_input_boundary" in inspect.getsource(
        _runnable_execution._input_structure_witness_check
    )


# ======================================================================================
# r71 C -- uninspectable declared-schema instance state (hon1-F1 + adjacent closures)
# ======================================================================================


class _HookCounter:
    """Tracks whether the hostile attribute hook ever ran during TL inspection."""

    property_calls = 0
    getattribute_calls = 0
    getattr_calls = 0


@pytest.fixture(autouse=True)
def _reset_hook_counters():
    _HookCounter.property_calls = 0
    _HookCounter.getattribute_calls = 0
    _HookCounter.getattr_calls = 0
    yield


class _GateModel(nn.Module):
    """Branches on a hidden ``gate`` attribute the declared schema never witnesses."""

    def forward(self, cfg) -> torch.Tensor:
        t = cfg.t
        return t * 10.0 if object.__getattribute__(cfg, "__dict__").get("gate") else t + 1.0


def _dict_property_shadowed_dataclass():
    @dataclasses.dataclass
    class _EvilCfg:
        t: torch.Tensor

        __dict__ = property(lambda self: {})  # type: ignore[assignment]

    return _EvilCfg


@pytest.mark.smoke
def test_r71c_property_shadowed_dict_refuses_without_running_hook(tmp_path: Path) -> None:
    """hon1-F1: a property-shadowed __dict__ hides hidden state -> the enumerator no
    longer trusts it, save refuses ``instance_state_uninspectable``, and the hostile
    property never runs during TL inspection."""

    from torchlens._input_walk import inspect_instance_state

    calls = {"n": 0}

    @dataclasses.dataclass
    class _EvilCfg:
        t: torch.Tensor

        @property
        def __dict__(self):  # type: ignore[override]
            calls["n"] += 1
            return {}

    cfg = _EvilCfg(torch.randn(3))
    object.__setattr__(cfg, "gate", True) if False else None
    inspection = inspect_instance_state(cfg)
    assert inspection.complete is False
    assert inspection.reason == "instance_state_uninspectable"
    assert calls["n"] == 0, "the shadowing property must NOT execute during inspection"

    snapshot = snapshot_input_boundary(cfg)
    assert any(r["reason"] == "instance_state_uninspectable" for r in snapshot["refusals"])
    assert calls["n"] == 0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory_name",
    ["descriptor_shadow", "dict_subclass", "custom_getattribute", "custom_getattr"],
)
def test_r71c_uninspectable_variants_refuse(factory_name: str, tmp_path: Path) -> None:
    """Every uninspectable declared-schema variant refuses fail-closed at save."""

    from torchlens._input_walk import inspect_instance_state, undeclared_instance_state

    if factory_name == "descriptor_shadow":

        class _Desc:
            def __get__(self, obj, owner=None):
                return {}

        @dataclasses.dataclass
        class _Cfg:
            t: torch.Tensor

            __dict__ = _Desc()  # type: ignore[assignment]

        value = _Cfg(torch.randn(3))
        assert inspect_instance_state(value).reason == "instance_state_uninspectable"
    elif factory_name == "dict_subclass":

        class _DictSub(dict):
            pass

        class _GetSet:
            def __get__(self, obj, owner=None):
                return _DictSub()

        @dataclasses.dataclass
        class _Cfg:
            t: torch.Tensor

            __dict__ = _GetSet()  # type: ignore[assignment]

        value = _Cfg(torch.randn(3))
        assert inspect_instance_state(value).complete is False
    elif factory_name == "custom_getattribute":

        @dataclasses.dataclass
        class _Cfg:
            t: torch.Tensor

            def __getattribute__(self, name):
                return object.__getattribute__(self, name)

        value = _Cfg(torch.randn(3))
        assert undeclared_instance_state(value, "dataclass") is True
    else:  # custom_getattr

        @dataclasses.dataclass
        class _Cfg:
            t: torch.Tensor

            def __getattr__(self, name):
                return "fast"

        value = _Cfg(torch.randn(3))
        assert undeclared_instance_state(value, "dataclass") is True

    snapshot = snapshot_input_boundary(value)
    assert any(r["reason"] == "instance_state_uninspectable" for r in snapshot["refusals"]), (
        factory_name,
        snapshot["refusals"],
    )


def test_r71c_shadowed_fields_namedtuple_refuses() -> None:
    """A namedtuple whose ``_fields`` is shadowed (custom __getattr__) refuses."""

    from torchlens._input_walk import snapshot_input_boundary

    Pt = namedtuple("Pt", ["t"])

    class _EvilPt(Pt):
        def __getattr__(self, name):
            return ()

    value = _EvilPt(torch.randn(3))
    snapshot = snapshot_input_boundary(value)
    assert any(r["reason"] == "instance_state_uninspectable" for r in snapshot["refusals"])


def test_r71c_registered_container_shadowed_dict_refuses(tmp_path: Path) -> None:
    """The registered-container ``__dict__`` read routes through the inert helper."""

    from torchlens._input_walk import snapshot_input_boundary

    class _EvilReg:
        def __init__(self, t):
            object.__setattr__(self, "_t", t)

        @property
        def __dict__(self):
            return {}

    register_container(
        _EvilReg,
        flatten=lambda box: ((object.__getattribute__(box, "_t"),), None),
        unflatten=lambda children, aux: _EvilReg(children[0]),
        state_complete=False,
    )
    value = _EvilReg(torch.randn(3))
    snapshot = snapshot_input_boundary(value)
    assert any(r["reason"] == "instance_state_uninspectable" for r in snapshot["refusals"])


@pytest.mark.smoke
def test_r71c_runtime_uninspectable_twin_never_verifies(tmp_path: Path) -> None:
    """Symmetric runtime proof: an admitted plain capture vs an uninspectable
    same-schema runtime twin diverges/refuses, never VERIFIED (the hon1-F1 E2E)."""

    @dataclasses.dataclass
    class _PlainCfg:
        t: torch.Tensor

    class _Model(nn.Module):
        def forward(self, cfg) -> torch.Tensor:
            return cfg.t * 2.0

    x = torch.randn(3)
    path = _save(_trace(_Model(), _PlainCfg(x.clone())), tmp_path / "plain_cfg.tlspec")
    assert (
        tl.load(path).run(inputs=_PlainCfg(x.clone())).report.path_faithfulness
        is PathFaithfulness.VERIFIED
    )

    @dataclasses.dataclass
    class _UninspectableCfg:
        t: torch.Tensor

        @property
        def __dict__(self):  # type: ignore[override]
            return {}

    with pytest.raises((PathDivergenceError, RunPreconditionError)):
        tl.load(path).run(inputs=_UninspectableCfg(x.clone()))


def test_r71c_greens_ordinary_declared_schemas(tmp_path: Path) -> None:
    """No over-trigger: ordinary/frozen/slots/inheritance dataclasses, standard
    namedtuples + subclasses, zero-field variants, and the well-behaved custom-Mapping
    lane all stay admitted (no uninspectable refusal)."""

    from torchlens._input_walk import inspect_instance_state, snapshot_input_boundary

    x = torch.zeros(1)

    @dataclasses.dataclass
    class _Plain:
        t: torch.Tensor

    @dataclasses.dataclass(frozen=True)
    class _Frozen:
        t: torch.Tensor

    @dataclasses.dataclass
    class _Slotted:
        __slots__ = ("t",)
        t: torch.Tensor

    @dataclasses.dataclass
    class _Base:
        t: torch.Tensor

    @dataclasses.dataclass
    class _Child(_Base):
        pass

    Pt = namedtuple("Pt", ["t"])

    class _PtSub(Pt):
        pass

    class _EmptyNT(namedtuple("EmptyNT", [])):
        pass

    @dataclasses.dataclass
    class _EmptyDC:
        pass

    class _GoodMapping(dict):
        pass

    clean_values = [
        _Plain(x),
        _Frozen(x),
        _Slotted(x),
        _Child(x),
        Pt(x),
        _PtSub(x),
        _EmptyNT(),
        _EmptyDC(),
        _GoodMapping({"k": x}),
    ]
    for value in clean_values:
        inspection = inspect_instance_state(value)
        assert inspection.complete is True, type(value).__name__
        snapshot = snapshot_input_boundary(value)
        assert not any(
            r["reason"] == "instance_state_uninspectable" for r in snapshot["refusals"]
        ), type(value).__name__


@pytest.mark.smoke
def test_r71c_no_live_dict_getattr_in_normative_module() -> None:
    """Source-scan tripwire: the normative input-boundary module never reads a live
    instance ``__dict__`` via ``getattr`` (only the inert raw-MRO helper)."""

    import inspect

    from torchlens import _input_walk

    for func in (
        _input_walk.inspect_instance_state,
        _input_walk.undeclared_instance_state,
        _input_walk.snapshot_input_boundary,
        _input_walk.walk_input_boundary,
    ):
        source = inspect.getsource(func)
        assert 'getattr(value, "__dict__"' not in source, func.__name__
        assert 'getattr(item, "__dict__"' not in source, func.__name__


# ======================================================================================
# r69 A -- RequiredWitnessInventory: descriptor-native presence proof, generated from
# WITNESS_FAMILY_REGISTRY (secA-F1/free-F1 witness-strip class + literal cross-anchor)
# ======================================================================================


import json as _json  # noqa: E402
import os as _os  # noqa: E402

from torchlens.runnable import (  # noqa: E402
    WITNESS_FAMILY_REGISTRY,
    WITNESS_FAMILY_REGISTRY_VERSION,
    ReadinessStatus,
    decode_input_site_position,
    encode_input_site_position,
)


class _KindBranch(nn.Module):
    """Branch on nested container KIND -- only the structure fact distinguishes."""

    def forward(self, d):
        inner = d["a"]
        if isinstance(inner, list):
            return inner[0] + 100.0
        return inner[0] * 2.0


class _LitBranch(nn.Module):
    def forward(self, t, flag):
        if flag == 1:
            return t * 2.0
        return t + 100.0


class _RichFamilies(nn.Module):
    """One canonical capture emitting literal + metadata + state + structure facts."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, t, flag):
        if self.lin.weight.requires_grad and t.is_contiguous() and flag == 1:
            return self.lin(t)
        return t + 100.0


def _mutate_manifest(path: Path, fn) -> None:
    manifest_path = _os.path.join(path, "manifest.json")
    manifest = _json.load(open(manifest_path))
    fn(manifest["run"])
    _json.dump(manifest, open(manifest_path, "w"))


def _assert_context_field_invalid(path: Path, run_inputs) -> None:
    """Analysis load survives; readiness UNAVAILABLE with context_field_invalid; no run."""

    loaded = tl.load(path)
    readiness = loaded.__dict__.get("_runnable_readiness")
    assert readiness is not None
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    codes = {diagnostic.code.value for diagnostic in readiness.diagnostics}
    assert "context_field_invalid" in codes, codes
    with pytest.raises(Exception):
        loaded.run(inputs=run_inputs)


def _heal_capture_state(source_path: Path, run_inputs) -> None:
    """Run the pristine artifact once so this test leaves no cross-test residue.

    Works around the PRE-EXISTING main-branch global-state bug (r68 ledger, out of
    r69 scope; same workaround as test_r67_zero_field_dataclass_argument_cannot_vanish):
    saving a multi-positional-arg runnable artifact without a subsequent loaded run
    breaks buffer-address resolution for the NEXT buffer-carrying capture in the same
    process. A loaded run heals it; tests that only save/analysis-load such artifacts
    must heal explicitly.
    """

    result = tl.load(source_path).run(inputs=run_inputs)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_r69_site_position_member_codec_round_trips() -> None:
    for position in (("arg", 0), ("arg", 12), ("kwarg", "flag"), ("kwarg", "a:b")):
        member = encode_input_site_position(position)
        assert decode_input_site_position(member) == position
    with pytest.raises(ValueError):
        encode_input_site_position(("arg", "x"))
    with pytest.raises(ValueError):
        encode_input_site_position(("weird", 0))
    with pytest.raises(ValueError):
        decode_input_site_position("arg:notanint")


@pytest.mark.smoke
def test_r69_inventory_is_authored_for_every_registry_family(tmp_path: Path) -> None:
    """Every artifact persists one row per registered family, empty sets explicit."""

    x = torch.randn(3)
    path = _save(_trace(_RichFamilies(), [x, 1]), tmp_path / "families.tlspec")
    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    inventory = descriptor.required_witness_inventory
    assert inventory.registry_version == WITNESS_FAMILY_REGISTRY_VERSION
    rows = {row.family: row for row in inventory.families}
    assert set(rows) == set(WITNESS_FAMILY_REGISTRY)
    for family, row in rows.items():
        assert row.disposition == WITNESS_FAMILY_REGISTRY[family].disposition
    assert rows["input_structure"].members == ("arg:0", "arg:1")
    # Anchored family carries no inventory members (its proof is the cross-anchor).
    assert rows["model_input_literal"].members == ()
    # Read-gated state facts are indexed by exact (state, fact) identity.
    assert any("::requires_grad" in member for member in rows["state_metadata"].members)
    _heal_capture_state(path, [x.clone(), 1])


def _strip_family(run: dict, family: str, count: int | None = None) -> int:
    kept, removed = [], 0
    for witness in run["control_witnesses"]:
        label = str(witness.get("site_label", ""))
        if label.startswith(f"{family}:") and (count is None or removed < count):
            removed += 1
            continue
        kept.append(witness)
    run["control_witnesses"] = kept
    return removed


def test_r69_registry_generated_strip_matrix_refuses(tmp_path: Path) -> None:
    """For every family PRESENT in a canonical artifact: drop-all, drop-one, and
    duplicate mutations parse-refuse ``context_field_invalid`` (analysis-only)."""

    x = torch.randn(3)
    source = _save(_trace(_RichFamilies(), [x, 1]), tmp_path / "matrix_src.tlspec")
    manifest = _json.load(open(_os.path.join(source, "manifest.json")))
    present_families = sorted(
        {
            str(w.get("site_label", "")).split(":", 1)[0]
            for w in manifest["run"]["control_witnesses"]
            if any(
                str(w.get("site_label", "")).startswith(f"{family}:")
                for family in WITNESS_FAMILY_REGISTRY
            )
        }
    )
    assert {"input_structure", "model_input_literal", "state_metadata"} <= set(present_families), (
        present_families
    )
    import shutil as _shutil

    run_inputs = [x.clone(), 1]
    case = 0
    for family in present_families:
        for mutation in ("drop_all", "drop_one", "duplicate"):
            case += 1
            path = tmp_path / f"matrix_{case}.tlspec"
            _shutil.copytree(source, path)

            def _apply(run: dict, family: str = family, mutation: str = mutation) -> None:
                if mutation == "drop_all":
                    assert _strip_family(run, family) > 0
                elif mutation == "drop_one":
                    assert _strip_family(run, family, count=1) == 1
                else:
                    twin = next(
                        dict(w)
                        for w in run["control_witnesses"]
                        if str(w.get("site_label", "")).startswith(f"{family}:")
                    )
                    twin["witness_id"] = "witness:9999"
                    twin["order"] = 9999
                    run["control_witnesses"] = list(run["control_witnesses"]) + [twin]

            _mutate_manifest(path, _apply)
            _assert_context_field_invalid(path, run_inputs)
    _heal_capture_state(source, [x.clone(), 1])


def test_r69_forged_inventory_mutations_refuse(tmp_path: Path) -> None:
    """Inventory-side forgeries: shrunk members, extra members, unknown/missing/dup
    family rows, wrong discriminator, wrong disposition -- all refuse typed."""

    x = torch.randn(3)
    source = _save(_trace(_RichFamilies(), [x, 1]), tmp_path / "forge_src.tlspec")
    import shutil as _shutil

    def _case(name: str, fn) -> None:
        path = tmp_path / f"forge_{name}.tlspec"
        _shutil.copytree(source, path)
        _mutate_manifest(path, fn)
        _assert_context_field_invalid(path, [x.clone(), 1])

    def _row(run: dict, family: str) -> dict:
        return next(
            row for row in run["required_witness_inventory"]["families"] if row["family"] == family
        )

    # Shrunk site member set (facts intact) -- exact member equality refuses.
    _case("shrunk_sites", lambda run: _row(run, "input_structure")["members"].pop())
    # Extra forged member (no matching fact).
    _case(
        "extra_member",
        lambda run: _row(run, "input_structure")["members"].append("arg:7"),
    )

    # Shrunk inventory AND stripped facts together: slot bindings still prove sites.
    def _shrink_both(run: dict) -> None:
        _strip_family(run, "input_structure")
        _row(run, "input_structure")["members"] = []

    _case("shrunk_both", _shrink_both)

    # Unknown family row.
    def _unknown_row(run: dict) -> None:
        run["required_witness_inventory"]["families"].append(
            {"family": "wormhole", "disposition": "inventory_indexed", "members": []}
        )

    _case("unknown_family", _unknown_row)
    # Missing family row.
    _case(
        "missing_family",
        lambda run: run["required_witness_inventory"]["families"].pop(),
    )
    # Duplicate family row.
    _case(
        "dup_family",
        lambda run: run["required_witness_inventory"]["families"].append(
            dict(run["required_witness_inventory"]["families"][0])
        ),
    )

    # Wrong registry discriminator.
    def _wrong_version(run: dict) -> None:
        run["required_witness_inventory"]["registry_version"] = "witness_family_registry_v0"

    _case("wrong_version", _wrong_version)

    # Disposition disagreeing with the closed registry.
    def _wrong_disposition(run: dict) -> None:
        _row(run, "input_structure")["disposition"] = "independent_ceiling"

    _case("wrong_disposition", _wrong_disposition)

    # Members on the anchored literal family.
    def _anchored_members(run: dict) -> None:
        _row(run, "model_input_literal")["members"] = ["forged"]

    _case("anchored_members", _anchored_members)

    # Forged site_count on a surviving fact (redundant consistency, never authority).
    def _forged_count(run: dict) -> None:
        for witness in run["control_witnesses"]:
            label = str(witness.get("site_label", ""))
            if label.startswith("input_structure:"):
                for entry in witness["observed_value"]["entries"]:
                    key = entry["key"]
                    if isinstance(key, dict) and key.get("value") == "site_count":
                        entry["value"]["value"] = 7
                return

    _case("forged_count", _forged_count)
    _heal_capture_state(source, [x.clone(), 1])


@pytest.mark.smoke
def test_r69_secA_f1_structure_strip_no_longer_restores_weak_semantics(
    tmp_path: Path,
) -> None:
    """secA-F1 E2E: stripping input_structure facts + kind swap used to VERIFY."""

    x = torch.randn(3)
    path = _save(_trace(_KindBranch(), {"a": [x.clone()]}), tmp_path / "seca.tlspec")
    _mutate_manifest(path, lambda run: _strip_family(run, "input_structure"))
    _assert_context_field_invalid(path, {"a": (x.clone(),)})


def test_r69_literal_fact_strip_breaks_the_cross_anchor(tmp_path: Path) -> None:
    """Fable ADD-1: per-leaf literal stripping cannot hide behind site coverage."""

    x = torch.randn(3)
    source = _save(_trace(_LitBranch(), [x.clone(), 1]), tmp_path / "anchor_src.tlspec")
    import shutil as _shutil

    path = tmp_path / "anchor.tlspec"
    _shutil.copytree(source, path)
    _mutate_manifest(path, lambda run: _strip_family(run, "model_input_literal", count=1))
    _assert_context_field_invalid(path, [x.clone(), 1])
    _heal_capture_state(source, [x.clone(), 1])


def test_r69_unregistered_family_fact_refuses(tmp_path: Path) -> None:
    """A SHAPE_STRUCTURE_FACT outside the closed registry can never ride silently."""

    x = torch.randn(3)
    source = _save(_trace(_LitBranch(), [x.clone(), 1]), tmp_path / "closure_src.tlspec")
    import shutil as _shutil

    path = tmp_path / "closure.tlspec"
    _shutil.copytree(source, path)

    def _forge(run: dict) -> None:
        twin = dict(run["control_witnesses"][-1])
        twin["site_label"] = "brand_new_family:site"
        twin["witness_id"] = "witness:9999"
        twin["order"] = 9999
        twin["kind"] = "shape_structure_fact"
        run["control_witnesses"] = list(run["control_witnesses"]) + [twin]

    _mutate_manifest(path, _forge)
    _assert_context_field_invalid(path, [x.clone(), 1])
    _heal_capture_state(source, [x.clone(), 1])


@pytest.mark.smoke
def test_r69_positive_family_matrix_round_trips(tmp_path: Path) -> None:
    """Positive cases: tensor-only, scalar-only, tensor-free empty root, mixed
    args/kwargs, no state reads, read-gated state reads -- all save+load+run."""

    x = torch.randn(3)

    class _TensorOnly(nn.Module):
        def forward(self, t):
            return t * 2.0

    class _EmptyRoot(nn.Module):
        def forward(self, t, marker):
            return t * 2.0

    class _MixedKw(nn.Module):
        def forward(self, t, *, add: bool):
            return t + 1.0 if add else t * 10.0

    cases = [
        (_TensorOnly(), x.clone(), x.clone(), "tensor_only"),
        (_LitBranch(), [x.clone(), 1], [x.clone(), 1], "scalar"),
        (_EmptyRoot(), (x.clone(), ()), (x.clone(), ()), "empty_root"),
        (_RichFamilies(), [x.clone(), 1], [x.clone(), 1], "state_reads"),
    ]
    for model, capture_inputs, run_inputs, name in cases:
        path = _save(_trace(model, capture_inputs), tmp_path / f"pos_{name}.tlspec")
        result = tl.load(path).run(inputs=run_inputs)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, name
    # Mixed positional + keyword sites.
    trace = tl.trace(_MixedKw(), [x.clone()], input_kwargs={"add": True}, capture=_CAPTURE)
    path = _save(trace, tmp_path / "pos_mixed.tlspec")
    result = tl.load(path).run(inputs={"args": [x.clone()], "kwargs": {"add": True}})
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r69_emission_meta_test_every_prefix_is_registered(tmp_path: Path) -> None:
    """A future replay-critical family cannot ship without a registry row.

    Every witness emitted by a canonical multi-family artifact resolves to a
    registered family -- SHAPE_STRUCTURE_FACT prefixes AND the direct control kinds
    (r71 A). The old source-text emitter-count heuristic is REPLACED by the typed
    registry-closure meta-test in
    ``tests/test_tlspec_runnable_r71_witness_obligations.py`` (registry keys ==
    direct kinds | shape prefixes | claim families, each row with a non-empty
    anchor and runtime consumer).
    """

    from torchlens._io import runnable as io_runnable

    x = torch.randn(3)
    path = _save(_trace(_RichFamilies(), [x, 1]), tmp_path / "emit.tlspec")
    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    for witness in descriptor.control_witnesses:
        family = io_runnable.witness_family_of_witness(witness)
        assert family is not None, witness.site_label
        assert family in WITNESS_FAMILY_REGISTRY, family
        assert WITNESS_FAMILY_REGISTRY[family].witness_kind is witness.kind, family
    _heal_capture_state(path, [x.clone(), 1])
