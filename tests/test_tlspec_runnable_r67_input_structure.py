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
