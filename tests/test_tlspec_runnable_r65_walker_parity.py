"""r65 Cluster Y: input-boundary walker container parity (single-sourced dispatch).

The three input-boundary walkers -- W1 (capture literal-leaf), W2 (capture
metadata-read-site), W3 (runtime literal-path) -- witness the SAME physical input
tree for different fact families, so a container kind handled by one walker but
missed by another silently drops a whole fact family for every leaf beneath it.
That is the r64 Finding-1 false-VERIFIED: W2 did not descend dataclass containers,
so ``box.x.is_contiguous()`` on a dataclass input field recorded no metadata
witness and a same-value non-contiguous twin replayed the captured branch as
VERIFIED. These tests pin:

1. the single-sourced dispatch STRUCTURALLY (source-scan meta-test: no private
   container dispatch inside any walker body -- a future container kind must land
   in ``torchlens/_input_walk.py`` and reaches every walker in lockstep),
2. container-kind parity across all three walkers BEHAVIORALLY (identical
   supported set incl. dataclass/namedtuple/dict/list/tuple/empty/opaque),
3. the declared DUAL mapping-key vocabulary (tagged literal vs raw fact-site,
   residual R6) exactly as declared in the one module,
4. the r64 repro end-to-end (dataclass metadata read witnessed; layout twin
   diverges under the default policy, never VERIFIED) and zero over-trigger
   (tensor-only dataclass with no metadata read stays VERIFIED + ATTESTED),
5. behavioral round-trip parity of the NON-rerouted W4/W5/W6 walkers against the
   shared dispatch on a torture container.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import textwrap
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_walk import (
    INPUT_CONTAINER_KINDS,
    classify_input_container,
    raw_mapping_key_component,
    tagged_mapping_key_component,
)
from torchlens._io.runnable import (
    BOOL_KEY_PATH_TAG,
    EMPTY_CONTAINER_PATH_MARKER,
    _UnsupportedLiteralError,
)
from torchlens._runnable_execution import (
    _container_kind,
    _container_leaf_paths,
    _runtime_nontensor_leaf_paths,
    _tensor_leaf_paths,
    _value_at_path,
)
from torchlens.capture.trace import (
    _OPAQUE_INPUT_LEAF,
    _record_runnable_input_literal_leaves,
    _record_runnable_input_tensor_sites,
)
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)

# ======================================================================================
# Shared fixtures: probe containers spanning the closed container-kind vocabulary
# ======================================================================================


class _Key(enum.Enum):
    """A mapping key outside the frozen literal grammar (non-representable)."""

    K = "k"


class _ProbeNT(NamedTuple):
    """Namedtuple probe carrying one tensor leaf and one scalar leaf."""

    t: torch.Tensor
    s: int


class _EmptyNT(NamedTuple):
    """Zero-field namedtuple (classified as an EMPTY container by kind)."""


@dataclasses.dataclass
class _ProbeBox:
    """Dataclass probe carrying one tensor leaf and one scalar leaf."""

    t: torch.Tensor
    s: int


class _FakeTrace:
    """Minimal weak-referenceable Trace stand-in for standalone walker runs."""

    intervention_ready = True


def _w1_leaves(root: Any) -> tuple[tuple[object, tuple[Any, ...], Any], ...]:
    """Run the capture literal-leaf walker (W1) standalone on one positional root."""

    trace = _FakeTrace()
    _record_runnable_input_literal_leaves(trace, [root], {})
    return trace.__dict__.get("_runnable_input_nontensor_leaves", ())


def _w1_paths(root: Any) -> set[tuple[Any, ...]]:
    """Return W1's recorded non-tensor leaf-path set for one positional root."""

    return {path for _position, path, _leaf in _w1_leaves(root)}


def _w2_paths(root: Any) -> set[tuple[Any, ...]]:
    """Run the capture metadata-site walker (W2) standalone; return its site paths."""

    trace = _FakeTrace()
    _record_runnable_input_tensor_sites(trace, [root], {})
    sites = trace.__dict__.get("_runnable_input_tensor_sites", {})
    return {path for _position, path in sites.values()}


# ======================================================================================
# 1. Source-scan meta-test: no private container dispatch inside any walker body
# ======================================================================================

_WALKER_FUNCTIONS = {
    "W1_capture_literal": _record_runnable_input_literal_leaves,
    "W2_capture_metadata_site": _record_runnable_input_tensor_sites,
    "W3_runtime_literal_path": _runtime_nontensor_leaf_paths,
}

#: Identifiers that constitute PRIVATE container dispatch. A walker body referencing
#: any of these is re-growing its own traversal -- the exact drift class that produced
#: the r64 false-VERIFIED. Container handling must live in torchlens/_input_walk.py.
_FORBIDDEN_DISPATCH_IDENTIFIERS = frozenset(
    {
        "is_dataclass",
        "fields",
        "_fields",
        "Mapping",
        "_Mapping",
        "Tensor",
        "empty_container_kind",
        "_encode_literal_key",
        "input_path_key_component",
    }
)


def _function_identifiers(func: Any) -> set[str]:
    """Collect every Name/Attribute/import identifier referenced in a function's AST.

    Docstrings and comments are inherently excluded (they are not Name nodes), so the
    scan cannot false-fire on prose mentioning a container kind.
    """

    source = textwrap.dedent(inspect.getsource(func))
    identifiers: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.alias):
            identifiers.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                identifiers.add(node.asname)
    return identifiers


@pytest.mark.smoke
@pytest.mark.parametrize("walker_name", sorted(_WALKER_FUNCTIONS))
def test_walker_bodies_have_no_private_container_dispatch(walker_name: str) -> None:
    """T-Y1: every input-boundary walker routes through the shared traversal ONLY."""

    identifiers = _function_identifiers(_WALKER_FUNCTIONS[walker_name])
    leaked = identifiers & _FORBIDDEN_DISPATCH_IDENTIFIERS
    assert not leaked, (
        f"{walker_name} performs private container dispatch via {sorted(leaked)}; "
        "container handling must be added to torchlens/_input_walk.py so every "
        "input-boundary walker gains it in lockstep (r64 Finding-1 drift class)."
    )
    assert "walk_input_boundary" in identifiers, (
        f"{walker_name} no longer routes through torchlens._input_walk."
        "walk_input_boundary -- the single-sourced container dispatch is mandatory."
    )


# ======================================================================================
# 2. Closed container-kind vocabulary + classification order pins
# ======================================================================================


@pytest.mark.smoke
def test_container_kind_vocabulary_is_closed_and_classified() -> None:
    """The normative dispatch classifies one probe per kind; the vocabulary is closed."""

    probes: dict[str, Any] = {
        "tensor": torch.zeros(1),
        "empty": {},
        "namedtuple": _ProbeNT(t=torch.zeros(1), s=5),
        "dataclass": _ProbeBox(t=torch.zeros(1), s=5),
        "mapping": {"k": 1},
        "sequence": [1],
        "leaf": object(),
    }
    for kind, value in probes.items():
        assert classify_input_container(value) == kind
    assert set(probes) == set(INPUT_CONTAINER_KINDS)
    # Order pins: EMPTY precedes namedtuple (a zero-field namedtuple is witnessed as an
    # empty container by KIND); a dataclass TYPE (not instance) is an opaque leaf.
    assert classify_input_container(_EmptyNT()) == "empty"
    assert classify_input_container(()) == "empty"
    assert classify_input_container([]) == "empty"
    assert classify_input_container(_ProbeBox) == "leaf"


# ======================================================================================
# 3. Three-walker container-kind parity (behavioral) + dual key vocabulary pins
# ======================================================================================

_PARITY_CASES: dict[str, Any] = {
    "namedtuple": lambda t: _ProbeNT(t=t, s=5),
    "dataclass": lambda t: _ProbeBox(t=t, s=5),
    "mapping": lambda t: {"t": t, "s": 5},
    "sequence_list": lambda t: [t, 5],
    "sequence_tuple": lambda t: (t, 5),
}
_EXPECTED_TENSOR_PATHS: dict[str, set[tuple[Any, ...]]] = {
    "namedtuple": {("t",)},
    "dataclass": {("t",)},
    "mapping": {("t",)},
    "sequence_list": {(0,)},
    "sequence_tuple": {(0,)},
}
_EXPECTED_LEAF_PATHS: dict[str, set[tuple[Any, ...]]] = {
    "namedtuple": {("s",)},
    "dataclass": {("s",)},
    "mapping": {("s",)},
    "sequence_list": {(1,)},
    "sequence_tuple": {(1,)},
}


@pytest.mark.smoke
@pytest.mark.parametrize("kind", sorted(_PARITY_CASES))
def test_three_walker_container_kind_parity(kind: str) -> None:
    """Every supported container kind is descended by ALL THREE walkers.

    The r64 bug class is a kind descended by the literal walkers but missed by the
    metadata-site walker: the literal fact family stays green while every metadata
    read beneath the container silently loses its witness.
    """

    tensor = torch.zeros(2)
    root = _PARITY_CASES[kind](tensor)
    w1 = _w1_paths(root)
    w3 = _runtime_nontensor_leaf_paths(root)
    assert w1 == _EXPECTED_LEAF_PATHS[kind]
    assert w3 == w1, "capture/runtime literal walkers must agree on the same object"
    w2 = _w2_paths(root)
    assert w2 == _EXPECTED_TENSOR_PATHS[kind], (
        f"metadata-site walker failed to descend {kind!r} -- the r64 Finding-1 class"
    )


@pytest.mark.smoke
def test_empty_container_and_opaque_leaf_parity() -> None:
    """EMPTY containers and opaque leaves are handled consistently by all walkers."""

    for empty in ({}, [], ()):
        assert _w1_paths([empty]) == {(0, EMPTY_CONTAINER_PATH_MARKER)}
        assert _runtime_nontensor_leaf_paths([empty]) == {(0, EMPTY_CONTAINER_PATH_MARKER)}
        assert _w2_paths([empty]) == set()  # nothing to index inside an empty container
    opaque = object()
    assert _w1_paths(opaque) == {()}
    assert _runtime_nontensor_leaf_paths(opaque) == {()}
    assert _w2_paths(opaque) == set()


@pytest.mark.smoke
def test_dual_mapping_key_vocabulary_declared_and_pinned() -> None:
    """R6 pin: the two mapping-key vocabularies behave exactly as declared.

    TAGGED (persisted literal, W1/W3): grammar-gated, bool keys type-distinct.
    RAW (metadata fact sites, W2): every key accepted verbatim; bool/int conflation is
    shielded by the r33 ``_type_strict_path`` symmetric input-tree belt.
    """

    assert tagged_mapping_key_component(True) == (BOOL_KEY_PATH_TAG, True)
    assert tagged_mapping_key_component(False) == (BOOL_KEY_PATH_TAG, False)
    assert tagged_mapping_key_component(1) == 1
    assert tagged_mapping_key_component("k") == "k"
    with pytest.raises(_UnsupportedLiteralError):
        tagged_mapping_key_component(_Key.K)
    with pytest.raises(_UnsupportedLiteralError):
        tagged_mapping_key_component(object())
    with pytest.raises(_UnsupportedLiteralError):
        tagged_mapping_key_component(b"kk")

    assert raw_mapping_key_component(True) is True
    assert raw_mapping_key_component(1) == 1
    assert raw_mapping_key_component(_Key.K) is _Key.K
    assert raw_mapping_key_component(b"kk") == b"kk"

    # Observable split: W1 persists the tagged bool-key path; W2 indexes the raw key.
    assert _w1_paths({True: 5}) == {((BOOL_KEY_PATH_TAG, True),)}
    assert _w2_paths({True: torch.zeros(1)}) == {(True,)}


@pytest.mark.smoke
def test_opaque_mapping_key_subtree_parity() -> None:
    """A non-representable key collapses the literal subtree to ONE opaque parent leaf.

    W2 still RAW-indexes the tensor beneath it (declared vocabulary): the fact site
    fails literal encoding at witness time while W1's opaque leaf independently
    ceilings the run to UNVERIFIABLE -- never a false VERIFIED.
    """

    tensor = torch.zeros(2)
    root = {_Key.K: {"deep": tensor}}
    assert _w1_paths(root) == {()}
    leaves = _w1_leaves(root)
    assert len(leaves) == 1
    assert leaves[0][2] is _OPAQUE_INPUT_LEAF
    assert _runtime_nontensor_leaf_paths(root) == {()}
    assert _w2_paths(root) == {(_Key.K, "deep")}


# ======================================================================================
# 4. r64 Finding-1 end-to-end behavioral pins (the money bug) + no-over-trigger
# ======================================================================================


@dataclasses.dataclass
class _LayoutBox:
    """Dataclass model input with a single tensor field (r64 Finding-1 shape)."""

    x: torch.Tensor


class _DataclassLayoutBranch(nn.Module):
    """Branch on a dataclass tensor FIELD's contiguity (r64 Finding-1 verbatim)."""

    def forward(self, box: _LayoutBox) -> torch.Tensor:
        """Add or subtract according to the field's memory layout."""

        if box.x.is_contiguous():
            return box.x + 10
        return box.x - 10


class _DataclassPassthrough(nn.Module):
    """Read the dataclass tensor field with NO metadata-predicate reads."""

    def forward(self, box: _LayoutBox) -> torch.Tensor:
        """Return ``box.x * 2`` without touching input metadata."""

        return box.x * 2.0


def _capture_runnable(model: nn.Module, box: _LayoutBox, path: Path) -> Any:
    """Capture one dataclass-input model and save a runnable artifact."""

    trace = tl.trace(
        model,
        box,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return trace


def _layout_twin(x: torch.Tensor) -> torch.Tensor:
    """Return a byte-identical, same-shape, NON-contiguous twin of a square tensor."""

    twin = x.t().contiguous().t()
    assert torch.equal(twin, x)
    assert not twin.is_contiguous()
    return twin


@pytest.mark.smoke
def test_r64_dataclass_metadata_read_witnessed_and_layout_twin_diverges(
    tmp_path: Path,
) -> None:
    """r64 Finding-1 verbatim: the dataclass-field metadata read is now witnessed.

    Pre-fix, W2 never indexed the dataclass field, no ``model_input_metadata`` fact
    was recorded, and the non-contiguous twin replayed the captured ``+10`` arm as
    VERIFIED. Post-fix the read is witnessed and the twin diverges (default policy
    raises), never VERIFIED, never ATTESTED.
    """

    x = torch.arange(16.0).reshape(4, 4).contiguous()
    path = tmp_path / "r64_dc_layout.tlspec"
    trace = _capture_runnable(_DataclassLayoutBranch(), _LayoutBox(x), path)
    reads = trace.__dict__.get("_runnable_input_metadata_reads")
    assert reads, (
        "the is_contiguous() read on the dataclass input field recorded no metadata "
        "witness -- the r64 Finding-1 walker gap has reopened"
    )

    twin = _layout_twin(x)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_LayoutBox(twin))
    diverged = tl.load(path).run(
        inputs=_LayoutBox(twin), on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.poisoned
    # A fresh model on the twin takes the OTHER arm; the replayed arm is wrong.
    fresh = _DataclassLayoutBranch()(_LayoutBox(twin))
    assert torch.equal(fresh, twin - 10)


@pytest.mark.smoke
def test_r64_dataclass_original_input_still_verifies_and_attests(tmp_path: Path) -> None:
    """Honest original-input dataclass metadata run stays VERIFIED + ATTESTED."""

    x = torch.arange(16.0).reshape(4, 4).contiguous()
    path = tmp_path / "r64_dc_layout_ok.tlspec"
    _capture_runnable(_DataclassLayoutBranch(), _LayoutBox(x), path)

    result = tl.load(path).run(inputs=_LayoutBox(x))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(result.output, x + 10)


@pytest.mark.smoke
def test_tensor_only_dataclass_without_metadata_read_stays_verified(
    tmp_path: Path,
) -> None:
    """No-over-trigger: tensor-only dataclass, no metadata read -> VERIFIED + ATTESTED."""

    x = torch.randn(3)
    path = tmp_path / "dc_passthrough.tlspec"
    _capture_runnable(_DataclassPassthrough(), _LayoutBox(x), path)

    result = tl.load(path).run(inputs=_LayoutBox(x))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(result.output, x * 2.0)


# ======================================================================================
# 5. W4/W5/W6 behavioral round-trip parity on a torture container (not rerouted in r65)
# ======================================================================================


class _TortureNT(NamedTuple):
    """Namedtuple limb of the torture container."""

    t2: torch.Tensor
    tag: str


@dataclasses.dataclass
class _Torture:
    """Torture container: dataclass/namedtuple/dict (bool+int+str keys)/list/empty/scalar."""

    t: torch.Tensor
    nt: _TortureNT
    m: dict[Any, Any]
    e: dict[str, Any]
    s: str


def _torture_root() -> tuple[_Torture, dict[tuple[Any, ...], torch.Tensor]]:
    """Build the torture container plus its expected path->tensor leaf map."""

    tensors: dict[tuple[Any, ...], torch.Tensor] = {
        ("t",): torch.zeros(2),
        ("nt", "t2"): torch.ones(2),
        ("m", "k", 0): torch.full((2,), 2.0),
        ("m", True): torch.full((2,), 3.0),
        ("m", 7): torch.full((2,), 4.0),
    }
    root = _Torture(
        t=tensors[("t",)],
        nt=_TortureNT(t2=tensors[("nt", "t2")], tag="x"),
        m={
            "k": [tensors[("m", "k", 0)]],
            True: tensors[("m", True)],
            7: tensors[("m", 7)],
        },
        e={},
        s="leaf",
    )
    return root, tensors


def test_w4_w5_w6_round_trip_parity_on_torture_container() -> None:
    """The NON-rerouted W4/W5/W6 walkers stay in lockstep with the shared dispatch.

    W4 (`_tensor_leaf_paths`, binding side) and W6 (`_container_leaf_paths`, output
    side) must enumerate the same physical tensor leaves the shared W2 dispatch
    indexes, and W5 (`_value_at_path`) must resolve every indexed site back to the
    IDENTICAL tensor object -- the round trip that makes a recorded fact land on the
    same physical leaf at run time.
    """

    root, tensors = _torture_root()

    # W2 (shared dispatch, raw keys) indexes exactly the physical tensor leaves.
    w2 = _w2_paths(root)
    assert w2 == set(tensors)

    # W4 mirrors W2 on this container (its (str, int) key filter admits bool as int).
    w4 = set(_tensor_leaf_paths(root))
    assert w4 == w2

    # W6 (producer/output twin) yields the same leaf paths and container kinds.
    assert set(_container_leaf_paths(root)) == w4
    assert _container_kind(root) == "dataclass"
    assert _container_kind(root.nt) == "namedtuple"
    assert _container_kind(root.m) == "dict"
    assert _container_kind(root.m["k"]) == "list"

    # W5 resolves EVERY indexed site back to the IDENTICAL tensor object.
    for leaf_path, tensor in tensors.items():
        assert _value_at_path(root, leaf_path) is tensor

    # W1/W3 stay in lockstep on the same root (literal fact family).
    expected_literals = {
        ("nt", "tag"),
        ("e", EMPTY_CONTAINER_PATH_MARKER),
        ("s",),
    }
    assert _w1_paths(root) == expected_literals
    assert _runtime_nontensor_leaf_paths(root) == expected_literals


@dataclasses.dataclass
class _OpaqueKeyBox:
    """Dataclass input whose mapping field carries a non-representable key."""

    d: dict[Any, Any]


class _EnumKeyedModel(nn.Module):
    """Read a tensor stored under a non-representable mapping key."""

    def forward(self, box: _OpaqueKeyBox) -> torch.Tensor:
        """Return ``box.d[_Key.K] * 2``."""

        return box.d[_Key.K] * 2.0


class _StrKeyedModel(nn.Module):
    """Read a str-keyed tensor while an opaque-keyed sibling entry sits unread."""

    def forward(self, box: _OpaqueKeyBox) -> torch.Tensor:
        """Return ``box.d['k'] * 2``."""

        return box.d["k"] * 2.0


def test_opaque_key_subtree_is_never_silently_witnessable(tmp_path: Path) -> None:
    """A subtree under a non-representable key can never yield VERIFIED.

    W2 RAW-indexes a tensor under such a key (declared vocabulary) while W4's key
    filter skips it, so the dual vocabulary is only honest because every path that
    could bless such an artifact fails closed: a CONSUMED opaque-keyed tensor refuses
    at save (binding container paths require str/int keys), and an opaque-keyed
    NON-tensor sibling saves but the literal walker's OPAQUE parent leaf ceilings the
    run -- never VERIFIED, never ATTESTED.
    """

    tensor = torch.randn(2)
    consumed = _OpaqueKeyBox({_Key.K: tensor})
    assert _w2_paths(consumed) == {("d", _Key.K)}
    assert set(_tensor_leaf_paths(consumed)) == set()  # W4's (str, int) key filter
    assert _w1_paths(consumed) == {("d",)}  # one opaque leaf at the mapping's path

    # Prong A: a consumed tensor under a non-representable key fails closed at SAVE
    # (the binding-path producer refuses non-(str, int) keys) -- no artifact exists
    # that could ever replay it as VERIFIED.
    trace = tl.trace(
        _EnumKeyedModel(),
        consumed,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    with pytest.raises(ValueError, match="string or integer keys"):
        trace.save(tmp_path / "enum_key.tlspec", level="runnable", include_activations=True)

    # Prong B: an opaque-keyed NON-tensor sibling (bytes key -- outside the literal
    # grammar but safely picklable) saves, and the opaque literal leaf then CEILINGS
    # the run on the ORIGINAL input: UNVERIFIABLE, never VERIFIED, never ATTESTED.
    sibling = _OpaqueKeyBox({"k": tensor, b"kk": 5})
    assert _w1_paths(sibling) == {("d",)}
    path = tmp_path / "bytes_key.tlspec"
    trace = tl.trace(
        _StrKeyedModel(),
        sibling,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    result = tl.load(path).run(
        inputs=_OpaqueKeyBox({"k": tensor, b"kk": 5}),
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.poisoned
