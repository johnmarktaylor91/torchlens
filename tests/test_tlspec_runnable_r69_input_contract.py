"""r69 B+D -- ONE scalar type-class classifier + ONE canonical mapping-key authority.

The r68 clusters B (scalar TYPE conflation -> false VERIFIED/ATTESTED) and D (exotic
mapping keys: NaN false-DIVERGED, numpy-key advertise-then-fail, enum-key admission)
share one root: four independent ``isinstance`` scalar dispatches. r69 single-sources
them in ``torchlens._input_walk.classify_scalar`` and makes ``encode_mapping_key`` the
sole key-identity authority end to end (snapshot ordered-key facts, literal paths,
tensor bindings, and runtime lookup all bind by exact canonical encoded token).

Locked policy table (PLAN_AGREED r69):

    | lattice class            | VALUE leaf                          | mapping KEY    |
    |--------------------------|-------------------------------------|----------------|
    | exact builtin atom       | value-equal (existing pinned rules) | codec-encode   |
    | stock numpy wrapper      | .item() value-equal (RATIFIED pin)  | REFUSE typed   |
    | semantic (enum/subclass) | REFUSE typed at save; runtime       | REFUSE typed   |
    |                          | cross-class DIVERGES both directions|                |
    | outside grammar          | opaque ceiling (unchanged)          | refuse (as-is) |

GREEN pins preserved verbatim: ``np.float64(2.0) <-> 2.0`` VERIFIES, ``-0.0/+0.0``
diverge, ``True != 1`` strict, int/float family separation, current NaN-value rule.
Plus the r69 cross-cutting belt: NO runnable-save input-boundary path may raise an
untyped error -- every cell of the leaf-type x position sweep either succeeds or
refuses through a typed ``RunnableErrorCode`` disposition.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
import struct
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_walk import (
    classify_scalar,
    decode_mapping_key,
    encode_mapping_key,
    snapshot_input_boundary,
)
from torchlens.errors import (
    PathDivergenceError,
    RunnablePreflightError,
    RunPreconditionError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _trace(model: nn.Module, inputs) -> tl.Trace:
    return tl.trace(model, inputs, capture=_CAPTURE)


def _save(trace: tl.Trace, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _assert_semantic_refusal(excinfo, *extra_needles: str) -> None:
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "missing_input_container_contract" in diagnostics
    for needle in extra_needles:
        assert needle in diagnostics, needle


# ======================================================================================
# The scalar lattice specimens (generated, no test-owned numpy list)
# ======================================================================================


class _PlainEnum(enum.Enum):
    A = 1


class _IntMode(enum.IntEnum):
    FAST = 1


class _Flags(enum.IntFlag):
    ON = 1


class _StrMode(enum.StrEnum):
    FAST = "fast"


class _FloatMode(float, enum.Enum):
    HALF = 0.5


class _CustomInt(int):
    pass


class _CustomFloat(float):
    pass


class _CustomStr(str):
    pass


class _StatefulFloat(float):
    """Scalar subclass carrying hidden instance state (Sol's r69 experiment)."""


_SEMANTIC_SPECIMENS: dict[str, Any] = {
    "plain_enum": _PlainEnum.A,
    "int_enum": _IntMode.FAST,
    "int_flag": _Flags.ON,
    "str_enum": _StrMode.FAST,
    "float_enum": _FloatMode.HALF,
    "custom_int": _CustomInt(1),
    "custom_float": _CustomFloat(2.0),
    "custom_str": _CustomStr("fast"),
    "np_str": np.str_("fast"),
}
_stateful = _StatefulFloat(2.0)
_stateful.mode = "fast"
_SEMANTIC_SPECIMENS["stateful_float"] = _stateful


def _np_float64_subclass_specimen():
    class _WeirdNp(np.float64):
        pass

    return _WeirdNp(2.0)


@pytest.mark.smoke
def test_r69_classifier_lattice_is_closed_and_exact() -> None:
    """Every specimen lands in its agreed lattice class; stock set is programmatic."""

    # Exact builtins.
    for value in (True, 0, -3, 2.5, "s"):
        assert classify_scalar(value)[0] == "plain", value
    for value in (None, Ellipsis):
        assert classify_scalar(value)[0] == "singleton", value
    # Enum members (checked FIRST, before numeric families) + scalar subclasses.
    for name, value in _SEMANTIC_SPECIMENS.items():
        assert classify_scalar(value)[0] == "semantic", name
    assert classify_scalar(_np_float64_subclass_specimen())[0] == "semantic"
    # Stock numpy numeric/bool wrappers: transparent, .item() exact.
    for wrapper in (
        np.bool_(True),
        np.int8(3),
        np.uint64(2**63),
        np.int64(-1),
        np.float16(0.5),
        np.float32(1.5),
        np.float64(2.0),
    ):
        kind, item = classify_scalar(wrapper)
        assert kind == "numpy", wrapper
        assert type(item) in (bool, int, float)
        assert item == wrapper.item()
    # Broad np.generic membership is NOT the admission test: stock non-numeric
    # wrappers stay in their existing opaque lane.
    for value in (np.datetime64(0, "s"), np.void(b""), np.bytes_(b"x"), np.complex128(1 + 2j)):
        assert classify_scalar(value)[0] == "opaque", value
    # Outside grammar entirely.
    for value in (b"x", 1 + 2j, Decimal("1.5"), object(), [1], {"k": 1}):
        assert classify_scalar(value)[0] == "opaque", value


# ======================================================================================
# B -- semantic captures refuse typed at root AND nested paths
# ======================================================================================


class _TwoArg(nn.Module):
    def forward(self, x, flag):
        return torch.relu(x) + 1.0


class _DictArg(nn.Module):
    def forward(self, x, cfg):
        return torch.relu(x) + 1.0


@dataclasses.dataclass
class _FlagBox:
    x: torch.Tensor
    flag: Any


class _BoxArg(nn.Module):
    def forward(self, box):
        return torch.relu(box.x) + 1.0


@pytest.mark.parametrize("name", sorted(_SEMANTIC_SPECIMENS))
def test_r69_semantic_scalar_capture_refuses_at_root_and_nested(name: str, tmp_path: Path) -> None:
    value = _SEMANTIC_SPECIMENS[name]
    x = torch.randn(3)
    # Root position. (A top-level str-FAMILY input is intercepted by the pre-existing
    # ``_input_coerce`` tokenizer-coercion lane before capture -- pre-r67, out of r69
    # scope -- so the root cell is exercised for non-str specimens; str-family
    # semantic leaves are covered by the nested cells below.)
    if not isinstance(value, str):
        with pytest.raises(RunnablePreflightError) as excinfo:
            _save(_trace(_TwoArg(), [x, value]), tmp_path / f"root_{name}.tlspec")
        _assert_semantic_refusal(excinfo, "semantic_scalar_type")
    # Nested mapping value.
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(_trace(_DictArg(), [x, {"flag": value}]), tmp_path / f"map_{name}.tlspec")
    _assert_semantic_refusal(excinfo, "semantic_scalar_type")
    # Nested dataclass field.
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(_trace(_BoxArg(), _FlagBox(x, value)), tmp_path / f"box_{name}.tlspec")
    _assert_semantic_refusal(excinfo, "semantic_scalar_type")


class _IntBranch(nn.Module):
    def forward(self, x, mode):
        if mode == 1:
            return x * 2.0
        return x + 100.0


class _FloatBranch(nn.Module):
    def forward(self, x, s):
        if s > 1.0:
            return x * 2.0
        return x + 100.0


@pytest.mark.smoke
def test_r69_admitted_capture_semantic_runtime_replacement_diverges(tmp_path: Path) -> None:
    """Both admitted lanes (plain + stock numpy) diverge on same-value semantic swaps."""

    x = torch.tensor([1.0, 2.0, 3.0])
    # plain int capture -> IntEnum / custom-int runtime swaps.
    path = _save(_trace(_IntBranch(), [x, 1]), tmp_path / "plain_int.tlspec")
    assert tl.load(path).run(inputs=[x, 1]).report.path_faithfulness is PathFaithfulness.VERIFIED
    for swap in (_IntMode.FAST, _CustomInt(1)):
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, swap])
        diverged = tl.load(path).run(
            inputs=[x, swap], on_divergence=DivergencePolicy.RETURN_DIVERGED
        )
        assert diverged.report.path_faithfulness is not PathFaithfulness.VERIFIED
        assert diverged.report.numeric_attestation is not NumericAttestationStatus.ATTESTED
        assert diverged.report.poisoned
    # stock numpy capture -> float-enum / custom-float runtime swaps.
    path = _save(_trace(_FloatBranch(), [x, np.float64(2.0)]), tmp_path / "np_float.tlspec")
    for swap in (_CustomFloat(2.0), _StatefulFloat(2.0)):
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, swap])


@pytest.mark.smoke
def test_r69_green_pins_stock_numpy_value_transparency(tmp_path: Path) -> None:
    """RATIFIED pins: np<->builtin same-value verifies; changed value diverges."""

    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(_trace(_FloatBranch(), [x, np.float64(2.0)]), tmp_path / "pin_np.tlspec")
    # np.float64(2.0) <-> 2.0 must VERIFY (the locked r67 heavy-gate pin).
    assert tl.load(path).run(inputs=[x, 2.0]).report.path_faithfulness is PathFaithfulness.VERIFIED
    assert (
        tl.load(path).run(inputs=[x, np.float64(2.0)]).report.path_faithfulness
        is PathFaithfulness.VERIFIED
    )
    # Changed value still diverges (both spellings).
    for changed in (0.5, np.float64(0.5)):
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, changed])
    # Reverse capture direction: plain float capture, numpy runtime twin verifies.
    path = _save(_trace(_FloatBranch(), [x, 2.0]), tmp_path / "pin_plain.tlspec")
    assert (
        tl.load(path).run(inputs=[x, np.float64(2.0)]).report.path_faithfulness
        is PathFaithfulness.VERIFIED
    )
    # bool/int and int/float family separation stays strict.
    path = _save(_trace(_IntBranch(), [x, 1]), tmp_path / "pin_int.tlspec")
    for swap in (True, 1.0, np.bool_(True), np.float64(1.0)):
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, swap])


# ======================================================================================
# B -- the ONE forced documented residual (contract section 11), executable pin
# ======================================================================================


class _WrapperSteer(nn.Module):
    """Branch on plain-vs-stock-numpy WRAPPER identity of a same-value scalar leaf."""

    def forward(self, x, v):
        if isinstance(v, np.floating):
            return x * 2.0
        return x + 100.0


@pytest.mark.smoke
def test_r69_expected_residual_plain_vs_stock_wrapper_type_steering(tmp_path: Path) -> None:
    """EXPECTED-RESIDUAL pin (contract section 11): wrapper-identity steering.

    The ratified ``np.float64(2.0) <-> 2.0`` value-verification pin fixes VERIFIED
    for the identical observable pair, so host control flow steered by the
    plain-builtin vs stock-transparent-NumPy-wrapper identity of a same-value
    scalar leaf CANNOT be caught without revoking that pin (no static fact can
    distinguish identical observables; ``type``/``isinstance`` are not
    interceptable completely). This is the documented FORCED residual -- scoped to
    the stock wrapper lane only: enum and custom scalar-subclass identity ARE
    caught (typed save refusal / runtime divergence, pinned above). If this test
    ever fails because the replay diverges, the residual was closed -- update
    contract section 11 in the same change.
    """

    x = torch.tensor([1.0, 2.0, 3.0])
    # Direction 1: numpy capture, plain replay -- residual false-VERIFIED.
    path = _save(_trace(_WrapperSteer(), [x, np.float64(2.0)]), tmp_path / "resid_np.tlspec")
    result = tl.load(path).run(inputs=[x, 2.0])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    # Direction 2: plain capture, numpy replay -- residual false-VERIFIED.
    path = _save(_trace(_WrapperSteer(), [x, 2.0]), tmp_path / "resid_plain.tlspec")
    result = tl.load(path).run(inputs=[x, np.float64(2.0)])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# D -- the mapping-key admission/round-trip matrix (the structural law)
# ======================================================================================


class _KeyedTensor(nn.Module):
    """Read the single tensor stored in the keyed dict (key identity is the input)."""

    def forward(self, x, d):
        return x + next(iter(d.values()))


_NAN = float("nan")
_ADMITTED_KEYS: dict[str, Any] = {
    "str": "k",
    "reserved_prefix_str": "\x00tlk:b:1",
    # r71 D: reserved-namespace sentinels are REAL dict keys that must round-trip
    # (never false-DIVERGE by positional sentinel interpretation).
    "empty_container_marker": "\x00tl_empty_container",
    "bool_key_marker": "\x00tl_bool_key",
    "null_prefixed_junk": "\x00junk",
    "int": 7,
    "bool_true": True,
    "float": 2.5,
    "float_one": 1.0,
    "neg_zero": -0.0,
    "pos_zero": 0.0,
    "inf": float("inf"),
    "neg_inf": float("-inf"),
    "nan": _NAN,
    "none": None,
    "nested_tuple": ("a", (True, 2.5)),
    "nan_in_tuple": (_NAN, 1),
    # free repro3c: a tuple carrying reserved strings round-trips.
    "reserved_in_tuple": ("\x00tl_empty_container", 1),
}


class _TupleSubclassKey(tuple):
    pass


_REFUSED_KEYS: dict[str, Any] = {
    "np_float64": np.float64(2.5),
    "np_int64": np.int64(7),
    "np_bool": np.bool_(True),
    "int_enum": _IntMode.FAST,
    "str_enum": _StrMode.FAST,
    "float_enum": _FloatMode.HALF,
    "plain_enum": _PlainEnum.A,
    "custom_int": _CustomInt(7),
    "custom_str": _CustomStr("k"),
    "tuple_subclass": _TupleSubclassKey(("a",)),
    "bytes": b"kk",
    "frozenset": frozenset({1}),
    "object": object(),
}


@pytest.mark.parametrize("name", sorted(_ADMITTED_KEYS))
def test_r69_admitted_key_saves_loads_and_runs_identical(name: str, tmp_path: Path) -> None:
    """Structural law, admitted half: save -> safe-load -> run identical VERIFIES.

    corr1recover-F2 regression class: no admitted key may false-DIVERGE (or crash)
    on the IDENTICAL captured input, and no successful save may be unloadable.
    """

    key = _ADMITTED_KEYS[name]
    x = torch.tensor([1.0, 2.0, 3.0])
    payload = torch.tensor([0.5, 0.5, 0.5])
    path = _save(_trace(_KeyedTensor(), [x, {key: payload}]), tmp_path / f"key_{name}.tlspec")
    result = tl.load(path).run(inputs=[x, {key: payload.clone()}])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, x + payload)


@pytest.mark.parametrize("name", sorted(_REFUSED_KEYS))
def test_r69_refused_key_refuses_typed_at_save(name: str, tmp_path: Path) -> None:
    """Structural law, refused half: typed at save -- never advertise-then-fail."""

    key = _REFUSED_KEYS[name]
    x = torch.tensor([1.0, 2.0, 3.0])
    payload = torch.tensor([0.5, 0.5, 0.5])
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(_trace(_KeyedTensor(), [x, {key: payload}]), tmp_path / f"ref_{name}.tlspec")
    _assert_semantic_refusal(excinfo, "opaque_mapping_key")


@pytest.mark.smoke
def test_r69_duplicate_same_bit_nan_keys_refuse_ambiguous(tmp_path: Path) -> None:
    """Two distinct same-bit NaN key objects -> one token -> ambiguous, refuse typed."""

    x = torch.tensor([1.0, 2.0, 3.0])
    n1, n2 = float("nan"), float("nan")
    d = {n1: torch.tensor([0.5, 0.5, 0.5]), n2: torch.tensor([1.0, 1.0, 1.0])}
    assert len(d) == 2  # hash-equal, eq-False: two live slots
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(_trace(_KeyedTensor(), [x, d]), tmp_path / "dup_nan.tlspec")
    _assert_semantic_refusal(excinfo, "ambiguous_mapping_key")


def test_r69_runtime_duplicate_token_matches_fail_closed() -> None:
    """Bind side: multiple runtime keys encoding to one token fail closed, typed."""
    from torchlens._runnable_execution import _value_at_path

    n1, n2 = float("nan"), float("nan")
    mapping = {n1: 1, n2: 2}
    token = encode_mapping_key(n1)
    with pytest.raises(KeyError, match="ambiguous_mapping_key"):
        _value_at_path(mapping, (token,))


@pytest.mark.smoke
def test_r69_mutated_keys_diverge(tmp_path: Path) -> None:
    """Signed-zero, bool/int/float, tuple-member, and semantic key swaps diverge."""

    x = torch.tensor([1.0, 2.0, 3.0])
    payload = torch.tensor([0.5, 0.5, 0.5])

    cases = [
        (-0.0, 0.0),
        (True, 1),
        (1, True),
        (1, 1.0),
        (("a", 1), ("a", 2)),
        (7, _IntMode.FAST),  # builtin -> semantic runtime key swap
        ("k", _CustomStr("k")),
    ]
    for index, (captured_key, runtime_key) in enumerate(cases):
        path = _save(
            _trace(_KeyedTensor(), [x, {captured_key: payload}]),
            tmp_path / f"mut_{index}.tlspec",
        )
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, {runtime_key: payload.clone()}])


def test_r69_nan_key_nested_and_bit_distinct_payloads() -> None:
    """NaN tokens carry binary64 bits: same-bit binds, different-payload differs."""

    quiet = struct.unpack(">d", bytes.fromhex("7ff8000000000000"))[0]
    payload_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    assert math.isnan(quiet) and math.isnan(payload_nan)
    token_quiet = encode_mapping_key(quiet)
    token_payload = encode_mapping_key(payload_nan)
    assert token_quiet != token_payload
    decoded = decode_mapping_key(token_quiet)
    assert struct.pack(">d", decoded) == struct.pack(">d", quiet)
    # Nested tuple recursion keeps the bit pattern.
    assert encode_mapping_key((quiet, 1)) != encode_mapping_key((payload_nan, 1))


@pytest.mark.smoke
def test_r71d_reserved_marker_keys_round_trip_without_false_divergence(tmp_path: Path) -> None:
    """free repro3c: a dict input keyed by a reserved sentinel (or a tuple containing
    one) saves/loads/runs VERIFIED on the UNCHANGED input; a changed value under such
    a key still diverges; the empty-container marker behavior is unaffected."""

    from torchlens.errors import PathDivergenceError, RunPreconditionError

    x = torch.tensor([1.0, 2.0, 3.0])
    payload = torch.tensor([0.5, 0.5, 0.5])
    reserved = ["\x00tl_empty_container", "\x00tl_bool_key", "\x00tlk:x", "\x00junk"]
    for index, key in enumerate(reserved):
        path = _save(_trace(_KeyedTensor(), [x, {key: payload}]), tmp_path / f"d_{index}.tlspec")
        result = tl.load(path).run(inputs=[x, {key: payload.clone()}])
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, key
        assert torch.equal(result.output, x + payload)
        # A CHANGED KEY (a different reserved marker) diverges: the token contract is
        # intact, the belt only escapes the key -- it never conflates two distinct
        # reserved keys.
        other = "\x00tl_bool_key" if key != "\x00tl_bool_key" else "\x00junk"
        with pytest.raises((PathDivergenceError, RunPreconditionError)):
            tl.load(path).run(inputs=[x, {other: payload.clone()}])

    # Tuple carrying a reserved string round-trips.
    tup_key = ("\x00tl_empty_container", 1)
    path = _save(_trace(_KeyedTensor(), [x, {tup_key: payload}]), tmp_path / "d_tup.tlspec")
    result = tl.load(path).run(inputs=[x, {tup_key: payload.clone()}])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    # A GENUINE empty-container marker still behaves (an added empty container
    # diverges) -- the belt escapes real keys, never the synthetic sentinel.
    class _EmptyBranch(nn.Module):
        def forward(self, x, cfg):
            return x * 2.0 if cfg.get("flag", {}) else x + 100.0

    path = _save(_trace(_EmptyBranch(), [x, {"flag": {}}]), tmp_path / "d_empty.tlspec")
    assert (
        tl.load(path).run(inputs=[x, {"flag": {}}]).report.path_faithfulness
        is PathFaithfulness.VERIFIED
    )
    with pytest.raises((PathDivergenceError, RunPreconditionError)):
        tl.load(path).run(inputs=[x, {"flag": {"present": 1}}])


@pytest.mark.smoke
def test_r71d_reserved_registry_meta_test() -> None:
    """Every sentinel interpreted before mapping lookup is in the reserved registry
    AND is never round-tripped raw by the key codec."""

    import inspect

    from torchlens import _runnable_execution
    from torchlens._input_walk import encode_mapping_key, reserved_input_path_components

    registry = reserved_input_path_components()
    # The one sentinel _value_at_path interprets positionally before mapping lookup.
    from torchlens._io.runnable import EMPTY_CONTAINER_PATH_MARKER

    assert EMPTY_CONTAINER_PATH_MARKER in registry
    # No reserved sentinel round-trips raw (each is escaped away from itself).
    for sentinel in registry:
        assert encode_mapping_key(sentinel) != sentinel, sentinel
    # The runtime path resolver still compares canonical tokens only (no decoded lane).
    bind_source = inspect.getsource(_runnable_execution._value_at_path)
    assert "decode_mapping_key" not in bind_source


def test_r69_key_codec_is_injective_and_round_trips() -> None:
    """Property: distinct admitted keys -> distinct tokens; grammar round-trips."""

    keys = [
        "k",
        "\x00tlk:x",
        "\x00tl_empty_container",
        "\x00tl_bool_key",
        "\x00junk",
        "",
        0,
        1,
        -7,
        True,
        False,
        2.5,
        1.0,
        -0.0,
        0.0,
        float("inf"),
        float("-inf"),
        _NAN,
        None,
        (),
        ("a",),
        ("a", 1),
        ("a", (True, 2.5)),
        (_NAN, 1),
    ]
    tokens = [encode_mapping_key(key) for key in keys]
    assert len(set(tokens)) == len(tokens), "codec must be injective over admitted keys"
    for key, token in zip(keys, tokens):
        assert isinstance(token, (str, int))
        decoded = decode_mapping_key(token)
        if isinstance(key, float) and math.isnan(key):
            assert struct.pack(">d", decoded) == struct.pack(">d", key)
        elif isinstance(key, tuple):
            assert type(decoded) is tuple and repr(decoded) == repr(key)
        else:
            assert type(decoded) is type(key) and decoded == key


# ======================================================================================
# Source/meta-tests: one classifier at every choke point; no decoded-value binding
# ======================================================================================


@pytest.mark.smoke
def test_r69_classifier_choke_points_source_scan() -> None:
    """Snapshot, literal encode, literal compare, and key codec share the classifier."""

    from torchlens import _runnable_execution
    from torchlens import _input_walk
    from torchlens._io import runnable as io_runnable

    for func in (
        _input_walk.snapshot_input_boundary,
        _input_walk.encode_mapping_key,
        io_runnable._encode_literal,
        io_runnable._normalize_numpy_scalars,
        _runnable_execution._literal_leaf_equal,
    ):
        assert "classify_scalar" in inspect.getsource(func), func.__qualname__

    # The classifier runs BEFORE any isinstance normalization in _encode_literal,
    # so no recipe path can launder a semantic scalar into a plain atom.
    encode_source = inspect.getsource(io_runnable._encode_literal)
    assert encode_source.index("classify_scalar") < encode_source.index("isinstance(")

    # No broad numpy-family admission at the comparison choke point.
    compare_source = inspect.getsource(_runnable_execution._literal_leaf_equal)
    assert "np.generic" not in compare_source
    assert "isinstance(value, np.generic)" not in compare_source

    # Runtime mapping binding is canonical-token-only: the decoded-value lookup
    # lane (``candidate == decoded``) is structurally gone.
    bind_source = inspect.getsource(_runnable_execution._value_at_path)
    assert "candidate == decoded" not in bind_source
    assert "decode_mapping_key" not in bind_source
    assert "encode_mapping_key" in bind_source


@pytest.mark.smoke
def test_r69_snapshot_key_tokens_paths_and_lookup_share_the_authority() -> None:
    """Snapshot ordered-key facts and leaf paths speak encode_mapping_key tokens."""

    from torchlens._runnable_execution import _value_at_path

    value = {2.5: {"inner": 1}, True: 2, "s": 3}
    snapshot = snapshot_input_boundary(value)
    root = snapshot["nodes"][0]
    assert root["keys"] == [encode_mapping_key(k) for k in value]
    for node in snapshot["nodes"][1:]:
        component = node["path"][0]
        assert component in root["keys"]
        # Every persisted component resolves through canonical-token lookup.
        _value_at_path(value, tuple(node["path"][:1]))


# ======================================================================================
# r69 A regression (heavy-gate): per-slot escape witnesses under ONE name-keyed label
# ======================================================================================


class _BufferNumpyWriteback(nn.Module):
    """r15-C2 shape: zero-copy numpy write-back into a registered buffer's storage."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 4.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.b.detach().numpy()[0] = 99.0
        return self.b * x


@pytest.mark.smoke
def test_r69_multi_slot_escape_witnesses_stay_runnable(tmp_path: Path) -> None:
    """A state name owning MULTIPLE escape-witnessed slots must not over-refuse.

    The producer legitimately emits one ``unbound_state_escape:<name>`` witness per
    qualifying SLOT (same site label, distinct per-fact ``slot_id``), so the
    inventory member identity is ``<name>::<slot_id>`` -- never the bare label
    (which would false-refuse the r15 buffer-numpy-writeback artifact as a
    duplicate member and silently LOSE its UNVERIFIABLE honesty verdict by
    degrading the load to analysis-only).
    """

    from torchlens._io.runnable import required_witness_family_members

    x = torch.tensor([3.0, 5.0])
    path = _save(_trace(_BufferNumpyWriteback(), x.clone()), tmp_path / "wb.tlspec")
    loaded = tl.load(path)
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor is not None, "artifact must stay RUNNABLE, never analysis-only"
    members = required_witness_family_members(descriptor.control_witnesses)["unbound_state_escape"]
    assert len(members) == len(set(members)), members
    assert len(members) >= 2, members  # same name, several slots -> distinct identities
    assert all(member.startswith("b::") for member in members), members
    # The r15 honesty verdict is preserved: the run executes and is NOT verified.
    result = loaded.run(inputs=x.clone(), seed=0, on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    # A genuinely DUPLICATED escape fact (same name AND slot) still refuses at parse.
    import copy as _copy
    import json as _json
    import os as _os

    manifest_path = _os.path.join(path, "manifest.json")
    manifest = _json.load(open(manifest_path))
    run = manifest["run"]
    twin = _copy.deepcopy(
        next(
            w
            for w in run["control_witnesses"]
            if str(w.get("site_label", "")).startswith("unbound_state_escape:")
        )
    )
    twin["witness_id"] = "witness:9999"
    twin["order"] = 9999
    run["control_witnesses"] = list(run["control_witnesses"]) + [twin]
    _json.dump(manifest, open(manifest_path, "w"))
    from torchlens.runnable import ReadinessStatus

    reloaded = tl.load(path)
    readiness = reloaded.__dict__.get("_runnable_readiness")
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert "context_field_invalid" in {d.code.value for d in readiness.diagnostics}


# ======================================================================================
# Cross-cutting r69 belt: NO untyped error on ANY input-boundary runnable save
# ======================================================================================


class _BeltStrSub(str):
    pass


_BELT_LEAVES: dict[str, Any] = {
    "str": "fast",
    "bytes": b"fast",
    "str_subclass": _BeltStrSub("fast"),
    "complex": 1 + 2j,
    "decimal": Decimal("1.5"),
    "enum": _PlainEnum.A,
    "np_scalar": np.float64(2.0),
    "object": object(),
}

_BELT_POSITIONS = ("top_level", "mapping_value", "dataclass_field", "tuple_element")


@pytest.mark.parametrize("position", _BELT_POSITIONS)
@pytest.mark.parametrize("leaf_name", sorted(_BELT_LEAVES))
def test_r69_no_untyped_error_input_boundary_save_sweep(
    leaf_name: str, position: str, tmp_path: Path
) -> None:
    """Every runnable-save cell either SUCCEEDS or refuses through a typed
    ``RunnableErrorCode`` disposition -- a bare AssertionError or untyped
    ``TorchLensIOError`` anywhere in the matrix is a failure (the belt that
    catches the NEXT DROP-order-shaped bug regardless of which leaf trips it).
    """

    from torchlens._io import TorchLensIOError

    leaf = _BELT_LEAVES[leaf_name]
    x = torch.randn(3)
    model: nn.Module = _DictArg()
    if position == "top_level":
        capture_inputs: Any = [x, leaf]
    elif position == "mapping_value":
        capture_inputs = [x, {"flag": leaf}]
    elif position == "dataclass_field":
        capture_inputs = _FlagBox(x, leaf)
        model = _BoxArg()
    else:
        capture_inputs = [x, (leaf,)]

    try:
        trace = _trace(model, capture_inputs)
    except Exception:
        # Pre-save CAPTURE-lane interception (e.g. the pre-existing top-level
        # str-family tokenizer coercion in _input_coerce) -- outside the runnable
        # SAVE belt's contract.
        pytest.skip(f"capture-lane interception for {leaf_name}@{position}")

    try:
        _save(trace, tmp_path / f"belt_{leaf_name}_{position}.tlspec")
    except RunnablePreflightError as exc:
        # Typed refusal: carries structured RunnableErrorCode diagnostics.
        assert exc.fields.get("diagnostics"), (leaf_name, position)
    except TorchLensIOError as exc:
        cause_chain = []
        cause = exc.__cause__
        while cause is not None:
            cause_chain.append(type(cause).__name__)
            cause = cause.__cause__
        raise AssertionError(
            f"untyped IO failure for {leaf_name}@{position}: {exc} (causes {cause_chain})"
        ) from exc
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001 -- the belt's whole point
        raise AssertionError(
            f"untyped {type(exc).__name__} for {leaf_name}@{position}: {exc}"
        ) from exc
