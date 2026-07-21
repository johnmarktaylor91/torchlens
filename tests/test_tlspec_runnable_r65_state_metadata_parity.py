"""State-metadata witness net == FULL input-net parity (r65 Cluster X) -- anti-edge pins.

Round-64 (both labs) proved the r63 state-metadata net mirrored only 5 of the input net's 20
accessors, leaving an "Nth unwitnessed state read" CLASS open (F2: zero-copy view exports leak
stride/offset with no accessor call; F3: a larger-base offset-0-contiguous view flips a
``storage_nbytes`` branch). r65 closes the CLASS structurally:

* ONE authoritative table (``STATE_METADATA_MIRROR``) whose keys are pinned BY TEST to the
  input-constant union AND the io-side fact vocabulary (T-X1) -- a future accessor added to any
  input constant without an explicit state disposition is a RED test, never a silent gap.
* Every mirror row chains into enforcement (T-X2): read kinds resolve to signature dims, dims
  are stamped by THE one signature helper, physical-scope dims are staging-guaranteed (T-X3).
* ``requires_grad`` (+ ``grad_fn`` presence) is a DECLARED-STATE FACT (locked F-1 ruling):
  witnessed, persisted, and REPRODUCED by staging -- never a refusal or ceiling (escape-gating
  it would refuse every frozen model). NOTE: TorchLens's grad-capture machinery runs the
  captured forward with param ``requires_grad`` temporarily enabled (pre-existing capture
  behavior), so the recorded fact is the bit the forward ACTUALLY read; the pin here is
  recorded == staged, exactly.
* Attribution families mirror the input net: layout/placement/geometry accessors attribute by
  STORAGE IDENTITY (a view's value is a pure function of the slot); the autograd/structural
  family attributes DIRECT-receiver-only -- its alias/view reads are the NAMED residual set,
  the state twin of the input net's leaf-only rule (TorchLens's own per-op bookkeeping reads
  ``requires_grad``/``grad_fn``/``_version`` on views and, pre-r65, on direct state receivers;
  the direct-receiver bookkeeping reads are now marked internal at their source, pinned by the
  no-over-trigger test below). ``w.contiguous() is w`` identity-aliasing stays a documented
  residual (locked ruling).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import runnable as io_runnable
from torchlens._io import runnable_load
from torchlens._runnable_state import (
    _STATE_METADATA_BIND_SCOPE,
    _STATE_METADATA_PHYSICAL_SCOPE,
    _STATE_METADATA_READ_REQUIRED_DIMS,
    _state_metadata_signature,
    prepare_runnable_state,
    recorded_state_metadata_facts,
    state_metadata_full_violations,
)
from torchlens.backends.torch.completeness_witness import (
    INPUT_METADATA_BOOL_METHODS,
    INPUT_METADATA_PREDICATE_FUNCS,
    INPUT_METADATA_PROPERTY_NAMES,
    STATE_METADATA_MIRROR,
    _STATE_METADATA_ALIAS_SAFE_STATE_NAMES,
    _STATE_METADATA_DIRECT_ONLY_NAMES,
    _STATE_ROUTE_DECLARED_FACT,
    _STATE_ROUTE_READ_KIND,
    _STATE_ROUTE_STRUCTURAL,
    host_escape_state_metadata_facts,
    host_escape_state_metadata_reads,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import ControlWitness, ControlWitnessKind, PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=_CAPTURE)


def _save(trace: tl.Trace, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _assert_refuses(trace: tl.Trace, path: Path, *needles: str) -> None:
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(trace, path)
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "state_metadata_mismatch" in diagnostics
    assert "producer_state_metadata" in diagnostics
    for needle in needles:
        assert needle in diagnostics, needle


def _assert_verified(model: nn.Module, x: torch.Tensor, path: Path) -> tl.Trace:
    loaded = tl.load(_save(_trace(model, x), path))
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    return loaded


# ======================================================================================
# T-X1 -- the headline set-identity invariant (mirror == input constants == fact names)
# ======================================================================================


@pytest.mark.smoke
def test_r65_mirror_keys_equal_input_constant_union_and_fact_vocabulary() -> None:
    """The ONE mirror covers EXACTLY the input net's accessor union -- by construction.

    A future accessor added to any input constant (or to the io fact vocabulary) without an
    explicit state disposition turns this test RED: the "Nth unwitnessed state read" class
    stays closed structurally, not by per-round enumeration.
    """

    input_union = (
        INPUT_METADATA_PREDICATE_FUNCS
        | INPUT_METADATA_BOOL_METHODS
        | INPUT_METADATA_PROPERTY_NAMES
        | {"storage_nbytes"}
    )
    assert set(STATE_METADATA_MIRROR) == input_union
    assert input_union == io_runnable._INPUT_METADATA_FACT_NAMES
    assert len(input_union) == 20


# ======================================================================================
# T-X2 -- chain completeness: every mirror row terminates in enforcement
# ======================================================================================


@pytest.mark.smoke
def test_r65_every_mirror_row_chains_into_enforcement() -> None:
    """read_kind rows -> required-dim rows -> signature dims; fact rows -> closed io vocab."""

    signature_keys = set(_state_metadata_signature(torch.zeros(2, 2)))
    for name, (route, detail) in STATE_METADATA_MIRROR.items():
        if route == _STATE_ROUTE_READ_KIND:
            assert detail in _STATE_METADATA_READ_REQUIRED_DIMS, name
        elif route == _STATE_ROUTE_DECLARED_FACT:
            assert detail in io_runnable._STATE_METADATA_FACT_NAMES, name
            assert detail in runnable_load._STATE_METADATA_FACT_ALLOWED_NAMES, name
        else:
            assert route == _STATE_ROUTE_STRUCTURAL, name
    for kind, (dim, _expected) in _STATE_METADATA_READ_REQUIRED_DIMS.items():
        assert dim in signature_keys, kind
    for dim, _expected in _STATE_METADATA_PHYSICAL_SCOPE + _STATE_METADATA_BIND_SCOPE:
        assert dim in signature_keys, dim
    # Every required-dim row outside the bind scope is a PHYSICAL dim (the unconditional
    # staged tripwire owns it), so an escape-gated refusal is always staging-guaranteed.
    physical_dims = {dim for dim, _ in _STATE_METADATA_PHYSICAL_SCOPE}
    for kind, (dim, _expected) in _STATE_METADATA_READ_REQUIRED_DIMS.items():
        assert dim in physical_dims, kind


@pytest.mark.smoke
def test_r65_named_residual_set_is_exactly_the_autograd_family() -> None:
    """The direct-only (alias-read residual) set is NAMED and justified, not incidental.

    These are the accessors whose alias/view reads on state cannot be attributed without
    over-trigger: TorchLens's own per-op bookkeeping reads ``requires_grad`` / ``grad_fn`` /
    ``_version`` on op-output views (r31/r33, re-verified r65), and a view's autograd state
    (``is_leaf`` False, fresh ``_version``, own ``grad`` slot) is not a pure function of the
    slot's canonical form. Direct reads -- the realistic ``self.w.requires_grad`` /
    ``self.b._version`` spellings -- are fully witnessed; the alias/view spellings are the
    state twin of the input net's leaf-only residual (contract residual note).
    """

    assert _STATE_METADATA_DIRECT_ONLY_NAMES == frozenset(
        {
            "requires_grad",
            "grad_fn",
            "is_leaf",
            "retains_grad",
            "_base",
            "_is_view",
            "output_nr",
            "grad",
            "_grad",
            "_version",
        }
    )
    # Together with the alias-safe family, the layout trio, the storage-geometry fact, and
    # the single structural row, the two attribution families tile the mirror exactly.
    layout_and_geometry = {"is_contiguous", "stride", "storage_offset", "storage_nbytes"}
    structural = {"is_coalesced"}
    assert (
        _STATE_METADATA_DIRECT_ONLY_NAMES
        | _STATE_METADATA_ALIAS_SAFE_STATE_NAMES
        | layout_and_geometry
        | structural
    ) == set(STATE_METADATA_MIRROR)


# ======================================================================================
# T-X3 -- staging canonicality: every tripwire dim is staging-guaranteed
# ======================================================================================


@pytest.mark.smoke
def test_r65_staged_state_satisfies_full_signature(tmp_path: Path) -> None:
    """Staged clones (user-bound AND prepared) exhibit ZERO full-signature violations.

    Pins that every PHYSICAL_SCOPE dim -- including the ten r65 additions -- is
    staging-guaranteed BEFORE it is tripwire-enforced, so the unconditional staged-runtime
    tripwire can never fire on TorchLens's own staging output.
    """

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 3))
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w.sum() + self.b

    x = torch.randn(3)
    loaded = tl.load(_save(_trace(Plain(), x), tmp_path / "plain.tlspec"))
    loaded.load_state_dict({"w": torch.ones(3, 2).t(), "b": torch.arange(8.0)[2:5]})
    for name, value in loaded.__dict__["_runnable_staged_user_state"].items():
        assert state_metadata_full_violations(value) == [], name
    prepared = prepare_runnable_state(loaded)
    for slot_id, value in prepared.slot_values.items():
        assert state_metadata_full_violations(value) == [], slot_id


def test_r65_staging_inside_inference_mode_stays_canonical(tmp_path: Path) -> None:
    """Binding/preparing state INSIDE an inference_mode region cannot mint inference clones."""

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w.sum()

    x = torch.randn(3)
    loaded = tl.load(_save(_trace(Plain(), x), tmp_path / "plain.tlspec"))
    with torch.inference_mode():
        loaded.load_state_dict({"w": torch.ones(2, 3)})
    for name, value in loaded.__dict__["_runnable_staged_user_state"].items():
        assert not value.is_inference(), name
        assert state_metadata_full_violations(value) == [], name


# ======================================================================================
# T-X4 -- behavioral matrix: read + non-canonical -> refuse; canonical read -> VERIFIED;
#         non-canonical UNREAD -> VERIFIED (the per-dim Option-A zero-collateral pin)
# ======================================================================================


def _shared_buffer() -> torch.Tensor:
    buffer = torch.arange(3.0)
    buffer.share_memory_()
    assert buffer.is_shared()
    return buffer


def _inference_buffer() -> torch.Tensor:
    with torch.inference_mode():
        buffer = torch.arange(3.0)
    assert buffer.is_inference()
    return buffer


def _view_buffer() -> torch.Tensor:
    base = torch.arange(3.0)
    buffer = base[:]  # full view: is_view True, _base set, storage stays TIGHT
    assert buffer._is_view() and buffer._base is not None
    return buffer


def _big_base_view_buffer() -> torch.Tensor:
    # The r64 F3 construction: offset 0, contiguous, default stride -- every r63 dim
    # canonical -- yet the BASE storage is far larger than numel * element_size.
    buffer = torch.arange(100.0)[:3]
    assert buffer.storage_offset() == 0 and buffer.is_contiguous()
    assert buffer.untyped_storage().nbytes() > buffer.numel() * buffer.element_size()
    return buffer


def _versioned_buffer() -> torch.Tensor:
    buffer = torch.arange(3.0)
    buffer.add_(1.0)  # pre-capture in-place mutation: _version now non-default
    assert buffer._version > 0
    return buffer


def _nonleaf_buffer() -> torch.Tensor:
    root = torch.ones(3, requires_grad=True)
    buffer = root * 2.0  # carries grad_fn: is_leaf False
    assert not buffer.is_leaf and buffer.grad_fn is not None
    return buffer


def _retains_grad_buffer() -> torch.Tensor:
    root = torch.ones(3, requires_grad=True)
    buffer = root * 2.0
    buffer.retain_grad()
    assert buffer.retains_grad
    return buffer


def _grad_param() -> nn.Parameter:
    param = nn.Parameter(torch.ones(3))
    param.grad = torch.zeros(3)
    return param


class _BufferReadModel(nn.Module):
    """Register one buffer and branch on one metadata accessor read of it."""

    def __init__(self, buffer: torch.Tensor, reader: str) -> None:
        super().__init__()
        self.register_buffer("b", buffer)
        self._reader = reader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.b
        flag = {
            "is_shared": lambda: b.is_shared(),
            "is_inference": lambda: b.is_inference(),
            "_is_view": lambda: b._is_view(),
            "_base": lambda: b._base is not None,
            "is_leaf": lambda: not b.is_leaf,
            "retains_grad": lambda: b.retains_grad,
            "_version": lambda: b._version > 0,
            "storage_nbytes": lambda: b.untyped_storage().nbytes() > 12,
            "none": lambda: False,
        }[self._reader]()
        return x - b.detach().sum() if flag else x + b.detach().sum()


_REFUSE_CASES = (
    ("is_shared", _shared_buffer, "is_shared"),
    ("is_inference", _inference_buffer, "is_inference"),
    ("_is_view", _view_buffer, "is_view"),
    ("_base", _view_buffer, "is_view"),
    ("is_leaf", _nonleaf_buffer, "is_leaf"),
    ("retains_grad", _retains_grad_buffer, "retains_grad"),
    ("_version", _versioned_buffer, "version_is_zero"),
    ("storage_nbytes", _big_base_view_buffer, "storage_nbytes_is_tight"),
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "reader, builder, dim", _REFUSE_CASES, ids=[case[0] for case in _REFUSE_CASES]
)
def test_r65_read_noncanonical_state_refuses_at_save(
    reader: str, builder, dim: str, tmp_path: Path
) -> None:
    """Each newly mirrored accessor read on a non-canonical captured slot refuses typed.

    Pre-r65 every one of these saved, transport normalized the dim away, and the loaded
    replay reported a false VERIFIED while a fresh oracle flips the branch.
    """

    trace = _trace(_BufferReadModel(builder(), reader), torch.randn(3))
    assert "b" in host_escape_state_metadata_reads(trace)
    _assert_refuses(trace, tmp_path / "refuse.tlspec", "b", dim)


@pytest.mark.parametrize(
    "reader",
    [
        "is_shared",
        "is_inference",
        "_is_view",
        "_base",
        "is_leaf",
        "retains_grad",
        "_version",
        "storage_nbytes",
    ],
)
def test_r65_canonical_state_read_stays_verified(reader: str, tmp_path: Path) -> None:
    """The same read on a CANONICAL captured slot saves and settles VERIFIED (no over-refusal)."""

    _assert_verified(
        _BufferReadModel(torch.arange(3.0), reader), torch.randn(3), tmp_path / "canon.tlspec"
    )


@pytest.mark.parametrize(
    "builder",
    [_shared_buffer, _inference_buffer, _view_buffer, _big_base_view_buffer, _versioned_buffer],
    ids=["shared", "inference", "view", "big_base_view", "versioned"],
)
def test_r65_unread_noncanonical_state_stays_verified(builder, tmp_path: Path) -> None:
    """A non-canonical slot that is NEVER read keeps saving + VERIFIED (r63 zero collateral).

    The per-dim Option-A pin: the new dims are escape-GATED, so the honest unread population
    (shared/pinned/inference/view/versioned state used only in tensor math) is untouched.
    """

    _assert_verified(
        _BufferReadModel(builder(), "none"), torch.randn(3), tmp_path / "unread.tlspec"
    )


def test_r65_grad_presence_read_refuses_when_grad_present(tmp_path: Path) -> None:
    """``self.w.grad is None`` on a grad-carrying captured param refuses; grad-free VERIFIED."""

    class GradRead(nn.Module):
        def __init__(self, with_grad: bool) -> None:
            super().__init__()
            self.w = _grad_param() if with_grad else nn.Parameter(torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w.sum() if self.w.grad is None else x - self.w.sum()

    trace = _trace(GradRead(True), torch.randn(3))
    assert "grad_presence" in host_escape_state_metadata_reads(trace).get("w", frozenset())
    _assert_refuses(trace, tmp_path / "grad.tlspec", "w", "grad_is_none")
    _assert_verified(GradRead(False), torch.randn(3), tmp_path / "nograd.tlspec")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_r65_pinned_read_refuses_when_pinned(tmp_path: Path) -> None:
    """``is_pinned()`` on a pinned captured buffer refuses (transport unpins)."""

    class PinnedRead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0).pin_memory())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x - self.b if self.b.is_pinned() else x + self.b

    trace = _trace(PinnedRead(), torch.randn(3))
    _assert_refuses(trace, tmp_path / "pinned.tlspec", "b", "is_pinned")


# ======================================================================================
# T-X5 -- the F2 immunizer: zero-copy view exports pin state layout
# ======================================================================================


class _NumpyFlagsRead(nn.Module):
    """The r64 F2 probe verbatim: branch on ``numpy()`` layout flags of a square buffer."""

    def __init__(self, transposed: bool) -> None:
        super().__init__()
        weight = torch.arange(9.0).reshape(3, 3)
        self.register_buffer("b", weight.t() if transposed else weight.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arr = self.b.detach().numpy()
        return x - self.b.sum() if not arr.flags["C_CONTIGUOUS"] else x + self.b.sum()


@pytest.mark.smoke
def test_r65_numpy_view_export_on_noncanonical_state_refuses(tmp_path: Path) -> None:
    """``numpy()`` off a transposed captured buffer refuses: the ndarray pins the layout."""

    trace = _trace(_NumpyFlagsRead(True), torch.randn(3))
    kinds = host_escape_state_metadata_reads(trace).get("b", frozenset())
    assert {"stride_exact", "storage_offset"} <= kinds
    _assert_refuses(trace, tmp_path / "f2.tlspec", "b", "stride_is_default")


def test_r65_numpy_view_export_on_canonical_state_stays_verified(tmp_path: Path) -> None:
    """The same ``numpy()`` read on a contiguous twin saves + VERIFIED (no over-refusal)."""

    _assert_verified(_NumpyFlagsRead(False), torch.randn(3), tmp_path / "f2c.tlspec")


def test_r65_array_and_dlpack_spellings_record_view_geometry() -> None:
    """``__array__`` / ``to_dlpack`` / ``from_dlpack`` spellings record the same read kinds."""

    class ArraySpelling(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            np.asarray(self.b.detach())
            return x + self.b

    class DlpackSpelling(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch.utils.dlpack.to_dlpack(self.b)
            return x + self.b

    for model_cls in (ArraySpelling, DlpackSpelling):
        trace = _trace(model_cls(), torch.randn(3))
        kinds = host_escape_state_metadata_reads(trace).get("b", frozenset())
        assert {"stride_exact", "storage_offset"} <= kinds, model_cls.__name__


# ======================================================================================
# T-X6 -- the F-1 immunizer: requires_grad / grad_fn presence = declared-state facts
# ======================================================================================


class _FrozenRequiresGradRead(nn.Module):
    """The r64 F1 scenario: a frozen model branching on its own trainable bit."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.lin.weight.requires_grad_(False)
        self.lin.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lin.weight.requires_grad:
            return self.lin(x) * 2
        return self.lin(x)


@pytest.mark.smoke
def test_r65_requires_grad_read_is_declared_fact_never_refusal(tmp_path: Path) -> None:
    """A ``requires_grad`` state read records a fact, saves, runs VERIFIED -- no ceiling.

    The locked F-1 ruling: the trainable bit is DECLARED state staging reproduces, so the
    frozen population never refuses and never ceilings. The recorded bit is whatever the
    captured forward actually read (TorchLens's grad-capture machinery runs the forward with
    param ``requires_grad`` temporarily enabled -- a pre-existing capture behavior), and the
    pin is exact reproduction: staged bit == recorded bit, with a user-supplied state dict
    carrying a different bit still staging the RECORDED one.
    """

    x = torch.randn(4, 3)
    trace = _trace(_FrozenRequiresGradRead(), x)
    facts = host_escape_state_metadata_facts(trace)
    assert "requires_grad" in facts.get("lin.weight", {})
    # The fact route never joins the escape read-kind ledger (no refusal machinery).
    assert "lin.weight" not in host_escape_state_metadata_reads(trace)
    path = _save(trace, tmp_path / "f1.tlspec")
    loaded = tl.load(path)
    descriptor = loaded.__dict__["_runnable_descriptor"]
    recorded = recorded_state_metadata_facts(descriptor)
    assert "requires_grad" in recorded.get("lin.weight", {})
    recorded_bit = recorded["lin.weight"]["requires_grad"]
    name_by_slot = {
        slot.slot_id: slot.state_binding.state_dict_name
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
    }
    prepared = prepare_runnable_state(loaded)
    staged_bits = {
        name_by_slot[slot_id]: value.requires_grad
        for slot_id, value in prepared.slot_values.items()
        if slot_id in name_by_slot
    }
    assert staged_bits["lin.weight"] == recorded_bit
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    # Recorded bit WINS over a user-supplied state dict carrying the opposite bit.
    flipped = tl.load(path)
    user_state = {
        name: torch.ones_like(value).requires_grad_(value.is_floating_point() and not recorded_bit)
        for name, value in _FrozenRequiresGradRead().state_dict().items()
    }
    flipped.load_state_dict(user_state)
    prepared_flipped = prepare_runnable_state(flipped)
    staged_flipped = {
        name_by_slot[slot_id]: value.requires_grad
        for slot_id, value in prepared_flipped.slot_values.items()
        if slot_id in name_by_slot
    }
    assert staged_flipped["lin.weight"] == recorded_bit


def test_r65_unread_bit_records_no_fact(tmp_path: Path) -> None:
    """A model that never reads state autograd metadata emits ZERO fact witnesses.

    The read-gated pin (and the internal-read-marker meta-pin): TorchLens's own per-op
    bookkeeping ``requires_grad``/``grad_fn``/``_version`` reads on state receivers are
    marked internal at their source, so the fact route stays sparse -- no witness bloat,
    zero artifact growth for the oblivious population.
    """

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 3)
            self.bn = nn.BatchNorm1d(3)
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(self.lin(x)) + self.b

    trace = _trace(Plain(), torch.randn(4, 3))
    assert host_escape_state_metadata_facts(trace) == {}
    assert host_escape_state_metadata_reads(trace) == {}
    loaded = tl.load(_save(trace, tmp_path / "plain.tlspec"))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert not [
        witness
        for witness in descriptor.control_witnesses
        if witness.site_label.startswith("state_metadata:")
    ]


def test_r65_grad_fn_present_state_read_refuses(tmp_path: Path) -> None:
    """A ``grad_fn``-presence read that observed True refuses: no staged leaf carries one."""

    class GradFnRead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", _nonleaf_buffer())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x - self.b.detach() if self.b.grad_fn is not None else x + self.b.detach()

    trace = _trace(GradFnRead(), torch.randn(3))
    assert host_escape_state_metadata_facts(trace).get("b", {}).get("grad_fn") is True
    _assert_refuses(trace, tmp_path / "gradfn.tlspec", "b", "grad_fn_present")


def test_r65_grad_fn_absent_state_read_stays_verified(tmp_path: Path) -> None:
    """The same read on an ordinary leaf param records fact False and stays VERIFIED."""

    class GradFnRead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x - self.w if self.w.grad_fn is not None else x + self.w

    x = torch.randn(3)
    trace = _trace(GradFnRead(), x)
    assert host_escape_state_metadata_facts(trace).get("w", {}).get("grad_fn") is False
    loaded = tl.load(_save(trace, tmp_path / "gradfn_ok.tlspec"))
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# Parse-side validation: the closed fact vocabulary is enforced before staging
# ======================================================================================


def _fact_witness(site_label: str, fact: object) -> ControlWitness:
    return ControlWitness(
        witness_id="witness:1",
        kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
        order=0,
        call_id=None,
        site_label=site_label,
        observed_value=io_runnable._encode_literal(fact),
    )


@pytest.mark.smoke
def test_r65_parse_validation_enforces_closed_fact_vocabulary() -> None:
    """Malformed persisted facts refuse at PARSE (``context_field_invalid``-class)."""

    good = _fact_witness(
        "state_metadata:w", {"state_metadata": True, "state": "w", "facts": {"requires_grad": True}}
    )
    runnable_load._validate_state_metadata_fact_witnesses((good,))
    bad_cases = [
        # Unknown fact name (outside the closed two-name vocabulary).
        {"state_metadata": True, "state": "w", "facts": {"stride": True}},
        # Non-bool fact value (run prep would apply an attacker-typed object).
        {"state_metadata": True, "state": "w", "facts": {"requires_grad": 1}},
        # Empty facts mapping.
        {"state_metadata": True, "state": "w", "facts": {}},
        # Missing state name.
        {"state_metadata": True, "facts": {"requires_grad": True}},
        # Malformed envelope.
        {"other": True},
    ]
    for fact in bad_cases:
        with pytest.raises(runnable_load.ContextFieldInvalidError):
            runnable_load._validate_state_metadata_fact_witnesses(
                (_fact_witness("state_metadata:w", fact),)
            )
    # Site label / embedded name disagreement.
    with pytest.raises(runnable_load.ContextFieldInvalidError):
        runnable_load._validate_state_metadata_fact_witnesses(
            (
                _fact_witness(
                    "state_metadata:other",
                    {"state_metadata": True, "state": "w", "facts": {"requires_grad": True}},
                ),
            )
        )


# ======================================================================================
# Residual pin -- identity-aliasing stays a residual, direct reads stay witnessed
# ======================================================================================


def test_r65_contiguous_identity_alias_is_documented_residual(tmp_path: Path) -> None:
    """``w.contiguous() is w`` (pure ``is`` identity) records nothing -- locked residual.

    No accessor fires on any tensor, so no Python-level net can see it; closing it would
    refuse every aliasing-polymorphic consumption of non-canonical state and break the r63
    channels-last zero-collateral pledge. The pin: the capture stays honest-but-unrefused
    (saves; the identity decision is invisible), and this test documents the boundary.
    """

    class IdentityAlias(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.w.contiguous()
            return x + self.w.sum() if y is self.w else x - self.w.sum()

    x = torch.randn(3)
    trace = _trace(IdentityAlias(), x)
    assert host_escape_state_metadata_reads(trace) == {}
    _save(trace, tmp_path / "residual.tlspec")
