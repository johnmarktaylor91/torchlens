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
    _ORACLE_POLICY_CLASSES,
    _STATE_METADATA_BIND_SCOPE,
    _STATE_METADATA_OBSERVED_PLACEMENT_KINDS,
    _STATE_METADATA_PHYSICAL_SCOPE,
    _STATE_METADATA_READ_REQUIRED_DIMS,
    ORACLE_POLICY_DECLARED_REPRODUCED,
    ORACLE_POLICY_ORACLE_CANONICAL,
    ORACLE_POLICY_REFUSE_ON_ANY_READ,
    ORACLE_POLICY_STRUCTURALLY_COVERED,
    STATE_METADATA_ORACLE_POLICY,
    _state_metadata_signature,
    _state_placement_canonical,
    prepare_runnable_state,
    recorded_state_metadata_facts,
    state_metadata_full_violations,
    state_metadata_read_violations,
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
            # r67 C4: every read kind is either an oracle_canonical row that resolves to a
            # signature dim, or a refuse_on_any_read row (``_version``) that terminates in
            # the unconditional producer refusal instead of a dim.
            policy = STATE_METADATA_ORACLE_POLICY.get(detail)
            if policy == ORACLE_POLICY_REFUSE_ON_ANY_READ:
                assert detail not in _STATE_METADATA_READ_REQUIRED_DIMS, name
                assert state_metadata_read_violations(None, [detail]) == [
                    (detail, "<refuse_on_any_read>", None)
                ], name
                continue
            assert policy == ORACLE_POLICY_ORACLE_CANONICAL, name
            if detail in _STATE_METADATA_OBSERVED_PLACEMENT_KINDS:
                # r67 C3: observed-value rows terminate in the observation predicate,
                # not a signature dim -- a missing/unknown observation refuses.
                assert detail not in _STATE_METADATA_READ_REQUIRED_DIMS, name
                assert state_metadata_read_violations(
                    _state_metadata_signature(torch.zeros(2)), [detail], {}
                ) == [(detail, "<observed_placement>", None)], name
                continue
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA staged slots require CUDA")
def test_r65_cuda_staged_state_satisfies_full_signature(tmp_path: Path) -> None:
    """Staged slots on a NON-CPU (CUDA) slot device exhibit ZERO full-signature violations.

    The r65 X regap immunizer: canonicality must be DEVICE-DERIVED -- what a fresh staging
    clone on the SLOT device actually produces. ``is_shared()`` is a device constant True on
    CUDA (not a user sharing signal), so a device-blind canonical-False pin false-fired the
    unconditional staged-runtime tripwire on every CUDA slot device (false
    ``PathDivergenceError`` on honest replays). Pins both staging paths -- embedded capture
    state AND a CPU user state dict staged to the CUDA slot device -- plus the end-to-end
    VERIFIED settle, so any future device-blind canonicality dim goes RED here, not only in
    the r36 regression suite.
    """

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 3))
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.w.sum() + self.b

    x = torch.randn(3, device="cuda")
    path = _save(_trace(Plain().cuda(), x), tmp_path / "cuda.tlspec")
    loaded = tl.load(path)
    prepared = prepare_runnable_state(loaded)
    assert prepared.slot_values
    for slot_id, value in prepared.slot_values.items():
        assert value.device.type == "cuda", slot_id
        assert state_metadata_full_violations(value) == [], slot_id
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    # A CPU user state dict staged ACROSS devices to the CUDA slot stays canonical too.
    rebound = tl.load(path)
    rebound.load_state_dict({"w": torch.ones(2, 3), "b": torch.arange(3.0)})
    prepared_rebound = prepare_runnable_state(rebound)
    for slot_id, value in prepared_rebound.slot_values.items():
        assert value.device.type == "cuda", slot_id
        assert state_metadata_full_violations(value) == [], slot_id
    rerun = rebound.run(inputs=x.clone())
    assert rerun.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA state read")
def test_r65_cuda_is_shared_read_stays_verified(tmp_path: Path) -> None:
    """A CUDA ``is_shared()`` state read saves and settles VERIFIED -- no read-divergence.

    On CUDA both the captured slot and the staged clone report the device constant True, so
    the read is reproduced exactly by staging: refusing it would be pure over-refusal, and
    the escape-gated producer check must see the slot as canonical BY DEVICE. The CPU twin
    (``share_memory_()`` backing read then staged-unshared) still refuses -- pinned by
    ``test_r65_read_noncanonical_state_refuses_at_save[is_shared]`` above.
    """

    model = _BufferReadModel(torch.arange(3.0), "is_shared").cuda()
    assert model.b.is_shared()  # the device constant, not user-chosen sharing
    x = torch.randn(3, device="cuda")
    trace = _trace(model, x)
    assert "is_shared" in host_escape_state_metadata_reads(trace).get("b", frozenset())
    loaded = tl.load(_save(trace, tmp_path / "cuda_shared_read.tlspec"))
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


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
    ("_version", _versioned_buffer, "refuse_on_any_read"),
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
        "storage_nbytes",
    ],
)
def test_r65_canonical_state_read_stays_verified(reader: str, tmp_path: Path) -> None:
    """The same read on a CANONICAL captured slot saves and settles VERIFIED (no over-refusal).

    ``_version`` is deliberately ABSENT here (r67 C4): it has NO canonical value -- oracle-1's
    default copy leaves 1 on a plain slot and 2 on an initialized-module slot, never the fresh
    constructor's 0 -- so a version-0 capture read refuses exactly like a versioned one
    (pinned by ``test_r67_version_read_refuses_for_both_construction_histories``).
    """

    _assert_verified(
        _BufferReadModel(torch.arange(3.0), reader), torch.randn(3), tmp_path / "canon.tlspec"
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "builder", [lambda: torch.arange(3.0), _versioned_buffer], ids=["version_zero", "versioned"]
)
def test_r67_version_read_refuses_for_both_construction_histories(builder, tmp_path: Path) -> None:
    """ANY attributed ``_version`` read refuses at save -- version-0 AND mutated captures alike.

    r66 hon1-F3: ``version_is_zero`` described a TorchLens-engineered staging clone, not
    oracle-1 -- the default ``load_state_dict(strict=True, assign=False)`` copy increments
    constructor-owned counters (0 -> 1 plain, 1 -> 2 initialized), so a captured
    ``_version == 0`` branch saved+VERIFIED yet the contract's OWN oracle could never read 0.
    ``_version`` is now ``refuse_on_any_read``: no construction history makes the read
    reproducible.
    """

    trace = _trace(_BufferReadModel(builder(), "_version"), torch.randn(3))
    assert "_version" in host_escape_state_metadata_reads(trace).get("b", frozenset())
    _assert_refuses(trace, tmp_path / "version.tlspec", "b", "refuse_on_any_read")


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
    """No-read models record ZERO read facts; emission is TOTALIZED per name (r71 E1).

    The internal-read-marker meta-pin survives r71: TorchLens's own per-op
    bookkeeping ``requires_grad``/``grad_fn``/``_version`` reads on state receivers
    are marked internal at their source, so the READ-fact tables stay empty for the
    oblivious population. r71 totalizes EMISSION over the declared state-name
    universe (presence is the anchor now): every declared name owns exactly one
    ``state_metadata`` witness carrying its capture-time ``requires_grad`` truth and
    ``grad_fn=False`` -- the parse-time domain totality that closes the r70
    matched-pair strip (hon1-F2).
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
    facts = recorded_state_metadata_facts(descriptor)
    declared_names = {
        binding.state_dict_name
        for slot in descriptor.tensor_slots
        if (binding := slot.state_binding) is not None
    }
    assert set(facts) == declared_names
    assert facts["lin.weight"] == {"grad_fn": False, "requires_grad": True}
    assert facts["b"] == {"grad_fn": False, "requires_grad": False}
    assert facts["bn.running_mean"]["requires_grad"] is False
    assert all(name_facts["grad_fn"] is False for name_facts in facts.values())


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


# ======================================================================================
# r67 C4 -- THE shared Oracle-1 post-copy parity matrix + the closed oracle-policy table
# ======================================================================================
#
# hon1-F3 root cause: ``version_is_zero`` described a TorchLens-ENGINEERED staging clone,
# not Oracle 1 -- the locked doctrine's fresh instance + default
# ``load_state_dict(strict=True, assign=False)`` copy. Default copy increments
# constructor-owned counters (0 -> 1 plain slots, 1 -> 2 initialized modules), so a
# version-0 canonical could save+VERIFY a branch the contract's OWN oracle can never take.
# The structural fix: every read-gated canonical dim must equal what ORACLE-1 actually
# produces (this matrix), and a dim with no artifact-independent oracle value is
# ``refuse_on_any_read`` -- no static expected scalar may exist without an oracle probe
# recipe exercised here.

from torchlens._runnable_state import _staged_state_clone  # noqa: E402


class _OracleProbeModule(nn.Module):
    """Fresh-instance oracle destination carrying every probe slot family.

    Plain zero-version parameter, initialized Linear/BatchNorm slots (constructor version
    1), a tied/alias parameter pair, a persistent buffer, and a non-persistent buffer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.plain = nn.Parameter(torch.zeros(2, 3))
        self.lin = nn.Linear(3, 3)
        self.bn = nn.BatchNorm1d(3)
        self.tied_a = nn.Parameter(torch.zeros(3))
        self.tied_b = self.tied_a  # tied/alias group: one allocation, two canonical names
        self.register_buffer("persistent_buf", torch.zeros(3))
        self.register_buffer("nonpersistent_buf", torch.zeros(3), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - probe only
        return x


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if device == "xpu":
        xpu = getattr(torch, "xpu", None)
        return (
            xpu is not None and callable(getattr(xpu, "is_available", None)) and xpu.is_available()
        )
    return False


def _probe_source_states(device: str) -> dict[str, dict[str, torch.Tensor]]:
    """Probe state variants: canonical, larger-base view, mutated-version, shared-memory."""

    base = {
        name: value.detach().clone().to(device)
        for name, value in _OracleProbeModule().state_dict().items()
    }
    variants: dict[str, dict[str, torch.Tensor]] = {"canonical": dict(base)}
    larger = dict(base)
    larger["plain"] = torch.arange(100.0, device=device)[:6].reshape(2, 3)
    assert larger["plain"].untyped_storage().nbytes() > 6 * larger["plain"].element_size()
    variants["larger_base_view"] = larger
    versioned = dict(base)
    mutated = base["lin.weight"].clone()
    mutated.add_(1.0)
    versioned["lin.weight"] = mutated
    variants["mutated_version"] = versioned
    if device == "cpu":
        shared = dict(base)
        shared_buf = base["persistent_buf"].clone().share_memory_()
        assert shared_buf.is_shared()
        shared["persistent_buf"] = shared_buf
        variants["shared_memory"] = shared
    return variants


@pytest.mark.smoke
def test_r67_oracle_policy_table_is_closed_and_total() -> None:
    """Every state-metadata row carries exactly one policy from the closed 4-class vocabulary.

    ``oracle_canonical`` rows resolve to a signature dim through the required-dims table;
    ``refuse_on_any_read`` rows (``_version``) deliberately do NOT -- there is no dim to
    compare, only the unconditional producer refusal.
    """

    assert set(STATE_METADATA_ORACLE_POLICY.values()) <= _ORACLE_POLICY_CLASSES
    expected_keys = set()
    for name, (route, detail) in STATE_METADATA_MIRROR.items():
        expected_keys.add(name if route == _STATE_ROUTE_STRUCTURAL else detail)
    assert set(STATE_METADATA_ORACLE_POLICY) == expected_keys
    for kind, policy in STATE_METADATA_ORACLE_POLICY.items():
        if policy == ORACLE_POLICY_ORACLE_CANONICAL:
            # An oracle_canonical row terminates in EXACTLY one validation mechanism:
            # a signature dim (pre-clone stamp) or an observed-value placement predicate.
            assert (kind in _STATE_METADATA_READ_REQUIRED_DIMS) != (
                kind in _STATE_METADATA_OBSERVED_PLACEMENT_KINDS
            ), kind
        else:
            assert kind not in _STATE_METADATA_READ_REQUIRED_DIMS, kind
            assert kind not in _STATE_METADATA_OBSERVED_PLACEMENT_KINDS, kind
    assert STATE_METADATA_ORACLE_POLICY["_version"] == ORACLE_POLICY_REFUSE_ON_ANY_READ
    assert STATE_METADATA_ORACLE_POLICY["requires_grad"] == ORACLE_POLICY_DECLARED_REPRODUCED
    assert STATE_METADATA_ORACLE_POLICY["grad_fn"] == ORACLE_POLICY_DECLARED_REPRODUCED
    assert STATE_METADATA_ORACLE_POLICY["is_coalesced"] == ORACLE_POLICY_STRUCTURALLY_COVERED


@pytest.mark.smoke
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "xpu"])
def test_r67_oracle1_post_copy_parity_matrix(device: str) -> None:
    """EVERY read-gated canonical dim equals what oracle-1's default copy actually produces.

    The shared machine-checked matrix: fresh instance + ``load_state_dict(strict=True,
    assign=False)`` over plain zero-version slots, initialized Linear/BatchNorm slots,
    tied/alias groups, larger-base views, shared-memory state, and persistent buffers;
    non-persistent buffers probe their reproduction mechanism (the staging clone). Every
    ``oracle_canonical`` row must satisfy its predicate on the REAL oracle destination --
    a static expected scalar with no probe observation here is RED.
    """

    if not _device_available(device):
        pytest.skip(f"device {device} unavailable")
    scope = _STATE_METADATA_BIND_SCOPE + _STATE_METADATA_PHYSICAL_SCOPE
    scope_dims = {dim for dim, _ in scope}
    observed_dims: set[str] = set()
    for variant_name, source_state in _probe_source_states(device).items():
        oracle = _OracleProbeModule().to(device)
        oracle.load_state_dict(source_state, strict=True, assign=False)
        entries = list(oracle.named_parameters(remove_duplicate=False))
        entries.extend(oracle.named_buffers(remove_duplicate=False))
        assert any(name == "tied_b" for name, _ in entries)  # alias group is in the walk
        for name, dest in entries:
            signature = _state_metadata_signature(dest)
            for dim, expected in scope:
                assert signature[dim] == expected, (variant_name, name, dim, signature[dim])
                observed_dims.add(dim)
            # r67 C3: the observed-value placement rows -- the REAL accessor reads on the
            # oracle destination must equal the device-defined predicate the producer
            # applies to user observations.
            for kind in sorted(_STATE_METADATA_OBSERVED_PLACEMENT_KINDS):
                canonical = _state_placement_canonical(kind, signature["device_type"])
                assert canonical is not None, (variant_name, name, kind)
                assert bool(getattr(dest, kind)()) == canonical, (variant_name, name, kind)
                observed_dims.add(kind)
    # Non-persistent buffers never ride oracle-1's copy; their reproduction mechanism is
    # the staging clone, which must satisfy the same rows.
    staged = _staged_state_clone(
        torch.arange(3.0, device=device), state_dict_name="nonpersistent_buf"
    )
    staged_signature = _state_metadata_signature(staged)
    for dim, expected in scope:
        assert staged_signature[dim] == expected, ("staged_nonpersistent", dim)
    for kind in sorted(_STATE_METADATA_OBSERVED_PLACEMENT_KINDS):
        canonical = _state_placement_canonical(kind, staged_signature["device_type"])
        assert bool(getattr(staged, kind)()) == canonical, ("staged_nonpersistent", kind)
    # No canonical dim without an oracle probe observation (the anti-static-scalar pin).
    assert observed_dims == scope_dims | _STATE_METADATA_OBSERVED_PLACEMENT_KINDS
    for kind, policy in STATE_METADATA_ORACLE_POLICY.items():
        if policy == ORACLE_POLICY_ORACLE_CANONICAL:
            if kind in _STATE_METADATA_OBSERVED_PLACEMENT_KINDS:
                assert kind in observed_dims
                continue
            dim, _expected = _STATE_METADATA_READ_REQUIRED_DIMS[kind]
            assert dim in observed_dims, kind


@pytest.mark.smoke
def test_r67_version_is_constructor_history_dependent_post_copy() -> None:
    """Oracle-1 default copy leaves DIFFERENT ``_version`` values per construction history.

    The recorded r66 evidence, machine-checked: a plain zero-version slot lands at 1 and an
    initialized Linear slot at 2 after the SAME default copy -- no artifact-independent
    scalar exists, so ``_version`` can only be ``refuse_on_any_read`` (never resurrect
    ``version_is_zero`` or any other static canonical).
    """

    oracle = _OracleProbeModule()
    source = {
        name: value.detach().clone() for name, value in _OracleProbeModule().state_dict().items()
    }
    assert oracle.plain._version == 0  # fresh constructor counter, pre-copy
    assert oracle.lin.weight._version > 0  # initializer already bumped it
    oracle.load_state_dict(source, strict=True, assign=False)
    assert oracle.plain._version > 0  # 0 -> 1: the oracle can never read 0
    assert oracle.lin.weight._version > oracle.plain._version  # 1 -> 2: history-dependent
    assert STATE_METADATA_ORACLE_POLICY["_version"] == ORACLE_POLICY_REFUSE_ON_ANY_READ


@pytest.mark.smoke
def test_r67_meta_destination_has_no_oracle_copy() -> None:
    """Meta-device destinations never receive oracle-1's default copy bytes.

    On current torch the default copy into a meta parameter is a warned NO-OP (older
    versions raise); either way NO oracle-1 destination value exists on meta, so the matrix
    honestly excludes meta rather than fabricating a canonical row.
    """

    with torch.device("meta"):
        oracle = _OracleProbeModule()
    source = {
        name: value.detach().clone() for name, value in _OracleProbeModule().state_dict().items()
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oracle.load_state_dict(source, strict=True, assign=False)
    except (RuntimeError, NotImplementedError):
        return  # raising torch versions: no destination exists, trivially no canonical
    entries = list(oracle.named_parameters(remove_duplicate=False))
    entries.extend(oracle.named_buffers(remove_duplicate=False))
    assert entries
    for name, dest in entries:
        assert dest.is_meta, name  # the copy was a no-op: no oracle value landed


# ======================================================================================
# r69 A -- state-metadata facts are inventory-indexed by exact (state, fact) identity
# (free-F1-secondary: a stripped declared fact silently omitted its staged bit)
# ======================================================================================


class _GradReadBranch(nn.Module):
    """Branch on a state requires_grad read (records one declared state fact)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.lin.weight.requires_grad:
            return self.lin(t)
        return t + 100.0


def _r69_save(tmp_path: Path, name: str) -> Path:
    trace = tl.trace(
        _GradReadBranch(),
        torch.randn(3),
        capture=_CAPTURE,
    )
    path = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _r69_mutate(path: Path, fn) -> None:
    import json

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fn(manifest["run"])
    manifest_path.write_text(json.dumps(manifest))


def _r69_assert_refused(path: Path) -> None:
    from torchlens.runnable import ReadinessStatus

    loaded = tl.load(path)
    readiness = loaded.__dict__.get("_runnable_readiness")
    assert readiness is not None
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert "context_field_invalid" in {d.code.value for d in readiness.diagnostics}
    with pytest.raises(Exception):
        loaded.run(inputs=torch.randn(3))


def test_r69_state_fact_strip_with_intact_inventory_refuses(tmp_path: Path) -> None:
    """Stripping a read-gated state fact whose identity the inventory declares refuses."""

    path = _r69_save(tmp_path, "strip_fact.tlspec")

    def _strip(run: dict) -> None:
        before = len(run["control_witnesses"])
        run["control_witnesses"] = [
            w
            for w in run["control_witnesses"]
            if not str(w.get("site_label", "")).startswith("state_metadata:")
        ]
        assert len(run["control_witnesses"]) < before

    _r69_mutate(path, _strip)
    _r69_assert_refused(path)


def test_r69_state_inventory_member_strip_with_intact_fact_refuses(tmp_path: Path) -> None:
    """Shrinking the state inventory while the fact survives breaks exact equality."""

    path = _r69_save(tmp_path, "strip_member.tlspec")

    def _shrink(run: dict) -> None:
        row = next(
            row
            for row in run["required_witness_inventory"]["families"]
            if row["family"] == "state_metadata"
        )
        assert row["members"], "state fact should have been read-gated-emitted"
        row["members"] = []

    _r69_mutate(path, _shrink)
    _r69_assert_refused(path)


def test_r69_forged_extra_state_fact_refuses(tmp_path: Path) -> None:
    """A forged state fact absent from the inventory refuses (no silent extra bit)."""

    path = _r69_save(tmp_path, "forge_fact.tlspec")

    def _forge(run: dict) -> None:
        import copy

        template = next(
            w
            for w in run["control_witnesses"]
            if str(w.get("site_label", "")).startswith("state_metadata:")
        )
        forged = copy.deepcopy(template)
        forged["witness_id"] = "witness:9999"
        forged["order"] = 9999
        forged["site_label"] = "state_metadata:lin.bias"
        value = forged["observed_value"]
        # Rewrite the embedded state name to match the forged site label.
        for entry in value["entries"]:
            if entry["key"].get("value") == "state":
                entry["value"]["value"] = "lin.bias"
        run["control_witnesses"] = list(run["control_witnesses"]) + [forged]

    _r69_mutate(path, _forge)
    _r69_assert_refused(path)


def test_r69_read_gated_emission_and_locked_staging_semantics_unchanged(
    tmp_path: Path,
) -> None:
    """r71 E1: emission is TOTALIZED (one member pair per declared name, explicit --
    never absence-means-default); the locked r65 F-1 application semantics (recorded
    requires_grad wins, capture truth staged) still hold."""

    class _NoRead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 3)

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.lin(t)

    trace = tl.trace(
        _NoRead(),
        torch.randn(3),
        capture=_CAPTURE,
    )
    path = tmp_path / "noread.tlspec"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    descriptor = tl.load(path).__dict__["_runnable_descriptor"]
    row = next(
        row
        for row in descriptor.required_witness_inventory.families
        if row.family == "state_metadata"
    )
    # Totalized domain: exact (name, fact) identity per declared name x both facts.
    assert row.members == (
        "lin.bias::grad_fn",
        "lin.bias::requires_grad",
        "lin.weight::grad_fn",
        "lin.weight::requires_grad",
    )
    facts = recorded_state_metadata_facts(descriptor)
    assert facts["lin.weight"] == {"grad_fn": False, "requires_grad": True}
    assert facts["lin.bias"] == {"grad_fn": False, "requires_grad": True}
    # Locked F-1: a read-gated capture still stages the recorded bit.
    read_path = _r69_save(tmp_path, "readgated.tlspec")
    loaded = tl.load(read_path)
    facts = recorded_state_metadata_facts(loaded.__dict__["_runnable_descriptor"])
    assert facts.get("lin.weight", {}).get("requires_grad") is True
    result = loaded.run(inputs=torch.randn(3))
    assert result.report.path_faithfulness.value == "verified"
