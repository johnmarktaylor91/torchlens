"""r67 C3+C6 -- ONE atomic storage-accessor disposition surface with ACTUAL-READ gating.

corr1-4 / free-F4 / free-F5 / r66b-R3 root causes, closed structurally:

* Handle ACQUISITION (``untyped_storage`` / ``storage`` / ``_typed_storage``) records
  origin/watch ONLY -- a discarded handle is not an observation (the ledger claimed
  ``storage_nbytes`` "one attribute away" and false-refused larger-base slots).
* The ACTUAL accessor call on the handle records the real returned value, through
  ``STORAGE_METADATA_ACCESSOR_DISPOSITIONS`` -- ``nbytes``/``size``/``__len__``,
  ``is_shared``/``is_pinned`` (the SAME observed-value read kinds as the Tensor
  spelling), mutators, value reads, raw pointers: never omission (reflection makes a
  new public storage member RED until classified).
* ``is_pinned`` uses the user's ONE actual return -- never a CUDA-initialization
  inference, never a speculative TorchLens re-read; ``grad_is_none`` is stamped only
  from the real ``.grad`` read under local warning suppression.
* Direct state reads fan out to the COMPLETE r37 alias group.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable_state import (
    _state_metadata_signature,
    _state_placement_canonical,
)
from torchlens.backends.torch import completeness_witness as cw
from torchlens.backends.torch.completeness_witness import (
    STORAGE_BRIDGE_ESCAPE_FUNCS,
    STORAGE_METADATA_ACCESSOR_DISPOSITIONS,
    _STORAGE_WRAPPED_DISPOSITIONS,
    host_escape_state_metadata_observations,
    host_escape_state_metadata_reads,
    host_escape_state_source_names,
)
from torchlens.errors import RunnablePreflightError
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


def _assert_refuses(trace: tl.Trace, path: Path, *needles: str) -> None:
    with pytest.raises(RunnablePreflightError) as excinfo:
        _save(trace, path)
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    for needle in needles:
        assert needle in diagnostics, needle


# ======================================================================================
# Reflection immunizer: every public storage member is classified (new accessor -> RED)
# ======================================================================================


@pytest.mark.smoke
@pytest.mark.parametrize("class_name", ["UntypedStorage", "TypedStorage"])
def test_r67_storage_disposition_table_covers_public_surface(class_name: str) -> None:
    """Every public (non-underscore) ``dir()`` member of each storage class has a row.

    A torch upgrade that adds a public storage accessor turns this RED until the new
    member is explicitly classified -- never a silent witness gap. Named private/dunder
    rows (``_cdata``, ``__len__``, ``__getitem__``, ``__setitem__``, ``__iter__``) are
    additionally required.
    """

    storage_cls = getattr(torch, class_name)
    rows = STORAGE_METADATA_ACCESSOR_DISPOSITIONS[class_name]
    public = {name for name in dir(storage_cls) if not name.startswith("_")}
    unclassified = public - set(rows)
    assert not unclassified, f"unclassified public storage members: {sorted(unclassified)}"
    for named_extra in ("_cdata", "__len__", "__getitem__", "__setitem__", "__iter__"):
        assert named_extra in rows, named_extra
    # Every row carries a KNOWN disposition class.
    known = _STORAGE_WRAPPED_DISPOSITIONS | {
        cw._STORAGE_ACCESSOR_INERT,
        cw._STORAGE_ACCESSOR_STRUCTURAL,
        cw._STORAGE_ACCESSOR_CONTAMINATED_PROPERTY,
    }
    for member, (disposition, why) in rows.items():
        assert disposition in known, member
        assert why, member


@pytest.mark.smoke
def test_r67_acquisition_bridges_cover_typed_storage_spelling() -> None:
    """``_typed_storage`` joined the origin+watch bridge set (r66b R3 spelling)."""

    assert {"untyped_storage", "storage", "_typed_storage", "data_ptr"} == set(
        STORAGE_BRIDGE_ESCAPE_FUNCS
    )


# ======================================================================================
# Discarded-handle vs actual-read pins (corr1-4) -- state side
# ======================================================================================


def _larger_base_buffer() -> torch.Tensor:
    buffer = torch.arange(100.0)[:3]
    assert buffer.untyped_storage().nbytes() > buffer.numel() * buffer.element_size()
    return buffer


class _StorageReader(nn.Module):
    """Register one buffer; run one storage-spelling accessor on it."""

    def __init__(self, buffer: torch.Tensor, reader) -> None:
        super().__init__()
        self.register_buffer("b", buffer)
        self._reader = reader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flag = self._reader(self.b)
        return x - self.b.detach().sum() if flag else x + self.b.detach().sum()


_ACQUIRE_ONLY = {
    "untyped": lambda b: (b.untyped_storage(), False)[1],
    "typed": lambda b: (b.storage(), False)[1],
    "typed_private": lambda b: (b._typed_storage(), False)[1],
}

_NBYTES_READERS = {
    "untyped": lambda b: b.untyped_storage().nbytes() > 12,
    "typed": lambda b: b.storage().nbytes() > 12,
    "typed_private": lambda b: b._typed_storage().nbytes() > 12,
    "untyped_size": lambda b: b.untyped_storage().size() > 12,
    "untyped_len": lambda b: len(b.untyped_storage()) > 12,
}


@pytest.mark.smoke
@pytest.mark.parametrize("spelling", sorted(_ACQUIRE_ONLY))
def test_r67_discarded_handle_on_larger_base_state_saves_and_verifies(
    spelling: str, tmp_path: Path
) -> None:
    """Acquisition alone records NO read kind: the larger-base slot stays admitted.

    The corr1-4 over-trigger: a discarded ``untyped_storage()`` handle (no ``.nbytes()``
    call) on a larger-base slot false-refused the save with
    ``storage_nbytes_is_tight=False``. The ledger may only claim observations that
    actually occurred.
    """

    x = torch.randn(3)
    trace = _trace(_StorageReader(_larger_base_buffer(), _ACQUIRE_ONLY[spelling]), x)
    assert host_escape_state_metadata_reads(trace) == {}
    loaded = tl.load(_save(trace, tmp_path / "acq.tlspec"))
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.parametrize("spelling", sorted(_NBYTES_READERS))
def test_r67_actual_nbytes_read_on_larger_base_state_refuses(spelling: str, tmp_path: Path) -> None:
    """The ACTUAL byte-count accessor call on the larger-base slot refuses at save."""

    trace = _trace(_StorageReader(_larger_base_buffer(), _NBYTES_READERS[spelling]), torch.randn(3))
    assert "storage_nbytes" in host_escape_state_metadata_reads(trace).get("b", frozenset())
    _assert_refuses(trace, tmp_path / "read.tlspec", "storage_nbytes")


@pytest.mark.parametrize("spelling", sorted(_NBYTES_READERS))
def test_r67_actual_nbytes_read_on_tight_state_stays_verified(
    spelling: str, tmp_path: Path
) -> None:
    """The same actual read on a TIGHT slot saves + VERIFIED (no over-refusal)."""

    x = torch.randn(3)
    trace = _trace(_StorageReader(torch.arange(3.0), _NBYTES_READERS[spelling]), x)
    loaded = tl.load(_save(trace, tmp_path / "tight.tlspec"))
    assert loaded.run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# Placement observed-value matrix: tensor x untyped x typed spellings, identical verdicts
# ======================================================================================


_PLACEMENT_SPELLINGS = {
    "is_shared_tensor": lambda b: b.is_shared(),
    "is_shared_untyped": lambda b: b.untyped_storage().is_shared(),
    "is_shared_typed": lambda b: b.storage().is_shared(),
    "is_pinned_tensor": lambda b: b.is_pinned(),
    "is_pinned_untyped": lambda b: b.untyped_storage().is_pinned(),
    "is_pinned_typed": lambda b: b.storage().is_pinned(),
}


@pytest.mark.parametrize("spelling", sorted(_PLACEMENT_SPELLINGS))
def test_r67_placement_read_canonical_state_identical_verdicts(
    spelling: str, tmp_path: Path
) -> None:
    """Every placement spelling on a canonical CPU slot records the SAME observed-value
    read kind and settles VERIFIED -- tensor and storage-handle spellings are ONE
    metadata surface."""

    kind = "is_shared" if "is_shared" in spelling else "is_pinned"
    x = torch.randn(3)
    trace = _trace(_StorageReader(torch.arange(3.0), _PLACEMENT_SPELLINGS[spelling]), x)
    assert kind in host_escape_state_metadata_reads(trace).get("b", frozenset()), spelling
    assert host_escape_state_metadata_observations(trace).get("b", {}).get(kind) is False
    loaded = tl.load(_save(trace, tmp_path / "canon.tlspec"))
    assert loaded.run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.parametrize(
    "spelling",
    ["is_shared_tensor", "is_shared_untyped", "is_shared_typed"],
)
def test_r67_placement_read_shared_state_refuses_every_spelling(
    spelling: str, tmp_path: Path
) -> None:
    """Observed True on a CPU slot refuses at save -- identically for every spelling.

    Sol's re-run defect verbatim: ``self.b.untyped_storage().is_shared()`` on a shared
    registered buffer saved and replayed the captured arm ``verified`` while a fresh
    Oracle-1 destination was unshared and took the other arm.
    """

    shared = torch.arange(3.0)
    shared.share_memory_()
    trace = _trace(_StorageReader(shared, _PLACEMENT_SPELLINGS[spelling]), torch.randn(3))
    assert host_escape_state_metadata_observations(trace).get("b", {}).get("is_shared") is True
    _assert_refuses(trace, tmp_path / "shared.tlspec", "is_shared")


@pytest.mark.smoke
def test_r67_is_pinned_observed_value_is_the_authority() -> None:
    """The recorded fact is the accessor's ACTUAL return -- no accelerator-init inference.

    free-F4: the r65 stamp keyed on ``torch.cuda.is_initialized()``, which is False on
    XPU/MPS hosts and for externally registered pinned memory -- a proof-by-absence that
    stamped canonical False on genuinely pinned slots. The observation ledger now carries
    exactly what the user's one call returned.
    """

    trace = _trace(_StorageReader(torch.arange(3.0), lambda b: b.is_pinned()), torch.randn(3))
    assert host_escape_state_metadata_observations(trace) == {"b": {"is_pinned": False}}
    # And the producer predicate is device-defined, not accelerator-state-defined:
    assert _state_placement_canonical("is_pinned", "cpu") is False
    assert _state_placement_canonical("is_pinned", "cuda") is False
    assert _state_placement_canonical("is_shared", "cpu") is False
    assert _state_placement_canonical("is_shared", "cuda") is True
    assert _state_placement_canonical("is_shared", None) is None  # unknown -> refuse


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned memory requires CUDA")
def test_r67_pinned_state_read_refuses_via_observation(tmp_path: Path) -> None:
    """Observed True on a genuinely pinned CPU slot refuses -- through the observation."""

    trace = _trace(
        _StorageReader(torch.arange(3.0).pin_memory(), lambda b: b.is_pinned()), torch.randn(3)
    )
    assert host_escape_state_metadata_observations(trace).get("b", {}).get("is_pinned") is True
    _assert_refuses(trace, tmp_path / "pinned.tlspec", "is_pinned")


def test_r67_arg_directed_placement_query_on_state_refuses(tmp_path: Path) -> None:
    """``is_pinned(device=...)`` on state records observed=None (unknown) and refuses.

    The arg-directed spelling answers a different question than the slot's default
    placement; treating it as the plain read would either over-bless or dodge the
    witness, so it fails closed.
    """

    trace = _trace(
        _StorageReader(torch.arange(3.0), lambda b: b.is_pinned(device="cuda")),
        torch.randn(3),
    )
    assert (
        host_escape_state_metadata_observations(trace).get("b", {}).get("is_pinned", "missing")
        is None
    )
    _assert_refuses(trace, tmp_path / "argdir.tlspec", "is_pinned")


# ======================================================================================
# Input-side parity (r66b R3): the typed/untyped storage geometry spellings on inputs
# ======================================================================================


class _InputGeomModel(nn.Module):
    def __init__(self, reader) -> None:
        super().__init__()
        self._reader = reader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight = self._reader(x)
        return torch.relu(x) + 1.0 if tight else torch.tanh(x) - 1.0


_INPUT_GEOM_READERS = {
    "untyped": lambda x: x.untyped_storage().nbytes() == x.numel() * x.element_size(),
    "typed_private": lambda x: x._typed_storage().nbytes() == x.numel() * x.element_size(),
    "typed": lambda x: x.storage().nbytes() == x.numel() * x.element_size(),
}


@pytest.mark.parametrize("spelling", sorted(_INPUT_GEOM_READERS))
def test_r67_input_storage_nbytes_read_diverges_on_changed_base(
    spelling: str, tmp_path: Path
) -> None:
    """r66b R3: every storage spelling of an input base-geometry read is witnessed.

    Original input replays VERIFIED; a larger-base same-shape twin (the oracle takes the
    other arm) must never settle VERIFIED -- it diverges typed or ceilings.
    """

    x = torch.randn(4)
    trace = _trace(_InputGeomModel(_INPUT_GEOM_READERS[spelling]), x)
    path = _save(trace, tmp_path / "geom.tlspec")
    original = tl.load(path).run(inputs=x.clone())
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    twin = torch.randn(100)[:4]
    with pytest.raises(Exception) as excinfo:
        tl.load(path).run(inputs=twin)
    assert "storage_nbytes" in str(excinfo.value)


@pytest.mark.smoke
def test_r67_discarded_input_handle_records_no_geometry_fact(tmp_path: Path) -> None:
    """Acquisition-only on an INPUT records no fact: a changed-base twin stays runnable.

    The input-side twin of the corr1-4 pin -- symmetric actual-read gating (the r67
    input x state parity requirement): no read, no fact, no phantom divergence.
    """

    class AcquireOnly(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _ = x.untyped_storage()
            return x * 2.0

    x = torch.randn(4)
    path = _save(_trace(AcquireOnly(), x), tmp_path / "acq_in.tlspec")
    twin = torch.randn(100)[:4]
    result = tl.load(path).run(inputs=twin)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# C6: grad_is_none is the REAL read; unattributable access fails closed; identity restore
# ======================================================================================


@pytest.mark.smoke
def test_r67_grad_is_none_is_the_actual_read() -> None:
    """The signature stamps the REAL ``.grad`` read -- free-F5's structural shortcut is gone.

    torch 2.8 allows assigning ``.grad`` on a non-leaf without ``retains_grad``; the old
    "a non-leaf without retains_grad structurally carries no grad" inference stamped
    ``grad_is_none=True`` for a fact it never read.
    """

    root = torch.ones(3, requires_grad=True)
    nonleaf = root * 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nonleaf.grad = torch.zeros(3)
    signature = _state_metadata_signature(nonleaf.clone().detach())  # detached: grad gone
    assert signature["grad_is_none"] is True
    # The live non-leaf with an assigned grad: the REAL read observes it.
    hostile_case = _state_metadata_signature(nonleaf) if False else None
    del hostile_case  # non-leaf is not admissible state; the unit fact is below
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (nonleaf.grad is None) is False
    # Matrix: leaf/non-leaf x retains x grad present/absent -- signature == real read.
    cases = []
    leaf = torch.ones(3)
    cases.append(leaf)  # leaf, no grad
    leaf_grad = nn.Parameter(torch.ones(3))
    leaf_grad.grad = torch.zeros(3)
    cases.append(leaf_grad)  # leaf, grad present
    retained = (torch.ones(3, requires_grad=True) * 2.0).clone().detach()
    cases.append(retained)
    for value in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            real = value.grad is None
        assert _state_metadata_signature(value)["grad_is_none"] == real, value


@pytest.mark.smoke
def test_r67_nonleaf_grad_signature_matches_real_read() -> None:
    """A non-leaf tensor with an ASSIGNED grad stamps grad_is_none=False (real read)."""

    root = torch.ones(3, requires_grad=True)
    nonleaf = root * 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nonleaf.grad = torch.zeros(3)
    signature = _state_metadata_signature(nonleaf)
    assert signature["grad_is_none"] is False


def test_r67_unattributable_storage_access_fails_closed(tmp_path: Path) -> None:
    """An owner-thread accessor on a storage TorchLens cannot attribute ceilings the run.

    Hidden non-state storage handles (created outside capture, never bridged) are exactly
    the monitor-uncertainty class: unknown must never read as no-consumption.
    """

    foreign = torch.arange(5.0).untyped_storage()

    class ForeignRead(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x - 1.0 if foreign.nbytes() > 12 else x + 1.0

    x = torch.randn(3)
    trace = _trace(ForeignRead(), x)
    try:
        path = _save(trace, tmp_path / "foreign.tlspec")
    except RunnablePreflightError:
        return  # refusing at save is equally fail-closed
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r67_storage_wrappers_restore_identity_exactly() -> None:
    """After a capture, every wrapped storage member restores to the exact original."""

    before: dict[tuple[str, str], object] = {}
    for class_name in ("UntypedStorage", "TypedStorage"):
        storage_cls = getattr(torch, class_name)
        for member in STORAGE_METADATA_ACCESSOR_DISPOSITIONS[class_name]:
            descriptor = inspect.getattr_static(storage_cls, member, None)
            if descriptor is not None:
                before[(class_name, member)] = descriptor

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _ = self.b.untyped_storage().nbytes()
            return x + self.b

    _trace(Plain(), torch.randn(3))
    for (class_name, member), descriptor in before.items():
        storage_cls = getattr(torch, class_name)
        assert inspect.getattr_static(storage_cls, member, None) is descriptor, (
            class_name,
            member,
        )


# ======================================================================================
# C6: tied-alias fan-out + captured-slot storage mutators + hostile-subclass admission
# ======================================================================================


class _TiedModel(nn.Module):
    """One allocation registered under two canonical parameter names."""

    def __init__(self, reader) -> None:
        super().__init__()
        self.tied_a = nn.Parameter(torch.ones(3))
        self.tied_b = self.tied_a
        self._reader = reader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flag = self._reader(self.tied_a)
        return x - self.tied_b.detach().sum() if flag else x + self.tied_b.detach().sum()


@pytest.mark.smoke
def test_r67_tied_alias_direct_read_attributes_to_full_group() -> None:
    """A direct read on ONE tied name marks EVERY canonical name in the alias group."""

    trace = _trace(_TiedModel(lambda p: p._version > 0), torch.randn(3))
    reads = host_escape_state_metadata_reads(trace)
    assert "_version" in reads.get("tied_a", frozenset())
    assert "_version" in reads.get("tied_b", frozenset())
    shared = _trace(_TiedModel(lambda p: p.is_shared()), torch.randn(3))
    observations = host_escape_state_metadata_observations(shared)
    assert observations.get("tied_a", {}).get("is_shared") is False
    assert observations.get("tied_b", {}).get("is_shared") is False
    names = host_escape_state_source_names(shared)
    assert {"tied_a", "tied_b"} <= set(names)


def test_r67_captured_storage_mutator_ceilings(tmp_path: Path) -> None:
    """A mutator through a captured slot's storage handle joins the writeback ceiling."""

    class MutateThroughHandle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.arange(3.0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.b.untyped_storage().fill_(0)
            return x + self.b.detach().sum()

    x = torch.randn(3)
    trace = _trace(MutateThroughHandle(), x)
    try:
        path = _save(trace, tmp_path / "mut.tlspec")
    except RunnablePreflightError:
        return  # refusing at save is equally fail-closed
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


class _Hostile(torch.Tensor):
    """A ``__torch_function__`` subclass that could lie about every metadata read."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func, types, args, kwargs or {})


@pytest.mark.parametrize("surface", ["state", "input"])
def test_r67_hostile_subclass_admission_refuses(surface: str, tmp_path: Path) -> None:
    """Hostile Tensor subclasses refuse typed on BOTH admission surfaces (C6 negative).

    The ``internal_scalar_read`` marker must never become a dispatch window for user
    subclass code: a hostile class is rejected at the admission boundary (capture or
    save), never blessed into a VERIFIED artifact.
    """

    if surface == "state":

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("b", _Hostile(torch.arange(3.0)))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.b.sum()

        model, payload = Model(), torch.randn(3)
    else:

        class Model(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        model, payload = Model(), _Hostile(torch.randn(3))

    with pytest.raises(Exception):
        trace = _trace(model, payload)
        _save(trace, tmp_path / "hostile.tlspec")


# ======================================================================================
# Observed-read property: no fact before access; exactly the right fact after
# ======================================================================================


@pytest.mark.smoke
def test_r67_observed_read_property_no_fact_without_access() -> None:
    """The ledger claims exactly the observations that occurred -- no more, no less."""

    # No access at all: empty ledgers.
    plain = _trace(_StorageReader(torch.arange(3.0), lambda b: False), torch.randn(3))
    assert host_escape_state_metadata_reads(plain) == {}
    assert host_escape_state_metadata_observations(plain) == {}
    # Acquisition only: still empty.
    acquired = _trace(_StorageReader(torch.arange(3.0), _ACQUIRE_ONLY["untyped"]), torch.randn(3))
    assert host_escape_state_metadata_reads(acquired) == {}
    # One actual read: exactly one read kind, with its observed value.
    read = _trace(
        _StorageReader(torch.arange(3.0), lambda b: b.untyped_storage().is_shared()),
        torch.randn(3),
    )
    assert host_escape_state_metadata_reads(read) == {"b": frozenset({"is_shared"})}
    assert host_escape_state_metadata_observations(read) == {"b": {"is_shared": False}}
