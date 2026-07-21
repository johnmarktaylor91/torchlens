"""State-binder metadata/layout completeness (r63 C1) -- whole-class immunizers.

Round-62/63 (both labs, JMT Option-A ruling) proved two false-VERIFIED families on the
sparse runnable-``.tlspec`` state surface:

* **Captured physical form, READ then lost in transport**: a captured param/buffer with a
  non-canonical PHYSICAL form (non-contiguous stride, nonzero ``storage_offset``, conj/neg
  lazy bit) that the model READS and branches on -- the snapshot clone and the embedded
  codec NORMALIZE the form away, so the loaded replay reported ``verified`` on the captured
  arm while a fresh oracle flips. Root cause: the four metadata accessors routed ONLY
  through the model-input observer, which ignores registered state -- no witness existed
  (Sol's r63 v4 probe). r63 closes the attribution gaps (each read now attributes a state
  escape + a per-slot read-kind fact) and refuses the save ESCAPE-GATED
  (``state_metadata_mismatch`` at ``producer_state_metadata``): only a READ non-canonical
  dim refuses; the UNREAD non-canonical population (channels-last conv weights) stays
  saveable + ``verified`` -- zero collateral is the Option-A guarantee.
* **User/embedded staged state whose metadata steers the fresh oracle**: a named source
  into an unnamed capture survives oracle-1's default ``load_state_dict`` copy and steers
  the fresh oracle while replay stays on the unnamed path; sparse/quantized/nested sources
  make the default load RAISE. The bind gate refuses the LOAD-SURVIVING subset
  ``(class_category, layout, names, is_quantized, is_nested)`` atomically before staging --
  and ONLY that subset: physical dims are normalized by default copy, so channels-last /
  non-contiguous / offset USER sources into canonical slots must keep binding (over-refusal
  guard).

Plus the structural belts: pre-clone signature stamping (the clone itself is a
normalization site), the canonical staging clone (default-copy semantics), the
in-transaction ``state_metadata:<slot_id>`` tripwire, and the source tripwire that keeps
every state-binding clone routed through the staging helper.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import (
    PathDivergenceError,
    RunnablePreflightError,
    StateBindingError,
)
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=_CAPTURE)


def _save(trace: tl.Trace, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace.save(path, level="runnable", include_weights=True)
    return path


def _save_load(model: nn.Module, x: torch.Tensor, path: Path) -> tl.Trace:
    return tl.load(_save(_trace(model, x), path))


def _noncontig_weight() -> torch.Tensor:
    weight = torch.arange(6.0).reshape(2, 3).t()  # stride (1, 3), non-contiguous
    assert not weight.is_contiguous()
    return weight.detach()


def _offset_buffer() -> torch.Tensor:
    buffer = torch.arange(8.0)[2:6]  # storage_offset 2
    assert buffer.storage_offset() == 2
    return buffer


def _conj_buffer() -> torch.Tensor:
    buffer = torch.ones(3, dtype=torch.cfloat).conj()
    assert buffer.is_conj()
    return buffer


def _neg_buffer() -> torch.Tensor:
    buffer = torch.randn(3, dtype=torch.cfloat).conj().imag  # real dtype, neg bit set
    assert buffer.is_neg()
    return buffer


# ======================================================================================
# The four READ physical dims -> state-escape witness emitted + save refused
# ======================================================================================


class IsContiguousReadParam(nn.Module):
    """Branch on ``self.w.is_contiguous()`` -- a non-contiguous captured param."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(_noncontig_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w.sum() if self.w.is_contiguous() else x - self.w.sum() * 2


class StrideReadParam(nn.Module):
    """Branch on ``self.w.stride()`` -- the exact-stride sibling read."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(_noncontig_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.w.sum() * 2 if self.w.stride() == (1, 3) else x + self.w.sum()


class StorageOffsetReadBuffer(nn.Module):
    """Branch on ``self.b.storage_offset()`` -- an offset captured buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", _offset_buffer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b.sum() if self.b.storage_offset() == 0 else x - self.b.sum()


class IsConjReadBuffer(nn.Module):
    """Branch on ``self.c.is_conj()`` -- a conj-bit captured buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", _conj_buffer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.c.real.sum() if self.c.is_conj() else x + self.c.real.sum()


class IsNegReadBuffer(nn.Module):
    """Branch on ``self.n.is_neg()`` -- the neg lazy-bit sibling."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("n", _neg_buffer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.n.sum() if self.n.is_neg() else x + self.n.sum()


_READ_CASES = (
    (IsContiguousReadParam, "w", "contiguous_default", torch.randn(3)),
    (StrideReadParam, "w", "stride_exact", torch.randn(3)),
    (StorageOffsetReadBuffer, "b", "storage_offset", torch.randn(4)),
    (IsConjReadBuffer, "c", "is_conj", torch.randn(3)),
    (IsNegReadBuffer, "n", "is_neg", torch.randn(3)),
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls, state_name, read_kind, x",
    _READ_CASES,
    ids=[case[2] for case in _READ_CASES],
)
def test_r63_state_metadata_read_emits_escape_witness(
    model_cls: type, state_name: str, read_kind: str, x: torch.Tensor
) -> None:
    """Each physical metadata read on registered state now attributes a state escape.

    The r62 gap: these reads recorded NOTHING (``host_escape_state_source_names`` empty,
    completeness COMPLETE, replay falsely VERIFIED). Post-fix each read joins the state
    escape names AND records its read kind for the producer gate.
    """

    from torchlens.backends.torch.completeness_witness import (
        host_escape_state_metadata_reads,
        host_escape_state_source_names,
    )

    trace = _trace(model_cls(), x)
    assert state_name in host_escape_state_source_names(trace)
    assert read_kind in host_escape_state_metadata_reads(trace).get(state_name, frozenset())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls, state_name, read_kind, x",
    _READ_CASES,
    ids=[case[2] for case in _READ_CASES],
)
def test_r63_read_noncanonical_capture_refuses_at_save(
    model_cls: type, state_name: str, read_kind: str, x: torch.Tensor, tmp_path: Path
) -> None:
    """A READ non-canonical captured physical form refuses the runnable save typed.

    Pre-fix these saved, transport normalized the form, and the loaded replay reported a
    false VERIFIED while the fresh oracle flipped (Sol r62/r63 probes verbatim).
    """

    trace = _trace(model_cls(), x)
    with pytest.raises(RunnablePreflightError) as excinfo:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace.save(tmp_path / "read.tlspec", level="runnable", include_weights=True)
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "state_metadata_mismatch" in diagnostics
    assert "producer_state_metadata" in diagnostics
    assert state_name in diagnostics


# ======================================================================================
# Zero collateral: the four UNREAD non-canonical captures save + VERIFIED
# ======================================================================================


class ChannelsLastConvUnread(nn.Module):
    """Channels-last conv weights, never inspected -- the Option-A protected population."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NonContigParamUnread(nn.Module):
    """Non-contiguous dense param used only in tensor math."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(_noncontig_weight())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.w * 2).sum()


class OffsetBufferUnread(nn.Module):
    """Offset buffer used only in tensor math."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", _offset_buffer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b


class ConjBufferUnread(nn.Module):
    """Conj-bit buffer used only in tensor math."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", _conj_buffer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c.real


_UNREAD_CASES = (
    (ChannelsLastConvUnread, torch.randn(1, 3, 8, 8)),
    (NonContigParamUnread, torch.randn(3)),
    (OffsetBufferUnread, torch.randn(4)),
    (ConjBufferUnread, torch.randn(3)),
)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls, x",
    _UNREAD_CASES,
    ids=[case[0].__name__ for case in _UNREAD_CASES],
)
def test_r63_unread_noncanonical_capture_saves_and_verifies(
    model_cls: type, x: torch.Tensor, tmp_path: Path
) -> None:
    """An UNREAD non-canonical capture must STILL save and settle VERIFIED (zero collateral).

    This is the regression guard distinguishing Option A from unconditional refusal: the
    physical form is destination-owned and value-invariant when unobserved, so refusing it
    would break honest channels-last CNNs.
    """

    loaded = _save_load(model_cls(), x, tmp_path / "unread.tlspec")
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# Bind gate: load-surviving subset refuses; excluded physical dims keep binding
# ======================================================================================


class PlainParam(nn.Module):
    """Canonical dense param -- the bind-gate destination."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w.sum()


def _plain_loaded(tmp_path: Path) -> tl.Trace:
    return _save_load(PlainParam(), torch.randn(3), tmp_path / "plain.tlspec")


def _named_source() -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.ones(2, 3).refine_names("out", "in")


class _CountingSubclass(torch.Tensor):
    """Hostile subclass: counts every metadata read routed through Python descriptors."""

    reads = 0

    @property
    def shape(self):  # type: ignore[override]
        type(self).reads += 1
        return super().shape

    @property
    def layout(self):  # type: ignore[override]
        type(self).reads += 1
        return super().layout

    @property
    def names(self):  # type: ignore[override]
        type(self).reads += 1
        return super().names

    def stride(self, *args, **kwargs):  # type: ignore[override]
        type(self).reads += 1
        return super().stride(*args, **kwargs)

    def is_contiguous(self, *args, **kwargs):  # type: ignore[override]
        type(self).reads += 1
        return super().is_contiguous(*args, **kwargs)


def _exotic_sources() -> dict[str, torch.Tensor]:
    sources: dict[str, torch.Tensor] = {
        "named": _named_source(),
        "sparse_coo": torch.ones(2, 3).to_sparse(),
        "sparse_csr": torch.ones(2, 3).to_sparse_csr(),
        "subclass": torch.ones(2, 3).as_subclass(_CountingSubclass),
    }
    try:
        sources["quantized"] = torch.quantize_per_tensor(torch.ones(2, 3), 0.1, 0, torch.qint8)
    except (RuntimeError, NotImplementedError):
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sources["nested"] = torch.nested.nested_tensor([torch.ones(3), torch.ones(3)])
    except (RuntimeError, NotImplementedError):
        pass
    try:
        sources["mkldnn"] = torch.ones(2, 3).to_mkldnn()
    except (RuntimeError, NotImplementedError):
        pass
    return sources


@pytest.mark.smoke
def test_r63_named_user_state_refuses_before_staging(tmp_path: Path) -> None:
    """Named source into an unnamed capture refuses at bind (Sol's 321-vs-3.0 repro class).

    Under oracle 1 the names SURVIVE default copy and steer the fresh oracle onto the named
    branch while replay stays unnamed -- the r62 bind-side false VERIFIED. The refusal is
    atomic: nothing is staged.
    """

    loaded = _plain_loaded(tmp_path)
    with pytest.raises(StateBindingError) as excinfo:
        loaded.load_state_dict({"w": _named_source()})
    message = str(excinfo.value)
    assert "state_metadata_mismatch" in message
    assert loaded.__dict__.get("_runnable_staged_user_state") is None


def test_r63_exotic_supplied_state_refuses_typed(tmp_path: Path) -> None:
    """Sparse COO/CSR, quantized, nested, mkldnn, and subclass sources refuse typed.

    Every one either raises under oracle-1's default ``load_state_dict`` copy or (subclass)
    could observe/steer through ``__torch_function__``; all refuse with
    ``state_metadata_mismatch`` before staging or execution.
    """

    for kind, source in _exotic_sources().items():
        loaded = _plain_loaded(tmp_path / kind)
        with pytest.raises(StateBindingError) as excinfo:
            loaded.load_state_dict({"w": source})
        assert "state_metadata_mismatch" in str(excinfo.value), kind


def test_r63_hostile_subclass_observes_zero_reads(tmp_path: Path) -> None:
    """Exact-class admission reads ``type()`` first: an unadmitted subclass observes nothing."""

    loaded = _plain_loaded(tmp_path)
    hostile = torch.ones(2, 3).as_subclass(_CountingSubclass)
    _CountingSubclass.reads = 0
    with pytest.raises(StateBindingError):
        loaded.load_state_dict({"w": hostile})
    assert _CountingSubclass.reads == 0


@pytest.mark.smoke
def test_r63_physical_form_user_sources_still_bind_and_verify(tmp_path: Path) -> None:
    """Non-contiguous and offset USER sources bind + VERIFIED (excluded dims stay excluded).

    Default ``load_state_dict`` copy normalizes physical form, so refusing these would
    over-refuse what PyTorch itself accepts -- the locked no-over-refusal constraint.
    """

    x = torch.randn(3)
    for name, source in (
        ("noncontig", torch.ones(3, 2).t()),
        ("offset", torch.ones(12)[4:10].reshape(2, 3)),
    ):
        loaded = _plain_loaded(tmp_path / name)
        loaded.load_state_dict({"w": source})
        result = loaded.run(inputs=x.clone())
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, name


def test_r63_channels_last_user_source_binds_into_conv(tmp_path: Path) -> None:
    """A channels-last USER source into a contiguous captured conv slot binds + VERIFIED."""

    x = torch.randn(1, 3, 8, 8)
    loaded = _save_load(nn.Conv2d(3, 4, 3, padding=1), x, tmp_path / "conv.tlspec")
    sd = {
        "weight": torch.randn(4, 3, 3, 3).to(memory_format=torch.channels_last),
        "bias": torch.randn(4),
    }
    loaded.load_state_dict(sd)
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_r63_dense_controls_bind_and_verify(tmp_path: Path) -> None:
    """Ordinary dense param + buffer + 0-dim int counter bind exact values + VERIFIED."""

    class DenseControls(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(2, 3))
            self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0]))
            self.register_buffer("counter", torch.tensor(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.scale + self.w.sum() + self.counter

    x = torch.randn(3)
    loaded = _save_load(DenseControls(), x, tmp_path / "dense.tlspec")
    loaded.load_state_dict(
        {
            "w": torch.full((2, 3), 4.0),
            "scale": torch.tensor([3.0, 3.0, 3.0]),
            "counter": torch.tensor(7),
        }
    )
    result = loaded.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# ======================================================================================
# Escaped-but-canonical pin + named-control ceiling pin
# ======================================================================================


class CanonicalIsContiguousRead(nn.Module):
    """``is_contiguous`` read on a CANONICAL param -- witnessed, reproducible, saveable."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.w.sum() if self.w.is_contiguous() else x - self.w.sum()


@pytest.mark.smoke
def test_r63_canonical_read_saves_verified_and_digest_witnesses(tmp_path: Path) -> None:
    """A metadata read on CANONICAL state saves + VERIFIED; changed staged state ceilings.

    The read is reproducible (staged state is always canonical), so the save is admitted;
    the slot is digest-witnessed exactly like an ``.item()`` value read, so a changed user
    state honestly reports UNVERIFIABLE rather than a blind VERIFIED.
    """

    x = torch.randn(3)
    path = _save(_trace(CanonicalIsContiguousRead(), x), tmp_path / "canon.tlspec")
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = tl.load(path)
    changed.load_state_dict({"w": torch.full((2, 3), 5.0)})
    result = changed.run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


class NamedControlNoTensorOp(nn.Module):
    """Named param read ONLY as Python control (``.names`` compare, no named-tensor op)."""

    def __init__(self) -> None:
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.w = nn.Parameter(torch.ones(2, 3).refine_names("out", "in"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 321.0 if self.w.names == ("out", "in") else x + 3.0


def test_r63_named_control_capture_never_verified(tmp_path: Path) -> None:
    """The named-as-control capture stays honestly non-VERIFIED, never a false VERIFIED.

    ``.names`` reads carry no accessor witness today. Post-r63 an ``include_weights`` save
    of NAMED captured state refuses typed at save -- the weight-blob writer validates the
    capture state through the same strict bind funnel, and names are a LOAD-SURVIVING dim
    the value-only embedding cannot represent (safetensors strips them), so embedding would
    silently declare a different state than captured. A weight-free save (or a legacy
    artifact) ceilings at the existing incomplete-escape UNVERIFIABLE. If future escape
    attribution learns to attribute ``.names`` reads, this pin forces routing into a typed
    refusal -- never into VERIFIED.
    """

    x = torch.randn(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = _trace(NamedControlNoTensorOp(), x)
        try:
            path = _save(trace, tmp_path / "named.tlspec")
        except (RunnablePreflightError, StateBindingError) as exc:
            assert "state_metadata_mismatch" in str(exc)
            return  # refusing the save outright is equally honest
        result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# ======================================================================================
# Structural belts: pre-clone stamping, runtime tripwire, source tripwire
# ======================================================================================


def test_r63_signatures_stamped_pre_clone() -> None:
    """Capture signatures preserve offset/conj facts the snapshot CLONE normalizes away."""

    class OffsetConjState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", _offset_buffer())
            self.register_buffer("c", _conj_buffer())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.b + self.c.real.sum()

    trace = _trace(OffsetConjState(), torch.randn(4))
    signatures = trace.__dict__.get("_runnable_capture_state_signatures")
    assert isinstance(signatures, dict)
    assert signatures["b"]["storage_offset_is_zero"] is False
    assert signatures["c"]["is_conj"] is True
    capture_state = trace.__dict__.get("_runnable_capture_state")
    assert capture_state["b"].storage_offset() == 0  # the clone normalized it
    assert not capture_state["c"].is_conj()


@pytest.mark.smoke
def test_r63_runtime_tripwire_catches_mutated_staged_state(tmp_path: Path) -> None:
    """Force-mutated staged state metadata fails ``state_metadata:<slot_id>`` typed."""

    loaded = _plain_loaded(tmp_path)
    loaded.load_state_dict({"w": torch.ones(2, 3)})
    staged = dict(loaded.__dict__["_runnable_staged_user_state"])
    slot_id = next(iter(staged))
    staged[slot_id] = torch.ones(3, 2).t()  # same shape/dtype, non-canonical stride
    loaded.__dict__["_runnable_staged_user_state"] = MappingProxyType(staged)
    with pytest.raises(PathDivergenceError) as excinfo:
        loaded.run(inputs=torch.randn(3))
    assert "canonical staged metadata signature" in str(excinfo.value)


def test_r63_staging_canonicalizes_physical_form(tmp_path: Path) -> None:
    """The staging clone always materializes the canonical destination form."""

    loaded = _plain_loaded(tmp_path)
    loaded.load_state_dict({"w": torch.ones(3, 2).t()})  # non-contiguous source
    staged = loaded.__dict__["_runnable_staged_user_state"]
    for value in staged.values():
        assert value.is_contiguous()
        assert value.stride() == (3, 1)
        assert value.storage_offset() == 0


def test_r63_source_tripwire_state_binding_uses_staging_helper() -> None:
    """No state-binding path calls bare ``_byte_guarded_clone`` around the staging helper."""

    import torchlens._runnable_state as module

    source = Path(module.__file__).read_text()

    def _body(name: str) -> str:
        match = re.search(rf"^def {name}\(.*?(?=^def |\Z)", source, flags=re.MULTILINE | re.DOTALL)
        assert match is not None, name
        return match.group(0)

    for binder in (
        "bind_embedded_trace_state",
        "bind_embedded_nonpersistent_buffers",
        "_validate_named_slot_mapping",
    ):
        body = _body(binder)
        assert "_byte_guarded_clone(" not in body, binder
        assert "_staged_state_clone(" in body, binder
    helper = _body("_staged_state_clone")
    assert "_byte_guarded_clone(" in helper  # the byte guard stays load-bearing inside
