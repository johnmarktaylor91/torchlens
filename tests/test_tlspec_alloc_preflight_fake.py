"""r57 C3 immunizer -- op-agnostic FakeTensorMode allocation preflight, allowlist DELETED.

r55 closed the allocation-bomb class with a ``FakeTensorMode(allow_non_fake_inputs=True)``
per-call projection, but GATED it behind a hand-maintained ``_SIZE_DRIVING_QUALNAME_TAILS``
op allowlist. r56 found the allowlist re-opened the class at every op it forgot
(``pad``/``constant_pad_nd``/``fold``/``*_window``/``tril_indices``/``triu_indices``/``one_hot``),
plus a FLOAT ``interpolate(scale_factor=...)`` slipping the integer-only sub-gate. r57 DELETES
the allowlist: the projection now runs for EVERY taken-path call carrying a non-bool numeric
literal (``_has_numeric_literal`` -- int OR finite float), and pure views / input-returning /
in-place ops are excluded STRUCTURALLY by fake output/input storage aliasing (not a view-name
list). A projected over-budget NEW allocation refuses typed at ``op_allocation_preflight``
BEFORE the real allocator runs.

This immunizer is ENUMERATION-FREE: it asserts the projection RAN and BOUNDED the call (a typed
refusal, with no allowlist behind it), never "op in a list". A broad op sample DELIBERATELY not
the old allowlist -- including every r56-missed op -- is refused; a huge PURE VIEW / input-return
/ in-place op is NOT over-refused (storage-alias -> 0 new bytes); a data-dependent op fails open;
benign models round-trip verified. Also covers the defensive run-side ``_decode_literal``
nesting-depth guard (C4).
"""

from __future__ import annotations

import copy
import json
import os
import resource
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens._runnable_execution as rex
from tests.test_tlspec_runnable_alloc_bomb import _tamper
from torchlens._runnable_execution import (
    _MAX_DECODE_NESTING_DEPTH,
    _decode_literal,
    _fake_tensor_mode_class,
    _has_numeric_literal,
    _input_storage_ids,
    _new_allocation_bytes,
    _preflight_call_allocation,
)
from torchlens.errors import (
    PayloadUnavailableError,
    RunCapabilityUnavailableError,
    RuntimeSignatureDriftError,
)
from torchlens.runnable import LiteralAtom, LiteralAtomKind, LiteralSequence, LiteralSequenceKind

_CAPTURE = {"capture": tl.options.CaptureOptions(intervention_ready=True)}


class _Factory(nn.Module):
    """A factory op (``torch.zeros(n)``) whose ``numel`` is a literal -- no tensor operand."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = torch.zeros(5)
        return x + pad.sum()


class _Nonzero(nn.Module):
    """A data-dependent op (``nonzero``) with no fake impl -- must FAIL OPEN and run."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(x > 0.0)
        return idx.float().sum() + x.sum()


class _HugeView(nn.Module):
    """A genuinely huge PURE VIEW (``expand`` of a size-1 dim) that allocates nothing.

    Its logical numel is enormous, but storage aliases the input, so the alloc
    preflights must charge it only the bytes the run really allocates and never
    over-refuse it (r51 anti-pattern).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**8, 4)  # 4e8-element view, no allocation
        return wide[0].sum() + x.sum()


def _build(
    tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor, **trace_kwargs: Any
) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE, **trace_kwargs)
    bundle = tmp_path / name
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    return bundle


def _tamper_literal(bundle: Path, kind: str, target: Any, replacement: Any) -> int:
    """Replace the first literal atom of ``(kind, value == target)`` inside any call's
    ``literal_arguments`` tree (recursing through nested list/tuple/dict literals).

    Returns the number of atoms mutated (must be >= 1 for the tamper to be meaningful).
    This edits ONLY the plaintext literal -- the recorded output slot shape is left
    honest and small, which is exactly the literal-only tamper the recorded-output
    bound cannot catch and the projection must.
    """

    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    count = 0

    def _walk(node: Any) -> bool:
        nonlocal count
        if isinstance(node, dict):
            if node.get("kind") == kind and node.get("value") == target:
                node["value"] = replacement
                count += 1
                return True
            return any(_walk(value) for value in node.values())
        if isinstance(node, list):
            return any(_walk(item) for item in node)
        return False

    for call in manifest["run"]["calls"]:
        for literal in call.get("literal_arguments", []):
            if _walk(literal):
                break
    path.write_text(json.dumps(manifest))
    return count


@contextmanager
def _rlimit_cap(extra: int = 1 << 30) -> Iterator[None]:
    """Cap address space so an un-refused bomb raises MemoryError instead of OOM-killing.

    A refusal at ``op_allocation_preflight`` fires BEFORE any allocation, so under this
    cap a refused call raises the typed ``RunCapabilityUnavailableError`` while an
    un-refused bomb would surface as a wrapped allocator failure -- the assertion that
    the former (not the latter) is raised is the proof that NO allocation occurred.
    """

    if not os.path.exists("/proc/self/status"):
        pytest.skip("address-space cap reads /proc/self/status (Linux only)")
    with open("/proc/self/status", encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _stub_call() -> Any:
    """A minimal stand-in carrying only the fields the refusal reads."""

    return types.SimpleNamespace(call_id="call:test", op_labels=("op:1",))


def _one_hot_fake_projection_supported() -> bool:
    """Return whether live FakeTensor projects explicit-width ``one_hot`` outputs."""

    mode_cls = _fake_tensor_mode_class()
    if mode_cls is None:
        return False
    try:
        with mode_cls(allow_non_fake_inputs=True) as mode:
            fake = rex._tree_to_fake(mode, torch.tensor([0, 1, 2, 3]))
            projected = F.one_hot(fake, num_classes=4)
    except Exception:  # noqa: BLE001 -- capability probe; any fake failure means absent.
        return False
    return tuple(projected.shape) == (4, 4)


_ONE_HOT_FAKE_PROJECTION_SUPPORTED = _one_hot_fake_projection_supported()


# --------------------------------------------------------------------------- #
# (a) the projection engine is available and closes the class                  #
# --------------------------------------------------------------------------- #


def test_fake_tensor_mode_available_and_projects_without_allocation() -> None:
    """The projection engine imports and projects a factory bomb without allocating."""

    mode_cls = _fake_tensor_mode_class()
    assert mode_cls is not None, "FakeTensorMode must be available on the supported torch range"
    with _rlimit_cap(), mode_cls(allow_non_fake_inputs=True):
        projected = torch.zeros(10**12)  # would be 4 TB if real -> refused by rlimit
    assert int(projected.numel()) == 10**12  # projected, never allocated


# --------------------------------------------------------------------------- #
# (b) ENUMERATION-FREE end-to-end refusal sweep: a BROAD op sample, DELIBERATELY  #
#     not the old allowlist, INCLUDING every r56-missed op. With NO allowlist, a  #
#     refusal PROVES the projection ran + bounded for that op -- never "op in a   #
#     list".                                                                       #
# --------------------------------------------------------------------------- #


class _Pad(nn.Module):  # r56 sec_1 / corr_2: F.pad was missing from the allowlist
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, [0, 13]).sum() + x.sum()


class _ConstantPadNd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.constant_pad_nd(x, (0, 17), 0.0).sum() + x.sum()


class _HannWindow(nn.Module):  # r56 free_1: window family was missing
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.hann_window(19).sum() + x.sum()


class _KaiserWindow(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.kaiser_window(23).sum() + x.sum()


class _TrilIndices(nn.Module):  # tall matrix -> nnz driven by ROW
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tril_indices(29, 3).float().sum() + x.sum()


class _TriuIndices(nn.Module):  # wide matrix -> nnz driven by COL
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.triu_indices(3, 31).float().sum() + x.sum()


class _InterpolateFloat(nn.Module):  # r56 free_1 vector 2: FLOAT scale_factor
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 1, 8)
        return F.interpolate(x, scale_factor=3.5, mode="nearest").sum() + x.sum()


class _OneHot(nn.Module):  # r56 corr_2: one_hot(num_classes=...) was missing
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = torch.tensor([0, 1, 2, 3])
        return F.one_hot(idx, num_classes=37).float().sum() + x.sum()


class _Zeros(nn.Module):  # already-covered factory (control)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(41).sum() + x.sum()


class _Arange(nn.Module):  # already-covered factory (control)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arange(43).float().sum() + x.sum()


class _Repeat(nn.Module):  # already-covered materializer (control)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(47).sum() + x.sum()


# (model_factory, input, literal kind, sentinel value, tampered replacement)
_REFUSAL_SWEEP = [
    ("pad", _Pad, torch.randn(4), "int", 13, 10**12),
    ("constant_pad_nd", _ConstantPadNd, torch.randn(4), "int", 17, 10**12),
    ("hann_window", _HannWindow, torch.randn(4), "int", 19, 10**12),
    ("kaiser_window", _KaiserWindow, torch.randn(4), "int", 23, 10**12),
    ("tril_indices", _TrilIndices, torch.randn(4), "int", 29, 10**11),
    ("triu_indices", _TriuIndices, torch.randn(4), "int", 31, 10**11),
    ("interpolate_float", _InterpolateFloat, torch.randn(1, 1, 8), "float", 3.5, 1e12),
    ("one_hot", _OneHot, torch.randn(4), "int", 37, 10**11),
    ("zeros", _Zeros, torch.randn(4), "int", 41, 10**12),
    ("arange", _Arange, torch.randn(4), "int", 43, 10**12),
    ("repeat", _Repeat, torch.randn(4), "int", 47, 10**11),
]


@pytest.mark.parametrize(
    "name, factory, x, kind, sentinel, replacement",
    _REFUSAL_SWEEP,
    ids=[row[0] for row in _REFUSAL_SWEEP],
)
def test_numeric_literal_bomb_refused_op_agnostically(
    tmp_path: Path,
    name: str,
    factory: type[nn.Module],
    x: torch.Tensor,
    kind: str,
    sentinel: Any,
    replacement: Any,
) -> None:
    """Each broad-sample op, literal-tampered huge, refuses at ``op_allocation_preflight``
    when the live FakeTensor surface supports its projection.

    Torch runtimes whose FakeTensor ``one_hot`` implementation still raises a
    data-dependent-output exception fail open by design, then re-type the real
    allocator refusal at ``op_allocation_execution``. The exception is feature-detected
    and scoped to that single operator; every projectable case still proves the
    enumeration-free preflight ran before allocation.
    """

    bundle = _build(tmp_path, f"{name}.tlspec", factory(), x)
    mutated = _tamper_literal(bundle, kind, sentinel, replacement)
    assert mutated >= 1, f"sentinel {sentinel!r} not found in {name} literal_arguments"
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            loaded.run(inputs=x.clone())
    expected_stage = (
        "op_allocation_preflight"
        if name != "one_hot" or _ONE_HOT_FAKE_PROJECTION_SUPPORTED
        else "op_allocation_execution"
    )
    assert caught.value.fields.get("detection_stage") == expected_stage
    if expected_stage == "op_allocation_preflight":
        assert int(caught.value.fields["required_bytes"]) > int(
            caught.value.fields["available_bytes"]
        )


class _Fold(nn.Module):
    """``F.fold`` -- its ``output_size`` is coupled to the input block count, so a
    literal-only tamper makes the projection RAISE (fail-open) AND the real op raise
    its own consistency check BEFORE allocating: caught with no allocation via
    signature drift, the documented fail-open residual (never an alloc bomb)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4, 9)
        return F.fold(x, output_size=(4, 4), kernel_size=(2, 2)).sum() + x.sum()


def test_fold_literal_tamper_caught_without_allocation(tmp_path: Path) -> None:
    """A ``fold`` output_size tamper is caught WITHOUT allocation. The projection fails
    open (output_size/block-count inconsistency), but real ``fold`` validates the same
    inconsistency before allocating -> typed ``RuntimeSignatureDriftError``, no bomb.
    This is the honest fail-open residual, not an ``op_allocation_preflight`` refusal."""

    x = torch.randn(1, 4, 9)
    bundle = _build(tmp_path, "fold.tlspec", _Fold(), x)
    mutated = _tamper_literal(bundle, "int", 4, 10**12)
    assert mutated >= 1
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
        # Caught (no unbounded allocation) -- via signature drift, not a MemoryError bomb.
        with pytest.raises(RuntimeSignatureDriftError):
            loaded.run(inputs=x.clone())


# --------------------------------------------------------------------------- #
# (c) ENUMERATION-FREE mechanism pins: the projection runs for ANY callable      #
#     regardless of its qualname/registry entry, and pure views / input-returns  #
#     / in-place ops are excluded by STORAGE ALIASING (no view-name list).       #
# --------------------------------------------------------------------------- #


def test_preflight_projects_regardless_of_callable_identity() -> None:
    """A factory bomb refuses even with ``entry=None`` and an ARBITRARY callable whose
    qualname is nothing special -- proving no op-name/registry gate remains."""

    call = _stub_call()
    with _rlimit_cap():
        with pytest.raises(RunCapabilityUnavailableError) as caught:
            _preflight_call_allocation(None, torch.zeros, [10**12], {}, call)
        assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"

        def _arbitrary_alloc(n: int) -> torch.Tensor:
            return torch.zeros(n)

        with pytest.raises(RunCapabilityUnavailableError):
            _preflight_call_allocation(None, _arbitrary_alloc, [10**12], {}, call)


def test_preflight_excludes_views_and_inplace_by_storage_alias() -> None:
    """Huge pure views / input-returning / in-place outputs alias a fake INPUT storage
    -> zero NEW bytes -> NOT refused (structural, no view allowlist). The ``_base``-gap
    cases (input-return, in-place) are included precisely because ``_base is None`` there."""

    call = _stub_call()
    x = torch.randn(4)
    over_triggers = [
        ("expand", torch.Tensor.expand, [x, 10**12, 4], {}),
        ("broadcast_to", torch.broadcast_to, [x, (10**12, 4)], {}),
        ("as_strided", torch.as_strided, [x, (10**6, 10**6), (0, 0)], {}),
        ("getitem", torch.Tensor.__getitem__, [x, 0], {}),  # input-returning-ish
        ("add_inplace", torch.Tensor.add_, [x.clone(), 1], {}),  # in-place, _base is None
    ]
    for _name, func, args, kwargs in over_triggers:
        # Must NOT raise -- a view/in-place op allocates nothing.
        _preflight_call_allocation(None, func, args, kwargs, call)


@pytest.mark.smoke
def test_preflight_fails_open_on_data_dependent_op() -> None:
    """A data-dependent op with no fake impl (``nonzero``) fails open -- not refused."""

    _preflight_call_allocation(None, torch.nonzero, [torch.randn(4)], {}, _stub_call())


def test_preflight_skips_calls_without_numeric_literal() -> None:
    """No size source AT ALL (no tensor operand AND no numeric literal, r61) -> the
    projection is skipped; a tensor-operand-only call now projects (see the r61
    immunizer file for the amplifier families the old has-literal-only skip gapped)."""

    _preflight_call_allocation(None, torch.zeros, [], {}, _stub_call())


def test_new_allocation_bytes_charges_only_non_aliasing_outputs() -> None:
    """Storage-alias accounting: a projected VIEW of the input contributes 0 bytes; a
    fresh projected tensor is charged; a mixed (view + fresh) output charges only fresh."""

    mode_cls = _fake_tensor_mode_class()
    assert mode_cls is not None
    with mode_cls(allow_non_fake_inputs=True) as mode:
        fake_in = mode.from_tensor(torch.randn(8))
        input_ids = _input_storage_ids([fake_in])
        view = fake_in.expand(10**6, 8)  # aliases input storage
        fresh = torch.zeros(1000)  # fresh fake allocation
        assert _new_allocation_bytes(view, input_ids) == {}  # pure view -> 0 new bytes
        fresh_bytes = _new_allocation_bytes(fresh, input_ids)
        assert sum(fresh_bytes.values()) == 1000 * fresh.element_size()
        mixed_bytes = _new_allocation_bytes([view, fresh], input_ids)
        assert sum(mixed_bytes.values()) == 1000 * fresh.element_size()


def test_has_numeric_literal_covers_int_and_finite_float_only() -> None:
    """``_has_numeric_literal`` fires for any non-bool int or FINITE float (closing the
    float ``interpolate(scale_factor=...)`` gap), and excludes bool / inf / nan."""

    assert _has_numeric_literal([5])
    assert _has_numeric_literal([3.5])
    assert _has_numeric_literal({"a": {"b": (2.0,)}})  # nested finite float
    assert not _has_numeric_literal([True, False])  # bool excluded
    assert not _has_numeric_literal([float("inf")])  # non-finite excluded
    assert not _has_numeric_literal([float("nan")])
    assert not _has_numeric_literal(["a string", None])


# --------------------------------------------------------------------------- #
# (d) absence pins -- the enumerated gate is GONE                                #
# --------------------------------------------------------------------------- #


def test_size_driving_allowlist_and_gate_are_deleted() -> None:
    """The projection is NOT gated by a callable qualname/tail allowlist: the r55
    enumeration symbols are absent, so no op-name list can silently re-open the class."""

    assert not hasattr(rex, "_SIZE_DRIVING_QUALNAME_TAILS")
    assert not hasattr(rex, "_call_is_size_driving")
    assert not hasattr(rex, "_has_integer_literal")  # renamed to _has_numeric_literal


# --------------------------------------------------------------------------- #
# (e) over-trigger guards -- legit ops / models must NOT be over-refused         #
# --------------------------------------------------------------------------- #


def test_untampered_factory_model_runs_verified(tmp_path: Path) -> None:
    """A legitimate factory-using model runs VERIFIED (no over-trigger)."""

    bundle = _build(tmp_path, "clean_factory.tlspec", _Factory(), torch.randn(4))
    result = tl.load(str(bundle)).run(inputs=torch.randn(4))
    assert result.report.path_faithfulness.value == "verified"


class _HugeMaterialized(nn.Module):
    """The counter-case: the same kind of huge expand, then a real materializing op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**7, 4) * 2.0  # 4e7 elements, 160 MB really allocated
        return wide[0].sum() + x.sum()


class _HugeSavedView(nn.Module):
    """The counter-case for retention: a huge view the capture saves is copied out."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**7, 4)  # 4e7 elements; its saved ``Op.out`` copy is 160 MB
        return wide[0].sum() + x.sum()


class _MedianOfView(nn.Module):
    """A huge unsaved view read by ``median``, which copies its operand internally."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        return x.expand(10**7, 4).median() + x.sum()


class _MmOfView(nn.Module):
    """A huge unsaved view read by ``mm``: BLAS repacks a stride-0 operand."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        return torch.mm(x.expand(10**7, 4), self.weight).sum() + x.sum()


class _NucNormOfView(nn.Module):
    """A huge unsaved view read by ``norm(p="nuc")``, which runs an SVD on a copy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        return x.expand(10**7, 4).norm(p="nuc") + x.sum()


class _SumAndMedianOfView(nn.Module):
    """A huge unsaved view read by ``sum`` and ``median`` (the alias tamper's base)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**7, 4)
        return wide.sum() + wide.median() + x.sum()


class _AddOutOfView(nn.Module):
    """A huge unsaved view read by ``torch.add(..., out=)``, which resizes ``out`` in full."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        out = torch.empty(0)
        torch.add(x.expand(10**7, 4), 1.0, out=out)
        return out[0].sum() + x.sum()


class _TwoHugeViews(nn.Module):
    """Two huge unsaved views, each admissible on its own, read by ``sum``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        return x.expand(10**7, 4).sum() + x.expand(10**7, 4).sum() + x.sum()


#: The host allocation budget the huge-view runs see: 64 MiB, fixed so the rows do not
#: depend on the runner's free memory. It is far below the views' logical sizes (1.6 GB
#: and 160 MB) and far above what an unsaved honest view really allocates (nothing).
_HUGE_VIEW_HOST_BUDGET_BYTES = 64 << 20


def _fix_host_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every module that reads the host allocation budget see the fixed figure."""

    import torchlens._runnable_call_arguments as call_arguments
    import torchlens._runnable_state as runnable_state

    def fixed_budget(device: torch.device) -> int:
        return _HUGE_VIEW_HOST_BUDGET_BYTES

    for module in (runnable_state, rex, call_arguments):
        monkeypatch.setattr(module, "_allocation_budget_bytes", fixed_budget)


def test_genuinely_huge_view_model_runs_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose real forward produces a HUGE view (4e8-element ``expand``) runs
    VERIFIED under a budget far below the view's 1.6 GB logical size: the op allocates
    nothing, the capture did not save the view so the run keeps no copy of it, and the
    run-prep bounds charge it nothing."""

    _fix_host_budget(monkeypatch)
    x = torch.randn(1, 4)
    # Retain only the scalar sums: the runnable save is tensor-value-free, and the default
    # exhaustive capture would materialize a contiguous copy of the 4e8-element view
    # (~1.6 GB), which trips the host-dependent save budget on a busy CI runner.
    # The subject here is the RUN-time alloc preflight, not capture retention.
    bundle = _build(tmp_path, "huge_view.tlspec", _HugeView(), x, save=tl.func("sum"))
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
    ops = {op.label.split("_")[0]: op for op in result.trace.layer_list}
    try:
        expand_out = ops["expand"].out
    except PayloadUnavailableError:
        expand_out = None
    assert expand_out is None  # unsaved at capture, so the run keeps no copy
    assert isinstance(ops["sum"].out, torch.Tensor)  # saved ops keep their snapshot


@pytest.mark.parametrize(
    "factory, save",
    [
        (_HugeMaterialized, tl.func("sum")),
        (_HugeSavedView, tl.func("expand")),
        (_MedianOfView, tl.func("sum")),
        (_MmOfView, tl.func("sum")),
        (_NucNormOfView, tl.func("sum")),
    ],
    ids=[
        "materializing_op",
        "saved_view",
        "view_read_by_median",
        "view_read_by_mm",
        "view_read_by_nuc_norm",
    ],
)
def test_huge_materialized_slot_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory: type[nn.Module], save: Any
) -> None:
    """A slot that really materializes past the budget is still refused at run-prep,
    before any op executes, charged its full 160 MB: a materializing op (``* 2.0``), a
    huge view the capture saved, whose ``Op.out`` copy the run makes, or an unsaved huge
    view read by an op that copies it internally (``median``, ``mm``, ``norm(p="nuc")``),
    which FakeTensor cannot see."""

    _fix_host_budget(monkeypatch)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_mat.tlspec", factory(), x, save=save)
    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) == 10**7 * 4 * 4
    assert int(caught.value.fields["available_bytes"]) == _HUGE_VIEW_HOST_BUDGET_BYTES


@pytest.mark.parametrize("failure", ["no_size_source", "projection_raises"])
def test_admitted_view_unprojectable_at_run_time_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A call producing a view run-prep admitted past the budget has no recorded-output
    bound behind it, so when its run-time projection cannot run (judged trivial, or the
    projection raises a non-allocation error) it is refused before it executes, where
    every other call fails open."""

    _fix_host_budget(monkeypatch)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_view.tlspec", _HugeView(), x, save=tl.func("sum"))
    loaded = tl.load(str(bundle))

    def _raise_projection(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("no fake implementation")

    if failure == "no_size_source":
        monkeypatch.setattr(rex, "_projection_required_by_arguments", lambda *_a: False)
    else:
        monkeypatch.setattr(rex, "_tree_to_fake", _raise_projection)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    labels = caught.value.fields.get("affected_op_labels") or ()
    assert any(str(label).startswith("expand") for label in labels)


def test_admitted_views_share_one_logical_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view work ceiling bounds the admitted views of a run together: two views each
    under it are refused once their sum passes it."""

    import torchlens._runnable_retention as retention

    _fix_host_budget(monkeypatch)
    monkeypatch.setattr(retention, "_VIEW_LOGICAL_ELEMENT_CEILING", 6 * 10**7)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "two_views.tlspec", _TwoHugeViews(), x, save=tl.func("sum"))
    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["logical_elements"]) == 2 * 4 * 10**7


def test_view_proof_projection_runs_without_real_kernel_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run-prep view proof never lets FakeTensor fall back to a real kernel on
    zero-filled operands of recorded, not yet bounded, shapes."""

    _fix_host_budget(monkeypatch)
    real_mode_class = rex._count_bounded_fake_tensor_mode_class()
    assert real_mode_class is not None
    seen: list[Any] = []

    class _Recording(real_mode_class):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.append(kwargs.get("allow_fallback_kernels", True))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rex, "_count_bounded_fake_tensor_mode_class", lambda: _Recording)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_view.tlspec", _HugeView(), x, save=tl.func("sum"))
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
    assert seen and seen[0] is False  # the first projection is the run-prep view proof


def _calls_by_op(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index a manifest's calls by the op name of their first label."""

    return {call["op_labels"][0].split("_")[0]: call for call in run["calls"]}


def _read_median_through_alias(manifest: dict[str, Any]) -> None:
    """Point ``median`` at a crafted 1-element version alias of the ``expand`` view."""

    run = manifest["run"]
    calls = _calls_by_op(run)
    view_id = calls["expand"]["output_slot_ids"][0]
    median = calls["median"]
    slots = {slot["slot_id"]: slot for slot in run["tensor_slots"]}
    alias = copy.deepcopy(slots[view_id])
    alias.update(
        slot_id=f"{view_id}:alias",
        shape=[1],
        rank=1,
        version_of=view_id,
        producer_slot_id=view_id,
        use_sites=[],
    )
    for argument in median["tensor_arguments"]:
        if argument["slot_id"] == view_id:
            argument["slot_id"] = alias["slot_id"]
            alias["use_sites"].append(
                {"call_id": median["call_id"], "argument_path": argument["argument_path"]}
            )
    slots[view_id]["use_sites"] = [
        site for site in slots[view_id]["use_sites"] if site["call_id"] != median["call_id"]
    ]
    run["tensor_slots"].append(alias)


def _record_out_add_small(manifest: dict[str, Any]) -> None:
    """Record every full-size slot but the ``expand`` view at a 4-element shape."""

    run = manifest["run"]
    view_id = _calls_by_op(run)["expand"]["output_slot_ids"][0]
    for slot in run["tensor_slots"]:
        if slot["slot_id"] != view_id and slot["shape"] == [10**7, 4]:
            slot["shape"] = [1, 4]


@pytest.mark.parametrize(
    "factory, mutate",
    [
        (_SumAndMedianOfView, _read_median_through_alias),
        (_AddOutOfView, _record_out_add_small),
    ],
    ids=["copying_reader_through_version_alias", "out_argument_resized_to_view"],
)
def test_crafted_hidden_copy_of_view_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory: type[nn.Module], mutate: Any
) -> None:
    """A crafted file cannot hide a full-size copy of an admitted view from run-prep: a
    copying reader reached through a version alias of the view, or an ``out=`` reader
    whose small recorded output the real call resizes to the view's size, keeps the view
    charged in full and refused before any op executes."""

    _fix_host_budget(monkeypatch)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "crafted.tlspec", factory(), x, save=tl.func("sum"))
    _tamper(bundle, mutate)
    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) == 10**7 * 4 * 4


def _admission_slot(slot_id: str, device_type: str = "cpu", alias_of: str | None = None) -> Any:
    return types.SimpleNamespace(
        slot_id=slot_id, device_type=device_type, version_of=alias_of, producer_slot_id=alias_of
    )


def _admission_call(
    call_id: str,
    reads: tuple[str, ...] = (),
    writes: tuple[str, ...] = (),
    out: str | None = None,
    inplace: bool = False,
) -> Any:
    arguments = [
        types.SimpleNamespace(slot_id=slot_id, argument_path=("args", index))
        for index, slot_id in enumerate(reads)
    ]
    if out is not None:
        arguments.append(types.SimpleNamespace(slot_id=out, argument_path=("kwargs", "out")))
    return types.SimpleNamespace(
        call_id=call_id,
        tensor_arguments=tuple(arguments),
        output_slot_ids=writes,
        is_inplace=inplace,
    )


def _resolved_op(namespace: str, name: str) -> Any:
    from torchlens._runnable_retention import _resolved_ops

    (func,) = _resolved_ops(((namespace, name),))
    return func


def _admits(
    monkeypatch: pytest.MonkeyPatch,
    view: Any,
    extra_slots: tuple[Any, ...],
    reader: Any,
    reader_op: tuple[str, str],
) -> bool:
    """Decide ``view`` (produced by ``expand``) with the projection pinned to "is a view"."""

    from torchlens._runnable_retention import _unretained_view_lookup

    monkeypatch.setattr(rex, "_pure_view_projector", lambda _d, _c: lambda _slot: True)
    producer = _admission_call("c:expand", writes=(view.slot_id,))
    callables = {"c:expand": _resolved_op("Tensor", "expand"), "c:read": _resolved_op(*reader_op)}
    trace = types.SimpleNamespace(_runnable=types.SimpleNamespace(callables_by_call_id=callables))
    descriptor = types.SimpleNamespace(tensor_slots=(view, *extra_slots), calls=(producer, reader))
    lookup = _unretained_view_lookup(trace, descriptor, frozenset())
    assert lookup is not None
    return lookup(view)


def test_view_admission_control_streaming_reader_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the routes below: a CPU view read only by ``sum`` is admitted."""

    view = _admission_slot("s:view")
    reader = _admission_call("c:read", reads=("s:view",), writes=("s:sum",))
    assert _admits(monkeypatch, view, (), reader, ("torch", "sum"))


@pytest.mark.parametrize(
    "route", ["cuda_view", "reader_through_alias", "out_reader", "inplace_reader", "norm_reader"]
)
def test_view_admission_refuses_hidden_copy_routes(
    monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """Each route that can hide a full-size copy of a view keeps the view charged: a view
    off CPU (the reader lists were measured on CPU), a copying reader of a version alias,
    a listed reader writing ``out=`` or in place, and ``norm`` (an SVD for ``p="nuc"``)."""

    view = _admission_slot("s:view", device_type="cuda" if route == "cuda_view" else "cpu")
    extra: tuple[Any, ...] = ()
    reader_op = ("torch", "add")
    reader = _admission_call("c:read", reads=("s:view",), writes=("s:out",))
    if route == "reader_through_alias":
        extra = (_admission_slot("s:alias", alias_of="s:view"),)
        reader = _admission_call("c:read", reads=("s:alias",), writes=("s:out",))
        reader_op = ("torch", "median")
    elif route == "out_reader":
        reader = _admission_call("c:read", reads=("s:view",), writes=("s:out",), out="s:buf")
    elif route == "inplace_reader":
        reader = _admission_call("c:read", reads=("s:view",), writes=("s:out",), inplace=True)
    elif route == "norm_reader":
        reader_op = ("Tensor", "norm")
    assert not _admits(monkeypatch, view, extra, reader, reader_op)


def _alias_slot(slots: dict[str, dict[str, Any]], target_id: str, slot_id: str) -> dict[str, Any]:
    """A crafted unread version alias of ``target_id``, recorded at the target's shape."""

    alias = copy.deepcopy(slots[target_id])
    alias.update(slot_id=slot_id, version_of=target_id, producer_slot_id=target_id, use_sites=[])
    return alias


def _alias_of_alias(manifest: dict[str, Any]) -> None:
    """Hang a two-level version-alias chain off the ``expand`` view."""

    run = manifest["run"]
    view_id = _calls_by_op(run)["expand"]["output_slot_ids"][0]
    slots = {slot["slot_id"]: slot for slot in run["tensor_slots"]}
    first = _alias_slot(slots, view_id, f"{view_id}:alias")
    slots[first["slot_id"]] = first
    run["tensor_slots"] += [first, _alias_slot(slots, first["slot_id"], f"{view_id}:alias2")]


def test_version_alias_of_alias_refused_at_load(tmp_path: Path) -> None:
    """Bind writes a produced value only into its direct version aliases, so a file whose
    alias names another alias is refused at load, which keeps the run's alias maps one
    level deep (a crafted chain cannot make run-prep quadratic)."""

    from torchlens._io.runnable_load import ContextFieldInvalidError

    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "alias_chain.tlspec", _HugeView(), x, save=tl.func("sum"))
    _tamper(bundle, _alias_of_alias)
    with pytest.raises(ContextFieldInvalidError, match="one level deep"):
        tl.load(str(bundle))


def test_alias_reader_map_is_linear_in_slots() -> None:
    """The view-admission reader map holds one entry per read and per direct alias read,
    however long a version-alias chain the descriptor carries."""

    from torchlens._runnable_retention import _readers_through_version_aliases

    chain = 2000
    slots = [_admission_slot("s:a0")] + [
        _admission_slot(f"s:a{index}", alias_of=f"s:a{index - 1}") for index in range(1, chain)
    ]
    readers = {
        "s:a0": [_admission_call("c:head", reads=("s:a0",))],
        f"s:a{chain - 1}": [_admission_call("c:tail", reads=(f"s:a{chain - 1}",))],
    }
    descriptor = types.SimpleNamespace(tensor_slots=tuple(slots))
    mapped = _readers_through_version_aliases(descriptor, readers)
    assert sum(len(calls) for calls in mapped.values()) <= 2 * len(readers)


def test_admitted_view_realized_off_cpu_refused_at_bind() -> None:
    """A view admitted past the budget as a CPU view whose realized value is on another
    device (the recorded label lied) is refused at bind, before any reader runs; an
    unadmitted output there, or an admitted one on CPU, passes."""

    from torchlens._runnable_retention import _check_unretained_extent

    labels = {"call_id": "c:expand", "slot_id": "s:view"}
    _check_unretained_extent(torch.empty(4, device="meta"), **labels)
    _check_unretained_extent(torch.empty(4), admitted_view=True, **labels)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        _check_unretained_extent(torch.empty(4, device="meta"), admitted_view=True, **labels)
    assert caught.value.fields.get("detection_stage") == "op_allocation_execution"
    assert caught.value.fields.get("device") == "meta"


def test_bind_tells_the_extent_check_which_outputs_are_admitted_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bind passes run-prep's admission to the extent check, so the device check above
    applies to exactly the admitted views."""

    _fix_host_budget(monkeypatch)
    real_check = rex._check_unretained_extent
    admitted_by_slot: dict[str, bool] = {}

    def recording_check(value: torch.Tensor, **kwargs: Any) -> None:
        admitted_by_slot[str(kwargs["slot_id"])] = bool(kwargs.get("admitted_view", False))
        real_check(value, **kwargs)

    monkeypatch.setattr(rex, "_check_unretained_extent", recording_check)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_view.tlspec", _HugeView(), x, save=tl.func("sum"))
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"
    admitted = sorted(slot for slot, flag in admitted_by_slot.items() if flag)
    assert admitted and all("expand" in slot for slot in admitted)


def test_data_dependent_op_fails_open_and_runs(tmp_path: Path) -> None:
    """A legit ``nonzero`` (fake impl raises ``DynamicOutputShapeException``) FAILS OPEN."""

    x = torch.randn(8)
    bundle = _build(tmp_path, "nonzero.tlspec", _Nonzero(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    # fail-open: the projection raised, the run still executes and is faithful.
    assert result.report.path_faithfulness.value == "verified"


# --------------------------------------------------------------------------- #
# (f) defensive run-side decode-depth guard (C4)                               #
# --------------------------------------------------------------------------- #


def test_decode_literal_depth_guard_raises_not_recursionerror() -> None:
    """An over-depth nested literal raises a typed ``ValueError``, never ``RecursionError``."""

    node: Any = LiteralAtom(kind=LiteralAtomKind.INT, value=1)
    for _ in range(_MAX_DECODE_NESTING_DEPTH + 5):
        node = LiteralSequence(kind=LiteralSequenceKind.TUPLE, items=(node,))
    with pytest.raises(ValueError, match="decode depth"):
        _decode_literal(node)


def test_decode_literal_normal_nesting_ok() -> None:
    """Ordinary shallow literal nesting decodes cleanly (guard never over-fires)."""

    node: Any = LiteralSequence(
        kind=LiteralSequenceKind.TUPLE,
        items=(
            LiteralAtom(kind=LiteralAtomKind.INT, value=2),
            LiteralAtom(kind=LiteralAtomKind.INT, value=3),
        ),
    )
    assert _decode_literal(node) == (2, 3)
