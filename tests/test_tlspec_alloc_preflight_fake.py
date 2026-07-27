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

import json
import resource
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchlens as tl
import torchlens._runnable_execution as rex
from torchlens._runnable_execution import (
    _MAX_DECODE_NESTING_DEPTH,
    _decode_literal,
    _fake_tensor_mode_class,
    _has_numeric_literal,
    _input_storage_ids,
    _new_allocation_bytes,
    _preflight_call_allocation,
)
from torchlens.errors import RunCapabilityUnavailableError, RuntimeSignatureDriftError
from torchlens.runnable import LiteralAtom, LiteralAtomKind, LiteralSequence, LiteralSequenceKind

pytestmark = pytest.mark.smoke

_CAPTURE = dict(intervention_ready=True)


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
    preflight must charge ZERO new bytes and never over-refuse it (r51 anti-pattern).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        wide = x.expand(10**8, 4)  # 4e8-element view, no allocation
        return wide[0].sum() + x.sum()


def _build(tmp_path: Path, name: str, model: nn.Module, x: torch.Tensor) -> Path:
    trace = tl.trace(model.eval(), x, **_CAPTURE)
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


def test_genuinely_huge_view_model_runs_verified(tmp_path: Path) -> None:
    """A model whose real forward produces a HUGE view (4e8-element ``expand``) runs
    VERIFIED -- the alloc preflight charges 0 new bytes for it (r51 over-catch avoided)."""

    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_view.tlspec", _HugeView(), x)
    result = tl.load(str(bundle)).run(inputs=x.clone())
    assert result.report.path_faithfulness.value == "verified"


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
