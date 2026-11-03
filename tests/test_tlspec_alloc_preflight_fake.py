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
import torchlens._runnable.call_arguments
import torchlens._runnable.call_outputs
import torchlens._runnable.execution as rex
import torchlens._runnable.state_context
from tests.test_tlspec_runnable_alloc_bomb import _tamper
from torchlens._runnable import (
    LiteralAtom,
    LiteralAtomKind,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralTorchSymbol,
)
from torchlens._runnable.execution import _MAX_DECODE_NESTING_DEPTH
from torchlens._runnable.state_context import _fake_tensor_mode_class
from torchlens._runnable.verification import _decode_literal
from torchlens.errors import (
    RunCapabilityUnavailableError,
    RunnableTLSPECError,
)

_CAPTURE = {"capture": tl.options.CaptureOptions(replayable=True)}


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
    trace.save(str(bundle), level="runnable", include_weights=True)
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
            fake = torchlens._runnable.call_arguments._tree_to_fake(
                mode, torch.tensor([0, 1, 2, 3])
            )
            projected = F.one_hot(fake, num_classes=4)
    except Exception:  # noqa: BLE001 -- capability probe; any fake failure means absent.
        return False
    return tuple(projected.shape) == (4, 4)


_ONE_HOT_FAKE_PROJECTION_SUPPORTED = _one_hot_fake_projection_supported()


# --------------------------------------------------------------------------- #
# (a) the projection engine is available and closes the class                  #
# --------------------------------------------------------------------------- #


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
    ("interpolate_float", _InterpolateFloat, torch.randn(1, 1, 8), "float", 3.5, 1e12),
    ("one_hot", _OneHot, torch.randn(4), "int", 37, 10**11),
    ("repeat", _Repeat, torch.randn(4), "int", 47, 10**11),
]


# Kept: sole branch coverage of _io/runnable_load.py, _runnable/call_arguments.py, backends/torch/_completeness_escape_state.py; a sole mutation kill.
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


# --------------------------------------------------------------------------- #
# (c) ENUMERATION-FREE mechanism pins: the projection runs for ANY callable      #
#     regardless of its qualname/registry entry, and pure views / input-returns  #
#     / in-place ops are excluded by STORAGE ALIASING (no view-name list).       #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (d) absence pins -- the enumerated gate is GONE                                #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (e) over-trigger guards -- legit ops / models must NOT be over-refused         #
# --------------------------------------------------------------------------- #


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


class _SumToFloat64OfView(nn.Module):
    """A huge unsaved view read by ``sum(dtype=torch.float64)``, which casts a full copy."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (1, 4)
        return x.expand(10**7, 4).sum(dtype=torch.float64) + x.sum()


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

    import torchlens._runnable.call_arguments as call_arguments
    import torchlens._runnable.state as runnable_state

    def fixed_budget(device: torch.device) -> int:
        return _HUGE_VIEW_HOST_BUDGET_BYTES

    for module in (runnable_state, rex, call_arguments):
        monkeypatch.setattr(module, "_allocation_budget_bytes", fixed_budget)


# Kept: sole branch coverage of _runnable/retention.py; its other params stay because the list is not trimmable per id.
@pytest.mark.parametrize(
    "factory, save",
    [
        (_HugeMaterialized, tl.func("sum")),
        (_HugeSavedView, tl.func("expand")),
        (_MedianOfView, tl.func("sum")),
        (_MmOfView, tl.func("sum")),
        (_NucNormOfView, tl.func("sum")),
        (_SumToFloat64OfView, tl.func("sum")),
    ],
    ids=[
        "materializing_op",
        "saved_view",
        "view_read_by_median",
        "view_read_by_mm",
        "view_read_by_nuc_norm",
        "view_read_by_sum_to_float64",
    ],
)
def test_huge_materialized_slot_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, factory: type[nn.Module], save: Any
) -> None:
    """A slot that really materializes past the budget is still refused at run-prep,
    before any op executes, charged its full 160 MB: a materializing op (``* 2.0``), a
    huge view the capture saved, whose ``Op.out`` copy the run makes, or an unsaved huge
    view read by an op that copies it internally (``median``, ``mm``, ``norm(p="nuc")``,
    ``sum(dtype=torch.float64)``), which FakeTensor cannot see."""

    _fix_host_budget(monkeypatch)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "huge_mat.tlspec", factory(), x, save=save)
    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["required_bytes"]) == 10**7 * 4 * 4
    assert int(caught.value.fields["available_bytes"]) == _HUGE_VIEW_HOST_BUDGET_BYTES


# Kept: sole branch coverage of _runnable/call_arguments.py.
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
        monkeypatch.setattr(
            torchlens._runnable.call_arguments,
            "_projection_required_by_arguments",
            lambda *_a: False,
        )
    else:
        monkeypatch.setattr(torchlens._runnable.call_arguments, "_tree_to_fake", _raise_projection)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    labels = caught.value.fields.get("affected_op_labels") or ()
    assert any(str(label).startswith("expand") for label in labels)


# Kept: sole branch coverage of _runnable/retention.py, _runnable/state.py.
def test_admitted_views_share_one_logical_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The view work ceiling bounds the admitted views of a run together: two views each
    under it are refused once their sum passes it."""

    import torchlens._runnable.retention as retention

    _fix_host_budget(monkeypatch)
    monkeypatch.setattr(retention, "_VIEW_LOGICAL_ELEMENT_CEILING", 6 * 10**7)
    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "two_views.tlspec", _TwoHugeViews(), x, save=tl.func("sum"))
    loaded = tl.load(str(bundle))
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=x.clone())
    assert caught.value.fields.get("detection_stage") == "op_allocation_preflight"
    assert int(caught.value.fields["logical_elements"]) == 2 * 4 * 10**7


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


# Kept: sole branch coverage of _runnable/retention.py; a sole mutation kill; its other params stay because the list is not trimmable per id.
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
    dtype: str | None = None,
    literal_paths: tuple[tuple[str, int | str], ...] = (),
) -> Any:
    arguments = [
        types.SimpleNamespace(slot_id=slot_id, argument_path=("args", index))
        for index, slot_id in enumerate(reads)
    ]
    if out is not None:
        arguments.append(types.SimpleNamespace(slot_id=out, argument_path=("kwargs", "out")))
    literals = tuple(
        types.SimpleNamespace(argument_path=path, value=LiteralAtom(LiteralAtomKind.INT, 1))
        for path in literal_paths
    )
    if dtype is not None:
        literals += (
            types.SimpleNamespace(
                argument_path=("kwargs", "dtype"), value=LiteralTorchSymbol(dtype)
            ),
        )
    return types.SimpleNamespace(
        call_id=call_id,
        tensor_arguments=tuple(arguments),
        literal_arguments=literals,
        output_slot_ids=writes,
        is_inplace=inplace,
    )


def _resolved_op(namespace: str, name: str) -> Any:
    from torchlens._runnable.retention import _resolved_ops

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

    from torchlens._runnable.retention import _unretained_view_lookup

    monkeypatch.setattr(
        torchlens._runnable.call_arguments,
        "_pure_view_projector",
        lambda _d, _c: lambda _slot: True,
    )
    producer = _admission_call("c:expand", writes=(view.slot_id,))
    callables = {"c:expand": _resolved_op("Tensor", "expand"), "c:read": _resolved_op(*reader_op)}
    trace = types.SimpleNamespace(_runnable=types.SimpleNamespace(callables_by_call_id=callables))
    descriptor = types.SimpleNamespace(tensor_slots=(view, *extra_slots), calls=(producer, reader))
    lookup = _unretained_view_lookup(trace, descriptor, frozenset())
    assert lookup is not None
    return lookup(view)


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


# Kept: sole branch coverage of _io/runnable_load.py; a sole mutation kill.
def test_version_alias_of_alias_refused_at_load(tmp_path: Path) -> None:
    """Bind writes a produced value only into its direct version aliases, so a file whose
    alias names another alias is refused at load, which keeps the run's alias maps one
    level deep (a crafted chain cannot make run-prep quadratic). The parse refusal is
    the load's readiness diagnostic, and the run refuses before any op executes."""

    x = torch.randn(1, 4)
    bundle = _build(tmp_path, "alias_chain.tlspec", _HugeView(), x, save=tl.func("sum"))
    _tamper(bundle, _alias_of_alias)
    loaded = tl.load(str(bundle))
    readiness = loaded.readiness
    assert readiness is not None and readiness.status.value != "ready"
    diagnostics = " ".join(f"{d.code.value} {d.message}" for d in readiness.diagnostics)
    assert "context_field_invalid" in diagnostics and "one level deep" in diagnostics
    with pytest.raises(RunnableTLSPECError):
        loaded.run(inputs=x.clone())


# Kept: sole branch coverage of _runnable/retention.py, _runnable/state.py.
def test_admitted_view_realized_off_cpu_refused_at_bind() -> None:
    """A view admitted past the budget as a CPU view whose realized value is on another
    device (the recorded label lied) is refused at bind, before any reader runs; an
    unadmitted output there, or an admitted one on CPU, passes."""

    from torchlens._runnable.retention import _check_unretained_extent

    labels = {"call_id": "c:expand", "slot_id": "s:view"}
    _check_unretained_extent(torch.empty(4, device="meta"), **labels)
    _check_unretained_extent(torch.empty(4), admitted_view=True, **labels)
    with pytest.raises(RunCapabilityUnavailableError) as caught:
        _check_unretained_extent(torch.empty(4, device="meta"), admitted_view=True, **labels)
    assert caught.value.fields.get("detection_stage") == "op_allocation_execution"
    assert caught.value.fields.get("device") == "meta"


# --------------------------------------------------------------------------- #
# (f) defensive run-side decode-depth guard (C4)                               #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _runnable/verification.py.
def test_decode_literal_depth_guard_raises_not_recursionerror() -> None:
    """An over-depth nested literal raises a typed ``ValueError``, never ``RecursionError``."""

    node: Any = LiteralAtom(kind=LiteralAtomKind.INT, value=1)
    for _ in range(_MAX_DECODE_NESTING_DEPTH + 5):
        node = LiteralSequence(kind=LiteralSequenceKind.TUPLE, items=(node,))
    with pytest.raises(ValueError, match="decode depth"):
        _decode_literal(node)
