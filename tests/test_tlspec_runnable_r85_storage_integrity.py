"""Label/activation-rung STORAGE integrity (r85, r84 Sol SOL-1 HIGH).

The state-provenance trust decision is a ``{rung} x {axis}`` matrix. Each rung
(param, buffer ``address``, label / ``buffer_source``) trusts a receiver as
internal model state -- exempting it from the input-taint / layout nets -- only
when TWO independent facts hold:

* IDENTITY -- is this the CURRENT-session object that was stamped/labeled as
  state? Closed rung-by-rung: r79 (param), r81 (buffer ``address``), r82/r83
  (label). r84 hon1/free/secA re-confirmed the whole identity axis SOUND.
* STORAGE INTEGRITY -- is the receiver's LIVE storage still the storage it held
  when it was stamped/labeled, i.e. was it NOT ``.data=``/``set_``-rebound to
  foreign/input-derived storage AFTER the stamp?

r81 enforced STORAGE INTEGRITY for the buffer ``address`` rung only
(``session_validated_buffer_address``). SOL-1 proved the label / activation rung
enforced IDENTITY but NOT storage integrity: a state-derived activation
(``state_rooted = self.b * 1.0``) whose storage is ``.data``-rebound to
input-derived data mid-forward kept its (legitimately current-session-anchored)
label, so it re-bound as its same-named graph parent and replay recomputed the
PRE-rebind value -- reported ``VERIFIED`` on the SAME captured input, off the
fresh oracle by 19.0 (13.0 on a changed input). PRE-EXISTING (byte-identical
false-VERIFIED at ``b5db3f95`` and ``10975d98``); the contract puts "a host write
WITHIN the forward into captured storage" IN scope.

THE r85 CLOSURE (unified, one belt for the whole storage column). Every label
stamp records a STRONG storage keeper (``TensorMeta.label_storage``) at
``set_tensor_label`` -- the same single choke point that writes the r83 identity
anchor -- and the same two accessor gates (``get_tensor_label`` /
``get_label_list``, via ``_session_storage_gate_blocks``) plus the provenance
predicate (``_tensor_has_known_provenance``) reject a current-session label whose
LIVE storage no longer matches its keeper. Keying on the storage OBJECT
(``data_ptr`` + ``nbytes`` + device) draws the SHARP line: a ``.data=`` / ``set_``
rebind swaps the pointer and FAILS (orphaning the receiver -> the existing break
marker -> save refusal / ``unverifiable``); an IN-PLACE write into the object's
OWN storage (tracked ``copy_``, EMA ``mul_().add_()``, ``buf[:] = ...``) keeps
the pointer and PASSES -- honest, journaled state mutation stays ``VERIFIED``.

Because the belt is one mechanism at the shared choke points, it closes the whole
column at once -- the buffer-derived, PARAM-derived, ``set_``, ``buffer_source``,
and view-after-rebind vehicles all ceil together (see the column-completeness
block), while the honest siblings (view-BEFORE-rebind reading the intact base,
a same-storage own-view rebind whose value is unchanged) stay VERIFIED. A
same-storage rebind that re-strides to an INPUT layout keeps the pointer (my
storage check correctly passes it) and is caught instead by the r73/r75 layout
net (``PathDivergenceError``), pinned in :func:`test_v6_same_storage_input_strided_rebind_ceils`.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError, RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


@contextmanager
def _allow_break_marker() -> Iterator[None]:
    """Permit the ``unattributed_tensor_args`` warning during an attack capture.

    A rebound receiver correctly loses provenance, so the break-marker warning
    firing IS the post-fix behaviour; it is asserted structurally by the save
    refusal below, not swallowed silently.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*no graph/source provenance.*")
        yield


def _assert_not_falsely_verified(result: Any, oracle: torch.Tensor) -> None:
    """Assert a replay is never blessed VERIFIED while disagreeing with oracle 1."""

    max_diff = (result.output - oracle).abs().max().item()
    if result.report.path_faithfulness is PathFaithfulness.VERIFIED:
        assert not result.report.poisoned
        assert max_diff < 1e-6, f"FALSE VERIFIED: replay disagrees with oracle 1 by {max_diff}"


def _assert_storage_rebind_closed(
    model_factory: Callable[[], nn.Module],
    capture_input: torch.Tensor,
    run_inputs: tuple[torch.Tensor, ...],
    tmp_path: Path,
) -> None:
    """Assert a storage-rebind vehicle NEVER produces a false VERIFIED.

    Accepts EITHER fail-closed form the contract allows: a save-side
    ``RunnablePreflightError`` (the receiver is orphaned so no faithful sparse
    descriptor can be built), or -- if the artifact does save -- a per-input
    ``unverifiable``/poisoned run that never disagrees with oracle 1 while
    VERIFIED. Both same and changed inputs are exercised.
    """

    path = tmp_path / "rebind.tlspec"
    with _allow_break_marker():
        trace = tl.trace(model_factory(), capture_input, capture=_CAPTURE)
    try:
        trace.save(path, level="runnable", include_weights=True)
    except RunnablePreflightError:
        return  # save refusal -- the strongest fail-closed form; no input can run
    loaded = tl.load(path)
    for run_input in run_inputs:
        oracle = model_factory()(run_input.clone())
        try:
            result = loaded.run(inputs=run_input.clone())
        except (RunnablePreflightError, PathDivergenceError):
            continue  # run-side fail-closed is equally acceptable
        _assert_not_falsely_verified(result, oracle)


def _verified(model: nn.Module, capture_input: torch.Tensor, tmp_path: Path) -> Any:
    """Capture, save runnable, load and run on the capture input; return RunResult."""

    path = tmp_path / "honest.tlspec"
    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path).run(inputs=capture_input.clone())


# --------------------------------------------------------------------------- #
# SOL-1 -- the HIGH: a state-derived activation ``.data``-rebound to input data.
# --------------------------------------------------------------------------- #


class _ReboundActivation(nn.Module):
    """Rebind a current-session state-rooted activation to input-derived storage."""

    def __init__(self) -> None:
        """Register the buffer whose product activation receives a live label."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a relabeled activation after rebinding its storage to input data."""

        state_rooted = self.b * 1.0
        input_derived = x * 2.0
        state_rooted.data = input_derived.detach()
        return state_rooted * 1.0


@pytest.mark.smoke
def test_sol1_state_activation_data_rebind_never_false_verified(tmp_path: Path) -> None:
    """SOL-1 RED: a ``.data``-rebound state activation must not replay as VERIFIED.

    Pre-fix (main ``00987798``): ``same_input VERIFIED`` output ``[1,1,1,1]`` vs
    oracle ``[20,20,20,20]`` (max-diff 19.0) AND ``changed_input VERIFIED``
    (max-diff 13.0) -- wrong on the SAME captured input. Post-fix the rebound
    receiver loses provenance, orphaning the model output, so the runnable save
    refuses (fail-closed); neither input can produce a false VERIFIED.
    """

    _assert_storage_rebind_closed(
        _ReboundActivation,
        torch.full((4,), 10.0),
        (torch.full((4,), 10.0), torch.full((4,), 7.0)),
        tmp_path,
    )


# --------------------------------------------------------------------------- #
# ZERO-COLLATERAL -- the sharp line: an IN-PLACE write into the receiver's OWN
# storage keeps the pointer and MUST stay VERIFIED (free's honest battery).
# --------------------------------------------------------------------------- #


class _ConvBN(nn.Module):
    """Conv + BatchNorm; running stats are state, updated in place during train."""

    def __init__(self) -> None:
        """Build the conv/BN pair."""

        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the batch-normed convolution."""

        return self.bn(self.c(x))


class _EMA(nn.Module):
    """EMA buffer updated with an in-place ``mul_().add_()`` (pointer preserved)."""

    def __init__(self) -> None:
        """Register the EMA state buffer."""

        super().__init__()
        self.register_buffer("ema", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update the EMA buffer in place and fold it into the output."""

        with torch.no_grad():
            self.ema.mul_(0.9).add_(x.mean(0), alpha=0.1)
        return x + self.ema


class _CopyBuf(nn.Module):
    """Buffer written by an in-place tracked ``copy_`` of an input reduction."""

    def __init__(self) -> None:
        """Register the destination buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy an input reduction into the buffer in place, then fold it in."""

        with torch.no_grad():
            self.b.copy_(x.mean(0))
        return x + self.b


class _SliceAssign(nn.Module):
    """Buffer written by an in-place ``buf[:] = ...`` slice assignment."""

    def __init__(self) -> None:
        """Register the destination buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Slice-assign an input reduction into the buffer, then fold it in."""

        with torch.no_grad():
            self.b[:] = x.mean(0)
        return x + self.b


class _StateActInplace(nn.Module):
    """A state-derived activation mutated by a TRACKED in-place op (keeps storage)."""

    def __init__(self) -> None:
        """Register the state root."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a state activation after an in-place add that keeps its pointer."""

        y = self.b * 1.0
        y.add_(3.0)
        return y + x


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory, capture_input, train",
    [
        (_ConvBN, torch.randn(2, 3, 8, 8), True),
        (_EMA, torch.randn(8, 4), False),
        (_CopyBuf, torch.randn(8, 4), False),
        (_SliceAssign, torch.randn(8, 4), False),
        (_StateActInplace, torch.randn(4), False),
    ],
)
def test_inplace_own_storage_writes_stay_verified(
    factory: Callable[[], nn.Module],
    capture_input: torch.Tensor,
    train: bool,
    tmp_path: Path,
) -> None:
    """Zero-collateral: an in-place write into own storage keeps its verdict.

    The sharp line -- ``copy_`` / EMA ``mul_().add_()`` / ``buf[:] = ...`` /
    ``add_`` keep the storage pointer, so the r85 belt passes them and honest
    journaled/tracked state mutation stays VERIFIED. A regression here would be
    the r85 belt over-triggering on the very cases free's 50-case battery pins.
    """

    torch.manual_seed(0)
    model = factory()
    if train:
        model.train()
    else:
        model.eval()
    result = _verified(model, capture_input, tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_convbn_eval_running_stats_stay_verified(tmp_path: Path) -> None:
    """Zero-collateral: ConvBN in eval reads frozen running stats -> VERIFIED."""

    torch.manual_seed(0)
    model = _ConvBN()
    model.train()
    model(torch.randn(2, 3, 8, 8))  # populate running stats
    model.eval()
    result = _verified(model, torch.randn(2, 3, 8, 8), tmp_path)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


# --------------------------------------------------------------------------- #
# COLUMN COMPLETENESS -- the ONE belt subsumes every storage-rebind sibling.
#
# The r85 belt is a single mechanism at the shared label choke points, so it
# ceils every vehicle that swaps a state receiver's storage POINTER -- not just
# SOL-1's exact buffer-derived spelling. These vehicles all reduce to the same
# closure (save refusal / unverifiable) with NO per-vehicle code; the honest
# siblings that keep the pointer stay VERIFIED. No sibling escapes the belt.
# --------------------------------------------------------------------------- #


class _ParamReboundActivation(nn.Module):
    """A PARAM-derived activation ``.data``-rebound to input-derived storage."""

    def __init__(self) -> None:
        """Register the parameter whose product activation is rebound."""

        super().__init__()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a param-rooted activation after rebinding its storage."""

        y = self.w * 1.0
        y.data = (x * 2.0).detach()
        return y * 1.0


class _SetReboundActivation(nn.Module):
    """A state activation whose storage is swapped via ``set_(foreign_storage)``."""

    def __init__(self) -> None:
        """Register the state root."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a state activation after ``set_``-ing it to input-derived storage."""

        y = self.b * 1.0
        src = (x * 2.0).detach().clone()
        y.set_(src.untyped_storage(), 0, y.shape, y.stride())
        return y * 1.0


class _ViewAfterRebind(nn.Module):
    """A view taken AFTER the base's storage is rebound reads input-derived data."""

    def __init__(self) -> None:
        """Register the state root."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the base to input storage, then read a view of it."""

        y = self.b * 1.0
        y.data = (x * 2.0).detach()
        v = y.view(4)
        return v * 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory",
    [_ParamReboundActivation, _SetReboundActivation, _ViewAfterRebind],
    ids=["param_derived_data_rebind", "set_foreign_storage", "view_after_base_rebind"],
)
def test_storage_rebind_sibling_vehicles_never_false_verified(
    factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """The single belt ceils every pointer-swap sibling (param / set_ / view-base).

    Each pre-fix would false-VERIFY the pre-rebind value exactly like SOL-1; the
    same storage-integrity belt orphans the rebound receiver so all three refuse.
    """

    _assert_storage_rebind_closed(
        factory,
        torch.full((4,), 10.0),
        (torch.full((4,), 10.0), torch.full((4,), 7.0)),
        tmp_path,
    )


def test_buffer_source_rung_honors_storage_pin() -> None:
    """The ``buffer_source`` rung carries the SAME storage pin as ``label_raw``.

    ``buffer_source`` is a promoted pre-buffer producer label (a tensor logged as
    an intermediate then registered as a buffer). It is gated by the identical
    ``label_storage_intact`` check in ``_tensor_has_known_provenance``, so a
    ``buffer_source``-carrying object whose storage is rebound is no longer
    trusted -- the sibling free warned could be "missed". Asserted directly on the
    internal API (mirroring r81/r83's rung-level probes): the pin survives the
    promotion and fails on rebind while an untouched object stays intact.
    """

    from torchlens.backends.torch import _tl

    _tl.begin_label_session()
    try:
        t = torch.ones(4)
        _tl.set_tensor_label(t, "mul_1_1_raw")
        _tl.promote_label_to_buffer_source_and_clear_label(t)
        meta = _tl.get_tensor_meta(t)
        assert meta is not None
        assert meta.buffer_source == "mul_1_1_raw"
        assert meta.label_raw is None
        assert _tl.session_label_storage_intact(meta, t) is True
        t.data = torch.arange(4, dtype=torch.float32).detach()
        assert _tl.session_label_storage_intact(meta, t) is False
    finally:
        _tl.end_label_session()


# --------------------------------------------------------------------------- #
# COLUMN COMPLETENESS -- honest siblings that KEEP the pointer stay VERIFIED.
# --------------------------------------------------------------------------- #


class _ViewBeforeRebind(nn.Module):
    """A view taken BEFORE the base is rebound still reads the intact base storage."""

    def __init__(self) -> None:
        """Register the state root."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """View the base first, THEN rebind the base object (the view is honest)."""

        y = self.b * 1.0
        v = y.view(4)
        y.data = (x * 2.0).detach()  # rebinds y only; v still reads the intact base
        return v * 1.0


class _SameStorageOwnViewRebind(nn.Module):
    """A ``.data=`` to a view of the object's OWN storage keeps the pointer + value."""

    def __init__(self) -> None:
        """Register the state root."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind to a view of the SAME storage -- value unchanged, pointer kept."""

        y = self.b * 1.0
        y.data = y.view(4)
        return y * 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory",
    [_ViewBeforeRebind, _SameStorageOwnViewRebind],
    ids=["view_before_base_rebind", "same_storage_own_view"],
)
def test_pointer_preserving_siblings_stay_verified(
    factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """Honest siblings that do NOT swap the receiver's storage pointer stay VERIFIED.

    A view captured before the base's ``.data=`` still reads the intact base
    storage; a ``.data=`` to a view of the receiver's OWN storage keeps both the
    pointer and the value. Both are honest and replay exactly -- the belt keys on
    the storage OBJECT, so neither is over-triggered.
    """

    torch.manual_seed(0)
    model = factory()
    capture_input = torch.full((4,), 10.0)
    result = _verified(model, capture_input, tmp_path)
    oracle = factory()(capture_input.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    assert (result.output - oracle).abs().max().item() < 1e-6


class _V6SameStorageInputStrided(nn.Module):
    """hon1 V6: a same-storage ``.data=`` re-strided to the INPUT's layout."""

    def __init__(self) -> None:
        """Register a 4-D state buffer."""

        super().__init__()
        self.register_buffer("b", torch.ones(2, 3, 4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind ``.data`` to a re-strided view of OWN storage matching x's strides."""

        y = self.b * 1.0
        y.data = torch.as_strided(y, y.shape, x.stride())
        return (y + 0.0).reshape(-1).sum() * 1.0


def test_v6_same_storage_input_strided_rebind_ceils(tmp_path: Path) -> None:
    """hon1 V6: same-storage input-strided ``.data=`` is caught by the layout net.

    The rebind keeps the storage POINTER, so the r85 belt correctly passes it (it
    is the object's own storage); the INPUT-layout dependency is instead caught by
    the r73/r75 layout net -- a changed-layout (channels_last) input diverges
    while the same-layout control stays honestly VERIFIED. Verifies the r85 belt
    and the layout net compose without a gap.
    """

    torch.manual_seed(0)
    model = _V6SameStorageInputStrided()
    capture_input = torch.randn(2, 3, 4, 4)
    path = tmp_path / "v6.tlspec"
    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    loaded = tl.load(path)

    control = loaded.run(inputs=capture_input.clone())
    assert control.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not control.report.poisoned

    twin = capture_input.clone().to(memory_format=torch.channels_last)
    with pytest.raises(PathDivergenceError):
        loaded.run(inputs=twin)


# --------------------------------------------------------------------------- #
# backend_address CLEANUP (r85, free FINDING-1 MED).
#
# r83's dee9d679 decoupled .backend_address from .address to fix the C2 buffer
# authority, but set the registered-only override UNCONDITIONALLY -- forcing
# EVERY op/activation node to None (contra its own commit "op/input ...
# unaffected") and leaving the model-output layer (built by copying the last op)
# reporting the coupled address while the aliasing op node reported None. r85
# applies the override ONLY to buffer-addressed nodes (Option a): an op node
# keeps backend_address == address (matching every other backend, where an op
# carries a jaxpr:/uop: handle the output layer inherits), and the C2 buffer
# decoupling is preserved verbatim.
# --------------------------------------------------------------------------- #


class _ConvBNAddress(nn.Module):
    """Conv + BatchNorm whose op nodes carry module addresses ('c', 'bn')."""

    def __init__(self) -> None:
        """Build a conv/BN pair with distinct module addresses."""

        super().__init__()
        self.c = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the batch-normed convolution (bn is the output-aliased module)."""

        return self.bn(self.c(x))


@pytest.mark.smoke
def test_backend_address_op_node_and_output_layer_agree() -> None:
    """free FINDING-1: an op node and the output layer for the SAME module agree.

    Pre-fix the batchnorm op node reported ``backend_address=None`` while the
    output layer aliasing the same ``bn`` module reported ``'bn'`` -- an
    intra-trace inconsistency on a public field. Post-fix a module-addressed op
    node keeps ``backend_address == address`` (not ``None``), matching the output
    layer.
    """

    torch.manual_seed(0)
    model = _ConvBNAddress()
    model.train()
    model(torch.randn(2, 3, 8, 8))
    model.eval()
    log = tl.trace(model, torch.randn(2, 3, 8, 8), capture=_CAPTURE)

    by_type = {layer.layer_type: layer for layer in log}
    conv = by_type["conv2d"]
    bn = by_type["batchnorm"]
    out = by_type["output"]

    # Op nodes keep backend_address coupled to their module address (not None).
    assert conv.backend_address == conv.address == "c"
    assert bn.backend_address == bn.address == "bn"
    # The output layer aliases the bn module and MUST agree with the bn op node.
    assert out.address == "bn"
    assert out.backend_address == bn.backend_address == "bn"


@pytest.mark.smoke
def test_backend_address_registered_vs_plain_attr_decoupling_preserved() -> None:
    """The C2 buffer decoupling is preserved: registered keeps its addr, plain-attr None.

    The r85 op-node fix must NOT disturb the r83 C2 intent -- a registered buffer
    still reports ``backend_address == address`` and a plain-attribute buffer
    (display address but not declared state) still reports ``None``.
    """

    class _MixedBuffers(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("rb", torch.ones(4))
            self.pa = torch.ones(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.rb + self.pa

    torch.manual_seed(0)
    log = tl.trace(_MixedBuffers(), torch.randn(4), capture=_CAPTURE)
    by_address = {layer.address: layer for layer in log if getattr(layer, "is_buffer", False)}
    assert by_address["rb"].backend_address == "rb"  # registered state keeps both
    assert by_address["pa"].backend_address is None  # plain-attr not declared state
