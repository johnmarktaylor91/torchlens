"""Label-rung current-session anchoring (r83 C1, r82 hon1/free/Sol HIGHs).

THREE INDEPENDENT r82 LANES broke the SAME class with three different vehicles:
label provenance was validated by TEXT MEMBERSHIP in the active capture's live
event index, with no per-object anchoring. Label text is deterministic per
op-kind + ordinal, so an ordinary op in a LATER, unrelated capture regenerates
the same string and a tensor still carrying a label from an EARLIER capture was
blessed as current-session model state.

* hon1 -- a stock ``register_forward_hook`` activation collector appending into
  a module-global list. SAME input, wrong parent BIND: ``VERIFIED``,
  ``poisoned=False``, replay off the fresh oracle by 18.0. The control with the
  label stripped (byte-identical value) correctly REFUSED at save, isolating the
  stale label as the sole delta.
* free -- an activation cached on a helper ``nn.Module``, then ``.data``-rebound
  into an input-derived layout branch: ``VERIFIED``, max-diff 20.0, where the
  fresh-tensor control ceils ``UNVERIFIABLE``/poisoned.
* Sol -- a tensor in a list inside a ``types.ModuleType``, with a live
  ``x + 100.0`` regenerating the colliding label: ``VERIFIED``, max-diff 100.15.

r79 gave the PARAM rung a per-session object-identity belt and r81 gave the
buffer ``address`` rung one (``session_validated_buffer_address``); the sibling
label components were left unanchored, which is exactly the asymmetry these
vehicles rode. All three are PRE-EXISTING (identical at the r79 base
``10975d98``), not r81 regressions.

THE r83 CLOSURE. ``TensorMeta.label_session`` records the monotonic token of the
capture that issued the object's label, written by ``set_tensor_label`` -- the
single choke point every torch-backend label stamp flows through, so a stamp
cannot exist without its anchor and no future stamp site can silently escape the
belt. The gate sits in ``get_tensor_label`` / ``get_label_list``, the accessors
every label consumer reads through, so the graph-parent binder, the layout
ancestry rooting rung, the dispatch-origin ladder, the host-escape attribution
ladder and the replay-template builder are all closed at once. Gating only the
two rungs in ``_tensor_has_known_provenance`` was empirically INSUFFICIENT --
free's launder rode the layout/origin rungs instead, which is pinned below.

ROOT A adds an inventory-driven sweep over the same session, so the leak
vehicles the reachability walk cannot reach -- a helper ``nn.Module``, an
``nn.Sequential``, an ``__slots__`` object, a module-global appended to from a
hook, a class attribute, a container nested in a ``types.ModuleType`` -- are
cleared too. It runs when the NEXT session is installed, not at capture cleanup:
``_cleanup_model_session`` precedes ``_postprocess``, and sweeping at cleanup
broke the output-attribution fallback in ``postprocess.graph_traversal``, which
still reads output-tensor labels. Deferring is equally complete for the leak
class, since a stale stamp can only ever matter to a subsequent capture.

Root A is defence-in-depth only. The belt is correct even when the sweep reaches
nothing, which ``test_stale_label_is_invisible_to_the_next_capture`` and
``test_unanchored_stamp_is_never_provenance`` assert directly by re-applying a
stale stamp with cleanup deliberately defeated.
"""

from __future__ import annotations

import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import (
    get_tensor_meta,
    raw_tensor_label,
    set_buffer_address,
    set_tensor_label,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _nchw() -> torch.Tensor:
    """Return a fixed contiguous NCHW probe input."""

    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)


def _twin(x: torch.Tensor) -> torch.Tensor:
    """Return the same-value channels_last layout twin of an input."""

    return x.clone().to(memory_format=torch.channels_last)


@contextmanager
def _allow_break_marker() -> Iterator[None]:
    """Permit the ``unattributed_tensor_args`` warning during an attack capture.

    Every vehicle here feeds the model a tensor with no current-session
    provenance, so the break marker firing is the CORRECT post-fix behaviour --
    it is asserted explicitly in
    :func:`test_hook_vehicle_leaves_the_unattributed_break_marker` rather than
    being silently swallowed everywhere.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*no graph/source provenance.*")
        yield


def _run_artifact(
    model: nn.Module,
    capture_input: torch.Tensor,
    run_input: torch.Tensor,
    path: Path,
) -> Any:
    """Capture, save runnable, load and run; return the ``RunResult``."""

    with _allow_break_marker():
        trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return tl.load(path).run(inputs=run_input)


def _assert_not_falsely_verified(result: Any, oracle: torch.Tensor) -> None:
    """Assert a replay is never blessed VERIFIED while disagreeing with oracle 1.

    The whole point of the sprint: a numerically wrong replay must ceil
    (``unverifiable``/poisoned), never report faithfulness.
    """

    max_diff = (result.output - oracle).abs().max().item()
    if result.report.path_faithfulness is PathFaithfulness.VERIFIED:
        assert not result.report.poisoned
        assert max_diff < 1e-6, f"FALSE VERIFIED: replay disagrees with oracle 1 by {max_diff}"


# --------------------------------------------------------------------------- #
# C1 -- hon1's vehicle: a stock forward-hook collector, SAME input, wrong BIND.
# --------------------------------------------------------------------------- #

_HOOK_FEATURES: list[torch.Tensor] = []


class _Inner(nn.Module):
    """Submodule whose output is harvested by an ordinary forward hook."""

    def __init__(self, w: float) -> None:
        """Store the scalar state whose product is harvested."""

        super().__init__()
        self.register_buffer("w", torch.full((4,), w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a state-rooted product carrying raw label ``mul_1_3_raw``."""

        return self.w * 3.0


class _Donor(nn.Module):
    """Capture-1 model whose inner output is collected by a stock hook."""

    def __init__(self) -> None:
        """Build the donor and attach a stock activation-collector hook."""

        super().__init__()
        self.inner = _Inner(7.0)
        self.inner.register_forward_hook(
            lambda _module, _args, output: _HOOK_FEATURES.append(output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the inner submodule and fold its output into the output."""

        return x + self.inner(x)


class _Consumer(nn.Module):
    """Capture-2 model whose OWN first op regenerates the donor's label text."""

    def __init__(self, foreign: torch.Tensor) -> None:
        """Hold the harvested foreign tensor behind an underscore attribute."""

        super().__init__()
        self.register_buffer("w", torch.full((4,), 1.0))
        self._f = [foreign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the foreign tensor alongside a same-labeled live op."""

        s = self.w * 3.0
        return x * 1.0 + self._f[0] + s * 0.0


@pytest.mark.smoke
def test_hook_collected_stale_label_is_not_current_session_provenance(
    tmp_path: Path,
) -> None:
    """hon1 RED: a hook-harvested tensor must never bind as a live-label parent.

    Pre-fix this reported ``VERIFIED``/``poisoned=False`` on the SAME input with
    replay off oracle 1 by 18.0 -- a silent WRONG VALUE bind, the most severe of
    the three vehicles. The control (same bytes, label stripped) refused at save;
    the stale label was the sole delta, so the fixed behaviour must match the
    control.
    """

    _HOOK_FEATURES.clear()
    x = torch.full((4,), 2.0)
    with _allow_break_marker():
        tl.trace(_Donor(), x, capture=_CAPTURE)
    harvested = _HOOK_FEATURES[0]

    # The vehicle only exists if the harvested tensor really kept a live label
    # text from capture 1 -- assert the collision is real, not assumed away.
    # (Root A sweeps it once the NEXT session opens; the belt below is what
    # makes it harmless in the meantime.)
    assert raw_tensor_label(harvested) == "mul_1_3_raw"

    model = _Consumer(harvested)
    with torch.no_grad():
        oracle = _Consumer(harvested)(x.clone())

    try:
        result = _run_artifact(model, x, x.clone(), tmp_path / "hook.tlspec")
    except RunnablePreflightError:
        return  # honest fail-closed, exactly what the label-stripped control does
    _assert_not_falsely_verified(result, oracle)


@pytest.mark.smoke
def test_hook_vehicle_leaves_the_unattributed_break_marker() -> None:
    """The positive signal: the foreign tensor must leave the break marker.

    Pre-fix the stale label SUPPRESSED ``unattributed_tensor_args`` -- the very
    ancestry-integrity marker the layout ladder and the save-side preflight key
    on. Asserting the warning fires proves the belt rejects the tensor at the
    provenance rung, rather than the ceiling arriving by some unrelated route.
    """

    _HOOK_FEATURES.clear()
    x = torch.full((4,), 2.0)
    with _allow_break_marker():
        tl.trace(_Donor(), x, capture=_CAPTURE)
    harvested = _HOOK_FEATURES[0]
    assert raw_tensor_label(harvested) == "mul_1_3_raw"

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        tl.trace(_Consumer(harvested), x, capture=_CAPTURE)


@pytest.mark.smoke
def test_hook_collected_control_without_stale_label_refuses(tmp_path: Path) -> None:
    """The isolating control: identical bytes with no stale label must refuse.

    Pins the finding airtight -- the ATTACK above may not pass by becoming
    *more* permissive than this control ever was.
    """

    _HOOK_FEATURES.clear()
    x = torch.full((4,), 2.0)
    with _allow_break_marker():
        tl.trace(_Donor(), x, capture=_CAPTURE)
    stripped = _HOOK_FEATURES[0].clone().detach()
    assert raw_tensor_label(stripped) is None

    with pytest.raises(RunnablePreflightError):
        _run_artifact(_Consumer(stripped), x, x.clone(), tmp_path / "hookctl.tlspec")


# --------------------------------------------------------------------------- #
# C1 -- free's vehicle: activation cached on a helper nn.Module, .data= launder.
# --------------------------------------------------------------------------- #

_SIDECAR = nn.Module()


class _SidecarDonor(nn.Module):
    """Capture-1 model caching a state-rooted activation on a helper module."""

    def __init__(self) -> None:
        """Register the buffer whose product is cached on the sidecar."""

        super().__init__()
        torch.manual_seed(3)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cache a state-rooted intermediate on an ordinary helper module."""

        t = self.b * 1.0
        _SIDECAR.cache = t
        return x + t


class _Launder(nn.Module):
    """Capture-2 model rebinding the leaked object to input-derived data."""

    def __init__(self, foreign: torch.Tensor) -> None:
        """Hold the foreign object and a same-labeled registered buffer."""

        super().__init__()
        torch.manual_seed(1)
        self.register_buffer("r", torch.randn(2, 4, 8, 8))
        self._holder = [foreign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a layout that is input-derived through the rebound object."""

        s = self.r * 1.0
        y = x * 2.0
        q = self._holder[0]
        q.data = y.detach()
        z = q * 1.0
        out = y + 10.0 if z.is_contiguous(memory_format=torch.channels_last) else y - 10.0
        return out + s * 0.0


@pytest.mark.parametrize("changed_input", [True, False], ids=["twin", "same"])
def test_helper_module_cached_stale_label_cannot_launder_layout(
    tmp_path: Path, changed_input: bool
) -> None:
    """free RED: a stale label on a helper-module cache must not bless a launder.

    Pre-fix: ``VERIFIED``/``poisoned=False`` at max-diff 20.0 (changed input) and
    at max-diff 0.0 on the SAME input, where the fresh-tensor control ceiled
    ``UNVERIFIABLE``/poisoned both ways. Both spellings are pinned because the
    same-input one is blessed-as-faithful when the honest verdict is a ceiling.
    """

    if hasattr(_SIDECAR, "cache"):
        del _SIDECAR.cache
    x = _nchw()
    with _allow_break_marker():
        tl.trace(_SidecarDonor(), x, capture=_CAPTURE)
    leaked = _SIDECAR.cache
    # A helper ``nn.Module`` value is skipped by the cleanup walk
    # (``isinstance(value, nn.Module) -> return``), so the stamp is still here.
    assert raw_tensor_label(leaked) is not None

    run_input = _twin(_nchw()) if changed_input else _nchw()
    model = _Launder(leaked)
    with torch.no_grad():
        oracle = _Launder(leaked)(run_input.clone())

    try:
        result = _run_artifact(model, x, run_input, tmp_path / f"free_{changed_input}.tlspec")
    except RunnablePreflightError:
        return
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    _assert_not_falsely_verified(result, oracle)


# --------------------------------------------------------------------------- #
# C1 -- Sol's vehicle: a tensor nested in a list inside a types.ModuleType.
# --------------------------------------------------------------------------- #


class _StashDonor(nn.Module):
    """Capture-1 model stashing a live op output in a ModuleType-nested list.

    r81's ModuleType cleanup sweep is SHALLOW, so a tensor one container deep
    inside the module namespace is never reached by it -- the leak Sol used.
    """

    def __init__(self, stash: types.ModuleType) -> None:
        """Hold the module namespace used as the stash."""

        super().__init__()
        self._stash = stash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Emit ``add_1_2_raw`` and stash it one container deep.

        The stashed tensor is deliberately NOT returned: an OUTPUT tensor's
        label is cleared by postprocess, which would defeat the vehicle.
        """

        y = x + 100.0
        self._stash.box = [y]
        return x * 0.0


class _StashConsumer(nn.Module):
    """Capture-2 model regenerating the donor's label text on its own first op."""

    def __init__(self, stash: types.ModuleType) -> None:
        """Hold the stash the donor wrote into."""

        super().__init__()
        self._stash = stash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the stale object's storage to input-derived data."""

        live = x + 100.0
        derived = x * 2.0
        q = self._stash.box[0]
        q.data = derived.detach()
        return q * 1.0 + live * 0.0


def test_moduletype_nested_stale_label_cannot_launder(tmp_path: Path) -> None:
    """Sol RED: a container-nested ModuleType stash must not resolve as provenance.

    Pre-fix: ``VERIFIED``, ``poisoned=False``, no witness-coverage gap, replay
    vs fresh-oracle max-diff 100.15 -- and it falsified the shipped contract's
    claim that deeper stashes are harmless because the belt gates them.
    """

    stash = types.ModuleType("r83_stash")
    x = _nchw()
    with _allow_break_marker():
        tl.trace(_StashDonor(stash), x, capture=_CAPTURE)
    # r81's ModuleType sweep is shallow and never reaches a tensor nested one
    # container deep, so the stamp survives the donor capture.
    assert raw_tensor_label(stash.box[0]) is not None

    run_input = _twin(_nchw())
    model = _StashConsumer(stash)
    with torch.no_grad():
        oracle = _StashConsumer(stash)(run_input.clone())

    try:
        result = _run_artifact(model, x, run_input, tmp_path / "sol.tlspec")
    except RunnablePreflightError:
        return
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    _assert_not_falsely_verified(result, oracle)


# --------------------------------------------------------------------------- #
# C1 -- the isolated consumer matrix (free's attack2). Injects the stale stamp
# directly, so the belt is tested independently of ANY leak vehicle: even if a
# future leak escapes cleanup entirely, an unanchored stamp is not provenance.
# --------------------------------------------------------------------------- #


class _InjectLaunder(nn.Module):
    """Layout launder consuming a directly-stamped foreign tensor."""

    def __init__(self, foreign: torch.Tensor) -> None:
        """Hold the foreign object behind an underscore attribute."""

        super().__init__()
        torch.manual_seed(1)
        self.register_buffer("r", torch.randn(2, 4, 8, 8))
        self._holder = [foreign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the rebound foreign object's layout."""

        y = x * 2.0 + self.r * 0.0
        q = self._holder[0]
        q.data = y.detach()
        z = q * 1.0
        return y + 10.0 if z.is_contiguous(memory_format=torch.channels_last) else y - 10.0


def _stamp_nothing(t: torch.Tensor) -> None:
    """Leave a tensor unstamped (the honest fail-closed control)."""


def _stamp_colliding_label(t: torch.Tensor) -> None:
    """Stamp a label whose TEXT collides with a live event in the launder."""

    set_tensor_label(t, "buffer_1_raw")


def _stamp_noncolliding_label(t: torch.Tensor) -> None:
    """Stamp a label whose text does NOT collide with any live event."""

    set_tensor_label(t, "mul_1_1_raw")


def _stamp_buffer_address(t: torch.Tensor) -> None:
    """Stamp a stale buffer address (the r81 belt's own component)."""

    set_buffer_address(t, "r")


def _stamp_label_and_address(t: torch.Tensor) -> None:
    """Stamp both a colliding label and a stale buffer address."""

    set_tensor_label(t, "buffer_1_raw")
    set_buffer_address(t, "r")


@pytest.mark.parametrize(
    "stamp",
    [
        _stamp_nothing,
        _stamp_colliding_label,
        _stamp_noncolliding_label,
        _stamp_buffer_address,
        _stamp_label_and_address,
    ],
    ids=[
        "no_stamp",
        "colliding_label",
        "noncolliding_label",
        "stale_address",
        "label_plus_address",
    ],
)
def test_unanchored_stamp_is_never_provenance(
    tmp_path: Path, stamp: Callable[[torch.Tensor], None]
) -> None:
    """Every stale-stamp spelling must ceil exactly like the no-stamp control.

    Pre-fix only two rows diverged from the control and both were the label
    rows: ``colliding_label`` and ``label_plus_address`` reported
    ``VERIFIED``/``poisoned=False`` at max-diff 20.0. ``noncolliding_label`` and
    ``stale_address`` already ceiled -- the r81 address belt holding, and direct
    positive evidence that the mechanism was the TEXT collision and nothing
    else. Keeping all five rows pins the diagnosis, not just the symptom.
    """

    torch.manual_seed(5)
    foreign = torch.randn(2, 4, 8, 8)
    stamp(foreign)

    x = _nchw()
    run_input = _twin(_nchw())
    model = _InjectLaunder(foreign)
    with torch.no_grad():
        oracle = _InjectLaunder(foreign)(run_input.clone())

    try:
        result = _run_artifact(model, x, run_input, tmp_path / "inject.tlspec")
    except RunnablePreflightError:
        return
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    _assert_not_falsely_verified(result, oracle)


@pytest.mark.smoke
def test_label_session_token_advances_and_anchors_per_capture() -> None:
    """The anchor itself: each capture issues a fresh token, retired at cleanup.

    The structural property the belt rests on -- a label issued by an earlier
    capture can never carry the active capture's token, because tokens are
    monotonic and never reused.
    """

    from torchlens.backends.torch import _tl

    assert _tl.active_label_session_token() is None  # retired outside a capture

    seen: list[int | None] = []

    class _Probe(nn.Module):
        """Model that samples the active session token from inside forward."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Record the active token, then emit one op."""

            seen.append(_tl.active_label_session_token())
            return x * 2.0

    tl.trace(_Probe(), torch.ones(4), capture=_CAPTURE)
    tl.trace(_Probe(), torch.ones(4), capture=_CAPTURE)

    assert len(seen) == 2
    assert seen[0] is not None and seen[1] is not None
    assert seen[1] > seen[0], "each capture must install a fresh, monotonic token"
    assert _tl.active_label_session_token() is None, "session must retire at cleanup"


@pytest.mark.smoke
def test_stale_label_is_invisible_to_the_next_capture() -> None:
    """A prior capture's label must not be readable as provenance in a new one.

    THE LOAD-BEARING PROPERTY, asserted with cleanup deliberately defeated: the
    stale stamp is re-applied by hand AFTER the donor capture, reconstructing
    exactly the state a future leak vehicle that escapes the inventory entirely
    would produce. The belt must reject it on the anchor alone -- root A closing
    today's known vehicles must never be what the correctness rests on.
    """

    from torchlens.backends.torch import _tl
    from torchlens.backends.torch._tl import TensorMeta, get_tensor_label

    donor_token: list[int | None] = []

    class _TokenDonor(nn.Module):
        """Donor that reports the token its own capture session issued."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sample the active session token and emit one op."""

            donor_token.append(_tl.active_label_session_token())
            return x * 3.0

    tl.trace(_TokenDonor(), torch.full((4,), 2.0), capture=_CAPTURE)
    assert donor_token[0] is not None

    # Re-apply the donor session's stamp, as though cleanup had never run.
    harvested = torch.full((4,), 21.0)
    harvested._tl = TensorMeta(label_raw="mul_1_3_raw", label_session=donor_token[0])
    assert raw_tensor_label(harvested) == "mul_1_3_raw"

    observed: list[Any] = []

    class _Reader(nn.Module):
        """Model that reads the foreign tensor's label from inside a capture."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sample the gated and ungated label views mid-capture."""

            observed.append((get_tensor_label(harvested), raw_tensor_label(harvested)))
            return x * 1.0

    with _allow_break_marker():
        tl.trace(_Reader(), torch.ones(4), capture=_CAPTURE)

    gated, ungated = observed[0]
    assert gated is None, "a prior capture's label must not read as provenance"
    assert ungated == "mul_1_3_raw", "the raw view stays diagnostic-only, not gated"


# --------------------------------------------------------------------------- #
# C1 ROOT A -- inventory-driven cleanup over free's leak-vehicle map.
#
# The reachability walk in ``model_prep._clear_session_tensor_metadata`` has
# structural blind spots: it returns immediately for any ``nn.Module`` value
# outside the traced tree and for any object with no ``__dict__``, r81's
# ``types.ModuleType`` sweep is shallow, and globals are reached only through
# ``forward.__code__.co_names``. Enumerating stamped objects by REGISTRATION
# reaches all of them. Defence-in-depth only -- the belt above is what makes an
# escaped stamp harmless.
# --------------------------------------------------------------------------- #

_LEAK_HELPER_MODULE = nn.Module()
_LEAK_SEQUENTIAL = nn.Sequential(nn.Identity())
_LEAK_MODULE_TYPE = types.ModuleType("r83_leak_modtype")
_LEAK_GLOBAL_LIST: list[torch.Tensor] = []


class _SlotsCache:
    """``__slots__`` holder -- no ``__dict__``, so the cleanup walk skips it."""

    __slots__ = ("cache",)


_LEAK_SLOTS = _SlotsCache()


class _LeakDonor(nn.Module):
    """Donor stashing one state-rooted product into each leak vehicle."""

    CLASS_CACHE: torch.Tensor | None = None

    def __init__(self) -> None:
        """Register the buffer whose products are stashed."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stash a distinct labeled tensor in every vehicle."""

        _LEAK_HELPER_MODULE.cache = self.b * 1.0
        _LEAK_SEQUENTIAL.cache = self.b * 2.0
        _LEAK_SLOTS.cache = self.b * 3.0
        _LEAK_MODULE_TYPE.box = [self.b * 4.0]
        _LEAK_GLOBAL_LIST.append(self.b * 5.0)
        type(self).CLASS_CACHE = self.b * 6.0
        return x + self.b


class _Passthrough(nn.Module):
    """Trivial model used purely to open a subsequent capture session."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Emit one op."""

        return x * 2.0


def test_leak_vehicles_are_swept_by_the_session_inventory() -> None:
    """Every vehicle in free's map must be cleared, walk blind spots included.

    Pre-fix, four of these LEAKED past cleanup: the helper ``nn.Module``, the
    ``nn.Sequential``, the ``__slots__`` object and the helper-function-reached
    object; the ModuleType-nested list defeated r81's shallow sweep. The sweep
    runs when the NEXT session is installed rather than at cleanup, because
    ``_cleanup_model_session`` precedes ``_postprocess`` and the output
    attribution fallback in ``postprocess.graph_traversal`` still reads
    output-tensor labels -- sweeping at cleanup broke it. Deferring is equally
    complete for the leak class: a stale stamp can only matter to a subsequent
    capture, and it is gone before that capture stamps or reads anything.
    """

    from torchlens.backends.torch import _tl

    _LEAK_GLOBAL_LIST.clear()
    _LeakDonor.CLASS_CACHE = None

    inventory_ids: list[set[int]] = []

    class _InventoryProbe(_LeakDonor):
        """Donor that samples the session inventory before the capture ends."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Stash into every vehicle, then read the inventory back."""

            out = super().forward(x)
            inventory_ids.append({id(t) for t in _tl.session_labeled_tensors()})
            return out

    tl.trace(_InventoryProbe(), torch.ones(4), capture=_CAPTURE)

    vehicles = {
        "helper nn.Module attr": _LEAK_HELPER_MODULE.cache,
        "nn.Sequential attr": _LEAK_SEQUENTIAL.cache,
        "__slots__ object attr": _LEAK_SLOTS.cache,
        "ModuleType-nested list": _LEAK_MODULE_TYPE.box[0],
        "module-global list": _LEAK_GLOBAL_LIST[0],
        # ``type(self).CLASS_CACHE`` writes onto the RUNNING class.
        "class attribute": _InventoryProbe.CLASS_CACHE,
    }
    assert all(t is not None for t in vehicles.values()), "a vehicle failed to stash"

    # ROOT A, stated directly: the inventory NAMES every vehicle, including the
    # four the reachability walk cannot reach and the ModuleType-nested one
    # r81's shallow sweep could not. That is the property the sweep rests on.
    stamped = inventory_ids[0]
    for name, tensor in vehicles.items():
        assert id(tensor) in stamped, f"{name} is not in the session inventory"

    # And the sweep clears all of them once the next session is installed.
    tl.trace(_Passthrough(), torch.ones(4), capture=_CAPTURE)
    for name, tensor in vehicles.items():
        assert raw_tensor_label(tensor) is None, f"{name} kept a stale stamp"
    assert _tl.active_label_session_token() is None


def test_sweep_never_touches_a_foreign_or_unstamped_tensor() -> None:
    """The sweep clears session stamps only; foreign ``_tl`` values survive.

    ``clear_meta`` preserves a non-TorchLens ``_tl``, and an object the session
    never stamped is not in the inventory at all.
    """

    from torchlens.backends.torch import _tl

    untouched = torch.ones(4)
    sentinel = object()
    untouched._tl = sentinel

    tl.trace(_Passthrough(), torch.ones(4), capture=_CAPTURE)
    tl.trace(_Passthrough(), torch.ones(4), capture=_CAPTURE)

    assert untouched._tl is sentinel
    assert _tl.sweep_retired_label_stamps() >= 0  # idempotent, never raises


# --------------------------------------------------------------------------- #
# ZERO COLLATERAL -- honest models that were VERIFIED before must stay VERIFIED.
# --------------------------------------------------------------------------- #


class _HonestLayoutBranch(nn.Module):
    """Registered buffer whose LAYOUT is read -- the residual-(3) rung itself."""

    def __init__(self) -> None:
        """Register the state buffer the branch reads."""

        super().__init__()
        torch.manual_seed(2)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the buffer's own (state-derived) layout."""

        z = self.b * 1.0
        return x + 1.0 if z.is_contiguous(memory_format=torch.channels_last) else x - 1.0


class _ConvBN(nn.Module):
    """Conv+BatchNorm, the canonical running-stats honesty case."""

    def __init__(self) -> None:
        """Build the conv/bn stack."""

        super().__init__()
        torch.manual_seed(4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the stack."""

        return self.bn(self.conv(x))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "factory, train",
    [
        (_HonestLayoutBranch, False),
        (_ConvBN, False),
        (_ConvBN, True),
    ],
    ids=["state_layout_branch", "convbn_eval", "convbn_train"],
)
def test_honest_models_stay_verified(
    tmp_path: Path, factory: Callable[[], nn.Module], train: bool
) -> None:
    """Honest state reads must not be ceiled by the label anchor gate.

    The gate must reject FOREIGN labels only; a live capture's own labels are
    anchored by construction, so nothing honest may change verdict.
    """

    model = factory()
    model.train(train)
    x = _nchw()
    result = _run_artifact(model, x, _nchw(), tmp_path / "honest.tlspec")
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_repeat_captures_of_one_instance_stay_verified(tmp_path: Path) -> None:
    """Three consecutive captures of ONE instance must all stay VERIFIED.

    The shape a session-scoped anchor is most likely to over-ceil: the second
    and third captures see tensors whose objects the previous session stamped.
    """

    model = _HonestLayoutBranch()
    for index in range(3):
        result = _run_artifact(model, _nchw(), _nchw(), tmp_path / f"repeat{index}.tlspec")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, f"capture {index}"
        assert not result.report.poisoned


@pytest.mark.smoke
def test_shared_buffer_object_across_instances_stays_verified(tmp_path: Path) -> None:
    """Three instances sharing ONE buffer object across captures stay VERIFIED."""

    torch.manual_seed(2)
    shared = torch.randn(2, 4, 8, 8)

    for index in range(3):
        model = _HonestLayoutBranch()
        model.b = shared
        result = _run_artifact(model, _nchw(), _nchw(), tmp_path / f"shared{index}.tlspec")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED, f"capture {index}"
        assert not result.report.poisoned


@pytest.mark.smoke
def test_live_capture_labels_are_anchored_and_readable() -> None:
    """A capture's OWN labels must read through the gate during that capture.

    The direct converse of the belt: gating must not make live provenance
    invisible, which would ceil every honest model.
    """

    from torchlens.backends.torch._tl import get_tensor_label

    samples: list[tuple[Any, Any]] = []

    class _SelfReader(nn.Module):
        """Model that reads its own intermediate's label mid-capture."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Read the live intermediate's label through both views."""

            y = x * 2.0
            samples.append((get_tensor_label(y), raw_tensor_label(y)))
            return y + 1.0

    tl.trace(_SelfReader(), torch.ones(4), capture=_CAPTURE)

    gated, ungated = samples[0]
    assert ungated is not None, "the live op output must carry a label"
    assert gated == ungated, "a live capture's own label must read as provenance"


@pytest.mark.smoke
def test_meta_carries_the_session_anchor() -> None:
    """``TensorMeta`` must expose the anchor written alongside every label."""

    from torchlens.backends.torch import _tl

    probe = torch.ones(4)
    set_tensor_label(probe, "outside_1_1_raw")
    meta = get_tensor_meta(probe)
    assert meta is not None
    assert meta.label_raw == "outside_1_1_raw"
    assert meta.label_session is None, "a stamp outside any capture is unanchored"
    assert _tl.session_meta_is_anchored(meta) is False
