"""Running budget for retained activation bytes, with a typed refusal.

``tl.trace(model, x)`` retains every operation's output by default. That is the
right default for the models TorchLens was designed around and a footgun at
frontier shapes: a first-time user pointing the default at a large model gets an
OOM kill (or an allocator ``RuntimeError`` from somewhere deep inside torch)
rather than an explanation.

This module makes that failure more predictable without claiming a general OOM
guarantee. The primary retained copy is admitted from source-tensor bytes before
allocation, then alias-aware physical storage is reconciled after user transforms.
The model forward, cross-device temporaries, and transform-only deltas can allocate
before their size is knowable.

The accounting is deliberately a **lower bound, labelled as one**. At the moment
of the trip the forward is incomplete, so the true footprint of the finished
capture would have been larger; the error says exactly that instead of
extrapolating a total it cannot know. Pre-allocation refusals label their figure
as projected; post-transform refusals label committed storage.

Budgets are per-device because saved payloads follow the tensors they copy
(``output_device="same"`` by default), so a CUDA capture spends VRAM and a CPU
capture spends host RAM. Devices whose headroom cannot be measured warn on their
first non-empty automatic charge and remain unbudgeted unless the user supplies
an absolute limit.

Lookback / ``followed_by`` window copies are retained RAM like any other saved
activation and are charged through the same admit/reconcile pair (site
``"lookback_window"``); releasing a retained payload — window eviction, cleanup,
or any other drop of the last live reference — credits its storage back.
"""

from __future__ import annotations

import contextlib
import os
import warnings
import weakref
from dataclasses import dataclass, field
from typing import Any

import torch

from ._errors import InvalidArgumentError
from .errors._base import CaptureError

__all__ = [
    "ACCOUNTING_PHASES",
    "DEFAULT_SAVE_BUDGET_FRACTION",
    "SaveBudget",
    "SaveBudgetExceededError",
    "SaveBudgetOption",
    "available_device_bytes",
    "format_bytes",
    "resolve_save_budget",
]

SaveBudgetOption = str | int | float | None
"""Accepted ``save_budget`` spellings: ``"auto"``, a fraction, bytes, or ``None``."""

DEFAULT_SAVE_BUDGET_FRACTION = 0.5
"""Fraction of a device's *available* memory the ``"auto"`` budget allows.

Half of free memory, not of total: the traced model's own parameters and
activations already occupy the difference. Half leaves headroom for the forward
pass itself, which allocates live intermediates alongside every payload
TorchLens retains.
"""

_BYTES_UNITS = (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1))


class SaveBudgetExceededError(CaptureError, RuntimeError):
    """Raised when retained activation bytes cross the configured save budget.

    The structured accounting is retained on ``fields`` (``accounted_bytes``,
    ``committed_bytes`` or ``projected_bytes``, ``budget_bytes``, ``device``,
    ``num_saved``, ``label``, ``accounting_phase``) so callers branch on numbers
    rather than parsing the message.
    """


def format_bytes(num_bytes: float) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Parameters
    ----------
    num_bytes:
        Byte count.

    Returns
    -------
    str
        Human-readable size such as ``"1.44 GB"``.
    """

    for unit, scale in _BYTES_UNITS:
        if abs(num_bytes) >= scale or unit == "B":
            if unit == "B":
                return f"{int(num_bytes)} B"
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{int(num_bytes)} B"


def _available_host_bytes() -> int | None:
    """Return available host RAM in bytes, or ``None`` when unmeasurable.

    Returns
    -------
    int | None
        Available bytes. Prefers ``MemAvailable`` from ``/proc/meminfo`` because
        it accounts for reclaimable cache, then the POSIX available-pages count,
        then ``psutil`` when installed. macOS has neither procfs nor
        ``SC_AVPHYS_PAGES``, so without the ``psutil`` fallback every default
        ``save_budget="auto"`` capture there warned that CPU budgeting was off.
    """

    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except Exception:
        pass
    try:
        import psutil
    except ImportError:
        return None
    try:
        available = int(psutil.virtual_memory().available)
    except Exception:
        return None
    return available if available > 0 else None


def available_device_bytes(device: torch.device) -> int | None:
    """Return the memory headroom for one device, or ``None`` when unmeasurable.

    Parameters
    ----------
    device:
        Device whose free memory is queried.

    Returns
    -------
    int | None
        Free bytes, or ``None`` when this device type exposes no headroom query.
        ``None`` means *unbudgeted and reported as such*, never "assume
        infinite silently".
    """

    device_type = device.type
    if device_type == "cpu":
        return _available_host_bytes()
    if device_type == "cuda":
        try:
            # Pin the query to THIS device explicitly (R36-7b): mem_get_info
            # resolves an index-less device to the CURRENT device, so a
            # capture saving to cuda:1 while cuda:0 is current read the wrong
            # card's headroom.
            with torch.cuda.device(device):
                free_bytes, _total = torch.cuda.mem_get_info(device)
            return int(free_bytes)
        except Exception:
            return None
    if device_type == "meta":
        # Meta tensors have no storage, so nothing is ever really committed.
        return None
    return None


@dataclass(frozen=True)
class _BudgetSpec:
    """Resolved budget policy.

    Parameters
    ----------
    fraction:
        Fraction of a device's available memory to allow, when the policy is
        headroom-relative.
    absolute_bytes:
        Fixed per-device byte cap, when the policy is absolute.
    source:
        Human-readable description of where the policy came from, quoted in the
        refusal so the user can tell a default from their own setting.
    """

    fraction: float | None
    absolute_bytes: int | None
    source: str


def resolve_save_budget(value: SaveBudgetOption) -> _BudgetSpec | None:
    """Resolve a ``save_budget`` option value into a budget policy.

    Parameters
    ----------
    value:
        ``"auto"`` for the default headroom fraction, a float in ``(0, 1]`` for a
        custom fraction of available memory, an int (``>= 1``) for an absolute
        per-device byte cap, or ``None`` to disable budgeting.

    Returns
    -------
    _BudgetSpec | None
        Resolved policy, or ``None`` when budgeting is disabled.

    Raises
    ------
    InvalidArgumentError
        (``ValueError`` lineage, ``code="save_budget_invalid"``) if ``value``
        is not one of the documented spellings. Invalid budgets fail loudly
        rather than silently disabling the guard.
    """

    if value is None:
        return None
    if isinstance(value, str):
        if value != "auto":
            raise InvalidArgumentError(
                f"save_budget string must be 'auto'; got {value!r}. Use a float in (0, 1] for a "
                "fraction of available memory, an int for absolute bytes, or None to disable.",
                code="save_budget_invalid",
                remedy="pass 'auto', a float in (0, 1], an int byte cap, or None",
                argument="save_budget",
            )
        return _BudgetSpec(
            fraction=DEFAULT_SAVE_BUDGET_FRACTION,
            absolute_bytes=None,
            source=(
                f"default save_budget='auto' "
                f"({DEFAULT_SAVE_BUDGET_FRACTION:.0%} of available memory)"
            ),
        )
    if isinstance(value, bool):
        raise InvalidArgumentError(
            "save_budget does not accept bool; use None to disable or 'auto' for the default.",
            code="save_budget_invalid",
            remedy="pass 'auto', a float in (0, 1], an int byte cap, or None",
            argument="save_budget",
        )
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise InvalidArgumentError(
                f"save_budget float must be a fraction in (0, 1]; got {value!r}. "
                "Pass an int for an absolute byte cap.",
                code="save_budget_invalid",
                remedy="pass 'auto', a float in (0, 1], an int byte cap, or None",
                argument="save_budget",
            )
        return _BudgetSpec(
            fraction=value,
            absolute_bytes=None,
            source=f"save_budget={value!r} ({value:.0%} of available memory)",
        )
    if isinstance(value, int):
        if value < 1:
            raise InvalidArgumentError(
                f"save_budget int must be at least 1 byte; got {value!r}. Use None to disable.",
                code="save_budget_invalid",
                remedy="pass 'auto', a float in (0, 1], an int byte cap, or None",
                argument="save_budget",
            )
        return _BudgetSpec(
            fraction=None,
            absolute_bytes=value,
            source=f"save_budget={value} ({format_bytes(value)} per device)",
        )
    raise InvalidArgumentError(
        f"save_budget must be 'auto', a float in (0, 1], an int of bytes, or None; "
        f"got {type(value).__name__}.",
        code="save_budget_invalid",
        remedy="pass 'auto', a float in (0, 1], an int byte cap, or None",
        argument="save_budget",
    )


@dataclass
class _RetainedStorageEntry:
    """One charged physical storage and the count of live retained payloads on it."""

    physical_bytes: int
    live_refs: int = 0


@dataclass
class _DeviceLedger:
    """Per-device running total and resolved limit."""

    committed_bytes: int = 0
    num_saved: int = 0
    limit_bytes: int | None = None
    available_bytes: int | None = None
    measured: bool = False
    retained_storage: dict[tuple[Any, ...], _RetainedStorageEntry] = field(default_factory=dict)


_ADMISSION_SITES = ("primary", "lookback_window")
"""Closed vocabulary of admission sites; each maps to one admission/reconciliation phase pair."""

_SITE_PHASES = {
    "primary": ("pre_allocation_admission", "post_transform_reconciliation"),
    "lookback_window": ("lookback_window_admission", "lookback_window_reconciliation"),
}

_ADMISSION_PHASES = frozenset(phases[0] for phases in _SITE_PHASES.values())

ACCOUNTING_PHASES: tuple[str, ...] = tuple(
    phase for phases in _SITE_PHASES.values() for phase in phases
)
"""Closed vocabulary of ``accounting_phase`` values on ``SaveBudgetExceededError``.

Every refusal's ``fields["accounting_phase"]`` is one of these; callers branch on
this vocabulary, never on message text.
"""


@dataclass(frozen=True)
class _BudgetReservation:
    """One pre-allocation admission reserved against a device ledger."""

    label: str
    device: torch.device
    num_bytes: int
    site: str = "primary"


def _credit_payload_release(ref: _PayloadWatcher) -> None:
    """Credit one payload release back to its budget when the payload dies.

    Parameters
    ----------
    ref:
        Fired release watcher carrying its charge coordinates.
    """

    budget = ref.budget_ref()
    if budget is None:
        return
    budget._payload_watchers.pop(id(ref), None)
    budget._credit_release(ref.ledger_key, ref.identity)


class _PayloadWatcher(weakref.ref):
    """Release watcher on one retained payload, carrying its charge coordinates.

    A ``weakref.ref`` subclass so one allocation covers the watcher AND its
    coordinates: the closure-based watcher this replaces cost a function
    object, three cells, and a fresh budget weakref per retained payload
    (~7 marginal objects/op on the default capture path — the R32 regression).
    The callback is the module-level :func:`_credit_payload_release`; the
    budget is held weakly through the accountant's one shared self-ref so the
    accountant never keeps itself alive through its own watchers.
    """

    __slots__ = ("budget_ref", "ledger_key", "identity")

    budget_ref: weakref.ref
    ledger_key: str
    identity: tuple[Any, ...]

    def __new__(
        cls,
        payload: torch.Tensor,
        budget_ref: weakref.ref,
        ledger_key: str,
        identity: tuple[Any, ...],
    ) -> _PayloadWatcher:
        self = super().__new__(cls, payload, _credit_payload_release)
        self.budget_ref = budget_ref
        self.ledger_key = ledger_key
        self.identity = identity
        return self

    def __init__(
        self,
        payload: torch.Tensor,
        budget_ref: weakref.ref,
        ledger_key: str,
        identity: tuple[Any, ...],
    ) -> None:
        # weakref.ref implements __init__ (not just __new__) and would refuse
        # the extra coordinate arguments; forward only its own pair.
        super().__init__(payload, _credit_payload_release)  # type: ignore[call-arg]


@dataclass
class SaveBudget:
    """Per-device running accountant for retained activation bytes.

    Parameters
    ----------
    spec:
        Resolved budget policy.

    Notes
    -----
    ``admit`` and ``commit`` are on the capture hot path, once per retained
    payload. Each is a few dict lookups, integer adds, and one compare in the
    common case; device headroom is measured lazily on a device's first charge,
    so a capture that retains nothing pays nothing.
    """

    spec: _BudgetSpec
    ledgers: dict[str, _DeviceLedger] = field(default_factory=dict)
    # Keyed by id(watcher): weakref containers hash/compare through the live
    # referent, and tensor ``==`` is elementwise (and wrapped during capture).
    _payload_watchers: dict[int, weakref.ref] = field(
        default_factory=dict, repr=False, compare=False
    )
    # One shared weakref on self, minted lazily on the first watched payload;
    # every _PayloadWatcher holds this instead of a fresh per-payload ref.
    _self_ref: weakref.ref | None = field(default=None, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with the process-local release watchers stripped.

        Returns
        -------
        dict[str, Any]
            Instance state whose ``_payload_watchers`` map is empty and whose
            per-device ``retained_storage`` identity maps are empty.

        Notes
        -----
        Watchers are ``weakref.ref`` objects on live payload tensors: process-local
        by construction and unpicklable. A restored accountant keeps its committed
        charges permanently (the same conservative direction as a payload that
        cannot be weak-referenced); the source accountant's live watchers are
        untouched.

        The retained-storage identity maps ride with the watchers: an identity
        key without its release watcher is a DEAD key that a later allocation
        recycling the same ``data_ptr`` at equal size dedupes against for a
        ZERO-byte commit — the exact ptr-reuse bug the watchers fixed.
        ``Trace.fork()`` copies this accountant through this state (deepcopy
        rides ``__getstate__``) and forks DO capture again (``run()`` /
        ``save_new_outs`` refresh), so stripping one without the other re-opens
        the corridor. With both stripped, a restored/forked accountant
        re-charges fresh storage (conservative: at worst an alias double-count,
        never a zero-commit).
        """

        state = self.__dict__.copy()
        state["_payload_watchers"] = {}
        state["_self_ref"] = None
        state["ledgers"] = {
            key: _DeviceLedger(
                committed_bytes=ledger.committed_bytes,
                num_saved=ledger.num_saved,
                limit_bytes=ledger.limit_bytes,
                available_bytes=ledger.available_bytes,
                measured=ledger.measured,
                retained_storage={},
            )
            for key, ledger in self.ledgers.items()
        }
        return state

    @classmethod
    def from_option(cls, value: SaveBudgetOption) -> SaveBudget | None:
        """Build a budget from a user-facing option value.

        Parameters
        ----------
        value:
            ``save_budget`` option value.

        Returns
        -------
        SaveBudget | None
            Accountant, or ``None`` when budgeting is disabled.
        """

        spec = resolve_save_budget(value)
        if spec is None:
            return None
        return cls(spec=spec)

    def _ledger_for(self, device: torch.device) -> _DeviceLedger:
        """Return (creating if needed) the ledger for one device.

        Parameters
        ----------
        device:
            Device whose ledger is requested.

        Returns
        -------
        _DeviceLedger
            Ledger with its limit resolved on first use.
        """

        # Canonicalize the ledger identity (R36-7b): ``torch.device("cuda")``
        # and ``torch.device("cuda:0")`` are the SAME physical device, but
        # ``str()`` keyed them into two ledgers, each enforcing only half the
        # footprint against a full-device limit.
        if device.type == "cuda" and device.index is None:
            with contextlib.suppress(Exception):
                device = torch.device("cuda", torch.cuda.current_device())
        key = str(device)
        ledger = self.ledgers.get(key)
        if ledger is not None:
            return ledger
        ledger = _DeviceLedger()
        if self.spec.absolute_bytes is not None:
            ledger.limit_bytes = self.spec.absolute_bytes
            ledger.measured = True
        else:
            available = available_device_bytes(device)
            ledger.available_bytes = available
            ledger.measured = available is not None
            if available is not None:
                fraction = self.spec.fraction or DEFAULT_SAVE_BUDGET_FRACTION
                ledger.limit_bytes = int(available * fraction)
            else:
                warnings.warn(
                    "TorchLens cannot measure available memory for device "
                    f"{device}; automatic save budgeting is disabled on that device. "
                    "Use an absolute save_budget=<bytes> to enforce a ceiling there.",
                    UserWarning,
                    stacklevel=4,
                )
        self.ledgers[key] = ledger
        return ledger

    def admit(
        self,
        label: str,
        device: torch.device,
        num_bytes: int,
        *,
        site: str = "primary",
    ) -> _BudgetReservation | None:
        """Reserve a projected retained payload before its allocation.

        Parameters
        ----------
        label:
            Operation label used in a refusal.
        device:
            Projected retention device.
        num_bytes:
            Source-tensor bytes used as the pre-allocation estimate.
        site:
            Admission site from the closed vocabulary: ``"primary"`` for the
            per-op retained copy, ``"lookback_window"`` for a bounded
            retroactive-save window copy.

        Returns
        -------
        _BudgetReservation | None
            Reservation to reconcile after allocation, or ``None`` for an empty payload.

        Raises
        ------
        SaveBudgetExceededError
            If the projected footprint crosses the configured ceiling.
        """

        if site not in _SITE_PHASES:
            raise ValueError(
                f"save-budget admission site must be one of {_ADMISSION_SITES}; got {site!r}"
            )
        if num_bytes <= 0:
            return None
        ledger = self._ledger_for(device)
        ledger.committed_bytes += int(num_bytes)
        ledger.num_saved += 1
        try:
            self._raise_if_over_budget(label, device, ledger, phase=_SITE_PHASES[site][0])
        except BaseException:
            # Refusal rollback (r8 R34, opus A): the refused reservation used
            # to stay CHARGED -- never-allocated bytes inflated every later
            # figure (a second refusal reported phantom activations) and,
            # under return_partial/attach_partial recovery, a fresh save that
            # genuinely fit was refused against ghost bytes.
            ledger.committed_bytes -= int(num_bytes)
            ledger.num_saved -= 1
            raise
        return _BudgetReservation(label=label, device=device, num_bytes=int(num_bytes), site=site)

    def commit(
        self,
        reservation: _BudgetReservation | None,
        payloads: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Replace one estimate with alias-aware physical retained storage.

        Parameters
        ----------
        reservation:
            Admission returned by :meth:`admit`.
        payloads:
            RAM-retained raw and transformed payloads after saving.

        Notes
        -----
        A transform can allocate an output whose size or alias behavior is unknowable
        before user code runs. The source-sized reservation protects the first retained
        allocation; reconciliation then charges any additional transform storage. That
        transform-only delta is necessarily post-allocation and is disclosed publicly.

        Each committed payload is watched with a weakref: when the last retained
        payload on a physical storage is released, the charge is credited back and the
        storage-identity key is pruned, so a later allocation that recycles the same
        ``data_ptr`` at the same size is charged rather than deduplicated to zero.
        """

        if reservation is None:
            return
        reserved_ledger = self._ledger_for(reservation.device)
        reserved_ledger.committed_bytes -= reservation.num_bytes
        reserved_ledger.num_saved -= 1

        self._charge_physical(reservation.label, payloads, phase=_SITE_PHASES[reservation.site][1])

    def charge_retained(
        self,
        label: str,
        payloads: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Charge already-allocated retained payloads with no prior admission.

        Parameters
        ----------
        label:
            Operation label used in a refusal.
        payloads:
            RAM-retained tensors (non-tensors are skipped).

        Notes
        -----
        For retained copies whose size is only knowable after they exist —
        ``save_arg_values`` argument snapshots are the motivating case. Charges
        are alias-aware and release-credited exactly like :meth:`commit`; the
        refusal phase is the post-allocation reconciliation phase, matching the
        transform-delta disclosure.
        """

        self._charge_physical(label, payloads, phase=_SITE_PHASES["primary"][1])

    def _charge_physical(
        self,
        label: str,
        payloads: tuple[torch.Tensor | None, ...],
        *,
        phase: str,
    ) -> None:
        """Charge alias-aware physical storage for retained payloads.

        Parameters
        ----------
        label:
            Operation label used in a refusal.
        payloads:
            Retained payloads; non-tensors are skipped.
        phase:
            Accounting phase for a refusal raised here.
        """

        for payload in payloads:
            if not isinstance(payload, torch.Tensor) or payload.is_meta:
                # A meta tensor has no physical storage to charge (see the
                # is_meta short-circuit in _retained_storage_identities,
                # which makes the identity loop below a no-op for it
                # anyway); skip BEFORE ever opening its device's ledger, so
                # a meta-only payload never spuriously creates (and warns
                # on) an unmeasurable "meta" save-budget ledger.
                continue
            ledger_key = str(payload.device)
            ledger = self._ledger_for(payload.device)
            # Component-granular identities (r8 R34, sol 1): the historical
            # ONE aggregate identity per sparse wrapper made two retained
            # sparse tensors sharing index storage charge the shared indices
            # TWICE ((indices+values_a) + (indices+values_b) instead of the
            # physical union) -- a false refusal at an exact union-byte
            # boundary. Each physical component storage now dedups on its
            # own identity; ``num_saved`` still counts payloads, not
            # components.
            payload_counted = False
            for identity, physical_bytes in _retained_storage_identities(payload):
                entry = ledger.retained_storage.get(identity)
                if entry is not None:
                    entry.live_refs += 1
                    self._watch_payload(payload, ledger_key, identity)
                    continue
                ledger.retained_storage[identity] = _RetainedStorageEntry(
                    physical_bytes=physical_bytes, live_refs=1
                )
                self._watch_payload(payload, ledger_key, identity)
                ledger.committed_bytes += physical_bytes
                if not payload_counted:
                    ledger.num_saved += 1
                    payload_counted = True
                self._raise_if_over_budget(label, payload.device, ledger, phase=phase)

    def _watch_payload(
        self,
        payload: torch.Tensor,
        ledger_key: str,
        identity: tuple[Any, ...],
    ) -> None:
        """Arm a release watcher that credits this payload's storage when it dies.

        Parameters
        ----------
        payload:
            Retained tensor payload just committed against ``identity``.
        ledger_key:
            Ledger key of the device the payload was charged on.
        identity:
            Storage identity the payload holds a live reference on.

        Notes
        -----
        The watcher holds the budget weakly so the accountant never keeps itself
        alive through its own callbacks. A payload that cannot be weak-referenced
        keeps the historical permanent charge (conservative: never a zero-commit).
        """

        budget_ref = self._self_ref
        if budget_ref is None:
            budget_ref = self._self_ref = weakref.ref(self)
        try:
            watcher = _PayloadWatcher(payload, budget_ref, ledger_key, identity)
        except TypeError:
            return
        self._payload_watchers[id(watcher)] = watcher

    def _credit_release(self, ledger_key: str, identity: tuple[Any, ...]) -> None:
        """Credit one payload release; prune and refund on the last release.

        Parameters
        ----------
        ledger_key:
            Ledger key of the device the storage was charged on.
        identity:
            Storage identity whose live reference count drops by one.
        """

        ledger = self.ledgers.get(ledger_key)
        if ledger is None:
            return
        entry = ledger.retained_storage.get(identity)
        if entry is None:
            return
        entry.live_refs -= 1
        if entry.live_refs > 0:
            return
        del ledger.retained_storage[identity]
        ledger.committed_bytes -= entry.physical_bytes
        ledger.num_saved -= 1

    def _raise_if_over_budget(
        self,
        label: str,
        device: torch.device,
        ledger: _DeviceLedger,
        *,
        phase: str,
    ) -> None:
        """Raise when one ledger exceeds its resolved limit.

        Parameters
        ----------
        label:
            Operation label used in the refusal.
        device:
            Device whose ledger is checked.
        ledger:
            Updated per-device ledger.
        phase:
            Accounting phase exposed in structured error fields.
        """

        limit = ledger.limit_bytes
        if limit is None or ledger.committed_bytes <= limit:
            return
        raise SaveBudgetExceededError(
            self._message(label, device, ledger, phase=phase),
            accounted_bytes=ledger.committed_bytes,
            committed_bytes=(None if phase in _ADMISSION_PHASES else ledger.committed_bytes),
            projected_bytes=(ledger.committed_bytes if phase in _ADMISSION_PHASES else None),
            budget_bytes=limit,
            available_bytes=ledger.available_bytes,
            device=str(device),
            num_saved=ledger.num_saved,
            label=label,
            accounting_phase=phase,
        )

    def _message(
        self,
        label: str,
        device: torch.device,
        ledger: _DeviceLedger,
        *,
        phase: str,
    ) -> str:
        """Build the refusal message.

        Parameters
        ----------
        label:
            Tripping layer label.
        device:
            Device whose budget was crossed.
        ledger:
            Ledger at the moment of the trip.
        phase:
            Accounting phase that detected the crossing.

        Returns
        -------
        str
            Explanatory message naming the committed footprint and the remedies.
        """

        limit = ledger.limit_bytes or 0
        mean_bytes = ledger.committed_bytes / max(ledger.num_saved, 1)
        headroom = (
            f" (device had {format_bytes(ledger.available_bytes)} available at first save)"
            if ledger.available_bytes is not None
            else ""
        )
        footprint_label = (
            "projected retained footprint (refused before the crossing allocation)"
            if phase in _ADMISSION_PHASES
            else "committed so far"
        )
        lookback_remedy = (
            "    - retain fewer window payloads: shrink lookback=<N> or keep "
            "lookback_payload_policy='metadata_only' so candidates hold no payloads\n"
            if phase.startswith("lookback_window")
            else ""
        )
        return (
            "torchlens stopped capture: retained activations crossed the save budget on "
            f"{device}.\n"
            f"  {footprint_label}: {format_bytes(ledger.committed_bytes)} across "
            f"{ledger.num_saved} saved activation(s), mean {format_bytes(mean_bytes)} each\n"
            f"  budget: {format_bytes(limit)} from {self.spec.source}{headroom}\n"
            f"  tripped while saving: {label}\n"
            "  This is a LOWER BOUND on the retained footprint at this point in the "
            "incomplete forward, not an extrapolated completed-capture total.\n"
            "  Remedies, cheapest first:\n"
            f"{lookback_remedy}"
            "    - save less: save=tl.func('relu') or save=tl.in_module('encoder') "
            "instead of saving every activation\n"
            "    - save less AND stream those selected payloads to disk: "
            "save=tl.func('relu'), storage=tl.to_disk('run.tlspec')\n"
            "    - keep metadata only: "
            "capture=tl.options.CaptureOptions(layers_to_save='none') "
            "(the graph is still captured)\n"
            "    - raise or lift the budget deliberately: "
            "capture=tl.options.CaptureOptions(save_budget=<bytes|fraction>), or "
            "save_budget=None to disable it"
        )


def _retained_storage_identities(tensor: torch.Tensor) -> list[tuple[tuple[Any, ...], int]]:
    """Return one device-scoped physical identity + byte size per storage.

    Parameters
    ----------
    tensor:
        Retained tensor payload.

    Returns
    -------
    list[tuple[tuple[Any, ...], int]]
        One ``(identity, physical_bytes)`` entry per physical storage the
        payload holds. Strided tensors have exactly one; sparse payloads
        (COO and compressed layouts) have no top-level storage and yield one
        entry PER COMPONENT (index tensor(s) AND values), so partially
        overlapping sparse tensors dedup on the shared component instead of
        double-charging it under one aggregate identity (r8 R34, sol 1).
        The historical fallback billed unreadable payloads at
        ``numel() * element_size()`` — LOGICAL dense bytes — under an
        id-based identity. A meta-device tensor has no physical storage at
        all and always yields an empty list (see the ``is_meta`` note below).
    """

    # A meta tensor carries no real bytes, ever -- ``retained_activation_bytes``
    # already skips meta payloads explicitly for this reason. Without this
    # early return, ``tensor.untyped_storage()`` is inconsistent for meta
    # tensors across the torch>=2.1 floor: on torch 2.1/2.2 it either raises
    # or hands back a storage whose ``nbytes()``/``data_ptr()`` are not
    # reliably zero, so the ``except Exception`` fallback below bills the
    # tensor's LOGICAL ``numel() * element_size()`` bytes against a device
    # that never actually committed any memory -- a save-budget admission for
    # "meta" that should never exist (observed: a save_arg_values capture
    # over a meta-device argument spuriously creates and warns on a "meta"
    # device ledger). Checking this up front is correct on every torch
    # version, not a version-gated shim.
    if tensor.is_meta:
        return []

    from ._state import pause_logging

    with pause_logging():
        try:
            if tensor.layout is not torch.strided:
                from .utils.tensor_utils import sparse_component_tensors

                device = str(tensor.device)
                return [
                    (
                        (
                            device,
                            int(component.untyped_storage().data_ptr()),
                            int(component.untyped_storage().nbytes()),
                        ),
                        int(component.untyped_storage().nbytes()),
                    )
                    for component in sparse_component_tensors(tensor)
                ]
            storage = tensor.untyped_storage()
            num_bytes = int(storage.nbytes())
            identity: tuple[Any, ...] = (
                str(tensor.device),
                int(storage.data_ptr()),
                num_bytes,
            )
            return [(identity, num_bytes)]
        except Exception:
            num_bytes = int(tensor.numel() * tensor.element_size())
            return [((str(tensor.device), "tensor", id(tensor), num_bytes), num_bytes)]


def _retained_storage_identity(tensor: torch.Tensor) -> tuple[tuple[Any, ...], int]:
    """Return one aggregate identity + total bytes (diagnostic compat shim).

    The charging path uses :func:`_retained_storage_identities`
    (component-granular); this aggregate view sums the components and keeps
    the historical single-identity shape for introspection/tests.
    """

    entries = _retained_storage_identities(tensor)
    if len(entries) == 1:
        return entries[0]
    total_bytes = sum(num_bytes for _, num_bytes in entries)
    aggregate: tuple[Any, ...] = (
        str(tensor.device),
        str(tensor.layout),
        tuple(identity for identity, _ in entries),
        total_bytes,
    )
    return aggregate, total_bytes


def retained_activation_bytes(ops: Any) -> int:
    """Return alias-aware physical bytes of the activations actually retained.

    The ONE byte model's aggregate read (F20, brainpipe memo D-17): a saved
    op contributes the physical storage of what capture actually kept -- the
    raw payload, the transformed payload, or both -- with each physical
    storage counted ONCE across the whole trace (dedup-shared and aliased
    payloads never double-charge). The historical aggregate summed the RAW
    ``activation_memory`` field for every saved op, over-reporting by the
    full reduction factor whenever a transform dropped the raw payload
    (87-190x measured on real models in the sweep's configuration).

    Runs on LIVE payload tensors only (the capture-time accounting seam);
    a payload-absent saved op (disk-only routes) falls back to its recorded
    transformed-else-raw metadata size, without alias awareness.

    Parameters
    ----------
    ops:
        Iterable of finished op records.

    Returns
    -------
    int
        Total retained activation bytes.
    """

    total = 0
    seen: set[tuple[Any, ...]] = set()
    for op in ops:
        if not getattr(op, "has_saved_activation", False) or getattr(op, "is_orphan", False):
            continue
        counted_payload = False
        for payload in (getattr(op, "out", None), getattr(op, "transformed_out", None)):
            if not isinstance(payload, torch.Tensor) or payload.is_meta:
                continue
            counted_payload = True
            for identity, num_bytes in _retained_storage_identities(payload):
                if identity in seen:
                    continue
                seen.add(identity)
                total += int(num_bytes)
        if not counted_payload:
            transformed_bytes = getattr(op, "transformed_activation_memory", None)
            if getattr(op, "out", None) is None and transformed_bytes:
                total += int(transformed_bytes)
            else:
                total += int(getattr(op, "activation_memory", 0) or 0)
    return total
