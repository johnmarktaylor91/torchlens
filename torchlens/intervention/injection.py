"""``log_injections`` stages 0-2: anchored injected-op recording (F01/F44).

The capture option ``CaptureOptions(log_injections=True)`` makes computation
performed INSIDE an intervention hook first-class: each torch call the hook
executes is recorded as an :class:`InjectedOp` -- it ran (fact, not
fabrication) -- carrying provenance "from intervention X, not the model".

STAGE 0 PREREQUISITES (both anchored-identity layers):

* Injected ops consume NO global label counters and NO site-key cohort
  ordinals -- BY CONSTRUCTION: the recorder is a session-scoped
  ``TorchFunctionMode`` installed inside the hook's ``pause_logging()``
  window, so the main capture journal and the live site-key minter never
  see these calls (the misfire test measures both halves).
* Each injected op gets the durable structural key
  ``(host_site_key, spec_rule_id, host_pass, firing_index, nesting_path,
  local_op_ordinal, output_slot)`` -- VERBATIM the C07 entry-dark
  ``Op.injection_provenance`` slot grammar (tlspec v9), which stage 2
  (lane F44) persists against with no further schema bump -- plus a
  human-readable label anchored to the host site
  (``relu_5_23/inj_1``; rendering is display sugar for the naming session).

STAGE 1 (in-memory): capture + immutable provenance + the query split
(``trace.model_ops`` / ``trace.injected_ops``). Injected ops cluster under
their intervention, never inside the model's module hierarchy; live
selectors never fire on them (they are outside the op stream by
construction); the ordinary validation tripwire passes UNCHANGED on a
logged in-memory trace (release gate, no new exemption, ever). Logged
injections during REPLAY stay deferred: no capture session exists during
replay, so nothing executed inside a replay hook can be recorded without
standing one up.

STAGE 2 (lane F44, persistence + attestation): analysis-level saves
persist the injected family through :mod:`torchlens._io.injection_codec`
(synthesized op rows writing the C07 ``Op.injection_provenance`` slot;
callable identity through the trusted-callable resolver; fire-time arg
snapshots for replay). Loads degrade to ``attestation="unattested"`` --
never refusing on an unresolvable callable, always refusing on forged
provenance -- and :func:`attest_injected_ops` is the validation door that
replays trusted injected callables against their recorded outputs.
Runnable saves refuse typed (the sparse core cannot carry the family).

Every spelling DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "InjectedOp",
    "InjectionAnchor",
    "InjectionAttestation",
    "InjectionAttestationRow",
    "InjectionProvenance",
    "attest_injected_ops",
    "injected_ops",
    "injection_recorder",
    "injection_state",
    "refuse_injection_logged_runnable_save",
    "refuse_injection_logged_stream_finalize",
]


@dataclass(frozen=True)
class InjectionProvenance:
    """The durable structural key of one injected op output (C07 slot grammar).

    Field-for-field the tlspec v9 entry-dark ``Op.injection_provenance``
    record (surgery memo 3.5 items 8-9, 12): stage 2 (lane F44) persists
    THIS shape with no further schema bump.
    """

    host_site_key: str | None
    spec_rule_id: str
    host_pass: int
    firing_index: int
    nesting_path: tuple[int, ...]
    local_op_ordinal: int
    output_slot: int


@dataclass(frozen=True)
class InjectedOp:
    """One recorded injected op output: it RAN, from an intervention.

    Parameters
    ----------
    label:
        Human-readable label anchored to the host site
        (``<host_label>/inj_<ordinal>``; multi-output slots append
        ``:<slot>``). Display sugar over the durable ``provenance`` key.
    host_label:
        The host op's (pass-qualified where applicable) label at fire time.
    func_name:
        The torch function the hook executed.
    layer_type:
        Normalized function name (the capture lane's layer-type spelling).
    out:
        The injected op's output tensor (detached snapshot; injected ops
        never enter the model's autograd or dataflow graph families).
    provenance:
        The durable structural key (:class:`InjectionProvenance`).
    callable_ref:
        The executed function's portable identity
        (:class:`~torchlens.intervention.types.FunctionRegistryKey`),
        minted by the trusted-callable resolver's canonical encoder at fire
        time and re-resolved through the same trust gate at replay/
        attestation time. ``None`` when the fired callable exposes no
        registrable identity.
    saved_args:
        Detached fire-time snapshots of the call's positional args, or
        ``None`` when any arg was unsnapshotable (the record is then
        replay-ineligible, disclosed at attestation).
    saved_kwargs:
        Detached fire-time snapshots of the call's keyword args as an
        immutable ``(name, value)`` pair tuple; ``None`` mirrors
        ``saved_args``.
    attestation:
        Evidence basis of the ``out`` claim: ``"recorded"`` (live capture,
        this session observed the call), ``"unattested"`` (loaded from an
        artifact, claim not yet corroborated -- the stage-2 degrade-to-
        unattested default), or ``"attested"`` (a trusted replay reproduced
        the recorded output exactly).
    attestation_reason:
        Machine-readable reason qualifying a non-``recorded`` status.
    """

    label: str
    host_label: str
    func_name: str
    layer_type: str
    out: Any
    provenance: InjectionProvenance
    callable_ref: Any = None
    saved_args: tuple[Any, ...] | None = None
    saved_kwargs: tuple[tuple[str, Any], ...] | None = None
    attestation: str = "recorded"
    attestation_reason: str | None = None
    fire_device: str | None = None


@dataclass(frozen=True)
class InjectionAnchor:
    """The host identity ONE hook firing anchors its injected ops to."""

    host_label: str
    host_site_key: str | None
    host_pass: int
    spec_rule_id: str
    firing_index: int


class injection_recorder:
    """Session-scoped recorder for the torch calls ONE hook firing executes.

    Installed by ``_execute_hook`` inside the hook's ``pause_logging()``
    window when the active capture armed ``log_injections=True``. The main
    capture journal is paused, so recording here consumes no global label
    counter and no site-key cohort ordinal; every intercepted call mints
    injection-local ordinals under the anchored host identity.
    """

    def __init__(self, trace: Any, anchor: InjectionAnchor) -> None:
        """Stage one recorder for one hook firing (nothing installed yet)."""

        self.trace = trace
        self.anchor = anchor
        self.local_ordinal = 0
        self._mode: Any = None

    def __enter__(self) -> injection_recorder:
        """Install the torch-function interceptor for this hook window."""

        import torch
        from torch.overrides import TorchFunctionMode

        from ..capture.arg_positions import _normalize_func_name
        from .binding import _has_tensor, _wrap_name_universe

        recorder = self
        wrap_universe = _wrap_name_universe()

        class _Mode(TorchFunctionMode):
            """Interceptor minting injected-op records for one firing."""

            def __torch_function__(
                self,
                func: Any,
                types: Any,
                args: tuple[Any, ...] = (),
                kwargs: dict[str, Any] | None = None,
            ) -> Any:
                """Snapshot the inputs, run the call, record its tensor outputs."""

                kwargs = kwargs or {}
                name = getattr(func, "__name__", None)
                if not name:
                    return func(*args, **kwargs)
                layer_type = _normalize_func_name(name)
                if layer_type not in wrap_universe:
                    return func(*args, **kwargs)
                # The replay evidence is snapshotted BEFORE the call runs: an
                # in-place callable (mul_, relu_, copy_, F.relu(inplace=True))
                # rewrites its own argument storage, so a post-call snapshot
                # would be the OUTPUT and the attestation replay would compute
                # f(f(x)) and call an honest capture forged (AUD-CODE 2.3a).
                snapshot = _snapshot_call_inputs(args, kwargs, torch.Tensor)
                out = func(*args, **kwargs)
                if not _has_tensor(out, torch.Tensor):
                    return out
                recorder._record_call((layer_type, name, func), snapshot, out, torch.Tensor)
                return out

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Remove the interceptor (exception cleanup included)."""

        if self._mode is not None:
            self._mode.__exit__(None, None, None)
            self._mode = None

    def _record_call(
        self,
        call: tuple[str, str, Any],
        snapshot: tuple[tuple[Any, ...] | None, tuple[tuple[str, Any], ...] | None],
        out: Any,
        tensor_type: type,
    ) -> None:
        """Mint one InjectedOp per tensor output slot of one intercepted call.

        ``call`` is the intercepted identity triple
        ``(layer_type, func_name, func)``; ``snapshot`` is the PRE-CALL
        ``(saved_args, saved_kwargs)`` pair from :func:`_snapshot_call_inputs`.
        """

        layer_type, func_name, func = call
        self.local_ordinal += 1
        ordinal = self.local_ordinal
        outputs: list[tuple[int, Any]]
        if isinstance(out, tensor_type):
            outputs = [(0, out)]
        elif isinstance(out, (list, tuple)):
            outputs = [
                (slot, item) for slot, item in enumerate(out) if isinstance(item, tensor_type)
            ]
        else:
            outputs = []
        from .resolver import function_registry_key_from_callable

        try:
            callable_ref = function_registry_key_from_callable(func)
        except (AttributeError, KeyError, TypeError, ValueError):
            callable_ref = None
        saved_args, saved_kwargs = snapshot
        anchor = self.anchor
        records = _injection_store(self.trace)
        for slot, value in outputs:
            suffix = f"/inj_{ordinal}" + (f":{slot}" if slot else "")
            records.append(
                InjectedOp(
                    label=f"{anchor.host_label}{suffix}",
                    host_label=anchor.host_label,
                    func_name=func_name,
                    layer_type=layer_type,
                    # detach().clone(): a bare detach aliases live storage,
                    # and a downstream in-place op (ResNet's relu_) would
                    # silently rewrite the recorded evidence -- caught by the
                    # stage-2 replay tripwire on the torchvision gate.
                    out=value.detach().clone(),
                    provenance=InjectionProvenance(
                        host_site_key=anchor.host_site_key,
                        spec_rule_id=anchor.spec_rule_id,
                        host_pass=anchor.host_pass,
                        firing_index=anchor.firing_index,
                        nesting_path=(),
                        local_op_ordinal=ordinal,
                        output_slot=slot,
                    ),
                    callable_ref=callable_ref,
                    saved_args=saved_args,
                    saved_kwargs=saved_kwargs,
                    fire_device=str(value.device),
                )
            )


def _snapshot_call_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any], tensor_type: type[Any]
) -> tuple[tuple[Any, ...] | None, tuple[tuple[str, Any], ...] | None]:
    """Snapshot one call's positional AND keyword inputs before it runs.

    Returns ``(saved_args, saved_kwargs)``; any unsnapshotable value on
    either side makes the WHOLE call replay-ineligible (``(None, None)``).
    """

    saved_args = _snapshot_call_values(args, tensor_type)
    if saved_args is None:
        return None, None
    snapped_kwarg_values = _snapshot_call_values(tuple(kwargs.values()), tensor_type)
    if snapped_kwarg_values is None:
        return None, None
    return saved_args, tuple(zip(kwargs.keys(), snapped_kwarg_values, strict=True))


def _snapshot_call_values(
    values: tuple[Any, ...], tensor_type: type[Any]
) -> tuple[Any, ...] | None:
    """Snapshot one intercepted call's argument values for later replay.

    Tensors become detached clones (a plain ``detach()`` would alias live
    storage a later in-place edit could rewrite under the record); the
    closed literal set and containers of snapshotable values copy
    structurally. ANY unsnapshotable value makes the WHOLE call
    replay-ineligible (``None``), never a silently partial arg tuple.
    """

    snapped: list[Any] = []
    for value in values:
        ok, snap = _snapshot_one_value(value, tensor_type)
        if not ok:
            return None
        snapped.append(snap)
    return tuple(snapped)


def _snapshot_one_value(value: Any, tensor_type: type[Any]) -> tuple[bool, Any]:
    """Snapshot one argument value; ``(False, None)`` when unsnapshotable."""

    import torch

    if isinstance(value, tensor_type):
        return True, value.detach().clone()
    literal_types = (bool, int, float, str, bytes, torch.dtype, torch.device)
    if value is None or isinstance(value, literal_types):
        return True, value
    if isinstance(value, (list, tuple)):
        return _snapshot_sequence(value, tensor_type)
    if isinstance(value, dict):
        return _snapshot_mapping(value, tensor_type)
    return False, None


def _snapshot_sequence(value: Any, tensor_type: type[Any]) -> tuple[bool, Any]:
    """Snapshot one list/tuple argument, preserving its container type."""

    items = []
    for item in value:
        ok, snap = _snapshot_one_value(item, tensor_type)
        if not ok:
            return False, None
        items.append(snap)
    return True, (tuple(items) if isinstance(value, tuple) else items)


def _snapshot_mapping(value: dict[Any, Any], tensor_type: type[Any]) -> tuple[bool, Any]:
    """Snapshot one string-keyed dict argument."""

    out: dict[str, Any] = {}
    for key, item in value.items():
        ok, snap = _snapshot_one_value(item, tensor_type) if isinstance(key, str) else (False, None)
        if not ok:
            return False, None
        out[key] = snap
    return True, out


def peek_injection_state(trace: Any) -> dict[str, Any] | None:
    """Read the trace's ``_tl_injection_state`` seam without arming it.

    ``_tl_injection_state`` is a declared ``FieldPolicy.DROP`` Trace field
    (``data_classes/_trace_components.py``) that is created lazily, on
    first use, by :func:`injection_state`; a trace that was never armed for
    injection simply never got the attribute. Read-only callers must
    tolerate that absence without materializing it (that mutation is
    reserved for :func:`injection_state`/:func:`arm_injection_logging`), so
    this is a direct private read guarded by ``AttributeError`` rather than
    a string-literal ``getattr`` default.
    """

    try:
        return trace._tl_injection_state
    except AttributeError:
        return None


def injection_state(trace: Any) -> dict[str, Any]:
    """The trace's ONE consolidated session-transient injection state.

    One declared ``FieldPolicy.DROP`` Trace field
    (``_tl_injection_state``) instead of five loose transients (the
    session-component consolidation pressure): ``armed`` (the capture
    knob), ``counters`` (per-rule firing indexes), ``current_rule`` (the
    predicate-door anchor), ``records`` (the injected-op ledger), and
    ``resolved`` (the post-capture anchoring marker).
    """

    state = peek_injection_state(trace)
    if state is None:
        state = {
            "armed": False,
            "counters": {},
            "current_rule": None,
            "records": [],
            "resolved": False,
        }
        trace._tl_injection_state = state
    return state


def arm_injection_logging(trace: Any, *, armed: bool) -> None:
    """Arm (or explicitly disarm) injected-op recording for one capture.

    The capture entry's one-line arming spelling (F01 stages 0-1): the ONE
    consolidated session-transient state dict (a declared ``FieldPolicy.DROP``
    Trace field) is created here at entry, and ``armed`` mirrors the
    ``log_injections`` capture knob. Injected records never persist at
    stage 1 -- the save door refuses typed, naming lane F44.
    """

    injection_state(trace)["armed"] = armed


def _injection_store(trace: Any) -> list[InjectedOp]:
    """The trace's session-transient injected-op ledger."""

    return injection_state(trace)["records"]


def injected_ops(trace: Any) -> tuple[InjectedOp, ...]:
    """Return the trace's injected-op records (empty when never armed).

    Records mint at FIRE time against the host's raw capture label (the
    only identity that exists mid-forward); this accessor RESOLVES each
    record once against the finished trace -- final host label, host
    site key, host pass -- and caches the anchored records back. The
    durable provenance ordinals (rule, firing, local op, slot) never
    change at resolution.
    """

    state = peek_injection_state(trace) or {}
    if state.get("pending_loaded_rows"):
        # Internal invariant: rehydrate_trace finalizes every split row; a
        # pending row here means a load path bypassed the finalize seam.
        raise RuntimeError(
            "loaded injected-op rows were split but never finalized; this "
            "artifact was loaded through a path that skipped rehydrate_trace"
        )
    records = state.get("records") or ()
    if not records:
        return ()
    if state.get("resolved", False):
        return tuple(records)
    import dataclasses

    ops = tuple(getattr(trace, "layer_list", ()) or ())
    by_raw: dict[str, Any] = {}
    for op in ops:
        raw = getattr(op, "raw_label", None)
        if isinstance(raw, str):
            by_raw.setdefault(raw, op)
    resolved: list[InjectedOp] = []
    for record in records:
        # live fire-time labels carry the transient "_raw" suffix the
        # finished records drop
        host = by_raw.get(record.host_label) or by_raw.get(record.host_label.removesuffix("_raw"))
        if host is None:
            host = _module_boundary_host(ops, record.host_label)
        if host is None:
            resolved.append(record)
            continue
        suffix = record.label[len(record.host_label) :]
        resolved.append(
            dataclasses.replace(
                record,
                label=f"{host.label}{suffix}",
                host_label=host.label,
                provenance=dataclasses.replace(
                    record.provenance,
                    host_site_key=getattr(host, "site_key", None),
                    host_pass=int(getattr(host, "pass_index", None) or record.provenance.host_pass),
                ),
            )
        )
    state["records"] = resolved
    state["resolved"] = True
    return tuple(resolved)


def _module_boundary_host(ops: tuple[Any, ...], host_label: str) -> Any | None:
    """Resolve a module-boundary firing's host op from its ``address:pass`` label.

    A ``tl.module(...)`` rule fires at a real module hook, where the only
    identity that exists is the module call label (``fc1:1``); no op carries
    that raw label. The host is the retained op that PRODUCED that module
    call's output: the ``interventionreplacement`` op the capture mints at
    module exit when the hook replaced the value (preferred -- it carries
    ``intervention_replaced=True``), else the module's own output op. Both
    list the module call in ``output_of_module_calls``. ``None`` when no
    retained op exits that module call (the record stays unanchored and the
    codec refuses to persist it typed).
    """

    fallback: Any | None = None
    for op in ops:
        exits = getattr(op, "output_of_module_calls", None) or ()
        if host_label not in exits:
            continue
        if getattr(op, "intervention_replaced", False):
            return op
        if fallback is None:
            fallback = op
    return fallback


def next_firing_index(trace: Any, spec_rule_id: str) -> int:
    """Advance and return the per-rule firing counter for one hook fire."""

    counters = injection_state(trace)["counters"]
    counters[spec_rule_id] = counters.get(spec_rule_id, 0) + 1
    return counters[spec_rule_id]


def refuse_injection_logged_runnable_save(trace: Any, save_level: str) -> None:
    """Refuse a RUNNABLE save of a trace carrying logged injected ops (typed).

    Analysis-level persistence is lane F44's shipped codec (the injected
    family rides the artifact as op rows writing the C07
    ``Op.injection_provenance`` slot). The runnable product is different:
    its sparse core is the model's taken-path DAG, injected computation is
    NOT on that path, and a runnable artifact silently dropping it would
    present an edited capture as a plain replayable one -- the exact
    stage-1 honesty rule, now narrowed to the one save level that cannot
    carry the family.

    Raises
    ------
    InvalidArgumentError
        ``injection_logged_runnable_unsupported``.
    """

    if save_level != "runnable":
        return
    state = peek_injection_state(trace) or {}
    records = state.get("records") or ()
    if not records:
        return
    from .._errors import InvalidArgumentError

    raise InvalidArgumentError(
        f"this trace carries {len(records)} logged injected op(s) "
        "(log_injections=True); the runnable sparse core is the model's "
        "taken-path DAG and cannot carry injected computation, so a "
        "runnable artifact would silently drop it",
        code="injection_logged_runnable_unsupported",
        remedy="save at the default analysis level (the injected family "
        "persists there), or re-capture without log_injections for a "
        "runnable artifact",
        argument="trace",
    )


def refuse_injection_logged_stream_finalize(trace: Any) -> None:
    """Refuse streamed-bundle finalize on a trace carrying injected ops (typed).

    The streamed ``to_disk`` writer scrubs and stages its own bundle without
    passing through the analysis save door that appends the injected family
    (:mod:`torchlens._io.injection_codec`), so a streamed artifact would
    silently drop recorded computation -- the stage-1 honesty rule. Today
    this door is defense-in-depth: producing injected records requires
    ``intervene=``, and streamed intervened captures fail earlier on the
    pre-existing fire-counter portability gap; the refusal arms the seam
    for the day that gap is fixed.

    Raises
    ------
    InvalidArgumentError
        ``injection_logged_stream_unsupported``.
    """

    state = peek_injection_state(trace) or {}
    records = state.get("records") or ()
    if not records:
        return
    from .._errors import InvalidArgumentError

    raise InvalidArgumentError(
        f"this trace carries {len(records)} logged injected op(s) "
        "(log_injections=True); the streamed to_disk writer does not carry "
        "the injected family, so finalizing would silently drop recorded "
        "computation",
        code="injection_logged_stream_unsupported",
        remedy="capture without storage=tl.to_disk(...) and save the "
        "finished trace with tl.save() (the analysis save persists the "
        "injected family), or re-capture without log_injections",
        argument="trace",
    )


_NONDETERMINISTIC_LAYER_TYPES = frozenset(
    {
        "alpha_dropout",
        "bernoulli",
        "cauchy",
        "dropout",
        "dropout2d",
        "dropout3d",
        "exponential",
        "feature_alpha_dropout",
        "geometric",
        "log_normal",
        "multinomial",
        "normal",
        "poisson",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "rrelu",
        "uniform",
    }
)
"""Normalized layer types whose replay legitimately consumes fresh RNG.

A replay of one of these can never corroborate the recorded output, so
attestation reports ``unattested(nondeterministic_callable)`` instead of a
false ``diverged`` verdict on an honest capture.
"""


@dataclass(frozen=True)
class InjectionAttestationRow:
    """One injected-op replay verdict.

    ``status`` is closed: ``"attested"`` (trusted replay reproduced the
    recorded output exactly), ``"unattested"`` (replay could not run --
    the reason discloses why), or ``"diverged"`` (a trusted replay RAN and
    produced a DIFFERENT value: the tripwire verdict; the report fails).
    """

    label: str
    status: str
    reason: str | None


@dataclass(frozen=True)
class InjectionAttestation:
    """The injected-op replay report for one trace.

    ``passed`` is False exactly when at least one row is ``diverged`` --
    a corroborated mismatch between an injected op's recorded output and
    its trusted replay. ``unattested`` rows never fail the report: they
    are disclosed inability to check, not evidence of forgery.
    """

    rows: tuple[InjectionAttestationRow, ...]
    passed: bool

    @property
    def attested(self) -> tuple[InjectionAttestationRow, ...]:
        """Rows whose trusted replay reproduced the recorded output."""

        return tuple(row for row in self.rows if row.status == "attested")

    @property
    def unattested(self) -> tuple[InjectionAttestationRow, ...]:
        """Rows the replay could not corroborate (reason disclosed)."""

        return tuple(row for row in self.rows if row.status == "unattested")

    @property
    def diverged(self) -> tuple[InjectionAttestationRow, ...]:
        """Rows whose trusted replay contradicted the recorded output."""

        return tuple(row for row in self.rows if row.status == "diverged")


def attest_injected_ops(
    trace: Any,
    *,
    trust_custom_callables: bool = False,
    allowed_custom_callable_modules: Any = None,
) -> InjectionAttestation:
    """Replay trusted injected callables against their recorded outputs.

    The stage-2 validation door (foldA s5 item 12): every injected-op
    record whose callable resolves through the trusted-callable resolver
    AND whose fire-time args were snapshotable is re-executed on those
    exact args, and the produced output is compared byte-exactly against
    the recorded one. Foreign callables stay unresolved (and unexecuted)
    unless explicitly trusted -- they report ``unattested
    (callable_untrusted)``; a resolvable-namespace callable that no longer
    imports reports ``unattested(callable_missing)``. A mismatch is the
    ``diverged`` tripwire verdict and fails the report; matched rows are
    promoted to ``attestation="attested"`` on the stored records.

    Ordinary validation (forward replay, metadata invariants) is UNCHANGED
    by this door -- it is additive, invoked explicitly.
    """

    state = injection_state(trace)
    records = injected_ops(trace)
    rows: list[InjectionAttestationRow] = []
    updated: list[InjectedOp] = []
    passed = True
    for record in records:
        status, reason = _attest_one_record(
            record,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
        rows.append(InjectionAttestationRow(label=record.label, status=status, reason=reason))
        if status == "diverged":
            passed = False
        if status == "attested" and record.attestation != "attested":
            import dataclasses

            record = dataclasses.replace(record, attestation="attested", attestation_reason=None)
        updated.append(record)
    state["records"] = updated
    return InjectionAttestation(rows=tuple(rows), passed=passed)


def _attest_one_record(
    record: InjectedOp,
    *,
    trust_custom_callables: bool,
    allowed_custom_callable_modules: Any,
) -> tuple[str, str | None]:
    """Return one record's ``(status, reason)`` replay verdict."""

    import torch

    precondition = _attest_precondition_reason(record, torch)
    if precondition is not None:
        return "unattested", precondition
    func, resolve_reason = _resolve_record_callable(
        record,
        trust_custom_callables=trust_custom_callables,
        allowed_custom_callable_modules=allowed_custom_callable_modules,
    )
    if func is None:
        return "unattested", resolve_reason
    return _replay_record(record, func, torch)


def _attest_precondition_reason(record: InjectedOp, torch: Any) -> str | None:
    """The reason one record cannot replay at all, or ``None``."""

    if record.out is None or not isinstance(record.out, torch.Tensor):
        return "payload_unavailable"
    if record.callable_ref is None:
        return "callable_ref_unavailable"
    if record.saved_args is None:
        return "args_unavailable"
    if record.layer_type in _NONDETERMINISTIC_LAYER_TYPES:
        return "nondeterministic_callable"
    if record.fire_device is not None and record.fire_device != str(record.out.device):
        return "device_changed"
    return None


_CALLABLE_RESOLUTION_FAILURES = (
    AttributeError,
    ImportError,
    KeyError,
    TypeError,
    ValueError,
)
"""Resolution failures classifying as ``callable_missing`` (never a crash)."""


def _resolve_record_callable(
    record: InjectedOp,
    *,
    trust_custom_callables: bool,
    allowed_custom_callable_modules: Any,
) -> tuple[Any, str | None]:
    """Resolve one record's callable through the trust gate, or classify."""

    from .errors import ReplayPreconditionError, UntrustedCallableError
    from .resolver import resolve_function_registry_key

    try:
        func = resolve_function_registry_key(
            record.callable_ref,
            trust_custom_callables=trust_custom_callables,
            allowed_custom_callable_modules=allowed_custom_callable_modules,
        )
    except UntrustedCallableError:
        return None, "callable_untrusted"
    except (ReplayPreconditionError, *_CALLABLE_RESOLUTION_FAILURES):
        return None, "callable_missing"
    return func, None


def _replay_record(record: InjectedOp, func: Any, torch: Any) -> tuple[str, str | None]:
    """Re-execute one trusted callable and compare byte-exactly."""

    kwargs = dict(record.saved_kwargs or ())
    try:
        with torch.no_grad():
            replayed = func(*(record.saved_args or ()), **kwargs)
    except Exception:  # noqa: BLE001 -- arbitrary user callable: any raise is
        # ambiguous evidence (signature drift, device absence) that neither
        # corroborates nor refutes the record; validation must not crash.
        return "unattested", "replay_raised"
    slot_value = _select_output_slot(replayed, record.provenance.output_slot, torch)
    if slot_value is None:
        return "diverged", "replay_output_structure"
    if _replay_values_match(slot_value, record.out, torch):
        return "attested", None
    return "diverged", "replay_value_mismatch"


def _replay_values_match(replayed: Any, recorded: Any, torch: Any) -> bool:
    """Exact positional equality that treats NaN as equal to NaN.

    ``torch.equal`` reads ``NaN != NaN``, so an HONEST injected op whose
    output legitimately holds NaN (``log`` of a negative value inside a
    hook) replayed to the identical result would be called ``diverged``
    (AUD-CODE 3.7b). Equality here is exact per element -- same dtype,
    shape, and device; every finite element equal; NaN exactly where the
    record has NaN. This is not a tolerance: no differing finite value
    passes.
    """

    if (
        replayed.dtype != recorded.dtype
        or tuple(replayed.shape) != tuple(recorded.shape)
        or replayed.device != recorded.device
    ):
        return False
    if torch.equal(replayed, recorded):
        return True
    if not (replayed.is_floating_point() or replayed.is_complex()):
        return False
    replayed_nan = torch.isnan(replayed)
    recorded_nan = torch.isnan(recorded)
    if not torch.equal(replayed_nan, recorded_nan):
        return False
    return bool(torch.equal(replayed[~replayed_nan], recorded[~recorded_nan]))


def _select_output_slot(replayed: Any, output_slot: int, torch: Any) -> Any:
    """Select the recorded output slot from a replayed call result."""

    if isinstance(replayed, torch.Tensor):
        return replayed if output_slot == 0 else None
    if isinstance(replayed, (list, tuple)) and 0 <= output_slot < len(replayed):
        candidate = replayed[output_slot]
        return candidate if isinstance(candidate, torch.Tensor) else None
    return None
