"""Exception restoration and concurrency refusals for mutable module globals.

The inventory half (every mutable module global in the package classified into one
lifecycle class, the weak/strong ledger and the hot-subset census) reads source only and
moved to ``scripts/checks/global_state_inventory.py``, run by ``scripts/checks/run_all.py``;
its rows live in ``scripts/checks/global_state_rows.py``. The tests here capture models.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, cast

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._capture import projections, trace as capture_trace
from torchlens._utils import _tensor_label as torch_tl
from torchlens.backends.torch import completeness_witness, rescue
from torchlens.errors import TraceCaptureError


def _capture_scope_snapshot() -> dict[str, Any]:
    """Return high-risk per-capture global state for restoration checks.

    Returns
    -------
    dict[str, Any]
        Values that must be identical before and after a failed capture.
    """

    return {
        "logging_enabled": _state._logging_enabled,
        "active_trace": _state._active_trace,
        "active_owner_thread_id": _state._active_owner_thread_id,
        "nonowner_belt_armed": _state._nonowner_belt_armed,
        "active_fast_run_collector": _state._active_fast_run_collector,
        "active_hook_plan": _state._active_hook_plan,
        "active_intervention_spec": _state._active_intervention_spec,
        "capture_replay_templates": _state._capture_replay_templates,
        "relationship_model_id": _state._relationship_model_id,
        "relationship_model_class": _state._relationship_model_class,
        "relationship_weight_fingerprint": _state._relationship_weight_fingerprint,
        "relationship_input_id": _state._relationship_input_id,
        "relationship_input_shape_hash": _state._relationship_input_shape_hash,
        "runnable_ledger_armed": _state._runnable_ledger_armed,
        "aten_recording_armed": _state._aten_recording_armed,
        "capture_reserved_by": _state._capture_reserved_by,
        "active_label_session": torch_tl._ACTIVE_LABEL_SESSION,
        "active_witness_state": completeness_witness._ACTIVE_WITNESS_STATE,
        "rescue_active": rescue._rescue_is_active(),
        "active_recording_state": projections._active_recording_state,
        "active_capture_backend": capture_trace._ACTIVE_CAPTURE_BACKEND,
    }


class _InjectedBaseFailure(BaseException):
    """BaseException subclass used to exercise the interruption cleanup arm."""


class _RaiseMidCapture(nn.Module):
    """Run one logged op before raising a selected exception class."""

    def __init__(self, error_type: type[BaseException]) -> None:
        """Store the exception type raised during ``forward``.

        Parameters
        ----------
        error_type:
            Exception class to raise after one tensor operation.
        """

        super().__init__()
        self.error_type = error_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one operation and then raise the injected exception.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            This path never returns.

        Raises
        ------
        BaseException
            Always raises ``self.error_type`` after the logged operation.
        """

        _ = torch.relu(x)
        raise self.error_type("injected mid-capture failure")


class _BlockingCapture(nn.Module):
    """Hold a public capture open while a concurrent capture is attempted."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        """Store synchronization events for the capture overlap.

        Parameters
        ----------
        entered:
            Event set after the first logged operation runs.
        release:
            Event that permits the model to finish.
        """

        super().__init__()
        self.entered = entered
        self.release = release

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log one operation and wait until the contender is refused.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Logged activation after the release event is set.
        """

        out = torch.relu(x)
        self.entered.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("concurrent capture test did not release the owner")
        return out


@pytest.mark.parametrize("error_type", [RuntimeError, _InjectedBaseFailure])
def test_mid_capture_failure_restores_process_state(
    error_type: type[BaseException],
) -> None:
    """Ordinary and interruption failures leave no stale capture owner.

    Parameters
    ----------
    error_type:
        Failure class injected after one recorded operation.
    """

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()

    # An ordinary forward exception escapes as the door's TraceCaptureError
    # chained from it; an interrupt-class BaseException is never wrapped.
    expected = TraceCaptureError if issubclass(error_type, Exception) else error_type
    with pytest.raises(expected, match="injected mid-capture failure") as excinfo:
        tl.trace(_RaiseMidCapture(error_type), torch.ones(2))
    if expected is TraceCaptureError:
        assert type(excinfo.value.__cause__) is error_type

    assert _capture_scope_snapshot() == before
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py, campaign mutant kills.
def test_concurrent_public_capture_refuses_without_corruption() -> None:
    """Overlapping public captures fail loudly and leave the owner intact."""

    tl.trace(nn.ReLU(), torch.ones(2))
    before = _capture_scope_snapshot()
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Run the capture that owns process-global logging state."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.trace(nn.ReLU(), torch.ones(2))
    finally:
        release.set()
        owner.join(timeout=5.0)

    assert not owner.is_alive(), "owner capture did not finish after release"
    assert owner_errors == []
    assert _capture_scope_snapshot() == before


# Kept by shrink-r4 (per-id subset rule): sole coverage of recording/_recorder.py.
def test_refused_concurrent_record_never_installs_recording_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused ``tl.record`` must not touch the fastlog recording global.

    The loser used to install its ``RecordingState`` (overwriting the admitted
    recorder's) and only then reach the inner admission refusal, projecting the
    winner's events into the loser's state for that window. The recorder-side
    reservation refuses before the install.
    """

    from torchlens.recording import _recorder as recorder_module

    installs: list[Any] = []
    real_install = recorder_module.active_recording_state

    def counting_install(state: Any) -> Any:
        """Record every recording-state install before delegating."""

        installs.append(state)
        return real_install(state)

    monkeypatch.setattr(recorder_module, "active_recording_state", counting_install)

    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Hold a live capture open while the record() loser is refused."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            tl.record(nn.ReLU(), torch.ones(2), save=tl.func("relu"))
    finally:
        release.set()
        owner.join(timeout=10.0)

    assert owner_errors == []
    assert installs == [], (
        "the refused record() installed its RecordingState before the admission refusal fired"
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
def test_publish_active_trace_refuses_concurrent_and_clears_owner() -> None:
    """Non-forward publication windows are admission-locked and owner-stamped.

    tf capture and paddle derived-grad replays publish ``_active_trace``
    without the logging toggle; a raw save/restore swap bypassed admission
    (silent corruption of a concurrent torch capture) and could republish a
    finished trace on restore. ``publish_active_trace`` must refuse typed
    against a live capture, set the owner thread id for the window, and clear
    both on exit.
    """

    sentinel = cast("Any", object())
    with _state.publish_active_trace(sentinel):
        assert _state._active_trace is sentinel
        assert _state._active_owner_thread_id == threading.get_ident()
        # A second publication (any thread) refuses while the window is open.
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            with _state.publish_active_trace(cast("Any", object())):
                pass  # pragma: no cover - refused above
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None

    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def run_owner() -> None:
        """Hold a live torch capture open for the publication refusal."""

        try:
            tl.trace(_BlockingCapture(entered, release), torch.ones(2))
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            owner_errors.append(error)

    owner = threading.Thread(target=run_owner)
    owner.start()
    assert entered.wait(timeout=5.0), "owner capture never reached its forward"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            with _state.publish_active_trace(sentinel):
                pass  # pragma: no cover - refused above
    finally:
        release.set()
        owner.join(timeout=10.0)
    assert owner_errors == []


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
def test_publish_backward_capture_nests_same_thread_and_restores() -> None:
    """The backward publication handle keeps raw-swap nesting semantics.

    The multi-trace backward bracket and an inner ``backward()`` inside a
    traced forward legitimately nest backward windows on ONE thread, so the
    admission-locked replacement for the raw ``_active_trace`` swap must
    save/restore LIFO on the same thread, restore exactly once (idempotent
    against stacked unwind arms), and clear the owner id at the end.
    """

    outer = cast("Any", object())
    inner = cast("Any", object())
    outer_publication = _state.publish_backward_capture(
        outer, hook_plan=None, intervention_spec=None
    )
    try:
        assert _state._active_trace is outer
        assert _state._active_owner_thread_id == threading.get_ident()
        inner_publication = _state.publish_backward_capture(
            inner, hook_plan=None, intervention_spec=None
        )
        assert _state._active_trace is inner
        inner_publication.restore()
        assert _state._active_trace is outer
        inner_publication.restore()  # idempotent: must NOT re-clobber to inner
        assert _state._active_trace is outer
    finally:
        outer_publication.restore()
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py, backends/torch/backward.py.
def test_backward_refuses_foreign_live_window_instead_of_wedging() -> None:
    """``backward`` concurrent with a foreign capture window refuses typed.

    The last unconverted raw ``_active_trace`` swap: interleaved with another
    thread's live window, the unlocked save could snapshot that window's trace
    as "previous" and the ``finally`` republished it after the window closed —
    ``_active_trace`` stayed permanently non-``None`` and EVERY later capture
    was refused (a process-global wedge). The admission-locked publication
    refuses the foreign window typed instead; after the window closes the same
    backward and later captures run untouched.
    """

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(2, 4, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()

    sentinel = cast("Any", object())
    entered = threading.Event()
    release = threading.Event()
    window_errors: list[BaseException] = []

    def hold_window() -> None:
        """Hold a foreign non-forward publication window open."""

        try:
            with _state.publish_active_trace(sentinel):
                entered.set()
                release.wait(timeout=20.0)
        except BaseException as error:  # pragma: no cover - surfaced by asserts
            window_errors.append(error)

    holder = threading.Thread(target=hold_window)
    holder.start()
    assert entered.wait(timeout=5.0), "foreign window never opened"
    try:
        with pytest.raises(_state.ReentrantTraceError, match="not re-entrant"):
            trace.backward(loss)
    finally:
        release.set()
        holder.join(timeout=10.0)
    assert window_errors == []

    # No wedge: the globals are clean, the refused backward runs now, and a
    # later capture is admitted.
    assert _state._active_trace is None
    assert _state._active_owner_thread_id is None
    trace.backward(loss)
    assert int(len(trace.autograd.backward_passes)) == 1
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/run.py, _state.py, _utils/arg_handling.py ....
def test_nested_public_capture_inside_reserved_window_refuses_typed() -> None:
    """A nested ``tl.trace`` from user code inside the reserved window refuses.

    R55 (r7 b8-sol): the same-thread reservation passthrough keyed on thread
    ident alone, so user code running inside the outer capture's reserved
    pre-admission window (here: a tensor-subclass ``__torch_function__`` fired
    by input setup) could start a nested PUBLIC capture that ran to completion
    -- BOTH captures settled (probe ``OUTER_OK``/``INNER_COMPLETED`` at the
    hunt-6 pin) instead of the documented ``ReentrantTraceError``. The
    passthrough now requires the recorder's continuation token; the nested
    entry holds none and refuses typed, and the refusal releases nothing it
    does not own.
    """

    fired: dict[str, object] = {"result": None}

    class _NestedTraceTensor(torch.Tensor):
        @classmethod
        def __torch_function__(
            cls, func: object, types: object, args: tuple = (), kwargs: dict | None = None
        ) -> object:
            kwargs = kwargs or {}
            if fired["result"] is None and _state._capture_reserved_by is not None:
                fired["result"] = "fired"
                try:
                    tl.trace(nn.ReLU(), torch.ones(2))
                    fired["result"] = "inner_completed"
                except _state.ReentrantTraceError:
                    fired["result"] = "inner_refused"
                    raise
            return super().__torch_function__(func, types, args, kwargs)

    inputs = torch.randn(2, 3).as_subclass(_NestedTraceTensor)
    with pytest.raises(_state.ReentrantTraceError):
        tl.trace(nn.Linear(3, 2), inputs)
    assert fired["result"] == "inner_refused"
    assert _state._capture_reserved_by is None
    # The refusal left admission clean: a fresh capture is admitted.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
def test_auto_name_counter_is_atomic_under_racing_threads() -> None:
    """Racing pre-admission ``_auto_name`` calls never mint duplicate names.

    R54 (r7 b8-sol): ``_auto_name`` runs during capture setup BEFORE
    admission, outside ``active_logging()``'s guard, and its unlocked
    read-modify-write let two threads read the same counter value and stamp
    two captures with the SAME auto name. The get+increment is now atomic
    under ``_state._naming_lock``.
    """

    class _RaceNamed:
        pass

    _state.reset_naming_counter("_racenamed")
    switch_interval = sys.getswitchinterval()
    names: list[str] = []
    names_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def mint(count: int) -> None:
        barrier.wait()
        local = [_state._auto_name(_RaceNamed()) for _ in range(count)]
        with names_lock:
            names.extend(local)

    sys.setswitchinterval(1e-6)
    try:
        workers = [threading.Thread(target=mint, args=(2000,)) for _ in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30.0)
    finally:
        sys.setswitchinterval(switch_interval)
        _state.reset_naming_counter("_racenamed")
    assert len(names) == 8000
    assert len(set(names)) == 8000, "duplicate auto names minted under the race"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _ir/container.py, types.py.
def test_register_container_racing_lookup_never_breaks_iteration() -> None:
    """``register_container`` racing a lookup never raises mid-iteration.

    R54 (r7 b2-sol): ``get_registered_container`` iterated the LIVE registry
    dict while public ``register_container`` mutated it from another thread --
    a registration landing mid-capture raised ``RuntimeError: dictionary
    changed size during iteration`` inside the forward walk. The reader now
    snapshots under the registry lock.
    """

    from torchlens._ir import container as container_mod

    registered_types: list[type] = []
    errors: list[BaseException] = []
    stop = threading.Event()
    switch_interval = sys.getswitchinterval()

    class _LookupProbe:
        pass

    def writer() -> None:
        try:
            for index in range(400):
                if stop.is_set():
                    break
                fresh = type(f"_RaceContainer{index}", (), {})
                registered_types.append(fresh)
                tl.types.register_container(
                    fresh,
                    lambda value: ([], None),
                    lambda aux, children: object(),
                )
        except BaseException as error:  # pragma: no cover - the defect signal
            errors.append(error)

    def reader() -> None:
        try:
            while not stop.is_set():
                container_mod.get_registered_container(_LookupProbe)
        except BaseException as error:  # pragma: no cover - the defect signal
            errors.append(error)

    sys.setswitchinterval(1e-6)
    try:
        reader_thread = threading.Thread(target=reader)
        writer_thread = threading.Thread(target=writer)
        reader_thread.start()
        writer_thread.start()
        writer_thread.join(timeout=30.0)
        stop.set()
        reader_thread.join(timeout=30.0)
    finally:
        sys.setswitchinterval(switch_interval)
        stop.set()
        with container_mod._CONTAINER_REGISTRY_LOCK:
            for registered in registered_types:
                container_mod._CONTAINER_REGISTRY.pop(registered, None)
    assert errors == [], f"registry race surfaced: {errors!r}"


# Kept by shrink-r4 (per-id subset rule): sole coverage of _state.py.
def test_capture_admission_runs_under_the_admission_lock() -> None:
    """Admission blocks while another thread holds the admission lock.

    ``active_logging``'s refusal check and its publication of ``_active_trace`` /
    ``_active_owner_thread_id`` / ``_logging_enabled`` are separate bytecodes.
    Unlocked, two threads entering together can both pass the check, and the
    loser then overwrites the winner's owner id — after which every op the winner
    logs is dropped by the wrapper's owner-thread fast path and its Trace is
    silently short, with no error anywhere. The check and the publication
    therefore have to happen under one lock; this asserts that directly, because
    the racing window itself is only a few bytecodes wide and a probabilistic
    probe cannot gate it reliably.
    """

    before = _capture_scope_snapshot()
    entered = threading.Event()
    blocked_for_lock = threading.Event()
    admitted = threading.Event()
    failures: list[BaseException] = []

    def admit() -> None:
        """Enter and immediately leave one capture session."""

        try:
            with _state.active_logging(cast("Any", object())):
                admitted.set()
        except BaseException as error:  # pragma: no cover - reported by the test
            failures.append(error)
            admitted.set()

    with _state._capture_admission_lock:
        entered.set()
        worker = threading.Thread(target=admit)
        worker.start()
        # The worker cannot reach the check, let alone publish, while the lock
        # is held here. If admission ran outside the lock it would sail through.
        blocked_for_lock.wait(timeout=0.5)
        assert not admitted.is_set(), (
            "active_logging admitted a capture while the admission lock was "
            "held: the check-then-publish sequence is not serialized"
        )
        assert _state._active_trace is None
        assert _state._active_owner_thread_id is None

    assert admitted.wait(timeout=10.0), "admission never completed after release"
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "admission worker hung"
    assert failures == [], f"admission failed after the lock was released: {failures!r}"
    assert _capture_scope_snapshot() == before


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/wrappers.py, campaign mutant kills.
def test_unwrap_torch_refuses_during_an_active_capture() -> None:
    """Mid-capture ``unwrap_torch()`` is a typed refusal, not a silent truncation.

    Reachable single-threaded: from a forward hook, an ``activation_transform``,
    or any user callback running inside the traced forward. Removing the
    wrappers there left the rest of the forward unlogged and returned a
    truncated Trace with no error at all.
    """

    from torchlens.backends.torch.wrappers import unwrap_torch

    seen: list[BaseException] = []

    class _UnwrapMidForward(nn.Module):
        """Attempt an unwrap between two logged operations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one op, try to unwrap, then run another op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = torch.relu(x)
            try:
                unwrap_torch()
            except BaseException as error:
                seen.append(error)
            return torch.relu(x)

    trace = tl.trace(_UnwrapMidForward(), torch.ones(2))

    assert len(seen) == 1, "unwrap_torch() mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureNotActiveError)
    assert error.fields["code"] == "unwrap_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused unwrap still truncated the capture"

    # The wrappers survived the refusal: the next capture needs no re-wrap.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_release_model_refuses_during_an_active_capture() -> None:
    """Mid-capture ``tl.release_model()`` is a typed refusal, not silent damage.

    The missing sibling of the ``unwrap_torch`` guard: releasing the model
    mid-forward (reachable single-threaded from a forward hook or
    ``activation_transform``) stripped the ``tl_*`` / ``._tl`` metadata the
    live capture's module attribution reads, and the capture then finished
    ``capture_verified`` with silently emptied module attribution.
    """

    seen: list[BaseException] = []

    class _ReleaseMidForward(nn.Module):
        """Attempt a self-release between two logged operations."""

        def __init__(self) -> None:
            """Build a submodule so module attribution has something to lose."""

            super().__init__()
            self.inner = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one module, try to release, then run another op.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = self.inner(x)
            try:
                tl.release_model(self)
            except BaseException as error:
                seen.append(error)
            return torch.relu(x)

    model = _ReleaseMidForward()
    trace = tl.trace(model, torch.ones(2))

    assert len(seen) == 1, "release_model() mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureNotActiveError)
    assert error.fields["code"] == "release_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused release still truncated the capture"
    # Module attribution survived: the submodule call is still attributed.
    assert any(op.modules for op in trace.compute_ops), (
        "the refused release still emptied module attribution"
    )

    # Releasing AFTER the capture stays the supported no-questions path.
    tl.release_model(model)
    recovered = tl.trace(model, torch.ones(2))
    assert any(op.modules for op in recovered.compute_ops)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/cleanup.py.
def test_cleanup_of_active_trace_refuses_mid_capture() -> None:
    """Husking the live capture's own trace mid-forward refuses typed.

    Sibling of the ``unwrap_torch`` / ``release_model`` guards: without the
    refusal the capture died later on a raw ``AttributeError`` (missing
    ``_wrapper_runtime_ws``) deep inside the commit path. Cleaning up a
    DIFFERENT, finished trace during a capture stays supported.
    """

    from torchlens import _state

    finished = tl.trace(nn.ReLU(), torch.ones(2))
    seen: list[BaseException] = []

    class _CleanupMidForward(nn.Module):
        """Attempt to husk the active trace between two logged ops."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Refused self-cleanup; allowed foreign-trace cleanup.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Activation after both operations.
            """

            x = torch.relu(x)
            try:
                _state._active_trace.cleanup()
            except BaseException as error:
                seen.append(error)
            finished.cleanup()  # foreign finished trace: must stay allowed
            return torch.relu(x)

    trace = tl.trace(_CleanupMidForward(), torch.ones(2))

    assert len(seen) == 1, "cleanup() of the active trace mid-capture did not refuse"
    error = seen[0]
    assert isinstance(error, tl.errors.CaptureNotActiveError)
    assert error.fields["code"] == "cleanup_during_active_capture"
    relu_ops = [op for op in trace.compute_ops if op.func_name == "relu"]
    assert len(relu_ops) == 2, "the refused cleanup still truncated the capture"
    trace.cleanup()  # post-capture cleanup stays the supported path


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/finalize/arms.py, backends/torch/_failed_session.py.
def test_interrupted_partial_diagnostics_still_restore_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interruption during partial-trace recovery still tears the session down.

    The failed-forward epilogue builds best-effort partial diagnostics and then
    restores the model. Both arms of that construction catch ``Exception``, so a
    ``KeyboardInterrupt`` raised inside it escaped straight past
    ``cleanup_model_session`` — leaving the user's model with TorchLens-forced
    ``requires_grad``, ``tl_*`` attributes, and an installed buffer tracker.
    """

    from torchlens import _partial as partial_module

    class _FailingModel(nn.Module):
        """Register a frozen parameter and then fail the forward."""

        def __init__(self) -> None:
            """Build a model with one explicitly frozen parameter."""

            super().__init__()
            self.weight = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run one logged op and then raise.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                This path never returns.

            Raises
            ------
            ValueError
                Always, after one logged operation.
            """

            _ = torch.relu(x * self.weight)
            raise ValueError("forward failure with interrupted diagnostics")

    def _interrupt_partial_construction(*_args: Any, **_kwargs: Any) -> Any:
        """Interrupt partial-trace construction the way Ctrl-C would."""

        raise KeyboardInterrupt("interrupted during partial construction")

    monkeypatch.setattr(partial_module.PartialTrace, "from_trace", _interrupt_partial_construction)

    model = _FailingModel()
    before = _capture_scope_snapshot()

    with pytest.raises(KeyboardInterrupt):
        tl.trace(model, torch.ones(2))

    assert model.weight.requires_grad is False, (
        "the interrupted epilogue left TorchLens-forced requires_grad on a frozen parameter"
    )
    assert not [name for name in vars(model) if name.startswith("tl_")], (
        "the interrupted epilogue left tl_* session metadata on the model"
    )
    assert _capture_scope_snapshot() == before
    # The process is still usable for the next capture.
    recovered = tl.trace(nn.ReLU(), torch.ones(2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)
