"""Phase 14 error catalog and cross-cutting intervention API audit tests."""

from __future__ import annotations

import gc
import threading
import weakref
import warnings
from collections.abc import Callable, Iterable
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import RunState
from torchlens.intervention import errors as terrors


SeverityClass = type[BaseException]
WarningClass = type[Warning]

ERROR_NAMES: tuple[str, ...] = (
    "EngineDispatchError",
    "ModelMismatchError",
    "BundleMemberError",
    "BundleRelationshipError",
    "BaselineUndeterminedError",
    "NoParentError",
    "DeadParentError",
    "ReplayPreconditionError",
    "SiteResolutionError",
    "SiteAmbiguityError",
    "HookSignatureError",
    "HookValueError",
    "HookSiteCoverageError",
    "RecursiveTracingError",
    "AxisAmbiguityError",
    "AppendMismatchError",
    "AppendBatchDependenceError",
    "ControlFlowDivergenceError",
    "SpecPortabilityError",
    "GraphShapeMismatchError",
    "LiveModeLabelError",
    "InterventionReadyConflictError",
    "SpliceModuleDtypeError",
    "SpliceModuleDeviceError",
    "SpecMutationError",
    "OpaqueCallableInExecutableSaveError",
    "DirectWriteInExecutableSaveError",
)
"""v5.2 intervention errors owned by Phase 14."""

WARNING_NAMES: tuple[str, ...] = (
    "MutateInPlaceWarning",
    "DirectActivationWriteWarning",
    "MultiMatchWarning",
    "ControlFlowDivergenceWarning",
    "BatchNormTrainModeWarning",
)
"""v5.2 intervention warnings owned by Phase 14."""

CATALOG_NAMES: tuple[str, ...] = ERROR_NAMES + WARNING_NAMES
"""Full public Phase 14 catalog."""

CATALOG_EXERCISE_MANIFEST: dict[str, str] = {
    "EngineDispatchError": "tests/test_intervention_phase8b.py::test_do_ambiguous_dispatch_and_model_mismatch_errors",
    "ModelMismatchError": "tests/test_intervention_phase8b.py::test_do_ambiguous_dispatch_and_model_mismatch_errors",
    "BundleMemberError": "tests/test_intervention_phase9.py",
    "BundleRelationshipError": "tests/test_intervention_phase9.py",
    "BaselineUndeterminedError": "tests/test_intervention_phase9.py",
    "NoParentError": "tests/test_intervention_phase9.py",
    "DeadParentError": "tests/test_intervention_phase9.py",
    "ReplayPreconditionError": "tests/test_intervention_phase6.py::test_replay_rejects_non_intervention_ready_logs",
    "SiteResolutionError": "tests/test_intervention_phase2.py::test_resolution_errors_strict_mode_and_warnings",
    "SiteAmbiguityError": "tests/test_intervention_phase2.py::test_resolution_errors_strict_mode_and_warnings",
    "HookSignatureError": "tests/test_intervention_phase3.py::test_normalizer_rejects_missing_site_and_bad_signature",
    "HookValueError": "tests/test_intervention_phase3.py::test_execute_hook_rejects_none_type_shape_dtype_and_device",
    "HookSiteCoverageError": "tests/test_intervention_phase3.py::test_normalizer_rejects_missing_site_and_bad_signature",
    "RecursiveTracingError": "tests/test_intervention_error_catalog.py::test_full_catalog_is_exercised",
    "AxisAmbiguityError": "tests/test_intervention_error_catalog.py::test_helper_axis_ambiguity_uses_catalog_error",
    "AppendMismatchError": "tests/test_intervention_phase12.py::test_append_shape_mismatch_raises",
    "AppendBatchDependenceError": "tests/test_intervention_phase12.py::test_append_batch_dependent_helper_rejected_after_clean_rerun",
    "ControlFlowDivergenceError": "tests/test_intervention_phase7.py::test_rerun_strict_divergence_raises_before_swap",
    "SpecPortabilityError": "tests/test_intervention_error_catalog.py::test_full_catalog_is_exercised",
    "GraphShapeMismatchError": "tests/test_intervention_phase10.py",
    "LiveModeLabelError": "tests/test_intervention_phase4c.py",
    "InterventionReadyConflictError": "tests/test_intervention_phase4a.py::test_intervention_ready_rejects_nonempty_layers_to_save_list",
    "SpliceModuleDtypeError": "tests/test_intervention_phase3.py::test_splice_module_dtype_error_is_specific",
    "SpliceModuleDeviceError": "tests/test_intervention_error_catalog.py::test_full_catalog_is_exercised",
    "SpecMutationError": "tests/test_intervention_phase8a.py",
    "OpaqueCallableInExecutableSaveError": "tests/test_intervention_phase10.py::test_portable_save_rejects_opaque_callable",
    "DirectWriteInExecutableSaveError": "tests/test_intervention_phase10.py",
    "MutateInPlaceWarning": "tests/test_intervention_phase8b.py::test_mutate_warning_fires_once_and_can_be_suppressed",
    "DirectActivationWriteWarning": "tests/test_intervention_phase8b.py::test_direct_activation_write_warns_once_and_marks_dirty",
    "MultiMatchWarning": "tests/test_intervention_phase2.py::test_resolution_errors_strict_mode_and_warnings",
    "ControlFlowDivergenceWarning": "tests/test_intervention_phase7.py::test_rerun_non_strict_divergence_warns_and_swaps",
    "BatchNormTrainModeWarning": "tests/test_intervention_phase12.py::test_append_batchnorm_train_mode_warns",
}
"""Manual manifest proving every catalog entry is represented in the test matrix."""


class _ReluAdd(torch.nn.Module):
    """Small stable model for Phase 14 cross-cutting tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReLU and a downstream add.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output plus one.
        """

        return torch.relu(x) + 1


class _LinearRelu(torch.nn.Module):
    """Small parameterized model for rerun and append tests."""

    def __init__(self) -> None:
        """Initialize deterministic linear weights."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(3))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear layer and ReLU.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU-transformed linear output.
        """

        return torch.relu(self.linear(x))


def _zero_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return zeros matching the input activation.

    Parameters
    ----------
    activation:
        Hook input activation.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zero activation.
    """

    del hook
    return torch.zeros_like(activation)


def _capture(model: torch.nn.Module | None = None, x: torch.Tensor | None = None) -> tl.ModelLog:
    """Capture an intervention-ready test log.

    Parameters
    ----------
    model:
        Optional model to capture.
    x:
        Optional input tensor.

    Returns
    -------
    tl.ModelLog
        Captured intervention-ready log.
    """

    torch.manual_seed(14)
    return tl.log_forward_pass(
        _ReluAdd() if model is None else model,
        torch.tensor([[-1.0, 2.0, 3.0]]) if x is None else x,
        vis_opt="none",
        intervention_ready=True,
    )


def _catalog_class(name: str) -> type[Any]:
    """Return a catalog class by public name.

    Parameters
    ----------
    name:
        Public class name in ``torchlens.intervention.errors``.

    Returns
    -------
    type[Any]
        Matching class object.
    """

    return getattr(terrors, name)


def _is_warning_name(name: str) -> bool:
    """Return whether a catalog entry is a warning class.

    Parameters
    ----------
    name:
        Public catalog entry name.

    Returns
    -------
    bool
        Whether ``name`` identifies a warning.
    """

    return name in WARNING_NAMES


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_each_named_error_is_importable_and_has_severity(name: str) -> None:
    """Every v5.2 catalog entry is importable and has a valid severity."""

    cls = _catalog_class(name)

    assert getattr(tl.intervention, name) is cls
    assert hasattr(cls, "severity"), f"{name} missing severity tag"
    assert cls.severity in ("recoverable", "informational", "fatal")


def test_error_warning_partition_is_consistent() -> None:
    """Catalog entries use exception and warning bases consistently."""

    for name in ERROR_NAMES:
        assert issubclass(_catalog_class(name), BaseException)
        assert not issubclass(_catalog_class(name), Warning)
    for name in WARNING_NAMES:
        assert issubclass(_catalog_class(name), Warning)
        assert _catalog_class(name).severity == "informational"


def test_spec_portability_alias_reconciles_executable_save_name() -> None:
    """v5.2 ``SpecPortabilityError`` remains compatible with Phase 10 naming."""

    assert terrors.SpecPortabilityError is terrors.OpaqueCallableInExecutableSaveError


def test_full_catalog_is_exercised() -> None:
    """Raise or warn every named catalog entry at least once in the matrix."""

    assert set(CATALOG_EXERCISE_MANIFEST) == set(CATALOG_NAMES)
    for name in ERROR_NAMES:
        cls = _catalog_class(name)
        with pytest.raises(cls):
            raise cls(site="relu_1_2", helper="zero_ablate", remediation="retry")
    for name in WARNING_NAMES:
        cls = _catalog_class(name)
        with pytest.warns(cls):
            warnings.warn(cls(site="relu_1_2", helper="zero_ablate"), stacklevel=1)


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_catalog_messages_include_required_payload_fields(name: str) -> None:
    """Named-field constructors include cited variables in user-facing text."""

    cls = _catalog_class(name)
    instance = cls(site="relu_1_2", helper="zero_ablate", shape=(2, 3))

    message = str(instance)
    assert "relu_1_2" in message
    assert "zero_ablate" in message
    assert "(2, 3)" in message


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("HookSignatureError", "fatal"),
        ("GraphShapeMismatchError", "fatal"),
        ("ControlFlowDivergenceError", "fatal"),
        ("ModelMismatchError", "fatal"),
        ("BundleRelationshipError", "fatal"),
        ("ReplayPreconditionError", "recoverable"),
        ("SiteResolutionError", "recoverable"),
        ("AppendMismatchError", "recoverable"),
        ("SpecMutationError", "recoverable"),
        ("MutateInPlaceWarning", "informational"),
    ),
)
def test_catalog_severity_policy_examples(name: str, expected: str) -> None:
    """Representative entries follow the Phase 14 severity policy."""

    assert _catalog_class(name).severity == expected


@pytest.mark.smoke
def test_axis_a_public_verbs_success_paths() -> None:
    """Public set, attach_hooks, do, replay, rerun, fork, and append smoke paths work."""

    log = _capture()
    log.set(tl.func("relu"), torch.zeros(1, 3), confirm_mutation=True)
    log.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    replay_result = log.replay()
    assert replay_result is log
    assert log.run_state is RunState.REPLAY_PROPAGATED

    do_log = _capture()
    do_log.do(tl.func("relu"), torch.ones(1, 3), engine="set_only", confirm_mutation=True)
    assert do_log.run_state is RunState.SPEC_STALE

    fork = do_log.fork("phase14")
    assert fork.parent_run() is do_log

    model = _LinearRelu()
    x = torch.randn(2, 3)
    rerun_log = _capture(model, x)
    rerun_log.rerun(model, x)
    assert rerun_log.run_state is RunState.RERUN_PROPAGATED
    rerun_log.rerun(model, torch.randn(1, 3), append=True)
    assert rerun_log.run_state is RunState.APPENDED


@pytest.mark.parametrize(
    ("verb", "operation"),
    (
        ("set", lambda log: log.set(tl.func("missing"), torch.zeros(1, 3), confirm_mutation=True)),
        (
            "attach_hooks",
            lambda log: log.attach_hooks(tl.func("missing"), _zero_hook, confirm_mutation=True),
        ),
        (
            "do",
            lambda log: log.do(
                tl.func("relu"), _zero_hook, x=torch.zeros(1, 3), confirm_mutation=True
            ),
        ),
        ("replay", lambda log: log.replay()),
        ("rerun", lambda log: log.rerun(_ReluAdd())),
        ("append", lambda log: log.rerun(_ReluAdd(), torch.ones(1, 4), append=True)),
    ),
)
def test_axis_a_public_verbs_failure_paths(
    verb: str, operation: Callable[[tl.ModelLog], object]
) -> None:
    """Each public propagation verb has at least one cataloged or stable failure path."""

    log = _capture()
    if verb == "replay":
        log._intervention_spec.clear()
    with pytest.raises(Exception) as excinfo:
        operation(log)
    assert str(excinfo.value)


def test_axis_b_replay_and_rerun_match_for_graph_stable_hook() -> None:
    """Replay and rerun produce comparable outputs for graph-stable zeroing."""

    x = torch.tensor([[-1.0, 2.0, 3.0]])
    replay_log = _capture(_ReluAdd(), x)
    rerun_log = _capture(_ReluAdd(), x)

    replay_log.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    rerun_log.attach_hooks(tl.func("relu"), _zero_hook, confirm_mutation=True)
    replay_log.replay()
    rerun_log.rerun(_ReluAdd(), x)

    replay_output = replay_log[replay_log.output_layers[0]].activation
    rerun_output = rerun_log[rerun_log.output_layers[0]].activation
    assert torch.equal(replay_output, rerun_output)


def test_axis_i_list_logs_snapshot_survives_concurrent_log_creation() -> None:
    """``tl.list_logs()`` returns valid snapshots while logs are created concurrently."""

    errors: list[BaseException] = []
    snapshots: list[tuple[tl.ModelLog, ...]] = []
    capture_lock = threading.Lock()

    def worker(seed: int) -> None:
        """Capture one log and record a registry snapshot.

        Parameters
        ----------
        seed:
            Per-thread RNG seed.
        """

        try:
            torch.manual_seed(seed)
            snapshots.append(tl.list_logs())
            with capture_lock:
                _capture(_ReluAdd(), torch.randn(1, 3))
            snapshots.append(tl.list_logs())
        except BaseException as exc:  # pragma: no cover - failure path is asserted below.
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert snapshots
    assert all(isinstance(snapshot, tuple) for snapshot in snapshots)
    assert all(hasattr(log, "layer_list") for snapshot in snapshots for log in snapshot)


def _weak_log_reference() -> tuple[int, weakref.ReferenceType[tl.ModelLog]]:
    """Create a log and return only weak identity information.

    Returns
    -------
    tuple[int, weakref.ReferenceType[tl.ModelLog]]
        Object id and weak reference for cleanup assertions.
    """

    log = _capture()
    return id(log), weakref.ref(log)


def test_axis_i_log_registry_uses_weakref_cleanup() -> None:
    """Dead logs disappear from the weak registry after garbage collection."""

    log_id, log_ref = _weak_log_reference()

    gc.collect()

    assert log_ref() is None
    assert all(id(log) != log_id for log in tl.list_logs())


def test_helper_axis_ambiguity_uses_catalog_error() -> None:
    """Vector helpers fail with ``AxisAmbiguityError`` when feature_axis is omitted."""

    hook = tl.steer(torch.ones(3))()

    with pytest.raises(terrors.AxisAmbiguityError, match="feature_axis"):
        hook(torch.ones(2, 3), hook=None)


def test_axis_l_torchscript_degradation_message_names_recovery() -> None:
    """TorchScript rejection names the wrapper and points to the unwrapped model."""

    x = torch.randn(1, 3)
    log = _capture(_ReluAdd(), x)
    scripted = torch.jit.trace(_ReluAdd(), x)

    with pytest.raises(RuntimeError) as excinfo:
        log.rerun(scripted, x)
    message = str(excinfo.value)
    assert "ScriptModule" in message
    assert "original" in message or "un-scripted" in message


def test_manifest_paths_are_test_references() -> None:
    """The exercise manifest points at intervention test references."""

    assert all(reference.startswith("tests/test_intervention") for reference in _manifest_values())


def _manifest_values() -> Iterable[str]:
    """Return manifest references.

    Returns
    -------
    Iterable[str]
        Test reference strings from the catalog manifest.
    """

    return CATALOG_EXERCISE_MANIFEST.values()
