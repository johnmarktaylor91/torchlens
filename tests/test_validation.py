"""Tests for the validation subpackage.

Covers: import paths, registry consistency, perturbation unit tests,
deep clone helpers, and integration tests through specific exemption paths.
"""

from collections import namedtuple
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import Trace, trace as trace_fn
from torchlens.validation import validate_forward_pass, validate_saved_outs
import torchlens.user_funcs as user_funcs
from torchlens.errors import MetadataInvariantError
from torchlens.fastlog import RecordContext
from torchlens.options import SaveOptions
from torchlens.validation import check_metadata_invariants
from torchlens.intervention.types import DictKey
from torchlens.validation.invariants import check_func_call_id_invariant
from torchlens.validation import validate_saved_outs as validate_from_subpkg
from torchlens.validation.exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    SKIP_PERTURBATION_ENTIRELY,
    STRUCTURAL_ARG_POSITIONS,
    CUSTOM_EXEMPTION_CHECKS,
)
from torchlens.validation.core import (
    _perturb_layer_outs,
    _deep_clone_tensors,
    _copy_validation_args,
    _execute_func_with_restored_state,
    _restore_live_parameter_args_for_replay,
    MAX_PERTURB_ATTEMPTS,
)
from torchlens.validation.status import (
    REGION_REPLAY_CLASS,
    REGION_REPLAY_CLASS_KEY,
    REGION_REPLAY_IMPORTER_PROVENANCE,
    REGION_REPLAY_PROVENANCE_KEY,
    ValidationReplayStatus,
)
from torchlens.utils.tensor_utils import tensor_nanequal

_TEST_FORWARD_GLOBAL_TENSOR: torch.Tensor | None = None
_TEST_FORWARD_GLOBAL_PAYLOAD: dict[str, torch.Tensor] | None = None


class _StaleLabelConstantModel(nn.Module):
    """Model that consumes an unregistered tensor carrying a stale label."""

    def __init__(self) -> None:
        """Initialize a constant tensor that is not a registered buffer."""

        super().__init__()
        self.constant = torch.ones(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the unregistered constant to the input.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        return x + self.constant


_TensorLiteralReturn = namedtuple("_TensorLiteralReturn", ("values", "valid"))


class _TensorLiteralOutputModel(nn.Module):
    """Model returning a tensor plus literal metadata in one container."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Return a namedtuple-style mixed output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, bool]
            Tensor output and literal validity flag.
        """

        return _TensorLiteralReturn(values=x + 1, valid=False)


# =============================================================================
# Import / binding tests
# =============================================================================


def test_validation_import_path():
    """from torchlens.validation import validate_saved_outs works."""
    assert callable(validate_from_subpkg)


def test_validation_replay_unverified_status_semantics() -> None:
    """Unverified replay status is available but never bool-coerced."""

    status = ValidationReplayStatus.unverified(
        backend="jax",
        source="live",
        reason="region_not_per_op_replayable",
        message="synthetic partial replay",
        replayed_node_count=3,
        unverified_node_count=2,
    )

    assert status.state == "unverified"
    assert status.available is True
    assert status.replayed_node_count == 3
    assert status.unverified_node_count == 2
    assert status.failed_node_count == 0
    with pytest.raises(TypeError, match="not a boolean"):
        bool(status)


def test_validation_replay_status_aggregate_fold() -> None:
    """Aggregate replay status keeps failures dominant over unverified regions."""

    mixed_status = ValidationReplayStatus.from_replay_counts(
        backend="jax",
        source="live",
        replayed_node_count=4,
        unverified_node_count=1,
    )
    failed_status = ValidationReplayStatus.from_replay_counts(
        backend="jax",
        source="live",
        replayed_node_count=0,
        unverified_node_count=3,
        failed_node_count=2,
    )
    passed_status = ValidationReplayStatus.from_replay_counts(
        backend="jax",
        source="live",
        replayed_node_count=4,
        unverified_node_count=0,
    )

    assert mixed_status.state == "unverified"
    assert mixed_status.replayed_node_count == 4
    assert mixed_status.unverified_node_count == 1
    assert failed_status.state == "failed"
    assert failed_status.failed_node_count == 2
    assert bool(failed_status) is False
    assert passed_status.state == "passed"
    assert bool(passed_status) is True


@pytest.mark.smoke
def test_validate_forward_pass_importable():
    """validate_forward_pass is importable from torchlens top-level."""
    assert callable(validate_forward_pass)


def test_validation_replay_restores_unchanged_live_parameter_args() -> None:
    """Replay args use live parameters only when saved snapshots still match."""

    live_param = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    saved_parent = live_param.detach().clone()
    saved_param = live_param.detach().clone()
    input_args = {"args": [saved_parent, saved_param], "kwargs": {}}
    layer = SimpleNamespace(
        is_inplace=False,
        _param_logs=[SimpleNamespace(handle=live_param)],
        parent_arg_positions={"args": {0: "input_1"}, "kwargs": {}},
    )

    _restore_live_parameter_args_for_replay(input_args, cast(Any, layer))

    assert input_args["args"][0] is saved_parent
    assert input_args["args"][1] is live_param

    stale_snapshot = live_param.detach().clone()
    with torch.no_grad():
        live_param.add_(1.0)
    stale_args = {"args": [stale_snapshot], "kwargs": {}}
    stale_layer = SimpleNamespace(
        is_inplace=False,
        _param_logs=[SimpleNamespace(handle=live_param)],
        parent_arg_positions={"args": {}, "kwargs": {}},
    )

    _restore_live_parameter_args_for_replay(stale_args, cast(Any, stale_layer))

    assert stale_args["args"][0] is stale_snapshot


def test_trace_clears_nested_cached_tensor_labels_between_sessions() -> None:
    """A second trace should not see stale labels on nested model-owned tensors."""

    class TensorCache:
        """Small model-owned cache that stores tensors below a custom object."""

        def __init__(self) -> None:
            """Initialize an empty tensor cache."""

            self.items: dict[str, torch.Tensor | None] = {"mask": None}

    class CachedTensorBlock(nn.Module):
        """Consume a cached tensor in a child module."""

        def forward(self, cached: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Add the cached tensor to the live input.

            Parameters
            ----------
            cached
                Tensor cached on the parent module during an earlier trace.
            x
                Live model input.

            Returns
            -------
            torch.Tensor
                Elementwise sum of the cached tensor and input.
            """

            return cached + x

    class NestedCachedTensorModel(nn.Module):
        """Model that keeps an op output in a nested custom cache."""

        def __init__(self) -> None:
            """Initialize the nested cache and child module."""

            super().__init__()
            self.cache = TensorCache()
            self.block = CachedTensorBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the model while reusing a nested cached tensor.

            Parameters
            ----------
            x
                Model input.

            Returns
            -------
            torch.Tensor
                Child-module output.
            """

            if self.cache.items["mask"] is None:
                self.cache.items["mask"] = torch.ones_like(x)
            return self.block(cast(torch.Tensor, self.cache.items["mask"]), x)

    model = NestedCachedTensorModel()
    x = torch.randn(2, 3)

    tl.trace(model, x, save=None, layers_to_save=None, inference_only=True)
    second_trace = tl.trace(model, x, save=None, layers_to_save=None, inference_only=True)

    assert second_trace.num_ops > 0


def test_trace_clears_forward_global_tensor_labels_between_sessions() -> None:
    """A second trace should not see stale labels on forward-global tensors."""

    global _TEST_FORWARD_GLOBAL_TENSOR

    class GlobalTensorBlock(nn.Module):
        """Consume a tensor read from the caller module's globals."""

        def forward(self, cached: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            """Add a global tensor to a live input.

            Parameters
            ----------
            cached
                Tensor captured through the parent forward function's globals.
            x
                Live model input.

            Returns
            -------
            torch.Tensor
                Elementwise sum of the cached tensor and input.
            """

            return cached + x

    class GlobalTensorForwardModel(nn.Module):
        """Model whose forward reads a tensor from function globals."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.block = GlobalTensorBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the model with a global tensor input to the child module.

            Parameters
            ----------
            x
                Model input.

            Returns
            -------
            torch.Tensor
                Child-module output.
            """

            return self.block(cast(torch.Tensor, _TEST_FORWARD_GLOBAL_TENSOR), x)

    x = torch.randn(2, 3)
    _TEST_FORWARD_GLOBAL_TENSOR = torch.ones_like(x)
    try:
        model = GlobalTensorForwardModel()

        tl.trace(model, x, save=None, layers_to_save=None, inference_only=True)
        second_trace = tl.trace(model, x, save=None, layers_to_save=None, inference_only=True)

        assert second_trace.num_ops > 0
    finally:
        _TEST_FORWARD_GLOBAL_TENSOR = None


def test_trace_clears_forward_global_container_tensor_labels_between_sessions() -> None:
    """A second trace should not see stale labels inside forward-global containers."""

    global _TEST_FORWARD_GLOBAL_PAYLOAD

    class GlobalPayloadForwardModel(nn.Module):
        """Model whose forward reads a tensor from a global container."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the model with a tensor stored in a global dict.

            Parameters
            ----------
            x
                Model input.

            Returns
            -------
            torch.Tensor
                Elementwise sum of the cached global tensor and input.
            """

            assert _TEST_FORWARD_GLOBAL_PAYLOAD is not None
            return x + _TEST_FORWARD_GLOBAL_PAYLOAD["value"]

    x = torch.randn(2, 3)
    _TEST_FORWARD_GLOBAL_PAYLOAD = {"value": torch.ones_like(x)}
    try:
        model = GlobalPayloadForwardModel()

        assert validate_forward_pass(model, x, validate_metadata=True) is True
        second_trace = tl.trace(model, x, save=None, layers_to_save=None, inference_only=True)

        assert second_trace.num_ops > 0
    finally:
        _TEST_FORWARD_GLOBAL_PAYLOAD = None


def test_validate_forward_pass_replays_tuple_output_identity_leaf() -> None:
    """Output identity replay should not index into an already-selected tensor leaf."""

    class TupleChunkOutputModel(nn.Module):
        """Return a tuple whose first element is a multi-output op leaf."""

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return two tensors in a tuple.

            Parameters
            ----------
            x
                Model input.

            Returns
            -------
            tuple[torch.Tensor, torch.Tensor]
                First chunk and a derived tensor.
            """

            first, _second = x.chunk(2, dim=1)
            return first, x + 1

    model = TupleChunkOutputModel()
    x = torch.randn(1, 4, 1)

    assert tl.validate_forward_pass(model, x, validate_metadata=True) is True


def test_validate_forward_pass_replays_dict_output_by_typed_path() -> None:
    """Validation replay indexes dict-returning outputs by typed container path."""

    class DictOutputBlock(nn.Module):
        """Return a dict of tensors from a submodule."""

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            """Run the block.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            dict[str, torch.Tensor]
                Dict output consumed by the parent module.
            """

            return {"hidden": x.relu(), "logits": x.sigmoid()}

    class DictOutputModel(nn.Module):
        """Consume one tensor from a dict-returning child module."""

        def __init__(self) -> None:
            """Initialize the model."""

            super().__init__()
            self.block = DictOutputBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the model.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Tensor selected from a dict output.
            """

            output = self.block(x)
            return output["logits"] * 2

    assert validate_forward_pass(DictOutputModel(), torch.tensor([-1.0, 2.0]))


def test_validation_recompute_selects_dict_output_by_container_path() -> None:
    """Validation replay selects a dict leaf by typed path, not raw integer index."""

    def return_dict(x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return a dict with two tensor leaves.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Dict output.
        """

        return {"hidden": x.relu(), "logits": x.sigmoid()}

    layer = SimpleNamespace(
        func=return_dict,
        func_rng_states=None,
        func_autocast_state=None,
        multi_output_index=0,
        container_path=(DictKey("logits"),),
    )

    recomputed = _execute_func_with_restored_state(
        layer=layer,
        input_args={"args": [torch.tensor([-1.0, 2.0])], "kwargs": {}},
        layers_to_perturb=[],
        layer_label="dict_output",
        verbose=False,
    )

    assert torch.allclose(recomputed, torch.tensor([-1.0, 2.0]).sigmoid())


def test_validate_forward_pass_output_aliasing_a_reassigned_buffer():
    """Regression: a model that RETURNS a registered buffer it reassigned must validate.

    Previously a false-negative: validate_forward_pass saved the ground-truth output by
    reference, then restored state_dict; load_state_dict writes buffers in-place, clobbering
    the saved ground-truth (which aliased the returned buffer) back to its initial value, so
    the (correct) traced output was compared against a corrupted zero tensor. Capture/replay
    were always correct; the validator now snapshots ground-truth outputs before the restore.
    """

    class RecurrentStateBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("h", torch.zeros(3))
            self.lin = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(4):
                h = cast(torch.Tensor, self._buffers["h"])
                self.h = torch.tanh(self.lin(x) + h)
            return cast(torch.Tensor, self._buffers["h"])

    torch.manual_seed(0)
    assert validate_forward_pass(RecurrentStateBuffer(), torch.randn(3)) is True

    class ReassignReturn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("b", torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.b = cast(torch.Tensor, self._buffers["b"]) + x
            return cast(torch.Tensor, self._buffers["b"])

    assert validate_forward_pass(ReassignReturn(), torch.randn(3)) is True


def test_validate_forward_pass_plain_attribute_mutable_state_isolated() -> None:
    """Plain attribute-held mutable tensors should not poison the traced run."""

    class PlainMutableState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = [torch.zeros(3)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.state[0].add_(1)
            return x + self.state[0]

    assert validate_forward_pass(PlainMutableState(), torch.randn(3)) is True


def test_validate_forward_pass_deepcopy_fallback_warns_for_registered_state() -> None:
    """Un-deepcopyable models fall back to state_dict restore with warning."""

    class UndeepcopyableRegisteredState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buf", torch.zeros(3))

        def __getstate__(self) -> dict[str, object]:
            """Make deepcopy fail like modules holding uncopyable resources."""

            raise TypeError("cannot deepcopy handle")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            buf = cast(torch.Tensor, self.buf)
            buf.add_(1)
            return x + buf

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert validate_forward_pass(UndeepcopyableRegisteredState(), torch.randn(3)) is True


def test_validate_forward_pass_deepcopy_fallback_restores_plain_attrs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback validation restores non-registered attr mutations before logging."""

    original_deepcopy = user_funcs.copy.deepcopy

    def fail_module_deepcopy(value: object) -> object:
        """Fail only model deepcopy so the validation fallback path is exercised."""

        if isinstance(value, nn.Module):
            raise TypeError("forced module deepcopy failure")
        return original_deepcopy(value)

    class StepCounterModel(nn.Module):
        """Model whose output depends on a plain Python step counter."""

        def __init__(self) -> None:
            """Initialize mutable plain state."""

            super().__init__()
            self.step = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Increment plain state and use it in the output."""

            self.step += 1
            return x + float(self.step)

    monkeypatch.setattr(user_funcs.copy, "deepcopy", fail_module_deepcopy)

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert validate_forward_pass(StepCounterModel(), torch.randn(3)) is True


def test_validate_forward_pass_deepcopy_fallback_tracks_function_attrs_by_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback validation accepts function attrs and detects reassignment by identity."""

    original_deepcopy = user_funcs.copy.deepcopy

    def fail_module_deepcopy(value: object) -> object:
        """Fail only model deepcopy so the validation fallback path is exercised."""

        if isinstance(value, nn.Module):
            raise TypeError("forced module deepcopy failure")
        return original_deepcopy(value)

    def identity_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Return the module output for a function-typed plain attribute."""

        del module
        return x + 1

    class FunctionAttrModel(nn.Module):
        """Model with a plain function attribute that deepcopy fallback snapshots."""

        def __init__(self) -> None:
            """Initialize the function-typed plain attribute."""

            super().__init__()
            self.forward_impl = identity_forward

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the function-typed plain attribute."""

            return self.forward_impl(self, x)

    monkeypatch.setattr(user_funcs.copy, "deepcopy", fail_module_deepcopy)

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert validate_forward_pass(FunctionAttrModel(), torch.randn(3)) is True


def test_validate_forward_pass_preserves_distinct_recurrent_output_labels() -> None:
    """Multi-output recurrent producers should validate against distinct outputs."""

    class SharedHeadTuple(nn.Module):
        """Return multiple outputs produced by repeated calls to one module."""

        def __init__(self) -> None:
            """Initialize the shared head."""

            super().__init__()
            self.head = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Run the shared head twice and return both tensors."""

            first = self.head(x)
            second = self.head(x + 1)
            return first, second

    assert validate_forward_pass(SharedHeadTuple(), torch.randn(2, 3)) is True


def test_validate_forward_pass_deepcopy_fallback_tripwire_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback plain-attr restore does not hide genuine logged-output breaks."""

    original_deepcopy = user_funcs.copy.deepcopy
    original_run = user_funcs._run_model_and_save_specified_outs

    def fail_module_deepcopy(value: object) -> object:
        """Fail only model deepcopy so the validation fallback path is exercised."""

        if isinstance(value, nn.Module):
            raise TypeError("forced module deepcopy failure")
        return original_deepcopy(value)

    def corrupt_logged_output(*args: object, **kwargs: object) -> Trace:
        """Corrupt the captured output to simulate a real capture break."""

        trace = original_run(*args, **kwargs)
        output_layer = trace[trace.output_layers[0]]
        output_layer.out = cast(torch.Tensor, output_layer.out) + 1.0
        return trace

    class StepCounterModel(nn.Module):
        """Model whose fallback path would false-alarm without plain restore."""

        def __init__(self) -> None:
            """Initialize mutable plain state."""

            super().__init__()
            self.step = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Increment plain state and use it in the output."""

            self.step += 1
            return x + float(self.step)

    monkeypatch.setattr(user_funcs.copy, "deepcopy", fail_module_deepcopy)
    monkeypatch.setattr(user_funcs, "_run_model_and_save_specified_outs", corrupt_logged_output)

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert validate_forward_pass(StepCounterModel(), torch.randn(3)) is False


def test_validate_forward_pass_deepcopy_fallback_restore_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback validation fails loudly if plain attr restore itself fails."""

    original_deepcopy = user_funcs.copy.deepcopy

    def fail_module_deepcopy(value: object) -> object:
        """Fail only model deepcopy so the validation fallback path is exercised."""

        if isinstance(value, nn.Module):
            raise TypeError("forced module deepcopy failure")
        return original_deepcopy(value)

    class RestoreBlockedModel(nn.Module):
        """Model that refuses normal assignment during the restore step."""

        def __init__(self) -> None:
            """Initialize mutable plain state and restore guard."""

            super().__init__()
            self._block_step_restore = False
            self.step = 0
            self._block_step_restore = True

        def __setattr__(self, name: str, value: object) -> None:
            """Block ``step`` assignment once validation tries to restore it."""

            if name == "step" and getattr(self, "_block_step_restore", False):
                raise RuntimeError("step restore blocked")
            super().__setattr__(name, value)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Mutate ``step`` without using normal assignment."""

            object.__setattr__(self, "step", self.step + 1)
            return x + float(self.step)

    monkeypatch.setattr(user_funcs.copy, "deepcopy", fail_module_deepcopy)

    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        with pytest.raises(RuntimeError, match="could not restore plain attribute"):
            validate_forward_pass(RestoreBlockedModel(), torch.randn(3))


def test_validation_tripwire_still_fails_on_wrong_output() -> None:
    """Trace validation still rejects genuinely wrong ground-truth outputs."""

    model = nn.Sequential(nn.Linear(3, 3), nn.ReLU()).eval()
    x = torch.randn(2, 3)
    trace = trace_fn(model, x)
    wrong_output = [trace[trace.output_layers[0]].out + 1]

    try:
        assert trace.validate_forward_pass(wrong_output, validate_metadata=False) is False
    finally:
        trace.cleanup()


def test_check_metadata_invariants_importable():
    """check_metadata_invariants and MetadataInvariantError importable from top-level."""
    assert callable(check_metadata_invariants)
    assert issubclass(MetadataInvariantError, ValueError)


def test_trace_validate_method_bound():
    """Trace.validate_saved_outs is callable."""
    assert hasattr(Trace, "validate_saved_outs")
    assert callable(Trace.validate_saved_outs)


def test_trace_check_metadata_method_bound():
    """Trace.check_metadata_invariants is callable."""
    assert hasattr(Trace, "check_metadata_invariants")
    assert callable(Trace.check_metadata_invariants)


def test_tensor_nanequal_uses_relative_tolerance_for_replay() -> None:
    """Validation replay should allow tiny relative floating-point drift."""
    saved = torch.tensor([1.0, 100.0], dtype=torch.float32)
    replayed = saved + torch.tensor([8e-6, 4e-3], dtype=torch.float32)
    mismatched = saved + torch.tensor([1e-3, 1e-1], dtype=torch.float32)

    assert not tensor_nanequal(saved, replayed, allow_tolerance=False)
    assert tensor_nanequal(saved, replayed, allow_tolerance=True)
    assert not tensor_nanequal(saved, mismatched, allow_tolerance=True)


# =============================================================================
# Registry consistency tests
# =============================================================================


def test_skip_validation_entirely_are_strings():
    assert len(SKIP_VALIDATION_ENTIRELY) > 0
    for entry in SKIP_VALIDATION_ENTIRELY:
        assert isinstance(entry, str) and len(entry) > 0


def test_skip_perturbation_entirely_are_strings():
    assert len(SKIP_PERTURBATION_ENTIRELY) > 0
    for entry in SKIP_PERTURBATION_ENTIRELY:
        assert isinstance(entry, str) and len(entry) > 0


def test_structural_arg_positions_values_are_sets_of_ints():
    for func_name, positions in STRUCTURAL_ARG_POSITIONS.items():
        assert isinstance(func_name, str) and len(func_name) > 0
        assert isinstance(positions, set)
        for pos in positions:
            assert isinstance(pos, int) and pos >= 0


def test_custom_exemption_checks_are_callable():
    for func_name, check_fn in CUSTOM_EXEMPTION_CHECKS.items():
        assert isinstance(func_name, str) and len(func_name) > 0
        assert callable(check_fn)


# =============================================================================
# Perturbation unit tests
# =============================================================================


@pytest.mark.smoke
def test_perturbation_changes_float_tensor():
    parent = torch.randn(10, 10)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.shape == parent.shape


def test_perturbation_changes_int_tensor():
    parent = torch.randint(0, 100, (10, 10))
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == parent.dtype


def test_perturbation_changes_bool_tensor():
    parent = torch.ones(10, 10, dtype=torch.bool)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    # With 100 elements all True, random should differ
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.bool


def test_perturbation_changes_complex_tensor():
    parent = torch.complex(torch.randn(5, 5), torch.randn(5, 5))
    output = torch.randn(5, 5)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.is_complex()


def test_perturbation_respects_dtype():
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.bool]:
        if dtype in (torch.int32, torch.int64):
            parent = torch.randint(0, 100, (5, 5), dtype=dtype)
        elif dtype == torch.bool:
            parent = torch.ones(5, 5, dtype=torch.bool)
        else:
            parent = torch.randn(5, 5, dtype=dtype)
        output = torch.randn(5, 5)
        perturbed = _perturb_layer_outs(parent, output)
        assert perturbed.dtype == dtype


def test_perturbation_handles_empty_tensor():
    parent = torch.tensor([])
    output = torch.tensor([])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.numel() == 0
    assert torch.equal(perturbed, parent)


def test_perturbation_terminates_on_scalar():
    """MAX_PERTURB_ATTEMPTS guard prevents infinite loop on single-element tensors."""
    # Single-element bool tensor: 50% chance each attempt matches original.
    # With MAX_PERTURB_ATTEMPTS=100, it should terminate regardless.
    parent = torch.tensor([True])
    output = torch.tensor([1.0])
    perturbed = _perturb_layer_outs(parent, output)
    assert perturbed.dtype == torch.bool
    assert perturbed.shape == parent.shape


# =============================================================================
# Deep clone tests
# =============================================================================


def test_deep_clone_nested_list_of_tensors():
    original = [torch.tensor([1.0, 2.0]), [torch.tensor([3.0]), torch.tensor([4.0])]]
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, list)
    assert isinstance(cloned[1], list)
    assert torch.equal(cloned[0], original[0])
    assert torch.equal(cloned[1][0], original[1][0])


def test_deep_clone_nested_dict_of_tensors():
    original = {"a": torch.tensor([1.0]), "b": {"c": torch.tensor([2.0])}}
    cloned = _deep_clone_tensors(original)
    assert isinstance(cloned, dict)
    assert isinstance(cloned["b"], dict)
    assert torch.equal(cloned["a"], original["a"])
    assert torch.equal(cloned["b"]["c"], original["b"]["c"])


def test_deep_clone_independence():
    """Modifying clone doesn't affect original."""
    original = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    cloned = _deep_clone_tensors(original)
    cloned[0][0] = 999.0
    assert original[0][0].item() == 1.0


def test_deep_clone_preserves_non_tensors():
    original = [42, "hello", None, (1, 2)]
    cloned = _deep_clone_tensors(original)
    assert cloned == original


def test_copy_validation_args():
    """_copy_validation_args deep-clones tensors in args and kwargs."""
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0])
    input_args = {
        "args": [t1, [t2, 42]],
        "kwargs": {"key": torch.tensor([5.0])},
    }
    copied = _copy_validation_args(input_args)

    # Independence
    copied["args"][0][0] = 999.0
    assert t1[0].item() == 1.0

    copied["kwargs"]["key"][0] = 999.0
    assert input_args["kwargs"]["key"][0].item() == 5.0


# =============================================================================
# Integration tests — validate full pipeline through specific exemption paths
# =============================================================================


class _GetItemTensorIndex(nn.Module):
    """Model that uses tensor indexing (__getitem__ with a tensor index)."""

    def forward(self, x):
        idx = torch.tensor([0, 2, 1])
        return x[idx]


class _ScatterModel(nn.Module):
    """Model that uses scatter_."""

    def forward(self, x):
        src = torch.ones(3, 5)
        index = torch.tensor([[0, 1, 2, 0, 0], [2, 0, 0, 1, 2], [0, 0, 1, 2, 0]])
        out = torch.zeros(3, 5)
        out.scatter_(1, index, src)
        return x + out


class _MaskedFillModel(nn.Module):
    """Model that uses masked_fill_."""

    def forward(self, x):
        mask = x > 0.5
        return x.masked_fill_(mask, 0.0)


class _FunctionalMaskedFillModel(nn.Module):
    """Model that uses masked_fill."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a boolean mask through the non-in-place masked_fill method."""
        mask = x > 0.5
        return x.masked_fill(mask, 0.0)


class _ZerosLikeModel(nn.Module):
    """Model that uses zeros_like."""

    def forward(self, x):
        z = torch.zeros_like(x)
        return x + z


class _NewTensorFactoryModel(nn.Module):
    """Model that uses the input tensor as a ``new_tensor`` factory."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a constant tensor using ``x`` only as a factory template.

        Parameters
        ----------
        x:
            Input tensor providing dtype, device, and layout metadata.

        Returns
        -------
        torch.Tensor
            Input shifted by a constant factory-created scalar.
        """

        values = x.new_tensor([1.0, 2.0, 3.0])
        return x + values.sum()


class _RemainderDividendBelowDivisorModel(nn.Module):
    """Model where the remainder divisor is locally value-invariant."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a remainder whose output is exactly the dividend.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Remainder output plus the input to keep both parents live.
        """

        dividend = torch.sigmoid(x)
        divisor = torch.full_like(dividend, 2.0)
        return torch.remainder(dividend, divisor) + x


class _EmptyLikeModel(nn.Module):
    """Model that uses empty_like (tests SKIP_VALIDATION_ENTIRELY)."""

    def forward(self, x):
        # empty_like output is nondeterministic — don't use it in computation
        _ = torch.empty_like(x)
        return x * 2


def test_validation_with_getitem_tensor_index():
    model = _GetItemTensorIndex()
    x = torch.randn(5, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_scatter():
    model = _ScatterModel()
    x = torch.randn(3, 5)
    assert validate_forward_pass(model, x)


def test_validation_with_masked_fill():
    model = _MaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_validation_with_functional_masked_fill() -> None:
    """Validate non-in-place masked_fill boolean masks as structural args."""
    model = _FunctionalMaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_save_arg_values_keeps_inplace_alias_contract_versions() -> None:
    """save_arg_values=True keeps alias-contract snapshots for in-place ops."""

    class InplaceModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run an in-place mutation after producing an intermediate."""

            y = x + 1
            y.relu_()
            return y * 2

    trace = trace_fn(InplaceModel(), torch.tensor([-2.0, 3.0]), save_arg_values=True)

    assert torch.equal(
        trace["add_1_1"].out_versions_by_child["relu_1_2"],
        torch.tensor([-1.0, 4.0]),
    )


def test_validation_with_zeros_like():
    model = _ZerosLikeModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_new_tensor_factory_source() -> None:
    """Perturbation skips the structural source tensor for ``new_tensor``."""

    model = _NewTensorFactoryModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_remainder_dividend_below_divisor() -> None:
    """Perturbation skips a divisor that is locally irrelevant to remainder."""

    model = _RemainderDividendBelowDivisorModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_empty_like():
    model = _EmptyLikeModel()
    x = torch.randn(3, 3)
    assert validate_forward_pass(model, x)


# =============================================================================
# Metadata invariant tests — standalone + corruption
# =============================================================================


class _SimpleFF(nn.Module):
    """Single-module feed-forward model for validation invariant tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.fc = nn.Linear(5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one attributed linear operation."""

        return self.fc(x)


class _FirstInputModel(nn.Module):
    """Return only the first input so the second input can remain unsaved."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the first input tensor."""

        return x


def _save_first_input_record(ctx: RecordContext) -> bool:
    """Select only first-input records for selective-save validation tests."""

    return ctx.label.startswith("input_1")


class _RootOnlyModel(nn.Module):
    """Root-only model with compute ops outside child modules."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple top-level tensor operation."""

        return x * 2


class _MidForwardGradModel(nn.Module):
    """Model that triggers ``autograd.grad`` before all forward ops exist."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.fc1 = nn.Linear(5, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a MAML-style forward with an inner differentiable grad."""

        hidden = torch.relu(self.fc1(x))
        inner_loss = hidden.square().mean()
        (weight_grad,) = torch.autograd.grad(
            inner_loss,
            self.fc1.weight,
            create_graph=True,
            retain_graph=True,
        )
        adapted_bias = weight_grad.mean()
        return self.fc2(hidden + adapted_bias)


class _BufferWriteValidationModel(nn.Module):
    """Model with one registered buffer read and write."""

    def __init__(self) -> None:
        """Initialize the registered buffer."""

        super().__init__()
        self.register_buffer("state", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read, mutate, and consume a registered buffer."""

        y = self.state + x
        self.state.copy_(y)
        return self.state * x


class _FuncCallSplitValidationModel(nn.Module):
    """Model with a multi-output torch call in plain capture."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split and recombine two tensor outputs."""

        left, right = torch.split(x, 1, dim=0)
        return left + right


def _make_clean_log() -> Trace:
    """Return a Trace with all outs and metadata for a simple FF model."""
    from torchlens import trace as trace_fn

    model = _SimpleFF()
    return trace_fn(model, torch.randn(2, 5), random_seed=42)


def _make_buffer_write_log() -> Trace:
    """Return a trace with static and write buffer versions."""

    return trace_fn(_BufferWriteValidationModel(), torch.ones(2), save_arg_values=True)


def _make_func_call_split_log() -> Trace:
    """Return a plain trace with a populated multi-output func_call_id group."""

    return trace_fn(_FuncCallSplitValidationModel(), torch.randn(2, 3), random_seed=42)


def _make_root_only_log() -> Trace:
    """Return a Trace for a model whose compute op is attributed only to root."""
    from torchlens import trace as trace_fn

    model = _RootOnlyModel()
    return trace_fn(model, torch.randn(2, 5), random_seed=42)


def _relabel_as_mlx_object_module(trace: Trace) -> Trace:
    """Relabel a torch trace as an MLX-style non-function-root trace.

    Parameters
    ----------
    trace:
        Source trace with torch-module metadata.

    Returns
    -------
    Trace
        Same trace object routed through backend-neutral object_module validation.
    """

    trace.backend = "mlx"
    trace.module_identity_mode = "object_module"
    trace.param_source = "native-module"
    return trace


def _relabel_as_function_root(trace: Trace, backend: str) -> Trace:
    """Relabel a root-only trace as a synthetic function-root backend trace.

    Parameters
    ----------
    trace:
        Root-only source trace.
    backend:
        Backend name whose spec supports ``function_root``.

    Returns
    -------
    Trace
        Same trace object routed through backend-neutral function_root validation.
    """

    trace.backend = backend
    trace.module_identity_mode = "function_root"
    trace.param_source = "none"
    return trace


def _make_backward_log() -> Trace:
    """Return a Trace with backward metadata for a simple FF model."""
    from torchlens import trace as trace_fn

    model = _SimpleFF()
    x = torch.randn(2, 5, requires_grad=True)
    log = trace_fn(model, x, save_grads="all", random_seed=42)
    log.log_backward(log[log.output_layers[0]].out.sum())
    return log


def _make_mid_forward_backward_log() -> Trace:
    """Return a Trace with a backward trigger that occurs mid-forward."""
    from torchlens import trace as trace_fn

    model = _MidForwardGradModel()
    x = torch.randn(2, 5, requires_grad=True)
    return trace_fn(model, x, save_grads="all", random_seed=42)


def test_clean_log_ops_all_invariants():
    """An uncorrupted Trace ops all invariant checks."""
    log = _make_clean_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_region_replay_annotation_on_plain_capture_fails_invariants() -> None:
    """Plain captures may not launder replay regions without importer provenance."""

    log = _make_clean_log()
    try:
        op = next(layer for layer in log.layer_list if not layer.is_input and not layer.is_output)
        op.annotations[REGION_REPLAY_CLASS_KEY] = REGION_REPLAY_CLASS

        with pytest.raises(MetadataInvariantError, match="region_replay_provenance"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_region_replay_annotation_with_importer_provenance_passes_tripwire() -> None:
    """Importer-owned synthetic region annotations satisfy the provenance invariant."""

    log = _make_clean_log()
    try:
        op = next(layer for layer in log.layer_list if not layer.is_input and not layer.is_output)
        log.annotations[REGION_REPLAY_PROVENANCE_KEY] = REGION_REPLAY_IMPORTER_PROVENANCE
        op.annotations[REGION_REPLAY_CLASS_KEY] = REGION_REPLAY_CLASS
        op.annotations[REGION_REPLAY_PROVENANCE_KEY] = REGION_REPLAY_IMPORTER_PROVENANCE

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_backend_identity_invariants_reject_torch_param_source_none_with_params() -> None:
    """Torch identity invariants reject ``param_source='none'`` with params present."""

    log = _make_clean_log()
    try:
        assert log.backend == "torch"
        assert log.module_identity_mode == "torch_module"
        assert log.param_source == "native-module"
        assert log.num_param_tensors > 0

        log.param_source = "none"

        with pytest.raises(MetadataInvariantError, match="param_source='none'"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backend_identity_invariants_pass_healthy_resnet() -> None:
    """Backend identity preconditions fire on a healthy torch ResNet trace."""

    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.resnet18(weights=None).eval()
    log = trace_fn(model, torch.randn(1, 3, 32, 32), random_seed=42)
    try:
        assert log.backend == "torch"
        assert log.module_identity_mode == "torch_module"
        assert log.param_source == "native-module"
        assert log.num_param_tensors > 0
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_backend_neutral_accessor_refs_reject_invalid_torch_status() -> None:
    """Structural neutral accessor checks reject bad torch resolver status."""

    log = _make_clean_log()
    try:
        victim = next(layer for layer in log.layer_list if not layer.is_input)
        assert victim.resolver_status == "resolved"

        victim.resolver_status = "bogus"

        with pytest.raises(MetadataInvariantError, match="invalid resolver_status"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backend_neutral_accessor_refs_reject_malformed_torch_dtype_ref() -> None:
    """Structural neutral accessor checks reject refs missing a name."""

    log = _make_clean_log()
    try:
        victim = next(layer for layer in log.layer_list if not layer.is_input)
        assert victim.dtype_ref is not None

        victim.dtype_ref = SimpleNamespace(backend="torch", name="")

        with pytest.raises(MetadataInvariantError, match="malformed dtype_ref"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backend_neutral_accessor_refs_pass_healthy_torch_trace() -> None:
    """Structural neutral accessor preconditions fire on a healthy torch trace."""

    log = _make_clean_log()
    try:
        records = [
            *log.layer_list,
            *list(log.layer_logs.values()),
            *list(log.param_logs.values()),
        ]
        assert any(getattr(record, "resolver_status", None) == "resolved" for record in records)
        assert any(getattr(record, "dtype_ref", None) is not None for record in records)
        assert any(getattr(record, "device_ref", None) is not None for record in records)
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_buffer_static_version_invariant_rejects_empty_versions() -> None:
    """Buffer static-version contract rejects an accessor entity with no versions."""

    log = _make_buffer_write_log()
    try:
        buffer = next(iter(log.buffers))
        assert buffer.versions

        buffer.versions = []

        with pytest.raises(MetadataInvariantError, match="has no version nodes"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_buffer_write_version_invariant_rejects_sparse_pass_set() -> None:
    """Buffer write-version contract rejects non-dense buffer_pass values."""

    log = _make_buffer_write_log()
    try:
        write_version = next(
            layer for layer in log.layer_list if layer.buffer_write_kind is not None
        )
        assert write_version.buffer_pass == 2

        write_version.buffer_pass = 3

        with pytest.raises(MetadataInvariantError, match="dense as a set"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_buffer_write_version_invariant_rejects_unresolved_source() -> None:
    """Buffer write-version contract rejects populated sources that do not resolve."""

    log = _make_buffer_write_log()
    try:
        write_version = next(
            layer for layer in log.layer_list if layer.buffer_write_kind is not None
        )
        assert write_version.buffer_source is not None

        write_version.buffer_source = "missing_raw_label"

        with pytest.raises(MetadataInvariantError, match="unresolved buffer_source"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_buffer_replay_validated_invariant_rejects_missing_evidence() -> None:
    """Replay-validated contract rejects a write version without saved source args."""

    log = _make_buffer_write_log()
    try:
        write_version = next(
            layer for layer in log.layer_list if layer.buffer_write_kind is not None
        )
        assert write_version.buffer_source is not None
        assert write_version.saved_args

        write_version.buffer_replay_validated = True
        write_version.saved_args = []

        with pytest.raises(MetadataInvariantError, match="saved source argument"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_buffer_integrity_preconditions_reach_real_trace() -> None:
    """Buffer integrity contracts fire on static and write versions in a real trace."""

    log = _make_buffer_write_log()
    try:
        buffer = next(iter(log.buffers))
        assert buffer.versions
        assert any(version.buffer_write_kind is None for version in buffer.versions)
        assert any(version.buffer_write_kind is not None for version in buffer.versions)
        assert {version.buffer_pass for version in buffer.versions} == {1, 2}
        assert any(version.buffer_source is not None for version in buffer.versions)
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_edge_use_invariant_rejects_invalid_existing_record_kind() -> None:
    """Edge-use contract rejects an invalid kind on a populated edge-use record."""

    log = _make_clean_log()
    try:
        layer = next(layer for layer in log.layer_list if layer._edge_uses)
        layer._edge_uses[0] = replace(layer._edge_uses[0], edge_use="bogus")

        with pytest.raises(MetadataInvariantError, match="invalid edge_use kind"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_edge_use_invariant_rejects_missing_parent_arg_reference() -> None:
    """Parent-arg contract rejects populated entries whose labels do not resolve."""

    log = _make_clean_log()
    try:
        layer = next(layer for layer in log.layer_list if layer.parent_arg_positions["args"])
        first_position = next(iter(layer.parent_arg_positions["args"]))
        layer.parent_arg_positions["args"][first_position] = "missing_parent_label"

        with pytest.raises(MetadataInvariantError, match="references missing parent"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_edge_use_parent_arg_preconditions_reach_real_trace() -> None:
    """Edge-use and parent-arg contracts inspect populated metadata on a real trace."""

    log = _make_clean_log()
    try:
        edge_use_layers = [layer for layer in log.layer_list if layer._edge_uses]
        parent_arg_layers = [
            layer
            for layer in log.layer_list
            if layer.parent_arg_positions["args"] or layer.parent_arg_positions["kwargs"]
        ]
        assert edge_use_layers
        assert parent_arg_layers
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_edge_use_invariant_allows_buffer_source_edge_without_record() -> None:
    """Buffer-source parent edges are valid even without per-edge use records."""

    log = _make_buffer_write_log()
    try:
        write_version = next(
            layer for layer in log.layer_list if layer.buffer_write_kind is not None
        )
        assert write_version.parents
        assert not write_version._edge_uses
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_stale_internal_source_label_is_relogged_not_reused() -> None:
    """A stale tensor label must not become a missing live-index parent."""

    from torchlens.backends.torch._tl import get_tensor_label, set_tensor_label

    model = _StaleLabelConstantModel()
    set_tensor_label(model.constant, "internalsource_1_2_raw")

    log = trace_fn(model, torch.zeros(2, 2))
    try:
        raw_labels = {entry._label_raw for entry in log.layer_list}
        assert "internalsource_1_2_raw" not in raw_labels
        assert get_tensor_label(model.constant) != "internalsource_1_2_raw"
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()

    set_tensor_label(model.constant, "internalsource_1_2_raw")
    assert validate_forward_pass(model, torch.zeros(2, 2), validate_metadata=True)


def test_module_output_structure_allows_literal_metadata_fields() -> None:
    """Module output-structure paths compare tensor leaves only."""

    log = trace_fn(
        _TensorLiteralOutputModel(),
        torch.zeros(2, 2),
        capture_container_structure=True,
    )
    try:
        root_call = log.module_calls["self:1"]
        assert root_call.output_ops == ["output_1"]
        assert log[root_call.output_ops[0]].container_path
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_param_deep_xref_rejects_missing_op_back_reference() -> None:
    """Param usage cannot point at an Op that no longer lists that Param."""

    log = _make_clean_log()
    try:
        param = next(param for param in log.param_logs if param.used_by_ops)
        op = log[param.used_by_ops[0]]
        assert any(candidate.address == param.address for candidate in op._param_logs)

        op._param_logs = [
            candidate for candidate in op._param_logs if candidate.address != param.address
        ]

        with pytest.raises(MetadataInvariantError, match="does not list the Param"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_param_deep_xref_preconditions_reach_real_trace() -> None:
    """Param deep-xref checks inspect populated reciprocal links on a real trace."""

    log = _make_clean_log()
    try:
        used_params = [param for param in log.param_logs if param.num_uses_by_ops > 0]
        assert used_params
        assert any(param.used_by_layers for param in used_params)
        assert log.layers_with_params
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_param_deep_xref_allows_recurrent_weight_sharing() -> None:
    """Recurrent/shared params pass with pass-qualified uses and deduped aggregates."""

    log = _make_recurrent_log()
    try:
        recurrent_param = next(param for param in log.param_logs if param.num_uses_by_ops > 1)
        assert recurrent_param.num_uses_by_ops == len(recurrent_param.used_by_ops)
        assert len(recurrent_param.used_by_layers) == 1
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_payload_metadata_invariant_rejects_transformed_shape_mismatch() -> None:
    """Live transformed payload metadata must match the retained tensor."""

    log = trace_fn(
        _SimpleFF(),
        torch.randn(2, 5),
        save=SaveOptions(activation_transform=lambda tensor: tensor.mean()),
        random_seed=42,
    )
    try:
        victim = next(layer for layer in log.layer_list if layer.transformed_out is not None)
        assert victim.transformed_out_shape == ()

        victim.transformed_out_shape = (999,)

        with pytest.raises(MetadataInvariantError, match="payload_metadata_invariants"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_payload_metadata_invariant_rejects_present_raw_memory_mismatch() -> None:
    """Live raw payload metadata must still be checked when the payload is retained."""

    log = trace_fn(_SimpleFF(), torch.randn(2, 5), random_seed=42)
    try:
        victim = next(layer for layer in log.layer_list if layer.has_saved_activation)
        assert victim.out is not None

        victim.activation_memory = 999

        with pytest.raises(MetadataInvariantError, match="payload_metadata_invariants"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_payload_metadata_preconditions_reach_real_saved_payload_trace() -> None:
    """Payload metadata checks inspect live raw and transformed activation payloads."""

    log = trace_fn(
        _SimpleFF(),
        torch.randn(2, 5),
        save=SaveOptions(activation_transform=lambda tensor: tensor.mean()),
        random_seed=42,
    )
    try:
        assert any(layer.out is not None for layer in log.layer_list)
        assert any(layer.transformed_out is not None for layer in log.layer_list)
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_payload_metadata_invariant_allows_selective_save_validation() -> None:
    """Metadata validation must not read intentionally unsaved selective-save payloads."""

    model = _FirstInputModel()
    x = torch.randn(2, 5)
    y = torch.randn(2, 5)
    log = trace_fn(model, [x, y], save=_save_first_input_record, random_seed=42)
    try:
        assert any(not layer.has_saved_activation for layer in log.layer_list)
        assert log.validate_forward_pass([x], validate_metadata=True) is True
    finally:
        log.cleanup()


def test_payload_metadata_invariant_allows_evicted_grad_metadata() -> None:
    """Gradient metadata without a live payload is legitimate after eviction."""

    log = _make_clean_log()
    try:
        victim = next(
            layer for layer in log.layer_list if not layer.is_input and not layer.is_output
        )
        victim.grad = None
        victim.transformed_grad = None
        victim.has_grad = False
        victim.grad_shape = (2, 3)
        victim.grad_dtype = torch.float32
        victim.gradient_memory = 24

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_func_call_id_invariant_rejects_missing_id_in_plain_capture() -> None:
    """Plain capture compute ops must retain populated func_call_id metadata."""

    log = _make_clean_log()
    try:
        victim = next(
            layer for layer in log.layer_list if not layer.is_input and not layer.is_output
        )
        assert victim.func_call_id is not None

        victim.func_call_id = None

        with pytest.raises(MetadataInvariantError, match="has no func_call_id"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_func_call_id_invariant_rejects_plain_group_func_name_mismatch() -> None:
    """Plain same-call groups compare function name and container spec only."""

    log = _make_func_call_split_log()
    try:
        split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
        assert len(split_layers) == 2
        assert len({layer.func_call_id for layer in split_layers}) == 1

        split_layers[1].func_name = "chunk"

        with pytest.raises(MetadataInvariantError, match="incompatible call metadata"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_func_call_id_plain_preconditions_reach_real_trace() -> None:
    """The func_call_id check runs on a plain exhaustive multi-output trace."""

    log = _make_func_call_split_log()
    try:
        split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
        assert len(split_layers) == 2
        assert len({layer.func_call_id for layer in split_layers}) == 1
        assert all(layer.container_path for layer in split_layers)
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_func_call_id_invariant_allows_sparse_recording_projection() -> None:
    """Sparse recording projections do not require templates or container specs."""

    recording = tl.record(
        _SimpleLinear(),
        torch.randn(2, 10),
        save=tl.func("linear"),
    )
    log = recording.to_trace()
    try:
        compute_layers = [
            layer for layer in log.layer_list if not (layer.is_input or layer.is_output)
        ]
        assert compute_layers
        assert any(layer.func_call_id is not None for layer in compute_layers)
        assert all(layer.args_template is None for layer in compute_layers)
        assert all(layer.container_spec is None for layer in compute_layers)
        result = check_func_call_id_invariant(log)
        assert result.passed is True
    finally:
        log.cleanup()


def test_clean_log_ops_as_method():
    """check_metadata_invariants works as a bound method on Trace."""
    log = _make_clean_log()
    assert log.check_metadata_invariants() is True
    log.cleanup()


def test_backward_invariants_simple_mlp() -> None:
    """Backward metadata invariants pass on a clean backward trace."""

    log = _make_backward_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_backward_invariants_with_intervening() -> None:
    """Backward metadata invariants allow intervening grad_fns."""

    log = _make_backward_log()
    assert any(not grad_fn_handle.has_op for grad_fn_handle in log.grad_fn_logs.values())
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_grad_fn_topology_rejects_missing_reverse_child_link() -> None:
    """Backward GradFn parent/child links must be bidirectional."""

    log = _make_backward_log()
    try:
        parent = next(grad_fn for grad_fn in log.grad_fn_logs.values() if grad_fn.children)
        child = next(
            grad_fn for grad_fn in log.grad_fn_logs.values() if grad_fn.label == parent.children[0]
        )
        child.parents = [label for label in child.parents if label != parent.label]

        with pytest.raises(MetadataInvariantError, match="does not list"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backward_pass_domain_rejects_invalid_status() -> None:
    """BackwardPass domain fields reject values outside event enums."""

    log = _make_backward_log()
    try:
        backward_pass = next(iter(log.backward_pass_logs.values()))
        backward_pass.status = "finished"

        with pytest.raises(MetadataInvariantError, match="invalid status"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backward_pass_domain_rejects_root_mismatch() -> None:
    """BackwardPass roots must agree with trace-level roots for the pass."""

    log = _make_backward_log()
    try:
        backward_pass = next(iter(log.backward_pass_logs.values()))
        backward_pass.root_grad_fn_ids = []

        with pytest.raises(MetadataInvariantError, match="root_grad_fn_ids do not match"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_backward_phase_d_preconditions_reach_backward_trace() -> None:
    """Phase-D backward checks run on a real backward-captured trace."""

    log = _make_backward_log()
    try:
        assert log.has_backward_pass
        assert log.grad_fn_logs
        assert log.backward_pass_logs
        assert any(
            grad_fn.children or grad_fn.parents or grad_fn.next_grad_fn_ids
            for grad_fn in log.grad_fn_logs.values()
        )
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_backward_invariants_allow_only_post_trigger_missing_backpointers() -> None:
    """Mid-forward backward triggers exempt only later-created forward layers."""

    log = _make_mid_forward_backward_log()
    try:
        trigger_positions = [
            event.forward_op_count_at_trigger
            for event in log._capture_events.backward_events
            if event.__class__.__name__ == "BackwardPassStart"
        ]
        assert trigger_positions
        last_trigger_position = max(
            position for position in trigger_positions if position is not None
        )
        assert any(
            layer.grad_fn_object_id is not None
            and layer.grad_fn is None
            and layer.step_index > last_trigger_position
            for layer in log.layer_list
        )

        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_bad_pre_trigger_layer_grad_fn_backpointer_raises() -> None:
    """A paired pre-trigger layer with a severed GradFn backpointer raises."""

    log = _make_backward_log()
    try:
        victim = next(
            layer
            for layer in log.layer_list
            if layer.grad_fn_object_id is not None and layer.grad_fn is not None
        )
        victim.grad_fn = None

        with pytest.raises(MetadataInvariantError, match="missing its GradFn backpointer"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_bad_grad_fn_order_raises() -> None:
    """Unknown grad_fn_handle ids in grad_fn_order raise an invariant error."""

    log = _make_backward_log()
    log.grad_fn_order.append(-1)
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_backward_root_grad_fn_id_raises() -> None:
    """An unknown backward root id raises an invariant error."""

    log = _make_backward_log()
    log.backward_root_grad_fn_object_ids = -1  # type: ignore[assignment]
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_layer_grad_fn_backpointer_raises() -> None:
    """A mismatched layer-to-grad_fn_handle backpointer raises an invariant error."""

    log = _make_backward_log()
    for grad_fn_handle in log.grad_fn_logs.values():
        if grad_fn_handle.has_op:
            grad_fn_handle.op_label = "missing_layer_1"
            break
    with pytest.raises(MetadataInvariantError, match="backward_graph_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_bad_higher_order_creator_chain_order_raises() -> None:
    """A resolved higher-order creator chain with the wrong order raises."""

    class HigherOrderModel(nn.Module):
        """Tiny nonlinear model for higher-order backward metadata."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a scalar nonlinear output."""

            return (torch.tanh(x) ** 3).sum()

    from torchlens import trace as trace_fn

    x = torch.randn(3, requires_grad=True)
    log = trace_fn(HigherOrderModel(), x, save_grads="all", random_seed=42)
    try:
        loss = log[log.output_layers[0]].out
        first_grad = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]
        torch.autograd.grad(first_grad.sum(), x, retain_graph=True)
        victim = next(
            grad_fn
            for grad_fn in log.grad_fns
            if grad_fn.creator_object_id is not None and grad_fn.order is not None
        )
        assert victim.order is not None
        victim.order = victim.order + 1

        with pytest.raises(MetadataInvariantError, match="creator order"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_corruption_parent_child_link():
    """Breaking a parent→child link raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with children and corrupt
    for lpl in log.layer_list:
        if lpl.children:
            child_label = lpl.children[0]
            child = log[child_label]
            # Remove the parent from the child's parents
            child.parents = [p for p in child.parents if p != lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="graph_topology"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_num_ops():
    """Mismatched num_ops raises MetadataInvariantError."""
    log = _make_clean_log()
    log.num_ops = 9999
    with pytest.raises(MetadataInvariantError, match="trace_self_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_module_back_reference():
    """Removing a layer from its module's layers raises MetadataInvariantError."""
    log = _make_clean_log()
    # Find a layer with a containing module and corrupt the Module
    for lpl in log.layer_list:
        cmo = lpl.module
        if cmo:
            # module may include pass (e.g. 'fc:1'), strip it
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            mod_log = log.modules._dict[cmo_addr]
            if lpl.layer_label in mod_log.layer_labels:
                mod_log.layer_labels = [x for x in mod_log.layer_labels if x != lpl.layer_label]
                # num_layers is a read-only property derived from layer_labels
                break
    with pytest.raises(MetadataInvariantError, match="module_layer_containment"):
        check_metadata_invariants(log)
    log.cleanup()


def test_module_call_boundary_rejects_output_outside_ops() -> None:
    """ModuleCall output ops must be produced inside the call."""

    log = _make_nested_log()
    try:
        victim = log.modules["fc"].ops["fc:1"]
        external_input = victim.input_ops[0]
        assert external_input not in victim.ops

        victim.output_ops = [external_input]

        with pytest.raises(MetadataInvariantError, match="outside its ops"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_module_call_boundary_allows_external_inputs_on_nested_modules() -> None:
    """ModuleCall input ops are allowed to come from the parent scope."""

    log = _make_nested_log()
    try:
        fc_call = log.modules["fc"].ops["fc:1"]
        assert fc_call.input_ops
        assert fc_call.input_ops[0] not in fc_call.ops
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_module_call_tree_preconditions_reach_nested_trace() -> None:
    """ModuleCall call-tree links and stacks are populated on a nested trace."""

    log = _make_nested_log()
    try:
        layer1_call = log.module_calls["layer1:1"]
        layer10_call = log.module_calls["layer1.0:1"]
        assert layer1_call.call_parent == "self:1"
        assert "layer1.0:1" in layer1_call.call_children
        assert layer10_call.module_call_stack == ["layer1:1"]
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_module_call_output_structure_precondition_reaches_real_trace() -> None:
    """Populated ModuleCall output structures are scoped to retained outputs."""

    log = _make_tuple_output_module_log()
    try:
        sub_call = log.module_calls["sub:1"]
        assert sub_call.output_structure is not None
        assert sub_call.output_ops
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_torch_trace_metadata_invariants_still_green() -> None:
    """Torch traces still validate through the unchanged torch invariant path."""

    log = _make_clean_log()
    try:
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


@pytest.mark.parametrize("backend", ["jax", "tinygrad"])
def test_function_root_module_invariants_green_for_synthetic_backends(backend: str) -> None:
    """Synthetic JAX/tinygrad function-root traces pass minimal module invariants."""

    log = _relabel_as_function_root(_make_root_only_log(), backend)
    try:
        assert check_metadata_invariants(log) is True
    finally:
        log.cleanup()


def test_non_function_root_drop_op_module_attribution_raises() -> None:
    """Dropping an op's module attribution fails before output validation matters."""

    log = _relabel_as_mlx_object_module(_make_clean_log())
    try:
        victim = next(layer for layer in log.layer_list if layer.module)
        victim.module = None
        victim.modules = []
        victim.module_call_stack = []
        victim.output_of_modules = []
        victim.output_of_module_calls = []
        victim.atomic_module_call = None

        with pytest.raises(MetadataInvariantError, match="module_attribution"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_non_function_root_cyclic_address_parent_raises() -> None:
    """Cyclic module address_parent links fail module containment logic."""

    log = _relabel_as_mlx_object_module(_make_clean_log())
    try:
        root = log.modules["self"]
        child = log.modules["fc"]
        root.address_parent = "fc"
        root.address_children = ["fc"]
        child.address_parent = "self"
        child.address_children = ["self"]

        with pytest.raises(MetadataInvariantError, match="module_containment_logic"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_plain_non_function_root_unattributed_compute_op_raises() -> None:
    """A non-function-root trace with an unattributed compute op is invalid."""

    log = _relabel_as_mlx_object_module(_make_root_only_log())
    try:
        with pytest.raises(MetadataInvariantError, match="module_attribution"):
            check_metadata_invariants(log)
    finally:
        log.cleanup()


def test_corruption_layer_num_calls():
    """Wrong layer_num_calls raises MetadataInvariantError."""
    log = _make_clean_log()
    # Corrupt one entry
    first_key = list(log.layer_num_calls.keys())[0]
    log.layer_num_calls[first_key] = 999
    with pytest.raises(MetadataInvariantError, match="recurrence_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_output_layers_empty():
    """Emptying output_layers raises MetadataInvariantError."""
    log = _make_clean_log()
    log.output_layers = []
    with pytest.raises(MetadataInvariantError, match="trace_self_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


# =============================================================================
# Phase 2: Complex semantic invariant corruption tests (M-R)
# =============================================================================


class _RecurrentFF(nn.Module):
    """Simple recurrent model for loop detection tests."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(3):
            x = self.relu(self.fc(x))
        return x


class _NestedModel(nn.Module):
    """Model with nested submodules for module containment tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(5, 4), nn.ReLU())
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.layer1(x)
        return self.fc(x)


class _TupleOutputSubmodule(nn.Module):
    """Submodule that returns a tuple of tensors."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two tensor leaves."""

        return x + 1, x + 2


class _TupleOutputModule(nn.Module):
    """Model with a real ModuleCall output structure."""

    def __init__(self) -> None:
        """Initialize the tuple-output submodule."""

        super().__init__()
        self.sub = _TupleOutputSubmodule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use one output from a tuple-returning child module."""

        left, _right = self.sub(x)
        return left * 2


def _make_recurrent_log():
    from torchlens import trace as trace_fn

    return trace_fn(_RecurrentFF(), torch.randn(2, 5), random_seed=42)


def _make_nested_log():
    from torchlens import trace as trace_fn

    return trace_fn(_NestedModel(), torch.randn(2, 5), random_seed=42)


def _make_tuple_output_module_log() -> Trace:
    """Return a trace with a ModuleCall output structure."""

    from torchlens import trace as trace_fn

    return trace_fn(_TupleOutputModule(), torch.randn(2, 5), random_seed=42)


# -- M. Graph ordering corruption --


def test_corruption_graph_ordering_duplicate_rt_num():
    """Duplicate raw_index triggers graph_ordering error."""
    log = _make_clean_log()
    # Set two layers to the same raw_index
    log.layer_list[0].raw_index = log.layer_list[1].raw_index
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_graph_ordering_topo_violation():
    """Parent with higher raw_index than child triggers error."""
    log = _make_clean_log()
    # Find a layer with parents and swap rt nums to break topo order
    for lpl in log.layer_list:
        if lpl.parents:
            parent = log[lpl.parents[0]]
            # Give parent a higher rt num than child
            parent.raw_index, lpl.raw_index = (
                lpl.raw_index,
                parent.raw_index,
            )
            break
    with pytest.raises(MetadataInvariantError, match="graph_ordering"):
        check_metadata_invariants(log)
    log.cleanup()


# -- N. Loop detection corruption --


def test_corruption_loop_detection_slo_empty():
    """Empty recurrent_ops triggers loop_detection error."""
    log = _make_clean_log()
    log.layer_list[0].recurrent_ops = []
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_slo_asymmetry():
    """Asymmetric recurrent_ops triggers loop_detection error."""
    log = _make_recurrent_log()
    # Find a multi-pass layer and corrupt one member's slo list
    for lpl in log.layer_list:
        if lpl.num_passes > 1:
            # Remove one member from slo
            lpl.recurrent_ops = [lpl.layer_label]
            break
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_loop_detection_ops_total():
    """Mismatched num_calls vs len(recurrent_ops) triggers error."""
    log = _make_clean_log()
    log.layer_list[0].num_passes = 99
    with pytest.raises(MetadataInvariantError, match="loop_detection"):
        check_metadata_invariants(log)
    log.cleanup()


# -- O. Distance / reachability corruption --


def test_corruption_distance_min_gt_max():
    """min_distance > max_distance triggers distance_invariants error."""
    log = _make_clean_log()
    # Find a non-input layer with distances set
    for lpl in log.layer_list:
        if (
            lpl.min_distance_from_input is not None
            and lpl.max_distance_from_input is not None
            and lpl.min_distance_from_input > 0
        ):
            lpl.min_distance_from_input = lpl.max_distance_from_input + 1
            break
    else:
        # If no layer has distances, skip (mark_layer_depths might be False)
        log.cleanup()
        return
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_distance_input_nonzero():
    """Input layer with nonzero distance_from_input triggers error."""
    log = _make_clean_log()
    if not log.mark_layer_depths:
        log.cleanup()
        return
    for label in log.input_layers:
        lpl = log[label]
        lpl.min_distance_from_input = 5
        lpl.max_distance_from_input = 5
        break
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_distance_ancestor_flag():
    """Mismatch between has_input_ancestor and input_ancestors triggers error."""
    log = _make_clean_log()
    if not log.mark_layer_depths:
        log.cleanup()
        return
    for lpl in log.layer_list:
        if lpl.has_input_ancestor and len(lpl.input_ancestors) > 0:
            lpl.has_input_ancestor = False
            break
    with pytest.raises(MetadataInvariantError, match="distance_invariants"):
        check_metadata_invariants(log)
    log.cleanup()


# -- P. Graph connectivity corruption --


def test_corruption_connectivity_parentless_layer():
    """Removing all parents from a computational layer triggers error."""
    log = _make_clean_log()
    for lpl in log.layer_list:
        if (
            not lpl.is_input
            and not lpl.is_buffer
            and not lpl.is_output
            and not lpl.is_internal_source
            and lpl.parents
        ):
            # Also fix the parent's child list to avoid graph_topology catching it first
            for p_label in lpl.parents:
                parent = log[p_label]
                parent.children = [c for c in parent.children if c != lpl.layer_label]
                parent.has_children = len(parent.children) > 0
            lpl.parents = []
            # has_parents is a read-only property derived from parents
            break
    with pytest.raises(MetadataInvariantError, match="graph_connectivity"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_connectivity_orphan_in_layer_list():
    """Adding a label to _orphan_labels that is also in layer_labels triggers error."""
    log = _make_clean_log()
    log._orphan_labels = [log.layer_labels[0]]
    with pytest.raises(MetadataInvariantError, match="graph_connectivity"):
        check_metadata_invariants(log)
    log.cleanup()


# -- Q. Module containment logic corruption --


def test_corruption_module_depth():
    """Wrong address_depth on a module triggers error."""
    log = _make_nested_log()
    for mod_log in log.modules:
        if mod_log.address != "self" and mod_log.address_depth > 0:
            mod_log.address_depth = 999
            break
    with pytest.raises(MetadataInvariantError, match="module_containment_logic"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_module_nested_path_leaf():
    """Last element of modules != module triggers error."""
    log = _make_nested_log()
    for lpl in log.layer_list:
        if len(lpl.modules) >= 2 and lpl.module:
            # Swap the last nested module to a different valid module so it
            # doesn't fail the module_layer_containment check but does fail
            # the leaf consistency check in module_containment_logic.
            # Use the first (parent) module as the last entry — valid module but wrong leaf
            lpl.modules[-1] = lpl.modules[0]
            break
    with pytest.raises(MetadataInvariantError, match="module_containment_logic"):
        check_metadata_invariants(log)
    log.cleanup()


# -- R. Lookup key consistency corruption --


def test_corruption_lookup_key_forward():
    """Adding a key to forward dict without reverse entry triggers error."""
    log = _make_clean_log()
    log._lookup_keys_to_layer_num_dict["bogus_key"] = 99999
    with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_lookup_key_raw_to_final():
    """Adding a raw→final mapping that points to invalid label triggers error."""
    log = _make_clean_log()
    log._raw_to_final_layer_labels["bogus_raw"] = "bogus_final"
    log._final_to_raw_layer_labels["bogus_final"] = "bogus_raw"
    with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
        check_metadata_invariants(log)
    log.cleanup()


def test_corruption_raw_label_asymmetry():
    """Mismatch between raw→final and final→raw triggers error."""
    log = _make_clean_log()
    if log._raw_to_final_layer_labels:
        first_raw = next(iter(log._raw_to_final_layer_labels))
        first_final = log._raw_to_final_layer_labels[first_raw]
        # Point the reverse to a different raw label
        log._final_to_raw_layer_labels[first_final] = "corrupted_raw"
        with pytest.raises(MetadataInvariantError, match="lookup_key_consistency"):
            check_metadata_invariants(log)
    log.cleanup()


# -- Clean recurrent and nested models pass all invariants --


def test_clean_recurrent_log_ops_all_invariants():
    """Recurrent model Trace ops all invariant checks."""
    log = _make_recurrent_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


def test_clean_nested_log_ops_all_invariants():
    """Nested model Trace ops all invariant checks."""
    log = _make_nested_log()
    assert check_metadata_invariants(log) is True
    log.cleanup()


# =============================================================================
# Bugfix regression tests
# =============================================================================


class _UnusedInputModel(nn.Module):
    """Model that ignores one of its keyword arguments."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x, unused_mask=None):
        return self.fc(x)


class _SharedParamDifferentOps(nn.Module):
    """Model where different operations consume the same parameter.

    The weight is used both in linear and also explicitly via torch.sum.
    """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        out = self.fc(x)
        weight_sum = torch.sum(self.fc.weight)
        return out + weight_sum


class _SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class _BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(10)
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(self.bn(x))


class TestValidationBugfixes:
    """Validation correctness."""

    def test_validation_basic(self):
        model = _SimpleLinear()
        x = torch.randn(2, 10)
        assert validate_forward_pass(model, x)

    def test_validation_batchnorm(self):
        """Validation with BatchNorm (has buffers) should work."""
        model = _BatchNormModel()
        x = torch.randn(4, 10)
        assert validate_forward_pass(model, x)

    def test_validation_unsaved_parent_no_crash(self):
        """Validation with layers_to_save subset should not crash on None parents."""
        from torchlens import trace as trace_fn

        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, layers_to_save="all")
        assert log is not None


class TestValidationNoSavedArgs:
    """Validation with save_arg_values=False."""

    def test_validation_no_args(self):
        """validate with save_arg_values=False should not crash."""
        from torchlens import trace as trace_fn

        model = _SimpleLinear()
        x = torch.randn(2, 10)
        log = trace_fn(model, x, save_arg_values=False)
        assert log is not None


class TestPosthocPerturbCheck:
    """posthoc_perturb_check correctly exempts layers with special-value args.

    Note: The original "return on first special arg" behavior is CORRECT —
    any single all-zeros/all-ones arg can explain output invariance.
    """

    def test_batchnorm_validation_with_buffers(self):
        model = _BatchNormModel()
        x = torch.randn(4, 10)
        result = validate_forward_pass(model, x)
        assert result is True


class TestUnusedInputValidation:
    """Regression: unused input kwargs should not crash validation.

    When a model ignores a kwarg (e.g. token_type_ids in DistilBert), the
    corresponding input layer has func=None and no children. Validation
    must skip replay for such layers instead of crashing on None().
    """

    def test_unused_kwarg_input_ops_validation(self):
        model = _UnusedInputModel()
        x = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        assert validate_forward_pass(model, x, input_kwargs={"unused_mask": mask})

    def test_unused_kwarg_input_annotations_ops(self):
        """Metadata invariants pass even with unused input layers."""
        from torchlens import trace as trace_fn

        model = _UnusedInputModel()
        x = torch.randn(2, 10)
        mask = torch.ones(2, 10)
        log = trace_fn(model, x, input_kwargs={"unused_mask": mask})
        check_metadata_invariants(log)


class TestSharedParamDifferentOps:
    """Regression: different operations consuming the same parameter should not
    violate loop_detection invariants.

    The param sharing invariant must group by (func_name, param_barcodes),
    not just param_barcodes alone. Otherwise, e.g. isinf(weight) and expand(weight)
    would be falsely flagged as needing the same layer_label.
    """

    def test_shared_param_different_ops_ops_validation(self):
        model = _SharedParamDifferentOps()
        x = torch.randn(2, 10)
        assert validate_forward_pass(model, x)

    def test_shared_param_different_ops_metadata_ops(self):
        from torchlens import trace as trace_fn

        model = _SharedParamDifferentOps()
        x = torch.randn(2, 10)
        log = trace_fn(model, x)
        check_metadata_invariants(log)


# =============================================================================
# Tripwire: plain capture must NOT emit functionless intervention placeholders
# =============================================================================


class _VmapMaskConsumer(nn.Module):
    """Submodule that consumes an externally built mask tensor."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Add the mask to ``x``."""

        return x + mask


class _VmapMaskModel(nn.Module):
    """Build a fresh mask tensor via ``torch.vmap`` and feed it to a submodule.

    This mirrors how HuggingFace transformers (Mistral, VITS, etc.) construct
    the 4D causal/sliding-window attention mask: the mask is materialized inside
    a ``torch.vmap`` transform, whose internal operations TorchLens cannot trace.
    The fully-formed mask then enters a downstream module untagged. Plain capture
    must register it as a clean internal source -- NOT a functionless
    ``intervention_replacement`` placeholder (which would be a silent capture gap
    papered over by a validation exemption).
    """

    def __init__(self) -> None:
        """Initialize the mask-consuming submodule."""

        super().__init__()
        self.consumer = _VmapMaskConsumer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Build a vmap mask and route it through the consumer submodule."""

        q_idx = torch.arange(x.shape[-1])
        kv_idx = torch.arange(x.shape[-1])

        def cell(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
            return (j <= i).float()

        mask = torch.vmap(torch.vmap(cell, in_dims=(None, 0)), in_dims=(0, None))(q_idx, kv_idx)
        return self.consumer(x, mask)


def _functionless_replacement_ops(log: "Trace") -> list:
    """Return ops that are auto-synthesized functionless intervention placeholders.

    A GENUINE raw forward-hook output replacement is legitimately functionless,
    but it carries ``func_name == "intervention_replacement"``. This helper
    targets exactly that placeholder shape so a capture gap (an untraced source
    surfacing as an intervention placeholder during plain tracing) is caught.
    """

    return [
        op
        for op in log.ops
        if getattr(op, "func_name", None) == "intervention_replacement"
        and getattr(op, "intervention_replaced", False)
    ]


def test_plain_trace_vmap_mask_has_no_functionless_replacement() -> None:
    """TRIPWIRE: an untraced vmap-built mask must not become a placeholder.

    Reintroducing the old behavior (logging the untagged mask as a functionless
    ``intervention_replacement`` op during plain capture) makes this fail loudly.
    """

    from torchlens import trace as trace_fn

    model = _VmapMaskModel().eval()
    x = torch.randn(4, 4)
    log = trace_fn(model, [x], {})

    # No genuine user intervention happened, so there must be zero functionless
    # intervention placeholders.
    assert _functionless_replacement_ops(log) == []
    assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []

    # The mask is instead logged as a transform boundary node with a clean parent edge.
    transform_ops = [op for op in log.ops if getattr(op, "is_transform", False)]
    assert [op.type for op in transform_ops] == ["vmap"]
    assert transform_ops[0].transform_chain == ("vmap", "vmap")
    assert transform_ops[0].parents
    assert all(parent.startswith("arange_") for parent in transform_ops[0].parents)
    transform_label = transform_ops[0].label.split(":")[0]
    assert any(transform_label in op.parents for op in log.ops if op.type == "add")

    # Validation passes legitimately (not via an exemption hiding the gap).
    check_metadata_invariants(log)
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_plain_trace_mistral_has_no_functionless_replacement():
    """TRIPWIRE on the real reproducer: tiny Mistral must trace cleanly.

    The HuggingFace Mistral attention mask is built inside ``torch.vmap``; plain
    tracing must surface it as an internal source, never a functionless
    intervention placeholder.
    """

    transformers = pytest.importorskip("transformers")
    from torchlens import trace as trace_fn

    cfg = transformers.MistralConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        sliding_window=16,
    )
    model = transformers.MistralForCausalLM(cfg).eval()
    log = trace_fn(model, [], {"input_ids": torch.randint(0, 100, (1, 16))})

    assert _functionless_replacement_ops(log) == []
    assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []
    check_metadata_invariants(log)
