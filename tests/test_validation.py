"""Tests for the validation subpackage.

Covers: import paths, registry consistency, perturbation unit tests,
deep clone helpers, and integration tests through specific exemption paths.
"""

from collections import defaultdict, deque, namedtuple
from dataclasses import replace
import threading
import warnings
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn

# Import the implementation entry point first: it breaks the standalone
# collection cycle between torchlens and torchlens._user_public_impls.
import torchlens.user_funcs as user_funcs
import torchlens as tl
import torchlens._user_public_impls as user_public_impls
from torchlens import Trace, trace as trace_fn
from torchlens.validation import (
    ValidationDiagnostic,
    get_validation_diagnostics,
    validate_forward_pass,
)
from torchlens.errors import MetadataInvariantError, TraceNotReproducibleWarning
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
    posthoc_perturb_check,
)
from torchlens.validation.core import (
    _perturb_layer_outs,
    _deep_clone_tensors,
    _copy_validation_args,
    _execute_func_with_restored_state,
    _check_perturbation_exemptions,
    _restore_live_parameter_args_for_replay,
    _op_reduction_depth,
    _deep_numeric_replay_matches_saved,
    completeness_backstop_counts,
    DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH,
    ValidationDecisionRecorder,
    validate_parents_of_saved_layer,
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


def test_validation_decision_recorder_counts_distinct_nodes() -> None:
    """Node coverage counts must not count replay phases as extra nodes."""

    recorder = ValidationDecisionRecorder()
    recorder.record(
        op_label="add_1:1",
        func_name="add",
        phase="replay",
        decision="validated",
        reason="replay_matched",
    )
    recorder.record(
        op_label="add_1:1",
        func_name="add",
        phase="perturbation",
        decision="validated",
        reason="perturbation_changed",
    )
    recorder.record(
        op_label="output_1:1",
        func_name="none",
        phase="ground_truth",
        decision="validated",
        reason="ground_truth_matched",
    )

    status = recorder.as_status()

    assert status.replayed_node_count == 2
    assert status.state == "passed"


def test_validation_decision_recorder_distinct_node_count_keeps_failure_tripwire() -> None:
    """Repeated failure decisions still produce one genuine failed node."""

    recorder = ValidationDecisionRecorder()
    for phase in ("replay", "perturbation"):
        recorder.record(
            op_label="broken_1:1",
            func_name="broken",
            phase=phase,
            decision="failed",
            reason="replay_mismatch",
        )

    status = recorder.as_status()

    assert status.failed_node_count == 1
    assert status.state == "failed"
    assert bool(status) is False


def test_validation_all_exempted_cannot_pass_without_replayed_nodes() -> None:
    """Exemptions alone must produce a non-boolean unverified terminal state."""

    recorder = ValidationDecisionRecorder()
    recorder.record(
        op_label="structural_1:1",
        func_name="structural",
        phase="perturbation",
        decision="exempted",
        reason="pre_perturbation_exemption",
    )

    status = recorder.as_status()

    assert status.state == "unverified"
    assert status.reason == "no_nodes_replay_validated"
    assert status.replayed_node_count == 0
    with pytest.raises(TypeError, match="not a boolean"):
        bool(status)


def test_validation_positive_replay_coverage_can_still_pass() -> None:
    """The zero-coverage guard must not block a genuinely validated node."""

    recorder = ValidationDecisionRecorder()
    recorder.record(
        op_label="identity_1:1",
        func_name="identity",
        phase="replay",
        decision="validated",
        reason="replay_matched",
    )

    status = recorder.as_status()

    assert status.state == "passed"
    assert status.replayed_node_count == 1
    assert bool(status) is True


@pytest.mark.smoke
def test_validate_forward_pass_importable():
    """validate_forward_pass is importable from torchlens top-level."""
    assert callable(validate_forward_pass)


class _ScatterAggregationModel(nn.Module):
    """A message-passing-style model whose forward aggregates messages into a
    destination tensor with an in-place ``index_add_`` (the GNN/molecular pattern).

    Under multi-threading, an in-place scatter perturbation can race so the
    perturbed output is indistinguishable from the original, spuriously failing
    the sensitivity check. The determinism stabilizer in ``_validate_forward_pass_torch``
    makes the perturbation reproducible so validation is stably True.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        messages = self.lin(x)
        aggregated = torch.zeros(4, 8)
        aggregated.index_add_(0, index, messages)
        return aggregated.sum(dim=1)


def test_validation_determinism_stabilizes_inplace_scatter_perturbation() -> None:
    """A scatter-aggregation model validates stably True across repeated runs.

    Exercises the determinism stabilizer (``torch.use_deterministic_algorithms(
    True, warn_only=True)`` around capture + replay + perturbation): the
    nondeterministic in-place-scatter perturbation flake (the GNN "regression"
    class) must not surface, so validation does not flake to False.
    """

    model = _ScatterAggregationModel().eval()
    x = torch.randn(10, 8)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])

    results = [validate_forward_pass(model, (x, index)) for _ in range(5)]
    assert all(results), results


def test_validation_restores_prior_deterministic_algorithms_setting() -> None:
    """The determinism stabilizer must save and restore the prior torch setting.

    Validation flips ``use_deterministic_algorithms`` on for its own capture +
    replay, but it must leave the process-wide setting exactly as it found it
    (both the enabled flag AND the warn_only flag) so it does not leak into the
    caller's environment.
    """

    model = _ScatterAggregationModel().eval()
    x = torch.randn(10, 8)
    index = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])

    prior_enabled = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        validate_forward_pass(model, (x, index))
        assert torch.are_deterministic_algorithms_enabled() is prior_enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is prior_warn_only

        # And it restores a non-default prior setting (enabled, not warn_only) too.
        torch.use_deterministic_algorithms(True, warn_only=False)
        validate_forward_pass(model, (x, index))
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.is_deterministic_algorithms_warn_only_enabled() is False
    finally:
        torch.use_deterministic_algorithms(prior_enabled, warn_only=prior_warn_only)


def test_validation_default_threads_and_explicit_single_thread_pin_restore() -> None:
    """LOAD-BEARING: default forwards use process threads; explicit pin restores.

    This is the host-independent mechanism gate for the inter-run multi-thread
    float-reduction-order fix. The drift it removes (~3e-7 ground-truth output
    disagreement straddling ``GROUND_TRUTH_OUTPUT_RTOL=1e-6``, plus the MoE
    masked-gate perturbation flake) is hardware/thread-count dependent and may
    not reproduce on every host, so we assert the retry mechanism directly
    rather than relying on a host reproducing the flake:

    * the default harness call does not override the process thread count;
    * ``num_threads=1`` is observed from inside the validation forwards;
    * the process-wide thread count is restored afterward, so the retry pin does
      not leak into the caller's (or later tests') environment.
    """

    observed_threads: list[int] = []

    class _ThreadProbeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            observed_threads.append(torch.get_num_threads())
            return self.lin(x)

    model = _ThreadProbeModel().eval()
    x = torch.randn(2, 4)

    prior_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, prior_num_threads))
        default_threads = torch.get_num_threads()
        result = validate_forward_pass(model, (x,))
        assert result is True
        assert observed_threads, "probe model forward was never invoked"
        assert all(t == default_threads for t in observed_threads), observed_threads
        assert torch.get_num_threads() == default_threads

        observed_threads.clear()
        result = user_funcs._validate_forward_pass_torch(model, (x,), num_threads=1)
        assert result is True
        # The explicit deterministic retry pin wraps both the ground-truth and
        # capture forwards inside the harness.
        assert observed_threads, "probe model forward was never invoked"
        assert all(t == 1 for t in observed_threads), observed_threads
        assert torch.get_num_threads() == default_threads
    finally:
        torch.set_num_threads(prior_num_threads)


class _SpectralGCNGroundTruthDriftModel(nn.Module):
    """A Chebyshev-spectral-conv-style model (MSTGCN / TGT-MSTGCN family) whose
    ground-truth output drifts ~3e-7 between two clean forwards under multi-threaded
    float reduction-order -- straddling the strict phase-0 ground-truth bar
    (``GROUND_TRUTH_OUTPUT_RTOL=1e-6``) -- yet goes bit-exact under a single thread.

    The forward stacks many repeated sparse-aggregation reductions (the Chebyshev
    polynomial recurrence over a graph adjacency) so the parallel accumulation order
    is unpinned; this is the structure that makes the spectral-GCN family FLAKY at
    the strict ground-truth tolerance without the harness single-thread pin.
    """

    def __init__(self, n: int = 64, hops: int = 6) -> None:
        super().__init__()
        torch.manual_seed(0)
        # A dense graph adjacency (the spectral operator); repeated matmuls against
        # it are the Chebyshev recurrence whose reduction order is thread-dependent.
        self.register_buffer("adj", torch.randn(n, n) / n)
        self.lin = nn.Linear(n, n)
        self.hops = hops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        acc = h
        prev = h
        cur = torch.matmul(h, self.adj)
        # Chebyshev recurrence T_k = 2 * adj @ T_{k-1} - T_{k-2}, accumulated.
        for _ in range(self.hops):
            acc = acc + cur
            nxt = 2.0 * torch.matmul(cur, self.adj) - prev
            prev, cur = cur, nxt
        return acc.sum(dim=0, keepdim=True)


def test_validation_spectral_gcn_ground_truth_determinism() -> None:
    """GATE (B/C-1): spectral-GCN ground-truth stability across N>=5 repeats.

    A Chebyshev-spectral-conv model (MSTGCN / TGT-MSTGCN family) -- the structure
    whose two clean forwards disagreed by ~3e-7 at the output under multi-threaded
    reduction order, straddling the strict phase-0 ground-truth bar
    (``GROUND_TRUTH_OUTPUT_RTOL=1e-6``) -- validates stably True across repeats now
    that the harness pins a single intra-op thread.

    NOTE: the underlying multi-thread reduction-order drift is hardware/thread-count
    dependent and does not reproduce on every host; the host-independent guard for
    the pin's mechanism is
    ``test_validation_pins_single_thread_inside_harness_and_restores``. This gate
    additionally pins down that the real spectral-GCN aggregation class validates
    GREEN (and reproducibly so) under the fix, WITHOUT loosening any tolerance --
    the strict 1e-6 bar is unchanged.
    """

    model = _SpectralGCNGroundTruthDriftModel().eval()
    x = torch.randn(32, 64)

    results = [validate_forward_pass(model, (x,)) for _ in range(5)]
    assert all(results), results


class _MoEMaskedGateModel(nn.Module):
    """A Mixture-of-Experts-style masked-gate model (minimax / nllb-moe family)
    whose routing builds a boolean mask via a comparison (``lt``) and gates expert
    outputs through a ``where`` / elementwise ``mul``.

    Under multi-threaded reduction order, perturbing the routing mask sometimes
    leaves the gated output unchanged (the perturbed position is already masked to
    zero / re-selects the same path), spuriously failing the perturbation
    sensitivity check. The harness single-thread pin makes the gating reproducible
    so the sensitivity outcome is stable.
    """

    def __init__(self, dim: int = 32, n_experts: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_experts))
        self.n_experts = n_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.router(x)  # (B, n_experts)
        # Boolean routing mask: keep experts scoring above the per-token mean.
        threshold = scores.mean(dim=-1, keepdim=True)
        gate = scores.lt(threshold)  # boolean mask feeding where/mul
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            keep = gate[:, i : i + 1]
            contrib = torch.where(keep, expert(x), torch.zeros_like(x))
            out = out + contrib * scores[:, i : i + 1]
        return out


def test_validation_moe_masked_gate_perturbation_determinism() -> None:
    """GATE (B/C-2): MoE masked-gate perturbation stability across N>=5 repeats.

    A Mixture-of-Experts masked-gate model (minimax / nllb-moe family) whose routing
    builds a boolean mask via ``lt`` and gates expert outputs through ``where`` /
    elementwise ``mul`` -- the structure whose perturbation sensitivity check flaked
    under multi-threaded reduction order (the perturbed mask sometimes left the
    output unchanged) -- validates stably True across repeats now that the harness
    pins a single intra-op thread.

    NOTE: as with the spectral-GCN gate, the underlying thread non-determinism is
    host-dependent; the mechanism guard is
    ``test_validation_pins_single_thread_inside_harness_and_restores``. This gate
    pins down that the real masked-gate routing class validates GREEN reproducibly
    under the fix, WITHOUT broadening the perturbation tolerance or skipping the
    check -- the sensitivity check still runs at full strictness.
    """

    model = _MoEMaskedGateModel().eval()
    x = torch.randn(16, 32)

    results = [validate_forward_pass(model, (x,)) for _ in range(5)]
    assert all(results), results


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

        # The global-container tensor genuinely has no graph/source provenance,
        # so each capture correctly emits the unattributed-tensor-args warning
        # (cert10 diagnostic); the concern under test is stale-label clearing.
        with pytest.warns(UserWarning, match="no graph/source provenance"):
            assert validate_forward_pass(model, x, validate_metadata=True) is True
        with pytest.warns(UserWarning, match="no graph/source provenance"):
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


def test_validate_forward_pass_uses_typed_ground_truth_leaf_order() -> None:
    """Ground-truth output leaves follow capture's typed container traversal."""

    UnpoolInfo = namedtuple("UnpoolInfo", ("edge_index", "cluster", "batch"))

    class NestedNamedtupleOutput(nn.Module):
        """Return a tuple with a nested namedtuple payload."""

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, UnpoolInfo]:
            """Run the model.

            Parameters
            ----------
            x
                Input tensor.

            Returns
            -------
            tuple[torch.Tensor, UnpoolInfo]
                Tensor output plus namedtuple metadata leaves.
            """

            edge_index = torch.stack((torch.arange(4), torch.arange(4).flip(0)))
            cluster = torch.zeros(4, dtype=torch.long)
            batch = torch.ones(4, dtype=torch.long)
            return x + 1.0, UnpoolInfo(edge_index=edge_index, cluster=cluster, batch=batch)

    assert validate_forward_pass(NestedNamedtupleOutput(), torch.randn(1, 3)) is True


def test_validate_forward_pass_accepts_nested_lstm_module_outputs() -> None:
    """Module output metadata preserves nn.LSTM's nested public return paths."""

    class TupleLSTM(nn.Module):
        """Return the raw nested output of an ``nn.LSTM`` module."""

        def __init__(self) -> None:
            """Initialize the LSTM fixture."""

            super().__init__()
            self.lstm = nn.LSTM(5, 7, batch_first=True)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Run the LSTM.

            Parameters
            ----------
            x
                Batch-major sequence input.

            Returns
            -------
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
                Raw ``nn.LSTM`` output and hidden-state tuple.
            """

            return self.lstm(x)

    model = TupleLSTM().eval()
    x = torch.randn(2, 3, 5)

    assert validate_forward_pass(model, x, validate_metadata=True) is True

    trace = trace_fn(model, x, save_arg_values=True)
    try:
        path_reprs = {repr(path) for path in trace.module_calls["lstm:1"].output_paths}

        assert path_reprs == {
            "(TupleIndex(index=0),)",
            "(TupleIndex(index=1), TupleIndex(index=0))",
            "(TupleIndex(index=1), TupleIndex(index=1))",
        }
    finally:
        trace.cleanup()


def test_detached_reference_patcher_ignores_opaque_defaults() -> None:
    """Detached-reference patching treats non-tuple defaults as opaque metadata."""

    from torchlens.backends.torch.wrappers import _patch_function_defaults

    class CallableWithOpaqueDefaults:
        """Callable object exposing a non-standard ``__defaults__`` value."""

        __defaults__ = object()

        def __call__(self) -> None:
            """Run the callable."""

    candidate = CallableWithOpaqueDefaults()
    original_defaults = candidate.__defaults__

    _patch_function_defaults(candidate, {id(original_defaults): "replacement"})

    assert candidate.__defaults__ is original_defaults


def test_posthoc_perturb_constant_check_supports_complex_outputs() -> None:
    """Posthoc perturbation checks run on complex outputs without crashing.

    Historically the blanket constant-output excuse called ``unique`` on the
    output, which is unsupported for complex tensors. cert10 (74318b5d)
    replaced the blanket constant-output/all-special-value excuses with narrow
    structured value proofs, so a constant complex output is now honestly
    NON-exempt -- the concern preserved here is that the check handles complex
    dtypes without raising and returns a structured decision.
    """

    layer = SimpleNamespace(
        func_name="fft",
        saved_args=(torch.ones(4, dtype=torch.complex64),),
        saved_kwargs={},
        dtype=torch.complex64,
        out=torch.ones(4, dtype=torch.complex64),
        layer_label="fft_1_1",
    )

    decision = posthoc_perturb_check(SimpleNamespace(), layer, ["input_1"], verbose=False)
    assert decision.exempt is False
    assert decision.reason == "no_posthoc_exemption"


def _cast_layer(source: torch.Tensor, target: object) -> SimpleNamespace:
    """Build a fake ``to()`` op record for posthoc-decision tests.

    Parameters
    ----------
    source:
        Saved source tensor (args[0]).
    target:
        Saved cast target (args[1]).

    Returns
    -------
    SimpleNamespace
        Minimal op-like record for ``posthoc_perturb_check``.
    """

    out = source.to(target) if isinstance(target, torch.dtype) else source
    return SimpleNamespace(
        func_name="to",
        saved_args=(source, target),
        saved_kwargs={},
        dtype=out.dtype,
        out=out,
        layer_label="to_1_1",
        parent_arg_positions={"args": {0: "input_1"}, "kwargs": {}},
        parents=["input_1"],
    )


def test_posthoc_integer_cast_quantization_is_exempt_only_for_float_to_int() -> None:
    """Float->int cast insensitivity is quantization; other casts stay strict.

    Positive: a floating source cast to an integer dtype is exempt (a
    same-integer-bucket perturbation cannot change the output by construction).
    Negatives (the bug-mask guards): float->float and int->int casts must NOT
    be exempt -- value insensitivity there would indicate a real capture bug.
    """

    float_to_int = posthoc_perturb_check(
        SimpleNamespace(), _cast_layer(torch.tensor([5.0]), torch.int64), ["input_1"]
    )
    assert float_to_int.exempt is True
    assert float_to_int.reason == "integer_cast_quantization"

    float_to_float = posthoc_perturb_check(
        SimpleNamespace(), _cast_layer(torch.tensor([5.0]), torch.float64), ["input_1"]
    )
    assert float_to_float.exempt is False

    int_to_int = posthoc_perturb_check(
        SimpleNamespace(), _cast_layer(torch.tensor([5]), torch.int32), ["input_1"]
    )
    assert int_to_int.exempt is False


def test_pure_out_kwarg_destination_perturbation_is_exempt_only_when_pure() -> None:
    """A parent occupying ONLY kwargs['out'] is exempt; mixed positions stay strict.

    Positive: torch's ``out=`` convention makes the destination a write-only
    storage target (PyG SAGPooling's ``cumsum(counts, out=ptr[1:])``), so its
    prior values never influence the result. Negatives (the bug-mask guards):
    a parent that also feeds a positional slot, or occupies a non-``out``
    keyword, feeds real values and must NOT be exempt.
    """

    from torchlens.validation.core import _perturbed_parents_only_occupy_out_kwarg

    pure_out = SimpleNamespace(
        parent_arg_positions={"args": {0: "data_1"}, "kwargs": {"out": "dest_1"}}
    )
    assert _perturbed_parents_only_occupy_out_kwarg(pure_out, ["dest_1"]) is True
    # The data parent stays strict even on the same op.
    assert _perturbed_parents_only_occupy_out_kwarg(pure_out, ["data_1"]) is False

    also_positional = SimpleNamespace(
        parent_arg_positions={"args": {0: "dest_1"}, "kwargs": {"out": "dest_1"}}
    )
    assert _perturbed_parents_only_occupy_out_kwarg(also_positional, ["dest_1"]) is False

    other_kwarg = SimpleNamespace(
        parent_arg_positions={"args": {}, "kwargs": {"out": "dest_1", "src": "dest_1"}}
    )
    assert _perturbed_parents_only_occupy_out_kwarg(other_kwarg, ["dest_1"]) is False

    assert _perturbed_parents_only_occupy_out_kwarg(pure_out, []) is False


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


def test_validate_forward_pass_pristine_replay_catches_mutation_masked_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planted replay bug cannot hide behind post-forward buffer state."""

    class CounterBuffer(nn.Module):
        """Model whose registered counter changes its structural forward path."""

        def __init__(self) -> None:
            """Initialize the mutable buffer."""

            super().__init__()
            self.register_buffer("step", torch.zeros(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a counter-selected branch, then advance the counter."""

            step = cast(torch.Tensor, self.step)
            if step.item() > 0:
                output = x * 2
            else:
                output = x + 1
            step.add_(1)
            return output

    def planted_state_sensitive_capture_bug(
        trace: Trace,
        ground_truth_output_tensors: list[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> bool:
        """Simulate a replay bug whose stale source state masks a bad capture."""

        del ground_truth_output_tensors, verbose, validate_metadata
        source_ref = trace._source_model_ref
        source_model = source_ref()
        assert source_model is not None
        return bool(cast(torch.Tensor, source_model.step).item() > 0)

    monkeypatch.setattr(Trace, "validate_forward_pass", planted_state_sensitive_capture_bug)
    with pytest.warns(TraceNotReproducibleWarning, match="stateful/non-reproducible"):
        assert (
            user_public_impls._validate_forward_pass_torch(
                CounterBuffer(),
                torch.randn(3),
                validate_metadata=False,
            )
            is False
        )

    original_restore = user_public_impls._restore_validation_replay_state

    def skip_restore(
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        plain_attr_snapshot: object | None,
    ) -> None:
        """Simulate the old validation behavior that replayed post-pass state."""

        del model, state_dict, plain_attr_snapshot

    monkeypatch.setattr(user_public_impls, "_restore_validation_replay_state", skip_restore)

    with pytest.warns(TraceNotReproducibleWarning, match="stateful/non-reproducible"):
        assert (
            user_public_impls._validate_forward_pass_torch(
                CounterBuffer(),
                torch.randn(3),
                validate_metadata=False,
            )
            is True
        )
    monkeypatch.setattr(user_public_impls, "_restore_validation_replay_state", original_restore)


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


def test_validate_forward_pass_replay_copy_fallback_warns_for_lock_attr() -> None:
    """Replay copy failure falls back to live state with an explicit warning."""

    class LockBackedModel(nn.Module):
        """Model holding an uncopyable external resource."""

        def __init__(self) -> None:
            """Initialize the lock and a simple registered buffer."""

            super().__init__()
            self.lock = threading.Lock()
            self.register_buffer("bias", torch.ones(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Use registered tensor state without mutating the lock."""

            return x + cast(torch.Tensor, self.bias)

    with pytest.warns(
        RuntimeWarning,
        match="validation replay against live model state; model could not be copied",
    ):
        assert validate_forward_pass(LockBackedModel(), torch.randn(3)) is True


def test_validate_forward_pass_warns_on_stateful_retrace_divergence() -> None:
    """Structural re-trace divergence emits a structured retained warning."""

    class ToggleBranch(nn.Module):
        """Model that changes control flow after one forward pass."""

        def __init__(self) -> None:
            """Initialize branch state."""

            super().__init__()
            self.use_mul = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a different branch after the first pass."""

            if self.use_mul:
                out = x * 2
            else:
                out = x + 1
            self.use_mul = True
            return out

    diagnostics: list[tuple[ValidationDiagnostic, ...]] = []

    def observe_trace(trace: Trace) -> None:
        """Retain diagnostics before validation cleans up its trace."""

        diagnostics.append(get_validation_diagnostics(trace))

    with pytest.warns(
        TraceNotReproducibleWarning,
        match="stateful/non-reproducible.*make the forward path state-independent",
    ) as caught:
        assert user_public_impls._validate_forward_pass_torch(
            ToggleBranch(),
            torch.randn(2, 3),
            _trace_observer=observe_trace,
        )

    warning = caught[0].message
    assert isinstance(warning, TraceNotReproducibleWarning)
    assert warning.fields["first_graph_hash"] != warning.fields["retrace_graph_hash"]
    assert warning.fields["first_divergence"] is not None
    assert len(diagnostics) == 1
    assert diagnostics[0][0].check == "trace_retrace_structure_mismatch"
    assert diagnostics[0][0].extra["first_op_count"] == warning.fields["first_op_count"]


def test_validate_forward_pass_no_retrace_warning_for_stateless_model() -> None:
    """Validation does not warn for a stateless repeated trace."""

    class StatelessToy(nn.Module):
        """Small stateless model with stable graph structure."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a deterministic stateless tensor computation."""

            return torch.relu(x + 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_forward_pass(StatelessToy(), torch.randn(2, 3)) is True

    messages = [str(warning.message) for warning in caught]
    assert not any("stateful/non-reproducible" in message for message in messages)


def test_validate_forward_pass_dropout_value_drift_does_not_warn_on_retrace() -> None:
    """RNG value drift alone does not trigger the structural re-trace warning."""

    diagnostics: list[tuple[ValidationDiagnostic, ...]] = []

    def observe_trace(trace: Trace) -> None:
        """Retain diagnostics before validation cleans up its trace."""

        diagnostics.append(get_validation_diagnostics(trace))

    model = nn.Dropout(p=0.5).train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert user_public_impls._validate_forward_pass_torch(
            model,
            torch.randn(4, 3),
            _trace_observer=observe_trace,
        )

    assert not any(issubclass(item.category, TraceNotReproducibleWarning) for item in caught)
    assert diagnostics == [()]


def test_validate_forward_pass_train_batch_norm_has_no_retrace_warning() -> None:
    """Same-option retracing must ignore train-mode BatchNorm value drift."""

    class BatchNormModel(nn.Module):
        """Model with a state-mutating registered BatchNorm buffer."""

        def __init__(self) -> None:
            """Initialize train-mode BatchNorm."""

            super().__init__()
            self.batch_norm = nn.BatchNorm1d(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Update BatchNorm running statistics and return normalized inputs."""

            return self.batch_norm(x)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert validate_forward_pass(BatchNormModel(), torch.randn(4, 3)) is True

    assert not any(issubclass(warning.category, TraceNotReproducibleWarning) for warning in caught)


def test_validate_forward_pass_uncopyable_model_does_not_retrace() -> None:
    """An uncopyable model skips only the new pristine re-trace diagnostic."""

    execution_count = [0]

    class UncopyableExecutionCounter(nn.Module):
        """Uncopyable model that records every full forward execution."""

        def __init__(self) -> None:
            """Initialize the execution counter."""

            super().__init__()

        def __deepcopy__(self, memo: dict[int, object]) -> "UncopyableExecutionCounter":
            """Reject validation's isolation copies."""

            del memo
            raise TypeError("cannot deepcopy model")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Count and execute a stable forward path."""

            execution_count[0] += 1
            return x + 1

    diagnostics: list[tuple[ValidationDiagnostic, ...]] = []

    def observe_trace(trace: Trace) -> None:
        """Retain diagnostics before validation cleans up its trace."""

        diagnostics.append(get_validation_diagnostics(trace))

    model = UncopyableExecutionCounter()
    with pytest.warns(RuntimeWarning, match="could not deepcopy the model"):
        assert user_public_impls._validate_forward_pass_torch(
            model,
            torch.randn(2, 3),
            _trace_observer=observe_trace,
        )

    assert execution_count == [2]
    assert len(diagnostics) == 1
    assert diagnostics[0][0].check == "trace_retrace_pristine_copy_unavailable"


def test_ground_truth_copy_fallback_warns_when_plain_attrs_cannot_be_snapshotted() -> None:
    """Ground-truth fallback skips only an unsnapshotable plain attribute."""

    class UncopyableOpaqueState(nn.Module):
        """Model that defeats both deepcopy and plain-attribute snapshots."""

        def __init__(self) -> None:
            """Initialize opaque state that fallback snapshotting cannot represent."""

            super().__init__()
            self.opaque_state = object()

        def __deepcopy__(self, memo: dict[int, object]) -> "UncopyableOpaqueState":
            """Reject isolated ground-truth copies."""

            del memo
            raise TypeError("cannot deepcopy model")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a stable output without using the opaque state."""

            return x + 1

    model = UncopyableOpaqueState()
    with pytest.warns(RuntimeWarning, match="skipping restoration for this attribute only"):
        fallback_model, snapshot = user_public_impls._model_for_ground_truth_validation(model)

    assert fallback_model is model
    assert snapshot is not None


def test_unsnapshotable_attr_restores_other_state_and_skips_pristine_retrace() -> None:
    """An opaque attr preserves replay while safely skipping only the new check."""

    class LockBackedToggle(nn.Module):
        """Stateful model with one unsnapshotable but unused lock."""

        def __init__(self) -> None:
            """Initialize the lock and branch counter."""

            super().__init__()
            self.lock = threading.Lock()
            self.step = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Change graph shape after the first execution."""

            output = x + 1 if self.step == 0 else x * 2
            self.step += 1
            return output

    diagnostics: list[tuple[ValidationDiagnostic, ...]] = []

    def observe_trace(trace: Trace) -> None:
        """Retain diagnostics before validation cleans up its trace."""

        diagnostics.append(get_validation_diagnostics(trace))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = user_public_impls._validate_forward_pass_torch(
            LockBackedToggle(),
            torch.randn(3),
            _trace_observer=observe_trace,
        )

    assert result is True
    assert any(
        "skipping restoration for this attribute only" in str(item.message) for item in caught
    )
    assert not any(issubclass(item.category, TraceNotReproducibleWarning) for item in caught)
    assert diagnostics[0][0].check == "trace_retrace_pristine_copy_unavailable"


def test_validate_forward_pass_ground_truth_copy_strips_traced_forward_wrappers() -> None:
    """Ground-truth validation must execute a previously traced nested-model copy."""

    class CountingChild(nn.Module):
        """Nested child with a source-visible forward counter."""

        def __init__(self) -> None:
            """Initialize the source execution counter."""

            super().__init__()
            self.executions = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Count each child forward execution."""

            self.executions += 1
            return x + 1

    class NestedModel(nn.Module):
        """Parent model used to install persistent TorchLens wrappers."""

        def __init__(self) -> None:
            """Initialize the nested child."""

            super().__init__()
            self.child = CountingChild()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Delegate to the nested child."""

            return self.child(x)

    model = NestedModel()
    prior_trace = trace_fn(model, torch.randn(2, 3))
    prior_trace.cleanup()
    model.child.executions = 0

    assert validate_forward_pass(model, torch.randn(2, 3)) is True
    assert model.child.executions == 0


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


def test_validation_dispatch_op_count_backstop_matches_normal_capture() -> None:
    """The dispatcher census agrees with a normal validation capture's op count."""

    class TwoOpModel(nn.Module):
        """Model with two independently dispatching captured operations."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply two tensor operations."""

            return torch.relu(x + 1)

    observed_counts: list[tuple[int, int]] = []

    def observe_trace(trace: Trace) -> None:
        """Save both operation counts before validation cleans up the trace."""

        observed_counts.append(
            (
                trace._validation_dispatch_op_count,
                trace._validation_captured_dispatchable_op_count,
            )
        )

    assert user_public_impls._validate_forward_pass_torch(
        TwoOpModel(),
        torch.randn(2, 3),
        validate_metadata=False,
        _trace_observer=observe_trace,
    )
    assert observed_counts == [(2, 2)]


def test_validation_dispatch_op_count_backstop_rejects_synthetic_missed_op() -> None:
    """A dispatcher/capture count mismatch is a hard completeness failure."""

    model = nn.Sequential(nn.ReLU()).eval()
    inputs = torch.randn(2, 3)
    trace = trace_fn(model, inputs, save_arg_values=True, save_rng_states=True)
    try:
        trace._validation_captured_dispatchable_op_count = len(
            {
                getattr(op, "func_call_id")
                for op in trace.layer_list
                if isinstance(getattr(op, "func_call_id", None), int)
            }
        )
        trace._validation_dispatch_op_count = trace._validation_captured_dispatchable_op_count + 1
        assert trace.validate_forward_pass([model(inputs)], validate_metadata=False) is False
        assert any(
            decision["reason"] == "dispatch_op_count_mismatch"
            for decision in trace.validation_replay_status.decisions
        )
    finally:
        trace.cleanup()


def test_validation_direct_aten_dispatch_drop_fails_the_public_gate() -> None:
    """Bug 1: a directly-dispatched aten op TorchLens dropped fails validation.

    ``torch.ops.aten.neg.default`` bypasses TorchLens wrapping, so ``neg`` is a
    real op that never enters the captured trace (only the following ``+ 1`` is
    captured). The completeness witness records it as an ``unowned_dispatch``; the
    public ``validate_forward_pass`` boolean MUST return ``False`` -- a detected
    real miss is exactly the silent drop the tripwire exists to catch.
    """

    class DirectAtenNeg(nn.Module):
        """Drop an op by calling a raw aten overload directly."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return ``-x + 1`` with the negate hidden from capture."""

            return torch.ops.aten.neg.default(x) + 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(DirectAtenNeg(), torch.tensor([1.0, -2.0, 3.0])) is False


def test_validation_same_shape_broadcast_tensors_validates() -> None:
    """Bug 2: a captured op that legitimately dispatches no aten op validates.

    ``torch.broadcast_tensors`` on already-equal shapes returns its inputs and
    dispatches no aten op, yet TorchLens captures the variadic call. That captured
    op has no dispatch counterpart by design; it must NOT be counted as a census
    mismatch on this correct, deterministic model.
    """

    class BenignBroadcast(nn.Module):
        """Broadcast two same-shape tensors (a dispatch-free capture)."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return the sum of two same-shape broadcast operands."""

            left, right = torch.broadcast_tensors(x + 1, x + 2)
            return left + right

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(BenignBroadcast(), torch.arange(4.0)) is True


def _raw_scale_replacement_hook(module, inputs, output):  # type: ignore[no-untyped-def]
    """Genuine raw output-replacement hook built from untraceable aten calls."""

    return torch.ops.aten.mul.Tensor(output, torch.tensor(0.5))


def test_validation_genuine_replacement_alone_validates() -> None:
    """Bug 3 control: a genuine output-replacement hook alone validates cleanly.

    The replacement's untraceable construction (raw ``aten.mul`` plus a
    python-wrapped ``torch.tensor`` orphaned out of the trace) all fires inside
    the torchlens ``wrapped_hook`` frame and is excused per-op, so a model whose
    only completeness residual is a genuine replacement passes.
    """

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.fc1(x))

    model = _Mlp().eval()
    model.relu.register_forward_hook(_raw_scale_replacement_hook)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, [torch.randn(3, 4)], input_kwargs={}) is True


def test_validation_genuine_replacement_plus_unrelated_drop_still_fails() -> None:
    """Bug 3: a real drop alongside a genuine replacement STILL fails the gate.

    The replacement carve-out is PER-OP scoped: it excuses only the untraceable
    dispatch that fired inside the replacement hook. An UNRELATED directly-
    dispatched aten drop in the forward body fires OUTSIDE the hook, so adding a
    legitimate output-replacement hook must NEVER flip that FAIL into a PASS.
    """

    class _MlpWithDrop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Unrelated real capture drop (raw aten in the forward body, NOT in a hook).
            hidden = torch.ops.aten.neg.default(self.fc1(x))
            return self.relu(hidden)

    model = _MlpWithDrop().eval()
    model.relu.register_forward_hook(_raw_scale_replacement_hook)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model, [torch.randn(3, 4)], input_kwargs={}) is False


def _backstop_op(
    func_call_id: int | None,
    *,
    transform_kind: str | None = None,
    func_name: str = "linear",
    intervention_replaced: bool = False,
    is_internal_source: bool = False,
) -> SimpleNamespace:
    """Build a minimal layer-op stand-in for the completeness backstop census."""

    return SimpleNamespace(
        func_call_id=func_call_id,
        transform_kind=transform_kind,
        func_name=func_name,
        intervention_replaced=intervention_replaced,
        is_internal_source=is_internal_source,
    )


def _backstop_dec(
    owner_func_call_id: int,
    *,
    capture_accounted: bool = True,
    in_replacement_hook: bool = False,
) -> dict:
    """Build a minimal dispatcher-census decomposition record."""

    return {
        "owner_func_call_id": owner_func_call_id,
        "capture_accounted": capture_accounted,
        "in_replacement_hook": in_replacement_hook,
    }


def _backstop_diag(
    *,
    reason: str = "unowned_dispatch",
    in_replacement_hook: bool = False,
    mutates: bool = False,
) -> dict:
    """Build a minimal unaccounted-dispatch witness diagnostic record."""

    return {"reason": reason, "in_replacement_hook": in_replacement_hook, "mutates": mutates}


def _backstop_trace(
    layer_list: list, decompositions: list, diagnostics: list | None = None
) -> SimpleNamespace:
    """Wrap census inputs in a trace-shaped object for the backstop helper."""

    return SimpleNamespace(
        layer_list=layer_list,
        completeness_decompositions=decompositions,
        completeness_diagnostics=diagnostics or [],
    )


def test_completeness_backstop_dispatchless_captured_op_is_benign() -> None:
    """A captured op with no owned aten dispatch is NOT a census mismatch (Bug 2).

    Both a ``torch.func`` transform boundary AND a benign no-op/view/meta call
    (e.g. same-shape ``torch.broadcast_tensors``) own a captured ``func_call_id``
    yet dispatch no aten op. Neither has a dispatch counterpart by design, so the
    captured census must exclude them -- counting them as "extra captured ops"
    false-fails a correct model.
    """

    # A transform boundary (fcid=3) owns a captured id but no owned dispatch -> match.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [
                _backstop_op(1),
                _backstop_op(2),
                _backstop_op(3, transform_kind="vmap", func_name="vmap"),
            ],
            [_backstop_dec(1), _backstop_dec(2)],
        )
    )
    assert (dispatch, captured) == (2, 2)

    # A NON-transform captured op (fcid=3) that dispatched no owned aten op is a
    # benign no-op (same-shape broadcast_tensors), NOT a gap -> match. This is the
    # exact false-positive the census must not raise.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [
                _backstop_op(1),
                _backstop_op(2),
                _backstop_op(3, transform_kind=None, func_name="broadcast_tensors"),
            ],
            [_backstop_dec(1), _backstop_dec(2)],
        )
    )
    assert dispatch == captured


def test_completeness_backstop_unowned_dispatch_fails_the_gate() -> None:
    """An UNOWNED aten dispatch is a real silent drop and MUST fail (Bug 1).

    A directly-dispatched ``torch.ops.aten.*`` call has no capturing owner, so the
    witness records an ``unowned_dispatch`` diagnostic. That real op is missing
    from the captured trace and must inflate the dispatch census so the backstop
    fails -- a captured op being present cannot mask it. An unowned dispatch that
    fired inside a genuine replacement hook is replacement construction, not a
    drop, and must NOT be counted.
    """

    # A clean capture (no unowned dispatch) matches.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
        )
    )
    assert dispatch == captured

    # An unowned dispatch outside a replacement hook is a real drop -> mismatch,
    # even though every captured op is accounted.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="unowned_dispatch", in_replacement_hook=False)],
        )
    )
    assert dispatch != captured

    # A real drop alongside a benign dispatchless captured op still fails (no
    # cancellation: the benign op is excluded, the unowned drop inflates dispatch).
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2), _backstop_op(3, func_name="broadcast_tensors")],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="unowned_dispatch", in_replacement_hook=False)],
        )
    )
    assert dispatch != captured

    # An unowned dispatch INSIDE a genuine replacement hook is construction, not a
    # drop -> match.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="unowned_dispatch", in_replacement_hook=True)],
        )
    )
    assert dispatch == captured

    # A benign ``owner_not_captured`` diagnostic (e.g. torch.equal control flow)
    # is NOT an unowned drop and must NOT fail the gate.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="owner_not_captured", in_replacement_hook=False)],
        )
    )
    assert dispatch == captured


def test_completeness_backstop_intervention_carveout_is_per_op_scoped() -> None:
    """The replacement carve-out is PER-OP scoped and never disarms plain capture.

    A genuine raw output-replacement hook builds its replacement with untraceable
    dispatch inside the torchlens ``wrapped_hook`` frame: python-wrapped calls
    become accounted owners orphaned out of the final trace (tagged
    ``in_replacement_hook``). Those exact orphaned owners are excused. But the
    exemption is scoped to the replacement's OWN ops -- an unrelated orphaned
    owner (a real silent drop) whose dispatch fired OUTSIDE the hook is NOT
    excused, and plain capture with no replacement at all is never relaxed.
    """

    # Orphaned owner (fcid=4) tagged in_replacement_hook -> excused -> match.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [
                _backstop_op(1),
                _backstop_op(3, func_name="relu"),
                _backstop_op(
                    None,
                    func_name="intervention_replacement",
                    intervention_replaced=True,
                ),
            ],
            [_backstop_dec(1), _backstop_dec(3), _backstop_dec(4, in_replacement_hook=True)],
        )
    )
    assert (dispatch, captured) == (2, 2)

    # Genuine replacement (orphan fcid=4 in-hook, excused) ALONGSIDE an UNRELATED
    # real drop (orphan fcid=6 OUTSIDE any hook) -> the unrelated drop still fires.
    # The presence of a replacement does NOT flip this FAIL to a PASS (Bug 3).
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [
                _backstop_op(1),
                _backstop_op(3, func_name="relu"),
                _backstop_op(
                    None,
                    func_name="intervention_replacement",
                    intervention_replaced=True,
                ),
            ],
            [
                _backstop_dec(1),
                _backstop_dec(3),
                _backstop_dec(4, in_replacement_hook=True),
                _backstop_dec(6, in_replacement_hook=False),
            ],
        )
    )
    assert dispatch != captured

    # SAME orphaned-dispatch shape during PLAIN capture (orphan not in any hook)
    # -> the tripwire STILL fires.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(3, func_name="relu")],
            [_backstop_dec(1), _backstop_dec(3), _backstop_dec(4, in_replacement_hook=False)],
        )
    )
    assert dispatch != captured

    # A non-genuine op merely NAMED intervention_replacement (its orphan never ran
    # in a real hook, so it is not tagged in_replacement_hook) does NOT enable the
    # carve-out -- the plain-capture drop still fires.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [
                _backstop_op(1),
                _backstop_op(3, func_name="relu"),
                _backstop_op(
                    None,
                    func_name="intervention_replacement",
                    intervention_replaced=False,
                ),
            ],
            [_backstop_dec(1), _backstop_dec(3), _backstop_dec(4, in_replacement_hook=False)],
        )
    )
    assert dispatch != captured


def test_completeness_backstop_mutating_owner_not_captured_fails_the_gate() -> None:
    """An uncaptured MUTATING owner_not_captured dispatch is a value-affecting drop.

    Round-3 strengthening: a benign ``owner_not_captured`` diagnostic (a wrapped
    op whose owner emitted no captured op) is legitimate PURE-READ control flow
    (``torch.equal`` / ``torch.allclose`` deciding a branch) and must NOT fail.
    But an ``owner_not_captured`` dispatch that MUTATES an argument is a
    value-affecting op the graph missed -- a genuine completeness failure that
    must inflate the dispatch census. The mutation signal (``mutates``) comes from
    the operator's own schema, so a pure-read comparison never trips it.
    """

    # Benign pure-read owner_not_captured (equal/allclose control flow) -> match.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="owner_not_captured", mutates=False)],
        )
    )
    assert dispatch == captured

    # A MUTATING owner_not_captured drop (e.g. an uncaptured in-place op that still
    # surfaced to the witness) is value-affecting -> mismatch -> FAIL.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="owner_not_captured", mutates=True)],
        )
    )
    assert dispatch != captured

    # A mutating drop alongside a benign captured no-op still fails (no cancel).
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2), _backstop_op(3, func_name="broadcast_tensors")],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="owner_not_captured", mutates=True)],
        )
    )
    assert dispatch != captured

    # A mutating owner_not_captured INSIDE a genuine replacement hook is
    # replacement construction, not a plain-capture drop -> excused -> match.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [_backstop_diag(reason="owner_not_captured", mutates=True, in_replacement_hook=True)],
        )
    )
    assert dispatch == captured

    # A legacy diagnostic dict with NO ``mutates`` key is treated as non-mutating
    # (benign) and must not spuriously fail -- backward compatibility.
    dispatch, captured = completeness_backstop_counts(
        _backstop_trace(
            [_backstop_op(1), _backstop_op(2)],
            [_backstop_dec(1), _backstop_dec(2)],
            [{"reason": "owner_not_captured", "in_replacement_hook": False}],
        )
    )
    assert dispatch == captured


def test_replay_validation_checks_every_recurrent_pass() -> None:
    """PIN: all five executions of a shared module are replay-validated.

    NOTE: this behavior already held before the 2026-07 dead-code removal (the
    pipeline enqueues pass-qualified labels); this test PINS it so it can never
    regress. It passes on the pre-refactor base by design.
    """

    class RecurrentLinear(nn.Module):
        """Apply one linear cell repeatedly."""

        def __init__(self) -> None:
            """Initialize the shared recurrent cell."""

            super().__init__()
            self.cell = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run five recurrent cell passes."""

            for _ in range(5):
                x = self.cell(x)
            return x

    model = RecurrentLinear()
    inputs = torch.randn(2, 3)
    trace = trace_fn(model, inputs, save_arg_values=True, save_rng_states=True)

    assert trace.validate_forward_pass([model(inputs)], validate_metadata=False) is True
    replayed_cell_passes = [
        decision
        for decision in trace.validation_replay_status.decisions
        if decision["phase"] == "replay"
        and decision["func_name"] == "linear"
        and decision["decision"] == "validated"
    ]
    assert len(replayed_cell_passes) == 5
    assert {decision["op_label"] for decision in replayed_cell_passes} == {
        "linear_1_1:1",
        "linear_1_1:2",
        "linear_1_1:3",
        "linear_1_1:4",
        "linear_1_1:5",
    }


def test_replay_validation_detects_corrupted_third_recurrent_pass_inputs() -> None:
    """Internal-helper hardening: bare-label direct calls validate every pass.

    The public pipeline already fails this corruption end-to-end (it enqueues
    pass-qualified labels). This test hardens the INTERNAL
    validate_parents_of_saved_layer entry point against bare multi-pass labels,
    which previously resolved through a single representative op.
    """

    class RecurrentLinear(nn.Module):
        """Apply one linear cell repeatedly."""

        def __init__(self) -> None:
            """Initialize the shared recurrent cell."""

            super().__init__()
            self.cell = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run five recurrent cell passes."""

            for _ in range(5):
                x = self.cell(x)
            return x

    model = RecurrentLinear()
    inputs = torch.randn(2, 3)
    trace = trace_fn(model, inputs, save_arg_values=True, save_rng_states=True)
    third_pass = next(op for op in trace.layer_list if op.label == "linear_1_1:3")
    assert third_pass.saved_args is not None
    third_pass.saved_args = (third_pass.saved_args[0] + 1.0, *third_pass.saved_args[1:])

    decision_recorder = ValidationDecisionRecorder()
    result = validate_parents_of_saved_layer(
        trace,
        "linear_1_1",
        set(),
        set(),
        defaultdict(set),
        deque(),
        decision_recorder=decision_recorder,
    )

    assert result.failed
    assert any(
        decision["op_label"] == "linear_1_1"
        and decision["phase"] == "replay"
        and decision["decision"] == "failed"
        for decision in decision_recorder.as_status().decisions
    )


def test_replay_validation_checks_every_train_batch_norm_pass() -> None:
    """PIN: train-mode BatchNorm replays all passes from pristine state.

    Passes on the pre-refactor base by design -- pins existing behavior against
    regression (see test_replay_validation_checks_every_recurrent_pass).
    """

    class RecurrentBatchNorm(nn.Module):
        """Apply one train-mode BatchNorm module repeatedly."""

        def __init__(self) -> None:
            """Initialize the shared normalization module."""

            super().__init__()
            self.batch_norm = nn.BatchNorm1d(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run five normalization passes."""

            for _ in range(5):
                x = self.batch_norm(x)
            return x

    model = RecurrentBatchNorm().train()
    inputs = torch.randn(8, 3)
    pristine_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    trace = trace_fn(model, inputs, save_arg_values=True, save_rng_states=True)
    model.load_state_dict(pristine_state)

    assert trace.validate_forward_pass([model(inputs)], validate_metadata=False) is True
    replayed_batch_norm_passes = [
        decision
        for decision in trace.validation_replay_status.decisions
        if decision["phase"] == "replay"
        and decision["func_name"] == "batch_norm"
        and decision["decision"] == "validated"
    ]
    assert len(replayed_batch_norm_passes) == 5


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
        output_layer = trace.layers[trace.output_layers[0]].ops[0]
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


def test_full_is_not_exempt_and_skip_perturbation_registry_is_pinned() -> None:
    """Keep ``full`` value-sensitive and pin the perturbation exemption registry."""

    assert sorted(SKIP_PERTURBATION_ENTIRELY) == [
        "broadcast_tensors",
        "deform_conv2d",
        "expand_as",
        "exponential_",
        "fill_",
        "full_like",
        "meshgrid",
        "new_ones",
        "new_zeros",
        "nms",
        "ones_like",
        "ps_roi_align",
        "ps_roi_pool",
        "rand_like",
        "randn_like",
        "roi_align",
        "roi_pool",
        "zero_",
        "zeros_like",
    ]
    assert "full" not in SKIP_VALIDATION_ENTIRELY
    assert "full" not in SKIP_PERTURBATION_ENTIRELY
    assert "full" not in CUSTOM_EXEMPTION_CHECKS
    assert "full" not in STRUCTURAL_ARG_POSITIONS


def test_copy_source_is_value_sensitive_and_destination_is_structural() -> None:
    """Healthy ``copy_`` validates while an ignored source still trips validation."""

    class CopySourceModel(nn.Module):
        """Copy a computed source into a fresh destination."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return values copied from a non-destination parent."""

            source = x + 2
            destination = torch.zeros_like(x)
            destination.copy_(source)
            return destination

    x = torch.tensor([2.0, 3.0, 4.0])
    healthy_trace = trace_fn(CopySourceModel(), x, save_arg_values=True)
    healthy_copy = next(op for op in healthy_trace.layer_list if op.func_name == "copy_")

    assert set(healthy_copy.parent_arg_positions["args"]) == {0, 1}
    assert (
        healthy_trace.validate_forward_pass(
            [healthy_trace[healthy_trace.output_layers[0]].out.detach().clone()],
            validate_metadata=False,
        )
        is True
    )

    broken_trace = trace_fn(CopySourceModel(), x, save_arg_values=True)
    broken_copy = next(op for op in broken_trace.layer_list if op.func_name == "copy_")
    saved_output = broken_copy.out.detach().clone()

    def ignore_source(destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Replay the saved bytes while deliberately ignoring the source parent."""

        del source
        return destination.copy_(saved_output)

    broken_copy.func = ignore_source
    broken_result = broken_trace.validate_forward_pass(
        [broken_trace[broken_trace.output_layers[0]].out.detach().clone()],
        validate_metadata=False,
    )

    assert broken_result is False
    assert broken_trace.validation_replay_status.state == "failed"
    assert any(
        decision["op_label"] == broken_copy.label
        and decision["phase"] == "perturbation"
        and decision["decision"] == "failed"
        for decision in broken_trace.validation_replay_status.decisions
    )


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


def test_perturbation_response_gate_treats_one_ulp_change_as_unequal() -> None:
    """The perturbation gate accepts only exact, NaN-aware output equality."""

    saved = torch.tensor([1.0], dtype=torch.float32)
    recomputed = torch.nextafter(saved, torch.tensor([float("inf")], dtype=torch.float32))

    assert not tensor_nanequal(recomputed, saved, allow_tolerance=False)
    assert tensor_nanequal(saved, saved.clone(), allow_tolerance=False)


@pytest.mark.smoke
def test_perturbation_changes_float_tensor() -> None:
    """Floating-point perturbation changes ordinary tensor values."""

    parent = torch.randn(10, 10)
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.shape == parent.shape


def test_perturbation_scales_near_constant_float_to_large_output() -> None:
    """Near-constant float perturbations scale up when child outputs are huge."""

    parent = torch.zeros(16, dtype=torch.float32)
    output = torch.full((16,), 1.0e30, dtype=torch.float32)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.abs().max() > 1.0e20
    assert not torch.equal(output - perturbed, output)


def test_perturbation_scales_tiny_float_range_to_large_output() -> None:
    """Tiny float ranges scale up when otherwise swallowed by huge operands."""

    parent = torch.linspace(-0.25, 0.25, 16, dtype=torch.float32)
    output = torch.full((16,), 1.0e30, dtype=torch.float32)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.abs().max() > 1.0e20
    assert not torch.equal(output + perturbed, output)


def test_validation_perturbs_zero_parent_at_large_float_scale() -> None:
    """Replay validation detects sensitivity when a zero parent meets a huge operand."""

    class HugeSubZero(nn.Module):
        """Model whose subtraction parent is zero but value-sensitive."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Subtract a data-derived zero tensor from a huge float tensor.

            Parameters
            ----------
            x:
                Input tensor used to shape the zero-valued parent.

            Returns
            -------
            torch.Tensor
                Huge float tensor with a zero-valued subtraction parent.
            """

            huge = torch.ones_like(x) * 1.0e30
            zero = x * 0.0
            return huge - zero

    assert validate_forward_pass(HugeSubZero(), torch.ones(2, 3), random_seed=123)


def test_perturbation_changes_int_tensor() -> None:
    """Integer perturbation changes tensor values while preserving dtype."""

    parent = torch.randint(0, 100, (10, 10))
    output = torch.randn(10, 10)
    perturbed = _perturb_layer_outs(parent, output)
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == parent.dtype


@pytest.mark.smoke
def test_perturbation_int64_saturated_max_does_not_overflow() -> None:
    """C1 regression: an int64 parent holding INT64_MAX must not crash randint.

    A legitimately captured int64 tensor that contains ``iinfo(int64).max``
    (common as PyG sentinel/cluster index values) used to make
    ``parent_outs.max() + 1`` wrap to ``INT64_MIN`` and raise
    ``RuntimeError("random_ expects 'from' to be less than 'to'...")``. The
    perturbation must run cleanly *and* still meaningfully perturb the tensor --
    a no-op would silently disarm the validation tripwire.
    """

    imax = torch.iinfo(torch.int64).max
    parent = torch.tensor([0, 5, imax, 100, imax, 42], dtype=torch.int64)
    output = torch.randn(parent.shape)

    perturbed = _perturb_layer_outs(parent, output)

    # (a) no raise (reached here), (b) genuinely perturbed, (c) dtype/shape kept.
    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.int64
    assert perturbed.shape == parent.shape


def test_perturbation_uint8_saturated_max_does_not_overflow() -> None:
    """C1 regression: a uint8 parent at its dtype max must clamp, not overflow."""

    umax = torch.iinfo(torch.uint8).max  # 255
    parent = torch.tensor([0, 5, umax, 10, umax], dtype=torch.uint8)
    output = torch.randn(parent.shape)

    perturbed = _perturb_layer_outs(parent, output)

    assert not torch.equal(perturbed, parent)
    assert perturbed.dtype == torch.uint8
    assert perturbed.shape == parent.shape


def test_sagpooling_validates_end_to_end_without_perturb_overflow() -> None:
    """C1 golden-model gate: a PyG SAGPooling graph validates green end-to-end.

    SAGPooling emits an int64 index tensor carrying ``INT64_MAX`` during its
    top-k selection. Before the fix this made the perturbation helper raise on a
    successfully-captured trace (a validation-machinery crash masking 21 PyG
    models). The tripwire must now run to a real pass/fail with no crash.
    """

    pytest.importorskip("torch_geometric")
    from torch_geometric.nn import GCNConv, SAGPooling

    torch.manual_seed(0)

    class SAGPoolNet(nn.Module):
        """Minimal GCN + SAGPooling graph that exercises the int64 sentinel path."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = GCNConv(8, 16)
            self.pool = SAGPooling(16, ratio=0.5)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            """Embed nodes, then pool the graph and return the pooled features."""

            x = self.conv(x, edge_index).relu()
            x, edge_index, _, _, _, _ = self.pool(x, edge_index)
            return x

    n_nodes = 10
    x = torch.randn(n_nodes, 8)
    edge_index = torch.randint(0, n_nodes, (2, 30), dtype=torch.long)

    # Returns True: the perturbation tripwire ran end-to-end with no overflow
    # crash and the replay/perturbation checks genuinely passed.
    assert validate_forward_pass(SAGPoolNet(), (x, edge_index), random_seed=1)


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


def test_deep_clone_preserves_namedtuple_type() -> None:
    """Namedtuple containers are reconstructed with positional fields."""

    point_type = namedtuple("Point", ["x", "y"])
    original = point_type(torch.tensor([1.0]), torch.tensor([2.0]))

    cloned = _deep_clone_tensors(original)

    assert isinstance(cloned, point_type)
    assert torch.equal(cloned.x, original.x)
    assert torch.equal(cloned.y, original.y)


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


class _GatherIndexModel(nn.Module):
    """Model that uses a tensor parent as a gather index."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather from ``x`` using a derived structural index tensor.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Gathered tensor plus a data dependency on ``x``.
        """

        index = torch.argmax(x, dim=1, keepdim=True).expand(-1, 2)
        gathered = torch.gather(x, 1, index)
        return gathered + x[:, :2]


class _FunctionalScatterKwargsModel(nn.Module):
    """Model that uses functional scatter with keyword index and src tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scatter values into a destination whose prior values are overwritten.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scatter result plus a data dependency on ``x``.
        """

        order = torch.argsort(x, dim=-1)
        src = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device).expand_as(x)
        dest = torch.zeros_like(x)
        scattered = dest.scatter(dim=-1, index=order, src=src)
        return scattered + x


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


class _WhereDifferentBranchesModel(nn.Module):
    """Model whose ``where`` branches are value-sensitive and non-equal."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select between distinct true and false branches.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Result of a value-sensitive ``torch.where`` call.
        """

        condition = x > 0
        true_branch = x + 10.0
        false_branch = x - 10.0
        return torch.where(condition, true_branch, false_branch)


class _WhereBranchSelectionModel(nn.Module):
    """Model whose ``where`` condition selectedness is configurable."""

    def __init__(self, condition_kind: str) -> None:
        """Initialize the condition kind.

        Parameters
        ----------
        condition_kind:
            One of ``"all_true"``, ``"all_false"``, or ``"mixed"``.
        """

        super().__init__()
        self.condition_kind = condition_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select between distinct branches with the configured condition.

        Parameters
        ----------
        x:
            Input tensor of shape ``(2, 3)``.

        Returns
        -------
        torch.Tensor
            Output of ``torch.where``.
        """

        if self.condition_kind == "all_true":
            condition = torch.ones((), dtype=torch.bool, device=x.device)
        elif self.condition_kind == "all_false":
            condition = torch.zeros((), dtype=torch.bool, device=x.device)
        elif self.condition_kind == "mixed":
            condition = torch.tensor(
                [[True, False, True], [False, True, False]],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            raise ValueError(f"Unknown condition kind: {self.condition_kind}")
        true_branch = x + 10.0
        false_branch = x - 10.0
        return torch.where(condition, true_branch, false_branch)


class _WhereSharedBranchModel(nn.Module):
    """Model passing the same branch tensor to both ``where`` value slots."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a mixed-condition ``where`` to the same branch tensor twice.

        Parameters
        ----------
        x:
            Input tensor of shape ``(2, 3)``.

        Returns
        -------
        torch.Tensor
            Output of ``torch.where``.
        """

        condition = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
            device=x.device,
        )
        branch = x + 1.0
        return torch.where(condition, branch, branch)


class _MaskedFillBranchSelectionModel(nn.Module):
    """Model whose ``masked_fill`` mask selectedness is configurable."""

    def __init__(self, mask_kind: str) -> None:
        """Initialize the mask kind.

        Parameters
        ----------
        mask_kind:
            One of ``"all_true"``, ``"all_false"``, or ``"mixed"``.
        """

        super().__init__()
        self.mask_kind = mask_kind

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``masked_fill`` with a 0-d tensor fill-value parent.

        Parameters
        ----------
        x:
            Input tensor of shape ``(2, 3)``.

        Returns
        -------
        torch.Tensor
            Output of ``x.masked_fill(mask, value)``.
        """

        if self.mask_kind == "all_true":
            mask = torch.ones_like(x, dtype=torch.bool)
        elif self.mask_kind == "all_false":
            mask = torch.zeros_like(x, dtype=torch.bool)
        elif self.mask_kind == "mixed":
            mask = torch.tensor(
                [[True, False, True], [False, True, False]],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            raise ValueError(f"Unknown mask kind: {self.mask_kind}")
        value = x.mean()
        return x.masked_fill(mask, value)


class _UnselectedWherePlaceholderModel(nn.Module):
    """Model with an unselected ``where`` branch used for placeholder tripwires."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route a real parent through an entirely unselected false branch.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output of a ``where`` whose false branch is never selected.
        """

        condition = torch.ones_like(x, dtype=torch.bool)
        selected = x + 1.0
        unselected = x - 1.0
        return torch.where(condition, selected, unselected)


class _RemainderDividendAtLeastDivisorModel(nn.Module):
    """Model where the remainder divisor must remain value-sensitive."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute a remainder whose output differs from the dividend.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Remainder output plus the input to keep the op live.
        """

        dividend = x.abs() + 5.0
        divisor = torch.full_like(dividend, 2.0)
        return torch.remainder(dividend, divisor) + x


class _NewTensorDataArgModel(nn.Module):
    """Model whose ``new_tensor`` data argument comes from a real parent op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a tensor from value data, not only from a template.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Data-derived ``new_tensor`` result.
        """

        data = x + 1.0
        return x.new_tensor(data) * 2.0


class _PopulatedContainerOutputModel(nn.Module):
    """Model returning a populated tuple/dict container of tensor leaves."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return tensor leaves through a populated nested container.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, dict[str, torch.Tensor]]
            Two real output leaves with non-empty container paths.
        """

        left = x + 1.0
        right = x * 2.0
        return left, {"right": right}


class _EmptyLikeModel(nn.Module):
    """Model that uses empty_like (tests SKIP_VALIDATION_ENTIRELY)."""

    def forward(self, x):
        # empty_like output is nondeterministic — don't use it in computation
        _ = torch.empty_like(x)
        return x * 2


class _InputDerivedFullModel(nn.Module):
    """Model whose ``full`` fill value is derived from the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fill an input-shaped tensor with the first input value.

        Parameters
        ----------
        x:
            Input tensor supplying the output shape and fill value.

        Returns
        -------
        torch.Tensor
            Tensor created by ``torch.full`` from an input-derived value.
        """

        return torch.full(x.shape, x[0])


def _only_layer_with_func_name(trace: Trace, func_name: str) -> Any:
    """Return the only layer in ``trace`` with the requested function name.

    Parameters
    ----------
    trace:
        Trace to search.
    func_name:
        Function name to locate.

    Returns
    -------
    Any
        The unique matching layer.
    """

    matches = [layer for layer in trace.layer_list if layer.func_name == func_name]
    assert len(matches) == 1
    return matches[0]


def _assert_custom_exemption_for_arg(
    model: nn.Module,
    x: torch.Tensor,
    func_name: str,
    arg_position: int,
    expected: bool,
) -> None:
    """Assert direct and registry perturbation exemption results for one arg.

    Parameters
    ----------
    model:
        Model to trace and validate.
    x:
        Input tensor.
    func_name:
        Captured function name to inspect.
    arg_position:
        Positional parent arg slot to perturb.
    expected:
        Expected exemption result.
    """

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        layer = _only_layer_with_func_name(trace, func_name)
        parent = layer.parent_arg_positions["args"][arg_position]

        assert CUSTOM_EXEMPTION_CHECKS[func_name](trace, layer, [parent]) is expected
        assert _check_perturbation_exemptions(trace, layer, [parent]) is expected
    finally:
        trace.cleanup()


def test_input_derived_full_validates_without_an_exemption() -> None:
    """``full`` is replay-validated through the pipeline rather than exempted."""

    model = _InputDerivedFullModel()
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        assert trace.validate_forward_pass([model(x).detach().clone()]) is True

        full_decisions = [
            decision
            for decision in trace.validation_replay_status.decisions
            if decision.get("func_name") == "full"
        ]
        assert full_decisions == [
            {
                "op_label": _only_layer_with_func_name(trace, "full").layer_label + ":1",
                "func_name": "full",
                "phase": "replay",
                "decision": "validated",
                "reason": "replay_matched",
            }
        ]
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Reduction-depth band-C eligibility predicate (validation/core.py)
#
# Band C grants a relaxed deep-numeric replay tolerance ONLY to ops that are
# reductions with per-output accumulation depth >= 64 (the depth at which FP32
# reorder round-off genuinely accrues). The eligibility is the FAITHFUL reduction
# depth, NOT a graph-position (step_index) proxy NOR a conv/matmul func allowlist.
# The tests below are the LOAD-BEARING gate: every injected-error case on a shallow
# op MUST be caught (the predicate must NOT mask it), and a gross error on a deep
# eligible op MUST still fail via band C's numeric caps. If any false-positive case
# below ever returns True, the tripwire is disarmed -- DO NOT loosen the test.
# ---------------------------------------------------------------------------


def _make_deep_numeric_layer(
    func_name: str,
    saved_args: list[Any],
    out: torch.Tensor,
    saved_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a minimal op stand-in for deep-numeric replay tests.

    ``_op_reduction_depth`` and ``_deep_numeric_replay_matches_saved`` read only
    ``func_name``, ``saved_args``, ``saved_kwargs`` and ``out``, so a namespace
    suffices and keeps the predicate tests fast and capture-independent.
    """

    return SimpleNamespace(
        func_name=func_name,
        saved_args=saved_args,
        saved_kwargs=saved_kwargs or {},
        out=out,
    )


def test_op_reduction_depth_reads_all_reduction_categories() -> None:
    """Per-output accumulation depth is read for every reduction category."""

    # conv2d: weight (out=4, in/groups=8, kH=3, kW=3) -> 8*3*3 = 72 accumulations.
    conv_layer = _make_deep_numeric_layer(
        "conv2d", [torch.zeros(1, 8, 5, 5), torch.zeros(4, 8, 3, 3)], torch.zeros(1, 4, 3, 3)
    )
    assert _op_reduction_depth(conv_layer) == 72

    # linear: weight (out=16, in=128) -> contracted K = 128.
    linear_layer = _make_deep_numeric_layer(
        "linear", [torch.zeros(2, 128), torch.zeros(16, 128)], torch.zeros(2, 16)
    )
    assert _op_reduction_depth(linear_layer) == 128

    # matmul: (n, K) @ (K, m); first operand last dim is K = 64.
    matmul_layer = _make_deep_numeric_layer(
        "matmul", [torch.zeros(3, 64), torch.zeros(64, 5)], torch.zeros(3, 5)
    )
    assert _op_reduction_depth(matmul_layer) == 64

    # scatter_add: max duplicate-index fan-in. Index 0 appears 1024 times -> 1024.
    deep_index = torch.zeros(1024, dtype=torch.long)
    scatter_layer = _make_deep_numeric_layer(
        "scatter_add", [torch.zeros(5), 0, deep_index, torch.ones(1024)], torch.zeros(5)
    )
    assert _op_reduction_depth(scatter_layer) == 1024

    # sum over a dim: numel of the reduced dim.
    sum_layer = _make_deep_numeric_layer("sum", [torch.zeros(3, 128)], torch.zeros(3), {"dim": 1})
    assert _op_reduction_depth(sum_layer) == 128

    # elementwise / structural ops have depth 1 and never reach the threshold.
    assert (
        _op_reduction_depth(_make_deep_numeric_layer("relu", [torch.zeros(4)], torch.zeros(4))) == 1
    )
    assert (
        _op_reduction_depth(_make_deep_numeric_layer("add", [torch.zeros(4)], torch.zeros(4))) == 1
    )


def test_op_reduction_depth_reads_keyword_passed_operands() -> None:
    """Depth must read operands passed by keyword, not just positionally.

    A ``conv2d(x, weight=w)`` / ``matmul(a, other=b)`` / ``scatter_add(out, dim,
    index=idx, src=s)`` call leaves the positional slot empty; reading positionally
    only would drop depth to 0 and wrongly withhold band C (a fresh false-negative).
    """

    conv_layer = _make_deep_numeric_layer(
        "conv2d",
        [torch.zeros(1, 8, 5, 5)],
        torch.zeros(1, 4, 3, 3),
        saved_kwargs={"weight": torch.zeros(4, 8, 3, 3)},
    )
    assert _op_reduction_depth(conv_layer) == 72

    linear_layer = _make_deep_numeric_layer(
        "linear",
        [torch.zeros(2, 128)],
        torch.zeros(2, 16),
        saved_kwargs={"weight": torch.zeros(16, 128)},
    )
    assert _op_reduction_depth(linear_layer) == 128

    matmul_layer = _make_deep_numeric_layer(
        "matmul",
        [torch.zeros(3, 64)],
        torch.zeros(3, 5),
        saved_kwargs={"other": torch.zeros(64, 5)},
    )
    assert _op_reduction_depth(matmul_layer) == 64

    deep_index = torch.zeros(1024, dtype=torch.long)
    scatter_layer = _make_deep_numeric_layer(
        "scatter_add",
        [torch.zeros(5), 0],
        torch.zeros(5),
        saved_kwargs={"index": deep_index, "src": torch.ones(1024)},
    )
    assert _op_reduction_depth(scatter_layer) == 1024


def test_op_reduction_depth_undeterminable_is_zero_fail_toward_strict() -> None:
    """When depth cannot be determined the predicate returns 0 (ineligible)."""

    assert _op_reduction_depth(_make_deep_numeric_layer("conv2d", [], torch.zeros(1))) == 0
    assert _op_reduction_depth(_make_deep_numeric_layer("matmul", [], torch.zeros(1))) == 0
    # scatter with no index operand cannot be measured.
    assert (
        _op_reduction_depth(
            _make_deep_numeric_layer("scatter_add", [torch.zeros(5), 0], torch.zeros(5))
        )
        == 0
    )


def test_deep_numeric_replay_recovers_deep_conv_cancellation_drift() -> None:
    """A physically-deep conv accepts sub-1e-4 replay drift (the resnet recovery).

    Mirrors the BiT ResNet-v2 (resnetv2_50x3_bit) false-negative: a wide,
    high-channel conv replays to FP32 round-off (abs error well under the
    deep-numeric atol of 1e-4 at activation scale ~90), concentrated on near-zero
    cancelling outputs. This is the LEGIT deep-conv cancellation drifter that MUST
    pass; the old step_index gate refused band C for the early such convs.
    """

    weight = torch.zeros(1536, 256, 1, 1)  # depth 256 >> threshold.
    saved_out = torch.empty(1, 1536, 8, 8).uniform_(-90.0, 90.0)
    recomputed = saved_out + torch.empty_like(saved_out).uniform_(-8.0e-5, 8.0e-5)
    layer = _make_deep_numeric_layer("conv2d", [torch.zeros(1, 256, 8, 8), weight], saved_out)

    assert _op_reduction_depth(layer) >= DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    # Strict band B (allow_tolerance) rejects this drift...
    assert not tensor_nanequal(recomputed, saved_out, allow_tolerance=True)
    # ...but band C accepts it for a deep conv (the recovery).
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is True


def test_deep_numeric_replay_false_positive_shallow_scatter_depth2() -> None:
    """LOAD-BEARING: a depth-2 scatter + ~5e-5 injection must NOT be masked.

    Codex's exact false-positive case. A scatter with a single duplicate index has
    fan-in depth 2 -- far below the threshold -- so band C is withheld and the
    injected error is caught by strict band B. (The category-arm predicate that
    granted band C to every scatter regardless of depth is what masked this; the
    depth-over-all-reductions predicate does not.)
    """

    index = torch.tensor([0, 0, 1, 2])  # index 0 duplicated -> fan-in depth 2.
    saved_out = torch.empty(3, 1).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed[0, 0] += 5.0e-5  # injected capture error.
    layer = _make_deep_numeric_layer(
        "scatter_add_", [torch.zeros(3, 1), 0, index.unsqueeze(1), torch.ones(4, 1)], saved_out
    )

    assert _op_reduction_depth(layer) < DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False


def test_deep_numeric_replay_false_positive_shallow_linear_depth16() -> None:
    """LOAD-BEARING: a depth-16 linear + 5e-5 injection must NOT be masked.

    A late shallow linear (the GNN bug-masking population the old step_index gate
    parachuted) is depth 16 < 64 -> band C withheld -> the injected error is caught.
    """

    saved_out = torch.empty(2, 8).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed[0, 0] += 5.0e-5
    layer = _make_deep_numeric_layer("linear", [torch.zeros(2, 16), torch.zeros(8, 16)], saved_out)

    assert _op_reduction_depth(layer) < DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False


def test_deep_numeric_replay_gross_error_on_deep_scatter_still_fails() -> None:
    """LOAD-BEARING: a deep ELIGIBLE scatter (depth 1024) + gross +10 still fails.

    Eligibility (depth >= 64) does NOT make band C unbounded: its outlier-fraction
    and scaled-diff caps reject a gross injected error even on a deep op.
    """

    deep_index = torch.zeros(1024, dtype=torch.long)
    saved_out = torch.empty(64).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed += 10.0  # gross error on every element.
    layer = _make_deep_numeric_layer(
        "scatter_add", [torch.zeros(64), 0, deep_index, torch.ones(1024)], saved_out
    )

    assert _op_reduction_depth(layer) >= DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False


def test_deep_numeric_replay_gross_error_on_deep_conv_still_fails() -> None:
    """LOAD-BEARING: a deep ELIGIBLE conv (depth 1536) + gross +10 still fails."""

    weight = torch.zeros(64, 1536, 1, 1)  # depth 1536 >> threshold.
    saved_out = torch.empty(1, 64, 4, 4).uniform_(-1.0, 1.0)
    recomputed = saved_out + 10.0
    layer = _make_deep_numeric_layer("conv2d", [torch.zeros(1, 1536, 4, 4), weight], saved_out)

    assert _op_reduction_depth(layer) >= DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False


def test_op_reduction_depth_scatter_fan_in_is_per_destination_coordinate() -> None:
    """LOAD-BEARING: an n-D scatter fan-in is counted PER destination coordinate.

    Codex's exact false-positive: a row-/feature-wise ``scatter_add`` writes many
    INDEPENDENT depth-2 destinations all at raw index ``0``. Counting the duplicate
    raw index value ``0`` GLOBALLY conflates them into one huge fan-in and wrongly
    grants band C to a genuinely shallow (depth-2) scatter, masking the ~5e-5 error
    class. Counting per actual destination COORDINATE reports depth 2 -> ineligible.
    """

    destination = torch.zeros(100, 3)
    index = torch.zeros(100, 2, dtype=torch.long)  # every row scatters to col 0.
    src = torch.ones(100, 2)
    saved_out = destination.scatter_add(1, index, src)  # each row col0 == 2.0.
    recomputed = saved_out.clone()
    recomputed[0, 0] += 5.0e-5  # injected capture error.
    layer = _make_deep_numeric_layer("scatter_add", [destination, 1, index, src], saved_out)

    # Two sources per destination -> depth 2 regardless of 200 raw duplicate zeros.
    assert _op_reduction_depth(layer) == 2
    assert _op_reduction_depth(layer) < DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False

    # A genuinely deep per-coordinate fan-in stays eligible.
    deep_index = torch.zeros(10, 200, dtype=torch.long)
    deep_src = torch.ones(10, 200)
    deep_dst = torch.zeros(10, 3)
    deep_out = deep_dst.scatter_add(1, deep_index, deep_src)
    deep_layer = _make_deep_numeric_layer(
        "scatter_add", [deep_dst, 1, deep_index, deep_src], deep_out
    )
    assert _op_reduction_depth(deep_layer) == 200  # 200 sources per (row, 0).
    assert _op_reduction_depth(deep_layer) >= DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH


def test_op_reduction_depth_conv_transpose_is_shallow() -> None:
    """LOAD-BEARING: transposed-conv depth uses the transposed weight layout.

    ``conv_transpose`` weight is ``[in_channels, out_channels/groups, *kernel]``;
    reusing the forward formula ``weight.numel() // weight.shape[0]`` reads the
    out-channel axis and over-reports. ``ConvTranspose2d(1, 128, 1)`` accumulates
    only ONE input element per output (depth 1), so it must be shallow/ineligible.
    """

    conv_t = nn.ConvTranspose2d(1, 128, kernel_size=1)
    saved_out = torch.empty(1, 128, 8, 8).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed.view(-1)[0] += 5.0e-5
    layer = _make_deep_numeric_layer(
        "conv_transpose2d", [torch.zeros(1, 1, 8, 8), conv_t.weight], saved_out
    )

    assert _op_reduction_depth(layer) == 1  # in_channels(1) * prod(kernel)(1).
    assert _op_reduction_depth(layer) < DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH
    assert _deep_numeric_replay_matches_saved(layer, recomputed) is False

    # A genuinely deep transposed conv (many in-channels) stays eligible.
    deep_conv_t = nn.ConvTranspose2d(256, 4, kernel_size=3)  # weight [256, 4, 3, 3].
    deep_out = torch.empty(1, 4, 10, 10).uniform_(-1.0, 1.0)
    deep_layer = _make_deep_numeric_layer(
        "conv_transpose2d", [torch.zeros(1, 256, 8, 8), deep_conv_t.weight], deep_out
    )
    assert _op_reduction_depth(deep_layer) == 256 * 9  # in_channels * prod(kernel).
    assert _op_reduction_depth(deep_layer) >= DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH


def test_op_reduction_depth_overwrite_scatter_is_ineligible() -> None:
    """LOAD-BEARING: plain overwrite ``scatter``/``scatter_`` get no band C.

    A plain scatter OVERWRITES the destination (last write wins) -- there is no FP32
    accumulation, so its replay must stay on the strict global tolerance. The deep
    duplicate-index pattern that would qualify an additive scatter must NOT qualify
    an overwrite scatter; it falls through to depth 1.
    """

    deep_index = torch.zeros(1024, dtype=torch.long)
    saved_out = torch.empty(64).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed[0] += 5.0e-5
    for func_name in ("scatter", "scatter_"):
        layer = _make_deep_numeric_layer(
            func_name, [torch.zeros(64), 0, deep_index, torch.ones(1024)], saved_out
        )
        assert _op_reduction_depth(layer) == 1, func_name
        assert _deep_numeric_replay_matches_saved(layer, recomputed) is False


def test_op_reduction_depth_non_additive_scatter_reduce_is_ineligible() -> None:
    """LOAD-BEARING: ``scatter_reduce`` max/min/prod modes get no band C.

    Only the ADDITIVE ``sum``/``mean`` reduce-modes accumulate FP32 products and
    drift with summation order; ``amax``/``amin``/``prod`` select or multiply and do
    NOT exhibit accumulation-order round-off, so they stay strict (depth 0). The
    additive modes remain eligible. The reduce mode is read from the ``reduce``
    keyword (TorchLens-captured form) or positional arg 4 (free-function form).
    """

    deep_index = torch.zeros(1024, dtype=torch.long)
    saved_out = torch.empty(64).uniform_(-1.0, 1.0)
    recomputed = saved_out.clone()
    recomputed[0] += 5.0e-5
    base_args = [torch.zeros(64), 0, deep_index, torch.ones(1024)]

    for mode in ("amax", "amin", "prod"):
        kw_layer = _make_deep_numeric_layer(
            "scatter_reduce", list(base_args), saved_out, saved_kwargs={"reduce": mode}
        )
        assert _op_reduction_depth(kw_layer) == 0, f"{mode} kwarg"
        assert _deep_numeric_replay_matches_saved(kw_layer, recomputed) is False
        pos_layer = _make_deep_numeric_layer("scatter_reduce", [*base_args, mode], saved_out)
        assert _op_reduction_depth(pos_layer) == 0, f"{mode} positional"

    for mode in ("sum", "mean"):
        kw_layer = _make_deep_numeric_layer(
            "scatter_reduce", list(base_args), saved_out, saved_kwargs={"reduce": mode}
        )
        assert _op_reduction_depth(kw_layer) == 1024, f"{mode} kwarg eligible"
        pos_layer = _make_deep_numeric_layer("scatter_reduce", [*base_args, mode], saved_out)
        assert _op_reduction_depth(pos_layer) == 1024, f"{mode} positional eligible"


def test_validation_with_getitem_tensor_index():
    model = _GetItemTensorIndex()
    x = torch.randn(5, 3)
    assert validate_forward_pass(model, x)


def test_validation_with_scatter():
    model = _ScatterModel()
    x = torch.randn(3, 5)
    assert validate_forward_pass(model, x)


def test_validation_with_gather_index_parent() -> None:
    """Validate gather index tensors as structural perturbation parents."""

    model = _GatherIndexModel()
    x = torch.randn(4, 5)
    assert validate_forward_pass(model, x, random_seed=123)


def test_validation_with_functional_scatter_kwargs_full_overwrite() -> None:
    """Validate functional scatter when index fully overwrites destination."""

    model = _FunctionalScatterKwargsModel()
    x = torch.randn(5, 7)
    assert validate_forward_pass(model, x, random_seed=123)


def test_validation_with_masked_fill():
    model = _MaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_validation_with_functional_masked_fill() -> None:
    """Validate non-in-place masked_fill boolean masks as structural args."""
    model = _FunctionalMaskedFillModel()
    x = torch.randn(4, 4)
    assert validate_forward_pass(model, x)


def test_masked_fill_mixed_mask_input_branch_is_not_exempt() -> None:
    """A mixed ``masked_fill`` mask leaves the input branch value-sensitive."""

    model = _MaskedFillBranchSelectionModel("mixed")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 0, False)


def test_masked_fill_mixed_mask_fill_value_branch_is_not_exempt() -> None:
    """A mixed ``masked_fill`` mask leaves the fill-value branch value-sensitive."""

    model = _MaskedFillBranchSelectionModel("mixed")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 2, False)


def test_masked_fill_all_true_mask_input_branch_is_exempt() -> None:
    """An all-true saved mask overwrites every input element."""

    model = _MaskedFillBranchSelectionModel("all_true")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 0, True)


def test_masked_fill_all_true_mask_fill_value_branch_is_not_exempt() -> None:
    """An all-true saved mask selects the fill-value branch everywhere."""

    model = _MaskedFillBranchSelectionModel("all_true")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 2, False)


def test_masked_fill_all_false_mask_fill_value_branch_is_exempt() -> None:
    """An all-false saved mask never selects the tensor fill-value branch."""

    model = _MaskedFillBranchSelectionModel("all_false")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 2, True)


def test_masked_fill_all_false_mask_input_branch_is_not_exempt() -> None:
    """An all-false saved mask selects the input branch everywhere."""

    model = _MaskedFillBranchSelectionModel("all_false")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "masked_fill", 0, False)


def test_validation_with_setitem_slice_full_overwrite() -> None:
    """Perturbing a fully overwritten ``__setitem__`` destination slice is exempt."""

    class SetItemSliceOverwriteModel(nn.Module):
        """Model that overwrites one destination slice with a value tensor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Assign a replacement tensor into a selected destination slice.

            Parameters
            ----------
            x:
                Input tensor with a singleton slice dimension.

            Returns
            -------
            torch.Tensor
                Tensor after an indexed slice assignment.
            """

            destination = x.clone()
            replacement = x[:, :, 0, :, :] + 1.0
            destination[:, :, 0, :, :] = replacement
            return destination

    model = SetItemSliceOverwriteModel()
    x = torch.randn(1, 3, 1, 4, 4)

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        setitem_layer = _only_layer_with_func_name(trace, "__setitem__")
        destination_parent = setitem_layer.parent_arg_positions["args"][0]
        replacement_parent = setitem_layer.parent_arg_positions["args"][2]

        assert _check_perturbation_exemptions(trace, setitem_layer, [destination_parent]) is True
        assert _check_perturbation_exemptions(trace, setitem_layer, [replacement_parent]) is False
    finally:
        trace.cleanup()


def test_validation_with_index_put_destination_full_overwrite() -> None:
    """Perturbing a fully overwritten ``index_put_`` destination is exempt.

    The destination's every row is overwritten by ``index_put_`` (accumulate
    False), so its prior value is provably irrelevant -- the exact analogue of
    the ``__setitem__`` destination-overwrite carve-out. Perturbing the VALUE
    parent must NOT be exempt.
    """

    class IndexPutOverwriteModel(nn.Module):
        """Model that fully overwrites a destination tensor via ``index_put_``."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Overwrite every destination row with a replacement tensor.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after a full ``index_put_`` overwrite.
            """

            destination = x.clone() + 0.5
            idx = torch.arange(x.shape[0])
            replacement = x[torch.arange(x.shape[0])] + 1.0
            destination.index_put_((idx,), replacement)
            return destination

    model = IndexPutOverwriteModel()
    x = torch.randn(4, 5)

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        index_put_layer = _only_layer_with_func_name(trace, "index_put_")
        destination_parent = index_put_layer.parent_arg_positions["args"][0]
        replacement_parent = index_put_layer.parent_arg_positions["args"][2]

        assert (
            CUSTOM_EXEMPTION_CHECKS["index_put_"](trace, index_put_layer, [destination_parent])
            is True
        )
        assert _check_perturbation_exemptions(trace, index_put_layer, [destination_parent]) is True
        assert _check_perturbation_exemptions(trace, index_put_layer, [replacement_parent]) is False
    finally:
        trace.cleanup()


def test_index_put_partial_overwrite_destination_is_not_exempt() -> None:
    """A genuinely-influential ``index_put_`` destination must still be perturbed.

    Only one row is overwritten; the un-indexed rows keep their prior value and
    flow to the output, so the destination is influential. The exemption must NOT
    fire (tripwire), and validation must still detect real sensitivity.
    """

    class IndexPutPartialModel(nn.Module):
        """Model that overwrites only part of its ``index_put_`` destination."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Overwrite a single destination row, leaving the rest intact.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)`` with ``rows > 1``.

            Returns
            -------
            torch.Tensor
                Destination with one overwritten row and surviving prior rows.
            """

            destination = x.clone() + 0.5
            idx = torch.tensor([0])
            replacement = x[[0]] + 1.0
            destination.index_put_((idx,), replacement)
            return destination

    model = IndexPutPartialModel()
    x = torch.randn(4, 5)

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        index_put_layer = _only_layer_with_func_name(trace, "index_put_")
        destination_parent = index_put_layer.parent_arg_positions["args"][0]

        # Tripwire: the influential (partially-surviving) destination is NOT exempt.
        assert (
            CUSTOM_EXEMPTION_CHECKS["index_put_"](trace, index_put_layer, [destination_parent])
            is False
        )
        assert _check_perturbation_exemptions(trace, index_put_layer, [destination_parent]) is False
    finally:
        trace.cleanup()


def test_index_put_value_parent_equal_to_destination_is_not_exempt() -> None:
    """LOAD-BEARING: a VALUE parent equal to the destination must still be perturbed.

    The index_put exemption identified its destination via ``torch.equal(perturbed,
    destination)`` alone; if a value-parent's CONTENTS happen to equal the
    destination, perturbing the value-parent was falsely exempted -- masking a
    genuine sensitivity. The exemption must require the perturbed parent to be the
    destination by ARG POSITION (``parent_arg_positions["args"][0]``), mirroring how
    ``__setitem__`` identifies its destination. Here destination and value share
    identical contents, so only arg-position disambiguates them.
    """

    class IndexPutValueEqualsDestModel(nn.Module):
        """Overwrite a destination with a value tensor of identical contents."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Build destination and values with EQUAL contents, then overwrite.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after a full ``index_put_`` overwrite by an
                equal-content value parent.
            """

            # Non-constant shared contents: cert10 removed the blanket
            # constant-output posthoc excuse (74318b5d), which previously
            # absorbed the genuinely-insensitive index-parent perturbation of a
            # constant output. Equal dest/value CONTENTS -- the arg-position
            # discrimination concern -- are preserved.
            base = x + 3.0
            destination = base.clone()
            values = base.clone()  # value-parent CONTENTS equal the destination.
            idx = torch.arange(x.shape[0])
            destination.index_put_((idx,), values)
            return destination

    model = IndexPutValueEqualsDestModel()
    x = torch.randn(4, 5)

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        index_put_layer = _only_layer_with_func_name(trace, "index_put_")
        arg_positions = index_put_layer.parent_arg_positions["args"]
        destination_parent = arg_positions[0]
        value_parent = arg_positions[2]

        # The value parent occupies arg slot 2, NOT 0 -- it must NOT be exempt even
        # though its contents equal the destination (the false-exemption tripwire).
        assert (
            CUSTOM_EXEMPTION_CHECKS["index_put_"](trace, index_put_layer, [value_parent]) is False
        )
        assert _check_perturbation_exemptions(trace, index_put_layer, [value_parent]) is False

        # The true destination (arg slot 0) remains a legitimate full-overwrite exempt.
        assert (
            CUSTOM_EXEMPTION_CHECKS["index_put_"](trace, index_put_layer, [destination_parent])
            is True
        )
    finally:
        trace.cleanup()


def test_index_put_accumulate_positional_destination_is_not_exempt() -> None:
    """``index_put_`` with positional ``accumulate=True`` keeps destination values live."""

    class IndexPutAccumulatePositionalModel(nn.Module):
        """Model that accumulates into every destination row."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Accumulate replacement rows into the destination.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after accumulating values.
            """

            destination = x.clone() + 0.5
            idx = torch.arange(x.shape[0])
            replacement = x + 1.0
            destination.index_put_((idx,), replacement, True)
            return destination

    model = IndexPutAccumulatePositionalModel()
    x = torch.randn(4, 5)

    _assert_custom_exemption_for_arg(model, x, "index_put_", 0, False)


def test_index_put_accumulate_kwarg_destination_is_not_exempt() -> None:
    """``index_put_`` with kwarg ``accumulate=True`` keeps destination values live."""

    class IndexPutAccumulateKwargModel(nn.Module):
        """Model that accumulates into every destination row via kwarg."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Accumulate replacement rows into the destination.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after accumulating values.
            """

            destination = x.clone() + 0.5
            idx = torch.arange(x.shape[0])
            replacement = x + 1.0
            destination.index_put_((idx,), replacement, accumulate=True)
            return destination

    model = IndexPutAccumulateKwargModel()
    x = torch.randn(4, 5)

    _assert_custom_exemption_for_arg(model, x, "index_put_", 0, False)


def test_index_put_duplicate_integer_indices_destination_is_not_exempt() -> None:
    """Duplicate integer indices do not prove full destination coverage."""

    class IndexPutDuplicateIndexModel(nn.Module):
        """Model whose duplicate row indices leave one destination row live."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Overwrite with duplicate integer row indices.

            Parameters
            ----------
            x:
                Input tensor of shape ``(4, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after an ``index_put_`` with duplicate rows.
            """

            destination = x.clone() + 0.5
            idx = torch.tensor([0, 0, 1, 2], device=x.device)
            replacement = x[idx] + 1.0
            destination.index_put_((idx,), replacement)
            return destination

    model = IndexPutDuplicateIndexModel()
    x = torch.randn(4, 5)

    _assert_custom_exemption_for_arg(model, x, "index_put_", 0, False)


def test_index_put_index_parent_is_not_exempt() -> None:
    """The ``index_put_`` exemption must never apply to the index parent."""

    class IndexPutOverwriteModel(nn.Module):
        """Model that fully overwrites a destination tensor via ``index_put_``."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Overwrite every destination row with a replacement tensor.

            Parameters
            ----------
            x:
                Input tensor of shape ``(rows, cols)``.

            Returns
            -------
            torch.Tensor
                Destination after a full ``index_put_`` overwrite.
            """

            destination = x.clone() + 0.5
            idx = torch.arange(x.shape[0])
            replacement = x + 1.0
            destination.index_put_((idx,), replacement)
            return destination

    model = IndexPutOverwriteModel()
    x = torch.randn(4, 5)

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        index_put_layer = _only_layer_with_func_name(trace, "index_put_")
        index_parent = index_put_layer.parent_arg_positions["args"][(1, 0)]

        assert (
            CUSTOM_EXEMPTION_CHECKS["index_put_"](trace, index_put_layer, [index_parent]) is False
        )
        assert _check_perturbation_exemptions(trace, index_put_layer, [index_parent]) is False
    finally:
        trace.cleanup()


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


def test_where_different_branches_are_not_condition_exempt() -> None:
    """Different ``where`` branches require real perturbation sensitivity."""

    model = _WhereDifferentBranchesModel()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        where_layer = _only_layer_with_func_name(trace, "where")
        condition_parent = where_layer.parent_arg_positions["args"][0]

        assert CUSTOM_EXEMPTION_CHECKS["where"](trace, where_layer, [condition_parent]) is False
        assert _check_perturbation_exemptions(trace, where_layer, [condition_parent]) is False
    finally:
        trace.cleanup()


def test_where_mixed_condition_true_branch_is_not_exempt() -> None:
    """A mixed saved condition leaves the true branch value-sensitive."""

    model = _WhereBranchSelectionModel("mixed")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 1, False)


def test_where_mixed_condition_false_branch_is_not_exempt() -> None:
    """A mixed saved condition leaves the false branch value-sensitive."""

    model = _WhereBranchSelectionModel("mixed")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 2, False)


def test_where_shared_branch_mixed_condition_is_not_branch_exempt() -> None:
    """Equal-content ``where`` branches do not create branch-parent exemptions."""

    model = _WhereSharedBranchModel()
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 1, False)
    _assert_custom_exemption_for_arg(model, x, "where", 2, False)


def test_where_all_false_condition_true_branch_is_exempt() -> None:
    """An all-false saved condition never selects the true branch."""

    model = _WhereBranchSelectionModel("all_false")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 1, True)


def test_where_all_false_condition_false_branch_is_not_exempt() -> None:
    """An all-false saved condition selects the false branch everywhere."""

    model = _WhereBranchSelectionModel("all_false")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 2, False)


def test_where_all_true_condition_false_branch_is_exempt() -> None:
    """An all-true saved condition never selects the false branch."""

    model = _WhereBranchSelectionModel("all_true")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 2, True)


def test_where_all_true_condition_true_branch_is_not_exempt() -> None:
    """An all-true saved condition selects the true branch everywhere."""

    model = _WhereBranchSelectionModel("all_true")
    x = torch.randn(2, 3)

    _assert_custom_exemption_for_arg(model, x, "where", 1, False)


def test_where_branch_selectedness_uses_saved_condition_not_parent_out() -> None:
    """Branch selectedness must use saved args, not mutable parent outputs."""

    model = _WhereBranchSelectionModel("all_false")
    x = torch.randn(2, 3)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        where_layer = _only_layer_with_func_name(trace, "where")
        condition_parent = where_layer.parent_arg_positions["args"][0]
        true_parent = where_layer.parent_arg_positions["args"][1]
        # Layer.out is read-only under cert10 strict accessors; simulate the
        # mutated parent out through the per-pass Op record instead.
        condition_op = trace[condition_parent].ops[0]
        condition_op.out = torch.ones_like(condition_op.out)

        assert CUSTOM_EXEMPTION_CHECKS["where"](trace, where_layer, [true_parent]) is True
        assert _check_perturbation_exemptions(trace, where_layer, [true_parent]) is True
    finally:
        trace.cleanup()


def test_funcless_placeholder_unselected_where_branch_still_fails_metadata() -> None:
    """TRIPWIRE: branch exemptions must not bless functionless placeholders."""

    model = _UnselectedWherePlaceholderModel()
    x = torch.randn(2, 3)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        where_layer = _only_layer_with_func_name(trace, "where")
        unselected_parent = where_layer.parent_arg_positions["args"][2]

        assert _check_perturbation_exemptions(trace, where_layer, [unselected_parent]) is True

        # Mutate the per-pass Op record: assigning on the Layer aggregate only
        # shadows the delegated attribute and never reaches the op the
        # invariant inspects.
        placeholder = trace[unselected_parent].ops[0]
        placeholder.func = None
        placeholder.func_name = "plain_placeholder"
        placeholder.intervention_replaced = False
        placeholder.is_internal_source = False

        # Layer 1 of the tripwire: the metadata invariant itself still rejects
        # a functionless computational op in plain capture.
        with pytest.raises(MetadataInvariantError, match="func is not callable"):
            check_metadata_invariants(trace)

        # Layer 2: end-to-end validation also FAILS. cert10 catches the
        # placeholder in the replay phase first (dedicated
        # 'functionless_computational_op' decision), before the metadata step
        # gets its turn, so the run fails without the invariant raising here.
        ground_truth = [model(x).detach().clone()]
        result = trace.validate_forward_pass(ground_truth, validate_metadata=True)
        assert bool(result) is False
        status = trace.validation_replay_status
        assert status.state == "failed"
        assert any(
            decision.get("reason") == "functionless_computational_op"
            and decision.get("decision") == "failed"
            for decision in status.decisions
        )
    finally:
        trace.cleanup()


def test_remainder_divisor_not_exempt_when_output_differs_from_dividend() -> None:
    """Remainder divisor perturbation is real when dividend is not the output."""

    model = _RemainderDividendAtLeastDivisorModel()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        remainder_layer = _only_layer_with_func_name(trace, "remainder")
        divisor_parent = remainder_layer.parent_arg_positions["args"][1]
        ground_truth = [model(x).detach().clone()]

        # PosthocPerturbDecision API (74318b5d): the divisor perturbation must
        # NOT be exempt when the saved dividend differs from the output.
        assert posthoc_perturb_check(trace, remainder_layer, [divisor_parent]).exempt is False

        remainder_op = remainder_layer.ops[0]
        remainder_op.out = remainder_op.out + 100.0
        result = trace.validate_forward_pass(ground_truth, validate_metadata=False)
        assert bool(result) is False
    finally:
        trace.cleanup()


# External torch copy-construct advisory: emitted by tensor.new_tensor(tensor)
# itself, but surfaced through the torchlens wrapper frame, so the suite's
# error::UserWarning:torchlens filter would escalate it. Not a torchlens warning.
@pytest.mark.filterwarnings("ignore:To copy construct from a tensor.*:UserWarning")
def test_new_tensor_data_arg_is_not_structural_exempt() -> None:
    """Only the ``new_tensor`` template arg is structural; data is value input."""

    model = _NewTensorDataArgModel()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        new_tensor_layer = _only_layer_with_func_name(trace, "new_tensor")
        data_parent = new_tensor_layer.parent_arg_positions["args"][1]

        assert _check_perturbation_exemptions(trace, new_tensor_layer, [data_parent]) is False
    finally:
        trace.cleanup()


def test_populated_multi_output_container_keeps_leaf_paths() -> None:
    """Populated output containers must enumerate and validate real leaves."""

    model = _PopulatedContainerOutputModel()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])

    assert validate_forward_pass(model, x, random_seed=123)

    trace = trace_fn(model, x, save_arg_values=True, random_seed=123)
    try:
        output_paths = {tuple(trace[label].container_path) for label in trace.output_layers}
        output_path_reprs = {repr(path) for path in output_paths}
        ground_truth = [x + 1.0, x * 2.0]

        assert len(trace.output_layers) == 2
        assert all(output_paths)
        assert output_path_reprs == {
            "(TupleIndex(index=0),)",
            "(TupleIndex(index=1), DictKey(key='right'))",
        }
        assert trace.validate_forward_pass(ground_truth, validate_metadata=True) is True
    finally:
        trace.cleanup()


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
    with pytest.warns(UserWarning, match="no graph/source provenance"):
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
            child = log.layer_dict_all_keys[child_label]
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
        lpl = log.layer_dict_all_keys[label]
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
                parent = log.layer_dict_all_keys[p_label]
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


# ---------------------------------------------------------------------------
# one-arg torch.where(condition) == nonzero index exemption (DUAL-LAB 2026-06-30)
# ---------------------------------------------------------------------------


def _where_layer(saved_args, saved_kwargs, dtype=torch.int64):
    """Build a minimal ``where`` op layer mock for the one-arg index discriminator."""

    return SimpleNamespace(
        func_name="where",
        saved_args=saved_args,
        saved_kwargs=saved_kwargs,
        dtype=dtype,
        out=torch.zeros(4, dtype=dtype),
        layer_label="where_1_1",
    )


def test_one_arg_where_positional_nonzero_is_index_exempt() -> None:
    """One-arg positional ``torch.where(cond)`` (nonzero index form) is recognized."""

    from torchlens.validation.exemptions import _check_one_arg_where_index_exempt

    layer = _where_layer((torch.tensor([1, 0, 1, 0], dtype=torch.int64),), {})
    assert _check_one_arg_where_index_exempt(layer) is True


def test_one_arg_where_keyword_nonzero_is_index_exempt() -> None:
    """Keyword ``torch.where(condition=cond)`` nonzero index form is recognized."""

    from torchlens.validation.exemptions import _check_one_arg_where_index_exempt

    layer = _where_layer((), {"condition": torch.tensor([1, 0, 1, 0], dtype=torch.int64)})
    assert _check_one_arg_where_index_exempt(layer) is True


def test_mixed_kwarg_value_select_where_is_not_one_arg_exempt() -> None:
    """LOAD-BEARING: ``torch.where(cond, input=a, other=b)`` (value select, branches in
    kwargs, captured as len(saved_args)==1 with int output) must NOT be treated as the
    one-arg index form -- the tripwire must stay armed for a genuine value-dependent branch."""

    from torchlens.validation.exemptions import _check_one_arg_where_index_exempt

    layer = _where_layer(
        (torch.tensor([True, False, True, False]),),
        {
            "input": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "other": torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        },
    )
    assert _check_one_arg_where_index_exempt(layer) is False


def test_three_arg_positional_value_select_where_is_not_one_arg_exempt() -> None:
    """A 3-arg positional value-select ``where`` is never the one-arg index form."""

    from torchlens.validation.exemptions import _check_one_arg_where_index_exempt

    layer = _where_layer(
        (
            torch.tensor([True, False, True, False]),
            torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            torch.tensor([5, 6, 7, 8], dtype=torch.int64),
        ),
        {},
    )
    assert _check_one_arg_where_index_exempt(layer) is False


def test_float_output_where_is_not_one_arg_index_exempt() -> None:
    """A non-integer output ``where`` is never treated as the one-arg index form."""

    from torchlens.validation.exemptions import _check_one_arg_where_index_exempt

    layer = _where_layer((torch.tensor([1.0, 0.0, 1.0, 0.0]),), {}, dtype=torch.float32)
    assert _check_one_arg_where_index_exempt(layer) is False


class _OneArgWhereModel(nn.Module):
    """Model exercising the one-arg ``torch.where(cond)`` nonzero index form."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a sum over the nonzero indices of a boolean condition."""

        condition = x > 0
        indices = torch.where(condition)
        return indices[0].sum() + x.sum()


def test_one_arg_where_model_validates_forward() -> None:
    """End-to-end: a model using one-arg ``torch.where`` passes forward replay validation."""

    torch.manual_seed(0)
    model = _OneArgWhereModel().eval()
    example = torch.randint(-3, 3, (8,))
    assert tl.validate(model, example, scope="forward") is True


class _ArangeInternalSourceOutputModel(nn.Module):
    """Return a parentless internal-source tensor as the final output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an arange source that has no tensor parents."""

        return torch.arange(x.shape[-1], device=x.device)


def test_plain_trace_internal_source_final_output_is_exempted() -> None:
    """A structurally proven internal source may be the final output."""

    model = _ArangeInternalSourceOutputModel().eval()
    x = torch.randn(4, 4)
    log = trace_fn(model, [x], {})
    try:
        assert log.output_layers == ["output_1"]
        output_parent = log[log["output_1"].parents[0]]
        assert output_parent.is_internal_source
        check_metadata_invariants(log)
        assert validate_forward_pass(model, [x], input_kwargs={})
    finally:
        log.cleanup()


class _RawAtenReluModule(nn.Module):
    """Its ``forward`` returns a tensor built via a raw aten dispatch call that
    bypasses TorchLens's python-level function wrapping, so the module output is
    untraceable at module exit (a plain-capture gap, like a vmap-built tensor)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an untraceable raw-ATen relu of the input."""

        return torch.ops.aten.relu.default(x)


class _RawAtenReluNet(nn.Module):
    """Wrap the untraceable module behind a traceable linear layer."""

    def __init__(self) -> None:
        """Build the fc -> raw-aten pipeline."""

        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.raw = _RawAtenReluModule()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc then the untraceable raw-aten module."""

        return self.raw(self.fc(x))


def test_genuine_raw_hook_untraceable_replacement_validates() -> None:
    """A GENUINE raw ``register_forward_hook`` that replaces a module's output
    with an untraceable (raw-ATen) tensor must validate CLEANLY.

    The wrapper synthesizes a legitimately functionless
    ``intervention_replacement`` op (``func_call_id=None``); the func_call_id
    invariant must exempt it -- previously it crashed with MetadataInvariantError
    on this documented, supported feature (cert round 3 MAJOR).
    """

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.fc1(x))

    def _untraceable_replacement_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        # Bypasses TorchLens python-level wrapping -- a real opaque replacement.
        return torch.ops.aten.mul.Tensor(output, torch.tensor(0.5))

    model = _Mlp().eval()
    model.relu.register_forward_hook(_untraceable_replacement_hook)
    x = torch.randn(3, 4)
    log = trace_fn(model, [x], {})
    try:
        # A genuine functionless replacement placeholder IS present here...
        placeholders = _functionless_replacement_ops(log)
        assert placeholders, "expected a genuine intervention_replacement placeholder"
        assert all(not getattr(op, "is_internal_source", False) for op in placeholders)
        # ...and validation must PASS (exemption scoped to genuine replacements).
        check_metadata_invariants(log)
    finally:
        log.cleanup()
    assert validate_forward_pass(model, [x], input_kwargs={})


class _NestedBlock(nn.Module):
    """A submodule nested two address levels deep (``net.block.norm``)."""

    def __init__(self) -> None:
        """Build a single LayerNorm submodule."""

        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the norm."""

        return self.norm(x)


class _NestedNet(nn.Module):
    """Wraps ``_NestedBlock`` so the hook target sits at address depth 2."""

    def __init__(self) -> None:
        """Build the block -> fc pipeline."""

        super().__init__()
        self.block = _NestedBlock()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run block then fc."""

        return self.fc(self.block(x))


def test_genuine_raw_hook_untraceable_replacement_validates_nested_depth() -> None:
    """The depth-1 case above must ALSO hold at nesting depth >= 2.

    Cert round 4 found this crashing with ``MetadataInvariantError`` because
    ``_ensure_module_output_tensor_logged``'s ``intervention_replacement``
    branch built a single-frame module stack instead of the full ancestor
    chain: by the time the raw hook fires, the hooked module's own frame has
    already been popped off ``trace._exhaustive_module_stack``, so a
    truncated single-frame stack wires the synthetic replacement op as a
    DIRECT CHILD OF ROOT while the module's real ops (which carry the correct
    full stack) simultaneously wire the same call label under its true
    parent -- a ``[module_hierarchy]`` bidirectionality conflict.
    """

    def _substituting_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        # Bypasses TorchLens's python-level wrapping -- a real opaque replacement.
        fresh = torch.ops.aten.zeros.default([*output.shape], dtype=output.dtype)
        return torch.ops.aten.add.Tensor(fresh, output)

    model = _NestedNet().eval()
    model.block.norm.register_forward_hook(_substituting_hook)  # nested 2 levels deep
    x = torch.randn(2, 8)
    log = trace_fn(model, [x], {})
    try:
        placeholders = _functionless_replacement_ops(log)
        assert placeholders, "expected a genuine intervention_replacement placeholder"
        assert all(not getattr(op, "is_internal_source", False) for op in placeholders)
        check_metadata_invariants(log)
    finally:
        log.cleanup()
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_genuine_raw_hook_untraceable_replacement_validates_depth_zero() -> None:
    """A raw ``register_forward_hook`` on the TOP-LEVEL model (depth 0) must
    ALSO validate cleanly, and produce a correct module-call record.

    Cert round 6 found this hard-crashing with an uninterpretable ``KeyError``
    at ``model_prep.py``'s ``trace._mod_call_index[id(module)]`` lookup: the
    root model's ``forward`` is deliberately never decorated
    (``_prepare_model_once``'s ``_visit_once`` -- "Root module is handled
    separately by trace"), so it is never registered in ``_mod_call_index``
    at all, unlike every non-root module (which the round-5 fix already
    covered at hook depths 1/2/3). This is the ROOT-specific gap: no amount
    of ancestor-stack reconstruction helps because the root is never in the
    dict in the first place.

    Beyond just not crashing, the synthesized placeholder's ``module`` field
    must be the semantically correct value: ``None``, matching the
    established convention for ops with no owning submodule (see
    ``sources.py``'s ``modules[-1] if modules else None``) -- NOT a bogus
    ``("", index)`` tuple, which would finalize into an uninterpretable
    ``":<index>"`` module-call label.
    """

    def _substituting_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        # Bypasses TorchLens's python-level wrapping -- a real opaque replacement.
        fresh = torch.ops.aten.zeros.default([*output.shape], dtype=output.dtype)
        return torch.ops.aten.add.Tensor(fresh, output)

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(self.fc1(x))

    model = _Mlp().eval()
    model.register_forward_hook(_substituting_hook)  # DEPTH 0: hook on the root model
    x = torch.randn(3, 4)
    log = trace_fn(model, [x], {})
    try:
        placeholders = _functionless_replacement_ops(log)
        assert placeholders, "expected a genuine intervention_replacement placeholder"
        assert all(not getattr(op, "is_internal_source", False) for op in placeholders)
        # The placeholder has no owning submodule -- it must carry `module=None`,
        # not a bogus `("", index)`/`":index"` label.
        assert all(op.module is None for op in placeholders)
        check_metadata_invariants(log)
    finally:
        log.cleanup()
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_plain_trace_noop_hook_untraceable_exit_is_internal_source_nested_depth() -> None:
    """TRIPWIRE at nesting depth >= 2: a plain-capture gap under a no-op
    observer hook, on a module nested 2+ address levels deep, must stay
    honest -- zero functionless ``intervention_replacement`` placeholders,
    zero ``intervention_replaced`` ops -- even after the depth-2 fix above.
    The fix must not make the tripwire pass silently on a genuine capture gap.
    """

    class _RawAtenGeluBlock(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.gelu.default(x)

    class _RawAtenGeluOuter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = _RawAtenGeluBlock()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

    class _RawAtenGeluNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(8, 8)
            self.mid = _RawAtenGeluOuter()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mid(self.fc(x))

    def _noop_observer_hook(module, inputs, output):  # type: ignore[no-untyped-def]
        output.sum()  # read-only side effect, never substitutes
        return None

    model = _RawAtenGeluNet().eval()
    model.mid.inner.register_forward_hook(_noop_observer_hook)  # nested 2 levels deep
    x = torch.randn(3, 8)
    log = trace_fn(model, [x], {})
    try:
        assert _functionless_replacement_ops(log) == []
        assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []
        internal_sources = [op for op in log.ops if getattr(op, "is_internal_source", False)]
        assert internal_sources, "untraceable raw-ATen output must be an internal source"
        assert all(getattr(op, "func_name", None) == "none" for op in internal_sources)
        check_metadata_invariants(log)
    finally:
        log.cleanup()
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_plain_trace_noop_hook_untraceable_exit_is_internal_source() -> None:
    """TRIPWIRE: a plain-capture gap under a NO-OP observer hook stays honest.

    An untraceable raw-ATen module output combined with a purely observational
    forward hook (returns ``None``, never substitutes) must be logged as a clean
    ``internal_source`` -- NEVER a functionless ``intervention_replacement``
    placeholder, and must NOT be marked ``intervention_replaced``. The old
    ``has _forward_hooks`` proxy mislabeled exactly this case (cert round 3
    coupled hazard); reintroducing it makes this fail loudly.
    """

    model = _RawAtenReluNet().eval()
    x = torch.randn(3, 4)
    log = trace_fn(model, [x], {})
    try:
        # No genuine intervention happened -> zero functionless placeholders and
        # zero intervention_replaced ops during plain capture.
        assert _functionless_replacement_ops(log) == []
        assert [op for op in log.ops if getattr(op, "intervention_replaced", False)] == []
        # The untraceable output is an honest internal source (func_name "none").
        internal_sources = [op for op in log.ops if getattr(op, "is_internal_source", False)]
        assert internal_sources, "untraceable raw-ATen output must be an internal source"
        assert all(getattr(op, "func_name", None) == "none" for op in internal_sources)
        # Validation passes legitimately (as a graph source, not via a hidden gap).
        check_metadata_invariants(log)
    finally:
        log.cleanup()
    assert validate_forward_pass(model, [x], input_kwargs={})


def test_func_call_id_exemption_is_scoped_to_genuine_replacement() -> None:
    """The func_call_id exemption for ``intervention_replacement`` is NARROW.

    It must exempt ONLY the genuine functionless-replacement shape
    (``func_name == "intervention_replacement"`` AND ``intervention_replaced``
    AND NOT ``is_internal_source`` -- mirroring the ``op_log_fields`` invariant),
    and must NOT blanket-exempt any op merely named ``intervention_replacement``.
    A placeholder lacking that shape must stay non-exempt so the invariant stays
    armed against future capture-gap synthesis.
    """

    from torchlens.validation.invariants import _is_func_call_id_exempt

    base = dict(is_input=False, is_output=False, is_buffer=False, func=None)
    genuine = SimpleNamespace(
        func_name="intervention_replacement",
        intervention_replaced=True,
        is_internal_source=False,
        **base,
    )
    assert _is_func_call_id_exempt(genuine) is True

    # Same func_name but NOT flagged as replaced -> not a genuine replacement.
    not_replaced = SimpleNamespace(
        func_name="intervention_replacement",
        intervention_replaced=False,
        is_internal_source=False,
        **base,
    )
    assert _is_func_call_id_exempt(not_replaced) is False

    # Same func_name + replaced but IS an internal source -> not a genuine
    # user replacement (internal sources carry func_name "none" in practice).
    internal = SimpleNamespace(
        func_name="intervention_replacement",
        intervention_replaced=True,
        is_internal_source=True,
        **base,
    )
    assert _is_func_call_id_exempt(internal) is False
