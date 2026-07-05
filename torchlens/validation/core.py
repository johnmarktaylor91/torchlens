"""Core validation logic for verifying saved outs.

Orchestrates a three-phase verification pipeline for every layer in the graph:

1. **Ground truth check** -- model outputs match a fresh forward pass.
2. **Forward replay** (BFS from outputs toward inputs) -- re-executing each
   layer's saved function on its saved parent outs reproduces the saved
   output tensor.
3. **Perturbation check** -- for each parent of a layer, substituting random
   "wrong" values into that parent slot changes the output, proving each
   parent genuinely influences the result.

Exemption decisions (which ops to skip, which args are structural) are
delegated to the registries in ``exemptions.py``.
"""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Optional,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Set,
    TYPE_CHECKING,
    Union,
    cast,
)

import torch

from ..data_classes.op import Op
from ..ir.events import is_control_edge_use
from ..ir.container import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

from ..utils.rng import execute_with_restored_rng_autocast
from ..utils.collections import assign_to_sequence_or_dict
from ..utils.tensor_utils import tensor_nanequal, tensor_all_nan
from .exemptions import (
    SKIP_VALIDATION_ENTIRELY,
    SKIP_PERTURBATION_ENTIRELY,
    STRUCTURAL_ARG_POSITIONS,
    CUSTOM_EXEMPTION_CHECKS,
    perturbed_layer_at_structural_position,
    posthoc_perturb_check,
)
from .status import ValidationReplayStatus

ValidationDecisionKind = Literal["validated", "failed", "unverified", "exempted"]
ValidationDecisionPhase = Literal["ground_truth", "replay", "perturbation", "metadata"]


@dataclass(frozen=True)
class ValidationDecision:
    """A single replay-validation decision for audit and golden tests.

    Parameters
    ----------
    op_label:
        Operation label associated with the decision, or ``None`` for a
        trace-level decision.
    func_name:
        Captured function name for the operation, when available.
    phase:
        Validation phase that produced the decision.
    decision:
        Stable decision kind.
    reason:
        Stable reason code explaining the decision.
    justification:
        Optional human-readable proof that an exemption applies by design.
    """

    op_label: str | None
    func_name: str | None
    phase: ValidationDecisionPhase
    decision: ValidationDecisionKind
    reason: str
    justification: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return a JSON-stable representation of this decision.

        Returns
        -------
        dict of str to str or None
            JSON-serializable decision payload.
        """

        decision = {
            "op_label": self.op_label,
            "func_name": self.func_name,
            "phase": self.phase,
            "decision": self.decision,
            "reason": self.reason,
        }
        if self.justification is not None:
            decision["justification"] = self.justification
        return decision


@dataclass
class ValidationDecisionRecorder:
    """Collect replay-validation decisions and coverage counts.

    Attributes
    ----------
    decisions:
        Ordered per-op decisions.
    """

    decisions: list[ValidationDecision] = field(default_factory=list)

    def record(
        self,
        *,
        op_label: str | None,
        func_name: str | None,
        phase: ValidationDecisionPhase,
        decision: ValidationDecisionKind,
        reason: str,
        justification: str | None = None,
    ) -> None:
        """Append a validation decision.

        Parameters
        ----------
        op_label:
            Operation label associated with the decision.
        func_name:
            Captured function name for the operation, when available.
        phase:
            Validation phase that produced the decision.
        decision:
            Stable decision kind.
        reason:
            Stable reason code explaining the decision.
        justification:
            Optional proof string for design exemptions.
        """

        self.decisions.append(
            ValidationDecision(
                op_label=op_label,
                func_name=func_name,
                phase=phase,
                decision=decision,
                reason=reason,
                justification=justification,
            )
        )

    def count(self, decision: ValidationDecisionKind) -> int:
        """Return the number of decisions of a given kind.

        Parameters
        ----------
        decision:
            Decision kind to count.

        Returns
        -------
        int
            Number of matching decisions.
        """

        return sum(item.decision == decision for item in self.decisions)

    def reason_counts(self, decision: ValidationDecisionKind) -> dict[str, int]:
        """Return reason-code counts for a given decision kind.

        Parameters
        ----------
        decision:
            Decision kind to summarize.

        Returns
        -------
        dict of str to int
            Counts keyed by stable reason code.
        """

        return dict(
            sorted(
                Counter(item.reason for item in self.decisions if item.decision == decision).items()
            )
        )

    def as_status(self, *, backend: str = "torch") -> ValidationReplayStatus:
        """Build a trace-level replay status from collected decisions.

        Parameters
        ----------
        backend:
            Backend name to attach to the status.

        Returns
        -------
        ValidationReplayStatus
            Aggregate replay-validation status.
        """

        return ValidationReplayStatus.from_replay_counts(
            backend=backend,
            source="live",
            replayed_node_count=self.count("validated"),
            unverified_node_count=self.count("unverified"),
            failed_node_count=self.count("failed"),
            unverified_reason_counts=self.reason_counts("unverified"),
            exempted_reason_counts=self.reason_counts("exempted"),
            decisions=tuple(item.as_dict() for item in self.decisions),
        )


@dataclass(frozen=True)
class ValidationCheckResult:
    """Internal replay result preserving pass/fail and coverage status.

    Parameters
    ----------
    decision:
        Stable decision kind.
    reason:
        Stable reason code explaining the decision.
    justification:
        Optional proof string for design exemptions.
    """

    decision: ValidationDecisionKind
    reason: str
    justification: str | None = None

    @property
    def failed(self) -> bool:
        """Return whether this result is a hard validation failure.

        Returns
        -------
        bool
            True for hard failures.
        """

        return self.decision == "failed"

    @classmethod
    def validated(cls, reason: str = "replay_matched") -> "ValidationCheckResult":
        """Build a validated result.

        Parameters
        ----------
        reason:
            Stable reason code.

        Returns
        -------
        ValidationCheckResult
            Validated result.
        """

        return cls("validated", reason)

    @classmethod
    def failed_result(cls, reason: str) -> "ValidationCheckResult":
        """Build a failed result.

        Parameters
        ----------
        reason:
            Stable reason code.

        Returns
        -------
        ValidationCheckResult
            Failed result.
        """

        return cls("failed", reason)

    @classmethod
    def unverified(cls, reason: str) -> "ValidationCheckResult":
        """Build an unverified result.

        Parameters
        ----------
        reason:
            Stable reason code.

        Returns
        -------
        ValidationCheckResult
            Unverified result.
        """

        return cls("unverified", reason)

    @classmethod
    def exempted(cls, reason: str, justification: str | None = None) -> "ValidationCheckResult":
        """Build an exempted result.

        Parameters
        ----------
        reason:
            Stable reason code.
        justification:
            Optional proof string for design exemptions.

        Returns
        -------
        ValidationCheckResult
            Exempted result.
        """

        return cls("exempted", reason, justification)


# Maximum number of random perturbation attempts before giving up on finding
# a value different from the original.  Relevant for integer/bool tensors
# where the value space may be small (e.g., a single-element int tensor).
MAX_PERTURB_ATTEMPTS = 100

# Deep reduction ops can replay the same FP32 op with tiny differences after
# many accumulated multiply-adds. Keep this local to validation replay so global
# tensor equality stays strict for graph bookkeeping and lower-tier tests.
#
# Band-C (deep-numeric) eligibility is the FAITHFUL reduction depth, not a graph-
# position or func-allowlist proxy: an op qualifies iff it is a REDUCTION whose
# per-output accumulation depth (FP32 multiply-adds summed into each output
# element) is at least DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH. FP32 reorder
# drift is driven by the contraction length, not by where the op sits in the
# graph (a wide channel-expansion conv accumulates hundreds of products per
# output whether it is op #41 or op #870) nor by a hand-picked conv/matmul list
# (a high-degree scatter/segment aggregation drifts by the same physics). This
# replaces BOTH the earlier DEEP_NUMERIC_REPLAY_FUNCS allowlist and the
# step_index >= 100 position proxy, which together mis-gated physically-deep
# early ops AND lent band C to shallow late ops (a bug-masking hole). The
# acceptance tolerances below are UNCHANGED, so a genuinely wrong replay still
# fails; only the qualification predicate is now the faithful depth measure.
#
# Depth 0 means the depth could not be determined; such ops are INELIGIBLE
# (fail-toward-strict) so band C is withheld and global tolerance stays strict.
DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH = 64

# Reduction-category func-name groups, used only to compute the per-output
# accumulation depth (NOT to gate eligibility -- a shallow op in any category
# below depth 64 stays strict). "Reduction" here means any op that sums many
# inputs into a single output element; everything else (elementwise, view,
# structural, copy) has depth 1 and never reaches the threshold.
_CONV_FUNC_PREFIX = "conv"  # conv1d/2d/3d + conv_transpose1d/2d/3d
_CONV_TRANSPOSE_FUNC_PREFIX = "conv_transpose"  # conv_transpose1d/2d/3d
_MATMUL_LINEAR_FUNCS = frozenset({"addmm", "baddbmm", "bmm", "linear", "matmul", "mm"})
# Scatter/segment/index aggregations that are band-C-eligible ONLY when they are
# ADDITIVE/mean accumulators -- i.e. they sum many FP32 source elements into one
# destination slot, so summation-order round-off genuinely accrues. Plain
# overwrite ``scatter``/``scatter_`` (no FP accumulation -- the last write wins)
# and ``max``/``min``/``amax``/``amin`` reductions (no summation-order drift)
# are EXCLUDED here and stay on the strict global tolerance. ``scatter_reduce*``
# is conditionally additive: only its ``reduce="sum"``/``"mean"`` modes accumulate
# (its ``"prod"``/``"amax"``/``"amin"`` modes do not), so it is gated dynamically
# inside the depth predicate rather than by membership here.
_SCATTER_REDUCE_FUNCS = frozenset(
    {
        "scatter_add",
        "scatter_add_",
        "scatter_reduce",
        "scatter_reduce_",
        "segment_reduce",
        "index_add",
        "index_add_",
    }
)
# scatter_reduce reduce-modes that ADD (FP reorder drift) vs that select/overwrite.
_ADDITIVE_SCATTER_REDUCE_MODES = frozenset({"sum", "mean"})
_DIM_REDUCE_FUNCS = frozenset(
    {
        "sum",
        "mean",
        "prod",
        "norm",
        "var",
        "std",
        "nansum",
        "nanmean",
        "logsumexp",
    }
)

DEEP_NUMERIC_REPLAY_RTOL = 1e-3
DEEP_NUMERIC_REPLAY_ATOL = 1e-4
DEEP_NUMERIC_REPLAY_OUTLIER_RTOL = 5e-2
DEEP_NUMERIC_REPLAY_OUTLIER_ATOL = 1e-2
DEEP_NUMERIC_REPLAY_MAX_OUTLIER_FRACTION = 1e-4
DEEP_NUMERIC_REPLAY_MAX_SCALED_DIFF = 5e-2
DEEP_NUMERIC_REPLAY_MAX_MEAN_SCALED_DIFF = 1e-3
GROUND_TRUTH_OUTPUT_RTOL = 1e-6
GROUND_TRUTH_OUTPUT_ATOL = 1e-8


def _raise_if_portable_bundle_log(self: Any) -> None:
    """Reject validation only when loaded logs lack replay callables.

    Parameters
    ----------
    self:
        Model log being validated.

    Raises
    ------
    TorchLensIOError
        If the log was loaded from a portable bundle without resolved
        ``func`` callables on computational nodes.
    """

    if not bool(getattr(self, "_loaded_from_bundle", False)):
        return
    unresolved = [
        getattr(layer, "layer_label", "<unknown>")
        for layer in getattr(self, "layer_list", [])
        if getattr(layer, "func", None) is None
        and getattr(layer, "func_name", "none") not in {"none", "input", "output", "buffer"}
    ]
    if unresolved:
        from .._io import TorchLensIOError

        raise TorchLensIOError(
            "validate_forward_pass requires resolved func callables; portable bundles "
            "drop them when functions cannot be represented. This bundle has unresolved "
            f"computational functions, e.g. {unresolved[:3]!r}."
        )


def _ground_truth_output_matches_saved(
    saved_output: torch.Tensor,
    ground_truth_output: torch.Tensor,
) -> bool:
    """Return whether a saved model output matches the direct forward output.

    The direct output check is exact first. For floating-point outputs, it then
    allows only sub-ULP wrapper noise, which covers models whose logged full
    forward produces numerically equivalent logits that differ at ~1e-11 scale.

    Parameters
    ----------
    saved_output:
        Output tensor saved by TorchLens logging.
    ground_truth_output:
        Output tensor from the direct model forward pass.

    Returns
    -------
    bool
        True if the outputs are exactly equal or differ only by the tight
        output-only floating-point tolerance.
    """
    if tensor_nanequal(saved_output, ground_truth_output, allow_tolerance=False):
        return True
    if saved_output.shape != ground_truth_output.shape:
        return False
    if saved_output.dtype != ground_truth_output.dtype:
        return False
    if not saved_output.is_floating_point():
        return False

    from .._state import pause_logging

    with pause_logging():
        if not torch.equal(saved_output.isnan(), ground_truth_output.isnan()):
            return False
        if not torch.equal(saved_output.isinf(), ground_truth_output.isinf()):
            return False
        saved_nonan = torch.nan_to_num(saved_output, 0.7234691827346)
        ground_truth_nonan = torch.nan_to_num(ground_truth_output, 0.7234691827346)
        return bool(
            torch.allclose(
                saved_nonan,
                ground_truth_nonan,
                rtol=GROUND_TRUTH_OUTPUT_RTOL,
                atol=GROUND_TRUTH_OUTPUT_ATOL,
            )
        )


def validate_saved_outs(
    self: "Trace",
    ground_truth_output_tensors: List[torch.Tensor],
    verbose: bool = False,
    validate_metadata: bool = True,
) -> ValidationReplayStatus:
    """Run the full validation pipeline on a completed Trace.

    The BFS traversal starts from two kinds of seed layers:
    - **output layers** -- whose values are verified against ``ground_truth_output_tensors``.
    - **internally terminated layers** -- dead-end tensors with no children outside the model.

    From each seed the BFS walks *backward* through parent edges.  A parent
    layer is enqueued once ALL of its child edges have been individually
    validated (edge-counting completion), ensuring diamond-shaped subgraphs
    are handled correctly.

    After out validation ops, optional metadata invariant checks
    (checks A-R in ``invariants.py``) run to verify structural/semantic
    consistency of the entire Trace.

    Args:
        ground_truth_output_tensors: Output tensors from a fresh forward pass,
            used to confirm the logged outputs are accurate before BFS begins.
        verbose: Whether to print warning messages on validation failure.
        validate_metadata: Whether to run metadata invariant checks (default True).

    Returns:
        Aggregate replay-validation status. Fully validated pass/fail results
        remain bool-compatible through callers that unwrap completed statuses.
    """
    _raise_if_portable_bundle_log(self)

    # Diagnostics side-channel: clear any stale failure from a prior run so a
    # report reflects THIS validation only. ADD-ONLY -- never affects the result.
    from .diagnostics import (
        CHECK_COMPLETENESS,
        CHECK_GROUND_TRUTH,
        CHECK_OUTPUT_MISSING,
        ValidationFailure,
        describe_tensor_mismatch,
        record_validation_failure,
        reset_validation_failure,
    )

    reset_validation_failure(self)
    decision_recorder = ValidationDecisionRecorder()

    # Phase 0: verify logged outputs match a fresh forward pass. Halted traces
    # deliberately stop at an internal frontier, so no full-model output exists.
    if not getattr(self, "halted", False):
        output_label_counts: dict[str, int] = defaultdict(int)
        for i, output_layer_label in enumerate(self.output_layers):
            output_layer = _resolve_output_entry_for_index(
                self, output_layer_label, output_label_counts
            )
            if output_layer.out is None:
                print(f"The {i}th output layer, {output_layer_label}, has no saved out.")
                record_validation_failure(
                    self,
                    ValidationFailure(
                        check=CHECK_OUTPUT_MISSING,
                        op_label=output_layer_label,
                        message=f"output layer #{i} has no saved out",
                    ),
                )
                decision_recorder.record(
                    op_label=output_layer_label,
                    func_name=getattr(output_layer, "func_name", None),
                    phase="ground_truth",
                    decision="failed",
                    reason="output_missing",
                )
                status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
                setattr(self, "_validation_replay_status", status)
                return status
            if not _ground_truth_output_matches_saved(
                output_layer.out, ground_truth_output_tensors[i]
            ):
                print(
                    f"The {i}th output layer, {output_layer_label}, does not match the ground truth output tensor."
                )
                record_validation_failure(
                    self,
                    describe_tensor_mismatch(
                        output_layer.out,
                        ground_truth_output_tensors[i],
                        check=CHECK_GROUND_TRUTH,
                        op_label=output_layer_label,
                        func_name=getattr(output_layer, "func_name", None),
                        message=f"output #{i} does not match ground truth",
                    ),
                )
                decision_recorder.record(
                    op_label=output_layer_label,
                    func_name=getattr(output_layer, "func_name", None),
                    phase="ground_truth",
                    decision="failed",
                    reason="ground_truth_mismatch",
                )
                status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
                setattr(self, "_validation_replay_status", status)
                return status
            decision_recorder.record(
                op_label=output_layer_label,
                func_name=getattr(output_layer, "func_name", None),
                phase="ground_truth",
                decision="validated",
                reason="ground_truth_matched",
            )

    # BFS backward from outputs + internally terminated layers.
    # Edge-counting approach: a parent is enqueued only after ALL its child
    # edges are validated (validated_child_edges == set(children)).
    validated_child_edges_for_each_layer: Dict[str, Set[str]] = defaultdict(set)
    seed_layers = list(
        dict.fromkeys(
            self[label].layer_label for label in self.output_layers + self.internal_sink_ops
        )
    )
    validated_layers = set(seed_layers)
    layers_to_validate_parents_for = deque(seed_layers)

    while len(layers_to_validate_parents_for) > 0:
        layer_to_validate_parents_for = layers_to_validate_parents_for.popleft()
        parents_valid = validate_parents_of_saved_layer(
            self,
            layer_to_validate_parents_for,
            validated_layers,
            validated_child_edges_for_each_layer,
            layers_to_validate_parents_for,  # type: ignore[arg-type]
            verbose,
            decision_recorder,
        )
        if parents_valid.failed:
            status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
            setattr(self, "_validation_replay_status", status)
            return status

    # Completeness check: BFS must visit every layer in the graph.
    expected_layers = {layer.layer_label for layer in self.layer_list}
    if len(validated_layers) < len(expected_layers):
        unreached = expected_layers - validated_layers
        print(
            f"All saved outs were accurate, but some layers were not reached (check that "
            f"child args logged accurately): {unreached}"
        )
        record_validation_failure(
            self,
            ValidationFailure(
                check=CHECK_COMPLETENESS,
                message=(
                    f"BFS reached {len(validated_layers)}/{len(expected_layers)} layers; "
                    f"{len(unreached)} unreached (e.g. {sorted(unreached)[:3]})"
                ),
                extra={"n_unreached": len(unreached)},
            ),
        )
        decision_recorder.record(
            op_label=None,
            func_name=None,
            phase="metadata",
            decision="failed",
            reason="bfs_incomplete",
        )
        status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
        setattr(self, "_validation_replay_status", status)
        return status

    # Metadata invariant checks (after out validation ops)
    if validate_metadata:
        from .invariants import check_metadata_invariants

        try:
            check_metadata_invariants(self)
        except Exception as invariant_error:
            # ADD-ONLY: record the firing invariant on the side-channel, then
            # RE-RAISE unchanged. The invariant remains a hard tripwire -- this
            # only captures its identity for downstream reporting.
            from .diagnostics import (
                CHECK_METADATA_INVARIANT,
                ValidationFailure,
                record_validation_failure,
            )

            record_validation_failure(
                self,
                ValidationFailure(
                    check=CHECK_METADATA_INVARIANT,
                    message=f"{type(invariant_error).__name__}: {invariant_error}",
                ),
            )
            decision_recorder.record(
                op_label=None,
                func_name=None,
                phase="metadata",
                decision="failed",
                reason="metadata_invariant_exception",
            )
            raise

    status = decision_recorder.as_status(backend=str(getattr(self, "backend", "torch")))
    setattr(self, "_validation_replay_status", status)
    return status


def validate_parents_of_saved_layer(
    self: "Trace",
    layer_to_validate_parents_for_label: str,
    validated_layers: Set[str],
    validated_child_edges_for_each_layer: Dict[str, Set[str]],
    layers_to_validate_parents_for: List[str],
    verbose: bool = False,
    decision_recorder: ValidationDecisionRecorder | None = None,
) -> ValidationCheckResult:
    """Validate a single layer's parent edges: argument logging, forward replay,
    and perturbation for each parent.

    This is the inner loop of the BFS. For the layer identified by
    ``layer_to_validate_parents_for_label``, this function:

    1. Checks that the parent layer outs are correctly logged in the
       argument map (``parent_arg_positions``).
    2. Re-executes the layer's function on saved parent values and confirms the
       output matches the saved tensor (forward replay, ``perturb=False``).
    3. For each parent, re-executes with that parent's out *perturbed*
       and confirms the output *changes* (perturbation check, ``perturb=True``).
       Ops in ``SKIP_PERTURBATION_ENTIRELY`` skip this step.

    After all checks pass, each parent's validated-child-edge set is updated.
    When a parent has ALL its child edges validated, it is added to the BFS
    work queue (unless it is an input layer or a parentless buffer).

    Args:
        layer_to_validate_parents_for_label: Label of the layer whose parent edges are being validated.
        validated_layers: Set of layer labels already validated; mutated in-place to add newly validated layers.
        validated_child_edges_for_each_layer: Dict mapping each layer label to the set of its child edges
            that have been validated so far; mutated in-place as child edges are confirmed.
        layers_to_validate_parents_for: Work queue of layer labels still needing parent validation;
            mutated in-place to append newly discovered layers.
        verbose: Whether to print warning messages on validation failure.

    Returns:
        Structured result for the parent-edge validation step.
    """
    layer_to_validate_parents_for = self[layer_to_validate_parents_for_label]
    ops_to_validate = _validation_ops_for_entry(layer_to_validate_parents_for)

    # Check that the arguments are logged correctly when the evidence is
    # available. Unverified evidence is recorded but does not preempt replay.
    arg_logging_result = _check_layer_arguments_logged_correctly(
        self, layer_to_validate_parents_for_label
    )
    skip_replay_after_arg_logging = False
    if arg_logging_result.decision == "unverified":
        if decision_recorder is not None:
            decision_recorder.record(
                op_label=layer_to_validate_parents_for_label,
                func_name=getattr(layer_to_validate_parents_for, "func_name", None),
                phase="replay",
                decision="unverified",
                reason=arg_logging_result.reason,
            )
        if arg_logging_result.reason == "missing_saved_args":
            skip_replay_after_arg_logging = True
    elif arg_logging_result.failed:
        print(
            f"Parent arguments for layer {layer_to_validate_parents_for_label} are not logged properly; "
            f"either a parent wasn't logged as an argument, or was logged an extra time"
        )
        from .diagnostics import CHECK_ARG_LOGGING, ValidationFailure, record_validation_failure

        record_validation_failure(
            self,
            ValidationFailure(
                check=CHECK_ARG_LOGGING,
                op_label=layer_to_validate_parents_for_label,
                func_name=getattr(layer_to_validate_parents_for, "func_name", None),
                message="parent not logged as an argument, or logged an extra time",
            ),
        )
        if decision_recorder is not None:
            decision_recorder.record(
                op_label=layer_to_validate_parents_for_label,
                func_name=getattr(layer_to_validate_parents_for, "func_name", None),
                phase="replay",
                decision=arg_logging_result.decision,
                reason=arg_logging_result.reason,
            )
        return arg_logging_result

    ops_to_replay = _representative_ops_for_replay(self, ops_to_validate)
    if not skip_replay_after_arg_logging:
        # Forward replay: re-execute with correct parent values, expect same output.
        for target_op in ops_to_replay:
            if _is_intentional_intervention_replacement(target_op):
                if decision_recorder is not None:
                    decision_recorder.record(
                        op_label=target_op.label,
                        func_name=getattr(target_op, "func_name", None),
                        phase="replay",
                        decision="exempted",
                        reason="intentional_intervention_replacement",
                    )
                continue
            replay_result = _check_whether_func_on_saved_parents_yields_saved_tensor(
                self, target_op.label, perturb=False
            )
            if decision_recorder is not None:
                decision_recorder.record(
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    phase="replay",
                    decision=replay_result.decision,
                    reason=replay_result.reason,
                    justification=replay_result.justification,
                )
            if replay_result.failed:
                return replay_result

        # Perturbation: for each parent, substitute random values and expect
        # the output to change, proving that parent genuinely influences this layer.

        representative_parent_edges = _representative_parent_edges(self, ops_to_replay)
        for target_op, perturb_layer in representative_parent_edges:
            if _is_intentional_intervention_replacement(target_op):
                if decision_recorder is not None:
                    decision_recorder.record(
                        op_label=target_op.label,
                        func_name=getattr(target_op, "func_name", None),
                        phase="perturbation",
                        decision="exempted",
                        reason="intentional_intervention_replacement",
                    )
                continue
            if target_op.func_name in SKIP_PERTURBATION_ENTIRELY:
                if decision_recorder is not None:
                    decision_recorder.record(
                        op_label=target_op.label,
                        func_name=getattr(target_op, "func_name", None),
                        phase="perturbation",
                        decision="exempted",
                        reason=f"skip_perturbation_entirely:{target_op.func_name}",
                    )
                continue
            perturb_result = _check_whether_func_on_saved_parents_yields_saved_tensor(
                self,
                target_op.label,
                perturb=True,
                layers_to_perturb=[perturb_layer],
                verbose=verbose,
            )
            if decision_recorder is not None:
                decision_recorder.record(
                    op_label=target_op.label,
                    func_name=getattr(target_op, "func_name", None),
                    phase="perturbation",
                    decision=perturb_result.decision,
                    reason=perturb_result.reason,
                    justification=perturb_result.justification,
                )
            if perturb_result.failed:
                return perturb_result

    # Record validated edges and enqueue parents whose ALL child edges are now validated.
    for parent_layer_label in layer_to_validate_parents_for.parents:
        parent_layer = self[parent_layer_label]
        validated_child_edges_for_each_layer[parent_layer_label].add(
            layer_to_validate_parents_for_label
        )
        # Enqueue a parent once at least one validated child proves a path to a
        # checked output or internal sink. Recurrent multi-pass layers can have
        # self/side child edges that are valid but not part of the current
        # representative validation frontier.
        if parent_layer_label not in validated_layers:
            validated_layers.add(parent_layer_label)
            # Don't enqueue terminal seeds (inputs, parentless buffers) --
            # they have no parents to validate further.
            if (not parent_layer.is_input) and not (
                parent_layer.is_buffer and (parent_layer.buffer_source is None)
            ):
                layers_to_validate_parents_for.append(parent_layer_label)

    return ValidationCheckResult.validated("parent_edges_validated")


def _is_intentional_intervention_replacement(layer: "Op") -> bool:
    """Return whether a layer's out was intentionally replaced by a hook.

    Parameters
    ----------
    layer:
        Operation pass being validated.

    Returns
    -------
    bool
        Whether validation should treat the op as an intervention boundary.
    """

    return bool(
        getattr(layer, "intervention_replaced", False)
        and not getattr(layer, "is_internal_source", False)
    )


def _is_provable_functionless_source_or_boundary(layer: "Op") -> bool:
    """Return whether a functionless op is a structural source or boundary.

    Parameters
    ----------
    layer:
        Operation pass being validated.

    Returns
    -------
    bool
        True only when captured metadata proves the absence of a callable is
        expected by design, rather than a lost computational function.
    """

    if getattr(layer, "func", None) is not None:
        return False
    if _is_intentional_intervention_replacement(layer):
        return True
    source_category = (
        bool(getattr(layer, "is_input", False))
        or bool(getattr(layer, "is_output", False))
        or bool(getattr(layer, "is_buffer", False))
        or bool(getattr(layer, "input_was_parameter", False))
        or bool(getattr(layer, "is_internal_source", False))
    )
    if not source_category:
        return False
    return str(getattr(layer, "func_name", "none")) in {"none", "input", "output", "buffer"}


def _resolve_output_entry_for_index(
    self: "Trace", output_layer_label: str, output_label_counts: dict[str, int]
) -> Op:
    """Resolve an output label occurrence to a concrete output Op.

    Parameters
    ----------
    output_layer_label:
        Layer label from ``Trace.output_layers``.
    output_label_counts:
        Mutable per-label occurrence counts used to map duplicate output layer
        labels to their pass-specific output ops.

    Returns
    -------
    Op
        Concrete output op for this output occurrence.
    """

    output_entry = self[output_layer_label]
    ops = getattr(output_entry, "ops", None)
    if not hasattr(ops, "_list"):
        return cast(Op, output_entry)
    op_list = cast(list[Op], cast(Any, ops)._list)
    occurrence_index = output_label_counts[output_layer_label]
    output_label_counts[output_layer_label] += 1
    return op_list[occurrence_index]


def _representative_ops_for_replay(self: "Trace", ops_to_validate: List[Op]) -> List[Op]:
    """Return concrete child ops for expensive forward replay validation.

    Parameters
    ----------
    ops_to_validate:
        Concrete op passes for the child Layer currently being validated.

    Returns
    -------
    list of Op
        A stable subset that covers the Layer's distinct parent-layer edges.
    """

    if len(ops_to_validate) <= 1:
        return ops_to_validate

    representative_ops: dict[str, Op] = {}
    fallback_op = ops_to_validate[0]
    for target_op in ops_to_validate:
        data_parents = _data_parent_labels(target_op)
        if not data_parents:
            representative_ops.setdefault(target_op.label, target_op)
        for parent_label in sorted(data_parents):
            parent_layer_label = self[parent_label].layer_label
            representative_ops.setdefault(parent_layer_label, target_op)
    if not representative_ops:
        return [fallback_op]
    return list(dict.fromkeys(representative_ops.values()))


def _representative_parent_edges(self: "Trace", ops_to_validate: List[Op]) -> List[tuple[Op, str]]:
    """Return one concrete op edge for each parent Layer edge.

    Parameters
    ----------
    ops_to_validate:
        Concrete op passes for the child Layer currently being validated.

    Returns
    -------
    list of tuple of Op and str
        Pairs of child op and pass-qualified parent label to perturb.
    """

    representative_edges: dict[str, tuple[Op, str]] = {}
    for target_op in ops_to_validate:
        for parent_label in sorted(_data_parent_labels(target_op)):
            parent_layer_label = self[parent_label].layer_label
            representative_edges.setdefault(parent_layer_label, (target_op, parent_label))
    return list(representative_edges.values())


def _data_parent_labels(op: Op) -> set[str]:
    """Return parent labels that should participate in value replay.

    Parameters
    ----------
    op
        Operation whose parents should be classified.

    Returns
    -------
    set[str]
        Parent labels excluding control-only dependencies.
    """

    parents = {label for label in op.parents if isinstance(label, str)}
    control_parents = {
        edge.parent_label
        for edge in getattr(op, "_edge_uses", ())
        if isinstance(getattr(edge, "parent_label", None), str) and is_control_edge_use(edge)
    }
    return parents - control_parents


def _validation_ops_for_entry(entry: Any) -> List[Op]:
    """Return pass-specific ops that should be used for validation.

    Parameters
    ----------
    entry:
        Trace entry resolved from a validation queue label. This is usually a
        ``Layer`` but may already be an ``Op`` for pass-qualified labels.

    Returns
    -------
    list of Op
        The concrete op passes whose saved args and outputs can be replayed.
    """

    ops = getattr(entry, "ops", None)
    if not hasattr(ops, "_list"):
        return [cast(Op, entry)]
    op_list = cast(list[Op], cast(Any, ops)._list)
    if len(op_list) == 0:
        return []
    return [op_list[0]]


def _check_layer_arguments_logged_correctly(
    self: "Trace", target_layer_label: str
) -> ValidationCheckResult:
    """Check whether the outs of the parent layers match the saved arguments of
    the target layer, and that the argument locations have been logged correctly.

    Args:
        target_layer_label: Layer to check

    Returns:
        Structured validation result for argument logging evidence.
    """
    target_entry = self[target_layer_label]
    target_ops = _representative_ops_for_replay(self, _validation_ops_for_entry(target_entry))

    for target_layer in target_ops:
        # Genuine functionless ops have no torch function whose arguments could
        # be reconstructed: graph sources (inputs, buffers, internally generated
        # sources such as a torch.vmap-built attention mask) and GENUINE raw
        # forward-hook output replacements injected by the user. This skip is
        # deliberately narrow -- it must NOT swallow a real op that merely lost
        # its func, which would mask a capture bug.
        if _is_provable_functionless_source_or_boundary(target_layer):
            continue

        # Make sure that all parent layers appear in at least one argument and
        # that no extra layers appear:
        data_parents = _data_parent_labels(target_layer)
        parents_in_args = set()
        for arg_type in ["args", "kwargs"]:
            parents_in_args.update(list(target_layer.parent_arg_positions[arg_type].values()))
        if parents_in_args != data_parents:
            try:
                _raise_if_replay_arg_version_data_incomplete(self, target_layer)
            except ValueError:
                return ValidationCheckResult.unverified("missing_saved_args")
            return ValidationCheckResult.failed_result("arg_logging_mismatch")

        argtype_dict = {
            "args": (enumerate, "saved_args"),
            "kwargs": (lambda x: x.items(), "saved_kwargs"),
        }

        # Check for each parent layer that it is logged as a saved argument when it matches an argument,
        # and is not logged when it does not match a saved argument.

        for parent_layer_label in data_parents:
            parent_layer = self[parent_layer_label]
            for arg_type in ["args", "kwargs"]:
                iterfunc, argtype_field = argtype_dict[arg_type]
                saved_values = getattr(target_layer, argtype_field)
                if saved_values is None:
                    return ValidationCheckResult.unverified("missing_saved_args")
                for key, val in iterfunc(saved_values):  # type: ignore[operator]
                    validation_result_for_arg_and_layer = _validate_layer_against_arg(
                        self, target_layer, parent_layer, arg_type, key, val
                    )
                    if validation_result_for_arg_and_layer.decision != "validated":
                        return validation_result_for_arg_and_layer
    return ValidationCheckResult.validated("arg_logging_matched")


def _raise_if_replay_arg_version_data_incomplete(self: "Trace", target_layer: Op) -> None:
    """Reject replay validation when sparse capture omitted arg-version data.

    Parameters
    ----------
    self:
        Trace being replay-validated.
    target_layer:
        Operation whose saved parents would need argument and child-version
        snapshots for validation.

    Raises
    ------
    ValueError
        If the trace is known to lack complete replay argument/version data.
    """

    if getattr(self, "_replay_arg_version_data_complete", True):
        return
    if not target_layer.parents:
        return
    raise ValueError(
        "Cannot validate saved layer "
        f"{target_layer.label}: this trace does not have complete saved argument "
        "values or child-version snapshots for replay validation. "
        "Use tl.trace(..., save_arg_values=True) for replay validation."
    )


def _validate_layer_against_arg(
    self: "Trace",
    target_layer: Op,
    parent_layer: Op,
    arg_type: str,
    key: Any,
    val: Any,
) -> ValidationCheckResult:
    """Validate whether a parent layer is correctly logged for a specific argument of a target layer.

    Handles nested argument structures (lists, tuples, dicts) by recursing into them
    and delegating to ``_check_arglocs_correct_for_arg`` for each leaf value.

    Args:
        target_layer: The child layer whose argument log is being checked.
        parent_layer: The parent layer being tested against the argument.
        arg_type: Either ``"args"`` or ``"kwargs"``.
        key: The positional index (for args) or keyword string (for kwargs) of the argument.
        val: The saved argument value to inspect.

    Returns:
        Structured validation result for this argument position.
    """
    if type(val) in [list, tuple]:
        for v, subval in enumerate(val):
            argloc_key = (key, v)
            validation_result_for_arg_and_layer = _check_arglocs_correct_for_arg(
                self, target_layer, parent_layer, arg_type, argloc_key, subval
            )
            if validation_result_for_arg_and_layer.decision != "validated":
                return validation_result_for_arg_and_layer

    elif type(val) == dict:
        for subkey, subval in val.items():
            argloc_key = (key, subkey)
            validation_result_for_arg_and_layer = _check_arglocs_correct_for_arg(
                self, target_layer, parent_layer, arg_type, argloc_key, subval
            )
            if validation_result_for_arg_and_layer.decision != "validated":
                return validation_result_for_arg_and_layer
    else:
        argloc_key = key
        validation_result_for_arg_and_layer = _check_arglocs_correct_for_arg(
            self, target_layer, parent_layer, arg_type, argloc_key, val
        )
        if validation_result_for_arg_and_layer.decision != "validated":
            return validation_result_for_arg_and_layer

    return ValidationCheckResult.validated("arg_logging_matched")


def _parent_logged_for_any_arg_alias(target_layer: Op, parent_layer_labels: set[str]) -> bool:
    """Return whether any parent label alias is logged at any arg location.

    Parameters
    ----------
    target_layer:
        Child op whose parent-arg map is being inspected.
    parent_layer_labels:
        Equivalent labels for the same parent, usually pass-qualified op label
        and bare parent layer label.

    Returns
    -------
    bool
        True when any alias appears in the target arg-position map.
    """

    return any(
        logged_parent in parent_layer_labels
        for arg_type in ("args", "kwargs")
        for logged_parent in target_layer.parent_arg_positions[arg_type].values()
    )


def _check_arglocs_correct_for_arg(
    self: "Trace",
    target_layer: Op,
    parent_layer: Op,
    arg_type: str,
    argloc_key: str | tuple[Any, ...],
    saved_arg_val: Any,
) -> ValidationCheckResult:
    """Check bidirectional consistency between a parent's tensor and a child's arg slot.

    Validates two directions:
    - If the parent's out matches the saved arg value AND the parent is
      not logged at that position, that is an error (unless the match is
      trivially coincidental -- e.g., empty tensor, bool tensor, all-NaN,
      all-zero, or all-abs-one, or another parent has identical values).
    - If the parent is logged at that position BUT its out does not
      match the saved arg value, that is an error (with a special exemption
      for in-place RNG ops like ``bernoulli_`` that mutate after
      logging).

    Args:
        target_layer: The child layer whose argument log is being checked.
        parent_layer: The parent layer being tested against the argument.
        arg_type: Either ``"args"`` or ``"kwargs"``.
        argloc_key: The position key (int, str, or tuple for nested args).
        saved_arg_val: The saved argument value at that position.

    Returns:
        Structured validation result for this argument location.
    """
    target_layer_label = target_layer.layer_label
    target_op_label = getattr(target_layer, "label", target_layer_label)
    parent_layer_label = parent_layer.layer_label
    parent_arg_labels = {getattr(parent_layer, "label", parent_layer_label), parent_layer_label}
    # out_versions_by_child stores per-child snapshots when an in-place
    # op modified the tensor between uses.  Fall back to out
    # when no child-specific version exists.
    if target_op_label in parent_layer.out_versions_by_child:
        parent_outs = parent_layer.out_versions_by_child[target_op_label]
    elif target_layer_label in parent_layer.out_versions_by_child:
        parent_outs = parent_layer.out_versions_by_child[target_layer_label]
    else:
        parent_outs = parent_layer.out
    if parent_outs is None:
        return ValidationCheckResult.unverified("missing_saved_parent_payload")

    if isinstance(saved_arg_val, torch.Tensor):
        parent_layer_matches_arg = tensor_nanequal(
            saved_arg_val, parent_outs, allow_tolerance=False
        )
    else:
        parent_layer_matches_arg = False
    parent_layerged_as_arg = (
        argloc_key in target_layer.parent_arg_positions[arg_type]
        and target_layer.parent_arg_positions[arg_type][argloc_key] in parent_arg_labels
    )

    # Case 1: parent matches the arg value but is NOT logged at this position.
    # This is only an error if the match is non-trivially coincidental.
    # Exemptions for trivially coincidental matches:
    #   - empty tensor (numel==0)
    #   - bool tensor (many ops produce True/False identity matches)
    #   - all-NaN, all-zero, or all-abs-one tensors (special values)
    #   - another parent has identical tensor values (ambiguous attribution)
    if (
        parent_layer_matches_arg
        and (not parent_layerged_as_arg)
        and (not _parent_logged_for_any_arg_alias(target_layer, parent_arg_labels))
        and (parent_outs.numel() != 0)
        and (parent_outs.dtype != torch.bool)
        and (not tensor_all_nan(parent_outs))
        and (not torch.all(parent_outs == 0))
        and (not torch.all(torch.abs(parent_outs) == 1))
        and not any(
            [
                torch.equal(parent_outs, self[other_parent].out)
                for other_parent in target_layer.parents
                if other_parent != parent_layer_label
            ]
        )
    ):
        print(
            f"Parent {parent_layer_label} of {target_layer_label} has outs that match "
            f"{arg_type} {argloc_key} for {target_layer_label}, but is not logged as "
            f"such in parent_arg_positions."
        )
        return ValidationCheckResult.failed_result("arg_logging_mismatch")

    # Case 2 exemption: in-place RNG ops (bernoulli_) mutate the tensor
    # AFTER it was logged as an arg, so the saved out no longer matches
    # the saved_args snapshot.  This is expected and not a real mismatch.
    if (
        not parent_layer_matches_arg
        and parent_layerged_as_arg
        and parent_layer.func_name == "bernoulli_"
    ):
        return ValidationCheckResult.validated("arg_logging_matched")

    # Case 3: parent is logged at this position but values don't match.
    if (not parent_layer_matches_arg) and parent_layerged_as_arg:
        print(
            f"Parent {parent_layer_label} of {target_layer_label} is logged as {arg_type} {argloc_key} to "
            f"{target_layer_label}, but its saved outs don't match the saved argument."
        )
        return ValidationCheckResult.failed_result("arg_logging_mismatch")

    return ValidationCheckResult.validated("arg_logging_matched")


def _check_perturbation_exemptions(
    self: "Trace",
    layer: Op,
    layers_to_perturb: List[str],
) -> bool:
    """Check whether a perturbation check should be skipped for registry-based reasons.

    Checks three exemption sources in order:
    1. Empty tensors (numel==0) -- perturbing an empty tensor is meaningless.
    2. Structural arg positions (``STRUCTURAL_ARG_POSITIONS``) -- the perturbed
       layer occupies a position that controls structure, not values (e.g.,
       index tensors for ``embedding``, ``index_select``).
    3. Custom exemption functions (``CUSTOM_EXEMPTION_CHECKS``) -- op-specific
       logic for complex cases like ``__getitem__``, ``lstm``, etc.

    Returns True if the perturbation is exempt (caller should skip), False otherwise.
    """
    # Empty tensors cannot be meaningfully perturbed.
    for perturbed_label in layers_to_perturb:
        p_entry = self[perturbed_label]
        if p_entry.out is not None and p_entry.out.numel() == 0:
            return True

    func_name = layer.func_name

    # Registry 3: structural arg positions (e.g., index tensor for embedding).
    if func_name in STRUCTURAL_ARG_POSITIONS:
        if perturbed_layer_at_structural_position(
            self, layer, layers_to_perturb, STRUCTURAL_ARG_POSITIONS[func_name]
        ):
            return True

    # Registry 4: custom per-op exemption checks.
    if func_name in CUSTOM_EXEMPTION_CHECKS:
        if CUSTOM_EXEMPTION_CHECKS[func_name](self, layer, layers_to_perturb):
            return True

    return False


def _execute_func_with_restored_state(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: List[str],
    layer_label: str,
    verbose: bool,
) -> Any:
    """Execute a layer's function with restored RNG and autocast state.

    Restores the exact RNG state that was active when this layer originally
    ran, so that stochastic ops (dropout, etc.) reproduce identical results
    during forward replay.

    **Exception handling**: catches ALL exceptions and returns ``None``.
    The caller treats ``None`` as:
    - For ``perturb=True``: an exempt perturbation (perturbation caused an
      invalid input combination, e.g., out-of-range indices -- this is
      expected and not a validation failure).
    - For ``perturb=False``: a failed validation (the layer's own function
      can't be replayed, which IS a problem).

    Returns the recomputed output tensor, or None on exception.
    """
    layer_func = layer.func

    try:
        recomputed_output = execute_with_restored_rng_autocast(
            layer_func,
            tuple(input_args["args"]),
            dict(input_args["kwargs"]),
            rng_states=layer.func_rng_states,
            autocast_state=layer.func_autocast_state,
        )
    except Exception as e:
        # Broad catch: perturbed values can trigger any exception (shape
        # mismatch, index OOB, dtype error, etc.).  Returning None lets the
        # caller decide whether this is acceptable.
        if verbose:
            print(
                f"Perturbation of {layers_to_perturb} for layer "
                f"{layer_label} caused {type(e).__name__}: {e}"
            )
        return None

    # In-place mutating ops (__setitem__, zero_, __delitem__) return None
    # from PyTorch but the "output" is the mutated first argument.
    if layer_func.__name__ in ("__setitem__", "zero_", "__delitem__"):
        recomputed_output = input_args["args"][0]

    # Multi-output functions may return typed containers; select the specific
    # output this layer represents using the captured typed path when available.
    container_path = tuple(getattr(layer, "container_path", ()) or ())
    if container_path and not isinstance(recomputed_output, torch.Tensor):
        recomputed_output = _slice_recomputed_output_by_path(recomputed_output, container_path)
    elif isinstance(recomputed_output, (list, tuple)):
        recomputed_output = recomputed_output[layer.multi_output_index]

    return recomputed_output


def _slice_recomputed_output_by_path(
    output: Any,
    path: tuple[OutputPathComponent, ...],
) -> Any:
    """Return the recomputed output leaf addressed by a typed container path.

    Parameters
    ----------
    output:
        Function return value.
    path:
        Captured output path for one tensor leaf.

    Returns
    -------
    Any
        Leaf value at the requested path.
    """

    current = output
    for component in path:
        current = _index_recomputed_output_component(current, component)
    return current


def _index_recomputed_output_component(output: Any, component: OutputPathComponent) -> Any:
    """Index one component into a recomputed output container.

    Parameters
    ----------
    output:
        Current output container.
    component:
        Typed path component.

    Returns
    -------
    Any
        Nested value.
    """

    if isinstance(component, TupleIndex):
        return output[component.index]
    if isinstance(component, DictKey):
        return output[component.key]
    if isinstance(component, NamedField):
        return getattr(output, component.name)
    if isinstance(component, DataclassField):
        return getattr(output, component.name)
    if isinstance(component, HFKey):
        return output[component.key]
    if isinstance(component, int):
        return output[component]
    if isinstance(component, str):
        return output[component]
    raise TypeError(f"Unsupported output path component {component!r}.")


def _reduced_numel_over_dims(tensor: torch.Tensor, dim: Any, default_all: bool) -> int:
    """Return how many input elements are summed into each output element.

    For a dimension-reducing op (``sum``/``mean``/...), the per-output
    accumulation depth is the product of the reduced dimension sizes. ``dim``
    follows torch's reduce conventions: ``None`` reduces over every dimension
    (when ``default_all``), an int or sequence of ints names the reduced dims.

    Runtime-verified torch behavior for an EXPLICIT empty ``dim=()`` (torch 2.x):
    for the reduce family this predicate gates (``sum``/``mean``/``nansum``/
    ``nanmean``/``var``/``std``/``norm``/``amax``/``amin``), ``dim=()`` is NOT an
    identity/no-op -- it reduces over EVERY dimension and returns a scalar (e.g.
    ``torch.sum(x, dim=()).shape == ()``). So ``dim=()`` is treated as reduce-all
    (depth = ``numel``), the same as ``dim=None`` with ``default_all`` -- not depth
    1. (``prod``/``logsumexp`` reject a tuple ``dim`` outright at runtime, so a
    captured op there will not present ``dim=()`` to this path.)
    """

    if dim is None:
        return int(tensor.numel()) if default_all else 1
    if isinstance(dim, int):
        dims: tuple[int, ...] = (dim,)
    else:
        try:
            dims = tuple(int(d) for d in dim)
        except TypeError:
            return 0
    if not dims:
        # Empty dim. For default_all reducers an explicit dim=() reduces over all
        # dims (runtime-verified scalar output), matching dim=None's reduce-all.
        return int(tensor.numel()) if default_all else 1
    ndim = tensor.dim()
    depth = 1
    for d in dims:
        depth *= int(tensor.shape[d % ndim]) if ndim else 1
    return depth


def _op_reduction_depth(layer: Op) -> int:
    """Return the per-output FP32 accumulation depth for a replay op.

    The reduction depth is the number of multiply-adds summed into each output
    element, which is what drives accumulation-order round-off between an original
    op and its faithful replay. It is read from the op's saved operand shapes /
    indices, consulting BOTH ``saved_args`` and ``saved_kwargs`` so that operands
    passed by keyword (``conv2d(x, weight=w)``, ``matmul(a, other=b)``,
    ``scatter_add(out, dim=d, index=idx, src=s)``) are still measured correctly --
    reading positionally only would drop the operand to depth 0 and wrongly
    withhold band C, a fresh false-negative.

    Depth by category:

    - forward ``conv*``: ``in_channels/groups * prod(kernel_size)`` =
      ``weight.numel() / weight.shape[0]`` products per output element (forward
      conv weight is ``[out_channels, in_channels/groups, *kernel]``, so the
      out-channel axis is ``shape[0]``).
    - ``conv_transpose*``: transposed-conv weight is
      ``[in_channels, out_channels/groups, *kernel]`` -- the IN-channel axis is
      ``shape[0]``. The true per-output accumulation depth is
      ``in_channels/groups * prod(kernel)`` = ``weight.shape[0] // groups *
      prod(weight.shape[2:])``, with ``groups`` read from positional arg 6 / kwarg
      ``groups`` (default 1; an unreadable or non-dividing ``groups`` returns 0,
      fail-toward-strict). The forward formula ``weight.numel() // weight.shape[0]``
      would read ``out_channels/groups * prod(kernel)``, and omitting the
      ``// groups`` divisor over-reports a grouped transpose (e.g.
      ``ConvTranspose2d(128, 128, 1, groups=128)`` is depth 1, not 128).
    - ``linear`` / ``addmm`` / ``mm`` / ``matmul`` / ``bmm`` / ``baddbmm``:
      the contracted dimension ``K`` of the first matrix operand.
    - ``scatter_add*`` / additive ``scatter_reduce*`` (``reduce="sum"``/``"mean"``)
      / ``segment_reduce`` / ``index_add``: the max number of source elements
      summed into a single destination COORDINATE (the graph fan-in = max
      duplicate destination-tuple count, counted per actual destination position,
      NOT per raw index value). Plain overwrite ``scatter``/``scatter_`` and
      ``max``/``min``/``amax``/``amin``/``prod`` reduce-modes do not accumulate and
      are not in the eligible set, so they stay strict. A depth-2 scatter (two
      sources per destination) stays well below threshold.
    - ``sum`` / ``mean`` / ``prod`` / ``norm`` / ``var`` / ``std`` and other
      dimension reductions: the numel of the reduced dimension(s).
    - elementwise / copy / view / structural ops: 1.

    Returns
    -------
    int
        Per-output accumulation depth, or ``0`` when it cannot be determined.
        A return of ``0`` conservatively withholds band C (fail-toward-strict).
    """

    saved_args: Sequence[Any] = getattr(layer, "saved_args", None) or ()
    saved_kwargs: Dict[str, Any] = getattr(layer, "saved_kwargs", None) or {}
    func_name = layer.func_name

    def _operand(index: int, *kwarg_names: str) -> Any:
        """Read an operand positionally, falling back to its keyword names."""
        if index < len(saved_args):
            return saved_args[index]
        for name in kwarg_names:
            if name in saved_kwargs:
                return saved_kwargs[name]
        return None

    # conv_transpose* (checked FIRST -- "conv_transpose" also startswith "conv"):
    # transposed weight is [in_channels, out_channels/groups, *kernel], so the true
    # per-output accumulation depth is in_channels/groups * prod(kernel) =
    # shape[0] // groups * prod(shape[2:]). groups is read from positional arg 6 /
    # kwarg "groups" (default 1). Omitting the // groups divisor over-reports a
    # grouped transpose (ConvTranspose2d(128,128,1,groups=128) is depth 1, not 128);
    # an unreadable / non-dividing groups returns 0 (fail-toward-strict).
    if func_name.startswith(_CONV_TRANSPOSE_FUNC_PREFIX):
        weight = _operand(1, "weight")
        if isinstance(weight, torch.Tensor) and weight.dim() >= 2 and weight.shape[0] > 0:
            kernel_numel = 1
            for kdim in weight.shape[2:]:
                kernel_numel *= int(kdim)
            groups = _operand(6, "groups")
            if groups is None:
                groups = 1
            if not isinstance(groups, int) or groups <= 0 or int(weight.shape[0]) % groups != 0:
                return 0
            return (int(weight.shape[0]) // groups) * kernel_numel
        return 0

    # forward conv*: weight is [out_channels, in_channels/groups, *kernel], so depth
    # = in_channels/groups * prod(kernel) = weight.numel() // weight.shape[0].
    if func_name.startswith(_CONV_FUNC_PREFIX):
        weight = _operand(1, "weight")
        if isinstance(weight, torch.Tensor) and weight.dim() >= 2 and weight.shape[0] > 0:
            return int(weight.numel() // weight.shape[0])
        return 0

    # matmul / linear family: depth = contracted dimension K.
    if func_name in _MATMUL_LINEAR_FUNCS:
        if func_name == "linear":
            weight = _operand(1, "weight")
            if isinstance(weight, torch.Tensor) and weight.dim() >= 1:
                return int(weight.shape[-1])
            first = _operand(0, "input")
            if isinstance(first, torch.Tensor) and first.dim() >= 1:
                return int(first.shape[-1])
            return 0
        if func_name == "addmm":
            mat1 = _operand(1, "mat1")
            if isinstance(mat1, torch.Tensor) and mat1.dim() >= 1:
                return int(mat1.shape[-1])
            return 0
        if func_name == "baddbmm":
            first_matrix = _operand(1, "batch1")
        elif func_name in {"mm", "bmm"}:
            first_matrix = _operand(0, "input", "mat1")
        else:  # matmul
            first_matrix = _operand(0, "input")
        if isinstance(first_matrix, torch.Tensor) and first_matrix.dim() >= 1:
            return int(first_matrix.shape[-1])
        return 0

    # Additive scatter / segment / index_add: depth = max per-coordinate fan-in.
    # scatter_reduce* is conditionally additive -- only its sum/mean modes accumulate;
    # a prod/amax/amin reduce-mode does not drift and is withheld (depth 0).
    if func_name in _SCATTER_REDUCE_FUNCS:
        if func_name.startswith("scatter_reduce"):
            # scatter_reduce(input, dim, index, src, reduce, *, include_self): the
            # reduce mode arrives as kwarg "reduce" (TorchLens-captured form) or as
            # positional arg 4 (free-function torch.scatter_reduce(..., reduce)).
            reduce_mode = saved_kwargs.get("reduce", None)
            if reduce_mode is None and len(saved_args) > 4:
                reduce_mode = saved_args[4]
            if reduce_mode not in _ADDITIVE_SCATTER_REDUCE_MODES:
                return 0
        return _scatter_fan_in_depth(func_name, saved_args, saved_kwargs)

    # dimension reductions (sum/mean/...): depth = numel of reduced dims.
    if func_name in _DIM_REDUCE_FUNCS:
        tensor = _operand(0, "input")
        if not isinstance(tensor, torch.Tensor):
            return 0
        dim = saved_kwargs.get("dim", None)
        if dim is None:
            dim = _operand(1, "dim")
        # sum/mean/prod with no dim reduce over all elements; var/std/norm too.
        return _reduced_numel_over_dims(tensor, dim, default_all=True)

    # Everything else (elementwise, view, structural, copy) has depth 1: a single
    # value flows through per output element, so no FP32 reorder drift accrues.
    return 1


def _scatter_fan_in_depth(
    func_name: str,
    saved_args: Any,
    saved_kwargs: Dict[str, Any],
) -> int:
    """Return the max number of source elements aggregated per destination slot.

    For ``scatter_add``-family ops the per-output accumulation depth is the graph
    fan-in: the maximum number of source elements that sum into a single
    destination COORDINATE. This MUST be counted per actual destination position,
    not per raw index value: a row-/feature-wise scatter writes many independent
    destinations at index ``0``, and counting the duplicate raw index value ``0``
    globally would conflate those independent destinations into one huge fan-in and
    wrongly grant band C to a genuinely shallow (e.g. depth-2) scatter.

    For an n-D ``scatter_add``/``scatter_reduce`` (``self``, ``index`` and ``src``
    all share rank), the destination coordinate of source element at position
    ``(i_0, ..., i_{n-1})`` is ``index[...]`` along the scatter ``dim`` and the
    SOURCE position ``i_d`` along every other dim. We rebuild the full destination
    tuple per source element and count the max number of identical tuples. A
    1-D ``index_add`` (index value IS the whole destination coordinate) and
    ``segment_reduce`` (max segment length) are already per-coordinate.
    """

    def _arg(index: int, *kwarg_names: str) -> Any:
        if index < len(saved_args):
            return saved_args[index]
        for name in kwarg_names:
            if name in saved_kwargs:
                return saved_kwargs[name]
        return None

    if func_name.startswith("index_add"):
        # index_add(input, dim, index, source): 1-D index value IS the full
        # destination coordinate along the indexed dim, so each raw index value
        # already names a distinct destination -- count raw duplicates directly.
        index = _arg(2, "index")
        return _max_raw_index_multiplicity(index)
    if func_name == "segment_reduce":
        # segment_reduce(data, reduce, lengths=...): fan-in = max segment length.
        lengths = saved_kwargs.get("lengths", None)
        if isinstance(lengths, torch.Tensor) and lengths.numel() > 0:
            return int(lengths.max().item())
        return 0

    # scatter_add*/scatter_reduce*: index is the 3rd operand, dim the 2nd.
    index = _arg(2, "index")
    if not isinstance(index, torch.Tensor) or index.numel() == 0:
        return 0
    dim = _arg(1, "dim")
    if not isinstance(dim, int):
        return 0
    return _max_destination_coordinate_fan_in(index, dim)


def _max_raw_index_multiplicity(index: Any) -> int:
    """Return the max count of any repeated raw index value (1-D index_add fan-in)."""

    if not isinstance(index, torch.Tensor) or index.numel() == 0:
        return 0
    try:
        flat = index.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat - flat.min())
        return int(counts.max().item())
    except (RuntimeError, ValueError):
        return 0


def _max_destination_coordinate_fan_in(index: torch.Tensor, dim: int) -> int:
    """Return the max number of n-D scatter source elements per destination tuple.

    Builds, for each source element, the full destination coordinate -- ``index``
    along the (normalized) scatter ``dim`` and the source position along every other
    dim -- then counts the largest group of identical destination coordinates. Two
    sources landing on the same destination report depth 2 no matter how many other
    destinations also happen to be indexed at the same raw value.
    """

    try:
        ndim = index.dim()
        if ndim == 0:
            return int(index.numel())
        scatter_dim = dim % ndim
        idx64 = index.to(torch.int64)
        # Per-axis destination coordinate: the index value along the scatter dim,
        # the source position (arange, broadcast) along every other dim.
        coord_axes = []
        for axis in range(ndim):
            if axis == scatter_dim:
                coord_axes.append(idx64.reshape(-1))
            else:
                shape = [1] * ndim
                shape[axis] = index.shape[axis]
                axis_positions = (
                    torch.arange(index.shape[axis], dtype=torch.int64)
                    .reshape(shape)
                    .expand_as(index)
                    .reshape(-1)
                )
                coord_axes.append(axis_positions)
        dest_tuples = torch.stack(coord_axes, dim=1)
        _, counts = torch.unique(dest_tuples, dim=0, return_counts=True)
        return int(counts.max().item())
    except (RuntimeError, ValueError):
        return 0


def _deep_numeric_replay_matches_saved(
    layer: Op,
    recomputed_output: torch.Tensor,
) -> bool:
    """Return whether a deep numeric replay matches within local relaxed tolerance.

    The standard validation tolerance remains the default for every layer. This
    fallback is intentionally narrow: it only applies to REDUCTION ops whose
    per-output accumulation depth is at least
    ``DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH`` (a deep conv/matmul/scatter/sum,
    where FP32 reorder round-off genuinely accrues). It first tries a modest
    ``allclose`` relaxation, then allows a tiny fraction of stricter-check
    outliers only when the overall scaled error is still very small. A shallow
    op (depth < 64) -- or any op whose depth cannot be determined (depth 0) --
    is INELIGIBLE and stays on the strict global tolerance (fail-toward-strict).

    Parameters
    ----------
    layer:
        Layer whose saved out is being replayed.
    recomputed_output:
        Output from re-executing ``layer.func`` on saved parent values.

    Returns
    -------
    bool
        True if this layer qualifies for the deep numeric replay tolerance and
        the recomputed output is close enough to the saved out.
    """
    saved_output = layer.out
    if saved_output is None:
        return False
    depth = _op_reduction_depth(layer)
    if depth < DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH:
        return False
    if recomputed_output.shape != saved_output.shape:
        return False
    if recomputed_output.dtype != saved_output.dtype:
        return False
    if not recomputed_output.is_floating_point():
        return False

    from .._state import pause_logging

    with pause_logging():
        if not torch.equal(recomputed_output.isnan(), saved_output.isnan()):
            return False
        if not torch.equal(recomputed_output.isinf(), saved_output.isinf()):
            return False

        recomputed_nonan = torch.nan_to_num(recomputed_output, 0.7234691827346)
        saved_nonan = torch.nan_to_num(saved_output, 0.7234691827346)

        if torch.allclose(
            recomputed_nonan,
            saved_nonan,
            rtol=DEEP_NUMERIC_REPLAY_RTOL,
            atol=DEEP_NUMERIC_REPLAY_ATOL,
        ):
            return True

        close = torch.isclose(
            recomputed_nonan,
            saved_nonan,
            rtol=DEEP_NUMERIC_REPLAY_OUTLIER_RTOL,
            atol=DEEP_NUMERIC_REPLAY_OUTLIER_ATOL,
        )
        outlier_fraction = (~close).sum().item() / close.numel()
        if outlier_fraction > DEEP_NUMERIC_REPLAY_MAX_OUTLIER_FRACTION:
            return False

        diff = (recomputed_nonan - saved_nonan).abs()
        scale = torch.maximum(recomputed_nonan.abs(), saved_nonan.abs()) + 1e-12
        scaled_diff = diff / scale
        return bool(
            scaled_diff.max().item() <= DEEP_NUMERIC_REPLAY_MAX_SCALED_DIFF
            and scaled_diff.mean().item() <= DEEP_NUMERIC_REPLAY_MAX_MEAN_SCALED_DIFF
        )


def _check_whether_func_on_saved_parents_yields_saved_tensor(
    self: "Trace",
    layer_to_validate_parents_for_label: str,
    perturb: bool = False,
    layers_to_perturb: Optional[List[str]] = None,
    verbose: bool = False,
) -> ValidationCheckResult:
    """Checks whether executing the saved function for a layer on the saved value of its parent layers
    in fact yields the saved outs for that layer.

    Args:
        layer_to_validate_parents_for_label: label of the layer to check the saved outs
        perturb: whether to perturb the saved outs
        layers_to_perturb: layers for which to perturb the saved outs

    Returns:
        Structured validation decision for this replay/perturbation attempt.
    """
    if layers_to_perturb is None:
        layers_to_perturb = []

    layer = self[layer_to_validate_parents_for_label]
    _raise_if_replay_arg_version_data_incomplete(self, layer)

    # Early exits for layers that cannot or should not be replayed.

    if layer.func is None:
        if _is_provable_functionless_source_or_boundary(layer):
            return ValidationCheckResult.exempted("functionless_source_or_boundary")
        return ValidationCheckResult.failed_result("functionless_computational_op")

    # Registry 1: skip ALL validation for nondeterministic ops (e.g., empty_like).
    if layer.func_name in SKIP_VALIDATION_ENTIRELY:
        return ValidationCheckResult.exempted(
            "uninitialized_by_design",
            justification=SKIP_VALIDATION_ENTIRELY[layer.func_name],
        )

    # Pre-execution perturbation exemptions (structural args, custom checks).
    if perturb and _check_perturbation_exemptions(self, layer, layers_to_perturb):
        return ValidationCheckResult.exempted("pre_perturbation_exemption")

    input_args, unverified_reason = _prepare_input_args_for_validating_layer(
        self, layer, layers_to_perturb
    )
    if input_args is None:
        return ValidationCheckResult.unverified(unverified_reason or "missing_saved_args")

    recomputed_output = _execute_func_with_restored_state(
        layer, input_args, layers_to_perturb, layer_to_validate_parents_for_label, verbose
    )

    # None means execution raised an exception (see _execute_func_with_restored_state).
    if recomputed_output is None:
        if not perturb:
            # Non-perturbed replay failure is a real validation failure.
            import warnings

            warnings.warn(
                f"Validation replay raised an exception for layer "
                f"{layer_to_validate_parents_for_label}; treating as failed validation."
            )
            from .diagnostics import (
                CHECK_REPLAY,
                ValidationFailure,
                record_validation_failure,
            )

            record_validation_failure(
                self,
                ValidationFailure(
                    check=CHECK_REPLAY,
                    op_label=layer_to_validate_parents_for_label,
                    func_name=getattr(layer, "func_name", None),
                    message="replay re-execution raised an exception (run verbose for traceback)",
                ),
            )
            return ValidationCheckResult.failed_result("replay_execution_exception")
        # Perturbed execution raised -- the perturbed values caused an invalid
        # input (e.g., wrong shape). It is unverified, not a pass.
        return ValidationCheckResult.unverified("perturbation_execution_exception")

    matches_saved = tensor_nanequal(recomputed_output, layer.out, allow_tolerance=True)
    if not matches_saved and not perturb and isinstance(recomputed_output, torch.Tensor):
        matches_saved = _deep_numeric_replay_matches_saved(layer, recomputed_output)

    # Forward replay failure (non-perturbed): saved outs don't match.
    if not matches_saved and not perturb:
        # Exemption: parent is an in-place RNG op that may have mutated its
        # tensor after the child logged it as an arg.
        parent_has_inplace_rng = any(self[p].func_name == "bernoulli_" for p in layer.parents)
        if parent_has_inplace_rng:
            return ValidationCheckResult.exempted("parent_inplace_rng_bernoulli")
        # Surface the computed reduction depth so a band-C miss is diagnosable:
        # depth < 64 means the op was (correctly) ineligible for the deep-numeric
        # tolerance; a large depth that still failed points at a real replay bug.
        reduction_depth = (
            _op_reduction_depth(layer) if isinstance(recomputed_output, torch.Tensor) else None
        )
        depth_note = (
            f" (reduction_depth={reduction_depth}, band-C eligibility threshold="
            f"{DEEP_NUMERIC_REPLAY_MIN_REDUCTION_DEPTH})"
            if reduction_depth is not None
            else ""
        )
        print(
            f"Saved outs for layer {layer_to_validate_parents_for_label} do not match the "
            f"values computed based on the parent layers {layer.parents}{depth_note}."
        )
        from .diagnostics import (
            CHECK_REPLAY,
            describe_tensor_mismatch,
            record_validation_failure,
        )

        record_validation_failure(
            self,
            describe_tensor_mismatch(
                layer.out,
                recomputed_output,
                check=CHECK_REPLAY,
                op_label=layer_to_validate_parents_for_label,
                func_name=getattr(layer, "func_name", None),
                reduction_depth=reduction_depth,
                message="isolated replay does not reproduce saved out",
            ),
        )
        return ValidationCheckResult.failed_result("replay_mismatch")

    # Perturbation produced identical output -- run posthoc checks to see if
    # there's a valid excuse (bool output, special-value args, type cast, etc.).
    # Uses exact equality (no tolerance) since any change should be detectable.
    if perturb and tensor_nanequal(recomputed_output, layer.out, allow_tolerance=False):
        perturb_ok = posthoc_perturb_check(self, layer, layers_to_perturb, verbose)
        if not perturb_ok:
            # ADD-ONLY: a genuine perturbation-insensitivity failure (the output
            # did not change when a parent's value was perturbed and no posthoc
            # excuse applied). Record the structured reason -- this NEVER changes
            # the decision posthoc_perturb_check already returned.
            from .diagnostics import (
                CHECK_PERTURBATION,
                ValidationFailure,
                record_validation_failure,
            )

            record_validation_failure(
                self,
                ValidationFailure(
                    check=CHECK_PERTURBATION,
                    op_label=layer_to_validate_parents_for_label,
                    func_name=getattr(layer, "func_name", None),
                    message=(
                        "output insensitive to perturbing parent(s) "
                        f"{layers_to_perturb}; the parent does not influence this op's value"
                    ),
                    extra={"perturbed_parents": list(layers_to_perturb)},
                ),
            )
            return ValidationCheckResult.failed_result("perturbation_insensitive")
        return ValidationCheckResult.exempted("posthoc_perturbation_exemption")

    return ValidationCheckResult.validated("perturbation_changed" if perturb else "replay_matched")


def _prepare_input_args_for_validating_layer(
    self: "Trace",
    layer_to_validate_parents_for: Op,
    layers_to_perturb: List[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """Build the input argument dict for replaying a layer's function.

    Starts from the layer's saved ``saved_args`` / ``saved_kwargs``,
    deep-clones all tensors to prevent in-place mutation during replay, then
    swaps in each parent's saved out (or a perturbed version) at the
    correct argument position.

    For nested argument positions (tuples, dicts nested inside args), the
    ``parent_arg_positions`` key is a tuple ``(outer_key, inner_key)`` and
    ``assign_to_sequence_or_dict`` handles the nested assignment.

    Args:
        layer_to_validate_parents_for: Layer being checked.
        layers_to_perturb: Layers for which to perturb the saved outs.

    Returns:
        Tuple of prepared replay args and an optional unverified reason code.
    """
    if layer_to_validate_parents_for.saved_args is None:
        return None, "missing_saved_args"
    input_args = {
        "args": list(layer_to_validate_parents_for.saved_args[:]),
        "kwargs": layer_to_validate_parents_for.saved_kwargs.copy(),
    }
    input_args = _copy_validation_args(input_args)
    _restore_live_parameter_args_for_replay(input_args, layer_to_validate_parents_for)

    # Swap in saved parent outs:

    for arg_type in ["args", "kwargs"]:
        for (
            key,
            parent_layer_arg,
        ) in layer_to_validate_parents_for.parent_arg_positions[arg_type].items():
            parent_layer = self[parent_layer_arg]
            target_op_label = getattr(layer_to_validate_parents_for, "label", None)
            if target_op_label in parent_layer.out_versions_by_child:
                parent_values = parent_layer.out_versions_by_child[target_op_label]
            elif layer_to_validate_parents_for.layer_label in parent_layer.out_versions_by_child:
                parent_values = parent_layer.out_versions_by_child[
                    layer_to_validate_parents_for.layer_label
                ]
            else:
                if _is_buffer_version_parent(parent_layer) and not _buffer_parent_source_equal(
                    parent_layer,
                    input_args,
                    arg_type,
                    key,
                ):
                    raise ValueError(
                        "Validation cannot source-match buffer-version parent "
                        f"'{parent_layer_arg}' for child "
                        f"'{layer_to_validate_parents_for.layer_label}'."
                    )
                parent_values = parent_layer.out
            if parent_values is None:
                if parent_layer_arg in layers_to_perturb:
                    return None, "missing_saved_parent_payload"
                continue
            parent_values = parent_values.detach().clone()

            if parent_layer_arg in layers_to_perturb:
                parent_layer_func_values = _perturb_parent_values_for_layer(
                    layer_to_validate_parents_for,
                    parent_layer_arg,
                    parent_values,
                )
            else:
                parent_layer_func_values = parent_values

            if type(key) != tuple:
                input_args[arg_type][key] = parent_layer_func_values
            else:
                input_args[arg_type][key[0]] = assign_to_sequence_or_dict(
                    input_args[arg_type][key[0]], key[1], parent_layer_func_values
                )

    return input_args, None


def _perturb_parent_values_for_layer(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor:
    """Return perturbed parent values tailored to the child op when needed.

    Parameters
    ----------
    layer:
        Child op being replayed.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved parent tensor values.

    Returns
    -------
    torch.Tensor
        Perturbed values intended to be valid inputs for ``layer`` while still
        differing from the saved parent.
    """

    one_hot_values = _perturb_one_hot_indices(layer, parent_label, parent_values)
    if one_hot_values is not None:
        return one_hot_values
    return _perturb_layer_outs(parent_values, layer.out)


def _perturb_one_hot_indices(
    layer: Op,
    parent_label: str,
    parent_values: torch.Tensor,
) -> torch.Tensor | None:
    """Return a valid alternate class-index tensor for ``one_hot`` replay.

    Parameters
    ----------
    layer:
        Candidate ``one_hot`` op.
    parent_label:
        Parent label selected for perturbation.
    parent_values:
        Saved class-index tensor.

    Returns
    -------
    torch.Tensor or None
        Valid class indices distinct from ``parent_values`` when this is the
        value parent of a ``one_hot`` op; otherwise ``None``.
    """

    if layer.func_name != "one_hot":
        return None
    if not _parent_label_occupies_arg_position(layer, parent_label, 0):
        return None
    if parent_values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return None
    num_classes = _one_hot_num_classes(layer)
    if num_classes is None or num_classes <= 1:
        return None
    return (parent_values + 1).remainder(num_classes)


def _parent_label_occupies_arg_position(layer: Op, parent_label: str, position: int) -> bool:
    """Return whether a parent label occupies a positional argument.

    Parameters
    ----------
    layer:
        Op whose parent-argument map should be inspected.
    parent_label:
        Parent label to find.
    position:
        Positional argument index.

    Returns
    -------
    bool
        True when ``parent_label`` is registered at ``args[position]``.
    """

    return (getattr(layer, "parent_arg_positions", None) or {}).get("args", {}).get(
        position
    ) == parent_label


def _one_hot_num_classes(layer: Op) -> int | None:
    """Return the captured ``num_classes`` for a ``one_hot`` op.

    Parameters
    ----------
    layer:
        Candidate ``one_hot`` op.

    Returns
    -------
    int or None
        Positive class count when captured, otherwise ``None``.
    """

    saved_kwargs = getattr(layer, "saved_kwargs", {}) or {}
    num_classes = saved_kwargs.get("num_classes")
    saved_args = cast(Sequence[Any], getattr(layer, "saved_args", ()) or ())
    if num_classes is None and len(saved_args) > 1:
        num_classes = saved_args[1]
    if isinstance(num_classes, int):
        return num_classes
    return None


def _restore_live_parameter_args_for_replay(input_args: dict[str, Any], layer: Op) -> None:
    """Restore live parameter objects into replay args when snapshots still match.

    Some PyTorch kernels, notably oneDNN-backed convolutions, can replay with
    slightly different accumulation when a model ``nn.Parameter`` is replaced
    by a detached tensor clone. The original call consumed the live parameter
    object, so validation should do the same when the source model is still
    available and the live parameter remains exactly equal to the saved
    argument snapshot. If the parameter has changed or cannot be resolved, the
    saved tensor clone is left in place so stale/corrupt captures still fail.

    Parameters
    ----------
    input_args:
        Replay argument mapping created from saved args/kwargs.
    layer:
        Operation being replayed.

    Returns
    -------
    None
        ``input_args`` is updated in place.
    """

    if getattr(layer, "is_inplace", False):
        return
    param_logs = tuple(getattr(layer, "_param_logs", ()) or ())
    if not param_logs:
        return

    skip_paths = _parent_arg_paths(layer)
    live_params = tuple(_available_live_params(param_logs))
    if not live_params:
        return

    input_args["args"] = [
        _restore_live_parameter_leaf(arg, live_params, skip_paths["args"], (index,))
        for index, arg in enumerate(input_args["args"])
    ]
    input_args["kwargs"] = {
        key: _restore_live_parameter_leaf(value, live_params, skip_paths["kwargs"], (key,))
        for key, value in input_args["kwargs"].items()
    }


def _parent_arg_paths(layer: Op) -> dict[str, set[tuple[Any, ...]]]:
    """Return argument paths occupied by dataflow parent tensors.

    Parameters
    ----------
    layer:
        Operation whose parent argument positions should be skipped.

    Returns
    -------
    dict[str, set[tuple[Any, ...]]]
        Path sets for positional and keyword arguments.
    """

    parent_positions = getattr(layer, "parent_arg_positions", {}) or {}
    paths: dict[str, set[tuple[Any, ...]]] = {"args": set(), "kwargs": set()}
    for arg_type in ("args", "kwargs"):
        for key in parent_positions.get(arg_type, {}):
            paths[arg_type].add(key if isinstance(key, tuple) else (key,))
    return paths


def _available_live_params(param_logs: tuple[Any, ...]) -> list[torch.nn.Parameter]:
    """Resolve live parameters still reachable from a trace's source model.

    Parameters
    ----------
    param_logs:
        Parameter metadata objects attached to an op.

    Returns
    -------
    list[torch.nn.Parameter]
        Live parameters that could be resolved.
    """

    live_params: list[torch.nn.Parameter] = []
    for param_log in param_logs:
        try:
            live_param = getattr(param_log, "handle", None)
        except Exception:
            live_param = None
        if isinstance(live_param, torch.nn.Parameter):
            live_params.append(live_param)
    return live_params


def _restore_live_parameter_leaf(
    value: Any,
    live_params: tuple[torch.nn.Parameter, ...],
    skip_paths: set[tuple[Any, ...]],
    path: tuple[Any, ...],
) -> Any:
    """Replace a saved parameter clone with its matching live parameter.

    Parameters
    ----------
    value:
        Candidate replay argument value.
    live_params:
        Live parameters used by the operation.
    skip_paths:
        Argument paths occupied by graph parents.
    path:
        Current path within args or kwargs.

    Returns
    -------
    Any
        ``value`` with matching parameter leaves restored where unambiguous.
    """

    if path in skip_paths:
        return value
    if isinstance(value, torch.Tensor):
        matches = [
            live_param
            for live_param in live_params
            if _live_parameter_matches_snapshot(live_param, value)
        ]
        if len(matches) == 1:
            return matches[0]
        return value
    if isinstance(value, list):
        return [
            _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        restored = [
            _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, index))
            for index, item in enumerate(value)
        ]
        return type(value)(restored)
    if isinstance(value, dict):
        return {
            key: _restore_live_parameter_leaf(item, live_params, skip_paths, (*path, key))
            for key, item in value.items()
        }
    return value


def _live_parameter_matches_snapshot(
    live_param: torch.nn.Parameter,
    snapshot: torch.Tensor,
) -> bool:
    """Return whether a live parameter exactly matches a saved tensor snapshot.

    Parameters
    ----------
    live_param:
        Live model parameter that was used by an operation.
    snapshot:
        Saved replay argument tensor.

    Returns
    -------
    bool
        Whether shape, dtype, device, and values match exactly.
    """

    if tuple(live_param.shape) != tuple(snapshot.shape):
        return False
    if live_param.dtype != snapshot.dtype:
        return False
    if live_param.device != snapshot.device:
        return False
    return tensor_nanequal(live_param, snapshot, allow_tolerance=False)


def _is_buffer_version_parent(parent_layer: Op) -> bool:
    """Return whether a parent is a written buffer-version node."""

    return bool(
        getattr(parent_layer, "is_buffer", False)
        and getattr(parent_layer, "buffer_write_kind", None) is not None
    )


def _buffer_parent_source_equal(
    parent_layer: Op,
    input_args: dict[str, Any],
    arg_type: str,
    key: Any,
) -> bool:
    """Return whether a child saved arg already equals the buffer-version value."""

    parent_out = parent_layer.out
    if parent_out is None:
        return False
    try:
        if type(key) != tuple:
            saved_arg_value = input_args[arg_type][key]
        else:
            saved_arg_value = input_args[arg_type][key[0]]
            for nested_key in key[1:]:
                saved_arg_value = saved_arg_value[nested_key]
    except Exception:
        return False
    return tensor_nanequal(saved_arg_value, parent_out, allow_tolerance=False)


def _deep_clone_tensors(val: Any) -> Any:
    """Recursively clone all tensors in a nested structure of lists/tuples/dicts.

    Non-tensor leaves are returned as-is (shared reference).  Tensor leaves
    are detached and cloned so that in-place ops during validation replay
    don't corrupt the original saved data.

    Preserves container types: a tuple input produces a tuple output, not a list.
    """
    if isinstance(val, torch.Tensor):
        return val.detach().clone()
    elif isinstance(val, (list, tuple)):
        cloned = [_deep_clone_tensors(v) for v in val]
        # Preserve the original container type (list vs tuple vs namedtuple).
        return type(val)(cloned)
    elif isinstance(val, dict):
        return {k: _deep_clone_tensors(v) for k, v in val.items()}
    return val


def _copy_validation_args(input_args: dict[str, Any]) -> dict[str, Any]:
    """Deep-clone all tensors in the input argument dict to avoid in-place mutation during validation.

    Args:
        input_args: Dict with ``"args"`` (list) and ``"kwargs"`` (dict) keys holding
            the raw creation arguments for a layer.

    Returns:
        A new dict with the same structure but with every tensor replaced by a detached clone.
    """
    return {
        "args": [_deep_clone_tensors(v) for v in input_args["args"]],
        "kwargs": {k: _deep_clone_tensors(v) for k, v in input_args["kwargs"].items()},
    }


def _perturb_layer_outs(parent_outs: torch.Tensor, output_outs: torch.Tensor) -> torch.Tensor:
    """Generate a random perturbation of a saved tensor for validation.

    The perturbation strategy varies by dtype to produce meaningful
    "wrong" values while respecting type constraints:

    - **Integer types**: sample uniformly from [min, max+1) of the original
      tensor, with the exclusive high bound clamped by ``torch.iinfo`` so the
      ``max + 1`` cannot overflow the dtype (a saturated range widens to the
      full valid dtype range instead).  If the tensor is all-one-value (single
      unique value), widen the range to [-10, 11) or [0, 11).  Retries up to
      ``MAX_PERTURB_ATTEMPTS`` to avoid accidentally reproducing the original.
    - **Bool**: random 0/1, retried to ensure difference.
    - **Float/complex**: scaled random normal, where the scale is derived from
      the output tensor's mean absolute value (plus random jitter) to produce
      perturbations of comparable magnitude.

    Args:
        parent_outs: The original parent tensor to perturb.
        output_outs: The child layer's output tensor, used to calibrate
            the perturbation scale for float types.

    Returns:
        A new tensor of the same shape/dtype with perturbed values.
    """
    device = parent_outs.device
    if parent_outs.numel() == 0:
        return parent_outs.detach().clone()

    if parent_outs.dtype in [
        torch.int,
        torch.long,
        torch.short,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        tensor_unique_vals = torch.unique(parent_outs)
        if len(tensor_unique_vals) > 1:
            # Multiple unique values: sample within the original range.
            #
            # Compute the bounds as Python ints and clamp the exclusive high
            # bound with ``torch.iinfo`` so ``max() + 1`` can never overflow the
            # dtype. A legitimately captured tensor holding ``iinfo.max`` (e.g.
            # PyG sentinel/cluster index tensors carry ``INT64_MAX``) would
            # otherwise wrap ``max() + 1`` to ``INT64_MIN`` and make
            # ``torch.randint`` raise "random_ expects 'from' to be less than
            # 'to'". This mirrors the ``torch.finfo`` clamping the float branch
            # already does; it only bounds the *sampling range*, never skips the
            # perturbation, so the tripwire stays non-vacuous.
            int_info = torch.iinfo(parent_outs.dtype)
            int_lo = int(parent_outs.min().item())
            int_hi = int(parent_outs.max().item())
            int_hi_excl = int_hi + 1 if int_hi < int_info.max else int_info.max
            if int_lo >= int_hi_excl:
                # Saturated range (e.g. the whole tensor sits at ``iinfo.max``):
                # widen to the full valid dtype range so we can still draw a
                # genuinely different value rather than degenerating to a no-op.
                int_lo, int_hi_excl = int_info.min, int_info.max
            perturbed_outs = parent_outs.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                perturbed_outs = torch.randint(
                    int_lo,
                    int_hi_excl,
                    size=parent_outs.shape,
                    device=device,
                ).type(parent_outs.dtype)
                if not torch.equal(perturbed_outs, parent_outs):
                    break
        else:
            # Single unique value: widen to a fixed range to guarantee a
            # different value can be produced.
            perturbed_outs = parent_outs.detach().clone()
            for _ in range(MAX_PERTURB_ATTEMPTS):
                if torch.min(parent_outs) < 0:
                    perturbed_outs = torch.randint(
                        -10, 11, size=parent_outs.shape, device=device
                    ).type(parent_outs.dtype)
                else:
                    perturbed_outs = torch.randint(
                        0, 11, size=parent_outs.shape, device=device
                    ).type(parent_outs.dtype)
                if not torch.equal(perturbed_outs, parent_outs):
                    break

    elif parent_outs.dtype == torch.bool:
        # Random bool, retried until different from original.
        perturbed_outs = parent_outs.detach().clone()
        for _ in range(MAX_PERTURB_ATTEMPTS):
            perturbed_outs = torch.randint(0, 2, size=parent_outs.shape, device=device).bool()
            if not torch.equal(perturbed_outs, parent_outs):
                break
    else:
        # Float/complex: uniform random within the original value range.
        # Using the original range ensures perturbed values stay in the
        # valid domain for range-restricted functions (e.g., bernoulli
        # requires probabilities in [0,1]).  For typical tensors with wide
        # range this produces meaningfully different values.
        if parent_outs.is_complex():
            real = parent_outs.real.float()
            imag = parent_outs.imag.float()
            r_lo, r_hi = real.min().item(), real.max().item()
            i_lo, i_hi = imag.min().item(), imag.max().item()
            if r_lo == r_hi:
                r_lo, r_hi = r_lo - 1.0, r_hi + 1.0
            if i_lo == i_hi:
                i_lo, i_hi = i_lo - 1.0, i_hi + 1.0
            perturbed_outs = torch.complex(
                torch.rand(parent_outs.shape, device=device) * (r_hi - r_lo) + r_lo,
                torch.rand(parent_outs.shape, device=device) * (i_hi - i_lo) + i_lo,
            ).type(parent_outs.dtype)
        else:
            lo = parent_outs.float().min().item()
            hi = parent_outs.float().max().item()
            finite_output = output_outs.detach().float().abs()
            finite_output = finite_output[torch.isfinite(finite_output)]
            output_scale = finite_output.max().item() if finite_output.numel() else 0.0
            dtype_info = torch.finfo(parent_outs.dtype)
            parent_scale = max(abs(lo), abs(hi), hi - lo, 1.0)
            scaled_expansion: float | None = None
            if hi - lo < max(1e-6, abs(lo) * 1e-6):
                # Near-constant tensor — range is too narrow for meaningful
                # perturbation at float32 precision. Expand by ±10% of the
                # parent magnitude, or by the child output scale when the parent
                # is near zero but the op output is huge. Without the output
                # scale, zero-valued broadcast parents can be perturbed by only
                # ~1 and then disappear under float32 rounding beside 1e38
                # operands.
                expansion = min(
                    max(1.0, abs(lo) * 0.1, output_scale * 0.1),
                    dtype_info.max * 0.25,
                )
                if output_scale > parent_scale * 1.0e6:
                    scaled_expansion = expansion
                else:
                    lo, hi = lo - expansion, hi + expansion
            elif output_scale > parent_scale * 1.0e6:
                # A non-constant but tiny parent range can also be invisible
                # when combined additively with enormous operands. Expand only
                # for extreme scale separation so normal range-restricted
                # tensors keep their original valid-domain perturbations.
                scaled_expansion = min(output_scale * 0.1, dtype_info.max * 0.25)
            if scaled_expansion is not None:
                signs = torch.where(
                    torch.rand_like(parent_outs.float(), device=device) < 0.5,
                    -1.0,
                    1.0,
                )
                magnitudes = (
                    torch.rand_like(parent_outs.float(), device=device) * 0.5 + 0.5
                ) * scaled_expansion
                perturbed_outs = parent_outs.float() + signs * magnitudes
            else:
                perturbed_outs = (
                    torch.rand_like(parent_outs.float(), device=device) * (hi - lo) + lo
                )
            perturbed_outs = perturbed_outs.type(parent_outs.dtype)

    return perturbed_outs
