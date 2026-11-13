"""Trace: the top-level container for a fully logged forward pass.

Trace is the root data structure returned by ``trace()``.
It owns every Op (per-operation entry), every Layer (per-layer
aggregate), the module hierarchy, parameter metadata, and graph-level
bookkeeping.

Key design patterns:

* **_tracing_finished behavioural switch** - Many custom_methods (``__len__``, ``__getitem__``,
  ``__str__``, ``__iter__``) behave differently during logging vs after
  postprocessing.  While logging is active (``_tracing_finished=False``), the
  model's tensors are keyed by their raw internal barcodes in
  transient raw graph state. After postprocessing flips ``_tracing_finished=True``,
  the friendly ``layer_list`` / ``layer_dict_all_keys`` / ``layer_logs``
  structures are populated and used instead.

* **Explicit Trace custom_methods** - Public custom_methods are defined directly on
  ``Trace``. Heavier implementations may delegate into subpackages
  through local imports, but users still call them as
  ``trace.draw(...)`` or ``trace.validate_forward_pass(...)``.

* **module build state** - A transient dict that accumulates module hierarchy
  information during the forward pass.  Consumed by ``_build_module_logs``
  (postprocessing step 16) and then cleared.  Initialised via
  ``_init_module_hierarchy_data()``.
"""

import pickle
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    TextIO,
    Union,
    cast,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from torchlens._runnable import ReadinessReport

    from .._io.streaming import BundleStreamWriter
    from .._output_slot import OutputSlot
    from .._runnable import ArchivedActivation, SparseRunDescriptor
    from ..debug._audit import HealthReport
    from ..intervention._types import FrozenInterventionSpec, TargetSpec
    from ..options import InterventionOptions, ReplayOptions
    from ._op_dedup import IdentityCacheEntry
    from .func_call_location import FuncCallLocation

from collections.abc import Collection, Iterable, Sequence
from functools import cached_property

from .. import _state
from .._captured_run import TraceLike
from .._constants import MODEL_LOG_FIELD_ORDER
from .._deprecations import MISSING, MissingType
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
)
from .._ir.workspaces import (
    ModuleCaptureWorkspace,
    RawGraphWorkspace,
    WrapperRuntimeWorkspace,
)
from .._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    DrawViewLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .._quantities import Bytes, Duration, Flops, Macs, as_duration
from .._runnable.seam import (
    RunnableTraceState,
    runnable_trace_state,
)
from .._save_budget import SaveBudget, SaveBudgetOption
from .._trace_state import TraceState
from .._utils.tensor_utils import SaveMode
from ..backends import BackendName
from ..intervention._types import (
    MODEL_LOG_FIELD_FORK_POLICY,
    InterventionSpec,
    Relationship,
)
from ..types import _ActivationPostfunc, _GradientPostfunc
from . import (
    _call_tree,
    _doors,
    _fork_lineage,
    _trace_annotations,
    _trace_aten,
    _trace_intervention_records,
    _trace_inventory,
    _trace_profile,
    _trace_rerun_state,
    _trace_state_io,
    _trace_stats_grad_fn_site,
    _trace_views,
)
from ._accessor_base import Accessor
from ._conditional_arms import ArmRef, ConditionalOutcome, arm_role_ops, derive_conditional_outcome
from ._draw_keywords import teach_unknown_draw_keywords
from ._nonfinite import nonfinite_coverage, nonfinite_op_labels
from ._preprocessing_provenance import status_of
from ._trace_accessors import (
    OrphanAccessor,
    TraceGradFnCallAccessor,
    TraceModuleCallAccessor,
    TraceOpAccessor,
)
from ._trace_docs import attach_trace_docs
from ._trace_field_spec import TRACE_PORTABLE_STATE_SPEC
from ._trace_fork import build_fork
from ._trace_grad_sites import armed_edits, armed_summary_line
from ._trace_groups import TraceAutograd, TraceCost, TraceGraph, TraceProvenance
from ._trace_groups_capture import TraceHealth, TraceSettings, TraceSource
from ._trace_intervention_records import _DoTransaction
from ._trace_missing_attrs import _raise_missing_trace_attribute
from ._trace_output import TraceOutputMixin
from ._trace_pickle_state import _initialize_output_metadata
from ._trace_pristine import derive_pristine
from ._trace_profile import ModelProfile
from ._trace_removal import batch_remove_log_entries, release_param_refs, remove_log_entry
from ._trace_source_introspection import inspect_source_attr
from ._trace_stack import StackedActivations, stack_traces
from ._trace_state_io import _MODEL_LOG_DEFAULT_FILL
from ._trace_views import _CallableList
from ._trace_viz_arguments import _forward_draw, _forward_summary
from ._transform_repr import _scrubbed_transform_repr
from .backward_pass import BackwardPass, BackwardPassAccessor
from .cleanup import (
    cleanup,
)
from .container import reconstruct_container, reconstruct_output
from .derived_grad import DerivedGradAccessor, IntermediateDerivedGradAccessor
from .field_policy import (
    build_record_field_policy_table,
    default_fill_state_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)
from .grad_fn import GradFn, GradFnAccessor
from .interface import (
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
)
from .layer import Layer
from .module import Module, ModuleAccessor
from .op import Op
from .param import ParamAccessor

if TYPE_CHECKING:
    import pandas as pd
    from PIL import Image

    from .._runnable import RunResult
    from .._trace_slice import TraceSlice
    from .._visualization.code_panel import CodePanelOption
    from ..intervention._types import FireRecord
    from ..receptive_field._types import (
        ReceptiveFieldDirection,
        ReceptiveFieldProfile,
        ReceptiveFieldStatus,
    )
    from ..report._explain import ExplainReport
    from ..report._profile import ProfileReport
    from ..report._trace_facts import _CallableDict
    from ..validation._status import ValidationStatus
    from ..viz._drawing import Drawing
    from .aten_op import AtenOp, _ModePausedInteriorGap
    from .buffer import BufferAccessor
    from .layer import LayerAccessor
    from .module import Module, ModuleCall


#: Sentinel: ``Trace.rerun`` keeps the transform recorded at capture time.
_USE_STORED_TRANSFORM = object()


@dataclass
class ResolvedPreprocessing:
    """Structured provenance for automatic input preprocessing.

    Attributes
    ----------
    source:
        Resolver source that selected the preprocessing transform.
    identifier:
        Model, weights, or default-policy identifier.
    verified:
        Whether the preprocessing came from model-specific metadata.
    config:
        Best-effort serializable preprocessing configuration.
    description:
        Human-readable one-line summary for trace summaries.
    """

    source: str
    identifier: str
    verified: bool
    config: dict[str, Any]
    description: str
    status = property(status_of)  # derived authority standing, never stored


@dataclass
class ResolvedPostprocessing:
    """Structured provenance for automatic output postprocessing.

    Attributes
    ----------
    source:
        Resolver source that selected the postprocessing transform.
    identifier:
        Model, weights, label bank, or default-policy identifier.
    verified:
        Whether the postprocessing came from model-specific metadata.
    config:
        Best-effort serializable postprocessing configuration.
    description:
        Human-readable one-line summary for trace summaries.
    style:
        Resolved output decoding style.
    selected_output_head:
        Selected output head name or path for multi-output models.
    label_source:
        Label source used for decoded outputs.
    label_source_version:
        Version or revision for the label source.
    confidence:
        Resolver confidence, when available.
    top_n_captured:
        Number of decoded rows captured per item.
    ambiguous:
        Whether detection found multiple plausible postprocessing choices.

    Examples
    --------
    >>> resolved = tl.postprocessing.ResolvedPostprocessing(
    ...     source="project",
    ...     identifier="my_labels",
    ...     verified=True,
    ...     config={"id2label": {0: "cat", 1: "dog"}},
    ...     description="project label bank",
    ...     style="classification",
    ... )
    >>> resolved.identifier, resolved.verified, resolved.style
    ('my_labels', True, 'classification')
    """

    source: str
    identifier: str
    verified: bool
    config: dict[str, Any]
    description: str
    style: str | None = None
    selected_output_head: str | None = None
    label_source: str | None = None
    label_source_version: str | None = None
    confidence: float | None = None
    top_n_captured: int | None = None
    ambiguous: bool = False


@dataclass
class ConditionalEvent:
    """Structured metadata for one conditional event in user source code."""

    # Declared so the scrubber walks this record field-by-field: ``source_file`` is the
    # ABSOLUTE forward-module path, and without a spec the relativizer never ran, leaking
    # host paths at every save level. ``_apply_conditional_source_policy`` scrubs
    # it; every other field is portable. ``ClassVar`` so ``@dataclass`` ignores it.
    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "id": FieldPolicy.KEEP,
        "kind": FieldPolicy.KEEP,
        "source_file": FieldPolicy.KEEP,
        "function_qualname": FieldPolicy.KEEP,
        "function_span": FieldPolicy.KEEP,
        "if_stmt_span": FieldPolicy.KEEP,
        "test_span": FieldPolicy.KEEP,
        "branch_ranges": FieldPolicy.KEEP,
        "branch_test_spans": FieldPolicy.KEEP,
        "call_depth": FieldPolicy.KEEP,
        "parent_conditional_id": FieldPolicy.KEEP,
        "parent_branch_kind": FieldPolicy.KEEP,
        "bool_layers": FieldPolicy.KEEP,
        # Runtime attributes stamped by phase-5c conditional attribution (not
        # declared dataclass fields). Retained verbatim, as they were before this
        # record gained a spec; only ``source_file`` is privacy-adjusted.
        "_bool_layers_raw": FieldPolicy.KEEP,
        "_arm_bool_indices": FieldPolicy.KEEP,
        "_arm_test_structures": FieldPolicy.KEEP,
    }

    id: int
    kind: Literal["if_chain", "ifexp"]
    source_file: str
    function_qualname: str
    function_span: tuple[int, int]
    if_stmt_span: tuple[int, int]
    test_span: tuple[int, int, int, int]
    branch_ranges: dict[str, tuple[int, int, int, int]]
    branch_test_spans: dict[str, tuple[int, int, int, int]]
    call_depth: int
    parent_conditional_id: int | None
    parent_branch_kind: str | None
    bool_layers: list[str] = field(default_factory=list)


@dataclass
class ConditionalRoleRef:
    """One op's participation in a conditional arm.

    Attributes
    ----------
    conditional_id:
        Stable id of the Conditional this op participates in.
    arm_index:
        Index of the arm within ``Conditional.arms``.
    arm_kind:
        Arm kind: ``"then"``, ``"elif"``, ``"else"`` or ``"fallthrough"``.
    role:
        Role within the arm: ``"evaluation"`` or ``"body"``.
    """

    conditional_id: str
    arm_index: int
    arm_kind: Literal["then", "elif", "else", "fallthrough"]
    role: Literal["evaluation", "body"]


@dataclass
class ConditionalArm:
    """One arm of an if-chain."""

    kind: Literal["then", "elif", "else", "fallthrough"]
    terminal_bool_op_label: str | None = None
    bool_value_at_run: bool | None = None
    condition_evaluated: bool = False
    evaluation_entry_edge: tuple[str, str] | None = None
    fired: bool = False
    execution_entry_edge: tuple[str, str] | None = None
    _trace: Any = field(default=ArmRef(), repr=False, compare=False)
    _conditional_id: str | None = field(default=None, repr=False, compare=False)
    _arm_index: int | None = field(default=None, repr=False, compare=False)

    @property
    def evaluation_ops(self) -> list[str]:
        """Return op labels that evaluate this arm's condition.

        Returns
        -------
        list[str]
            Labels for ops with an evaluation role in this arm.
        """

        return arm_role_ops(self._trace, self._conditional_id, self._arm_index, "evaluation")

    @property
    def execution_ops(self) -> list[str]:
        """Return op labels that execute this arm's body.

        Returns
        -------
        list[str]
            Labels for ops with a body role in this arm.
        """

        return arm_role_ops(self._trace, self._conditional_id, self._arm_index, "body")

    def _bind(self, trace: Any, conditional_id: str, arm_index: int) -> None:
        """Bind this arm to its Trace and owning conditional identity.

        Parameters
        ----------
        trace:
            Trace containing the role-bearing ops.
        conditional_id:
            Owning Conditional id.
        arm_index:
            Position of this arm within ``Conditional.arms``.
        """

        self._trace = trace
        self._conditional_id = conditional_id
        self._arm_index = arm_index


@dataclass
class Conditional:
    """One if-chain at one source location."""

    # Declared so the scrubber walks this record and applies the source-privacy
    # policy to ``source_file`` (the absolute forward-module path) instead of
    # persisting it verbatim. ``ClassVar`` so ``@dataclass`` ignores it.
    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "id": FieldPolicy.KEEP,
        "arms": FieldPolicy.KEEP,
        "fired_arm_index": FieldPolicy.KEEP,
        "fired_arm_kind": FieldPolicy.KEEP,
        "source_file": FieldPolicy.KEEP,
        "source_line": FieldPolicy.KEEP,
    }

    id: str
    arms: list[ConditionalArm]
    fired_arm_index: int | None
    fired_arm_kind: Literal["then", "elif", "else", "fallthrough"] | None
    source_file: str | None
    source_line: int | None

    @property
    def source_location(self) -> str | None:
        """Combined ``file:line`` location, if available."""

        if self.source_file is None or self.source_line is None:
            return None
        return f"{self.source_file}:{self.source_line}"

    @property
    def fired_arm(self) -> ConditionalArm | None:
        """Direct access to the fired arm, if any."""

        if self.fired_arm_index is None:
            return None
        if self.fired_arm_index < 0 or self.fired_arm_index >= len(self.arms):
            return None
        return self.arms[self.fired_arm_index]

    @property
    def outcome(self) -> ConditionalOutcome:
        """Which way this conditional went, derived from its arms; never persisted."""

        return derive_conditional_outcome(self.arms)

    @property
    def has_else(self) -> bool:
        """Whether this conditional has an else arm."""

        return any(arm.kind == "else" for arm in self.arms)

    @property
    def has_elif(self) -> bool:
        """Whether this conditional has one or more elif arms."""

        return any(arm.kind == "elif" for arm in self.arms)

    @property
    def num_arms(self) -> int:
        """Number of arms."""

        return len(self.arms)

    @property
    def num_elifs(self) -> int:
        """Number of elif arms."""

        return sum(arm.kind == "elif" for arm in self.arms)


class ConditionalAccessor:
    """Dict-like accessor for Conditional records."""

    # Declared so the scrubber descends into ``_list``/``_dict`` and reaches each
    # ``Conditional`` (whose ``source_file`` must be privacy-scrubbed) rather than
    # persisting the whole accessor subtree verbatim. ``_list`` and ``_dict``
    # share the same ``Conditional`` objects, so the scrub memo keeps them identical.
    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "_list": FieldPolicy.KEEP,
        "_dict": FieldPolicy.KEEP,
    }

    def __init__(self, conditionals: list[Conditional] | None = None) -> None:
        """Initialize from conditionals in trace order.

        Parameters
        ----------
        conditionals:
            Conditional records to expose.
        """

        self._list = list(conditionals or [])
        self._dict = {conditional.id: conditional for conditional in self._list}

    def __getitem__(self, key: int | str) -> Conditional:
        """Return a Conditional by ordinal or id."""

        if isinstance(key, int):
            return self._list[key]
        return self._dict[key]

    def __len__(self) -> int:
        """Return the number of conditionals."""

        return len(self._list)

    def __iter__(self) -> Iterator[Conditional]:
        """Iterate conditionals in trace order."""

        return iter(self._list)

    def keys(self) -> list[str]:
        """Return conditional ids."""

        return list(self._dict.keys())

    def values(self) -> list[Conditional]:
        """Return conditional records."""

        return list(self._list)

    def items(self) -> list[tuple[str, Conditional]]:
        """Return ``(id, Conditional)`` pairs."""

        return [(conditional.id, conditional) for conditional in self._list]


@dataclass(init=False, repr=False, eq=False)
class Trace(TraceOutputMixin, _trace_rerun_state.TraceAppendMixin, TraceLike):
    """Top-level container for a logged forward pass.

    Serves double duty: during the forward pass it accumulates raw tensor
    metadata in transient raw graph state; after postprocessing (``_tracing_finished=True``)
    it presents a clean, user-facing view via ``layer_list``, ``layer_dict_all_keys``,
    ``layer_logs``, ``modules``, ``params``, and ``buffers``.

    Supports ``len()``, iteration, and flexible ``__getitem__`` lookup by
    integer index, layer label, module address, or substring.

    Examples
    --------
    >>> trace = tl.trace(model, x)
    >>> trace.layer_labels
    ['input_1', 'conv2d_1_1', 'relu_1_2', 'conv2d_2_3', 'output_1']
    >>> tuple(trace["relu_1_2"].out.shape)
    (1, 4, 8, 8)
    """

    @property
    def readiness(self) -> "ReadinessReport | None":
        """Return non-executing sparse-run readiness for a loaded artifact.

        Returns
        -------
        ReadinessReport | None
            Structured load-time report, or ``None`` for a live Trace.
        """

        return cast("ReadinessReport | None", runnable_trace_state(self).readiness)

    @property
    def _sparse_run_descriptor(self) -> "SparseRunDescriptor | None":
        """Return the parsed sparse descriptor retained by a loaded artifact.

        Returns
        -------
        SparseRunDescriptor | None
            Parsed descriptor, or ``None`` for analysis-only/live traces and
            structurally unparseable runnable descriptors.
        """

        return cast("SparseRunDescriptor | None", runnable_trace_state(self).descriptor)

    @property
    def archived_activations(self) -> Mapping[str, "ArchivedActivation"]:
        """Return inspection-only activation payloads from a runnable archive.

        Returns
        -------
        Mapping[str, ArchivedActivation]
            Mapping keyed by ``"<slot_id>:<field>"``. Values are never read by
            the sparse scheduler or used as intermediate execution inputs.
        """

        return cast(
            Mapping[str, "ArchivedActivation"],
            runnable_trace_state(self).archived_activations or {},
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Strictly validate and atomically stage sparse-run state.

        Parameters
        ----------
        state_dict:
            Canonically named parameter and persistent-buffer tensor mapping.

        Raises
        ------
        StateBindingError
            If names, module paths, roles, shapes, dtypes, or aliases violate
            the recorded state-slot contract.

        Notes
        -----
        This method stages transient run state only. It does not execute the
        graph or write tensor values into the sparse descriptor.
        """

        from .._runnable.state import load_trace_state_dict

        load_trace_state_dict(self, state_dict)

    def find_nan(self) -> Any:
        """Return the first NaN or Inf among saved outputs in execution order.

        Selectively saved traces identify this as the first finding among
        saved tensors and report unsaved ancestor operations as an uncertainty
        zone rather than claiming the true birth operation.

        Returns
        -------
        FindNanReport
            Structured non-finite diagnostic. Source locations are surfaced
            from the operation's existing recorded code context when available.
        """
        return _trace_views.find_nan(self)

    def audit(self) -> "HealthReport":
        """Return an honest one-call health report for this completed trace.

        Payload-dependent and gradient-dependent diagnostics are listed as
        skipped when this capture cannot support them, rather than being used
        to claim a clean result.

        Returns
        -------
        torchlens.report.HealthReport
            Prioritized findings plus checks run and skipped-check reasons.
        """
        return _trace_annotations.audit(self)

    def using(self, name: str) -> Any:
        """Return the appliance registered as ``name``, bound to this trace.

        The one accessor door for appliances (DOCUMENTED-UNSTABLE); register with
        ``torchlens.appliances.register``. An unknown name refuses typed
        (``appliance_unknown``) and lists the registered names.
        """
        return _trace_annotations.using(self, name=name)

    def __getattr__(self, name: str) -> Any:
        """Explain common missing memory attributes before raising.

        Parameters
        ----------
        name:
            Missing attribute name.

        Returns
        -------
        Any
            This path never returns; the annotation preserves static typing
            for explicitly installed session fields.
        """

        return _raise_missing_trace_attribute(self, name)

    if TYPE_CHECKING:

        def __setattr__(self, name: str, value: Any) -> None:
            """Declare dynamically installed session fields to static tooling.

            Parameters
            ----------
            name:
                Session field name.
            value:
                Session field value.
            """

            ...

    @property
    def _capture_events(self) -> Any:
        """Return the retained raw capture event stream, if present.

        Returns
        -------
        Any
            Retained ``CaptureEvents`` instance.

        Raises
        ------
        AttributeError
            If no retained stream is installed.
        """

        if "_capture_events" not in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute '_capture_events'"
            )
        return self.__dict__["_capture_events"]

    @_capture_events.setter
    def _capture_events(self, value: Any) -> None:
        """Install the retained raw capture event stream.

        Parameters
        ----------
        value:
            ``CaptureEvents`` instance retained by this Trace.
        """

        self.__dict__["_capture_events"] = value

    @_capture_events.deleter
    def _capture_events(self) -> None:
        """Release the retained event stream and its working projection.

        Returns
        -------
        None
            The retained stream is removed and its working lanes are cleared.
        """

        from .._captured_run import forget_event_stream

        forget_event_stream(self)

    @property
    def _buffer_write_events(self) -> list[Any]:
        """Return capture-journal buffer writes through their read surface.

        Returns
        -------
        list[Any]
            Buffer-write events retained in the active or completed journal.
        """

        stream = self.__dict__.get("capture_events") or self.__dict__.get("_capture_events")
        return list(getattr(stream, "buffer_write_events", ()) or ())

    backend: BackendName
    backend_runtime_config: dict[str, Any] | None
    backend_runtime_device_summary: dict[str, Any] | None
    backend_runtime_version: str | None
    module_identity_mode: Literal["torch_module", "pytree_module", "function_root", "object_module"]
    param_source: Literal["native-module", "pytree-derived", "none"]
    state: TraceState
    tlspec_version: int
    annotations: dict[str, Any]
    input_preprocessor: ResolvedPreprocessing | None
    output_postprocessor: ResolvedPostprocessing | None
    output_id2label: dict[int, str] | None
    output_num_classes: int | None
    _output_slot: "OutputSlot | None"
    input_object_id: int | None
    model_object_id: int | None
    input_signature_hash: str | None
    state_history: list[Any]
    _replay_frontier: dict[str, torch.Tensor]
    backward_ready: bool
    inference_only: bool
    chunked_forward: bool
    profile_enabled: bool
    save_arg_templates: bool
    op_equivalence_classes: dict[str, set[str]]
    last_run: Any | None
    capture_start_time: float
    capture_end_time: float
    _runnable: RunnableTraceState
    _raw_graph_ws: RawGraphWorkspace
    _module_capture_ws: ModuleCaptureWorkspace
    _wrapper_runtime_ws: WrapperRuntimeWorkspace
    _fast_run_session: Any | None
    _distributed_plane_p: Any | None
    distributed_scope: str | None
    _primitive_op_profile: Any | None
    backward_root_grad_fn_object_ids: list[int]
    _backward_pass_logs: dict[int, BackwardPass]
    code_context: list["FuncCallLocation"]
    jax_closed_jaxpr: Any
    jax_equation_captures: tuple[Any, ...]
    jax_outvar_key_to_capture_index: dict[str, int]
    _jax_capture_index_to_raw_op_label: dict[int, str]
    jax_capture_index_to_final_op_label: dict[int, str]
    jax_inlined_call_primitives: tuple[str, ...]
    jax_static_argnums: tuple[int, ...]
    input_structure: Any
    _containers: dict[int, Any]
    _annotation_blobs: dict[str, Any] | None
    _last_sibling_ordering_decision: Any
    _last_encoding_state: Any
    _last_render_geometry: Any
    _module_call_accessor: Any
    _op_accessor_cache: Any
    _layer_accessor_cache: Any
    _receptive_field_solution: Any
    _rf_directional_solutions: Any
    _tl_rf_probe_active: Any
    _cooked_from: str | None = None  # "recording" on a Trace that Recording.to_trace cooked

    _PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = TRACE_PORTABLE_STATE_SPEC
    _FIELD_POLICY = build_record_field_policy_table(
        MODEL_LOG_FIELD_ORDER,
        _PORTABLE_STATE_SPEC,
        fork_policy=MODEL_LOG_FIELD_FORK_POLICY,
        default_fill_state=_MODEL_LOG_DEFAULT_FILL,
        schema_key="trace",
    )
    _PORTABLE_STATE_SPEC = portable_state_spec_from_policy(_FIELD_POLICY)

    def __init__(
        self,
        model_class_name: str,
        output_device: str = "same",
        activation_transform: _ActivationPostfunc | None = None,
        grad_transform: _GradientPostfunc | None = None,
        save_raw_activations: bool = True,
        save_raw_grads: bool = True,
        save_mode: SaveMode = "copy",
        keep_orphans: bool = False,
        save_args: bool = False,
        save_grads: Any = None,
        capture_tensor_grad_hooks: bool = True,
        detach_saved_activations: bool = False,
        mark_layer_depths: bool = True,
        source_context_lines: int = 7,
        optimizer: torch.optim.Optimizer | None = None,
        save_source_context: bool = False,
        save_rng: bool = False,
        recurrence_detection: bool = True,
        verbose: bool = False,
        backward_ready: bool = False,
        inference_only: bool = False,
        chunked_forward: bool = False,
        emit_nvtx: bool = False,
        measure_python_peak_memory: bool = False,
        distributed_witness: str = "none",
        save_budget: SaveBudgetOption = "auto",
        facet_functions: Any | None = None,
        input_transform: Callable[[Any], Any] | None = None,
        input: Any | None = None,
        save_raw_input: str | bool = "small",
        batch_render: str = "auto",
        output_transform: Callable[[Any], Any] | None = None,
        transformed_output: Any | None = None,
        save_transformed_output: str | bool = "small",
        layer_visualizers: Mapping[Any, Callable[..., Any]] | None = None,
        save_visualizations: bool = False,
    ) -> None:
        """Initialise a fresh Trace for a new logging session.

        Args:
            model_class_name: Human-readable name of the model being logged.
            output_device: Device to move saved outs to ("same" keeps original device).
            activation_transform: Optional function applied to each tensor before saving.
            grad_transform: Optional function applied to each grad before saving.
            save_raw_activations: Whether raw outs are retained when a transform is set.
            save_raw_grads: Whether raw grads are retained when a transform is set.
            save_mode: Tensor retention mode for saved activation and gradient payloads.
            keep_orphans: If True, orphan island ops remain in raw metadata and
                are exposed via ``trace.graph.orphans``.
            save_args: Whether to deep-copy each operation's input arguments.
            save_grads: Which backward gradients should be retained. ``True``
                saves all gradients, ``False``/``None`` saves no payloads, and
                selectors/predicates save matching gradient records.
            capture_tensor_grad_hooks: Whether forward tensors receive
                tensor-level backward hooks for implicit backward events and
                per-op gradient payloads. Grad-fn registration remains enabled.
            detach_saved_activations: Whether to detach saved tensors from the autograd graph.
            mark_layer_depths: Whether to compute BFS distances from
                inputs/outputs for each layer.
            source_context_lines: Number of source-code context lines to capture
                around each function call (used by FuncCallLocation).
            optimizer: Optional torch optimizer, used to annotate which params
                have optimizers attached.
            verbose: If True, print timed progress messages at each major pipeline stage.
            backward_ready: Session-time flag for training-compatible out retention.
                Portable bundle load restores the default ``False`` value.
            inference_only: Whether the forward was captured under ``torch.no_grad()``,
                requested explicitly or implied by a capture entered with autograd off.
            chunked_forward: Whether the trace was assembled from forward chunks.
            emit_nvtx: Whether decorated torch operations should emit NVTX ranges
                around captured torch calls. This is a profiling aid for CUDA/Nsight
                workflows and does not change graph construction or saved payloads.
            measure_python_peak_memory: Session-time flag selecting whether the
                CPU/MPS ``forward_peak_memory`` measurement also runs a
                ``tracemalloc`` Python-allocation probe. Off by default because the
                allocator hook taxes every traced operation. Portable bundle load
                restores the default ``False`` value.
            distributed_witness: Session-time witness level for collective
                boundary records ("none" or "digest"; "payload" reserved for
                the merge artifact story). Portable bundle load restores the
                default "none".
            save_budget: Session-time per-device ceiling on retained activation
                bytes. ``"auto"`` allows half of each device's available memory;
                a float sets another fraction, an int an absolute byte cap, and
                ``None`` disables the guard. Crossing it raises
                ``SaveBudgetExceededError`` mid-capture. Portable bundle load
                restores the default ``"auto"``.
            facet_functions: Immutable facet recipe snapshot captured for
                this trace.
            input_transform: Optional callable used to convert raw user input into
                model-ready input.
            input: Original user input before ``input_transform`` was applied.
            save_raw_input: Portable save policy for ``input``.
            batch_render: Raw-input batch rendering policy for visualization.
            output_transform: Optional callable used to convert model output into
                human-readable metadata.
            transformed_output: Human-readable model output after ``output_transform``.
            save_transformed_output: Portable save policy for ``transformed_output``.
            layer_visualizers: Optional mapping of selectors to visualizer callables.
            save_visualizations: Whether rendered visualizations should persist in bundles.
        """
        # General info. Callables are effectively immutable - deepcopy is unnecessary.
        self.name: str | None = None
        self._source_class_name = model_class_name
        self.model_name = model_class_name
        self._provenance_backend: BackendName = "torch"
        self._provenance_backend_config: dict[str, Any] | None = None
        self.backend_runtime_device_summary: dict[str, Any] | None = None
        self._provenance_backend_version: str | None = None
        self._options_module_identity_mode: Literal[
            "torch_module", "pytree_module", "function_root", "object_module"
        ] = "torch_module"
        self._provenance_param_source: Literal["native-module", "pytree-derived", "none"] = (
            "native-module"
        )
        self._autograd_derived_grads = DerivedGradAccessor()
        self.num_context_lines = source_context_lines
        self._optimizer = optimizer
        self.tlspec_version = TLSPEC_VERSION
        # _tracing_finished is the master behavioural switch: False during logging,
        # True after postprocessing.  Many custom_methods (len, getitem, str, iter)
        # branch on this flag to choose raw-barcode vs final-label access.
        self._tracing_finished = False
        self._raw_graph_ws = RawGraphWorkspace()
        self._module_capture_ws = ModuleCaptureWorkspace()
        self._wrapper_runtime_ws = WrapperRuntimeWorkspace()
        self._primitive_op_profile = None
        from .._ir.workspaces import _init_module_hierarchy_data

        self._module_capture_ws.module_build_data = _init_module_hierarchy_data()
        self._options_mode: Literal["exhaustive", "predicate"] = "exhaustive"
        # L7a: a structure-only capture; value consumers gate through
        # torchlens._capture.structure_only. DOCUMENTED-UNSTABLE spelling.
        self._options_structure_only: bool = False
        # tlspec v9 entry-dark slots: the structure-only evidence
        # envelope and the (path, digest) source-snapshot table.
        # Declared, persisted, load-validated; no writer yet.
        self._graph_structure_evidence: dict[str, Any] | None = None
        self._source_snapshots: list[dict[str, Any]] = []
        # L6: session-time audit rows of resolved-selection interventions (query repr,
        # resolve digest, per-site relations). DROP under v7. DOCUMENTED-UNSTABLE.
        self.intervention_audit: list[dict[str, Any]] = []
        self._runnable = RunnableTraceState()
        self._fast_run_session: Any | None = None
        # Merge-ranks plane-P: session-time physical dispatch observation
        # journal for armed captures (census criteria 2-4 evidence). DROP
        # under its private name; never survives save/load. DOCUMENTED-UNSTABLE.
        self._distributed_plane_p: Any | None = None
        self.is_stopped = False
        self.stop_reason: str | None = None
        self._graph_stop_frontier: str | None = None
        self._layers_logged = False
        self._layers_saved = False
        self._options_keep_orphans = keep_orphans
        self.is_replayable = False
        self._options_save_arg_templates = False
        self.input = input
        self._provenance_input_preprocessor: ResolvedPreprocessing | None = None
        self._transform = input_transform
        self._options_input_transform = (
            repr(input_transform) if input_transform is not None else None
        )
        self._options_save_raw_input = save_raw_input
        self.batch_render = batch_render
        self.transformed_output = transformed_output
        self._decoded_output: Any | None = None
        self._provenance_output_postprocessor: ResolvedPostprocessing | None = None
        _initialize_output_metadata(self)
        self._output_transform = output_transform
        self._output_slot = None
        self._options_save_transformed_output = save_transformed_output
        self.layer_visualizers = layer_visualizers
        self.save_visualizations = save_visualizations
        self._visualizer_dir: str | None = None
        self._options_activation_transform = activation_transform
        self._activation_transform_repr = _scrubbed_transform_repr(activation_transform)
        self._options_save_raw_activations = save_raw_activations
        self._provenance_input_annotations: dict[str, Any] = {}
        self._options_grad_transform = grad_transform
        self._options_save_raw_grads = save_raw_grads
        self._options_save_mode = save_mode
        self._source_code_blob: dict[str, str] = {}
        self._source_model_ref: weakref.ReferenceType[nn.Module] | None = None
        self._parent_run: weakref.ReferenceType[Trace] | None = None
        self._provenance_model_object_id: int | None = None
        self._source_class_qualname: str | None = None
        self._provenance_param_hash_quick: str | None = None
        self._provenance_root_entry_point: str | None = None
        self._provenance_param_hash_full: str | None = None
        self._provenance_input_object_id: int | None = None
        self._provenance_input_signature_hash: str | None = None
        self._options_seed: int | None = None
        self._options_output_device = output_device
        self._options_detach_saved_activations = detach_saved_activations
        self._options_backward_ready = backward_ready
        self._options_inference_only = inference_only
        self._options_chunk_size = chunked_forward
        self._options_emit_nvtx = emit_nvtx
        self.measure_python_peak_memory = measure_python_peak_memory
        self.distributed_witness = distributed_witness
        self.save_budget = save_budget
        # Built once per capture; ``None`` when budgeting is disabled; charged on the save hot path.
        self._save_budget_accountant = SaveBudget.from_option(save_budget)
        self.facet_functions = facet_functions
        self._options_on_nonfinite: bool = False
        self.track_nonfinite: bool = False
        self.track_device_memory: bool = False
        self.measure_host_peak_memory: bool = False
        self._options_aten_ops: bool | str = False
        self.annotations: dict[str, Any] = {}
        self._source_frames: list[FuncCallLocation] = []
        self._graph_manual_tensor_connections: list[tuple[str, str]] = []
        self._source_forward_file: str | None = None
        self._source_forward_line: int | None = None
        self._source_class_file: str | None = None
        self._source_class_line: int | None = None
        self._source_init_file: str | None = None
        self._source_init_line: int | None = None
        self._source_class_docstring: str | None = None
        self._source_init_signature: str | None = None
        self._source_init_docstring: str | None = None
        self._source_forward_signature: str | None = None
        self._source_forward_docstring: str | None = None
        self._provenance_cache_hit: bool = False
        self._provenance_cache_key: str | None = None
        self._provenance_cache_path: str | None = None
        self._provenance_recording_kept: bool = True
        self._out_dedup_mode: Literal["identity", "content", "none"] = "identity"
        self._out_identity_cache: dict[int, IdentityCacheEntry] = {}
        self._out_hash_cache: dict[str, tuple[str, torch.Tensor]] = {}
        self._code_context_cache: dict[Any, tuple[Any, ...]] = {}
        self._halt_returns_partial_trace = False
        self._replay_arg_version_data_complete = True
        self._options_save_args = save_args
        self._options_save_grads = "all" if save_grads is True else save_grads
        self.capture_tensor_grad_hooks = capture_tensor_grad_hooks
        self._options_save_source_context = save_source_context
        self._options_save_rng = save_rng
        self._options_recurrence_detection = recurrence_detection
        # L1 grouping surface: "structural" is the only entry-legal knob
        # value in wave 0 (others refuse typed at trace entry); the stamp is
        # written by each producer once step-7 grouping settles.
        self._options_grouping = "structural"
        self._options_grouping_policy: dict[str, Any] | None = None
        # Set to "rank_local_shard" by capture when any param/input
        # carries a non-Replicate DTensor placement (the sharded predicate in
        # torchlens.distributed._dtensor). Unreachable until the D-L8-CAP
        # capture relaxation: every sharded capture still refuses at entry.
        self._provenance_distributed_scope: str | None = None
        # L9 backward residuals: per-fire timing clock provenance
        # ("unmeasured" until a timing prehook arms) and the checkpoint-
        # invocation witness summary (None until torch capture builds it).
        # Both DROP-gated pre-bump; spellings DOCUMENTED-UNSTABLE.
        self.grad_fn_timing_provenance: str = "unmeasured"
        self.checkpoint_invocation_witness: dict[str, Any] | None = None
        self._options_verbose = verbose
        self.profile_enabled = False
        self._autograd_has_grads = False
        self._options_mark_layer_depths = mark_layer_depths
        self._provenance_fingerprint: str | None = None
        self._intervention_spec: InterventionSpec | None = InterventionSpec()
        self.state_history: list[Any] = []
        self._provenance_observer_spans: list[dict[str, Any]] = list(
            _state._active_record_spans.get()
        )
        self.last_run: Any | None = None
        self._append_history: list[dict[str, Any]] = []
        self._has_direct_writes = False
        self._warned_direct_write = False
        self._warned_mutate_in_place = False
        self._warned_once: set[str] = set()
        self._raw_transform_escape_detected = False
        self._raw_dynamo_region_detected = False
        self._raw_call_fingerprint: tuple[int, int] | None = None
        self._spec_revision = 0
        self._out_recipe_revision = 0
        self._append_sequence_id = 0
        self._last_hook_handle_ids: tuple[str, ...] = ()
        self.state = TraceState.PRISTINE
        self._provenance_is_appended = False
        self._graph_relationship_evidence: dict[str, Relationship] = {
            "model": Relationship.UNKNOWN,
            "weights": Relationship.UNKNOWN,
            "input": Relationship.UNKNOWN,
            "graph": Relationship.UNKNOWN,
        }
        self._replay_frontier: dict[str, torch.Tensor] = {}
        self._output_container_specs_by_raw_label: dict[str, Any] = {}
        self._out_writer: BundleStreamWriter | None = None
        self._keep_outs_in_memory: bool = True
        self._defer_streaming_bundle_finalization: bool = False
        self._out_sink: Callable[[str, torch.Tensor], None] | None = None
        # Model structure info (computed @properties: is_recurrent,
        # max_layer_op_count, is_branching, has_conditional_branching)

        # Tensor Tracking - post-processed (populated after _tracing_finished=True):
        self._layer_list: list[Op] = []  # ordered list of all layer ops
        self._layer_dict_main_keys: dict[str, Op] = OrderedDict()  # primary label -> entry
        self._layer_dict_all_keys: dict[str, Op] = OrderedDict()  # all lookup keys -> entry
        self._layer_logs: dict[str, Layer] = OrderedDict()  # no-pass label -> aggregate Layer
        self.op_labels: list[str] = []  # pass-qualified labels (e.g. "conv2d_1_1:1")
        self.layer_labels: list[str] = []  # pass-stripped labels (e.g. "conv2d_1_1")
        self._layer_num_calls: dict[str, int] = OrderedDict()  # no-pass label -> pass count
        self._by_pass: dict[int, list[int]] = {}
        self._layer_nums_to_save: list[int] = []  # ordinal positions of layers to save
        self._grad_op_nums_to_save: list[int] | str = []
        self._num_ops: int = 0  # total operations after postprocessing

        # Mapping between raw barcodes and final human-readable labels
        # (populated during postprocessing's label-assignment step):
        self._raw_to_final_layer_labels: dict[str, str] = {}
        self._raw_to_final_parent_layer_labels: dict[str, str] = {}
        self._raw_to_final_op_labels: dict[str, str] = {}
        self._final_to_raw_layer_labels: dict[str, str] = {}
        self._lookup_keys_to_layer_num_dict: dict[str, int] = {}
        self._layer_num_to_lookup_keys_dict: dict[int, list[str]] = defaultdict(list)
        self._ambiguous_lookup_keys: dict[str, list[int]] = {}

        # Special Layers:
        self.input_layers: list[str] = []
        self.output_layers: list[str] = []
        self._annotation_blobs: dict[str, Any] | None = None
        self.buffer_layers: list[str] = []
        self._buffer_num_calls: dict[str, int] = {}
        self._buffer_accessor = None
        self._buffer_write_tracker: Any | None = None
        self._buffer_initial_values: dict[str, Any] = {}
        self.internal_source_ops: list[str] = []
        self.internal_sink_ops: list[str] = []
        self.internally_terminated_bool_ops: list[str] = []
        self._graph_conditional_branch_edges: list[tuple[str, str]] = []
        self._graph_conditional_records: list[ConditionalEvent] = []
        self._graph_conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]] = {}
        self._graph_conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = {}
        self._graph_conditionals = ConditionalAccessor()
        self._orphan_labels: list[str] = []
        self._orphan_logs: tuple[Op, ...] = ()
        self._orphan_records: list[dict[str, Any]] = []
        self._saved_grad_labels: set[str] = set()
        self.layers_with_params: dict[str, list[Any]] = defaultdict(list)
        # Maps equivalence_class -> set of layer labels that share
        # that equivalence type (populated by loop_detection.py).
        self._graph_op_equivalence_classes: dict[str, set[str]] = defaultdict(set)

        # Aggregate tensor statistics (computed during postprocessing):
        self._cost_total_activation_memory: Bytes = Bytes(0)
        self._cost_total_grad_memory: Bytes = Bytes(0)
        self._cost_total_backward_memory: Bytes = Bytes(0)
        self._cost_total_autograd_memory: Bytes | None = None
        self._cost_pinned_autograd_memory: Bytes | None = None  # pinned-autograd disclosure
        self._num_saved_ops: int = 0  # layers with has_saved_activation=True
        self._cost_saved_activation_memory: Bytes = Bytes(0)
        self._cost_saved_grad_memory: Bytes = Bytes(0)

        self._init_param_time_and_backward_state()
        _state.register_log(self)

    def _init_param_time_and_backward_state(self) -> None:
        """Initialise the param, module, timing and backward-record fields of a fresh Trace."""

        # Param info:
        self._param_logs: ParamAccessor = ParamAccessor({})
        self._num_param_tensors: int = 0
        self._num_layers_with_params: int = 0
        self.num_params: int = 0
        self.num_params_trainable: int = 0
        self.num_params_frozen: int = 0
        self._cost_total_param_memory: Bytes = Bytes(0)
        self._cost_total_param_grad_memory: Bytes = Bytes(0)
        # a CPU forward delta of 0 is recorded as ``None`` with
        # ``forward_peak_memory_reason`` naming measure_python_peak_memory=True.
        self._cost_forward_peak_memory: Bytes | None = Bytes(0)
        self._provenance_forward_memory_backend: str = "unknown"
        self._cost_forward_peak_memory_reason: str | None = None
        self._forward_peak_memory_pair: dict[str, Any] | None = None

        # Structured module info:
        self._module_logs: ModuleAccessor = ModuleAccessor({})

        # Time elapsed:
        self._provenance_start_time: float = 0
        self._provenance_end_time: float = 0
        self._phase_timings: dict[str, dict[str, float | int]] = {}
        self._provenance_setup_duration: Duration = Duration(0)
        self._cost_forward_duration: Duration = Duration(0)
        self._provenance_cleanup_duration: Duration = Duration(0)
        self._cost_func_calls_duration: Duration = Duration(0)
        self._autograd_has_backward_pass: bool = False
        self._grad_fn_logs: dict[int, GradFn] = OrderedDict()
        self._autograd_grad_fn_order: list[int] = []
        self._backward_pass_logs: dict[int, BackwardPass] = OrderedDict()
        self._grad_fn_param_refs: dict[str, str] = {}
        self._param_log_by_pid: dict[int, str] = {}
        # The RECORDED prep inventories of stamped parameters/buffers, the source
        # for stamp cleanup; emptied by ``_cleanup_model_session``. Never portable.
        self._session_param_inventory: list[Any] = []
        self._session_buffer_inventory: list[Any] = []
        # Id(tensor) -> _SessionBufferStamp for every buffer stamp written this
        # session (storage-identity belt); emptied by ``_cleanup_model_session``.
        self._session_buffer_identity: dict[int, Any] = {}
        self._autograd_backward_root_grad_fn_object_ids: list[int] = []
        self._cost_backward_durations: list[Duration] = []
        self._num_backward_passes: int = 0
        self._cost_backward_peak_memory: Bytes | None = Bytes(0)  # CPU None: see the reason
        self._provenance_backward_memory_backend: str = "unknown"
        self.backward_peak_memory_reason: str | None = None

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************

    def __len__(self) -> int:
        """Number of layer-pass entries. Uses final list after postprocessing, raw dict during logging."""
        if self._tracing_finished:
            return len(self._layer_list)
        else:
            return len(self._raw_graph_ws.raw_layer_dict)

    def __getitem__(self, ix: Any) -> Any:
        """Return one record, or the Selection a selector names.

        After the pass an int, slice or string addresses records; a set-naming
        selector (``tl.func``, ``tl.where``, ``tl.in_module``, compositions)
        returns a ``tl.Selection`` of every match; a one-thing spelling must match
        exactly one. A key gives the kind it names: a layer label its Op (a bare
        multi-pass label refuses; ``label:N``, ``:last``, ``:all``), never a
        ``Layer`` (that door is ``trace.layers[...]``); a module address its
        Module, ``address:N`` its ModuleCall, a parameter, buffer or grad_fn key
        its record. Typed doors convert: ``trace.ops[address]`` is the output Op.

        Args:
            ix: index, slice, label, address, substring or selector.

        Returns:
            Op, Selection, Module, ModuleCall, Param, Buffer or GradFn.
        """
        if self._tracing_finished:
            return _getitem_after_pass(self, ix)
        else:
            return _getitem_during_pass(self, ix)

    def find_sites(self, selector: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Return a table of intervention sites matching a query.

        Parameters
        ----------
        selector:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        _SiteTable
            Ordered table of matching layer-pass records.
        """
        return _trace_views.find_sites(
            self, selector=selector, strict=strict, max_fanout=max_fanout
        )

    def _resolve_sites(self, selector: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Resolve the record-grain site table for internal planning (private).

        The public door is ``trace[selector]``: a set-naming selector returns a
        ``tl.Selection`` of every match, never capped; a one-thing spelling
        returns its one record.

        Parameters
        ----------
        selector:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        _SiteTable
            Ordered table of matching layer-pass records.
        """
        return _trace_views.resolve_site_table(
            self, selector=selector, strict=strict, max_fanout=max_fanout
        )

    def check_plan(self, plan: Any) -> Any:
        """Audit-only plan check (nnsight-scan parity).

        Resolves each plan entry's selector against this trace, checks
        multiplicity, and compares declared replacement geometry (a meta
        tensor or any shape/dtype-bearing spec) against the recorded
        shape/dtype hypotheses. The report carries the source comparison
        digest, the evidence status, every checked site, and
        ``executable=False`` — checking a plan NEVER sets
        ``replayable``, arms, replays, or lifts the late-bind
        refusal. Refuses typed on callables, value-derived selection, and
        REFUTED sources. Spelling DOCUMENTED-UNSTABLE (the public verb name
        belongs to the UI sprint).

        Parameters
        ----------
        plan:
            Iterable of selector entries or ``(selector, replacement)``
            pairs.

        Returns
        -------
        torchlens._capture._plan_check.PlanCheckReport
            Frozen audit report.
        """
        return _trace_views.check_plan(self, plan=plan)

    def annotate(
        self,
        selector: Any,
        *,
        data: Any = None,
        image: "str | Path | Image.Image | None" = None,
        max_fanout: int = 1_000_000,
        copy: bool = False,
    ) -> "Trace":
        """Attach user-owned annotation data to selected graph nodes.

        Parameters
        ----------
        selector:
            Selector, target spec, frozen target spec, or non-strict bare string.
        data:
            JSON-serializable metadata or a portable tensor payload. Tensor
            payloads are persisted under ``_annotation_blobs`` and are supported
            only for torch traces.
        image:
            Optional picture for the selected nodes: an image file path or a PIL image.
        max_fanout:
            Explicit maximum number of selected sites. The default is
            intentionally large so annotation can fan out across many layers.
        copy:
            If ``True``, annotate and return an owned fork instead of mutating
            this trace.

        Returns
        -------
        Trace
            The annotated trace. This is ``self`` unless ``copy=True``.
        """
        return _trace_annotations.annotate(
            self, selector=selector, data=data, image=image, max_fanout=max_fanout, copy=copy
        )

    def _validate_annotation_tensor(self, tensor: torch.Tensor) -> None:
        """Validate that a tensor annotation matches the active payload codec.

        Parameters
        ----------
        tensor:
            Tensor annotation candidate.

        Returns
        -------
        None
            Raises if the tensor is not portable for this trace.
        """
        return _trace_annotations.validate_annotation_tensor(self, tensor=tensor)

    #: Key namespaces of DERIVED render-time annotation blobs (computed from a
    #: specific run's activations). Bare ``layer:``/``op:`` keys are USER
    #: ``annotate(data=...)`` blobs. Rerun refreshes drop the derived
    #: namespaces (stale against the new activations) and keep user blobs.
    _DERIVED_ANNOTATION_BLOB_PREFIXES: ClassVar[tuple[str, ...]] = (
        "mds:",
        "rdm:",
        "scree:",
        "featmap:",
    )

    def _user_annotation_blobs(self) -> dict[str, Any] | None:
        """Return only user-attached annotation blobs.

        Returns
        -------
        dict[str, Any] | None
            Blobs outside every derived render namespace, or ``None`` when no
            user blob exists (matching the unannotated default).
        """
        return _trace_annotations.user_annotation_blobs(self)

    def _mark_annotations_mutated(self) -> None:
        """Invalidate render-only caches after an annotation mutation.

        Returns
        -------
        None
            The cached sibling-ordering decision is discarded.
        """
        return _trace_annotations.mark_annotations_mutated(self)

    @property
    def uncalled_modules(self) -> _CallableList:
        """Return registered modules that were not exercised in the captured pass.

        Returns
        -------
        _CallableList
            Module addresses present on the source model but absent from the
            captured module accessor, recorded when the capture finished.
            Returns an empty list when that record is missing (a loaded or
            unpickled trace) and the source model is no longer available;
            ``num_uncalled_modules`` reads ``None`` in that case.
        """
        return _trace_views.uncalled_modules(self)

    @property
    def outcome(self) -> Any | None:
        """Return the settled capture outcome for this trace, when one exists.

        Returns
        -------
        CaptureOutcome | None
            The settlement authority's typed record: an attested settle stamp
            for live products, an adopted or lattice-derived record for loaded
            artifacts (load derivation writes the sidecar), or ``None`` on a
            live trace that has not settled yet (the documented pre-settlement
            state; capability gates independently treat it as UNKNOWN).
        """
        return _trace_annotations.outcome(self)

    @property
    def _source_cls(self) -> type[Any] | None:
        """Return the live source model class when the model is still alive.

        Returns
        -------
        type[Any] | None
            Runtime class of the source model, or ``None`` after the weakref dies.
        """

        source_ref = None if self.__dict__.get("_tl_cleaned_up", False) else self._source_model_ref
        model = source_ref() if source_ref is not None else None
        return None if model is None else type(model)

    @property
    def parent_trace(self) -> "Trace | None":
        """Return the parent Trace in a fork/rerun lineage, if any.

        Returns
        -------
        Trace | None
            Parent Trace resolved from the legacy ``parent_run`` weakref, or
            ``None`` for root traces and deserialized traces.
        """

        parent_ref = None if self.__dict__.get("_tl_cleaned_up", False) else self._parent_run
        if isinstance(parent_ref, weakref.ReferenceType):
            parent = parent_ref()
            return parent if isinstance(parent, Trace) else None
        return None

    @property
    def root_trace(self) -> "Trace | None":
        """Return the ultimate root Trace in this fork/rerun lineage.

        Returns
        -------
        Trace | None
            The oldest reachable Trace ancestor, or ``None`` when this Trace
            has no parent.
        """

        root = self.parent_trace
        while root is not None and root.parent_trace is not None:
            root = root.parent_trace
        return root

    @property
    def _options_save(self) -> str | list[str]:
        """Return the public layer-save selection represented by this Trace.

        Returns
        -------
        str | list[str]
            ``"all"`` when all layers were requested, otherwise saved
            pass-qualified Op labels in execution order.
        """

        layer_nums = getattr(self, "_layer_nums_to_save", [])
        if layer_nums == "all":
            return "all"
        selected_nums = set(layer_nums)
        return [op.label for op in self._layer_list if op.raw_index in selected_nums]

    @property
    def _source_class_docstring(self) -> str | None:
        """Return the source model class docstring."""

        return self.__dict__.get("_source_class_docstring") or inspect_source_attr(
            self, "_source_class_docstring"
        )

    @_source_class_docstring.setter
    def _source_class_docstring(self, value: str | None) -> None:
        """Store the source model class docstring."""

        self.__dict__["_source_class_docstring"] = value

    @_source_class_docstring.deleter
    def _source_class_docstring(self) -> None:
        """Delete the stored source model class docstring."""

        self.__dict__.pop("_source_class_docstring", None)

    @property
    def _source_init_signature(self) -> str | None:
        """Return the source model ``__init__`` signature."""

        return self.__dict__.get("_source_init_signature") or inspect_source_attr(
            self, "_source_init_signature"
        )

    @_source_init_signature.setter
    def _source_init_signature(self, value: str | None) -> None:
        """Store the source model ``__init__`` signature."""

        self.__dict__["_source_init_signature"] = value

    @_source_init_signature.deleter
    def _source_init_signature(self) -> None:
        """Delete the stored source model ``__init__`` signature."""

        self.__dict__.pop("_source_init_signature", None)

    @property
    def _source_init_docstring(self) -> str | None:
        """Return the source model ``__init__`` docstring."""

        return self.__dict__.get("_source_init_docstring") or inspect_source_attr(
            self, "_source_init_docstring"
        )

    @_source_init_docstring.setter
    def _source_init_docstring(self, value: str | None) -> None:
        """Store the source model ``__init__`` docstring."""

        self.__dict__["_source_init_docstring"] = value

    @_source_init_docstring.deleter
    def _source_init_docstring(self) -> None:
        """Delete the stored source model ``__init__`` docstring."""

        self.__dict__.pop("_source_init_docstring", None)

    @property
    def _source_forward_signature(self) -> str | None:
        """Return the source model ``forward`` signature."""

        return self.__dict__.get("_source_forward_signature") or inspect_source_attr(
            self, "_source_forward_signature"
        )

    @_source_forward_signature.setter
    def _source_forward_signature(self, value: str | None) -> None:
        """Store the source model ``forward`` signature."""

        self.__dict__["_source_forward_signature"] = value

    @_source_forward_signature.deleter
    def _source_forward_signature(self) -> None:
        """Delete the stored source model ``forward`` signature."""

        self.__dict__.pop("_source_forward_signature", None)

    @property
    def _source_forward_docstring(self) -> str | None:
        """Return the source model ``forward`` docstring."""

        return self.__dict__.get("_source_forward_docstring") or inspect_source_attr(
            self, "_source_forward_docstring"
        )

    @_source_forward_docstring.setter
    def _source_forward_docstring(self, value: str | None) -> None:
        """Store the source model ``forward`` docstring."""

        self.__dict__["_source_forward_docstring"] = value

    @_source_forward_docstring.deleter
    def _source_forward_docstring(self) -> None:
        """Delete the stored source model ``forward`` docstring."""

        self.__dict__.pop("_source_forward_docstring", None)

    def __str__(self) -> str:
        """Human-readable summary; delegates to post-pass or mid-pass formatter."""
        if self._tracing_finished:
            return _str_after_pass(self)
        else:
            return _str_during_pass(self)

    def _repr_card_rows(self) -> list[tuple[str, str]]:
        """Stored summary facts shared with the notebook renderer."""
        return _doors.report()._trace_summary._trace_rows(self)

    def __repr__(self) -> str:
        """One-line identity and stored health; never triggers an analysis."""

        return _doors.report()._trace_summary._trace_repr(self)

    def _repr_honesty_suffix(self) -> str:
        """Outcome + health suffix tokens for the identity repr."""

        return _doors.report()._trace_summary._trace_honesty_suffix(self)

    def __treescope_repr__(self, path: Any, subtree_renderer: Any) -> Any:
        """Thin lazy treescope hook: the dataclass-dump guard.

        Delegates to the bridge; falls through to ``NotImplemented`` on the
        ``disabled()`` scope or any fault -- never an exception in a cell.
        """
        try:
            from ..bridge.treescope import treescope_repr

            return treescope_repr(self, path, subtree_renderer)
        except Exception:  # noqa: BLE001 - degrade to treescope's default
            return NotImplemented

    def __iter__(self) -> Iterator[Any]:
        """Loops through all tensors in the log."""
        if self._tracing_finished:
            return iter(self._layer_list)
        else:
            return iter(list(self._raw_graph_ws.raw_layer_dict.values()))

    def save(
        self,
        path: str | Path,
        *,
        level: str = "portable",
        overwrite: bool = False,
        include_outs: bool | None = None,
        include_weights: bool = False,
        include_grads: bool | None = None,
        include_saved_args: bool | None = None,
        include_rng_states: bool | None = None,
        include_activations: bool = False,
        include_source: bool = True,
        include_custom_attributes: bool = True,
        include_buffer_values: bool | None = None,
        strict: bool = True,
    ) -> None:
        """Save this trace as a ``.tlspec`` directory; ``tl.load(path)`` reads it back.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory.
        level:
            ``"audit"``, ``"executable_with_callables"``, ``"portable"`` or
            ``"runnable"``.
        overwrite:
            Replace an existing directory at ``path``.
        include_outs:
            Persist activation payloads (``None`` follows the level default).
        include_weights:
            For runnable saves, bundle the full capture-time ``state_dict``: all
            named parameters and persistent buffers, as state rather than a
            reconstructed model.
        include_grads:
            Persist gradient payloads (``None`` follows the level default).
        include_saved_args:
            Persist saved call arguments (``None`` follows the level default).
        include_rng_states:
            Persist RNG states (``None`` follows the level default).
        include_activations:
            Archive exactly the capture-time ``save=``-selected payloads for
            inspection and eligible byte-exact attestation; those payloads never
            seed execution.
        include_source:
            Embed the captured model source code (default ``True``); ``False``
            strips verbatim source, docstrings, and source-file references.
            Absolute source paths are always relativized to a bare basename.
        include_custom_attributes:
            Persist user-set custom attributes.
        include_buffer_values:
            Persist buffer values (``None`` follows the level default).
        strict:
            Refuse typed on any field the level cannot persist.

        Warning
        -------
        Portable bundles contain a pickle file, but the default load path
        decodes it through a restricted, default-deny unpickler; foreign
        ``custom`` callable modules are never imported unless explicitly
        trusted (``trust_custom_callables=True`` /
        ``allowed_custom_callable_modules=...``). Only grant that trust to
        artifacts whose provenance you trust.
        """

        from .._io.bundle import save as save_bundle

        save_bundle(
            self,
            path,
            level=level,
            overwrite=overwrite,
            include_outs=include_outs,
            include_weights=include_weights,
            include_grads=include_grads,
            include_saved_args=include_saved_args,
            include_rng_states=include_rng_states,
            include_activations=include_activations,
            include_source=include_source,
            include_custom_attributes=include_custom_attributes,
            include_buffer_values=include_buffer_values,
            strict=strict,
        )

    def reconstruct_output(self, values: Literal["out", "transformed"] = "out") -> Any:
        """Reconstruct the traced model's final Python output object.

        Parameters
        ----------
        values:
            Leaf value source: ``"out"`` or ``"transformed"``.

        Returns
        -------
        Any
            Reconstructed model return value.
        """

        return reconstruct_output(self, values=values)

    def reconstruct_container(
        self,
        *,
        site: Any = None,
        role: Any = None,
        values: Literal["out", "transformed"] = "out",
    ) -> Any:
        """Reconstruct a captured container selected by site and role.

        Parameters
        ----------
        site:
            Optional boundary site selector.
        role:
            Optional boundary role selector.
        values:
            Leaf value source: ``"out"`` or ``"transformed"``.

        Returns
        -------
        Any
            Reconstructed Python container.
        """

        return reconstruct_container(self, site=site, role=role, values=values)

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with non-picklable weakref-backed accessors stripped."""
        return _trace_state_io.trace_getstate(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Trace":
        """Return a detached deep copy using the supported pickle semantics.

        Parameters
        ----------
        memo:
            Standard deepcopy memo populated with the cloned trace.

        Returns
        -------
        Trace
            Detached trace copy whose tensor payloads no longer carry autograd history.
        """

        existing = memo.get(id(self))
        if existing is not None:
            return cast("Trace", existing)
        cloned = cast(
            "Trace",
            pickle.loads(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)),
        )
        # A deepcopy stays inside the live session, so the SAME-process copy
        # keeps the full verification verdict -- the pickle path deliberately
        # refuses to persist a positive claim.
        cloned._provenance_verified = self._provenance_verified
        cloned._provenance_verification_reason = self._provenance_verification_reason
        memo[id(self)] = cloned
        return cloned

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and rebuild weakref-backed links."""
        return _trace_state_io.trace_setstate(self, state, conditional_accessor=ConditionalAccessor)

    def replace_state_from(self, trace: "Trace") -> None:
        """Atomically replace this log's run-state from another ``Trace``.

        This method is intended for intervention rerun. The rerun engine builds
        ``trace`` off to the side and calls this only after validation ops.
        The final state replacement uses one state-restore pass over the new fields
        to minimize torn-state windows. Concurrent reads during rerun are
        unsupported; no lock is taken.

        Parameters
        ----------
        trace:
            Fully postprocessed fresh log whose graph, layer containers,
            accessors, output metadata, shape/hash fields, and per-pass entries
            should replace this log's current run-state.

        Returns
        -------
        None
            This log is mutated in place.
        """
        return _trace_rerun_state.replace_state_from(self, trace=trace)

    def _refresh_matching_rerun_state_from(self, new_log: "Trace", *, moved: bool = False) -> bool:
        """Refresh payload-bearing fields from a same-shape rerun.

        Parameters
        ----------
        new_log:
            Fully captured and postprocessed rerun candidate.
        moved:
            True when the caller discards ``new_log`` afterwards, so values move.

        Returns
        -------
        bool
            True when refreshed in place; False when labels differ (``replace_state_from``).
        """
        return _trace_rerun_state.refresh_matching_rerun_state_from(self, new_log, moved=moved)

    def _copy_user_annotations(self, annotations: Any) -> dict[str, Any] | None:
        """Copy the reserved user annotation namespace from a mapping.

        Parameters
        ----------
        annotations:
            Existing annotation mapping.

        Returns
        -------
        dict[str, Any] | None
            Copied user namespace, or ``None`` when absent.
        """
        return _trace_rerun_state.copy_user_annotations(self, annotations=annotations)

    def _merge_user_annotations(
        self,
        fresh_annotations: Any,
        preserved_user_annotations: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Merge preserved user annotations into fresh internal annotations.

        Parameters
        ----------
        fresh_annotations:
            Annotation mapping from the fresh rerun.
        preserved_user_annotations:
            Previously stored ``annotations["user"]`` mapping.

        Returns
        -------
        dict[str, Any]
            Fresh annotations plus the preserved user namespace.
        """
        return _trace_rerun_state.merge_user_annotations(
            self,
            fresh_annotations=fresh_annotations,
            preserved_user_annotations=preserved_user_annotations,
        )

    def _copy_rerun_value(self, value: Any) -> Any:
        """Copy rerun metadata while keeping tensor payload identities.

        Parameters
        ----------
        value:
            Value copied from the fresh rerun trace.

        Returns
        -------
        Any
            Copied metadata container, or the original tensor/object when
            copying would be incorrect or unnecessary.
        """
        return _trace_rerun_state.copy_rerun_value(self, value=value)

    def _append_state_from(self, trace: "Trace") -> None:
        """Merge compatible chunk outs from ``trace`` into this log.

        Parameters
        ----------
        trace:
            Freshly captured append chunk whose topology and tensor metadata
            have already been validated against this log.
        """
        return _trace_rerun_state.append_state_from(self, trace=trace)

    # ********************************************
    # ******** Public Convenience Methods ********
    # ********************************************

    def discharge_against(self, real_trace: "Trace") -> Any:
        """Discharge this structure-only trace's hypotheses against a real run.

        DOCUMENTED-UNSTABLE surface (no deprecation shim owed on
        rename). Only defined on a structure-only capture; ``real_trace``
        must be an ordinary settled COMPLETE capture of the same graph.
        Returns a frozen ``StructureDischarge`` record (per-claim table +
        overall verdict); neither trace is mutated — the verdict registers in
        a weak-keyed side table consulted by the structure-only capability
        chokepoint (a REFUTED discharge flips hypothesis consumers to typed
        refusals). See ``torchlens._capture.structure_only.discharge_against``
        for the full join/precondition contract.
        """
        return _trace_rerun_state.discharge_against(self, real_trace=real_trace)

    @property
    def cost(self) -> "TraceCost":
        """Compute, memory and timing totals of a Trace."""

        return TraceCost(self)

    @property
    def graph(self) -> "TraceGraph":
        """Graph-shape facts of a Trace: edges, degrees, branching and conditionals."""

        return TraceGraph(self)

    @property
    def autograd(self) -> "TraceAutograd":
        """Backward-pass records and grad_fn facts of a Trace."""

        return TraceAutograd(self)

    @property
    def source(self) -> "TraceSource":
        """Where the traced model's source lives: class, ``__init__`` and ``forward`` facts."""

        return TraceSource(self)

    @property
    def provenance(self) -> "TraceProvenance":
        """How, when and from what this Trace was captured."""
        return TraceProvenance(self)

    @property
    def options(self) -> "TraceSettings":
        """The capture options this Trace ran with (the stored snapshot)."""

        return TraceSettings(self)

    @property
    def health(self) -> "TraceHealth":
        """Honesty diagnostics of the capture: witness and escape-detector facts."""

        return TraceHealth(self)

    @property
    def _graph_is_recurrent(self) -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self._layer_num_calls.values())

    @property
    def _cost_forward_peak_memory_pair(self) -> "dict[str, Any] | None":
        """Session-time (live, resident) forward peak pair with backend named."""
        from .._capture.peak_memory import read_peak_pair  # report/ may not read peaks

        return read_peak_pair(self)

    @property
    def _layers_recurrent(self) -> "LayerAccessor":
        """Access Layers with more than one captured pass."""

        return _trace_views.layers_recurrent(self)

    @property
    def _graph_max_ops_per_layer(self) -> int:
        """Maximum number of ops for any layer."""
        return max(self._layer_num_calls.values(), default=1)

    @property
    def _graph_is_branching(self) -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.children) > 1 for entry in self._layer_list)

    @property
    def _graph_has_conditional_branching(self) -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.control_flow.is_in_conditional_body for entry in self._layer_list)

    @property
    def _graph_has_conditionals(self) -> bool:
        """Whether this Trace contains conditional-flow records."""

        return len(self._graph_conditionals) > 0

    @property
    def _graph_is_dynamic(self) -> bool:
        """Whether execution depends on runtime tensor values."""

        return self.graph.has_conditionals

    @property
    def _source_forward_location(self) -> str | None:
        """Combined forward source location."""

        if self._source_forward_file is None or self._source_forward_line is None:
            return None
        return f"{self._source_forward_file}:{self._source_forward_line}"

    @property
    def _source_class_location(self) -> str | None:
        """Combined model class source location."""

        if self._source_class_file is None or self._source_class_line is None:
            return None
        return f"{self._source_class_file}:{self._source_class_line}"

    @property
    def _source_init_location(self) -> str | None:
        """Combined model ``__init__`` source location."""

        if self._source_init_file is None or self._source_init_line is None:
            return None
        return f"{self._source_init_file}:{self._source_init_line}"

    @property
    def _cost_last_backward_duration(self) -> Duration | None:
        """Most recent backward-pass duration, if any."""

        if not self._cost_backward_durations:
            return None
        return as_duration(self._cost_backward_durations[-1])

    @property
    def _cost_total_backward_duration(self) -> Duration:
        """Sum of all captured backward-pass durations."""

        return Duration(sum(self._cost_backward_durations))

    @property
    def _autograd_last_backward_root_grad_fn_object_id(self) -> int | None:
        """Most recent backward root grad_fn_handle object id, if any."""

        if not self._autograd_backward_root_grad_fn_object_ids:
            return None
        return self._autograd_backward_root_grad_fn_object_ids[-1]

    @property
    def _cost_overhead_duration(self) -> Duration:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self._provenance_duration - self._cost_func_calls_duration

    @property
    def _provenance_duration(self) -> Duration:
        """Total capture-phase duration in seconds."""

        if not self._provenance_start_time or not self._provenance_end_time:
            return Duration(0)
        return Duration(self._provenance_end_time - self._provenance_start_time)

    @property
    def _provenance_kind(self) -> str:
        """Return this product's capture kind: ``"episode"`` or ``"plain"``."""

        return _doors.report()._trace_facts.provenance_kind(self)

    @property
    def episode(self) -> Any | None:
        """Return the parsed episode step ledger, or ``None``."""

        return _doors.report()._trace_facts.episode(self)

    @property
    def episode_coupling(self) -> Any | None:
        """Return the attested coupling verdict, or ``None``."""

        return _doors.report()._trace_facts.episode_coupling(self)

    @property
    def _cost_total_flops_forward(self) -> Flops:
        """Total forward FLOPs (the canonical identity-partition aggregation)."""

        return _doors.report()._trace_facts.cost_total_flops_forward(self)

    @property
    def _cost_total_flops_backward(self) -> Flops:
        """Total backward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(
                entry.cost.flops_backward
                for entry in self._layer_list
                if entry.cost.flops_backward is not None
            )
        )

    @property
    def _cost_total_flops(self) -> Flops:
        """Total FLOPs (forward + backward)."""
        return Flops(self.cost.total_flops_forward + self.cost.total_flops_backward)

    @property
    def _cost_flops_by_op_type(self) -> "_CallableDict":
        """Group FLOPs by layer type."""

        return _doors.report()._trace_facts.cost_flops_by_op_type(self)

    @property
    def _cost_total_macs_forward(self) -> Macs:
        """Total TRUE forward MACs over ops with a known MAC split."""

        return _doors.report()._trace_facts.cost_total_macs_forward(self)

    @property
    def factcore(self) -> Any:
        """The ONE numbers substrate for this trace."""

        return _doors.report()._trace_facts.factcore(self)

    @property
    def health_facts(self) -> Any:
        """Normalized, serializable health observations."""

        return _doors.report()._trace_facts.health_facts(self)

    @property
    def nonfinite_verdict(self) -> str:
        """Three-state health verdict: found / checked_and_clean / not_checked."""

        return _doors.report()._trace_facts.nonfinite_verdict(self)

    @property
    def unknown_flop_ops(self) -> tuple[Any, ...]:
        """Named ledger of ops with unknown FLOPs, grouped by op name."""

        return _doors.report()._trace_facts.unknown_flop_ops(self)

    @property
    def compute_coverage(self) -> dict[str, int]:
        """Four-way compute coverage counts over every layer-list row."""

        return _doors.report()._trace_facts.compute_coverage(self)

    @property
    def macs_unknown_split_ops(self) -> tuple[str, ...]:
        """Labels of compute ops whose exact MAC split cannot be derived."""

        return _doors.report()._trace_facts.macs_unknown_split_ops(self)

    @property
    def _cost_total_macs_backward(self) -> Macs | None:
        """Backward MACs are not derivable and always ``None`` (estimates only)."""

        return None

    @property
    def _cost_total_macs(self) -> Macs | None:
        """Total MACs are not derivable and always ``None`` (see total_macs_backward)."""

        return None

    @property
    def _cost_macs_by_kind(self) -> "_CallableDict":
        """Group TRUE forward MACs by layer type."""

        return _doors.report()._trace_facts.cost_macs_by_kind(self)

    @property
    def params(self) -> ParamAccessor:
        """Access parameter metadata by address, short name, or index."""
        return self._param_logs

    @property
    def _graph_tied_param_groups(self) -> tuple[tuple[str, ...], ...]:
        """Alias-address groups of parameters registered at more than one path."""

        return _trace_views.graph_tied_param_groups(self)

    @property
    def ops(self) -> TraceOpAccessor:
        """Access per-invocation Op records by label or index."""

        return _trace_views.ops(self)

    @property
    def transforms(self) -> tuple[Op, ...]:
        """Return transform-boundary operation records."""

        return _trace_views.transforms(self)

    @property
    def model_profile(self) -> ModelProfile:
        """Return a computed semantic I/O profile descriptor."""

        return _trace_profile.model_profile(self)

    def profile(
        self,
        level: Literal["op", "module", "call"] = "op",
        *,
        sort_by: Literal["time", "flops", "activation_memory", "param_count"] = "time",
        ascending: bool = False,
        k: int | None = None,
        group_by: Literal["source_line"] | None = None,
    ) -> "ProfileReport":
        """Return a unified resource profile assembled from this trace."""

        return _doors.report()._trace_facts.profile(
            self, level, sort_by=sort_by, ascending=ascending, k=k, group_by=group_by
        )

    def sites(self) -> Any:
        """Return the tabular view of this trace's structural sites."""

        return _trace_inventory.build_sites_table(self)

    def bill_of_materials(self) -> dict[str, Any]:
        """Return the inventory of what this trace actually contains."""

        return _doors.report()._trace_facts.bill_of_materials(self)

    def explain(
        self,
        *,
        audience: Literal["researcher", "practitioner", "auto"] = "auto",
        max_tokens: int | None = None,
    ) -> "ExplainReport":
        """Explain this trace in plain language (DOCUMENTED-UNSTABLE)."""

        return _doors.report()._trace_facts.explain(self, audience=audience, max_tokens=max_tokens)

    def capabilities(self) -> Any:
        """Return the report-family capability card for THIS trace."""

        return _doors.report()._trace_facts.capabilities(self)

    def receptive_fields(
        self,
        level: Literal["op", "layer", "call", "module"] = "op",
        *,
        input_op: "Op | None" = None,
        statuses: "Collection[ReceptiveFieldStatus] | None" = None,
        sort_by: str | None = None,
        ascending: bool = True,
        direction: "ReceptiveFieldDirection | str" = "receptive",
    ) -> "ReceptiveFieldProfile":
        """Return the trace-wide geometric receptive-field table."""

        return _doors.receptive_field()._trace_tables.receptive_fields(
            self,
            level,
            input_op=input_op,
            statuses=statuses,
            sort_by=sort_by,
            ascending=ascending,
            direction=direction,
        )

    def projective_fields(
        self,
        level: Literal["op", "layer", "call", "module"] = "op",
        *,
        input_op: "Op | None" = None,
        statuses: "Collection[ReceptiveFieldStatus] | None" = None,
        sort_by: str | None = None,
        ascending: bool = True,
    ) -> "ReceptiveFieldProfile":
        """Return the trace-wide source-anchored projective-field table."""

        return _doors.receptive_field()._trace_tables.projective_fields(
            self, level, input_op=input_op, statuses=statuses, sort_by=sort_by, ascending=ascending
        )

    @property
    def layers(self) -> "LayerAccessor":
        """Access aggregate per-layer metadata by label, index, or pass notation."""

        return _trace_views.layers(self)

    @property
    def modules(self) -> "ModuleAccessor":
        """Access structured per-module metadata by address, index, or pass notation."""
        return self._module_logs

    @property
    def module_collapse_order(self) -> list[tuple[str, float]]:
        """Every module's smart-collapse priority (``Module.collapse_score``), highest first."""

        return _doors.visualization()._trace_collapse.module_collapse_order(self)

    def _collapse_schedule(self, context: Any = None) -> Any:
        """Private door for ``lenses._budget``: the collapse schedule under ``context``.

        Reached through ``_doors.visualization()`` so the lens modules import no collapse code.
        """

        return _doors.visualization()._trace_collapse.collapse_schedule(self, context)

    def _suggest_layer_labels(self, selector: str, *, limit: int = 10) -> list[str]:
        """Did-you-mean layer labels for a failed lookup (the refusal text's candidates)."""

        return _trace_views.suggest_layer_labels(self, selector, limit=limit)

    def modules_with_facet(self, name: str) -> Iterator[Any]:
        """Yield Modules with facet ``name`` available now, reused ones included."""

        return _trace_views.modules_with_facet(self, name)

    def attention_blocks(self) -> Iterator[Any]:
        """Yield Modules with an available query-projection facet, reused ones included."""

        return self.modules_with_facet("q")

    @property
    def module_calls(self) -> TraceModuleCallAccessor:
        """Access per-invocation ModuleCall records by call label or index."""

        return _trace_views.module_calls(self)

    def _root_call(self) -> "ModuleCall":
        """Return the top-level ModuleCall for internal call-tree traversal."""

        return _call_tree.root_call(self)

    @property
    def call_tree(self) -> "ModuleCall":
        """Return the root ModuleCall for the nested dynamic call tree."""

        return self._root_call()

    def walk_calls(self) -> Iterator["ModuleCall"]:
        """Yield every ModuleCall in call-tree depth-first order."""

        return _call_tree.walk_calls(self)

    def show_call_tree(
        self,
        max_depth: int | None = None,
        include_atomic: bool = True,
        show_call_index: bool = True,
        file: TextIO | None = None,
    ) -> None:
        """Print this Trace's full ModuleCall tree as an ASCII tree."""

        return _call_tree.show_call_tree(self, max_depth, include_atomic, show_call_index, file)

    @property
    def root_module(self) -> "Module":
        """The root module (the model itself)."""
        return cast("Module", self._module_logs["self"])

    @property
    def num_edges(self) -> int:
        """Distinct edges in the per-pass Op graph, including boundary edges."""

        return _trace_views.num_edges(self)

    @property
    def _graph_num_compute_edges(self) -> int:
        """Distinct Op graph edges whose endpoints are both compute Ops."""

        return _trace_views.graph_num_compute_edges(self)

    @property
    def _graph_num_buffer_edges(self) -> int:
        """Distinct Op graph edges with at least one buffer endpoint."""

        return _trace_views.graph_num_buffer_edges(self)

    @property
    def _graph_num_layer_edges(self) -> int:
        """Distinct edges in the aggregate Layer graph."""

        return _trace_views.graph_num_layer_edges(self)

    @property
    def _autograd_num_backward_edges(self) -> int | None:
        """Distinct edges in the backward GradFn graph, or ``None`` if ungated."""

        return _trace_views.autograd_num_backward_edges(self)

    @property
    def _graph_branching_factor(self) -> float:
        """Mean fan-out: children (consumers) per compute Op."""

        return _trace_views.graph_branching_factor(self)

    @property
    def _graph_max_in_degree(self) -> int:
        """Maximum number of parents over compute Ops."""

        return _trace_views.graph_max_in_degree(self)

    @property
    def _graph_max_out_degree(self) -> int:
        """Maximum number of children over compute Ops."""

        return _trace_views.graph_max_out_degree(self)

    @property
    def _autograd_intermediate_derived_grads(self) -> IntermediateDerivedGradAccessor:
        """Access exact op-level derived gradient records, when a backend provides them."""

        return _trace_views.autograd_intermediate_derived_grads(self)

    @_autograd_intermediate_derived_grads.setter
    def _autograd_intermediate_derived_grads(self, value: IntermediateDerivedGradAccessor) -> None:
        """Store exact op-level derived gradient records."""

        self.__dict__["_intermediate_derived_grads"] = value

    @property
    def has_trainable_params(self) -> bool:
        """Whether this Trace contains at least one trainable parameter."""

        return self.num_params_trainable > 0

    @property
    def has_frozen_params(self) -> bool:
        """Whether this Trace contains at least one frozen parameter."""

        return self.num_params_frozen > 0

    @property
    def buffers(self) -> "BufferAccessor":
        """Access buffer metadata by address, short name, or index."""
        return self._buffer_accessor  # type: ignore[return-value]

    @property
    def _graph_orphans(self) -> OrphanAccessor:
        """Access retained orphan island operations by raw or final label."""

        return _trace_views.graph_orphans(self)

    @property
    def _autograd_grad_fns(self) -> GradFnAccessor:
        """Access backward grad_fn_handle metadata by label, index, pass label, or substring."""

        return _trace_views.autograd_grad_fns(self)

    @property
    def _autograd_grad_fn_calls(self) -> TraceGradFnCallAccessor:
        """Access per-invocation GradFnCall records by qualified label or index."""

        return _trace_views.autograd_grad_fn_calls(self)

    @property
    def _autograd_grad_fn_fire_timings(self) -> "OrderedDict[str, Duration | None]":
        """Return live per-fire backward timing spans keyed like ``grad_fn_calls``."""
        return _trace_stats_grad_fn_site.grad_fn_fire_timings(self)

    @property
    def grad_fn_site_summary(self) -> "OrderedDict[str | None, dict[str, Any]]":
        """Return the read-only per-site rollup of backward grad-fn facts."""

        return _trace_stats_grad_fn_site.grad_fn_site_summary(self)

    @property
    def _autograd_backward_passes(self) -> BackwardPassAccessor:
        """Access backward pass records by 0-based position or named pass number."""

        return _trace_views.autograd_backward_passes(self)

    @property
    def _autograd_last_backward_pass(self) -> BackwardPass | None:
        """Return the most recent backward pass record, if any."""

        backward_passes = self.autograd.backward_passes
        if not backward_passes:
            return None
        return backward_passes[-1]

    def _sync_backward_projection_if_needed(self) -> None:
        """Synchronize lazy backward projections from runtime sidecar events."""
        return _doors.backends_torch()._trace_backward.sync_backward_projection_if_needed(self)

    @property
    def saved_ops(self) -> Accessor[Op]:
        """Access Ops with saved activations."""

        return _trace_views.saved_ops(self)

    @property
    def saved_grad_ops(self) -> Accessor[Op]:
        """Access Ops with saved gradients."""

        return _trace_views.saved_grad_ops(self)

    @property
    def saved_layers(self) -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved activation."""

        return _trace_views.saved_layers(self)

    @property
    def _layers_saved_grad(self) -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved gradient."""

        return _trace_views.layers_saved_grad(self)

    @property
    def saved_module_calls(self) -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved activations."""

        return self._module_calls_with_output_in(set(self.saved_ops.keys()))

    def _module_calls_with_output_in(self, op_labels: set[str]) -> Accessor[Any]:
        """Access ModuleCalls with at least one output Op among ``op_labels``."""
        return _trace_views.module_calls_with_output_in(self, op_labels)

    @property
    def saved_modules(self) -> Accessor[Any]:
        """Access Modules with at least one saved-activation ModuleCall."""

        return _trace_views.saved_modules(self)

    @property
    def saved_grad_module_calls(self) -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved gradients."""

        return self._module_calls_with_output_in(set(self.saved_grad_ops.keys()))

    @property
    def saved_grad_modules(self) -> Accessor[Any]:
        """Access Modules with at least one saved-gradient ModuleCall."""

        return _trace_views.saved_grad_modules(self)

    @property
    def _autograd_saved_grad_fn_calls(self) -> Accessor[Any]:
        """Access GradFnCall records with saved gradient inputs or outputs."""

        return _trace_views.autograd_saved_grad_fn_calls(self)

    @property
    def _autograd_saved_grad_fns(self) -> GradFnAccessor:
        """Access GradFns containing at least one saved GradFnCall."""

        return _trace_views.autograd_saved_grad_fns(self)

    @property
    def compute_ops(self) -> Accessor[Op]:
        """Access Ops that are not graph-boundary sentinels."""

        return _trace_views.compute_ops(self)

    @property
    def _layers_compute(self) -> Accessor[Layer]:
        """Access Layers whose representative Op is not a boundary sentinel."""

        return _trace_views.layers_compute(self)

    @property
    def input_ops(self) -> Accessor[Op]:
        """Access flat input-boundary Ops."""

        return _trace_views.input_ops(self)

    @property
    def num_input_layers(self) -> int:
        """Number of distinct input-boundary Layers (a multi-pass Layer counts once)."""

        return len(dict.fromkeys(self.input_layers))

    @property
    def output_ops(self) -> Accessor[Op]:
        """Access flat output-boundary Ops."""

        return _trace_views.output_ops(self)

    @property
    def num_output_layers(self) -> int:
        """Number of distinct output-boundary Layers (a multi-pass Layer counts once)."""

        return len(dict.fromkeys(self.output_layers))

    @property
    def num_buffer_layers(self) -> int:
        """Number of distinct buffer-boundary Layers."""

        return len(dict.fromkeys(self.buffer_layers))

    @property
    def buffer_read_ops(self) -> list[str]:
        """Labels for buffer Ops that read registered-buffer values into the graph."""

        return _trace_views.buffer_read_ops(self)

    @property
    def buffer_write_ops(self) -> list[str]:
        """Labels for buffer Ops that record registered-buffer write events."""

        return _trace_views.buffer_write_ops(self)

    @property
    def _layers_internal_source(self) -> Accessor[Layer]:
        """Access Layers representing internal-source positions."""

        return _trace_views.layers_internal_source(self)

    @property
    def _layers_internal_sink(self) -> Accessor[Layer]:
        """Access Layers representing internal-sink positions."""

        return _trace_views.layers_internal_sink(self)

    @property
    def num_uncalled_modules(self) -> int | None:
        """Number of registered source-model modules not called in this Trace."""

        return _trace_views.num_uncalled_modules(self)

    @property
    def ops_with_params(self) -> Accessor[Op]:
        """Access Ops that use at least one parameter tensor."""

        return _trace_views.ops_with_params(self)

    def between(self, sources: Any, sinks: Any) -> "TraceSlice":
        """Return the sub-DAG view carrying influence from sources to sinks."""

        return _doors.trace_slice().build_slice_between(self, sources, sinks)

    def subgraph(self, selection: Any) -> "TraceSlice":
        """Return the sub-DAG view of any ACT region of this trace."""

        return _doors.trace_slice().build_slice_from_selection(self, selection)

    def stack(self, selector: "Any") -> "StackedActivations":
        """Stack selector-matched saved outputs in recorded execution order."""

        return stack_traces(self, selector=selector)

    @property
    def injected_ops(self) -> tuple[Any, ...]:
        """Injected-op records from ``log_injections=True`` hooks."""

        return _doors.intervention()._trace_edits.injected_ops(self)

    @property
    def model_ops(self) -> tuple[Any, ...]:
        """The MODEL half of the injections query split."""

        return tuple(self._layer_list)

    def save_intervention(
        self,
        path: str | Path,
        *,
        level: str = "executable_with_callables",
        allow_direct_writes: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Save this log's intervention recipe to a ``.tlspec`` directory."""

        return _doors.intervention()._trace_edits.save_intervention(
            self, path, level=level, allow_direct_writes=allow_direct_writes, overwrite=overwrite
        )

    @cached_property
    def recipe(self) -> "FrozenInterventionSpec":
        """Return an immutable snapshot of this log's intervention recipe."""

        return self._ensure_intervention_spec().freeze()

    def _history_site_payload(self, site: Any) -> Any:
        """Forward to ``_trace_intervention_records.history_site_payload``."""

        return _trace_intervention_records.history_site_payload(site)

    def _attach_hooks(
        self,
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
    ) -> tuple[str, ...]:
        """Stage sticky hooks on the recipe (the ``do`` attach path); returns the handle ids."""

        return _doors.intervention()._trace_hooks.attach_sticky_hooks(
            self,
            hooks_or_site,
            hook,
            *extra_hooks,
            strict=strict,
        )

    def _refuse_chunked_batch_coupling(self, entries: list[Any]) -> None:
        """Forward to ``_doors.intervention()._edit_guards.refuse_chunked_batch_coupling``."""

        return _doors.intervention()._edit_guards.refuse_chunked_batch_coupling(self, entries)

    def __enter__(self) -> "Trace":
        """Enter a scoped edit: ``with trace.fork() as scoped: scoped.do(...)``."""

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Leave the scope: detach the sticky hooks the last ``do()`` inside it staged."""

        _doors.intervention()._trace_edits.detach_last_do_hooks(self)

    def clear_edits(
        self,
        site: Any = None,
        handle: Any = None,
        *,
        strict: bool = False,
    ) -> "Trace":
        """Detach sticky hooks by site or handle."""

        return _doors.intervention()._trace_edits.clear_edits(self, site, handle, strict=strict)

    def _clear_hooks(self) -> "Trace":
        """Clear every sticky hook from the recipe (internal; ``clear_edits`` is the public door)."""

        return _doors.intervention()._trace_hooks.clear_hooks(self)

    @property
    def _graph_edges(self) -> tuple[Any, ...]:
        """Forward to ``_doors.intervention()._trace_edits.graph_edges``."""

        return _doors.intervention()._trace_edits.graph_edges(self)

    def _settle_selection_do(
        self, txn: "_DoTransaction", selected_engine: str, mutation_kind: str
    ) -> None:
        """Forward to ``_doors.intervention()._trace_edits.settle_selection_do``."""

        return _doors.intervention()._trace_edits.settle_selection_do(
            self, txn, selected_engine, mutation_kind
        )

    def do(
        self,
        site: Any,
        edit: Any = None,
        /,
        *,
        model: nn.Module | None = None,
        inputs: Any = None,
        intervention: "InterventionOptions | None" = None,
        loss: Callable[["Trace"], torch.Tensor] | None = None,
    ) -> "Trace":
        """Apply an intervention and dispatch to replay, rerun, or set-only."""

        return _doors.intervention()._trace_edits.do(
            self,
            site,
            edit,
            model=model,
            inputs=inputs,
            intervention=intervention,
            loss=loss,
        )

    def _do_with_options(
        self,
        site: Any,
        value_or_hook: Any,
        *,
        model: nn.Module | None,
        x: Any,
        intervention_options: "InterventionOptions",
    ) -> "Trace":
        """Forward to ``_doors.intervention()._trace_edits.do_with_options``."""

        return _doors.intervention()._trace_edits.do_with_options(
            self,
            site,
            value_or_hook,
            model=model,
            x=x,
            intervention_options=intervention_options,
        )

    @property
    def armed_edits(self) -> tuple[Any, ...]:
        """The gradient rules in this trace's recipe, as ``Rule`` clauses."""

        return armed_edits(self)

    def _armed_summary_line(self) -> str | None:
        """Summary text while an armed gradient rule has not fired (or ``None``)."""

        return armed_summary_line(self)

    def _record_do_intervention_event(
        self,
        txn: _DoTransaction,
        *,
        status: str | None,
        error: str | None = None,
        staged_only: bool = False,
        extra_fields: dict[str, Any] | None = None,
        event_row: bool = False,
    ) -> dict[str, Any]:
        """Forward to ``_doors.intervention()._trace_edits.record_do_intervention_event``."""

        return _doors.intervention()._trace_edits.record_do_intervention_event(
            self,
            txn,
            status=status,
            error=error,
            staged_only=staged_only,
            extra_fields=extra_fields,
            event_row=event_row,
        )

    def fork(self, name: str | None = None) -> "Trace":
        """Create a copy-on-write intervention fork of this log."""

        return _fork_lineage.fork_trace(self, name)

    def _record_operation(self, op: str, **payload: Any) -> None:
        """Forward to ``_trace_intervention_records.record_operation``."""

        return _trace_intervention_records.record_operation(self, op, **payload)

    def _warn_if_root_mutation(self) -> None:
        """Forward to ``_doors.intervention()._edit_guards.warn_if_root_mutation``."""

        return _doors.intervention()._edit_guards.warn_if_root_mutation(self)

    def _select_do_engine(self, engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Forward to ``_doors.intervention()._edit_guards.select_do_engine``."""

        return _doors.intervention()._edit_guards.select_do_engine(self, engine, model=model, x=x)

    def _apply_do_mutation(
        self,
        site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_edits.apply_do_mutation``."""

        return _doors.intervention()._trace_edits.apply_do_mutation(
            self,
            site,
            value_or_hook,
            engine=engine,
            strict=strict,
        )

    @staticmethod
    def _is_selection_batch(value: Any) -> bool:
        """Forward to ``_doors.intervention()._trace_selection_edits.is_selection_batch``."""

        return _doors.intervention()._trace_selection_edits.is_selection_batch(value)

    @staticmethod
    def _selection_batch_kind(selection: Any) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.selection_batch_kind``."""

        return _doors.intervention()._trace_selection_edits.selection_batch_kind(selection)

    def _apply_selection_batch_do(
        self, pairs: Any, *, engine: str, strict: bool
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_batch_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_batch_do(
            self, pairs, engine=engine, strict=strict
        )

    def _batch_act_plan(
        self, index: int, selection: Any, edit: Any
    ) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.batch_act_plan``."""

        return _doors.intervention()._trace_selection_edits.batch_act_plan(
            self, index, selection, edit
        )

    def _apply_cross_kind_batch_do(
        self, pairs: list[tuple[Any, Any]], kinds: list[str], *, engine: str, strict: bool
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_cross_kind_batch_do``."""

        return _doors.intervention()._trace_selection_edits.apply_cross_kind_batch_do(
            self, pairs, kinds, engine=engine, strict=strict
        )

    def _apply_selection_do(
        self, selection: Any, edit: Any, *, engine: str, strict: bool
    ) -> tuple[str, tuple[str, ...]]:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_do(
            self, selection, edit, engine=engine, strict=strict
        )

    def _attach_selection_plan_hooks(
        self, hook_items: list[dict[str, Any]], *, strict: bool
    ) -> tuple[str, ...]:
        """Forward to ``_doors.intervention()._trace_selection_edits.attach_selection_plan_hooks``."""

        return _doors.intervention()._trace_selection_edits.attach_selection_plan_hooks(
            self, hook_items, strict=strict
        )

    def _apply_selection_edge_do(
        self, selection: Any, lifted: Any, edit: Any, *, engine: str, strict: bool
    ) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_edge_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_edge_do(
            self, selection, lifted, edit, engine=engine, strict=strict
        )

    def _apply_selection_param_do(
        self, selection: Any, lifted: Any, edit: Any, *, engine: str, strict: bool
    ) -> str:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_selection_param_do``."""

        return _doors.intervention()._trace_selection_edits.apply_selection_param_do(
            self, selection, lifted, edit, engine=engine, strict=strict
        )

    def _apply_leaf_selection_edits(self, leaf_items: list[dict[str, Any]]) -> None:
        """Forward to ``_doors.intervention()._trace_selection_edits.apply_leaf_selection_edits``."""

        return _doors.intervention()._trace_selection_edits.apply_leaf_selection_edits(
            self, leaf_items
        )

    def _validate_supplied_model_matches_capture(self, model: nn.Module) -> None:
        """Forward to ``_doors.intervention()._trace_edits.validate_supplied_model_matches_capture``."""

        return _doors.intervention()._trace_edits.validate_supplied_model_matches_capture(
            self, model
        )

    def _fork_trace(self, *, name: str | None) -> "Trace":
        """Build a copy-on-write fork of this Trace (the builder)."""

        return build_fork(self, name=name)

    def _next_fork_name(self) -> str:
        """Forward to ``_fork_lineage.next_fork_name``."""

        return _fork_lineage.next_fork_name(self)

    def _rebind_fork_owner_refs(self) -> None:
        """Forward to ``_fork_lineage.rebind_fork_owner_refs``."""

        return _fork_lineage.rebind_fork_owner_refs(self)

    def _recipe_is_clean(self) -> bool:
        """Forward to ``_trace_intervention_records.recipe_is_clean``."""

        return _trace_intervention_records.recipe_is_clean(self)

    def _ensure_intervention_spec(self) -> InterventionSpec:
        """Forward to ``intervention._trace_edits.ensure_intervention_spec``."""

        return _doors.intervention()._trace_edits.ensure_intervention_spec(self)

    def _mark_intervention_spec_mutated(self, *, outs_stale: bool = True) -> None:
        """Forward to ``_trace_intervention_records.mark_intervention_spec_mutated``."""

        return _trace_intervention_records.mark_intervention_spec_mutated(
            self, outs_stale=outs_stale
        )

    def _validate_intervention_site(self, site: Any, *, strict: bool) -> None:
        """Forward to ``_doors.intervention()._trace_edits.validate_intervention_site``."""

        return _doors.intervention()._trace_edits.validate_intervention_site(
            self, site, strict=strict
        )

    def _target_spec_from_site(self, site: Any, *, strict: bool) -> "TargetSpec":
        """Forward to ``_doors.intervention()._trace_edits.target_spec_from_site``."""

        return _doors.intervention()._trace_edits.target_spec_from_site(self, site, strict=strict)

    def _save_new_outs(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor | list[Any],
        input_kwargs: dict[Any, Any] | None = None,
        save: str | list[str] = "all",
        save_grads: str | list[str] | None = "all",
        seed: int | None = None,
        backward_ready: bool | None = None,
        _run_until_plan: Any | None = None,
        _staged_plan: Any | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only outs (internal refresh engine)."""
        return _doors.validation()._trace_checks.save_new_outs(
            self,
            model,
            inputs,
            input_kwargs=input_kwargs,
            save=save,
            save_grads=save_grads,
            seed=seed,
            backward_ready=backward_ready,
            _run_until_plan=_run_until_plan,
            _staged_plan=_staged_plan,
        )

    def validate_forward_pass(
        self,
        ground_truth_output_tensors: list[torch.Tensor] | torch.Tensor,
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationStatus"]:
        """Validate saved outs against ground-truth model outputs."""
        return _doors.validation()._trace_checks.validate_forward_pass(
            self, ground_truth_output_tensors, verbose=verbose, validate_metadata=validate_metadata
        )

    @property
    def validation_replay_status(self) -> "ValidationStatus":
        """Return replay-validation availability or last completed result."""
        return _doors.validation()._trace_checks.validation_replay_status(self)

    @property
    def is_pristine(self) -> bool | None:
        """Whether nothing ever intervened on this trace, read from its saved evidence."""

        return derive_pristine(self)

    def run(
        self,
        model: Any = None,
        x: Any = None,
        /,
        *,
        chunk_paths: Any | None = None,
        replay: "ReplayOptions | None" = None,
        input_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        on_zero_fire: str | MissingType = MISSING,
    ) -> "Trace":
        """Rerun the live model on new input and return a new Trace; this Trace is unchanged."""
        options = {k: v for k, v in locals().items() if k not in ("self", "model", "x")}
        result = _doors.intervention()._trace_run.run(self, model, x, **options, transaction=False)
        return cast("Trace", result)

    def run_transaction(
        self,
        *,
        inputs: Any,
        seed: int | None = None,
        fast: bool = False,
        carry_state: bool = False,
        until: Any = None,
        save: Any = None,
        on_divergence: Literal["raise", "return"] = "raise",
        input_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        on_zero_fire: str | MissingType = MISSING,
    ) -> "RunResult":
        """Run a verified fresh execution with a report, leaving this Trace unchanged."""
        options = {name: value for name, value in locals().items() if name != "self"}
        return cast("RunResult", _doors.intervention()._trace_run.run(self, **options))

    def check_metadata_invariants(self) -> bool:
        """Run metadata invariant checks on this completed model log."""
        return _doors.validation()._trace_checks.check_metadata_invariants(self)

    def cleanup(self) -> None:
        """Delete log data, break cycles, and free cached GPU memory."""
        return cleanup(self)

    def _release_param_refs(self, *, allow_iter_rehydrate: bool = False) -> None:
        """Release live ``nn.Parameter`` references held by ParamLogs."""
        return release_param_refs(self, allow_iter_rehydrate=allow_iter_rehydrate)

    def _postprocess(
        self, output_tensors: list[torch.Tensor], output_tensor_addresses: list[str]
    ) -> None:
        """Run postprocessing on a completed raw capture pass."""
        from .._postprocess import postprocess as _impl

        self._postprocessing_active = True
        try:
            return _impl(
                self,
                output_tensors=output_tensors,
                output_tensor_addresses=output_tensor_addresses,
            )
        finally:
            self._postprocessing_active = False

    def _run_and_log_inputs_through_model(
        self,
        model: torch.nn.Module,
        input_args: torch.Tensor | list[Any],
        input_kwargs: dict[Any, Any] | None = None,
        layers_to_save: str | list[str | int] | None = "all",
        grad_layers_to_save: str | list[str | int] | None = "all",
        random_seed: int | None = None,
        postprocess: bool = True,
        reservation_resume: object | None = None,
    ) -> Any:
        """Run a forward pass and capture it into this model log."""
        from .._capture.run import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            seed=random_seed,
            postprocess=postprocess,
            reservation_resume=reservation_resume,
        )

    def backward(self, loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` and populate this Trace with backward metadata."""

        return _doors.backends_torch()._trace_backward.backward(self, loss, **backward_kwargs)

    def capture_backward(self) -> Any:
        """Return a context manager that captures user-managed backward calls."""
        return _doors.backends_torch()._trace_backward.capture_backward(self)

    def _disarm_autograd(self) -> None:
        """Detach this Trace from global autograd backward interception."""
        _doors.intervention()._trace_run.disarm(self)

    def _remove_log_entry(
        self,
        log_entry: Op,
        remove_references: bool = True,
        replacement_labels: dict[str, str] | None = None,
    ) -> None:
        """Remove a single layer-pass entry and scrub graph references."""
        return remove_log_entry(
            self,
            log_entry,
            remove_references=remove_references,
            replacement_labels=replacement_labels,
        )

    def _batch_remove_log_entries(
        self,
        entries_to_remove: Iterable[Op],
        remove_references: bool = True,
        replacement_labels: dict[str, str] | None = None,
    ) -> None:
        """Remove multiple layer-pass entries using single-pass filtering."""
        return batch_remove_log_entries(
            self,
            entries_to_remove,
            remove_references=remove_references,
            replacement_labels=replacement_labels,
        )

    @property
    def logged_values(self) -> dict[str, Any]:
        """Return values recorded via ``log_value`` during this capture."""

        values = (getattr(self, "annotations", {}) or {}).get("logged_values", {})
        return dict(values) if isinstance(values, dict) else {}

    def to_agent_json(self, *, max_ops: int | None = None) -> dict[str, Any]:
        """Return a self-describing machine-readable dump of this trace."""

        return _doors.report()._agent_json.build_agent_json(self, max_ops=max_ops)

    def _attach_dataframe_honesty(self, frame: Any) -> Any:
        """Stamp ``frame.attrs`` with this trace's capture-honesty facts."""

        from .._capture_honesty import attach_dataframe_honesty

        return attach_dataframe_honesty(frame, self)

    def to_pandas(self, include_decoded_output_summary: bool = False) -> "pd.DataFrame":
        """Return a dataframe containing one row per layer pass."""

        return _doors.stats()._trace_tables.to_pandas(
            self, include_decoded_output_summary=include_decoded_output_summary
        )

    def decode_output(
        self,
        k: int | None = None,
        *,
        style: str | None = None,
        head: str | None = None,
        tokenizer: Any | None = None,
    ) -> Any:
        """Return decoded output rows: the captured rows, or a decode on demand."""

        return _doors.stats()._trace_tables.decode_output(
            self, k=k, style=style, head=head, tokenizer=tokenizer
        )

    def output_table(
        self,
        k: int = 5,
        batch_items: int | Sequence[int] | None = None,
        *,
        style: str | None = None,
        head: str | None = None,
    ) -> "pd.DataFrame":
        """Return decoded output as a batch top-k dataframe."""

        return _doors.stats()._trace_tables.output_table(
            self, k=k, batch_items=batch_items, style=style, head=head
        )

    def show(self, kind: str | None = None, **kwargs: Any) -> Any:
        """Render this trace using a lightweight notebook-friendly dispatcher."""

        return _doors.visualization()._trace_draw.show(self, kind=kind, **kwargs)

    @teach_unknown_draw_keywords
    def draw(  # noqa: PLR0913 - the one draw door: every VisualizationOptions field is a keyword
        self,
        path: str | None = None,
        *,
        graph: Literal["forward", "backward", "combined"] = "forward",
        rolled: bool = False,
        depth: int = 1000,
        module: "Module | str | None" = None,
        collapse: CollapseLiteral = "none",
        fold_repeats: FoldRepeatsLiteral = None,
        fold_patterns: Any = None,
        theme: str = "torchlens",
        layout: VisNodePlacementLiteral = "auto",
        renderer: VisRendererLiteral = "graphviz",
        direction: VisDirectionLiteral | None = None,
        node_style: VisNodeModeLiteral = "default",
        show_buffers: BufferVisibilityLiteral = "meaningful",
        show_edge_shapes: bool = False,
        show_arg_labels: bool = True,
        show_legend: bool | None = None,
        color_by: Any = None,
        size_by: str | Callable[[Any], Any] | None = None,
        size_scale: str | None = None,
        stack_by: str | bool | Callable[[Any], Any] | None = None,
        node_overlay: str | Mapping[str, Any] | Callable[[Any], Any] | None = None,
        node_label_fields: list[str] | None = None,
        graph_overrides: dict[str, Any] | None = None,
        edge_overrides: dict[str, Any] | None = None,
        grad_edge_overrides: dict[str, Any] | None = None,
        module_overrides: dict[str, Any] | None = None,
        node_spec_fn: Callable[..., Any] | None = None,
        collapsed_node_spec_fn: Callable[..., Any] | None = None,
        collapse_fn: Callable[..., Any] | None = None,
        skip_fn: Callable[..., Any] | None = None,
        intervention_mode: VisInterventionModeLiteral = "node_mark",
        show_cone: bool = False,
        code_panel: "CodePanelOption" = False,
        font_size: int | None = None,
        dpi: int | None = None,
        order_siblings: bool = True,
        show_containers: Literal[False, "labels", "collapsed", "nodes"] = False,
        container_max_inline: int = 12,
        show_input_transform_summary: bool = False,
        show_orphans: bool = False,  # Invariants support flipping this; owner visual review pending.
        show_redundant_args: bool = False,
        show_saved_for_backward: bool = False,
        bwd: int | Iterable[int] | None = None,
        show_module_paths: bool = False,
        show_grad_values: bool = False,
        backward_node_spec_fn: Callable[..., Any] | None = None,
        intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
        file_format: str | None = None,
        open_viewer: bool = False,
        view: DrawViewLiteral | None = None,
    ) -> "Drawing":
        """Draw this trace and return the picture as a :class:`~torchlens.viz.Drawing`."""
        return _forward_draw(_doors.visualization()._trace_draw.draw, locals())

    def animate(
        self,
        site: Any,
        path: str | None = None,
        *,
        open_viewer: bool = False,
    ) -> "Drawing":
        """Animate the passes at ``site`` and return the picture as a :class:`~torchlens.viz.Drawing`."""
        return _doors.visualization()._trace_draw.animate(
            self, site, path=path, open_viewer=open_viewer
        )

    @property
    def nonfinite_ops(self) -> tuple[str, ...]:
        """Return pass-qualified labels of ops whose output held NaN or Inf."""

        return nonfinite_op_labels(self)

    @property
    def nonfinite_coverage(self) -> Any:
        """Return the evidence basis and coverage behind :attr:`nonfinite_ops`."""

        return nonfinite_coverage(self)

    def first_nonfinite(self, link_format: Literal["terminal", "html", "text"] = "text") -> str:
        """Return a text answer describing the first saved non-finite out."""

        return _doors.report()._trace_summary.first_nonfinite(self, link_format=link_format)

    def preview_fastlog(
        self,
        predicate: Callable[..., Any] | None = None,
        keep_op: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Render a fastlog predicate preview for this model graph."""

        return _doors.visualization()._trace_draw.preview_fastlog(
            self, predicate=predicate, keep_op=keep_op, **kwargs
        )

    def last_run_records(self) -> tuple["FireRecord", ...]:
        """Return fire records from the most recent replay, rerun, or live capture."""

        return _doors.visualization()._trace_draw.last_run_records(self)

    def _repr_html_(self) -> str:
        """Return the notebook HTML representation for this model log."""

        return _doors.report()._trace_summary.repr_html(self)

    def summary(  # noqa: PLR0913 - the ratified public grammar: every axis is a named keyword
        self,
        level: str | None = None,
        *,
        # Rebuilt grammar. Spellings
        # DOCUMENTED-UNSTABLE pending naming-session ratification.
        view: str | None = None,
        depth: Any = None,
        columns: list[str] | None = None,
        where: Any = None,
        buffers: str | None = None,
        fold_repeats: Any = None,
        max_rows: int | None = None,
        flop_convention: str | None = None,
        units: str | None = None,
        style: str | None = None,
        max_scan_elements: int | None = None,
        # One-call-only spellings (typed errors here, never no-ops).
        input_shape: Any = None,
        execution_mode: str | None = None,
        grad_mode: str | None = None,
        # Internal plumbing from the one-call door (not public grammar).
        _execution_note: str | None = None,
        _input_synthesis: str | None = None,
        # Refuse unknown names with their replacement or nearest grammar option.
        **unsupported: Any,
    ) -> str:
        """Render a summary of this capture."""

        return _forward_summary(_doors.report()._trace_summary.summary, locals())

    def narrate(self, last: "int | None" = None, *, select: "Any" = None) -> "str":
        """Render this trace's events in the live-narration grammar."""
        return _doors.report()._narrate.narrate(self, last=last, select=select)

    @property
    def has_aten_ops(self) -> bool:
        """Whether this trace carries recorded ATen operations."""
        return _trace_aten.has_aten_ops(self)

    @property
    def aten_ops(self) -> "tuple[AtenOp, ...]":
        """All recorded ATen operations in dispatch order (forward, then backward)."""
        return _trace_aten.aten_ops(self)

    @property
    def aten_gaps(self) -> "tuple[_ModePausedInteriorGap, ...]":
        """Disclosed stretches where ATen rows are missing by construction."""
        return _trace_aten.aten_gaps(self)


attach_trace_docs(Trace)
Trace._FIELD_FORK_POLICY = fork_policy_from_policy(Trace._FIELD_POLICY)  # type: ignore[attr-defined]
Trace._DEFAULT_FILL_STATE = default_fill_state_from_policy(Trace._FIELD_POLICY)  # type: ignore[attr-defined]

# The tlspec v8 coordinated bump retired this class's pre-release
# registrations: every formerly gated Trace field above now declares its
# persisting policy directly. The registrar mechanism itself stays for
# future sprints (torchlens/_io/prerelease.py).
