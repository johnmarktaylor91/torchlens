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
  structures are populated and used instead.  ``_tracing_finished`` also
  persists across the fast pass on purpose: fast-path postprocessing
  relies on the fully-populated lookup dicts from the exhaustive pass.

* **Explicit Trace custom_methods** - Public custom_methods are defined directly on
  ``Trace``. Heavier implementations may delegate into subpackages
  through local imports, but users still call them as
  ``trace.draw(...)`` or ``trace.validate_forward_pass(...)``.

* **module build state** - A transient dict that accumulates module hierarchy
  information during the forward pass.  Consumed by ``_build_module_logs``
  (postprocessing step 16) and then cleared.  Initialised via
  ``_init_module_hierarchy_data()``.
"""

import copy
import inspect
import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import difflib
from pathlib import Path
import weakref
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Tuple,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from .._io.streaming import BundleStreamWriter
    from .func_call_location import FuncCallLocation

from .. import _state
from ..backends import BackendName
from .._trace_state import TraceState
from .._io import (
    FieldPolicy,
    TLSPEC_VERSION,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from ..constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from ..captured_run import CapturedRun
from ..ir.trace_build_state import TraceBuildState
from ..intervention.types import (
    MODEL_LOG_FIELD_FORK_POLICY,
    InterventionSpec,
    Relationship,
)
from ..types import ActivationPostfunc, GradientPostfunc
from ..utils.tensor_utils import SaveMode
from ..quantities import Bytes, Duration
from .module import ModuleAccessor
from .param import ParamAccessor
from .interface import (
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
)
from .backward_pass import BackwardPass
from .derived_grad import DerivedGradAccessor
from .field_policy import (
    build_record_field_policy_table,
    default_fill_state_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)
from .grad_fn import GradFn
from .layer import Layer
from .op import Op
from ._state_adapter import state_items, state_restore
from ._trace_accessors import (
    _TRACE_LAYER_ACCESSOR_CACHE,
    _TRACE_OP_ACCESSOR_CACHE,
)

if TYPE_CHECKING:

    class _TraceMixinTypingBase:
        """Typing-only permissive base for mechanically extracted Trace mixins."""

        def __getattr__(self, name: str) -> Any:
            """Return any dynamic Trace attribute for type checking."""

            raise AttributeError(name)

        def __getitem__(self, key: Any) -> Any:
            """Return any dynamic Trace item for type checking."""

            raise KeyError(key)

        def __iter__(self) -> Iterator[Any]:
            """Iterate dynamic Trace entries for type checking."""

            return iter(())

        def __len__(self) -> int:
            """Return dynamic Trace length for type checking."""

            return 0

    class TraceStatsMixin(_TraceMixinTypingBase):
        """Typing-only TraceStatsMixin stand-in."""

    class TraceInterventionMixin(_TraceMixinTypingBase):
        """Typing-only TraceInterventionMixin stand-in."""

    class TraceValidationMixin(_TraceMixinTypingBase):
        """Typing-only TraceValidationMixin stand-in."""

    class TraceExportMixin(_TraceMixinTypingBase):
        """Typing-only TraceExportMixin stand-in."""

    class TraceVisualizationMixin(_TraceMixinTypingBase):
        """Typing-only TraceVisualizationMixin stand-in."""

else:
    from ._trace_export import TraceExportMixin
    from ._trace_intervention import TraceInterventionMixin
    from ._trace_stats import TraceStatsMixin
    from ._trace_validation import TraceValidationMixin
    from ._trace_viz import TraceVisualizationMixin


_MODEL_LOG_DEFAULT_FILL: dict[str, Any] = {
    "trace_label": None,
    "model_label": None,
    "backend": "torch",
    "module_identity_mode": "torch_module",
    "param_source": "native-module",
    "derived_grads": DerivedGradAccessor(),
    "intervention_ready": False,
    "save_arg_templates": False,
    "raw_input": None,
    "_transform": None,
    "save_raw_input": "small",
    "batch_render": "auto",
    "raw_output": None,
    "_output_transform": None,
    "save_raw_output": "small",
    "layer_visualizers": None,
    "save_visualizations": False,
    "_visualizer_dir": None,
    "parent_run": None,
    "_intervention_spec": None,
    "state_history": [],
    "last_run": None,
    "append_history": [],
    "_has_direct_writes": False,
    "_warned_direct_write": False,
    "_warned_mutate_in_place": False,
    "_spec_revision": 0,
    "_out_recipe_revision": 0,
    "_annotation_blobs": None,
    "_append_sequence_id": 0,
    "_last_hook_handle_ids": (),
    "state": TraceState.PRISTINE,
    "model_object_id": None,
    "model_class_qualname": None,
    "param_hash_quick": None,
    "param_hash_full": None,
    "input_object_id": None,
    "input_signature_hash": None,
    "graph_shape_hash": None,
    "module_filter": None,
    "emit_nvtx": False,
    "raise_on_nan": False,
    "keep_orphans": False,
    "annotations": {},
    "observer_spans": [],
    "manual_tensor_connections": [],
    "forward_source_line": None,
    "forward_source_file": None,
    "class_source_file": None,
    "class_source_line": None,
    "init_source_file": None,
    "init_source_line": None,
    "class_docstring": None,
    "init_signature": None,
    "init_docstring": None,
    "forward_signature": None,
    "forward_docstring": None,
    "code_context": [],
    "capture_cache_hit": False,
    "capture_cache_key": None,
    "capture_cache_path": None,
    "recording_kept": True,
    "facet_registry_snapshot": None,
    "_out_dedup_mode": "identity",
    "_out_identity_cache": {},
    "_out_hash_cache": {},
    "_code_context_cache": {},
    "capture_tensor_grad_hooks": True,
    "save_grads": None,
    "inference_only": False,
    "chunked_forward": False,
    "is_appended": False,
    "relationship_evidence": {},
    "replay_frontier": {},
    "_ambiguous_lookup_keys": {},
    "total_gradient_memory": 0,
    "total_backward_memory": 0,
    "saved_gradient_memory": 0,
    "num_saved_layers": 0,
    "num_saved_module_calls": 0,
    "num_saved_grad_fns": 0,
    "num_saved_grad_fn_calls": 0,
    "total_param_gradient_memory": 0,
    "forward_peak_memory": 0,
    "forward_memory_backend": "unknown",
    "_phase_timings": {},
    "_replay_arg_version_data_complete": True,
    "_grad_fn_param_refs": {},
}
# Typed container defaults for every non-Optional container field in
# `MODEL_LOG_FIELD_ORDER`. The blanket ``{field: None}`` base below is wrong for
# these: an absent (legacy/partial-state) container field would restore as
# ``None`` instead of its declared list/dict/set/tuple and then crash on first
# touch (``.values()``/iteration), or -- for ``layer_list``/``layer_logs`` --
# crash inside ``__setstate__`` itself. Plain builtin types are used
# deliberately: they let ``coerce_container_typed_state`` also repair a
# present-but-wrong-typed legacy value (e.g. ``None`` or a ``dict`` where a
# ``list`` is now declared). Fields whose declared type is a genuine
# ``OrderedDict``/``defaultdict`` at runtime restore correctly as a plain dict
# for legacy states (still ``.values()``/``.items()``-usable); fresh captures
# always carry the exact runtime container, so this only affects legacy fill.
_MODEL_LOG_CONTAINER_DEFAULTS: dict[str, Any] = {
    "annotations": {},
    "observer_spans": [],
    "manual_tensor_connections": [],
    "code_context": [],
    "_out_identity_cache": {},
    "_out_hash_cache": {},
    "_code_context_cache": {},
    "_grad_op_nums_to_save": [],
    "input_annotations": {},
    "_source_code_blob": {},
    "state_history": [],
    "append_history": [],
    "_last_hook_handle_ids": (),
    "relationship_evidence": {},
    "replay_frontier": {},
    "layer_list": [],
    "layer_dict_main_keys": {},
    "layer_dict_all_keys": {},
    "layer_logs": {},
    "layer_labels": [],
    "op_labels": [],
    "layer_num_calls": {},
    "by_pass": {},
    "_raw_to_final_layer_labels": {},
    "_raw_to_final_parent_layer_labels": {},
    "_raw_to_final_op_labels": {},
    "_final_to_raw_layer_labels": {},
    "_lookup_keys_to_layer_num_dict": {},
    "_layer_num_to_lookup_keys_dict": {},
    "_ambiguous_lookup_keys": {},
    "input_layers": [],
    "output_layers": [],
    "buffer_layers": [],
    "buffer_num_calls": {},
    "internal_source_ops": [],
    "internal_sink_ops": [],
    "internally_terminated_bool_ops": [],
    "conditional_branch_edges": [],
    "conditional_records": [],
    "conditional_arm_entry_edges": {},
    "conditional_edge_call_indices": {},
    "layers_with_params": {},
    "op_equivalence_classes": {},
    "_orphan_labels": [],
    "_orphan_logs": (),
    "orphan_records": [],
    "_phase_timings": {},
    "grad_fn_logs": {},
    "grad_fn_order": [],
    "backward_pass_logs": {},
    "_grad_fn_param_refs": {},
    "backward_root_grad_fn_object_ids": [],
    "backward_durations": [],
}
_MODEL_LOG_DEFAULT_FILL = {
    **{field_name: None for field_name in MODEL_LOG_FIELD_ORDER},
    **_MODEL_LOG_CONTAINER_DEFAULTS,
    **_MODEL_LOG_DEFAULT_FILL,
}
_MODEL_LOG_DEFAULT_FILL["tlspec_version"] = TLSPEC_VERSION


def _legacy_save_grads_from_state(state: dict[str, Any]) -> Any:
    """Return the canonical ``save_grads`` value for legacy trace state.

    Parameters
    ----------
    state:
        Pickled or tlspec-restored trace state, possibly containing pre-P3
        gradient-save aliases.

    Returns
    -------
    Any
        Canonical ``save_grads`` policy.
    """

    if "save_grads" in state:
        return state["save_grads"]
    if not state.get("save_gradients", False):
        return None
    gradients_to_save = state.get("gradients_to_save", "all")
    return "all" if gradients_to_save is True else gradients_to_save


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


def _init_module_hierarchy_data() -> dict[str, Any]:
    """Create the transient dict used to accumulate module hierarchy data during logging.

    Consumed by ``_build_module_logs`` (step 16) and then cleared.
    """
    return {
        "addresses": [],
        "module_types": {},
        "module_ops": [],
        "module_num_calls": defaultdict(lambda: 1),
        "top_level_modules": [],
        "top_level_module_ops": [],
        "module_children": defaultdict(list),
        "module_pass_children": defaultdict(list),
        "module_nparams": defaultdict(lambda: 0),
        "module_nparams_trainable": defaultdict(lambda: 0),
        "module_nparams_frozen": defaultdict(lambda: 0),
        "module_num_tensors": defaultdict(lambda: 0),
        "module_call_index_tensors": defaultdict(lambda: 0),
        "module_layers": defaultdict(list),
        "module_pass_layers": defaultdict(list),
        "module_output_structures": {},
        "module_layer_argnames": defaultdict(list),
        "module_training_modes": {},
        "module_forward_start_times": {},
        "module_forward_durations": {},
        "module_code_contexts": {},
        "module_call_stacks": {},
    }


@dataclass
class ConditionalEvent:
    """Structured metadata for one conditional event in user source code."""

    id: int
    kind: Literal["if_chain", "ifexp"]
    source_file: str
    function_qualname: str
    function_span: Tuple[int, int]
    if_stmt_span: Tuple[int, int]
    test_span: Tuple[int, int, int, int]
    branch_ranges: Dict[str, Tuple[int, int, int, int]]
    branch_test_spans: Dict[str, Tuple[int, int, int, int]]
    call_depth: int
    parent_conditional_id: Optional[int]
    parent_branch_kind: Optional[str]
    bool_layers: List[str] = field(default_factory=list)


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
        Arm kind: ``"then"``, ``"elif"``, or ``"else"``.
    role:
        Role within the arm: ``"evaluation"`` or ``"body"``.
    """

    conditional_id: str
    arm_index: int
    arm_kind: Literal["then", "elif", "else"]
    role: Literal["evaluation", "body"]


@dataclass
class ConditionalArm:
    """One arm of an if-chain."""

    kind: Literal["then", "elif", "else"]
    terminal_bool_op_label: str | None = None
    bool_value_at_run: bool | None = None
    condition_evaluated: bool = False
    evaluation_entry_edge: tuple[str, str] | None = None
    fired: bool = False
    execution_entry_edge: tuple[str, str] | None = None
    _trace: Any = field(default=None, repr=False, compare=False)
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

        return self._role_ops("evaluation")

    @property
    def execution_ops(self) -> list[str]:
        """Return op labels that execute this arm's body.

        Returns
        -------
        list[str]
            Labels for ops with a body role in this arm.
        """

        return self._role_ops("body")

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

    def _role_ops(self, role: Literal["evaluation", "body"]) -> list[str]:
        """Return op labels participating in this arm with a given role.

        Parameters
        ----------
        role:
            Conditional role to collect.

        Returns
        -------
        list[str]
            Participating op labels.
        """

        if self._trace is None or self._conditional_id is None or self._arm_index is None:
            return []
        labels: list[str] = []
        for op in self._trace.layer_list:
            if any(
                ref.conditional_id == self._conditional_id
                and ref.arm_index == self._arm_index
                and ref.role == role
                for ref in op.in_conditionals or []
            ):
                labels.append(op.layer_label)
        return labels


@dataclass
class Conditional:
    """One if-chain at one source location."""

    id: str
    arms: list[ConditionalArm]
    fired_arm_index: int | None
    fired_arm_kind: Literal["then", "elif", "else"] | None
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


class _CallableList(list[Any]):
    """List that returns a plain list when called.

    This keeps rare report surfaces callable for user ergonomics without adding
    extra callable custom_methods to the Trace method state_history.
    """

    def __call__(self) -> list[Any]:
        """Return a plain-list copy of this report.

        Returns
        -------
        list[Any]
            Plain list containing this report's items.
        """

        return list(self)


def _normalize_conditional_arm_entry_edges(
    value: Any,
) -> dict[tuple[int, str], list[tuple[str, str]]]:
    """Return conditional arm edges in canonical flat-key form.

    Parameters
    ----------
    value:
        Stored conditional arm edge state from current or older portable bundles.

    Returns
    -------
    dict[tuple[int, str], list[tuple[str, str]]]
        Mapping from ``(conditional_id, arm_kind)`` to edge tuples.
    """

    normalized: dict[tuple[int, str], list[tuple[str, str]]] = {}
    if not isinstance(value, Mapping):
        return normalized
    for raw_key, raw_edges in value.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 2 and isinstance(raw_key[1], str):
            normalized[(int(raw_key[0]), raw_key[1])] = list(raw_edges or [])
            continue
        if not isinstance(raw_edges, Mapping):
            continue
        conditional_id = int(raw_key)
        for arm_kind, arm_edges in raw_edges.items():
            normalized[(conditional_id, str(arm_kind))] = list(arm_edges or [])
    return normalized


def _append_conditional_arm_edge(
    conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]],
    key: tuple[int, str],
    edge: tuple[str, str],
) -> None:
    """Append one conditional arm edge, replacing malformed legacy values.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical edge mapping to mutate.
    key:
        ``(conditional_id, arm_kind)`` edge bucket.
    edge:
        ``(parent_label, child_label)`` edge tuple.
    """

    edges = conditional_arm_entry_edges.setdefault(key, [])
    if not isinstance(edges, list):
        edges = []
        conditional_arm_entry_edges[key] = edges
    edges.append(edge)


@dataclass(init=False, repr=False, eq=False)
class Trace(
    TraceStatsMixin,
    TraceInterventionMixin,
    TraceValidationMixin,
    TraceExportMixin,
    TraceVisualizationMixin,
    CapturedRun,
):
    """Top-level container for a logged forward pass.

    Serves double duty: during the forward pass it accumulates raw tensor
    metadata in transient raw graph state; after postprocessing (``_tracing_finished=True``)
    it presents a clean, user-facing view via ``layer_list``, ``layer_dict_all_keys``,
    ``layer_logs``, ``modules``, ``params``, and ``buffers``.

    Supports ``len()``, iteration, and flexible ``__getitem__`` lookup by
    integer index, layer label, module address, or substring.
    """

    def _ensure_build_state(self) -> TraceBuildState:
        """Return the transient capture/postprocess build state.

        Returns
        -------
        TraceBuildState
            Private state holder used only while capture or postprocessing is active.
        """

        build_state = self.__dict__.get("_build_state")
        if not isinstance(build_state, TraceBuildState):
            build_state = TraceBuildState()
            build_state.module_build_data = _init_module_hierarchy_data()
            self.__dict__["_build_state"] = build_state
        elif not build_state.module_build_data:
            build_state.module_build_data = _init_module_hierarchy_data()
        return build_state

    @staticmethod
    def _build_state_attr_map() -> dict[str, str]:
        """Map legacy transient attribute names to build-state field names."""

        return {
            "_raw" + "_layer_dict": "raw_layer_dict",
            "_raw" + "_layer_labels_list": "raw_layer_labels_list",
            "_layer" + "_counter": "layer_counter",
            "_raw" + "_layer_type_counter": "raw_layer_type_counter",
            "_current" + "_func_barcode": "current_func_barcode",
            "_mod" + "_call_index": "mod_call_index",
            "_mod" + "_call_labels": "mod_call_labels",
            "_mod" + "_entered": "mod_entered",
            "_mod" + "_exited": "mod_exited",
            "_module" + "_build_data": "module_build_data",
            "_module" + "_metadata": "module_metadata",
            "_module" + "_forward_args": "module_forward_args",
            "_grad" + "_fn_strong_refs": "grad_fn_strong_refs",
            "_in" + "_exhaustive_pass": "in_exhaustive_pass",
            "_module" + "_containment_engine": "module_containment_engine",
            "_exhaustive" + "_module_stack": "exhaustive_module_stack",
            "_input" + "_tensor_addresses": "input_tensor_addresses",
        }

    def __getattr__(self, name: str) -> Any:
        """Route transient capture attributes through private build state."""

        state_field = self._build_state_attr_map().get(name)
        if state_field is None:
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
        if (
            name != "_in_exhaustive_pass"
            and "_build_state" not in self.__dict__
            and self.__dict__.get("_tracing_finished", True)
        ):
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
        return getattr(self._ensure_build_state(), state_field)

    def __setattr__(self, name: str, value: Any) -> None:
        """Route transient capture attribute writes through private build state."""

        state_field = self._build_state_attr_map().get(name)
        if state_field is None:
            super().__setattr__(name, value)
            return
        if (
            name != "_in_exhaustive_pass"
            and "_build_state" not in self.__dict__
            and self.__dict__.get("_tracing_finished", True)
        ):
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
        setattr(self._ensure_build_state(), state_field, value)

    def __delattr__(self, name: str) -> None:
        """Delete transient capture attributes from private build state."""

        state_field = self._build_state_attr_map().get(name)
        if state_field is None:
            super().__delattr__(name)
            return
        build_state = self.__dict__.get("_build_state")
        if build_state is not None and hasattr(build_state, state_field):
            default_state = TraceBuildState()
            setattr(build_state, state_field, getattr(default_state, state_field))

    backend: BackendName
    backend_runtime_config: dict[str, Any] | None
    backend_runtime_device_summary: dict[str, Any] | None
    backend_runtime_version: str | None
    module_identity_mode: Literal["torch_module", "pytree_module", "function_root", "object_module"]
    param_source: Literal["native-module", "pytree-derived", "none"]
    state: TraceState
    tlspec_version: int
    annotations: Dict[str, Any]
    input_preprocessor: ResolvedPreprocessing | None
    output_postprocessor: ResolvedPostprocessing | None
    output_id2label: dict[int, str] | None
    output_num_classes: int | None
    input_object_id: int | None
    model_object_id: int | None
    input_signature_hash: str | None
    state_history: list[Any]
    replay_frontier: dict[str, torch.Tensor]
    backward_ready: bool
    inference_only: bool
    chunked_forward: bool
    profile_enabled: bool
    save_arg_templates: bool
    op_equivalence_classes: Dict[str, set[str]]
    last_run: Any | None
    capture_start_time: float
    capture_end_time: float
    backward_root_grad_fn_object_ids: list[int]
    backward_pass_logs: Dict[int, BackwardPass]
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
    _annotation_revision: int
    _last_sibling_ordering_decision: Any

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "trace_label": FieldPolicy.KEEP,
        "model_class_name": FieldPolicy.KEEP,
        "model_label": FieldPolicy.KEEP,
        "backend": FieldPolicy.KEEP,
        "backend_runtime_config": FieldPolicy.KEEP,
        "backend_runtime_device_summary": FieldPolicy.KEEP,
        "backend_runtime_version": FieldPolicy.KEEP,
        "_paddle_capture_depth": FieldPolicy.DROP,
        "_paddle_op_captures": FieldPolicy.DROP,
        "_paddle_alias_annotations": FieldPolicy.DROP,
        "_paddle_capture_gap_markers": FieldPolicy.DROP,
        "_tf_source_records": FieldPolicy.DROP,
        "_tf_unresolved_producers": FieldPolicy.DROP,
        "_tf_init_op_labels": FieldPolicy.DROP,
        "_tf_op_type_counts": FieldPolicy.DROP,
        "_tf_op_captures": FieldPolicy.DROP,
        "_tf_validation_result": FieldPolicy.DROP,
        "module_identity_mode": FieldPolicy.KEEP,
        "param_source": FieldPolicy.KEEP,
        "derived_grads": FieldPolicy.KEEP,
        "num_context_lines": FieldPolicy.KEEP,
        "_optimizer": FieldPolicy.DROP,
        "tlspec_version": FieldPolicy.KEEP,
        "_tracing_finished": FieldPolicy.KEEP,
        "capture_mode": FieldPolicy.KEEP,
        "halted": FieldPolicy.KEEP,
        "halt_reason": FieldPolicy.KEEP,
        "halt_frontier": FieldPolicy.KEEP,
        "_layers_logged": FieldPolicy.KEEP,
        "_layers_saved": FieldPolicy.KEEP,
        "keep_orphans": FieldPolicy.KEEP,
        "intervention_ready": FieldPolicy.KEEP,
        "save_arg_templates": FieldPolicy.KEEP,
        "raw_input": FieldPolicy.KEEP,
        "input_preprocessor": FieldPolicy.KEEP,
        "_transform": FieldPolicy.DROP,
        "transform_repr": FieldPolicy.KEEP,
        "save_raw_input": FieldPolicy.KEEP,
        "batch_render": FieldPolicy.KEEP,
        "raw_output": FieldPolicy.KEEP,
        "decoded_output": FieldPolicy.KEEP,
        "output_postprocessor": FieldPolicy.KEEP,
        "output_id2label": FieldPolicy.KEEP,
        "output_num_classes": FieldPolicy.KEEP,
        "_output_transform": FieldPolicy.DROP,
        "save_raw_output": FieldPolicy.KEEP,
        "layer_visualizers": FieldPolicy.DROP,
        "save_visualizations": FieldPolicy.KEEP,
        "_visualizer_dir": FieldPolicy.DROP,
        "activation_transform": FieldPolicy.DROP,
        "_activation_transform_repr": FieldPolicy.KEEP,
        "save_raw_activations": FieldPolicy.KEEP,
        "save_mode": FieldPolicy.KEEP,
        "input_annotations": FieldPolicy.KEEP,
        "_source_code_blob": FieldPolicy.KEEP,
        "_source_model_ref": FieldPolicy.DROP,
        "parent_run": FieldPolicy.DROP,
        "model_object_id": FieldPolicy.KEEP,
        "model_class_qualname": FieldPolicy.KEEP,
        "param_hash_quick": FieldPolicy.KEEP,
        "param_hash_full": FieldPolicy.KEEP,
        "input_object_id": FieldPolicy.KEEP,
        "input_signature_hash": FieldPolicy.KEEP,
        "random_seed": FieldPolicy.KEEP,
        "output_device": FieldPolicy.KEEP,
        "detach_saved_activations": FieldPolicy.KEEP,
        "backward_ready": FieldPolicy.DROP,
        "inference_only": FieldPolicy.KEEP,
        "chunked_forward": FieldPolicy.KEEP,
        "module_filter": FieldPolicy.DROP,
        "emit_nvtx": FieldPolicy.KEEP,
        "raise_on_nan": FieldPolicy.KEEP,
        "annotations": FieldPolicy.KEEP,
        "observer_spans": FieldPolicy.KEEP,
        "manual_tensor_connections": FieldPolicy.KEEP,
        "forward_source_file": FieldPolicy.KEEP,
        "forward_source_line": FieldPolicy.KEEP,
        "class_source_file": FieldPolicy.KEEP,
        "class_source_line": FieldPolicy.KEEP,
        "init_source_file": FieldPolicy.KEEP,
        "init_source_line": FieldPolicy.KEEP,
        "class_docstring": FieldPolicy.KEEP,
        "init_signature": FieldPolicy.KEEP,
        "init_docstring": FieldPolicy.KEEP,
        "forward_signature": FieldPolicy.KEEP,
        "forward_docstring": FieldPolicy.KEEP,
        "code_context": FieldPolicy.KEEP,
        "capture_cache_hit": FieldPolicy.KEEP,
        "capture_cache_key": FieldPolicy.KEEP,
        "capture_cache_path": FieldPolicy.KEEP,
        "recording_kept": FieldPolicy.KEEP,
        "facet_registry_snapshot": FieldPolicy.DROP,
        "_out_dedup_mode": FieldPolicy.DROP,
        "_out_identity_cache": FieldPolicy.DROP,
        "_out_hash_cache": FieldPolicy.DROP,
        "_code_context_cache": FieldPolicy.DROP,
        "save_arg_values": FieldPolicy.KEEP,
        "save_grads": FieldPolicy.KEEP,
        "capture_tensor_grad_hooks": FieldPolicy.KEEP,
        "_grad_op_nums_to_save": FieldPolicy.KEEP,
        "grad_transform": FieldPolicy.DROP,
        "grad_transform_repr": FieldPolicy.KEEP,
        "save_raw_gradients": FieldPolicy.KEEP,
        "save_code_context": FieldPolicy.KEEP,
        "save_rng_states": FieldPolicy.KEEP,
        "recurrence_detection": FieldPolicy.KEEP,
        "verbose": FieldPolicy.KEEP,
        "profile_enabled": FieldPolicy.KEEP,
        "has_gradients": FieldPolicy.KEEP,
        "mark_layer_depths": FieldPolicy.KEEP,
        "graph_shape_hash": FieldPolicy.KEEP,
        "_intervention_spec": FieldPolicy.DROP,
        "state_history": FieldPolicy.KEEP,
        "last_run": FieldPolicy.DROP,
        "append_history": FieldPolicy.KEEP,
        "_has_direct_writes": FieldPolicy.KEEP,
        "_warned_direct_write": FieldPolicy.DROP,
        "_warned_mutate_in_place": FieldPolicy.DROP,
        "_spec_revision": FieldPolicy.KEEP,
        "_out_recipe_revision": FieldPolicy.KEEP,
        "_append_sequence_id": FieldPolicy.KEEP,
        "_last_hook_handle_ids": FieldPolicy.DROP,
        "_predicate_save_options": FieldPolicy.DROP,
        "_predicate_history_size": FieldPolicy.DROP,
        "_predicate_history": FieldPolicy.DROP,
        "_predicate_all_contexts": FieldPolicy.DROP,
        "_predicate_lookback": FieldPolicy.DROP,
        "_predicate_lookback_payload_policy": FieldPolicy.DROP,
        "_capture_config": FieldPolicy.DROP,
        "_stop_directive": FieldPolicy.DROP,
        "_halt_returns_partial_trace": FieldPolicy.DROP,
        "_predicate_save_decisions": FieldPolicy.DROP,
        "_predicate_contexts_by_label": FieldPolicy.DROP,
        "_predicate_current_contexts": FieldPolicy.DROP,
        "_predicate_lookback_candidates": FieldPolicy.DROP,
        "_postprocessing_active": FieldPolicy.DROP,
        "_raw_transform_escape_detected": FieldPolicy.DROP,
        "_raw_event_shape_hash": FieldPolicy.DROP,
        "_replay_arg_version_data_complete": FieldPolicy.KEEP,
        "state": FieldPolicy.KEEP,
        "is_appended": FieldPolicy.KEEP,
        "relationship_evidence": FieldPolicy.KEEP,
        "replay_frontier": FieldPolicy.DROP,
        "_output_container_specs_by_raw_label": FieldPolicy.DROP,
        "layer_list": FieldPolicy.KEEP,
        "layer_dict_main_keys": FieldPolicy.KEEP,
        "layer_dict_all_keys": FieldPolicy.KEEP,
        "layer_logs": FieldPolicy.KEEP,
        "layer_labels": FieldPolicy.KEEP,
        "op_labels": FieldPolicy.KEEP,
        "layer_num_calls": FieldPolicy.KEEP,
        "by_pass": FieldPolicy.KEEP,
        "_layer_nums_to_save": FieldPolicy.KEEP,
        "num_ops": FieldPolicy.KEEP,
        "num_modules": FieldPolicy.DROP,
        "_raw_to_final_layer_labels": FieldPolicy.KEEP,
        "_raw_to_final_parent_layer_labels": FieldPolicy.KEEP,
        "_raw_to_final_op_labels": FieldPolicy.KEEP,
        "_final_to_raw_layer_labels": FieldPolicy.KEEP,
        "_lookup_keys_to_layer_num_dict": FieldPolicy.KEEP,
        "_layer_num_to_lookup_keys_dict": FieldPolicy.KEEP,
        "_ambiguous_lookup_keys": FieldPolicy.KEEP,
        "input_layers": FieldPolicy.KEEP,
        "output_layers": FieldPolicy.KEEP,
        "input_structure": FieldPolicy.BLOB_RECURSIVE,
        "_containers": FieldPolicy.BLOB_RECURSIVE,
        "_annotation_blobs": FieldPolicy.BLOB_RECURSIVE,
        "buffer_layers": FieldPolicy.KEEP,
        "buffer_num_calls": FieldPolicy.KEEP,
        "_buffer_accessor": FieldPolicy.DROP,
        "_buffer_write_events": FieldPolicy.DROP,
        "_buffer_write_tracker": FieldPolicy.DROP,
        "_buffer_initial_values": FieldPolicy.BLOB_RECURSIVE,
        "internal_source_ops": FieldPolicy.KEEP,
        "internal_sink_ops": FieldPolicy.KEEP,
        "internally_terminated_bool_ops": FieldPolicy.KEEP,
        "conditional_branch_edges": FieldPolicy.KEEP,
        "conditional_records": FieldPolicy.KEEP,
        "conditional_arm_entry_edges": FieldPolicy.KEEP,
        "conditional_edge_call_indices": FieldPolicy.KEEP,
        "conditionals": FieldPolicy.KEEP,
        "_orphan_labels": FieldPolicy.KEEP,
        "_orphan_logs": FieldPolicy.KEEP,
        "orphan_records": FieldPolicy.BLOB_RECURSIVE,
        "_saved_grad_labels": FieldPolicy.DROP,
        "layers_with_params": FieldPolicy.KEEP,
        "ops_with_params": FieldPolicy.KEEP,
        "op_equivalence_classes": FieldPolicy.KEEP,
        "total_activation_memory": FieldPolicy.KEEP,
        "total_gradient_memory": FieldPolicy.KEEP,
        "total_backward_memory": FieldPolicy.KEEP,
        "total_autograd_memory": FieldPolicy.KEEP,
        "num_saved_ops": FieldPolicy.KEEP,
        "saved_activation_memory": FieldPolicy.KEEP,
        "saved_gradient_memory": FieldPolicy.KEEP,
        "num_saved_layers": FieldPolicy.KEEP,
        "num_saved_module_calls": FieldPolicy.KEEP,
        "num_saved_grad_fns": FieldPolicy.KEEP,
        "num_saved_grad_fn_calls": FieldPolicy.KEEP,
        "param_logs": FieldPolicy.KEEP,
        "num_param_tensors": FieldPolicy.KEEP,
        "num_layers_with_params": FieldPolicy.KEEP,
        "num_params": FieldPolicy.KEEP,
        "num_params_trainable": FieldPolicy.KEEP,
        "num_params_frozen": FieldPolicy.KEEP,
        "total_param_memory": FieldPolicy.KEEP,
        "total_param_gradient_memory": FieldPolicy.KEEP,
        "forward_peak_memory": FieldPolicy.KEEP,
        "forward_memory_backend": FieldPolicy.KEEP,
        "_raw" + "_layer_dict": FieldPolicy.DROP,
        "_raw" + "_layer_labels_list": FieldPolicy.DROP,
        "_layer" + "_counter": FieldPolicy.DROP,
        "_raw" + "_layer_type_counter": FieldPolicy.DROP,
        "_current" + "_func_barcode": FieldPolicy.DROP,
        "_mod" + "_call_index": FieldPolicy.DROP,
        "_mod" + "_call_labels": FieldPolicy.DROP,
        "_mod" + "_entered": FieldPolicy.DROP,
        "_mod" + "_exited": FieldPolicy.DROP,
        "_module" + "_build_data": FieldPolicy.DROP,
        "_module" + "_metadata": FieldPolicy.DROP,
        "_module" + "_forward_args": FieldPolicy.DROP,
        "_grad" + "_fn_strong_refs": FieldPolicy.DROP,
        "_in" + "_exhaustive_pass": FieldPolicy.DROP,
        "_module" + "_containment_engine": FieldPolicy.DROP,
        "_exhaustive" + "_module_stack": FieldPolicy.DROP,
        "_module_logs": FieldPolicy.DROP,
        "_param_logs_by_module": FieldPolicy.DROP,
        "_build_state": FieldPolicy.DROP,
        "_pre_forward_rng_states": FieldPolicy.DROP,
        "_mlx_saved_payloads": FieldPolicy.DROP,
        "_mlx_capture_depth": FieldPolicy.DROP,
        "_mlx_type_counts": FieldPolicy.DROP,
        "_out_writer": FieldPolicy.DROP,
        "_keep_outs_in_memory": FieldPolicy.DROP,
        "_grad_stream_retain_in_memory": FieldPolicy.DROP,
        "_defer_streaming_bundle_finalization": FieldPolicy.DROP,
        "_out_sink": FieldPolicy.DROP,
        "_capture_events": FieldPolicy.DROP,
        "_tl_backward_hooked_tensor_keys": FieldPolicy.DROP,
        "_active_backward_pass_index": FieldPolicy.DROP,
        "_backward_roots_by_pass": FieldPolicy.DROP,
        "_backward_projection_event_count": FieldPolicy.DROP,
        "_implicit_backward_pass_open": FieldPolicy.DROP,
        "_warned_implicit_backward_pass": FieldPolicy.DROP,
        "_tl_backward_triggers_disarmed": FieldPolicy.DROP,
        "capture_start_time": FieldPolicy.KEEP,
        "capture_end_time": FieldPolicy.KEEP,
        "_phase_timings": FieldPolicy.KEEP,
        "setup_duration": FieldPolicy.KEEP,
        "forward_duration": FieldPolicy.KEEP,
        "cleanup_duration": FieldPolicy.KEEP,
        "func_calls_duration": FieldPolicy.KEEP,
        "has_backward_pass": FieldPolicy.KEEP,
        "grad_fn_logs": FieldPolicy.KEEP,
        "grad_fn_order": FieldPolicy.KEEP,
        "backward_pass_logs": FieldPolicy.KEEP,
        "_grad_fn_param_refs": FieldPolicy.KEEP,
        "_grad_fn_param_refs_by_object_id": FieldPolicy.DROP,
        "_param_log_by_pid": FieldPolicy.DROP,
        "backward_root_grad_fn_object_ids": FieldPolicy.KEEP,
        "backward_durations": FieldPolicy.KEEP,
        "num_backward_passes": FieldPolicy.KEEP,
        "backward_peak_memory": FieldPolicy.KEEP,
        "backward_memory_backend": FieldPolicy.KEEP,
        "_backward_gradfn_refs": FieldPolicy.DROP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        MODEL_LOG_FIELD_ORDER,
        PORTABLE_STATE_SPEC,
        fork_policy=MODEL_LOG_FIELD_FORK_POLICY,
        default_fill_state=_MODEL_LOG_DEFAULT_FILL,
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)

    def __init__(
        self,
        model_class_name: str,
        output_device: str = "same",
        activation_transform: Optional[ActivationPostfunc] = None,
        grad_transform: Optional[GradientPostfunc] = None,
        save_raw_activations: bool = True,
        save_raw_gradients: bool = True,
        save_mode: SaveMode = "copy",
        keep_orphans: bool = False,
        save_arg_values: bool = False,
        save_grads: Any = None,
        capture_tensor_grad_hooks: bool = True,
        detach_saved_activations: bool = False,
        mark_layer_depths: bool = True,
        num_context_lines: int = 7,
        optimizer: torch.optim.Optimizer | None = None,
        save_code_context: bool = False,
        save_rng_states: bool = False,
        recurrence_detection: bool = True,
        verbose: bool = False,
        backward_ready: bool = False,
        inference_only: bool = False,
        chunked_forward: bool = False,
        module_filter: Callable[[Any], bool] | None = None,
        emit_nvtx: bool = False,
        facet_registry_snapshot: Any | None = None,
        transform: Callable[[Any], Any] | None = None,
        raw_input: Any | None = None,
        save_raw_input: str | bool = "small",
        batch_render: str = "auto",
        output_transform: Callable[[Any], Any] | None = None,
        raw_output: Any | None = None,
        save_raw_output: str | bool = "small",
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
            save_raw_gradients: Whether raw grads are retained when a transform is set.
            save_mode: Tensor retention mode for saved activation and gradient payloads.
            keep_orphans: If True, orphan island ops remain in raw metadata and
                are exposed via ``trace.orphans``.
            save_arg_values: Whether to deep-copy each operation's input arguments.
            save_grads: Which backward gradients should be retained. ``True``
                saves all gradients, ``False``/``None`` saves no payloads, and
                selectors/predicates save matching gradient records.
            capture_tensor_grad_hooks: Whether forward tensors receive
                tensor-level backward hooks for implicit backward events and
                per-op gradient payloads. Grad-fn registration remains enabled.
            detach_saved_activations: Whether to detach saved tensors from the autograd graph.
            mark_layer_depths: Whether to compute BFS distances from
                inputs/outputs for each layer.
            num_context_lines: Number of source-code context lines to capture
                around each function call (used by FuncCallLocation).
            optimizer: Optional torch optimizer, used to annotate which params
                have optimizers attached.
            verbose: If True, print timed progress messages at each major pipeline stage.
            backward_ready: Session-time flag for training-compatible out retention.
                Portable bundle load restores the default ``False`` value.
            inference_only: Whether the forward was captured under ``torch.no_grad()``.
            chunked_forward: Whether the trace was assembled from forward chunks.
            emit_nvtx: Whether decorated torch operations should emit NVTX ranges
                around captured torch calls. This is a profiling aid for CUDA/Nsight
                workflows and does not change graph construction or saved payloads.
            facet_registry_snapshot: Immutable facet recipe snapshot captured for
                this trace.
            transform: Optional callable used to convert raw user input into
                model-ready input.
            raw_input: Original user input before ``transform`` was applied.
            save_raw_input: Portable save policy for ``raw_input``.
            batch_render: Raw-input batch rendering policy for visualization.
            output_transform: Optional callable used to convert model output into
                human-readable metadata.
            raw_output: Human-readable model output after ``output_transform``.
            save_raw_output: Portable save policy for ``raw_output``.
            layer_visualizers: Optional mapping of selectors to visualizer callables.
            save_visualizations: Whether rendered visualizations should persist in bundles.
        """
        # Callables are effectively immutable - deepcopy is unnecessary.

        # General info
        self.trace_label: str | None = None
        self.model_class_name = model_class_name
        self.model_label = model_class_name
        self.backend: BackendName = "torch"
        self.backend_runtime_config: dict[str, Any] | None = None
        self.backend_runtime_device_summary: dict[str, Any] | None = None
        self.backend_runtime_version: str | None = None
        self.module_identity_mode: Literal[
            "torch_module", "pytree_module", "function_root", "object_module"
        ] = "torch_module"
        self.param_source: Literal["native-module", "pytree-derived", "none"] = "native-module"
        self.derived_grads = DerivedGradAccessor()
        self.num_context_lines = num_context_lines
        self._optimizer = optimizer
        self.tlspec_version = TLSPEC_VERSION
        # _tracing_finished is the master behavioural switch: False during logging,
        # True after postprocessing.  Many custom_methods (len, getitem, str, iter)
        # branch on this flag to choose raw-barcode vs final-label access.
        # It intentionally persists across the fast pass so fast-path
        # postprocessing can use the exhaustive pass's lookup dicts.
        self._tracing_finished = False
        # "exhaustive" captures all metadata; "fast" reuses exhaustive-pass
        # structure, only re-capturing tensor contents.
        self.capture_mode: Literal["exhaustive", "fast", "predicate"] = "exhaustive"
        self.halted = False
        self.halt_reason: str | None = None
        self.halt_frontier: str | None = None
        self._layers_logged = False
        self._layers_saved = False
        self.keep_orphans = keep_orphans
        self.intervention_ready = False
        self.save_arg_templates = False
        self.raw_input = raw_input
        self.input_preprocessor: ResolvedPreprocessing | None = None
        self._transform = transform
        self.transform_repr = repr(transform) if transform is not None else None
        self.save_raw_input = save_raw_input
        self.batch_render = batch_render
        self.raw_output = raw_output
        self.decoded_output: Any | None = None
        self.output_postprocessor: ResolvedPostprocessing | None = None
        self.output_id2label: dict[int, str] | None = None
        self.output_num_classes: int | None = None
        self._output_transform = output_transform
        self.save_raw_output = save_raw_output
        self.layer_visualizers = layer_visualizers
        self.save_visualizations = save_visualizations
        self._visualizer_dir: str | None = None
        self.activation_transform = activation_transform
        self._activation_transform_repr = (
            repr(activation_transform) if activation_transform is not None else None
        )
        self.save_raw_activations = save_raw_activations
        self.input_annotations: Dict[str, Any] = {}
        self.grad_transform = grad_transform
        self.grad_transform_repr = repr(grad_transform) if grad_transform is not None else None
        self.save_raw_gradients = save_raw_gradients
        self.save_mode = save_mode
        self._source_code_blob: dict[str, str] = {}
        self._source_model_ref: weakref.ReferenceType[nn.Module] | None = None
        self.parent_run: weakref.ReferenceType["Trace"] | None = None
        self.model_object_id: int | None = None
        self.model_class_qualname: str | None = None
        self.param_hash_quick: str | None = None
        self.param_hash_full: str | None = None
        self.input_object_id: int | None = None
        self.input_signature_hash: str | None = None
        self.random_seed = None
        self.output_device = output_device
        self.detach_saved_activations = detach_saved_activations
        self.backward_ready = backward_ready
        self.inference_only = inference_only
        self.chunked_forward = chunked_forward
        self.module_filter = module_filter
        self.emit_nvtx = emit_nvtx
        self.facet_registry_snapshot = facet_registry_snapshot
        self.raise_on_nan: bool = False
        self.annotations: Dict[str, Any] = {}
        self.code_context: list["FuncCallLocation"] = []
        self.manual_tensor_connections: List[Tuple[str, str]] = []
        self.forward_source_file: str | None = None
        self.forward_source_line: int | None = None
        self.class_source_file: str | None = None
        self.class_source_line: int | None = None
        self.init_source_file: str | None = None
        self.init_source_line: int | None = None
        self.class_docstring: str | None = None
        self.init_signature: str | None = None
        self.init_docstring: str | None = None
        self.forward_signature: str | None = None
        self.forward_docstring: str | None = None
        self.capture_cache_hit: bool = False
        self.capture_cache_key: str | None = None
        self.capture_cache_path: str | None = None
        self.recording_kept: bool = True
        self._out_dedup_mode: Literal["identity", "content", "none"] = "identity"
        self._out_identity_cache: Dict[int, Tuple[torch.Tensor, str, torch.Tensor, int | None]] = {}
        self._out_hash_cache: Dict[str, Tuple[str, torch.Tensor]] = {}
        self._code_context_cache: dict[Any, tuple[Any, ...]] = {}
        self._halt_returns_partial_trace = False
        self._replay_arg_version_data_complete = True
        self.save_arg_values = save_arg_values
        self.save_grads = "all" if save_grads is True else save_grads
        self.capture_tensor_grad_hooks = capture_tensor_grad_hooks
        self.save_code_context = save_code_context
        self.save_rng_states = save_rng_states
        self.recurrence_detection = recurrence_detection
        self.verbose = verbose
        self.profile_enabled = False
        self.has_gradients = False
        self.mark_layer_depths = mark_layer_depths
        self.graph_shape_hash: str | None = None
        self._intervention_spec: InterventionSpec | None = InterventionSpec()
        self.state_history: list[Any] = []
        self.observer_spans: list[dict[str, Any]] = list(_state._active_record_spans)
        self.last_run: Any | None = None
        self.append_history: list[dict[str, Any]] = []
        self._has_direct_writes = False
        self._warned_direct_write = False
        self._warned_mutate_in_place = False
        self._raw_transform_escape_detected = False
        self._spec_revision = 0
        self._out_recipe_revision = 0
        self._append_sequence_id = 0
        self._last_hook_handle_ids: tuple[str, ...] = ()
        self.state = TraceState.PRISTINE
        self.is_appended = False
        self.relationship_evidence: dict[str, Relationship] = {
            "model": Relationship.UNKNOWN,
            "weights": Relationship.UNKNOWN,
            "input": Relationship.UNKNOWN,
            "graph": Relationship.UNKNOWN,
        }
        self.replay_frontier: dict[str, torch.Tensor] = {}
        self._output_container_specs_by_raw_label: dict[str, Any] = {}
        self._out_writer: Optional["BundleStreamWriter"] = None
        self._keep_outs_in_memory: bool = True
        self._defer_streaming_bundle_finalization: bool = False
        self._out_sink: Optional[Callable[[str, torch.Tensor], None]] = None
        # Model structure info (computed @properties: is_recurrent,
        # max_layer_op_count, is_branching, has_conditional_branching)

        # Tensor Tracking - post-processed (populated after _tracing_finished=True):
        self.layer_list: List[Op] = []  # ordered list of all layer ops
        self.layer_dict_main_keys: Dict[str, Op] = OrderedDict()  # primary label -> entry
        self.layer_dict_all_keys: Dict[str, Op] = OrderedDict()  # all lookup keys -> entry
        self.layer_logs: Dict[str, Layer] = OrderedDict()  # no-pass label -> aggregate Layer
        self.op_labels: List[str] = []  # pass-qualified labels (e.g. "conv2d_1_1:1")
        self.layer_labels: List[str] = []  # pass-stripped labels (e.g. "conv2d_1_1")
        self.layer_num_calls: Dict[str, int] = OrderedDict()  # no-pass label -> pass count
        self.by_pass: dict[int, list[int]] = {}
        self._layer_nums_to_save: List[int] = []  # ordinal positions of layers to save
        self._grad_op_nums_to_save: List[int] | str = []
        self.num_ops: int = 0  # total operations after postprocessing

        # Mapping between raw barcodes and final human-readable labels
        # (populated during postprocessing's label-assignment step):
        self._raw_to_final_layer_labels: Dict[str, str] = {}
        self._raw_to_final_parent_layer_labels: Dict[str, str] = {}
        self._raw_to_final_op_labels: Dict[str, str] = {}
        self._final_to_raw_layer_labels: Dict[str, str] = {}
        self._lookup_keys_to_layer_num_dict: Dict[str, int] = {}
        self._layer_num_to_lookup_keys_dict: Dict[int, List[str]] = defaultdict(list)
        self._ambiguous_lookup_keys: Dict[str, List[int]] = {}

        # Special Layers:
        self.input_layers: List[str] = []
        self.output_layers: List[str] = []
        self._annotation_blobs: dict[str, Any] | None = None
        self.buffer_layers: List[str] = []
        self.buffer_num_calls: Dict[str, int] = {}
        self._buffer_accessor = None
        self._buffer_write_events: list[Any] = []
        self._buffer_write_tracker: Any | None = None
        self._buffer_initial_values: Dict[str, Any] = {}
        self.internal_source_ops: List[str] = []
        self.internal_sink_ops: List[str] = []
        self.internally_terminated_bool_ops: List[str] = []
        self.conditional_branch_edges: List[Tuple[str, str]] = []
        self.conditional_records: List[ConditionalEvent] = []
        self.conditional_arm_entry_edges: Dict[Tuple[int, str], List[Tuple[str, str]]] = {}
        self.conditional_edge_call_indices: Dict[Tuple[str, str, int, str], List[int]] = {}
        self.conditionals = ConditionalAccessor()
        self._orphan_labels: List[str] = []
        self._orphan_logs: tuple[Op, ...] = ()
        self.orphan_records: list[dict[str, Any]] = []
        self._saved_grad_labels: set[str] = set()
        self.layers_with_params: Dict[str, List[Any]] = defaultdict(list)
        # Maps equivalence_class -> set of layer labels that share
        # that equivalence type (populated by loop_detection.py).
        self.op_equivalence_classes: Dict[str, set[str]] = defaultdict(set)

        # Aggregate tensor statistics (computed during postprocessing):
        self.total_activation_memory: Bytes = Bytes(0)
        self.total_gradient_memory: Bytes = Bytes(0)
        self.total_backward_memory: Bytes = Bytes(0)
        self.total_autograd_memory: Bytes | None = None
        self.num_saved_ops: int = 0  # layers with has_saved_activation=True
        self.saved_activation_memory: Bytes = Bytes(0)
        self.saved_gradient_memory: Bytes = Bytes(0)
        self.num_saved_layers: int = 0
        self.num_saved_module_calls: int = 0
        self.num_saved_grad_fns: int = 0
        self.num_saved_grad_fn_calls: int = 0

        # Param info:
        self.param_logs: "ParamAccessor" = ParamAccessor({})
        self.num_param_tensors: int = 0
        self.num_layers_with_params: int = 0
        self.num_params: int = 0
        self.num_params_trainable: int = 0
        self.num_params_frozen: int = 0
        self.total_param_memory: Bytes = Bytes(0)
        self.total_param_gradient_memory: Bytes = Bytes(0)
        self.forward_peak_memory: Bytes = Bytes(0)
        self.forward_memory_backend: str = "unknown"

        # Structured module info:
        self._module_logs: ModuleAccessor = ModuleAccessor({})

        # Time elapsed:
        self.capture_start_time: float = 0
        self.capture_end_time: float = 0
        self._phase_timings: dict[str, dict[str, float | int]] = {}
        self.setup_duration: Duration = Duration(0)
        self.forward_duration: Duration = Duration(0)
        self.cleanup_duration: Duration = Duration(0)
        self.func_calls_duration: Duration = Duration(0)
        self.has_backward_pass: bool = False
        self.grad_fn_logs: Dict[int, GradFn] = OrderedDict()
        self.grad_fn_order: List[int] = []
        self.backward_pass_logs: Dict[int, BackwardPass] = OrderedDict()
        self._grad_fn_param_refs: dict[str, str] = {}
        self._param_log_by_pid: dict[int, str] = {}
        self.backward_root_grad_fn_object_ids: list[int] = []
        self.backward_durations: list[Duration] = []
        self.num_backward_passes: int = 0
        self.backward_peak_memory: Bytes = Bytes(0)
        self.backward_memory_backend: str = "unknown"
        _state._register_log(self)

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************

    def __len__(self) -> int:
        """Number of layer-pass entries. Uses final list after postprocessing, raw dict during logging."""
        if self._tracing_finished:
            return len(self.layer_list)
        else:
            return len(getattr(self, "_raw" + "_layer_dict"))

    def __getitem__(self, ix: Any) -> Any:
        """Returns an object logging a model layer given an index. If the pass is finished,
        it'll do this intelligently; if not, it simply queries based on the layer's raw barcode.

        Args:
            ix: desired index

        Returns:
            Tensor log entry object with info about specified layer.
        """
        if self._tracing_finished:
            return _getitem_after_pass(self, ix)
        else:
            return _getitem_during_pass(self, ix)

    def find_sites(self, query: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Return a table of intervention sites matching a query.

        Parameters
        ----------
        query:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        SiteTable
            Ordered table of matching layer-pass records.
        """

        from ..intervention.resolver import find_sites

        return find_sites(self, query, strict=strict, max_fanout=max_fanout)

    def resolve_sites(self, query: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
        """Resolve intervention sites matching a query.

        Parameters
        ----------
        query:
            Selector, target spec, frozen target spec, or non-strict bare string.
        strict:
            Whether to reject non-portable query forms.
        max_fanout:
            Maximum number of matching sites.

        Returns
        -------
        SiteTable
            Ordered table of matching layer-pass records.
        """

        from ..intervention.resolver import resolve_sites

        return resolve_sites(self, query, strict=strict, max_fanout=max_fanout)

    def annotate(
        self,
        selector: Any,
        *,
        data: Any = None,
        image: str | Path | None = None,
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
            Optional local image path used by the existing ``NodeSpec.image``
            render hook.
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

        target = self.fork(name=None) if copy else self
        target._annotate_in_place(selector, data=data, image=image, max_fanout=max_fanout)
        return target

    def with_annotations(
        self,
        selector: Any,
        *,
        data: Any = None,
        image: str | Path | None = None,
        max_fanout: int = 1_000_000,
    ) -> "Trace":
        """Return an owned annotated copy of this trace.

        Parameters
        ----------
        selector:
            Selector, target spec, frozen target spec, or non-strict bare string.
        data:
            JSON-serializable metadata or a portable tensor payload.
        image:
            Optional local image path used by the existing ``NodeSpec.image``
            render hook.
        max_fanout:
            Explicit maximum number of selected sites.

        Returns
        -------
        Trace
            Forked trace carrying the requested annotations.
        """

        return self.annotate(selector, data=data, image=image, max_fanout=max_fanout, copy=True)

    def _annotate_in_place(
        self,
        selector: Any,
        *,
        data: Any,
        image: str | Path | None,
        max_fanout: int,
    ) -> None:
        """Apply annotation updates directly to this trace.

        Parameters
        ----------
        selector:
            Selector resolved against this trace.
        data:
            Annotation data supplied by the caller.
        image:
            Optional image path supplied by the caller.
        max_fanout:
            Explicit fan-out limit passed to the site resolver.

        Returns
        -------
        None
            This trace is mutated in place.
        """

        if data is None and image is None:
            raise ValueError("annotate() requires data=, image=, or both.")
        sites = self.resolve_sites(selector, max_fanout=max_fanout)
        image_value = str(image) if image is not None else None
        data_kind = self._annotation_data_kind(data)
        for site in sites:
            key = self._annotation_key_for_site(site)
            if data is not None:
                if data_kind == "blob":
                    self._store_annotation_blob(key, data)
                else:
                    self._store_annotation_breadcrumb(site, "data", data)
            if image_value is not None:
                self._store_annotation_breadcrumb(site, "image", image_value)
        self._mark_annotations_mutated()

    def _annotation_data_kind(self, data: Any) -> str:
        """Classify and validate annotation data.

        Parameters
        ----------
        data:
            Candidate annotation payload.

        Returns
        -------
        str
            ``"none"``, ``"blob"``, or ``"json"``.
        """

        if data is None:
            return "none"
        if isinstance(data, torch.Tensor):
            self._validate_annotation_tensor(data)
            return "blob"
        self._validate_annotation_json(data)
        return "json"

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

        backend_name = str(getattr(self, "backend", "torch"))
        if backend_name != "torch":
            raise ValueError(
                "annotate(data=torch.Tensor) is supported only for torch traces in this "
                f"release; this trace uses backend={backend_name!r}."
            )
        from .._io.payload_codec import get_payload_codec

        decision = get_payload_codec(backend_name).validate_for_save(tensor, strict=True)
        if decision.__class__.__name__ != "Ok":
            reason = getattr(decision, "text", "unsupported tensor payload")
            raise ValueError(
                f"Annotation tensor is not portable for backend {backend_name!r}: {reason}."
            )

    @staticmethod
    def _validate_annotation_json(data: Any) -> None:
        """Validate that an annotation breadcrumb can be persisted as JSON data.

        Parameters
        ----------
        data:
            Candidate JSON breadcrumb.

        Returns
        -------
        None
            Raises if ``data`` is not JSON-serializable.
        """

        try:
            json.dumps(data, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "annotate(data=...) must be JSON-serializable or a torch.Tensor. "
                "Convert arrays to torch.Tensor for blob persistence; values are never "
                "silently stringified."
            ) from exc

    def _annotation_key_for_site(self, site: Any) -> str:
        """Return the persistent annotation blob key for a resolved site.

        Parameters
        ----------
        site:
            Resolved layer-pass record.

        Returns
        -------
        str
            ``layer:<layer_label>`` for single-pass layers, otherwise
            ``op:<op.label>``.
        """

        layer_label = str(getattr(site, "layer_label"))
        if self.layer_num_calls.get(layer_label, 1) == 1:
            return f"layer:{layer_label}"
        return f"op:{getattr(site, 'label')}"

    def _store_annotation_blob(self, key: str, data: Any) -> None:
        """Store a blob annotation under ``_annotation_blobs``.

        Parameters
        ----------
        key:
            Namespaced annotation key.
        data:
            Codec-validated payload.

        Returns
        -------
        None
            This trace's blob mapping is mutated in place.
        """

        if self._annotation_blobs is None:
            self._annotation_blobs = {}
        self._annotation_blobs[key] = data

    def _store_annotation_breadcrumb(self, site: Any, name: str, value: Any) -> None:
        """Store a small user breadcrumb on the selected Op and Layer.

        Parameters
        ----------
        site:
            Resolved layer-pass record.
        name:
            User-namespace field name.
        value:
            JSON-compatible breadcrumb value.

        Returns
        -------
        None
            The selected Op and aggregate Layer annotation dicts are mutated.
        """

        self._user_annotation_dict(site.annotations)[name] = value
        layer_log = self.layer_logs.get(str(getattr(site, "layer_label")))
        if layer_log is not None:
            self._user_annotation_dict(layer_log.annotations)[name] = value

    @staticmethod
    def _user_annotation_dict(annotations: dict[str, Any]) -> dict[str, Any]:
        """Return the reserved user annotation namespace.

        Parameters
        ----------
        annotations:
            Op, Layer, or Trace annotation mapping.

        Returns
        -------
        dict[str, Any]
            Mutable ``annotations["user"]`` mapping.
        """

        user_annotations = annotations.setdefault("user", {})
        if not isinstance(user_annotations, dict):
            raise ValueError('annotations["user"] must be a dict to store user annotations.')
        return user_annotations

    def _mark_annotations_mutated(self) -> None:
        """Bump the annotation revision and invalidate render-only caches.

        Returns
        -------
        None
            This trace's annotation revision is incremented.
        """

        self._annotation_revision = int(getattr(self, "_annotation_revision", 0)) + 1
        self.__dict__.pop("_last_sibling_ordering_decision", None)

    def find_layers(self, query: str, *, limit: int = 10) -> List[str]:
        """Return layer labels matching a fuzzy query.

        Parameters
        ----------
        query:
            Layer-label substring or approximate layer name.
        limit:
            Maximum number of labels to return.

        Returns
        -------
        List[str]
            Matching no-pass layer labels in execution order, followed by close
            fuzzy matches when substring matches are insufficient.
        """

        query_text = str(query).lower()
        labels = list(self.layer_labels)
        substring_matches = [label for label in labels if query_text in label.lower()]
        if len(substring_matches) >= limit:
            return substring_matches[:limit]
        fuzzy_matches = difflib.get_close_matches(str(query), labels, n=limit, cutoff=0.25)
        result = substring_matches[:]
        for label in fuzzy_matches:
            if label not in result:
                result.append(label)
            if len(result) >= limit:
                break
        return result

    @property
    def uncalled_modules(self) -> _CallableList:
        """Return registered modules that were not exercised in the captured pass.

        Returns
        -------
        _CallableList
            Module addresses present on the source model but absent from the
            captured module accessor. Returns an empty list when the source
            model is no longer available.
        """

        source_ref = getattr(self, "_source_model_ref", None)
        model = source_ref() if source_ref is not None else None
        if model is None:
            return _CallableList()
        registered = {address or "self" for address, _module in model.named_modules()}
        called = set(getattr(self._module_logs, "_dict", {}).keys())
        called.update(getattr(self._module_logs, "_alias_dict", {}).keys())
        return _CallableList(sorted(registered - called))

    @property
    def model_cls(self) -> type[Any] | None:
        """Return the live source model class when the model is still alive.

        Returns
        -------
        type[Any] | None
            Runtime class of the source model, or ``None`` after the weakref dies.
        """

        source_ref = getattr(self, "_source_model_ref", None)
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

        parent_ref = getattr(self, "parent_run", None)
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

        parent = self.parent_trace
        if parent is None:
            return None
        root = parent
        while root.parent_trace is not None:
            root = root.parent_trace
        return root

    @property
    def layers_to_save(self) -> str | list[str]:
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
        return [op.label for op in self.layer_list if op.raw_index in selected_nums]

    def _source_model_class(self) -> type[Any] | None:
        """Return the live source model class if it is still retained.

        Returns
        -------
        type[Any] | None
            Source model class, or ``None`` if the weakref is unavailable.
        """

        source_ref = getattr(self, "_source_model_ref", None)
        model = source_ref() if source_ref is not None else None
        return None if model is None else type(model)

    def _inspect_source_attr(self, attr_name: str) -> str | None:
        """Inspect one source-model attribute when stored metadata is absent.

        Parameters
        ----------
        attr_name:
            One of the Trace source-introspection field names.

        Returns
        -------
        str | None
            Inspected metadata, or ``None`` when the source model is gone or
            the callable cannot be inspected.
        """

        model_cls = self._source_model_class()
        if model_cls is None:
            return None
        if attr_name == "class_docstring":
            return model_cls.__doc__
        if attr_name in {"init_signature", "init_docstring"}:
            target = getattr(model_cls, "__init__", None)
        else:
            target = getattr(model_cls, "forward", None)
        if target is None:
            return None
        if attr_name.endswith("_docstring"):
            return getattr(target, "__doc__", None)
        try:
            return str(inspect.signature(target))
        except (TypeError, ValueError):
            return None

    @property
    def class_docstring(self) -> str | None:
        """Return the source model class docstring."""

        return self.__dict__.get("class_docstring") or self._inspect_source_attr("class_docstring")

    @class_docstring.setter
    def class_docstring(self, value: str | None) -> None:
        """Store the source model class docstring."""

        self.__dict__["class_docstring"] = value

    @class_docstring.deleter
    def class_docstring(self) -> None:
        """Delete the stored source model class docstring."""

        self.__dict__.pop("class_docstring", None)

    @property
    def init_signature(self) -> str | None:
        """Return the source model ``__init__`` signature."""

        return self.__dict__.get("init_signature") or self._inspect_source_attr("init_signature")

    @init_signature.setter
    def init_signature(self, value: str | None) -> None:
        """Store the source model ``__init__`` signature."""

        self.__dict__["init_signature"] = value

    @init_signature.deleter
    def init_signature(self) -> None:
        """Delete the stored source model ``__init__`` signature."""

        self.__dict__.pop("init_signature", None)

    @property
    def init_docstring(self) -> str | None:
        """Return the source model ``__init__`` docstring."""

        return self.__dict__.get("init_docstring") or self._inspect_source_attr("init_docstring")

    @init_docstring.setter
    def init_docstring(self, value: str | None) -> None:
        """Store the source model ``__init__`` docstring."""

        self.__dict__["init_docstring"] = value

    @init_docstring.deleter
    def init_docstring(self) -> None:
        """Delete the stored source model ``__init__`` docstring."""

        self.__dict__.pop("init_docstring", None)

    @property
    def forward_signature(self) -> str | None:
        """Return the source model ``forward`` signature."""

        return self.__dict__.get("forward_signature") or self._inspect_source_attr(
            "forward_signature"
        )

    @forward_signature.setter
    def forward_signature(self, value: str | None) -> None:
        """Store the source model ``forward`` signature."""

        self.__dict__["forward_signature"] = value

    @forward_signature.deleter
    def forward_signature(self) -> None:
        """Delete the stored source model ``forward`` signature."""

        self.__dict__.pop("forward_signature", None)

    @property
    def forward_docstring(self) -> str | None:
        """Return the source model ``forward`` docstring."""

        return self.__dict__.get("forward_docstring") or self._inspect_source_attr(
            "forward_docstring"
        )

    @forward_docstring.setter
    def forward_docstring(self, value: str | None) -> None:
        """Store the source model ``forward`` docstring."""

        self.__dict__["forward_docstring"] = value

    @forward_docstring.deleter
    def forward_docstring(self) -> None:
        """Delete the stored source model ``forward`` docstring."""

        self.__dict__.pop("forward_docstring", None)

    def __str__(self) -> str:
        """Human-readable summary; delegates to post-pass or mid-pass formatter."""
        if self._tracing_finished:
            return _str_after_pass(self)
        else:
            return _str_during_pass(self)

    def __repr__(self) -> str:
        """Short identity-card representation for REPL display."""
        from ..visualization._summary_internal import format_model_repr

        return format_model_repr(self)

    def _repr_html_(self) -> str:
        """Return the notebook HTML representation for this model log.

        Returns
        -------
        str
            HTML fragment for IPython/Jupyter display.

        Falls back to ``repr(self)`` when the notebook extra is unavailable.
        """
        try:
            import IPython  # noqa: F401
        except ImportError:
            return repr(self)

        from html import escape

        layers = len(getattr(self, "layer_logs", {}) or {})
        ops = getattr(self, "num_ops", 0)
        save_level = "all" if getattr(self, "_layers_saved", False) else "selected"
        if getattr(self, "num_saved_ops", 0) == 0:
            save_level = "metadata only"
        nonfinite = self.first_nonfinite(link_format="html")
        nonfinite_summary = (
            "No non-finite saved outs" if nonfinite.startswith("No non-finite") else nonfinite
        )
        title = escape(str(getattr(self, "trace_label", None) or self.model_label))
        state = escape(str(getattr(getattr(self, "state", None), "name", "UNKNOWN")))
        return (
            "<div style='border:1px solid #d0d7de;border-radius:8px;"
            "padding:10px 12px;font-family:system-ui,sans-serif;max-width:560px'>"
            f"<div style='font-weight:700;margin-bottom:6px'>TorchLens Trace: {title}</div>"
            "<div style='display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:4px 12px'>"
            f"<div><b>Layers</b>: {layers}</div>"
            f"<div><b>Ops</b>: {ops}</div>"
            f"<div><b>Save level</b>: {escape(save_level)}</div>"
            f"<div><b>Run state</b>: {state}</div>"
            "</div>"
            f"<div style='margin-top:8px'><b>NaN/Inf</b>: {nonfinite_summary}</div>"
            "</div>"
        )

    def __iter__(self) -> Iterator[Any]:
        """Loops through all tensors in the log."""
        if self._tracing_finished:
            return iter(self.layer_list)
        else:
            return iter(list(getattr(self, "_raw" + "_layer_dict").values()))

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Call :func:`torchlens.save` for this model log.

        Warning
        -------
        Portable bundles contain a pickle file. Only load bundles from trusted
        sources. Loading an untrusted bundle can execute arbitrary code.
        """

        from .._io.bundle import save as save_bundle

        save_bundle(self, path, **kwargs)

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

        from .container import reconstruct_output

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

        from .container import reconstruct_container

        return reconstruct_container(self, site=site, role=role, values=values)

    def __getstate__(self) -> Dict[str, Any]:
        """Return pickle state with non-picklable weakref-backed accessors stripped."""
        state = self.__dict__.copy()
        state["_module_logs"] = None
        state["_buffer_accessor"] = None
        state["_source_model_ref"] = None
        state["parent_run"] = None
        state["last_run"] = None
        state["_out_identity_cache"] = {}
        state["_out_hash_cache"] = {}
        state["_code_context_cache"] = {}
        state.pop("_container_ordinals_by_output_op_label", None)
        state.pop("_container_ordinals_by_input_func_call_id", None)
        state.pop("_build_state", None)
        state["_backward_gradfn_refs"] = []
        state["_tl_backward_hooked_tensor_keys"] = set()
        state["_pending_live_fire_records"] = []
        state["_last_hook_handle_ids"] = ()
        state["_activation_transform_repr"] = (
            repr(self.activation_transform) if self.activation_transform is not None else None
        )
        state["tlspec_version"] = TLSPEC_VERSION
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore pickle state and rebuild weakref-backed links."""
        read_tlspec_version(state, cls_name=type(self).__name__)
        containers_were_serialized = "_containers" in state and state["_containers"] is not None
        setstate_defaults = {
            **_MODEL_LOG_DEFAULT_FILL,
            "tlspec_version": TLSPEC_VERSION,
            "transform_repr": None,
            "decoded_output": None,
            "output_postprocessor": None,
            "output_id2label": None,
            "output_num_classes": None,
            "_activation_transform_repr": None,
            "module_identity_mode": "torch_module",
            "param_source": "native-module",
            "save_raw_activations": True,
            "save_mode": "copy",
            "raw_output": None,
            "_output_transform": None,
            "save_raw_output": "small",
            "layer_visualizers": None,
            "save_visualizations": False,
            "_visualizer_dir": None,
            "input_annotations": {},
            "grad_transform": None,
            "grad_transform_repr": None,
            "save_raw_gradients": True,
            "save_grads": _legacy_save_grads_from_state(state),
            "capture_tensor_grad_hooks": True,
            "_grad_op_nums_to_save": [],
            "has_backward_pass": False,
            "grad_fn_logs": OrderedDict(),
            "grad_fn_order": [],
            "backward_pass_logs": OrderedDict(),
            "backward_root_grad_fn_object_ids": [],
            "backward_durations": [],
            "num_backward_passes": 0,
            "backward_peak_memory": 0,
            "backward_memory_backend": "unknown",
            "total_autograd_memory": None,
            "_buffer_accessor": None,
            "_module_logs": None,
            "_module" + "_build_data": None,
            "_out_writer": None,
            "_keep_outs_in_memory": True,
            "_defer_streaming_bundle_finalization": False,
            "_out_sink": None,
            "append_history": [],
            "_in" + "_exhaustive_pass": False,
            "_module" + "_containment_engine": "hook_stack",
            "_exhaustive" + "_module_stack": [],
            "_source_code_blob": {},
            "_source_model_ref": None,
            "backward_ready": False,
            "inference_only": False,
            "chunked_forward": False,
            "module_filter": None,
            "raise_on_nan": False,
            "keep_orphans": False,
            "annotations": {},
            "observer_spans": [],
            "manual_tensor_connections": [],
            "forward_source_file": None,
            "forward_source_line": None,
            "class_source_file": None,
            "class_source_line": None,
            "init_source_file": None,
            "init_source_line": None,
            "class_docstring": None,
            "init_signature": None,
            "init_docstring": None,
            "forward_signature": None,
            "forward_docstring": None,
            "code_context": [],
            "capture_cache_hit": False,
            "capture_cache_key": None,
            "capture_cache_path": None,
            "recording_kept": True,
            "_out_dedup_mode": "identity",
            "_out_identity_cache": {},
            "_out_hash_cache": {},
            "_code_context_cache": {},
            "_last_hook_handle_ids": (),
            "conditionals": ConditionalAccessor(),
            "total_gradient_memory": 0,
            "saved_gradient_memory": 0,
            "total_param_gradient_memory": 0,
            "forward_peak_memory": 0,
            "forward_memory_backend": "unknown",
            "_postprocessing_active": False,
            # `_backward_gradfn_refs` is `FieldPolicy.DROP` (never part of
            # `MODEL_LOG_FIELD_ORDER`) and `__getstate__` always emits an
            # empty `list` for it -- but any artifact serialized by code
            # older than this line still has it baked in as a `dict`
            # (`__getstate__` used to hardcode `{}`). Listing it here
            # lets `coerce_container_typed_state` below fix that
            # present-but-wrong-typed legacy shape, not just absence.
            "_backward_gradfn_refs": [],
        }
        default_fill_state(state, defaults=setstate_defaults)
        coerce_container_typed_state(
            state,
            setstate_defaults,
            exclude={
                # `List[int] | str` -- a bare string sentinel (``"all"``) is
                # a legitimate value, not a type-mismatch bug; coercing it
                # to ``list("all")`` would silently corrupt real data.
                "_grad_op_nums_to_save",
            },
        )
        if "_grad_layer_nums_to_save" in state and "_grad_op_nums_to_save" not in state:
            state["_grad_op_nums_to_save"] = state.pop("_grad_layer_nums_to_save")
        if "_saved_grads_set" in state and "_saved_grad_labels" not in state:
            state["_saved_grad_labels"] = state.pop("_saved_grads_set")
        state.pop("save_gradients", None)
        state.pop("gradients_to_save", None)
        state.pop("_keep_grads_in_memory", None)
        state.pop("_grad_stream_retain_in_memory", None)
        if state.get("_intervention_spec") is None:
            state["_intervention_spec"] = InterventionSpec()
        if not state.get("relationship_evidence"):
            state["relationship_evidence"] = {
                "model": Relationship.UNKNOWN,
                "weights": Relationship.UNKNOWN,
                "input": Relationship.UNKNOWN,
                "graph": Relationship.UNKNOWN,
            }
        if state["backward_ready"] is None:
            state["backward_ready"] = False
        if state["inference_only"] is None:
            state["inference_only"] = False
        if state["chunked_forward"] is None:
            state["chunked_forward"] = False
        for field_name in (
            "setup_duration",
            "forward_duration",
            "cleanup_duration",
            "func_calls_duration",
        ):
            state[field_name] = Duration(state.get(field_name) or 0.0)
        state["backward_durations"] = [
            Duration(duration) for duration in state.get("backward_durations", [])
        ]
        conditional_arm_entry_edges = _normalize_conditional_arm_entry_edges(
            state.get("conditional_arm_entry_edges") or {}
        )
        for parent, child in state.pop("conditional_then_entry_edges", []) or []:
            _append_conditional_arm_edge(conditional_arm_entry_edges, (0, "then"), (parent, child))
        for conditional_id, elif_index, parent, child in (
            state.pop("conditional_elif_entry_edges", []) or []
        ):
            _append_conditional_arm_edge(
                conditional_arm_entry_edges,
                (conditional_id, f"elif_{elif_index}"),
                (parent, child),
            )
        for conditional_id, parent, child in state.pop("conditional_else_entry_edges", []) or []:
            _append_conditional_arm_edge(
                conditional_arm_entry_edges,
                (conditional_id, "else"),
                (parent, child),
            )
        state["conditional_arm_entry_edges"] = conditional_arm_entry_edges
        for field_name in (
            "total_activation_memory",
            "total_gradient_memory",
            "total_backward_memory",
            "saved_activation_memory",
            "saved_gradient_memory",
            "total_param_memory",
            "total_param_gradient_memory",
            "forward_peak_memory",
            "backward_peak_memory",
        ):
            state[field_name] = Bytes(state.get(field_name, 0) or 0)
        if state.get("total_autograd_memory") is not None:
            state["total_autograd_memory"] = Bytes(state["total_autograd_memory"])
        self.__dict__.update(state)
        if not containers_were_serialized:
            self.__dict__.pop("_containers", None)
        if self.__dict__.get("_module_logs") is None:
            self._module_logs = ModuleAccessor({})
        if "_buffer_accessor" not in self.__dict__:
            self._buffer_accessor = None
        for layer_log in self.layer_logs.values():
            layer_log.source_trace = self
            for layer_pass in layer_log.ops.values():
                layer_pass.source_trace = self
        for layer_pass in self.layer_list:
            layer_pass.source_trace = self
        for grad_fn_handle in self.grad_fn_logs.values():
            grad_fn_handle.source_trace = self
            if grad_fn_handle.op is not None:
                grad_fn_handle.op.grad_fn_handle = grad_fn_handle
                op_passes = getattr(grad_fn_handle.op, "ops", None)
                if op_passes is not None and hasattr(op_passes, "values"):
                    for layer_pass in op_passes.values():
                        layer_pass.grad_fn_handle = grad_fn_handle
        _state._register_log(self)

    def replace_state_from(self, new_log: "Trace") -> None:
        """Atomically replace this log's run-state from another ``Trace``.

        This method is intended for intervention rerun. The rerun engine builds
        ``new_log`` off to the side and calls this only after validation ops.
        The final state replacement uses one state-restore pass over the new fields
        to minimize torn-state windows. Concurrent reads during rerun are
        unsupported; no lock is taken.

        Parameters
        ----------
        new_log:
            Fully postprocessed fresh log whose graph, layer containers,
            accessors, output metadata, shape/hash fields, and per-pass entries
            should replace this log's current run-state.

        Returns
        -------
        None
            This log is mutated in place.
        """

        preserved_fields = (
            "trace_label",
            "parent_run",
            "_intervention_spec",
            "_transform",
            "save_raw_input",
            "batch_render",
            "_output_transform",
            "save_raw_output",
            "state_history",
            "_warned_direct_write",
            "_warned_mutate_in_place",
            "model_object_id",
            "model_class_qualname",
            "param_hash_quick",
            "param_hash_full",
            "input_object_id",
            "input_signature_hash",
            "is_appended",
            "_append_sequence_id",
            "append_history",
            "relationship_evidence",
            "_source_model_ref",
            "_has_direct_writes",
            "_spec_revision",
            "_out_recipe_revision",
            "input_annotations",
            "_annotation_blobs",
        )
        current_state = dict(state_items(self))
        preserved_trace_user_annotations = self._copy_user_annotations(
            current_state.get("annotations")
        )
        preserved_state = {
            field_name: current_state.get(field_name) for field_name in preserved_fields
        }
        if "_annotation_revision" in current_state:
            preserved_state["_annotation_revision"] = current_state["_annotation_revision"]
        replacement_state = dict(state_items(new_log))
        replacement_state.update(preserved_state)
        replacement_state["annotations"] = self._merge_user_annotations(
            self._copy_rerun_value(getattr(new_log, "annotations", {})),
            preserved_trace_user_annotations,
        )
        state_restore(self, replacement_state)
        self.__dict__.pop("_validation_replay_status", None)
        _TRACE_OP_ACCESSOR_CACHE.pop(self, None)
        _TRACE_LAYER_ACCESSOR_CACHE.pop(self, None)
        self._rebind_fork_owner_refs()

    def _refresh_matching_rerun_state_from(self, new_log: "Trace") -> bool:
        """Refresh payload-bearing fields from a same-shape rerun.

        Parameters
        ----------
        new_log:
            Fully captured and postprocessed rerun candidate.

        Returns
        -------
        bool
            True when the existing graph containers were refreshed in place.
            False means labels did not match closely enough and callers should
            use ``replace_state_from``.
        """

        old_raw_labels = tuple(layer._layer_label_raw for layer in self.layer_list)
        new_raw_labels = tuple(layer._layer_label_raw for layer in new_log.layer_list)
        old_final_labels = tuple(layer.layer_label for layer in self.layer_list)
        new_final_labels = tuple(layer.layer_label for layer in new_log.layer_list)
        if old_raw_labels != new_raw_labels or old_final_labels != new_final_labels:
            return False

        new_by_raw = {layer._layer_label_raw: layer for layer in new_log.layer_list}
        for layer in self.layer_list:
            self._refresh_rerun_op_from(layer, new_by_raw[layer._layer_label_raw])
        self._refresh_rerun_layer_logs_from(new_log)
        self._refresh_rerun_trace_fields_from(new_log)
        self.__dict__.pop("_validation_replay_status", None)
        _TRACE_OP_ACCESSOR_CACHE.pop(self, None)
        _TRACE_LAYER_ACCESSOR_CACHE.pop(self, None)
        self._rebind_fork_owner_refs()
        return True

    def _refresh_rerun_op_from(self, layer: Any, new_layer: Any) -> None:
        """Copy rerun fields into one existing ``Op``.

        Parameters
        ----------
        layer:
            Existing operation record retained by the fast path.
        new_layer:
            Fresh rerun operation record supplying current payloads and
            per-call metadata.
        """

        preserved_fields = {
            "source_trace",
            "_source_trace_ref",
            "input_ops",
            "input_activations",
            "input_shapes",
            "input_dtypes",
            "input_memory",
            "num_inputs",
            "is_in_conditional_body",
        }
        new_layer_state = dict(state_items(new_layer))
        preserved_user_annotations = self._copy_user_annotations(
            getattr(layer, "annotations", None)
        )
        for field_name in LAYER_PASS_LOG_FIELD_ORDER:
            if field_name in preserved_fields:
                continue
            value = self._copy_rerun_value(new_layer_state.get(field_name))
            if field_name == "annotations":
                value = self._merge_user_annotations(value, preserved_user_annotations)
            layer._internal_set(
                field_name,
                value,
            )
        for field_name in (
            "out_ref",
            "grad_ref",
            "_pending_blob_id",
            "_pending_transformed_out_blob_id",
            "_pending_grad_blob_id",
            "_pending_transformed_grad_blob_id",
            "annotations",
            "interventions",
            "container_spec",
            "args_template",
            "kwargs_template",
            "_edge_uses",
        ):
            if hasattr(new_layer, field_name):
                value = self._copy_rerun_value(new_layer_state.get(field_name))
                if field_name == "annotations":
                    value = self._merge_user_annotations(value, preserved_user_annotations)
                layer._internal_set(
                    field_name,
                    value,
                )
        layer.source_trace = self

    def _refresh_rerun_layer_logs_from(self, new_log: "Trace") -> None:
        """Refresh aggregate ``Layer`` records from a same-shape rerun.

        Parameters
        ----------
        new_log:
            Fresh rerun trace with layer aggregates already postprocessed.
        """

        new_layer_logs = getattr(new_log, "layer_logs", {}) or {}
        for label, layer_log in self.layer_logs.items():
            new_layer_log = new_layer_logs.get(label)
            if new_layer_log is None:
                continue
            preserved_user_annotations = self._copy_user_annotations(
                getattr(layer_log, "annotations", None)
            )
            for field_name, value in state_items(new_layer_log):
                if field_name in {"_source_trace_ref", "ops"}:
                    continue
                copied_value = self._copy_rerun_value(value)
                if field_name == "annotations":
                    copied_value = self._merge_user_annotations(
                        copied_value,
                        preserved_user_annotations,
                    )
                setattr(layer_log, field_name, copied_value)
            layer_log.source_trace = self

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

        if not isinstance(annotations, dict):
            return None
        user_annotations = annotations.get("user")
        if not isinstance(user_annotations, dict):
            return None
        return self._copy_rerun_value(user_annotations)

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

        merged = fresh_annotations if isinstance(fresh_annotations, dict) else {}
        if preserved_user_annotations is not None:
            merged["user"] = self._copy_rerun_value(preserved_user_annotations)
        return merged

    def _refresh_rerun_trace_fields_from(self, new_log: "Trace") -> None:
        """Refresh trace-level run fields without replacing graph containers.

        Parameters
        ----------
        new_log:
            Fresh rerun trace supplying current run metadata.
        """

        field_names = (
            "raw_output",
            "save_raw_output",
            "has_gradients",
            "random_seed",
            "chunked_forward",
            "input_object_id",
            "input_signature_hash",
            "graph_shape_hash",
            "_raw_event_shape_hash",
            "num_saved_ops",
            "saved_activation_memory",
            "total_activation_memory",
            "saved_gradient_memory",
            "total_gradient_memory",
            "total_backward_memory",
            "total_autograd_memory",
            "forward_peak_memory",
            "backward_peak_memory",
            "output_layers",
            "output_layers_by_pass",
            "output_layers_by_module_call",
            "_output_container_specs_by_raw_label",
        )
        for field_name in field_names:
            if hasattr(new_log, field_name):
                setattr(self, field_name, self._copy_rerun_value(getattr(new_log, field_name)))
        self.facet_registry_snapshot = getattr(new_log, "facet_registry_snapshot", None)

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

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, list):
            return [self._copy_rerun_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_rerun_value(item) for item in value)
        if isinstance(value, dict):
            return {
                self._copy_rerun_value(key): self._copy_rerun_value(item)
                for key, item in value.items()
            }
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def append_state_from(self, new_log: "Trace") -> None:
        """Merge compatible chunk outs from ``new_log`` into this log.

        Parameters
        ----------
        new_log:
            Freshly captured append chunk whose topology and tensor metadata
            have already been validated against this log.
        """

        new_by_raw = {layer._layer_label_raw: layer for layer in new_log.layer_list}
        old_by_label = {
            key: layer
            for layer in self.layer_list
            for key in (layer._layer_label_raw, layer.layer_label)
        }
        for layer in self.layer_list:
            new_layer = new_by_raw[layer._layer_label_raw]
            if not (
                getattr(layer, "is_buffer", False)
                or self._is_append_buffer_side_effect_layer(layer, old_by_label)
            ):
                layer._append_tensor_from(new_layer, "out")
                layer._append_tensor_from(new_layer, "transformed_out")
            self._copy_append_last_chunk_fields(layer, new_layer)
            self._refresh_appended_tensor_metadata(layer)
        self.has_gradients = self.has_gradients or new_log.has_gradients
        self.random_seed = new_log.random_seed
        self.input_object_id = new_log.input_object_id
        self.input_signature_hash = new_log.input_signature_hash
        self._rebind_fork_owner_refs()

    def _is_append_buffer_side_effect_layer(
        self, layer: Any, layer_by_label: dict[str, Any]
    ) -> bool:
        """Return whether ``layer`` only feeds buffer version side effects.

        Parameters
        ----------
        layer:
            Candidate layer being considered for append tensor concatenation.
        layer_by_label:
            Mapping from raw and final labels to layers in this trace.

        Returns
        -------
        bool
            True when every tracked child is a buffer version node created by a
            buffer write.
        """

        child_labels = list(getattr(layer, "children", []))
        if not child_labels:
            return False
        saw_buffer_write = False
        for child_label in child_labels:
            child_layer = layer_by_label.get(child_label)
            if child_layer is None:
                return False
            if not (
                getattr(child_layer, "is_buffer", False)
                and getattr(child_layer, "buffer_write_kind", None) is not None
            ):
                return False
            saw_buffer_write = True
        return saw_buffer_write

    def _copy_append_last_chunk_fields(self, layer: Any, new_layer: Any) -> None:
        """Copy per-call metadata fields from the last appended chunk.

        Parameters
        ----------
        layer:
            Existing accumulated layer pass.
        new_layer:
            New chunk layer pass supplying per-call state.
        """

        for field_name in (
            "func_duration",
            "flops_forward",
            "flops_backward",
            "func_rng_states",
            "func_autocast_state",
            "arg_names",
            "num_args_total",
            "num_pos_args",
            "num_kwargs",
            "non_tensor_pos_args",
            "non_tensor_kwargs",
            "func_non_tensor_args",
            "is_inplace",
            "grad_fn_class_name",
            "grad_fn_object_id",
            "interventions",
            "annotations",
        ):
            if hasattr(new_layer, field_name):
                layer._internal_set(
                    field_name, self._copy_append_metadata_value(getattr(new_layer, field_name))
                )

    def _copy_append_metadata_value(self, value: Any) -> Any:
        """Copy metadata from the last chunk without failing on non-leaf tensors.

        Parameters
        ----------
        value:
            Metadata value from the new chunk.

        Returns
        -------
        Any
            Best-effort copied value.
        """

        if isinstance(value, torch.Tensor):
            from ..utils.tensor_utils import safe_copy

            return safe_copy(value, detach_tensor=True)
        if isinstance(value, list):
            return [self._copy_append_metadata_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._copy_append_metadata_value(item) for item in value)
        if isinstance(value, dict):
            return {
                self._copy_append_metadata_value(key): self._copy_append_metadata_value(item)
                for key, item in value.items()
            }
        try:
            return copy.deepcopy(value)
        except RuntimeError:
            return value

    def _refresh_appended_tensor_metadata(self, layer: Any) -> None:
        """Refresh shape, dtype, and memory fields after tensor concatenation.

        Parameters
        ----------
        layer:
            Layer pass whose tensor fields may have been concatenated.
        """

        for tensor_field, shape_field, dtype_field, memory_field in (
            ("out", "shape", "dtype", "activation_memory"),
            (
                "transformed_out",
                "transformed_out_shape",
                "transformed_out_dtype",
                "transformed_activation_memory",
            ),
            ("grad", "grad_shape", "grad_dtype", "gradient_memory"),
            (
                "transformed_grad",
                "transformed_grad_shape",
                "transformed_grad_dtype",
                "transformed_gradient_memory",
            ),
        ):
            value = getattr(layer, tensor_field, None)
            if isinstance(value, torch.Tensor):
                from ..utils.tensor_utils import get_memory_amount

                layer._internal_set(shape_field, tuple(value.shape))
                layer._internal_set(dtype_field, value.dtype)
                layer._internal_set(memory_field, Bytes(get_memory_amount(value)))
            else:
                layer._internal_set(shape_field, None)
                layer._internal_set(dtype_field, None)
                layer._internal_set(memory_field, None)

    # ********************************************
    # ******** Public Convenience Methods ********
    # ********************************************


Trace.FIELD_FORK_POLICY = fork_policy_from_policy(Trace.FIELD_POLICY)  # type: ignore[attr-defined]
Trace.DEFAULT_FILL_STATE = default_fill_state_from_policy(Trace.FIELD_POLICY)  # type: ignore[attr-defined]
