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
  (postprocessing step 17) and then cleared.  Initialised via
  ``_init_module_hierarchy_data()``.
"""

import copy
import inspect
import json
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import difflib
from functools import cached_property
from html import escape
from pathlib import Path
import time
import uuid
import weakref
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    TYPE_CHECKING,
    TextIO,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    import pandas as pd

    from .._io.streaming import BundleStreamWriter
    from ..experimental.dagua._bridge import TorchLensRenderAudit
    from ..fastlog.types import ModuleStackFrame
    from ..intervention.types import FireRecord
    from ..validation.status import ValidationReplayStatus
    from ..visualization.code_panel import CodePanelOption
    from .buffer import BufferAccessor
    from .func_call_location import FuncCallLocation
    from .layer import LayerAccessor
    from .module import Module, ModuleCall

from .._deprecations import MISSING, MissingType, warn_deprecated_alias
from .. import _state
from ..backends import BackendName
from .._trace_state import TraceState
from .._training_validation import reject_compiled_model
from .._literals import (
    BufferVisibilityLiteral,
    VisDirectionLiteral,
    VisInterventionModeLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from .._io import FieldPolicy, TLSPEC_VERSION, default_fill_state, read_tlspec_version
from ..constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from ..captured_run import CapturedRun
from ..ir.events import TraceBuildState
from ..options import (
    InterventionOptions,
    ReplayOptions,
    merge_intervention_options,
    merge_replay_options,
)
from ..intervention.types import (
    FrozenInterventionSpec,
    ForkFieldPolicy,
    MODEL_LOG_FIELD_FORK_POLICY,
    InterventionSpec,
    Relationship,
    TargetSpec,
)
from ..types import ActivationPostfunc, GradientPostfunc
from ..utils.tensor_utils import SaveMode
from ..quantities import Bytes, Duration, Flops, Macs, as_duration
from .._errors import AmbiguousOpLookupError
from .cleanup import (
    _LIST_FIELDS_TO_CLEAN,
    _clear_entry_attributes,
    _label_for_reference_removal,
    _remove_log_entry_references,
    _scrub_conditional_fields_after_removal,
    _scrub_per_op_equivalence_lists,
    cleanup,
)
from .module import ModuleAccessor
from .param import ParamAccessor
from ._accessor_base import Accessor
from .interface import (
    _format_conditional_branch_stack,
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
)
from .backward_pass import BackwardPass, BackwardPassAccessor
from .derived_grad import DerivedGradAccessor, IntermediateDerivedGradAccessor
from .grad_fn import GradFnAccessor, GradFn
from .layer import Layer, OpAccessor
from .op import Op, TensorLog
from ._state_adapter import state_items, state_new, state_restore
from .._source_links import file_line_text, terminal_file_line_link, vscode_file_line_link

_USE_STORED_TRANSFORM = object()
_JAX_VALIDATION_REPLAY_BACKEND = "jax"
_MLX_VALIDATION_REPLAY_BACKEND = "mlx"
_TINYGRAD_VALIDATION_REPLAY_BACKEND = "tinygrad"


def _output_role_from_container_path(path: Any) -> str | None:
    """Return the final output-container role for a dataframe row.

    Parameters
    ----------
    path:
        Captured typed container path.

    Returns
    -------
    str | None
        Last key/index/field component, or ``None`` for non-container rows.
    """

    components = tuple(path or ())
    if not components:
        return None
    component = components[-1]
    for attr_name in ("index", "key", "name"):
        if hasattr(component, attr_name):
            return str(getattr(component, attr_name))
    return str(component)


def _is_batch_topk_decoded(value: Any) -> bool:
    """Return whether ``value`` is the typed decoded batch top-k representation.

    Parameters
    ----------
    value:
        Candidate decoded output value.

    Returns
    -------
    bool
        Whether the value has ``{"kind": "batch_topk", "rows": [...]}`` shape.
    """

    return (
        isinstance(value, Mapping)
        and value.get("kind") == "batch_topk"
        and isinstance(value.get("rows"), list)
    )


def _decoded_batch_topk_rows(value: Any) -> list[dict[str, Any]] | None:
    """Return decoded batch top-k rows from typed or legacy representations.

    Parameters
    ----------
    value:
        Decoded output value.

    Returns
    -------
    list[dict[str, Any]] | None
        Batch top-k rows, or ``None`` when the decoded value is another kind.
    """

    if _is_batch_topk_decoded(value):
        rows = value["rows"]
    elif isinstance(value, list):
        rows = value
    else:
        return None
    if all(
        isinstance(row, dict) and {"batch_item", "rank", "label", "prob"} <= set(row)
        for row in rows
    ):
        return cast(list[dict[str, Any]], rows)
    return None


def _normalize_batch_items(
    batch_items: int | Sequence[int] | None, rows: Sequence[Mapping[str, Any]]
) -> set[int] | None:
    """Normalize an output-table batch item selector.

    Parameters
    ----------
    batch_items:
        ``None``, a count, or explicit batch item indices.
    rows:
        Decoded rows used to infer available item indices.

    Returns
    -------
    set[int] | None
        Selected batch item indices, or ``None`` for all.
    """

    if batch_items is None:
        return None
    available = sorted({int(row.get("batch_item", 0)) for row in rows})
    if isinstance(batch_items, int):
        if batch_items < 0:
            raise ValueError("batch_items count must be >= 0.")
        return set(available[:batch_items])
    return {int(item) for item in batch_items}


def _retained_output_logits(trace: "Trace") -> torch.Tensor | None:
    """Return a retained output tensor suitable for classification re-decode.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    torch.Tensor | None
        Retained logits tensor, if exactly one output tensor is available.
    """

    tensors = [
        getattr(op, "out", None)
        for op in getattr(trace, "output_ops", [])
        if isinstance(getattr(op, "out", None), torch.Tensor)
    ]
    if len(tensors) != 1:
        return None
    return tensors[0]


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
    "total_gradient_memory": 0,
    "total_backward_memory": 0,
    "saved_gradient_memory": 0,
    "num_saved_layers": 0,
    "num_saved_module_calls": 0,
    "num_saved_grad_fns": 0,
    "num_saved_grad_fn_calls": 0,
    "total_param_gradient_memory": 0,
    "forward_peak_memory": 0,
}
_MODEL_LOG_DEFAULT_FILL = {
    **{field_name: None for field_name in MODEL_LOG_FIELD_ORDER},
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


@dataclass(frozen=True)
class ModelProfile:
    """Computed descriptor for recognized semantic I/O profiles.

    Attributes
    ----------
    input_modality:
        Conservative modality inferred from raw input and preprocessing
        provenance.
    input_preprocessing_source:
        ``ResolvedPreprocessing.source`` when present.
    output_postprocessing_source:
        ``ResolvedPostprocessing.source`` when present.
    output_label_count:
        Number of known output labels/classes, if available.
    has_output_labels:
        Whether a label space is available from ``output_id2label``.
    num_stimuli:
        Inferred leading batch/stimulus count from ``raw_input``.
    has_raw_images:
        Whether raw input appears to contain image objects.
    keystone_applicable:
        Whether the image-classifier keystone cascade has the required input
        and output metadata.
    """

    input_modality: str
    input_preprocessing_source: str | None
    output_postprocessing_source: str | None
    output_label_count: int | None
    has_output_labels: bool
    num_stimuli: int | None
    has_raw_images: bool
    keystone_applicable: bool


_IMAGE_PREPROCESSING_SOURCES = frozenset(
    {
        "hf_auto_image_processor",
        "torchvision_weights",
        "imagenet_default",
        "timm",
    }
)
_TEXT_PREPROCESSING_SOURCES = frozenset({"hf_auto_tokenizer"})


def _is_pil_image_like(value: Any) -> bool:
    """Return whether ``value`` looks like a PIL image instance.

    Parameters
    ----------
    value:
        Candidate raw input value.

    Returns
    -------
    bool
        Whether the value has the minimal PIL image surface used by TorchLens
        raw-input rendering.
    """

    return hasattr(value, "mode") and hasattr(value, "size") and hasattr(value, "copy")


def _raw_input_contains_images(value: Any) -> bool:
    """Return whether raw input contains one or more image-like objects.

    Parameters
    ----------
    value:
        Raw input value captured on a trace.

    Returns
    -------
    bool
        Whether an image-like object is present.
    """

    if _is_pil_image_like(value):
        return True
    if isinstance(value, Mapping):
        return any(_raw_input_contains_images(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_raw_input_contains_images(item) for item in value)
    return False


def _raw_input_num_stimuli(value: Any) -> int | None:
    """Infer a leading stimulus count from raw input.

    Parameters
    ----------
    value:
        Raw input value captured on a trace.

    Returns
    -------
    int | None
        Inferred stimulus count, or ``None`` when unavailable.
    """

    if value is None:
        return None
    if _is_pil_image_like(value) or isinstance(value, str):
        return 1
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, Mapping):
        counts = {
            count for item in value.values() if (count := _raw_input_num_stimuli(item)) is not None
        }
        return counts.pop() if len(counts) == 1 else None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value)
    return None


def _infer_input_modality(raw_input: Any, preprocessing_source: str | None) -> str:
    """Infer a conservative input modality descriptor.

    Parameters
    ----------
    raw_input:
        Trace raw input value.
    preprocessing_source:
        Resolved preprocessing source string, if any.

    Returns
    -------
    str
        ``"image"``, ``"text"``, ``"tensor"``, or ``"unknown"``.
    """

    if preprocessing_source in _IMAGE_PREPROCESSING_SOURCES or _raw_input_contains_images(
        raw_input
    ):
        return "image"
    if preprocessing_source in _TEXT_PREPROCESSING_SOURCES or isinstance(raw_input, str):
        return "text"
    if isinstance(raw_input, torch.Tensor) or isinstance(raw_input, np.ndarray):
        return "tensor"
    return "unknown"


# Op fields deliberately omitted from ``Trace.to_pandas()`` columns. Every
# field in ``LAYER_PASS_LOG_FIELD_ORDER`` must either appear as a dataframe
# column or be listed here -- ``tests/test_io_pandas.py`` enforces this so new
# Op fields can never silently fail to reach the layer-pass table again
# (regression gate for TO-PANDAS-NEW-FIELDS).
_TO_PANDAS_EXCLUDED_OP_FIELDS: frozenset[str] = frozenset(
    {
        # Internal / private bookkeeping (not part of the user-facing table):
        "_label_raw",
        "_layer_label_raw",
        "_tracing_finished",
        "_construction_done",
        "_param_barcodes",
        "_param_logs",
        "_edge_uses",
        "_address_normalized",
        "source_trace",
        # Tensor payloads (bulk data; access via ``layer.out`` / ``layer.grad``):
        "out",
        "transformed_out",
        "transformed_grad",
        "input_activations",
        "saved_args",
        "saved_kwargs",
        "out_versions_by_child",
        # Live objects, callables, and rich records (not scalar table cells):
        "input_ops",  # live accessor over parent Op records; needs an attached
        # source trace and raises on disk-loaded logs -- the `parents` label
        # column already carries the relationship.
        "func",
        "grad_fn",
        "grad_fn_handle",
        "grad_fn_object_id",
        "activation_transform",
        "interventions",
        "code_context",
        "transform_fn_source",
        "args_template",
        "kwargs_template",
        "container_spec",
        "parent_param_ops",
        "atomic_module_call",
        "module_entry_arg_keys",
        "annotations",
        # Live-trace-derived views (raise on detached/rehydrated logs; the
        # ``parents`` column plus parent rows carry the same information):
        "input_ops",
        "input_shapes",
        "input_dtypes",
        "input_memory",
        # Bulky state snapshots and internal viz paths:
        "func_rng_states",
        "func_autocast_state",
        "visualizer_path",
    }
)


def _init_module_hierarchy_data() -> dict[str, Any]:
    """Create the transient dict used to accumulate module hierarchy data during logging.

    Consumed by ``_build_module_logs`` (step 17) and then cleared.
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


def _loaded_non_torch_validation_replay_unavailable(trace: Any) -> bool:
    """Return whether loaded non-torch replay validation cannot run.

    Parameters
    ----------
    trace:
        Trace-like object being checked.

    Returns
    -------
    bool
        True for loaded non-torch traces whose backend runtime replay
        capture lists were stripped by portable save.
    """

    if not bool(getattr(trace, "_loaded_from_bundle", False)):
        return False
    backend = str(getattr(trace, "backend", "torch"))
    if backend == _MLX_VALIDATION_REPLAY_BACKEND:
        return True
    if backend == _JAX_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "jax_equation_captures", ()))
    if backend == _TINYGRAD_VALIDATION_REPLAY_BACKEND:
        return not bool(getattr(trace, "tinygrad_uop_captures", ()))
    return False


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


class OrphanAccessor(Accessor[Op]):
    """Dict-like accessor for retained orphan ``Op`` records."""

    def __init__(self, _orphan_labels: Mapping[str, Op] | None = None) -> None:
        """Initialize from raw orphan labels.

        Parameters
        ----------
        _orphan_labels:
            Mapping from raw orphan labels to retained orphan operation logs.
        """

        super().__init__(_orphan_labels or {})

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve by any orphan label variant or unique substring.

        Parameters
        ----------
        key:
            Lookup key or substring.

        Returns
        -------
        Op | None
            Matching orphan operation, or ``None`` if not found or ambiguous.
        """

        exact_matches = [
            orphan
            for orphan in self._dict.values()
            if key
            in {
                orphan.layer_label,
                orphan.layer_label_short,
                orphan.label,
                orphan.label_short,
                orphan.layer_label,
                orphan.layer_label_short,
                orphan._label_raw,
            }
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]

        substring_matches = [
            orphan
            for orphan in self._dict.values()
            if any(
                label is not None and key.lower() in str(label).lower()
                for label in (
                    orphan.layer_label,
                    orphan.layer_label_short,
                    orphan.label,
                    orphan.label_short,
                    orphan.layer_label,
                    orphan.layer_label_short,
                    orphan._label_raw,
                )
            )
        ]
        if len(substring_matches) == 1:
            return substring_matches[0]
        return None

    @property
    def _item_kind(self) -> str:
        """Return display name used in generic ``KeyError`` messages."""

        return "orphan"


class TraceOpAccessor(Accessor[Op]):
    """Trace-level accessor for type-strict Op lookups."""

    def __init__(self, ops: Sequence[Op], layer_num_calls: Mapping[str, int]) -> None:
        """Initialize from ordered Op records.

        Parameters
        ----------
        ops:
            Ordered Op records.
        layer_num_calls:
            Mapping from parent Layer label to number of Op passes.
        """

        op_lookup: OrderedDict[str, Op] = OrderedDict()
        self._raw_index_lookup: dict[int, Op] = {}
        for op in ops:
            op_lookup[op.label] = op
            self._raw_index_lookup[op.raw_index] = op
        super().__init__(op_lookup, item_list=list(ops))
        self._layer_num_calls = dict(layer_num_calls)

    def by_raw_index(self, raw_index: int) -> Op:
        """Return an Op by its realtime raw capture index.

        Parameters
        ----------
        raw_index:
            One-based raw capture index stored on the Op.

        Returns
        -------
        Op
            Matching operation record.
        """

        try:
            return self._raw_index_lookup[raw_index]
        except KeyError as exc:
            raise KeyError(f"Op raw_index {raw_index} not found.") from exc

    def _resolve_pass_qualified(self, key: str) -> Op | None:
        """Resolve pass-qualified Op labels without returning parent Layers."""

        if key in self._dict:
            return self._dict[key]
        return None

    def _resolve_substring(self, key: str) -> Op | None:
        """Resolve exact long/short Op labels or unique bare parent labels."""

        for op in self._list:
            if key in {op.label, op.label_short, op._label_raw, op.raw_label}:
                return op
            if self._layer_num_calls.get(op.layer_label, 0) == 1 and key in {
                op.layer_label,
                op.layer_label_short,
            }:
                return op
        parent_matches = [op for op in self._list if key in {op.layer_label, op.layer_label_short}]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            parent_label = parent_matches[0].layer_label
            qualified = ", ".join(op.label for op in parent_matches[:10])
            suffix = "..." if len(parent_matches) > 10 else ""
            raise AmbiguousOpLookupError(
                f"Layer '{parent_label}' has {len(parent_matches)} ops. Use a 0-based "
                "integer position or a pass-qualified Op label such as "
                f"{qualified}{suffix}."
            )
        return None


class TraceModuleCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict ModuleCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed ModuleCalls."""

        super().__init__(calls)

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare Module address to its only ModuleCall."""

        parent_matches = [call for call in self._list if key == getattr(call, "address", None)]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise ValueError(
                f"Module '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


class TraceGradFnCallAccessor(Accessor[Any]):
    """Trace-level accessor for type-strict GradFnCall lookups."""

    def __init__(self, calls: Mapping[str, Any]) -> None:
        """Initialize from call-label keyed GradFnCalls."""

        super().__init__(calls)

    def _resolve_substring(self, key: str) -> Any | None:
        """Resolve unique bare GradFn label to its only GradFnCall."""

        parent_matches = [call for call in self._list if key == getattr(call, "label", None)]
        if len(parent_matches) == 1:
            return parent_matches[0]
        if len(parent_matches) > 1:
            raise ValueError(
                f"GradFn '{key}' has {len(parent_matches)} calls. Use a 0-based integer "
                f"position or a call-qualified label like '{key}:1'."
            )
        return None


_TRACE_OP_ACCESSOR_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, TraceOpAccessor]] = (
    weakref.WeakKeyDictionary()
)
_TRACE_LAYER_ACCESSOR_CACHE: weakref.WeakKeyDictionary[Any, tuple[int, Any]] = (
    weakref.WeakKeyDictionary()
)
_TRACE_MODULE_CALL_ACCESSOR_CACHE: weakref.WeakKeyDictionary[
    Any, tuple[int, TraceModuleCallAccessor]
] = weakref.WeakKeyDictionary()


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


class _CallableDict(dict[Any, Any]):
    """Dict that returns a plain dict when called.

    This preserves legacy ``log.report_by_type()`` ergonomics for budgeted
    report properties that should not remain inspectable custom_methods.
    """

    def __call__(self) -> dict[Any, Any]:
        """Return a plain-dict copy of this report.

        Returns
        -------
        dict[Any, Any]
            Plain dict containing this report's items.
        """

        return dict(self)


def _legacy_conditional_then_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[str, str]]:
    """Return the legacy THEN-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[str, str]]
        Legacy ``(parent, child)`` THEN-edge view.
    """

    return [
        edge
        for (_conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind == "then"
        for edge in edges
    ]


def _legacy_conditional_elif_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, int, str, str]]:
    """Return the legacy ELIF-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, int, str, str]]
        Legacy ``(cond_id, elif_index, parent, child)`` ELIF-edge view.
    """

    return [
        (conditional_id, int(branch_kind.split("_", 1)[1]), parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind.startswith("elif_")
        for parent, child in edges
    ]


def _legacy_conditional_else_entry_edges(
    conditional_arm_entry_edges: Mapping[Tuple[int, str], List[Tuple[str, str]]],
) -> List[Tuple[int, str, str]]:
    """Return the legacy ELSE-edge view from canonical conditional arm edges.

    Parameters
    ----------
    conditional_arm_entry_edges:
        Canonical ``(cond_id, branch_kind) -> edge list`` mapping.

    Returns
    -------
    List[Tuple[int, str, str]]
        Legacy ``(cond_id, parent, child)`` ELSE-edge view.
    """

    return [
        (conditional_id, parent, child)
        for (conditional_id, branch_kind), edges in conditional_arm_entry_edges.items()
        if branch_kind == "else"
        for parent, child in edges
    ]


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
class Trace(CapturedRun):
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
        if "_build_state" not in self.__dict__ and self.__dict__.get("_tracing_finished", True):
            raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")
        return getattr(self._ensure_build_state(), state_field)

    def __setattr__(self, name: str, value: Any) -> None:
        """Route transient capture attribute writes through private build state."""

        state_field = self._build_state_attr_map().get(name)
        if state_field is None:
            super().__setattr__(name, value)
            return
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
        "_initial_hook_plan": FieldPolicy.DROP,
        "_predicate_save_options": FieldPolicy.DROP,
        "_predicate_history_size": FieldPolicy.DROP,
        "_predicate_history": FieldPolicy.DROP,
        "_predicate_all_contexts": FieldPolicy.DROP,
        "_predicate_lookback": FieldPolicy.DROP,
        "_predicate_lookback_payload_policy": FieldPolicy.DROP,
        "_halt_returns_partial_trace": FieldPolicy.DROP,
        "_predicate_save_decisions": FieldPolicy.DROP,
        "_predicate_contexts_by_label": FieldPolicy.DROP,
        "_predicate_current_contexts": FieldPolicy.DROP,
        "_predicate_lookback_candidates": FieldPolicy.DROP,
        "_postprocessing_active": FieldPolicy.DROP,
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
            emit_nvtx: Whether decorated torch operations should emit NVTX ranges.
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
        self._out_identity_cache: Dict[int, Tuple[torch.Tensor, str, torch.Tensor]] = {}
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
            only for torch traces in this sprint.
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

    def save_intervention(
        self,
        path: str | Path,
        *,
        level: str = "executable_with_callables",
        allow_direct_writes: bool = False,
        overwrite: bool = False,
    ) -> None:
        """Save this log's intervention recipe to a ``.tlspec`` directory.

        Parameters
        ----------
        path:
            Destination ``.tlspec`` directory path.
        level:
            Save level: ``"audit"``, ``"executable_with_callables"``, or
            ``"portable"``.
        allow_direct_writes:
            Whether executable saves may proceed after direct out writes.
        overwrite:
            Whether an existing destination may be replaced.
        """

        from ..intervention.save import save_intervention

        save_intervention(
            self,
            path,
            level=level,
            allow_direct_writes=allow_direct_writes,
            overwrite=overwrite,
        )

    @cached_property
    def intervention_spec(self) -> FrozenInterventionSpec:
        """Return an immutable snapshot of this log's intervention recipe.

        Returns
        -------
        FrozenInterventionSpec
            Frozen public view of the current mutable intervention spec.
        """

        return self._ensure_intervention_spec().freeze()

    def set(
        self,
        site: Any,
        value: Any,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Set a site out recipe without propagating it.

        Parameters
        ----------
        site:
            Selector-like target for the out to replace.
        value:
            Static replacement tensor or one-shot callable accepting the
            matched out and returning a replacement tensor.
        strict:
            Whether site resolution should reject non-portable selectors.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log, with a stale intervention recipe.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.hooks import is_facet_target

        if is_facet_target(site):

            def _facet_replacement_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
                """Return a static or callable facet replacement slice.

                Parameters
                ----------
                out:
                    Facet slice supplied by the facet hook wrapper.
                hook:
                    Hook context supplied by TorchLens.

                Returns
                -------
                torch.Tensor
                    Replacement facet slice.
                """

                del hook
                if callable(value):
                    return cast(torch.Tensor, value(out))
                return cast(torch.Tensor, value)

            self.attach_hooks(
                site,
                _facet_replacement_hook,
                strict=strict,
                confirm_mutation=True,
            )
            self._record_operation(
                "set",
                site=repr(site),
                value_kind=type(value).__name__,
                strict=strict,
                callable=callable(value),
                facet_scatter=True,
            )
            return self
        self._validate_intervention_site(site, strict=strict)
        metadata = {"created_by": "set_callable_one_shot"} if callable(value) else {}
        self._ensure_intervention_spec().add_set(
            self._target_spec_from_site(site, strict=strict),
            value,
            metadata=metadata,
        )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "set",
            site=repr(site),
            value_kind=type(value).__name__,
            strict=strict,
            callable=callable(value),
        )
        return self

    def attach_hooks(
        self,
        hooks_or_site: Any,
        hook: Any = None,
        *extra_hooks: Any,
        strict: bool = False,
        prepend: bool = False,
        confirm_mutation: bool = False,
    ) -> Any:
        """Attach sticky hooks to the current intervention spec.

        Raw PyTorch ``register_forward_hook`` remains supported for users who
        need module-local replacement logic outside this API; during active
        TorchLens captures, returned replacement tensors are instrumented so the
        graph can continue through downstream ops.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        hook:
            Optional hook for the ``(site, hook)`` input shape.
        *extra_hooks:
            Additional hooks to compose at ``hooks_or_site`` in left-to-right order.
        strict:
            Whether site resolution should reject non-portable selectors.
        prepend:
            Whether new sticky hooks should run before existing sticky hooks.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Any
            Scoped removable hook handle.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import HookSignatureError
        from ..intervention.handles import HookHandle
        from ..intervention.hooks import normalize_hook_plan

        if extra_hooks:
            if hook is None:
                raise HookSignatureError("extra hooks require an initial hook argument.")
            entries = normalize_hook_plan(
                [(hooks_or_site, hook_like) for hook_like in (hook, *extra_hooks)]
            )
        else:
            entries = normalize_hook_plan(hooks_or_site, hook)
        from ..intervention.hooks import expand_facet_hook_entries

        entries = expand_facet_hook_entries(self, entries)
        for entry in entries:
            self._validate_intervention_site(entry.site_target, strict=strict)
        spec = self._ensure_intervention_spec()
        handle_ids: list[str] = []
        for entry in entries:
            handle_id = f"hook-{uuid.uuid4().hex}"
            handle_ids.append(handle_id)
            metadata = dict(entry.metadata)
            spec.add_hook(
                self._target_spec_from_site(entry.site_target, strict=strict),
                entry.helper_spec if entry.helper_spec is not None else entry.normalized_callable,
                helper=entry.helper_spec,
                handle=handle_id,
                metadata=metadata,
                prepend=prepend,
            )
        self._mark_intervention_spec_mutated()
        self._record_operation(
            "attach_hooks",
            hook_count=len(entries),
            sites=tuple(repr(entry.site_target) for entry in entries),
            strict=strict,
            prepend=prepend,
            handles=tuple(handle_ids),
        )
        self._last_hook_handle_ids = tuple(handle_ids)
        scoped_handle = HookHandle(self, tuple(handle_ids), confirm_mutation=confirm_mutation)
        if extra_hooks:
            return scoped_handle
        return self

    def remove(self) -> None:
        """Remove the most recent legacy-returned hook attachment.

        Returns
        -------
        None
            Hook specs attached by the last single-hook ``attach_hooks`` call
            are detached.
        """

        for handle_id in self._last_hook_handle_ids:
            self.detach_hooks(handle=handle_id, confirm_mutation=True)
        self._last_hook_handle_ids = ()

    def __enter__(self) -> "Trace":
        """Enter a legacy scoped hook attachment.

        Returns
        -------
        Trace
            This log, acting as the most recent hook handle.
        """

        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Clean up a legacy scoped hook attachment.

        Parameters
        ----------
        exc_type:
            Exception type, if the body raised.
        exc:
            Exception value, if the body raised.
        traceback:
            Exception traceback, if the body raised.
        """

        self.remove()

    def detach_hooks(
        self,
        site: Any = None,
        handle: Any = None,
        *,
        strict: bool = False,
        confirm_mutation: bool = False,
    ) -> "Trace":
        """Detach sticky hooks by site or handle.

        Parameters
        ----------
        site:
            Optional selector-like target. When provided, all sticky hooks for
            that target are removed.
        handle:
            Optional hook handle returned by ``attach_hooks``.
        strict:
            Whether no-op detach requests should raise.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log, with a stale recipe if hooks were removed.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        from ..intervention.errors import SpecMutationError

        if site is None and handle is None:
            if strict:
                raise SpecMutationError("detach_hooks requires a site or handle in strict mode.")
            return self

        target_spec = None
        if site is not None:
            self._validate_intervention_site(site, strict=strict)
            target_spec = self._target_spec_from_site(site, strict=strict)

        handle_values = (
            tuple(getattr(handle, "handle_ids", (str(handle),))) if handle is not None else (None,)
        )
        removed = 0
        for handle_value in handle_values:
            removed += self._ensure_intervention_spec().remove_hook(
                site_target=target_spec,
                handle=handle_value,
            )
        if removed == 0 and strict:
            raise SpecMutationError("detach_hooks did not match any sticky hooks.")
        if removed > 0:
            self._mark_intervention_spec_mutated()
        self._record_operation(
            "detach_hooks",
            site=repr(site) if site is not None else None,
            handle=str(handle) if handle is not None else None,
            removed=removed,
            strict=strict,
        )
        return self

    def clear_hooks(self, *, confirm_mutation: bool = False) -> "Trace":
        """Clear all sticky hooks from the current intervention spec.

        Parameters
        ----------
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.

        Returns
        -------
        Trace
            This model log with hook specs cleared and marked stale.
        """

        self._warn_if_root_mutation(confirm_mutation=confirm_mutation)
        self._ensure_intervention_spec().clear()
        self._mark_intervention_spec_mutated()
        self._record_operation("clear_hooks")
        return self

    def do(
        self,
        hooks_or_site: Any,
        value_or_hook: Any = None,
        *,
        model: nn.Module | None = None,
        x: Any = None,
        engine: str | MissingType = MISSING,
        confirm_mutation: bool | MissingType = MISSING,
        strict: bool | MissingType = MISSING,
        intervention: InterventionOptions | None = None,
    ) -> "Trace":
        """Apply an intervention and dispatch to replay, rerun, or set-only.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional hook for the ``(site, hook)`` input shape.
        model:
            Model required when ``engine="rerun"``.
        x:
            Input required when ``engine="rerun"``.
        engine:
            ``"auto"``, ``"replay"``, ``"rerun"``, or ``"set_only"``.
        confirm_mutation:
            Suppress the once-per-root mutate-in-place warning for callers that
            intentionally mutate this log.
        strict:
            Whether selector and propagation checks should raise.

        Returns
        -------
        Trace
            This model log after the selected propagation engine runs.
        """

        from ..intervention.errors import EngineDispatchError

        intervention_options = merge_intervention_options(
            intervention=intervention,
            engine=engine,
            confirm_mutation=confirm_mutation,
            strict=strict,
        )
        engine_value = intervention_options.engine
        confirm_mutation_value = intervention_options.confirm_mutation
        strict_value = intervention_options.strict

        if engine_value not in {"auto", "replay", "rerun", "set_only"}:
            raise ValueError(
                "do(..., engine=...) must be 'auto', 'replay', 'rerun', or 'set_only'."
            )

        selected_engine = self._select_do_engine(engine_value, model=model, x=x)
        if selected_engine == "rerun":
            if model is None:
                raise EngineDispatchError("do(..., engine='rerun') requires model= and x=.")
            self._validate_supplied_model_matches_capture(model)
        mutation_kind = self._apply_do_mutation(
            hooks_or_site,
            value_or_hook,
            engine=selected_engine,
            strict=strict_value,
            confirm_mutation=confirm_mutation_value,
        )
        self._record_operation(
            "do",
            mutation_kind=mutation_kind,
            engine=selected_engine,
            requested_engine=engine_value,
            model_supplied=model is not None,
            x_supplied=x is not None,
            strict=strict_value,
        )

        if selected_engine == "set_only":
            return self
        if selected_engine == "replay":
            return self.replay(strict=strict_value)
        assert model is not None
        return self.rerun(model, x, strict=strict_value)

    def fork(self, name: str | None = None) -> "Trace":
        """Create a copy-on-write intervention fork of this log.

        Parameters
        ----------
        name:
            Optional name for the forked log.

        Returns
        -------
        Trace
            Forked model log.
        """

        fork = self._fork_trace(name=name)
        self._record_operation("fork", source_id=id(self), name=fork.trace_label)
        return fork

    def _record_operation(self, op: str, **payload: Any) -> None:
        """Append a structured operation record to ``state_history``.

        Parameters
        ----------
        op:
            Operation name.
        **payload:
            Operation-specific metadata.

        Returns
        -------
        None
            The history list is mutated in place.
        """

        self.state_history.append(
            {
                "op": op,
                "spec_revision": self._spec_revision,
                "timestamp": time.monotonic(),
                **payload,
            }
        )

    def _warn_if_root_mutation(self, *, confirm_mutation: bool) -> None:
        """Emit the once-per-root mutate-in-place warning when appropriate.

        Parameters
        ----------
        confirm_mutation:
            Whether the caller explicitly accepted in-place mutation.
        """

        if confirm_mutation or self.parent_run is not None or self._warned_mutate_in_place:
            return
        from ..intervention.errors import MutateInPlaceWarning
        from ..options import suppress_mutate_warnings

        if suppress_mutate_warnings.is_suppressed:
            return
        warnings.warn(
            "MutateInPlaceWarning: Trace mutators modify root logs in place. "
            "Use log.fork(...) for isolated edits or pass confirm_mutation=True.",
            MutateInPlaceWarning,
            stacklevel=3,
        )
        self._warned_mutate_in_place = True

    def _select_do_engine(self, engine: str, *, model: nn.Module | None, x: Any) -> str:
        """Resolve the concrete ``do`` engine from caller arguments.

        Parameters
        ----------
        engine:
            Requested engine name.
        model:
            Optional model supplied for rerun.
        x:
            Optional input supplied for rerun.

        Returns
        -------
        str
            Concrete engine name.

        Raises
        ------
        EngineDispatchError
            If the engine cannot be inferred from an incomplete model/input pair.
        """

        from ..intervention.errors import EngineDispatchError

        if engine != "auto":
            if engine == "rerun" and (model is None or x is None):
                raise EngineDispatchError(
                    "do(..., engine='rerun') requires both model= and x=. "
                    "Pass both, or use engine='replay' if full rerun is not intended."
                )
            return engine
        if (model is None) != (x is None):
            raise EngineDispatchError(
                "do(engine='auto') needs both model= and x= for rerun, or neither for "
                "replay. Pass both, or use engine='replay' if rerun is not intended."
            )
        return "rerun" if model is not None else "replay"

    def _apply_do_mutation(
        self,
        hooks_or_site: Any,
        value_or_hook: Any,
        *,
        engine: str,
        strict: bool,
        confirm_mutation: bool,
    ) -> str:
        """Apply the mutation part of ``do`` and report its kind.

        Parameters
        ----------
        hooks_or_site:
            Mapping/list batch input or selector-like site.
        value_or_hook:
            Optional value or hook.
        engine:
            Concrete engine selected by ``_select_do_engine``.
        strict:
            Whether selector checks should be strict.
        confirm_mutation:
            Whether root mutation warnings should be suppressed.

        Returns
        -------
        str
            ``"set"`` or ``"attach_hooks"``.
        """

        if engine == "set_only" and value_or_hook is not None:
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        if value_or_hook is not None and not callable(value_or_hook):
            self.set(
                hooks_or_site,
                value_or_hook,
                strict=strict,
                confirm_mutation=confirm_mutation,
            )
            return "set"
        self.attach_hooks(
            hooks_or_site,
            value_or_hook,
            strict=strict,
            confirm_mutation=confirm_mutation,
        )
        return "attach_hooks"

    def _validate_supplied_model_matches_capture(self, model: nn.Module) -> None:
        """Validate rerun model evidence against the captured source model.

        Parameters
        ----------
        model:
            Candidate model for rerun.

        Raises
        ------
        ModelMismatchError
            If available class or weight-fingerprint evidence differs.
        """

        from ..intervention.errors import ModelMismatchError
        from ..user_funcs import _fingerprint_model_weights, _qualname_for_model

        expected_class = getattr(self, "model_class_qualname", None)
        actual_class = _qualname_for_model(model)
        if expected_class is not None and actual_class != expected_class:
            raise ModelMismatchError(
                "Supplied model class does not match captured model class: "
                f"expected {expected_class!r}, got {actual_class!r}."
            )

        expected_fingerprint = getattr(self, "param_hash_quick", None)
        if expected_fingerprint is None:
            return
        actual_fingerprint = _fingerprint_model_weights(model)
        if actual_fingerprint != expected_fingerprint:
            raise ModelMismatchError(
                "Supplied model weight fingerprint does not match captured model weights."
            )

    def _fork_trace(
        self,
        *,
        name: str | None,
        deep_copy_layer_labels: Set[str] | None = None,
    ) -> "Trace":
        """Build a forked Trace with policy-driven field handling.

        Parameters
        ----------
        name:
            Optional fork name.
        deep_copy_layer_labels:
            Optional layer labels that require full Op field copies. When
            provided, other Op shells are still forked but their fields use a
            shallow copy to avoid deep-copying untouched replay context.

        Returns
        -------
        Trace
            Forked log whose mutable containers are independent.
        """

        fork = state_new(type(self))
        fork_state = {
            field_name: self._fork_model_field(field_name, value)
            for field_name, value in state_items(self)
        }
        state_restore(fork, fork_state)
        fork.parent_run = weakref.ref(self)
        fork.trace_label = name or self._next_fork_name()
        fork._intervention_spec = copy.deepcopy(self._ensure_intervention_spec())
        fork.state_history = copy.deepcopy(self.state_history)
        fork.relationship_evidence = copy.deepcopy(self.relationship_evidence)
        fork._out_recipe_revision = self._out_recipe_revision
        fork._spec_revision = self._spec_revision
        fork.state = self.state
        fork._warned_mutate_in_place = False
        fork._warned_direct_write = False

        layer_map = fork._fork_layer_ops_from(
            self,
            deep_copy_layer_labels=deep_copy_layer_labels,
        )
        fork._rebuild_fork_layer_collections(self, layer_map)
        fork._rebind_fork_owner_refs()
        _state._register_log(fork)
        return fork

    def _next_fork_name(self) -> str:
        """Return a deterministic default fork name for this parent log."""

        base_name = self.trace_label or "trace"
        fork_count = sum(
            1
            for record in self.state_history
            if isinstance(record, dict) and record.get("op") == "fork"
        )
        return f"{base_name}_fork_{fork_count + 1}"

    def _fork_model_field(self, field_name: str, value: Any) -> Any:
        """Apply the Trace fork policy to a single field.

        Parameters
        ----------
        field_name:
            Field being copied.
        value:
            Current field value.

        Returns
        -------
        Any
            Field value for the fork.
        """

        policy = MODEL_LOG_FIELD_FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _fork_layer_ops_from(
        self,
        parent: "Trace",
        *,
        deep_copy_layer_labels: Set[str] | None = None,
    ) -> dict[int, Op]:
        """Fork every Op and return an old-object-id map.

        Parameters
        ----------
        parent:
            Parent log whose layer ops are being forked.
        deep_copy_layer_labels:
            Optional layer labels that require normal fork policy copies. Ops
            outside this set receive distinct shells with shallow-copied fields.

        Returns
        -------
        dict[int, Op]
            Mapping from ``id(parent_pass)`` to forked pass.
        """

        layer_map: dict[int, Op] = {}
        fork_equivalent_ops = self.op_equivalence_classes
        for parent_pass in parent.layer_list:
            fork_pass = state_new(Op)
            deep_copy_fields = (
                deep_copy_layer_labels is None or parent_pass.layer_label in deep_copy_layer_labels
            )
            state_restore(
                fork_pass,
                {
                    field_name: self._fork_layer_pass_field(
                        field_name,
                        value,
                        deep_copy=deep_copy_fields,
                    )
                    for field_name, value in state_items(parent_pass)
                },
            )
            fork_pass.source_trace = self
            eq_type = getattr(fork_pass, "equivalence_class", None)
            if eq_type in fork_equivalent_ops:
                fork_pass.equivalent_ops = fork_equivalent_ops[eq_type]
            object.__setattr__(fork_pass, "_construction_done", True)
            layer_map[id(parent_pass)] = fork_pass
        return layer_map

    def _fork_layer_pass_field(self, field_name: str, value: Any, *, deep_copy: bool = True) -> Any:
        """Apply the Op fork policy to a single field.

        Parameters
        ----------
        field_name:
            Op field being copied.
        value:
            Current field value.
        deep_copy:
            Whether to honor the normal copy policy. False shares field values
            for untouched differentiable-replay ops while still creating a
            distinct Op shell.

        Returns
        -------
        Any
            Field value for the forked pass.
        """

        if field_name == "_source_trace_ref":
            return None
        if not deep_copy:
            return self._copy_shallow_fork_value(value)
        policy = Op.FIELD_FORK_POLICY.get(field_name, self._default_fork_policy(value))
        if policy is ForkFieldPolicy.FORK_SHARE:
            return value
        if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
            return None
        return self._copy_fork_value(value)

    def _rebuild_fork_layer_collections(self, parent: "Trace", layer_map: dict[int, Op]) -> None:
        """Rebuild layer lookup containers so they point at forked ops.

        Parameters
        ----------
        parent:
            Parent log whose containers are being mirrored.
        layer_map:
            Mapping from parent pass object id to forked pass.
        """

        def remap_pass(value: Any) -> Any:
            """Map a parent pass object to its forked counterpart.

            Parameters
            ----------
            value:
                Candidate parent-layer object or another value.

            Returns
            -------
            Any
                Forked layer pass when ``value`` is known, otherwise ``value``.
            """
            return layer_map.get(id(value), value)

        self.layer_list = [remap_pass(layer) for layer in parent.layer_list]
        self.layer_dict_main_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_main_keys.items()
        )
        self.layer_dict_all_keys = OrderedDict(
            (key, remap_pass(layer)) for key, layer in parent.layer_dict_all_keys.items()
        )
        fork_layer_logs: dict[str, Layer] = OrderedDict()
        for label, parent_layer in parent.layer_logs.items():
            fork_layer_log = state_new(type(parent_layer))
            state_restore(
                fork_layer_log,
                {key: self._copy_fork_value(value) for key, value in state_items(parent_layer)},
            )
            fork_layer_log.source_trace = self
            fork_layer_log.ops = OpAccessor(
                OrderedDict(
                    (call_index, remap_pass(layer_pass))
                    for call_index, layer_pass in parent_layer.ops.items()
                )
            )
            if getattr(fork_layer_log, "equivalence_class", None) in self.op_equivalence_classes:
                fork_layer_log.equivalent_ops = self.op_equivalence_classes[
                    fork_layer_log.equivalence_class
                ]
            fork_layer_logs[label] = fork_layer_log
        self.layer_logs = fork_layer_logs

    def _rebind_fork_owner_refs(self) -> None:
        """Rebind weak owner references on forked child objects to this fork."""

        for layer_pass in self.layer_list:
            layer_pass.source_trace = self
            del layer_pass.facets
        for layer_log in self.layer_logs.values():
            layer_log.source_trace = self
        for module_log in self.modules:
            module_log._source_trace = self
            module_log.__dict__.pop("_facets_cache", None)
            for module_call in module_log.calls.values():
                module_call._source_trace = self

    @staticmethod
    def _copy_fork_value(value: Any) -> Any:
        """Copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.

        Returns
        -------
        Any
            Fork-safe copy.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        try:
            return copy.deepcopy(value)
        except Exception:
            return copy.copy(value)

    @staticmethod
    def _copy_shallow_fork_value(value: Any) -> Any:
        """Shallow-copy a fork field while preserving tensor and callable identity.

        Parameters
        ----------
        value:
            Value to copy.

        Returns
        -------
        Any
            Lightweight fork-safe copy for replay-unaffected Op fields.
        """

        if isinstance(value, torch.Tensor) or callable(value):
            return value
        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return value
        try:
            return copy.copy(value)
        except Exception:
            return value

    @staticmethod
    def _default_fork_policy(value: Any) -> ForkFieldPolicy:
        """Choose a conservative fork policy for fields outside policy tables.

        Parameters
        ----------
        value:
            Field value without an explicit policy.

        Returns
        -------
        ForkFieldPolicy
            Default share/copy decision.
        """

        if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
            return ForkFieldPolicy.FORK_SHARE
        if isinstance(value, torch.Tensor) or callable(value):
            return ForkFieldPolicy.FORK_SHARE
        return ForkFieldPolicy.FORK_COPY

    def _recipe_is_clean(self) -> bool:
        """Return whether propagated outs match the current spec revision.

        Returns
        -------
        bool
            ``True`` when the current out recipe revision equals the
            mutable intervention spec revision.
        """

        return self._spec_revision == self._out_recipe_revision

    def _ensure_intervention_spec(self) -> InterventionSpec:
        """Return the mutable intervention spec, creating one if needed.

        Returns
        -------
        InterventionSpec
            Mutable intervention recipe owned by this log.
        """

        if self._intervention_spec is None:
            self._intervention_spec = InterventionSpec()
        return self._intervention_spec

    def _mark_intervention_spec_mutated(self) -> None:
        """Invalidate cached frozen views and mark the spec stale.

        Returns
        -------
        None
            This model log is mutated in place.
        """

        self._spec_revision += 1
        self.__dict__.pop("intervention_spec", None)
        self.__dict__.pop("_frozen_intervention_spec", None)
        self.__dict__.pop("_cached_frozen_intervention_spec", None)
        self.state = TraceState.SPEC_STALE

    def _validate_intervention_site(self, site: Any, *, strict: bool) -> None:
        """Validate that a mutator site resolves on this log.

        Parameters
        ----------
        site:
            Selector-like target to validate.
        strict:
            Whether selector resolution should be strict.

        Returns
        -------
        None
            Raises when the site cannot resolve.
        """

        from ..intervention.resolver import _selector_resolution_direction

        try:
            if _selector_resolution_direction(site) == "backward":
                return
        except Exception:
            pass
        max_fanout = max(1, len(self.layer_list))
        self.resolve_sites(site, strict=strict, max_fanout=max_fanout)

    def _target_spec_from_site(self, site: Any, *, strict: bool) -> TargetSpec:
        """Convert a selector-like site to a mutable target spec.

        Parameters
        ----------
        site:
            Selector-like target, target spec, or layer pass.
        strict:
            Whether the resulting target should carry strict resolution.

        Returns
        -------
        TargetSpec
            Mutable target spec stored in the intervention recipe.
        """

        if isinstance(site, TargetSpec):
            target = copy.copy(site)
            target.strict = strict or target.strict
            return target
        if hasattr(site, "to_target_spec"):
            target = site.to_target_spec()
            target.strict = strict or target.strict
            return cast("TargetSpec", target)
        if hasattr(site, "layer_label"):
            return TargetSpec("label", str(site.layer_label), strict=strict)
        return TargetSpec("label", site, strict=strict)

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
        state["_backward_gradfn_refs"] = {}
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
        default_fill_state(
            state,
            defaults={
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
                "_postprocessing_active": False,
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
        _TRACE_OP_ACCESSOR_CACHE.pop(self, None)
        _TRACE_LAYER_ACCESSOR_CACHE.pop(self, None)
        _TRACE_MODULE_CALL_ACCESSOR_CACHE.pop(self, None)
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
        _TRACE_OP_ACCESSOR_CACHE.pop(self, None)
        _TRACE_LAYER_ACCESSOR_CACHE.pop(self, None)
        _TRACE_MODULE_CALL_ACCESSOR_CACHE.pop(self, None)
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
    # ********** Computed Properties *************
    # ********************************************

    @property
    def conditional_then_entry_edges(self) -> List[Tuple[str, str]]:
        """Deprecated THEN-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[str, str]]
            Legacy ``(parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_then_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_then_entry_edges(self.conditional_arm_entry_edges)

    @conditional_then_entry_edges.setter
    def conditional_then_entry_edges(self, value: List[Tuple[str, str]]) -> None:
        """Set the deprecated THEN-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(parent, child)`` edge list. Edges are assigned to
            conditional id 0 because the legacy view did not carry ids.
        """

        warn_deprecated_alias("conditional_then_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if key[1] != "then"
        }
        if value:
            self.conditional_arm_entry_edges[(0, "then")] = list(value)

    @property
    def conditional_elif_entry_edges(self) -> List[Tuple[int, int, str, str]]:
        """Deprecated ELIF-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[int, int, str, str]]
            Legacy ``(cond_id, elif_index, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_elif_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_elif_entry_edges(self.conditional_arm_entry_edges)

    @conditional_elif_entry_edges.setter
    def conditional_elif_entry_edges(self, value: List[Tuple[int, int, str, str]]) -> None:
        """Set the deprecated ELIF-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, elif_index, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_elif_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if not key[1].startswith("elif_")
        }
        for conditional_id, elif_index, parent, child in value:
            self.conditional_arm_entry_edges.setdefault(
                (conditional_id, f"elif_{elif_index}"), []
            ).append((parent, child))

    @property
    def conditional_else_entry_edges(self) -> List[Tuple[int, str, str]]:
        """Deprecated ELSE-edge view derived from ``conditional_arm_entry_edges``.

        Returns
        -------
        List[Tuple[int, str, str]]
            Legacy ``(cond_id, parent, child)`` edge view.
        """

        warn_deprecated_alias("conditional_else_entry_edges", "conditional_arm_entry_edges")
        return _legacy_conditional_else_entry_edges(self.conditional_arm_entry_edges)

    @conditional_else_entry_edges.setter
    def conditional_else_entry_edges(self, value: List[Tuple[int, str, str]]) -> None:
        """Set the deprecated ELSE-edge view by updating canonical arm edges.

        Parameters
        ----------
        value:
            Legacy ``(cond_id, parent, child)`` edge list.
        """

        warn_deprecated_alias("conditional_else_entry_edges", "conditional_arm_entry_edges")
        self.conditional_arm_entry_edges = {
            key: edges
            for key, edges in self.conditional_arm_entry_edges.items()
            if key[1] != "else"
        }
        for conditional_id, parent, child in value:
            self.conditional_arm_entry_edges.setdefault((conditional_id, "else"), []).append(
                (parent, child)
            )

    @property
    def is_recurrent(self) -> bool:
        """Whether any layer has more than one pass."""
        return any(v > 1 for v in self.layer_num_calls.values())

    @property
    def recurrent_layers(self) -> "LayerAccessor":
        """Access Layers with more than one captured pass.

        Returns
        -------
        LayerAccessor
            Accessor containing aggregate Layer records whose ``num_passes`` is
            greater than 1.
        """
        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer) for label, layer in self.layer_logs.items() if layer.num_passes > 1
            ),
            source_trace=self,
        )

    @property
    def max_layer_op_count(self) -> int:
        """Maximum number of ops for any layer."""
        return max(self.layer_num_calls.values(), default=1)

    @property
    def is_branching(self) -> bool:
        """Whether any layer has more than one child."""
        return any(len(entry.children) > 1 for entry in self.layer_list)

    @property
    def has_conditional_branching(self) -> bool:
        """Whether any layer is in a conditional branch."""
        return any(entry.is_in_conditional_body for entry in self.layer_list)

    @property
    def has_conditionals(self) -> bool:
        """Whether this Trace contains conditional-flow records."""

        return len(self.conditionals) > 0

    @property
    def num_conditionals(self) -> int:
        """Number of conditional-flow records."""

        return len(self.conditionals)

    @property
    def is_dynamic_graph(self) -> bool:
        """Whether execution depends on runtime tensor values."""

        return self.has_conditionals

    @property
    def forward_source_location(self) -> str | None:
        """Combined forward source location."""

        if self.forward_source_file is None or self.forward_source_line is None:
            return None
        return f"{self.forward_source_file}:{self.forward_source_line}"

    @property
    def class_source_location(self) -> str | None:
        """Combined model class source location."""

        if self.class_source_file is None or self.class_source_line is None:
            return None
        return f"{self.class_source_file}:{self.class_source_line}"

    @property
    def init_source_location(self) -> str | None:
        """Combined model ``__init__`` source location."""

        if self.init_source_file is None or self.init_source_line is None:
            return None
        return f"{self.init_source_file}:{self.init_source_line}"

    @property
    def num_tensors(self) -> int:
        """Total number of tensor operations."""
        return len(self)

    @property
    def last_backward_duration(self) -> Duration | None:
        """Most recent backward-pass duration, if any."""

        if not self.backward_durations:
            return None
        return as_duration(self.backward_durations[-1])

    @property
    def total_backward_duration(self) -> Duration:
        """Sum of all captured backward-pass durations."""

        return Duration(sum(self.backward_durations))

    @property
    def last_backward_root_grad_fn_object_id(self) -> int | None:
        """Most recent backward root grad_fn_handle object id, if any."""

        if not self.backward_root_grad_fn_object_ids:
            return None
        return self.backward_root_grad_fn_object_ids[-1]

    @property
    def overhead_duration(self) -> Duration:
        """Time spent on TorchLens overhead (total minus function calls)."""
        return self.capture_duration - self.func_calls_duration

    @property
    def capture_duration(self) -> Duration:
        """Total capture-phase duration in seconds."""

        if not self.capture_start_time or not self.capture_end_time:
            return Duration(0)
        return Duration(self.capture_end_time - self.capture_start_time)

    # ********************************************
    # ************* FLOPs Properties *************
    # ********************************************
    # FLOPs are estimated per-operation during logging (flops_forward,
    # flops_backward on each Op).  These properties aggregate
    # across the entire model.  Layers with None FLOPs (unknown ops) are
    # skipped, so the totals may undercount.

    @property
    def total_flops_forward(self) -> Flops:
        """Total forward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(entry.flops_forward for entry in self.layer_list if entry.flops_forward is not None)
        )

    @property
    def total_flops_backward(self) -> Flops:
        """Total backward FLOPs across all layers (skipping None/unknown)."""
        return Flops(
            sum(
                entry.flops_backward
                for entry in self.layer_list
                if entry.flops_backward is not None
            )
        )

    @property
    def total_flops(self) -> Flops:
        """Total FLOPs (forward + backward)."""
        return Flops(self.total_flops_forward + self.total_flops_backward)

    @property
    def flops_by_op_type(self) -> _CallableDict:
        """Group FLOPs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int | Flops]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": Flops(0), "backward": Flops(0), "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += entry.flops_forward
            if entry.flops_backward is not None:
                result[lt]["backward"] += entry.flops_backward
        return _CallableDict(result)

    # ********************************************
    # ************** MACs Properties *************
    # ********************************************
    # MACs (multiply-accumulate operations) = FLOPs / 2.

    @property
    def total_macs_forward(self) -> Macs:
        """Total forward MACs across all layers (skipping None/unknown)."""
        return Macs(self.total_flops_forward // 2)

    @property
    def total_macs_backward(self) -> Macs:
        """Total backward MACs across all layers (skipping None/unknown)."""
        return Macs(self.total_flops_backward // 2)

    @property
    def total_macs(self) -> Macs:
        """Total MACs (forward + backward)."""
        return Macs(self.total_flops // 2)

    @property
    def macs_by_op_type(self) -> _CallableDict:
        """Group MACs by layer type.

        Returns:
            Callable dict mapping layer_type to forward/backward/count totals.
        """
        result: Dict[str, Dict[str, int | Macs]] = {}
        for entry in self.layer_list:
            lt = entry.layer_type
            if lt not in result:
                result[lt] = {"forward": Macs(0), "backward": Macs(0), "count": 0}
            result[lt]["count"] += 1
            if entry.flops_forward is not None:
                result[lt]["forward"] += Macs(entry.flops_forward // 2)
            if entry.flops_backward is not None:
                result[lt]["backward"] += Macs(entry.flops_backward // 2)
        return _CallableDict(result)

    # ********************************************
    # ************* Params Accessor **************
    # ********************************************

    @property
    def params(self) -> ParamAccessor:
        """Access parameter metadata by address, short name, or index."""
        return self.param_logs

    @property
    def ops(self) -> TraceOpAccessor:
        """Access per-invocation Op records by label or index."""

        cache_key = len(self.layer_list)
        cache_entry = _TRACE_OP_ACCESSOR_CACHE.get(self)
        if cache_entry is None or cache_entry[0] != cache_key:
            accessor = TraceOpAccessor(self.layer_list, self.layer_num_calls)
            _TRACE_OP_ACCESSOR_CACHE[self] = (cache_key, accessor)
            return accessor
        return cache_entry[1]

    @property
    def transforms(self) -> tuple[Op, ...]:
        """Return transform-boundary operation records.

        Returns
        -------
        tuple[Op, ...]
            Ops whose ``is_transform`` role flag is true.
        """

        return tuple(op for op in self.ops if getattr(op, "is_transform", False))

    @property
    def model_profile(self) -> ModelProfile:
        """Return a computed semantic I/O profile descriptor.

        Returns
        -------
        ModelProfile
            Runtime-only descriptor derived from preprocessing, raw-input, and
            output-label metadata. This property is not persisted.
        """

        input_preprocessor = self.input_preprocessor
        output_postprocessor = self.output_postprocessor
        input_source = input_preprocessor.source if input_preprocessor is not None else None
        output_source = output_postprocessor.source if output_postprocessor is not None else None
        has_output_labels = bool(self.output_id2label)
        output_label_count = (
            len(self.output_id2label)
            if self.output_id2label is not None
            else self.output_num_classes
        )
        has_raw_images = _raw_input_contains_images(self.raw_input)
        input_modality = _infer_input_modality(self.raw_input, input_source)
        return ModelProfile(
            input_modality=input_modality,
            input_preprocessing_source=input_source,
            output_postprocessing_source=output_source,
            output_label_count=output_label_count,
            has_output_labels=has_output_labels,
            num_stimuli=_raw_input_num_stimuli(self.raw_input),
            has_raw_images=has_raw_images,
            keystone_applicable=bool(
                input_modality == "image"
                and has_raw_images
                and has_output_labels
                and output_label_count is not None
            ),
        )

    @property
    def layers(self) -> "LayerAccessor":
        """Access aggregate per-layer metadata by label, index, or pass notation."""
        from .layer import LayerAccessor

        cache_key = len(self.layer_logs)
        cache_entry = _TRACE_LAYER_ACCESSOR_CACHE.get(self)
        if cache_entry is None or cache_entry[0] != cache_key:
            accessor = LayerAccessor(self.layer_logs, source_trace=self)
            _TRACE_LAYER_ACCESSOR_CACHE[self] = (cache_key, accessor)
            return accessor
        return cast("LayerAccessor", cache_entry[1])

    @property
    def modules(self) -> "ModuleAccessor":
        """Access structured per-module metadata by address, index, or pass notation."""
        return self._module_logs

    def modules_with_facet(self, name: str) -> Iterator[Any]:
        """Yield Modules whose semantic facet view has a facet available now.

        Parameters
        ----------
        name:
            Facet name to look up using the current capture's available values.
        """

        return (module for module in self.modules if module.facets.has(name))

    def attention_blocks(self) -> Iterator[Any]:
        """Yield Modules with an available query-projection facet."""

        return self.modules_with_facet("q")

    @property
    def module_calls(self) -> TraceModuleCallAccessor:
        """Access per-invocation ModuleCall records by call label or index."""

        num_calls = sum(len(module.calls) for module in self._module_logs)
        cache_entry = _TRACE_MODULE_CALL_ACCESSOR_CACHE.get(self)
        if cache_entry is not None and cache_entry[0] == num_calls:
            return cache_entry[1]
        calls: OrderedDict[str, Any] = OrderedDict()
        for module in self._module_logs:
            for call in module.calls.values():
                calls[call.call_label] = call
        accessor = TraceModuleCallAccessor(calls)
        _TRACE_MODULE_CALL_ACCESSOR_CACHE[self] = (num_calls, accessor)
        return accessor

    @property
    def num_module_calls(self) -> int:
        """Total number of module invocations recorded in this Trace."""

        return len(self.module_calls)

    def _root_call(self) -> "ModuleCall":
        """Return the top-level ModuleCall for internal call-tree traversal.

        When multiple top-level calls exist, the first one in trace insertion
        order is returned.
        """

        for call in self.module_calls:
            if call.call_parent is None:
                return cast("ModuleCall", call)
        raise RuntimeError("Trace has no root ModuleCall")

    def walk_calls(self) -> Iterator["ModuleCall"]:
        """Yield every ModuleCall in call-tree depth-first order.

        Yields
        ------
        ModuleCall
            ModuleCall records ordered by dynamic call-tree traversal.
        """

        if not self.module_calls:
            return
        yield from self._root_call().walk_descendants(include_self=True)

    def show_call_tree(
        self,
        max_depth: int | None = None,
        include_atomic: bool = True,
        show_call_index: bool = True,
        file: TextIO | None = None,
    ) -> None:
        """Print this Trace's full ModuleCall tree as an ASCII tree.

        Parameters
        ----------
        max_depth:
            Maximum descendant depth to print, or ``None`` for no limit.
        include_atomic:
            Whether to include atomic module leaves.
        show_call_index:
            Whether to include the ``:N`` call suffix in labels.
        file:
            Optional output stream. ``None`` prints to stdout.
        """

        if not self.module_calls:
            return
        self._root_call().show_call_tree(
            max_depth=max_depth,
            include_atomic=include_atomic,
            show_call_index=show_call_index,
            file=file,
        )

    @property
    def root_module(self) -> "Module":
        """The root module (the model itself)."""
        return cast("Module", self._module_logs["self"])

    @property
    def num_layers(self) -> int:
        """Number of distinct Layer records in this Trace."""

        return len(self.layers)

    @property
    def num_compute_layers(self) -> int:
        """Number of compute Layers in this Trace."""

        return len(self.compute_layers)

    @property
    def num_compute_ops(self) -> int:
        """Number of compute Ops in this Trace."""

        return len(self.compute_ops)

    @property
    def num_edges(self) -> int:
        """Distinct edges in the per-pass Op graph, including boundary edges."""

        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.ops
                for child_label in op.children
            }
        )

    @property
    def num_compute_edges(self) -> int:
        """Distinct Op graph edges whose endpoints are both compute Ops."""

        compute_labels = {op.label for op in self.compute_ops}
        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.compute_ops
                for child_label in op.children
                if self.ops[child_label].label in compute_labels
            }
        )

    @property
    def num_buffer_edges(self) -> int:
        """Distinct Op graph edges with at least one buffer endpoint."""

        return len(
            {
                (op.label, self.ops[child_label].label)
                for op in self.ops
                for child_label in op.children
                if op.is_buffer or self.ops[child_label].is_buffer
            }
        )

    @property
    def num_layer_edges(self) -> int:
        """Distinct edges in the aggregate Layer graph."""

        return len(
            {
                (layer.layer_label, child_label)
                for layer in self.layers
                for child_label in layer.children
            }
        )

    @property
    def num_backward_edges(self) -> int | None:
        """Distinct edges in the backward GradFn graph, or ``None`` if ungated."""

        if not self.has_backward_pass:
            return None
        return len(
            {
                (grad_fn.label, child_label)
                for grad_fn in self.grad_fns
                for child_label in grad_fn.children
            }
        )

    @property
    def branching_factor(self) -> float:
        """Mean fan-out: children (consumers) per compute Op.

        Computed over a single, consistent node set (compute Ops) so the ratio is
        a coherent mean out-degree: ~1.0 for a plain chain, >1.0 when ops are
        reused (residual streams, shared embeddings, dense connectivity). Counts
        every child of each compute Op (including boundary/buffer consumers).
        """

        compute_ops = list(self.compute_ops)
        if not compute_ops:
            return 0.0
        return sum(op.num_children for op in compute_ops) / len(compute_ops)

    @property
    def max_in_degree(self) -> int:
        """Maximum number of parents over compute Ops."""

        return max((op.num_parents for op in self.compute_ops), default=0)

    @property
    def max_out_degree(self) -> int:
        """Maximum number of children over compute Ops."""

        return max((op.num_children for op in self.compute_ops), default=0)

    @property
    def num_saved_grad_ops(self) -> int:
        """Number of Ops with saved gradients."""

        return len(self.saved_grad_ops)

    @property
    def intermediate_derived_grads(self) -> IntermediateDerivedGradAccessor:
        """Access exact op-level derived gradient records, when a backend provides them.

        Returns
        -------
        IntermediateDerivedGradAccessor
            Records keyed by pass-qualified op label. Backends that did not run
            an intermediate-derived-gradient pass return an empty accessor.
        """

        records = self.__dict__.get("_intermediate_derived_grads")
        if isinstance(records, IntermediateDerivedGradAccessor):
            return records
        return IntermediateDerivedGradAccessor()

    @intermediate_derived_grads.setter
    def intermediate_derived_grads(self, value: IntermediateDerivedGradAccessor) -> None:
        """Store exact op-level derived gradient records.

        Parameters
        ----------
        value
            Accessor to expose through ``trace.intermediate_derived_grads``.
        """

        self.__dict__["_intermediate_derived_grads"] = value

    @property
    def num_saved_grad_layers(self) -> int:
        """Number of Layers containing at least one saved-gradient Op."""

        return len(self.saved_grad_layers)

    @property
    def num_param_tensors_trainable(self) -> int:
        """Number of trainable parameter tensors in this Trace."""

        return sum(1 for param in self.params if param.is_trainable)

    @property
    def num_param_tensors_frozen(self) -> int:
        """Number of frozen parameter tensors in this Trace."""

        return sum(1 for param in self.params if not param.is_trainable)

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
    def num_modules(self) -> int:
        """Total number of registered source-model submodules."""

        return len(self._module_logs)

    @num_modules.deleter
    def num_modules(self) -> None:
        """Ignore cleanup deletion for derived module count."""

    @property
    def orphans(self) -> OrphanAccessor:
        """Access retained orphan island operations by raw or final label."""

        orphan_dict = OrderedDict(
            (log.layer_label, log) for log in self._orphan_logs if getattr(log, "is_orphan", False)
        )
        return OrphanAccessor(orphan_dict)

    @property
    def grad_fns(self) -> GradFnAccessor:
        """Access backward grad_fn_handle metadata by label, index, pass label, or substring."""
        self._sync_backward_projection_if_needed()
        return GradFnAccessor(self.grad_fn_logs, self.grad_fn_order)

    @property
    def grad_fn_calls(self) -> TraceGradFnCallAccessor:
        """Access per-invocation GradFnCall records by qualified label or index."""

        self._sync_backward_projection_if_needed()
        calls: OrderedDict[str, Any] = OrderedDict()
        for grad_fn_handle in self.grad_fns:
            for call_index, call in grad_fn_handle.calls.items():
                call.source_trace = self
                calls[f"{grad_fn_handle.label}:{call_index}"] = call
        return TraceGradFnCallAccessor(calls)

    @property
    def backward_passes(self) -> BackwardPassAccessor:
        """Access backward pass records by 0-based position or named pass number."""

        if getattr(self, "backend", "torch") in {"jax", "mlx", "tinygrad"}:
            raise ValueError(
                f"{getattr(self, 'backend', 'backend')} traces do not support true backward "
                "capture. Use trace.derived_grads for leaf-level derived gradients computed "
                "by the backend preview."
            )
        self._sync_backward_projection_if_needed()
        return BackwardPassAccessor(self.backward_pass_logs)

    @property
    def last_backward_pass(self) -> BackwardPass | None:
        """Return the most recent backward pass record, if any."""

        backward_passes = self.backward_passes
        if not backward_passes:
            return None
        return backward_passes[-1]

    def _sync_backward_projection_if_needed(self) -> None:
        """Synchronize lazy backward projections from runtime sidecar events."""

        if getattr(self, "_tl_active_backward_bracket", False):
            return
        if not getattr(self, "backward_events", ()):
            return
        from ..backends.torch.backward import (
            _close_implicit_backward_pass_if_open,
            _materialize_backward_projections,
        )

        _close_implicit_backward_pass_if_open(self)
        _materialize_backward_projections(self)

    @property
    def num_grad_fn_calls(self) -> int:
        """Total number of GradFnCall records in this Trace."""

        return len(self.grad_fn_calls)

    @property
    def saved_ops(self) -> Accessor[Op]:
        """Access Ops with saved activations."""

        return TraceOpAccessor(
            [op for op in self.layer_list if op.has_saved_activation],
            self.layer_num_calls,
        )

    @property
    def saved_grad_ops(self) -> Accessor[Op]:
        """Access Ops with saved gradients."""

        if getattr(self, "backend", "torch") in {"jax", "mlx", "tinygrad"}:
            raise ValueError(
                f"{getattr(self, 'backend', 'backend')} traces do not expose op-level saved "
                "gradients. Use trace.derived_grads for leaf-level derived gradients computed "
                "by the backend preview."
            )
        return TraceOpAccessor(
            [op for op in self.layer_list if op.has_grad],
            self.layer_num_calls,
        )

    @property
    def saved_layers(self) -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved activation."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if any(op.has_saved_activation for op in layer.ops.values())
            ),
            source_trace=self,
        )

    @property
    def saved_grad_layers(self) -> Accessor[Layer]:
        """Access Layers containing at least one Op with a saved gradient."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if any(op.has_grad for op in layer.ops.values())
            ),
            source_trace=self,
        )

    @property
    def saved_module_calls(self) -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved activations."""

        saved_labels = set(self.saved_ops.keys())
        calls: OrderedDict[str, Any] = OrderedDict()
        for call in self.module_calls:
            if any(label in saved_labels for label in getattr(call, "output_ops", [])):
                calls[call.call_label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def saved_modules(self) -> Accessor[Any]:
        """Access Modules with at least one saved-activation ModuleCall."""

        saved_addresses = {call.address for call in self.saved_module_calls}
        return ModuleAccessor(
            OrderedDict(
                (module.address, module)
                for module in self.modules
                if module.address in saved_addresses
            )
        )

    @property
    def num_saved_modules(self) -> int:
        """Number of Modules with at least one saved-activation ModuleCall."""

        return len(self.saved_modules)

    @property
    def saved_grad_module_calls(self) -> Accessor[Any]:
        """Access ModuleCalls whose outputs include saved gradients."""

        saved_labels = set(self.saved_grad_ops.keys())
        calls: OrderedDict[str, Any] = OrderedDict()
        for call in self.module_calls:
            if any(label in saved_labels for label in getattr(call, "output_ops", [])):
                calls[call.call_label] = call
        return TraceModuleCallAccessor(calls)

    @property
    def saved_grad_modules(self) -> Accessor[Any]:
        """Access Modules with at least one saved-gradient ModuleCall."""

        saved_addresses = {call.address for call in self.saved_grad_module_calls}
        return ModuleAccessor(
            OrderedDict(
                (module.address, module)
                for module in self.modules
                if module.address in saved_addresses
            )
        )

    @property
    def num_saved_grad_module_calls(self) -> int:
        """Number of ModuleCalls whose outputs include saved gradients."""

        return len(self.saved_grad_module_calls)

    @property
    def num_saved_grad_modules(self) -> int:
        """Number of Modules with at least one saved-gradient ModuleCall."""

        return len(self.saved_grad_modules)

    @property
    def saved_grad_fn_calls(self) -> Accessor[Any]:
        """Access GradFnCall records with saved gradient inputs or outputs."""

        calls: OrderedDict[str, Any] = OrderedDict()
        for label, call in self.grad_fn_calls.items():
            if (
                getattr(call, "grad_inputs", None) is not None
                or getattr(call, "grad_outputs", None) is not None
            ):
                calls[label] = call
        return TraceGradFnCallAccessor(calls)

    @property
    def saved_grad_fns(self) -> GradFnAccessor:
        """Access GradFns containing at least one saved GradFnCall."""

        items = OrderedDict(
            (grad_fn_object_id, grad_fn_handle)
            for grad_fn_object_id, grad_fn_handle in self.grad_fn_logs.items()
            if any(
                getattr(call, "grad_inputs", None) is not None
                or getattr(call, "grad_outputs", None) is not None
                for call in grad_fn_handle.calls.values()
            )
        )
        return GradFnAccessor(items, list(items))

    @property
    def compute_ops(self) -> Accessor[Op]:
        """Access Ops that are not graph-boundary sentinels."""

        return TraceOpAccessor(
            [op for op in self.layer_list if not (op.is_input or op.is_output or op.is_buffer)],
            self.layer_num_calls,
        )

    @property
    def compute_layers(self) -> Accessor[Layer]:
        """Access Layers whose representative Op is not a boundary sentinel."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (label, layer)
                for label, layer in self.layer_logs.items()
                if label in {op.layer_label for op in self.compute_ops}
            ),
            source_trace=self,
        )

    @property
    def input_ops(self) -> Accessor[Op]:
        """Access flat input-boundary Ops."""

        return TraceOpAccessor([self[label] for label in self.input_layers], self.layer_num_calls)

    @property
    def num_input_layers(self) -> int:
        """Number of input-boundary Layers."""

        return len(self.input_layers)

    @property
    def num_input_ops(self) -> int:
        """Number of flat input-boundary Ops."""

        return len(self.input_ops)

    @property
    def output_ops(self) -> Accessor[Op]:
        """Access flat output-boundary Ops."""

        return TraceOpAccessor([self[label] for label in self.output_layers], self.layer_num_calls)

    @property
    def num_output_layers(self) -> int:
        """Number of output-boundary Layers."""

        return len(self.output_layers)

    @property
    def num_output_ops(self) -> int:
        """Number of flat output-boundary Ops."""

        return len(self.output_ops)

    @property
    def num_buffer_layers(self) -> int:
        """Number of buffer-boundary Layers."""

        return len(self.buffer_layers)

    @property
    def buffer_read_ops(self) -> list[str]:
        """Labels for buffer Ops that read registered-buffer values into the graph."""

        return [op.label for op in self.layer_list if op.is_buffer and op.buffer_write_kind is None]

    @property
    def buffer_write_ops(self) -> list[str]:
        """Labels for buffer Ops that record registered-buffer write events."""

        return [
            op.label for op in self.layer_list if op.is_buffer and op.buffer_write_kind is not None
        ]

    @property
    def num_buffer_read_ops(self) -> int:
        """Number of buffer Ops that read registered-buffer values into the graph."""

        return len(self.buffer_read_ops)

    @property
    def num_buffer_write_ops(self) -> int:
        """Number of buffer Ops that record registered-buffer write events."""

        return len(self.buffer_write_ops)

    @property
    def internal_source_layers(self) -> Accessor[Layer]:
        """Access Layers representing internal-source positions."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (self[label].layer_label, self.layers[self[label].layer_label])
                for label in self.internal_source_ops
            ),
            source_trace=self,
        )

    @property
    def num_internal_source_layers(self) -> int:
        """Number of internal-source Layers."""

        return len(self.internal_source_layers)

    @property
    def num_internal_source_ops(self) -> int:
        """Number of internal-source Ops."""

        return len(self.internal_source_ops)

    @property
    def internal_sink_layers(self) -> Accessor[Layer]:
        """Access Layers representing internal-sink positions."""

        from .layer import LayerAccessor

        return LayerAccessor(
            OrderedDict(
                (self[label].layer_label, self.layers[self[label].layer_label])
                for label in self.internal_sink_ops
            ),
            source_trace=self,
        )

    @property
    def num_internal_sink_layers(self) -> int:
        """Number of internal-sink Layers."""

        return len(self.internal_sink_layers)

    @property
    def num_internal_sink_ops(self) -> int:
        """Number of internal-sink Ops."""

        return len(self.internal_sink_ops)

    @property
    def num_uncalled_modules(self) -> int:
        """Number of registered source-model modules not called in this Trace."""

        return len(self.uncalled_modules)

    @property
    def ops_with_params(self) -> Accessor[Op]:
        """Access Ops that use at least one parameter tensor."""

        return TraceOpAccessor(
            [op for op in self.layer_list if op.num_params > 0],
            self.layer_num_calls,
        )

    @property
    def num_ops_with_params(self) -> int:
        """Number of Ops that use at least one parameter tensor."""

        return len(self.ops_with_params)

    @property
    def num_grad_fns(self) -> int:
        """Number of unique autograd grad_fn_handle nodes discovered."""
        return len(self.grad_fn_logs)

    @property
    def num_grad_fns_with_op(self) -> int:
        """Number of GradFn records paired with a forward Op."""

        return sum(1 for grad_fn_handle in self.grad_fn_logs.values() if grad_fn_handle.has_op)

    @property
    def num_grad_fns_without_op(self) -> int:
        """Number of grad_fn_handle nodes without a corresponding forward Layer."""
        return sum(1 for grad_fn_handle in self.grad_fn_logs.values() if not grad_fn_handle.has_op)

    # ********************************************
    # ******** Public Convenience Methods ********
    # ********************************************

    def show(self, method: str = "graph", **kwargs: Any) -> str | None:
        """Render this trace using a lightweight notebook-friendly dispatcher.

        Parameters
        ----------
        method:
            Display method. ``"graph"`` delegates to :meth:`draw`, ``"repr"``
            returns ``repr(self)``, and ``"html"`` returns ``_repr_html_()``.
        **kwargs:
            Visualization keyword arguments forwarded to :meth:`draw`. The
            legacy ``vis_opt`` spelling is accepted as an alias for ``vis_mode``.

        Returns
        -------
        str | None
            Rendered representation, Graphviz DOT source, or ``None`` when
            rendering is explicitly disabled with ``vis_opt="none"`` or
            ``vis_mode="none"``.
        """

        vis_opt = kwargs.pop("vis_opt", None)
        if vis_opt is not None and "vis_mode" not in kwargs:
            kwargs["vis_mode"] = vis_opt
        if kwargs.get("vis_mode") == "none":
            return None
        if method == "repr":
            return repr(self)
        if method == "html":
            return self._repr_html_()
        return cast("str | None", self.draw(**kwargs))

    def draw(
        self,
        vis_opt: VisModeLiteral | MissingType = MISSING,
        view: VisModeLiteral | MissingType = MISSING,
        depth: int | MissingType = MISSING,
        renderer: VisRendererLiteral | MissingType = MISSING,
        layout: VisNodePlacementLiteral | MissingType = MISSING,
        node_style: VisNodeModeLiteral | MissingType = MISSING,
        vis_mode: VisModeLiteral = "unrolled",
        vis_call_depth: int = 1000,
        vis_outpath: str = "modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        module: "Module | str | None" = None,
        node_mode: VisNodeModeLiteral = "default",
        vis_node_mode: VisNodeModeLiteral | MissingType = MISSING,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        collapsed_node_spec_fn: Optional[Callable[..., Any]] = None,
        collapse_fn: Optional[Callable[..., Any]] = None,
        skip_fn: Optional[Callable[..., Any]] = None,
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_grad_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_module_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffers: BufferVisibilityLiteral | bool | MissingType = MISSING,
        show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
        vis_direction: VisDirectionLiteral | MissingType = MISSING,
        direction: VisDirectionLiteral = "bottomup",
        vis_node_placement: VisNodePlacementLiteral = "auto",
        vis_renderer: VisRendererLiteral = "graphviz",
        vis_theme: str = "torchlens",
        vis_intervention_mode: VisInterventionModeLiteral = "node_mark",
        vis_show_cone: bool = True,
        code_panel: "CodePanelOption" = False,
        node_overlay: str | Mapping[str, Any] | None = None,
        node_label_fields: list[str] | None = None,
        show_legend: bool = False,
        font_size: int | None = None,
        dpi: int | None = None,
        for_paper: bool = False,
        return_graph: bool = False,
        order_siblings: bool = True,
        show_containers: Literal[False, "labels", "cluster", "collapsed", "auto"] = False,
        container_max_inline: int = 12,
        show_input_transform_summary: bool = False,
    ) -> Any:
        """Render the computational graph for this model log.

        Parameters
        ----------
        vis_mode, vis_call_depth, vis_outpath, vis_graph_overrides, module, node_mode, \
        node_spec_fn, collapsed_node_spec_fn, collapse_fn, skip_fn, vis_edge_overrides, \
        vis_grad_edge_overrides, vis_module_overrides, vis_save_only, vis_fileformat, \
        show_buffer_layers, direction, vis_node_placement, vis_renderer, vis_theme, \
        vis_intervention_mode, vis_show_cone, code_panel, order_siblings, show_containers,
        container_max_inline, show_input_transform_summary:
            Forwarded unchanged to :func:`torchlens.visualization.rendering.draw`.
            ``show_buffer_layers`` accepts ``"never"``, ``"meaningful"``, or
            ``"always"``. Legacy bools are deprecated but supported by the
            Graphviz renderer.

        Returns
        -------
        Any
            Graphviz DOT source, renderer-specific output, or renderer object
            when ``return_graph=True``.
        """
        from ..visualization.rendering import draw as _impl

        if vis_opt is not MISSING:
            vis_mode = cast(VisModeLiteral, vis_opt)
        if view is not MISSING:
            vis_mode = cast(VisModeLiteral, view)
        if depth is not MISSING:
            vis_call_depth = cast(int, depth)
        if renderer is not MISSING:
            vis_renderer = cast(VisRendererLiteral, renderer)
        if layout is not MISSING:
            vis_node_placement = cast(VisNodePlacementLiteral, layout)
        if node_style is not MISSING:
            node_mode = cast(VisNodeModeLiteral, node_style)
        if vis_node_mode is not MISSING:
            node_mode = cast(VisNodeModeLiteral, vis_node_mode)
        if vis_buffers is not MISSING:
            show_buffer_layers = cast(BufferVisibilityLiteral | bool, vis_buffers)
        if vis_direction is not MISSING:
            direction = cast(VisDirectionLiteral, vis_direction)

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            module=module,
            node_mode=node_mode,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            collapse_fn=collapse_fn,
            skip_fn=skip_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_grad_edge_overrides=vis_grad_edge_overrides,
            vis_module_overrides=vis_module_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            vis_node_placement=vis_node_placement,
            vis_renderer=vis_renderer,
            vis_theme=vis_theme,
            vis_intervention_mode=vis_intervention_mode,
            vis_show_cone=vis_show_cone,
            code_panel=code_panel,
            node_overlay=node_overlay,
            node_label_fields=node_label_fields,
            show_legend=show_legend,
            font_size=font_size,
            dpi=dpi,
            for_paper=for_paper,
            return_graph=return_graph,
            order_siblings=order_siblings,
            show_containers=show_containers,
            container_max_inline=container_max_inline,
            show_input_transform_summary=show_input_transform_summary,
        )

    def add_node_overlay(
        self,
        scores: Mapping[str, Any],
        *,
        name: str = "overlay",
    ) -> "Trace":
        """Register external per-node overlay scores for later rendering.

        Parameters
        ----------
        scores:
            Mapping from layer labels to scalar or displayable values.
        name:
            Overlay name stored on the log for discoverability.

        Returns
        -------
        Trace
            This log, allowing chained calls before ``draw``.
        """

        self._node_overlay_scores = dict(scores)
        self._node_overlay_name = name
        return self

    def animate_ops(self, site: Any) -> str:
        """Return a minimal HTML animation for repeated ops at ``site``.

        Parameters
        ----------
        site:
            Layer label, pass-qualified layer label, or object with a
            ``layer_label`` attribute.

        Returns
        -------
        str
            Self-contained HTML fragment with play/pause controls.

        Raises
        ------
        KeyError
            If the requested site cannot be resolved.
        """

        label = str(getattr(site, "layer_label", site))
        base_label = label.split(":", 1)[0]
        if base_label not in self.layer_logs:
            raise KeyError(f"Unknown layer site {label!r}.")
        layer = self.layer_logs[base_label]
        pass_entries = list(getattr(layer, "ops", ()) or [])
        if not pass_entries:
            pass_entries = [self[label]]
        frames = [
            {
                "pass": int(getattr(entry, "pass_index", index + 1) or index + 1),
                "label": str(getattr(entry, "layer_label", base_label)),
                "shape": "x".join(str(dim) for dim in getattr(entry, "shape", ()) or ()),
                "memory": str(getattr(entry, "activation_memory", "")),
            }
            for index, entry in enumerate(pass_entries)
        ]
        frame_markup = "".join(
            "<li data-frame='{idx}'>{label} pass {call_index}: {shape} {memory}</li>".format(
                idx=index,
                label=escape(str(frame["label"])),
                call_index=frame["pass"],
                shape=escape(str(frame["shape"] or "scalar")),
                memory=escape(str(frame["memory"])),
            )
            for index, frame in enumerate(frames)
        )
        return (
            "<div class='tl-pass-animation' data-site='"
            + escape(base_label)
            + "'><button type='button' data-action='play'>Play</button>"
            + "<button type='button' data-action='pause'>Pause</button><ol>"
            + frame_markup
            + "</ol><script>(function(){var root=document.currentScript.parentElement;"
            + "var items=root.querySelectorAll('li');var i=0,t=null;"
            + "function show(){items.forEach(function(x,j){x.style.display=j===i?'':'none';});}"
            + "show();root.querySelector('[data-action=play]').onclick=function(){"
            + "if(t)return;t=setInterval(function(){i=(i+1)%items.length;show();},500);};"
            + "root.querySelector('[data-action=pause]').onclick=function(){clearInterval(t);t=null;};"
            + "})();</script></div>"
        )

    def first_nonfinite(self, link_format: Literal["terminal", "html", "text"] = "terminal") -> str:
        """Return a text answer describing the first saved non-finite out.

        Parameters
        ----------
        link_format:
            Source-location link style. ``"terminal"`` emits OSC 8 hyperlinks,
            ``"html"`` emits VS Code URI anchors, and ``"text"`` emits plain
            ``path:line`` text.

        Returns
        -------
        str
            Human-readable single-paragraph answer naming the layer, operation,
            module, shape, dtype, parents, and source location.
        """

        for layer in getattr(self, "layer_list", []) or []:
            out = getattr(layer, "out", None)
            if not isinstance(out, torch.Tensor) or out.numel() == 0:
                continue
            try:
                has_nonfinite = bool((~torch.isfinite(out.detach())).any().item())
            except (RuntimeError, TypeError):
                continue
            if not has_nonfinite:
                continue
            stack = getattr(layer, "code_context", None) or []
            location = "source unavailable"
            if stack:
                frame = stack[0]
                file_path = str(getattr(frame, "file", "unknown"))
                line_number = getattr(frame, "line_number", "unknown")
                if link_format == "terminal":
                    location = terminal_file_line_link(file_path, line_number)
                elif link_format == "html":
                    location = vscode_file_line_link(file_path, line_number)
                elif link_format == "text":
                    location = file_line_text(file_path, line_number)
                else:
                    raise ValueError("link_format must be 'terminal', 'html', or 'text'.")
            parents = ", ".join(getattr(layer, "parents", None) or []) or "none"
            module = getattr(layer, "module", None) or "no module"
            return (
                f"First non-finite saved out is in layer {layer.layer_label} "
                f"(op {getattr(layer, 'func_name', 'unknown')}, module {module}), "
                f"shape={getattr(layer, 'shape', None)}, "
                f"dtype={getattr(layer, 'dtype', None)}, parents={parents}, "
                f"source={location}."
            )
        return "No non-finite tensor values found in saved outs."

    def draw_backward(
        self,
        vis_outpath: str = "backward_modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        collapsed_node_spec_fn: Optional[Callable[..., Any]] = None,
        vis_node_mode: VisNodeModeLiteral = "default",
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_direction: VisDirectionLiteral = "topdown",
        code_panel: "CodePanelOption" = False,
        vis_mode: VisModeLiteral = "rolled",
        bwd: int | Iterable[int] | None = None,
    ) -> str:
        """Render the captured backward grad_fn_handle graph.

        Parameters
        ----------
        vis_outpath, vis_graph_overrides, node_spec_fn, collapsed_node_spec_fn, \
        vis_node_mode, vis_edge_overrides, vis_save_only, vis_fileformat, \
        vis_direction, code_panel, vis_mode, bwd:
            Forwarded unchanged to
            :func:`torchlens.visualization.rendering.render_backward_graph`.
            ``collapsed_node_spec_fn`` and ``vis_node_mode`` are accepted for
            forward-visualization API symmetry but are not applied because
            backward graphs do not render collapsed module nodes.

        Returns
        -------
        str
            Graphviz DOT source.
        """
        from ..visualization.rendering import render_backward_graph as _impl

        return _impl(
            self,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            collapsed_node_spec_fn=collapsed_node_spec_fn,
            vis_node_mode=vis_node_mode,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            direction=vis_direction,
            code_panel=code_panel,
            vis_mode=vis_mode,
            bwd=bwd,
        )

    def draw_combined(
        self,
        vis_outpath: str = "combined_modelgraph",
        vis_graph_overrides: Optional[Dict[str, Any]] = None,
        node_spec_fn: Optional[Callable[..., Any]] = None,
        backward_node_spec_fn: Optional[Callable[..., Any]] = None,
        vis_edge_overrides: Optional[Dict[str, Any]] = None,
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_direction: VisDirectionLiteral = "leftright",
        vis_mode: VisModeLiteral = "unrolled",
        intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
        show_buffer_layers: BufferVisibilityLiteral | bool = "meaningful",
        bwd: int | Iterable[int] | None = None,
    ) -> str:
        """Render forward ops and backward grad_fns in one graph.

        Parameters
        ----------
        vis_outpath, vis_graph_overrides, node_spec_fn, backward_node_spec_fn, \
        vis_edge_overrides, vis_save_only, vis_fileformat, vis_direction, \
        vis_mode, intervening_cluster, show_buffer_layers, bwd:
            Forwarded unchanged to
            :func:`torchlens.visualization.rendering.render_combined_graph`.

        Returns
        -------
        str
            Graphviz DOT source.
        """
        from ..visualization.rendering import render_combined_graph as _impl

        return _impl(
            self,
            vis_outpath=vis_outpath,
            vis_graph_overrides=vis_graph_overrides,
            node_spec_fn=node_spec_fn,
            backward_node_spec_fn=backward_node_spec_fn,
            vis_edge_overrides=vis_edge_overrides,
            vis_save_only=vis_save_only,
            vis_fileformat=vis_fileformat,
            direction=vis_direction,
            vis_mode=vis_mode,
            intervening_cluster=intervening_cluster,
            show_buffer_layers=show_buffer_layers,
            bwd=bwd,
        )

    def preview_fastlog(
        self,
        predicate: Optional[Callable[..., Any]] = None,
        keep_op: Optional[Callable[..., Any]] = None,
        keep_module: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Render a fastlog predicate preview for this model graph.

        Parameters
        ----------
        predicate, keep_op, keep_module:
            Predicate callables that receive synthesized fastlog ``RecordContext``
            objects.
        **kwargs:
            Forwarded to :func:`torchlens.visualization.fastlog_preview.preview_fastlog`.

        Returns
        -------
        str
            Graphviz DOT source.
        """

        from ..visualization.fastlog_preview import preview_fastlog as _impl

        return _impl(
            self,
            predicate=predicate,
            keep_op=keep_op,
            keep_module=keep_module,
            **kwargs,
        )

    def last_run_records(self) -> tuple["FireRecord", ...]:
        """Return fire records from the most recent replay, rerun, or live capture.

        Returns
        -------
        tuple[FireRecord, ...]
            Immutable snapshot of matching intervention fire records.
        """

        ctx = getattr(self, "last_run", None)
        if not isinstance(ctx, dict):
            return ()
        timestamp = ctx.get("timestamp")
        if not isinstance(timestamp, (int, float)):
            return ()
        records = []
        for layer in getattr(self, "layer_list", []) or []:
            for record in getattr(layer, "interventions", []) or []:
                record_timestamp = getattr(record, "timestamp", None)
                if isinstance(record_timestamp, (int, float)) and record_timestamp >= timestamp:
                    records.append(record)
        return tuple(records)

    def summary(
        self,
        level: Literal[
            "overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall", "output"
        ] = "overview",
        *,
        fields: Optional[List[str]] = None,
        mode: Literal["auto", "rolled", "unrolled"] = "auto",
        show_ops: bool = False,
        preset: Optional[
            Literal[
                "overview",
                "graph",
                "memory",
                "control_flow",
                "compute",
                "cost",
                "waterfall",
                "output",
            ]
        ] = None,
        columns: Optional[List[str]] = None,
        include_ops: Optional[bool] = None,
        max_rows: Optional[int] = 200,
        print_to: Optional[Callable[[str], None]] = None,
        count_fma_as_two: bool = False,
        show_input_preprocessing_details: bool = False,
    ) -> str:
        """Render a concise text summary of the logged model.

        Parameters
        ----------
        level:
            Summary level to render.
        fields:
            Explicit column selection for the primary table.
        mode:
            Operation aggregation mode. ``"rolled"`` uses aggregate layer rows,
            while ``"unrolled"`` uses per-pass operation rows.
        show_ops:
            Whether to append an operation table.
        preset:
            Alias for ``level`` retained for compatibility with the design spec.
        columns:
            Alias for ``fields``.
        include_ops:
            Alias for ``show_ops`` retained for compatibility with the design spec.
        max_rows:
            Maximum number of rows to render per table. ``None`` disables truncation.
        print_to:
            Optional callable that receives the rendered summary text.
        count_fma_as_two:
            FLOP/MAC convention marker for summary consumers. Current captured
            counts are displayed as stored; this flag reserves the public
            convention toggle without changing saved metadata.
        show_input_preprocessing_details:
            Whether to include verification/source detail for input
            preprocessing records.

        Returns
        -------
        str
            Rendered summary string.
        """
        from ..visualization._summary_internal import render_model_summary

        del count_fma_as_two
        return render_model_summary(
            self,
            level=level,
            preset=preset,
            fields=fields,
            columns=columns,
            mode=mode,
            show_ops=show_ops,
            include_ops=include_ops,
            max_rows=max_rows,
            print_to=print_to,
            show_input_preprocessing_details=show_input_preprocessing_details,
        )

    def render_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        vis_outpath: str = "graph.gv",
        vis_save_only: bool = False,
        vis_fileformat: str = "pdf",
        vis_buffers: bool = False,
        vis_direction: str = "bottomup",
        vis_theme: str = "torchlens",
    ) -> str:
        """Render this model log with the experimental Dagua backend.

        Parameters
        ----------
        vis_mode, vis_call_depth, vis_outpath, vis_save_only, vis_fileformat, \
        vis_buffers, vis_direction, vis_theme:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.render_trace_with_dagua`.

        Returns
        -------
        str
            Serialized Dagua graph output or the rendered artifact path.
        """
        from ..experimental.dagua import render_trace_with_dagua as _impl

        return cast(
            str,
            _impl(
                self,
                vis_mode=vis_mode,
                vis_call_depth=vis_call_depth,
                vis_outpath=vis_outpath,
                vis_save_only=vis_save_only,
                vis_fileformat=vis_fileformat,
                vis_buffers=vis_buffers,
                vis_direction=vis_direction,
                vis_theme=vis_theme,
            ),
        )

    def to_dagua_graph(
        self,
        vis_mode: str = "unrolled",
        vis_call_depth: int = 1000,
        show_buffer_layers: bool = False,
        direction: str = "bottomup",
        include_grad_edges: Optional[bool] = None,
    ) -> Any:
        """Translate this model log into an experimental Dagua graph.

        Parameters
        ----------
        vis_mode, vis_call_depth, show_buffer_layers, direction, include_grad_edges:
            Forwarded unchanged to
            :func:`torchlens.experimental.dagua.trace_to_dagua_graph`.

        Returns
        -------
        Any
            Dagua graph object.
        """
        from ..experimental.dagua import trace_to_dagua_graph as _impl

        return _impl(
            self,
            vis_mode=vis_mode,
            vis_call_depth=vis_call_depth,
            show_buffer_layers=show_buffer_layers,
            direction=direction,
            include_grad_edges=include_grad_edges,
        )

    def visualization_field_audit(self) -> "TorchLensRenderAudit":
        """Return the visualization field-usage audit for this model log.

        Returns
        -------
        TorchLensRenderAudit
            Audit of used and unused fields in the visualization bridge.
        """
        from ..experimental.dagua import build_render_audit as _impl

        return _impl(self)

    def to_pandas(self, include_decoded_output_summary: bool = False) -> "pd.DataFrame":
        """Return a dataframe containing one row per layer pass.

        Parameters
        ----------
        include_decoded_output_summary:
            If ``True``, add a gated ``decoded_output_summary`` column populated
            only on output-node rows. The default remains schema-stable.

        Returns
        -------
        pd.DataFrame
            Layer-pass table for this model log.

        Raises
        ------
        RuntimeError
            If called before the forward pass has completed.
        """
        if not self._tracing_finished:
            raise RuntimeError(
                "to_pandas() cannot be called before the forward pass is complete. "
                "Please wait until trace has returned."
            )
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with "
                "`pip install torchlens[tabular]`."
            ) from e

        # Identity columns lead the table; the remaining columns follow the
        # canonical Op field order minus the documented exclusions, plus the
        # handful of property-backed convenience columns that are not stored
        # fields. ``dict.fromkeys`` removes duplicates while keeping the lead
        # block's positions.
        lead_columns = [
            "layer_label",
            "label",
            "layer_label_short",
            "label_short",
            "layer_type",
        ]
        property_columns = [
            "has_parents",
            "siblings",
            "has_siblings",
            "co_parents",
            "has_co_parents",
            "uses_params",
            "is_module_input",
            "output_role",
            "num_saved_grads",
        ]
        if include_decoded_output_summary:
            property_columns.append("decoded_output_summary")
        fields_for_df = list(
            dict.fromkeys(
                lead_columns
                + [
                    field_name
                    for field_name in LAYER_PASS_LOG_FIELD_ORDER
                    if field_name not in _TO_PANDAS_EXCLUDED_OP_FIELDS
                ]
                + property_columns
            )
        )

        fields_to_change_type = {
            "type_index": int,
            "step_index": int,
            "num_passes": int,
            "pass_index": int,
            "is_inplace": bool,
            "is_input": bool,
            "is_output": bool,
            "is_buffer": bool,
            "in_multi_output": bool,
            "has_parents": bool,
            "has_children": bool,
            "has_siblings": bool,
            "has_co_parents": bool,
            "uses_params": bool,
            "num_params": int,
            "param_memory": int,
            "activation_memory": int,
            "is_module_input": bool,
            "is_module_output": bool,
            "conditional_branch_depth": int,
            "is_terminal_conditional_bool": bool,
        }

        model_df_dictlist: list[dict[str, Any]] = []
        for layer_entry in self.layer_list:
            layer_dict: dict[str, Any] = {}
            for field_name in fields_for_df:
                if field_name == "conditional_branch_stack":
                    layer_dict[field_name] = _format_conditional_branch_stack(
                        layer_entry.conditional_branch_stack
                    )
                elif field_name == "grad":
                    grad_records = getattr(layer_entry, "_grad_records", ())
                    layer_dict[field_name] = (
                        next(iter(grad_records)).grad if len(grad_records) == 1 else None
                    )
                elif field_name == "num_saved_grads":
                    layer_dict[field_name] = len(getattr(layer_entry, "_grad_records", ()))
                elif field_name == "output_role":
                    layer_dict[field_name] = _output_role_from_container_path(
                        getattr(layer_entry, "container_path", ())
                    )
                elif field_name == "decoded_output_summary":
                    layer_dict[field_name] = (
                        self._decoded_output_summary()
                        if bool(getattr(layer_entry, "is_output", False))
                        else None
                    )
                else:
                    layer_dict[field_name] = getattr(layer_entry, field_name)
            model_df_dictlist.append(layer_dict)
        model_df = pd.DataFrame(model_df_dictlist)

        for column_name in fields_to_change_type:
            model_df[column_name] = model_df[column_name].astype(fields_to_change_type[column_name])
        model_df["terminal_conditional_id"] = model_df["terminal_conditional_id"].astype("Int64")

        return model_df

    def save_new_outs(
        self,
        model: torch.nn.Module,
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: str | List[str] = "all",
        grad_layers_to_save: str | List[str] | None = "all",
        random_seed: Optional[int] = None,
        backward_ready: bool | None = None,
    ) -> None:
        """Re-run the model with new inputs, saving only outs.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed, backward_ready:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.save_new_outs`.
        """
        from ..capture.trace import save_new_outs as _impl

        if backward_ready is True:
            reject_compiled_model(model, api_name="Trace.save_new_outs")

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            backward_ready=backward_ready,
        )

    def validate_saved_outs(
        self,
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationReplayStatus"]:
        """Deprecated alias for :meth:`validate_forward_pass`.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to :meth:`validate_forward_pass`.

        Returns
        -------
        bool or ValidationReplayStatus
            ``True`` if validation succeeds. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status.
        """
        warn_deprecated_alias(
            "Trace.validate_saved_outs",
            "Trace.validate_forward_pass",
        )
        return self.validate_forward_pass(
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )

    def validate_forward_pass(
        self,
        ground_truth_output_tensors: List[torch.Tensor],
        verbose: bool = False,
        validate_metadata: bool = True,
    ) -> Union[bool, "ValidationReplayStatus"]:
        """Validate saved outs against ground-truth model outputs.

        Parameters
        ----------
        ground_truth_output_tensors, verbose, validate_metadata:
            Forwarded unchanged to
            :func:`torchlens.validation.core.validate_saved_outs`.

        Returns
        -------
        bool or ValidationReplayStatus
            ``True`` if validation succeeds. Loaded non-torch traces whose
            runtime replay captures were stripped return an explicit
            unavailable status instead of a pass/fail bool.
        """
        from ..backends import get_backend_spec

        status = self.validation_replay_status
        if bool(getattr(self, "_loaded_from_bundle", False)) and not status.available:
            setattr(self, "_validation_replay_status", status)
            return status
        spec = get_backend_spec(getattr(self, "backend", "torch"))
        return spec.validate_trace(
            self,
            ground_truth_output_tensors=ground_truth_output_tensors,
            verbose=verbose,
            validate_metadata=validate_metadata,
        )

    @property
    def validation_replay_status(self) -> "ValidationReplayStatus":
        """Return replay-validation availability or last completed result.

        Returns
        -------
        ValidationReplayStatus
            Status object distinguishing live replay validation from loaded
            traces whose runtime replay captures were stripped during save.
        """

        from ..backends import get_backend_spec
        from ..validation.status import ValidationReplayStatus

        cached_status = getattr(self, "_validation_replay_status", None)
        if isinstance(cached_status, ValidationReplayStatus):
            return cached_status
        backend = str(getattr(self, "backend", "torch"))
        if _loaded_non_torch_validation_replay_unavailable(self):
            return ValidationReplayStatus.unavailable_loaded_runtime_stripped(
                backend=backend,
                payload_load_status=getattr(self, "payload_load_status", None),
            )
        spec = get_backend_spec(backend)
        if not spec.capabilities.validation_replay:
            return ValidationReplayStatus.unavailable_unsupported(backend=backend)
        return ValidationReplayStatus.available_live(backend=backend)

    def replay(
        self,
        strict: bool | MissingType = MISSING,
        hooks: dict[Any, Any] | None | MissingType = MISSING,
        differentiable: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Replay the saved DAG cone affected by hooks.

        Parameters
        ----------
        strict:
            Whether replay divergence warnings should raise.
        hooks:
            Optional mapping from selector-like targets to hook callables.
        differentiable:
            If true, return a new Trace whose replayed tensors remain
            differentiable from fresh replay-frontier leaves.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        replay_options = merge_replay_options(
            replay=replay,
            strict=strict,
            hooks=hooks,
            differentiable=differentiable,
        )

        from ..intervention.replay import replay as _impl

        return _impl(self, replay=replay_options)

    def replay_from(
        self,
        site: Any,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
    ) -> "Trace":
        """Replay downstream from a pre-mutated site.

        Parameters
        ----------
        site:
            Layer pass or selector resolving to one origin. The origin's
            current out is preserved and used as the override.
        strict:
            Whether replay divergence warnings should raise.

        Returns
        -------
        Trace
            This model log, mutated in place.
        """

        replay_options = merge_replay_options(replay=replay, strict=strict)

        from ..intervention.replay import replay_from as _impl

        return _impl(self, site, replay=replay_options)

    def decode_output(self, top_n: int | None = None) -> Any:
        """Return captured decoded output rows when available.

        Parameters
        ----------
        top_n:
            Optional maximum rank to return per batch item.

        Returns
        -------
        Any
            JSON-primitive decoded output captured during tracing.

        Raises
        ------
        ValueError
            If decoded output was not captured or the requested ``top_n`` exceeds
            the capture-time bound.
        """

        if self.decoded_output is None:
            recomputed = self._recompute_decoded_output(top_n=top_n)
            if recomputed is None:
                raise ValueError(
                    "logits not retained; re-decode unavailable. Capture with semantic "
                    "output decoding enabled and retained logits to recompute."
                )
            return recomputed
        if top_n is None:
            return self.decoded_output
        captured_top_n = getattr(self.output_postprocessor, "top_n_captured", None)
        if captured_top_n is not None and top_n > captured_top_n:
            recomputed = self._recompute_decoded_output(top_n=top_n)
            if recomputed is not None:
                return recomputed
            raise ValueError(
                f"logits not retained; re-decode unavailable above captured top_n={captured_top_n}."
            )
        if _is_batch_topk_decoded(self.decoded_output):
            return {
                "kind": "batch_topk",
                "rows": [
                    row
                    for row in self.decoded_output["rows"]
                    if isinstance(row, dict) and int(row.get("rank", 1)) <= top_n
                ],
            }
        if not isinstance(self.decoded_output, list):
            return self.decoded_output
        return [
            row
            for row in self.decoded_output
            if not isinstance(row, dict) or int(row.get("rank", 1)) <= top_n
        ]

    def output_table(
        self, top_n: int = 5, batch_items: int | Sequence[int] | None = None
    ) -> "pd.DataFrame":
        """Return decoded output as a batch top-k dataframe.

        Parameters
        ----------
        top_n:
            Maximum number of class rows per batch item.
        batch_items:
            ``None`` returns all captured batch items, an integer returns the
            first ``N`` batch items, and a sequence selects explicit batch item
            indices.

        Returns
        -------
        pd.DataFrame
            Table with columns ``batch_item``, ``rank``, ``label``, and ``prob``.

        Raises
        ------
        ValueError
            If classification logits were not decoded or cannot be recomputed.
        """

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
            ) from e

        if top_n < 1:
            raise ValueError("top_n must be >= 1.")
        decoded = self.decode_output(top_n=top_n)
        rows = _decoded_batch_topk_rows(decoded)
        if rows is None:
            raise ValueError("decoded output is not a batch top-k classification table.")
        selected_items = _normalize_batch_items(batch_items, rows)
        filtered_rows = [
            {
                "batch_item": int(row["batch_item"]),
                "rank": int(row["rank"]),
                "label": str(row["label"]),
                "prob": float(row["prob"]),
            }
            for row in rows
            if int(row.get("rank", 0)) <= top_n
            and (selected_items is None or int(row.get("batch_item", -1)) in selected_items)
        ]
        return pd.DataFrame(filtered_rows, columns=["batch_item", "rank", "label", "prob"])

    def _recompute_decoded_output(self, top_n: int | None) -> Any | None:
        """Best-effort decoded-output recompute from retained output logits.

        Parameters
        ----------
        top_n:
            Requested number of top rows per batch item.

        Returns
        -------
        Any | None
            Typed decoded representation when recompute succeeds, otherwise
            ``None``.
        """

        postprocessor = getattr(self, "output_postprocessor", None)
        if postprocessor is None or getattr(postprocessor, "style", None) == "hf_text":
            return None
        logits = _retained_output_logits(self)
        if logits is None:
            return None
        from ..autoroute._builtin_output import _decode_classification, _labels_for_resolved

        labels = _labels_for_resolved(postprocessor)
        if labels is None:
            return None
        rows = _decode_classification(logits, labels, top_n=top_n or 5)
        if rows is None:
            return None
        return {"kind": "batch_topk", "rows": rows}

    def _decoded_output_summary(self) -> str | None:
        """Return a compact decoded-output summary for gated dataframe export.

        Returns
        -------
        str | None
            First decoded row summary, or ``None`` when unavailable.
        """

        rows = _decoded_batch_topk_rows(getattr(self, "decoded_output", None))
        if not rows:
            return None
        first_item = int(rows[0].get("batch_item", 0))
        item_rows = [row for row in rows if int(row.get("batch_item", -1)) == first_item][:3]
        return "; ".join(
            f"{int(row.get('rank', 0))}. {row.get('label')} ({float(row.get('prob', 0.0)):.1%})"
            for row in item_rows
        )

    def rerun(
        self,
        model: Any = None,
        x: Any = None,
        *,
        append: bool | MissingType = MISSING,
        chunk_size: int | None | MissingType = MISSING,
        chunk_paths: Any | None = None,
        strict: bool | MissingType = MISSING,
        replay: ReplayOptions | None = None,
        transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
        output_transform: Callable[[Any], Any] | bool | object = _USE_STORED_TRANSFORM,
    ) -> "Trace":
        """Re-execute a model with this log's active intervention spec.

        Parameters
        ----------
        model:
            Model to execute through TorchLens decorated wrappers. When omitted,
            the live model captured by this ``Trace`` is reused if still available.
        x:
            Forward input. If ``model`` is omitted, the first positional argument
            is treated as the new user input.
        append:
            If true, append a compatible chunk along batch dimension 0.
        chunk_size:
            If supplied, split positional tensor input into chunks of this size,
            rerun the first chunk normally, then append remaining chunks.
        chunk_paths:
            Optional explicit tensor leaf paths to split.
        strict:
            Whether graph-shape divergence should raise instead of warn.
        transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit input
            transform callable for this rerun.
        output_transform:
            Stored-transform sentinel, ``False`` to bypass, or explicit output
            transform callable for this rerun.

        Returns
        -------
        Trace
            This model log, mutated in place after a validated atomic swap.
        """

        rerun_model: nn.Module | None
        if isinstance(model, nn.Module):
            rerun_model = model
            user_input = x
        else:
            source_ref = getattr(self, "_source_model_ref", None)
            user_input = model
            if x is not None:
                raise TypeError("Pass either rerun(model, x) or rerun(new_user_input), not both.")
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)
            rerun_model = source_ref() if source_ref is not None else None
            if rerun_model is None:
                raise RuntimeError(
                    "This Trace does not retain a live model reference. Pass the model as "
                    "`trace.rerun(model, input)`."
                )
        replay_options = merge_replay_options(
            replay=replay,
            append=append,
            chunk_size=chunk_size,
            strict=strict,
        )
        if isinstance(model, nn.Module):
            transformed_input = self._apply_rerun_transform(user_input, transform=transform)

        from ..intervention.rerun import rerun as _impl

        resolved_output_transform = self._resolve_rerun_output_transform(output_transform)
        result = _impl(
            self,
            rerun_model,
            transformed_input,
            replay=replay_options,
            chunk_paths=chunk_paths,
            output_transform=resolved_output_transform,
        )
        # Atomic swap rebuilds Trace state; restore raw_input to the new
        # user-supplied value so visualization / save-load report the
        # current input rather than the prior trace's.
        result.raw_input = user_input
        return result

    def _apply_rerun_transform(
        self,
        user_input: Any,
        *,
        transform: Callable[[Any], Any] | bool | object,
    ) -> Any:
        """Apply the stored or explicit input transform for ``rerun``.

        Parameters
        ----------
        user_input:
            New user input supplied to ``rerun``.
        transform:
            Sentinel to reuse the stored transform, ``False`` to bypass, or an
            explicit callable to use for this rerun.

        Returns
        -------
        Any
            Model-ready rerun input.
        """

        stored_transform = getattr(self, "_transform", None)
        if transform is _USE_STORED_TRANSFORM and stored_transform is not None:
            return stored_transform(user_input)
        if transform is False:
            return user_input
        if callable(transform):
            return transform(user_input)
        return user_input

    def _resolve_rerun_output_transform(
        self,
        output_transform: Callable[[Any], Any] | bool | object,
    ) -> Callable[[Any], Any] | None:
        """Resolve the output transform callable for ``rerun``.

        Parameters
        ----------
        output_transform:
            Sentinel to reuse the stored output transform, ``False`` to bypass,
            or an explicit callable to use for this rerun.

        Returns
        -------
        Callable[[Any], Any] | None
            Output transform to apply to the fresh model output, or ``None``.
        """

        stored_transform = getattr(self, "_output_transform", None)
        if output_transform is _USE_STORED_TRANSFORM:
            return stored_transform
        if output_transform is False:
            return None
        if callable(output_transform):
            return output_transform
        return None

    def check_metadata_invariants(self) -> bool:
        """Run metadata invariant checks on this completed model log.

        Returns
        -------
        bool
            ``True`` if all invariants pass.
        """
        from ..validation.invariants import check_metadata_invariants as _impl

        return _impl(self)

    def cleanup(self) -> None:
        """Delete log data, break cycles, and free cached GPU memory.

        Returns
        -------
        None
            This method mutates the model log in place.
        """
        return cleanup(self)

    def release_param_refs(self, *, allow_iter_rehydrate: bool = False) -> None:
        """Release live ``nn.Parameter`` references held by ParamLogs.

        Parameters
        ----------
        allow_iter_rehydrate:
            If ``True``, iterating ``param_logs`` may lazily restore live
            references from the source model. Public explicit releases leave
            this disabled.

        Returns
        -------
        None
            This method mutates ParamLogs in place.
        """
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = False
        for param_log in self.param_logs.values():
            param_log.release_param_ref()
        if hasattr(self.param_logs, "_rehydrate_on_iter"):
            self.param_logs._rehydrate_on_iter = allow_iter_rehydrate

    def _postprocess(
        self,
        output_tensors: List[torch.Tensor],
        output_tensor_addresses: List[str],
    ) -> None:
        """Run postprocessing on a completed raw capture pass.

        Parameters
        ----------
        output_tensors:
            Output tensors returned by the model.
        output_tensor_addresses:
            Hierarchical addresses for those outputs.
        """
        from ..postprocess import postprocess as _impl

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
        input_args: torch.Tensor | List[Any],
        input_kwargs: Optional[Dict[Any, Any]] = None,
        layers_to_save: Optional[str | List[str | int]] = "all",
        grad_layers_to_save: Optional[str | List[str | int]] = "all",
        random_seed: Optional[int] = None,
        postprocess: bool = True,
    ) -> Any:
        """Run a forward pass and capture it into this model log.

        Parameters
        ----------
        model, input_args, input_kwargs, layers_to_save, grad_layers_to_save, random_seed:
            Forwarded unchanged to
            :func:`torchlens.capture.trace.run_and_log_inputs_through_model`.
        """
        from ..capture.trace import run_and_log_inputs_through_model as _impl

        return _impl(
            self,
            model=model,
            input_args=input_args,
            input_kwargs=input_kwargs,
            layers_to_save=layers_to_save,
            grad_layers_to_save=grad_layers_to_save,
            random_seed=random_seed,
            postprocess=postprocess,
        )

    def log_backward(self, loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` while capturing first-class backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This model log, for chaining.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec

        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import log_backward as _impl

        return cast("Trace", _impl(self, loss, **backward_kwargs))

    def backward(self, loss: torch.Tensor, **backward_kwargs: Any) -> "Trace":
        """Run backward from ``loss`` and populate this Trace with backward metadata.

        Parameters
        ----------
        loss:
            Tensor whose ``grad_fn_handle`` roots the backward graph.
        **backward_kwargs:
            Keyword arguments forwarded to ``torch.Tensor.backward``.

        Returns
        -------
        Trace
            This Trace, for chaining.
        """

        return self.log_backward(loss, **backward_kwargs)

    def recording_backward(self) -> Any:
        """Return a context manager that captures user-managed backward calls.

        Returns
        -------
        Any
            Backward recording context manager.
        """
        from ..backends import BackendUnsupportedError, get_backend_spec

        spec = get_backend_spec(getattr(self, "backend", "torch"))
        if not spec.capabilities.backward_capture:
            raise BackendUnsupportedError(
                f"Backend {spec.name!r} does not support backward capture. "
                "Use trace.derived_grads when this backend exposes leaf-level "
                "derived gradients."
            )
        from ..backends.torch.backward import recording_backward as _impl

        return _impl(self)

    def disarm_triggers(self) -> None:
        """Detach this Trace from global autograd backward interception.

        Returns
        -------
        None
            Future plain ``loss.backward()`` or ``torch.autograd.*`` calls will
            not record into this Trace.
        """
        from ..backends.torch.backward import disarm_triggers as _impl

        _impl(self)

    def _remove_log_entry(
        self,
        log_entry: Op,
        remove_references: bool = True,
    ) -> None:
        """Remove a single layer-pass entry and scrub graph references.

        Parameters
        ----------
        log_entry:
            Entry to remove.
        remove_references:
            Whether to scrub all graph references to the removed entry.
        """
        tensor_label = _label_for_reference_removal(log_entry, self._tracing_finished)
        if remove_references:
            _remove_log_entry_references(self, tensor_label)
        _clear_entry_attributes(log_entry)

    def _batch_remove_log_entries(
        self,
        entries_to_remove: Iterable[Op],
        remove_references: bool = True,
    ) -> None:
        """Remove multiple layer-pass entries using single-pass filtering.

        Parameters
        ----------
        entries_to_remove:
            Entries to remove.
        remove_references:
            Whether to scrub all graph references to the removed entries.
        """
        entries_to_remove = list(entries_to_remove)
        surviving_entries = [entry for entry in self if entry not in entries_to_remove]

        labels_to_remove = set()
        for entry in entries_to_remove:
            labels_to_remove.add(_label_for_reference_removal(entry, self._tracing_finished))
            _clear_entry_attributes(entry)

        if not remove_references:
            return

        _scrub_conditional_fields_after_removal(self, labels_to_remove, surviving_entries)

        for field_name in _LIST_FIELDS_TO_CLEAN:
            collection = getattr(self, field_name)
            collection[:] = [label for label in collection if label not in labels_to_remove]

        self.conditional_branch_edges = [
            edge
            for edge in self.conditional_branch_edges
            if edge[0] not in labels_to_remove and edge[1] not in labels_to_remove
        ]

        for param_group, tensor_labels in list(self.layers_with_params.items()):
            self.layers_with_params[param_group] = [
                label for label in tensor_labels if label not in labels_to_remove
            ]
        self.layers_with_params = {
            param_group: tensor_labels
            for param_group, tensor_labels in self.layers_with_params.items()
            if len(tensor_labels) > 0
        }

        for equiv_group, equivalent_label_set in list(self.op_equivalence_classes.items()):
            equivalent_label_set -= labels_to_remove
        self.op_equivalence_classes = {
            equiv_group: equivalent_label_set
            for equiv_group, equivalent_label_set in self.op_equivalence_classes.items()
            if len(equivalent_label_set) > 0
        }

        _scrub_per_op_equivalence_lists(surviving_entries, labels_to_remove)


Trace.FIELD_FORK_POLICY = MODEL_LOG_FIELD_FORK_POLICY  # type: ignore[attr-defined]
Trace.DEFAULT_FILL_STATE = _MODEL_LOG_DEFAULT_FILL  # type: ignore[attr-defined]
