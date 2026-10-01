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

import copy
import difflib
import inspect
import json
import pickle
import re
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
    cast,
)

import torch
from torch import nn

if TYPE_CHECKING:
    from .._io.streaming import BundleStreamWriter
    from ..debug._audit import TraceAudit
    from ..runnable import (
        ArchivedActivation,
        ReadinessReport,
        SparseRunDescriptor,
    )
    from .func_call_location import FuncCallLocation

from .. import _state
from .._errors import InvalidArgumentError
from .._io import (
    TLSPEC_VERSION,
    FieldPolicy,
    coerce_container_typed_state,
    default_fill_state,
    read_tlspec_version,
)
from .._runnable_seam import (
    RunnableTraceState,
    normalize_runnable_trace_state,
    runnable_trace_state,
)
from .._save_budget import SaveBudget, SaveBudgetOption
from .._trace_state import TraceState
from ..backends import BackendName
from ..captured_run import CapturedRun
from ..constants import LAYER_PASS_LOG_FIELD_ORDER, MODEL_LOG_FIELD_ORDER
from ..intervention.types import (
    MODEL_LOG_FIELD_FORK_POLICY,
    InterventionSpec,
    Relationship,
)
from ..ir.workspaces import (
    LEGACY_TRACE_BUILD_STATE_KEYS,
    ModuleCaptureWorkspace,
    RawGraphWorkspace,
    WrapperRuntimeWorkspace,
)
from ..quantities import Bytes, Duration
from ..types import ActivationPostfunc, GradientPostfunc
from ..utils.tensor_utils import SaveMode
from ._state_adapter import state_items, state_restore
from ._trace_accessors import (
    _invalidate_trace_module_call_accessor_cache,
    _invalidate_trace_op_layer_accessor_caches,
)
from ._trace_legacy_state import TRACE_PORTABLE_STATE_ALIASES, pop_retired_legacy_keys
from .backward_pass import BackwardPass
from .derived_grad import DerivedGradAccessor
from .field_policy import (
    build_record_field_policy_table,
    default_fill_state_from_policy,
    fork_policy_from_policy,
    portable_state_spec_from_policy,
)
from .grad_fn import GradFn
from .interface import (
    _getitem_after_pass,
    _getitem_during_pass,
    _str_after_pass,
    _str_during_pass,
)
from .layer import Layer
from .module import ModuleAccessor
from .op import Op
from .param import ParamAccessor

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

    class TraceStackMixin(_TraceMixinTypingBase):
        """Typing-only TraceStackMixin stand-in."""

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
    from ._trace_stack import TraceStackMixin
    from ._trace_stats import TraceStatsMixin
    from ._trace_validation import TraceValidationMixin
    from ._trace_viz import TraceVisualizationMixin


_MODEL_LOG_DEFAULT_FILL: dict[str, Any] = {
    "grouping": "structural",
    "grouping_policy": None,
    "distributed_scope": None,
    "trace_label": None,
    "model_label": None,
    "backend": "torch",
    "module_identity_mode": "torch_module",
    "param_source": "native-module",
    "derived_grads": DerivedGradAccessor(),
    "_runnable": None,
    "_fast_run_session": None,
    "_distributed_plane_p": None,
    "_buffer_persistence": {},
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
    "_warned_once": set(),
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
    "measure_python_peak_memory": False,
    "distributed_witness": "none",
    "save_budget": "auto",
    "raise_on_nan": False,
    "track_nonfinite": False,
    "structure_only": False,
    "intervention_audit": [],
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
    "escape_diagnostics": [],
    "completeness_diagnostics": [],
    "completeness_decompositions": [],
    "capture_guard_passes": [],
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
    **dict.fromkeys(MODEL_LOG_FIELD_ORDER),
    **_MODEL_LOG_CONTAINER_DEFAULTS,
    **_MODEL_LOG_DEFAULT_FILL,
}
_MODEL_LOG_DEFAULT_FILL["tlspec_version"] = TLSPEC_VERSION

# Plausible-but-absent attribute names, mapped to the fields that answer them. A
# frontier-scale user's first question is "how big is this capture?", and the
# singular ``activation_memory`` spelling (which IS an ``Op`` field, meaning that
# one op's payload bytes) has no single correct Trace-level meaning: the whole
# forward's tensors and the subset ``save=`` retained are different numbers, and
# collapsing them into one alias would make the answer ambiguous rather than
# available. So the names route to the real fields instead of becoming one.
_MISSING_ATTR_HINTS: dict[str, str] = {
    "activation_memory": (
        "use total_activation_memory for every tensor computed in the forward, or "
        "saved_activation_memory for just the payloads save= retained "
        "(Op.activation_memory is the per-op figure; forward_peak_memory is the "
        "measured runtime peak)."
    ),
    "memory": (
        "use total_activation_memory / saved_activation_memory for activations, "
        "total_param_memory for parameters, or forward_peak_memory for the measured "
        "runtime peak."
    ),
    "total_memory": (
        "use total_activation_memory for activations and total_param_memory for "
        "parameters; forward_peak_memory is the measured runtime peak."
    ),
    "footprint": (
        "use total_activation_memory / saved_activation_memory / total_param_memory, "
        "or forward_peak_memory for the measured runtime peak."
    ),
}


def _raise_missing_trace_attribute(trace: "Trace", name: str) -> Any:
    """Raise the canonical error for one missing Trace attribute.

    Parameters
    ----------
    trace:
        Trace on which attribute lookup failed.
    name:
        Missing attribute name.

    Raises
    ------
    AttributeError
        Always, with an actionable memory-field hint when available.
    """

    if getattr(trace, "__dict__", {}).get("_tl_cleaned_up", False):
        # One typed refusal for every reader of a husked trace, instead of a
        # raw AttributeError naming whichever private field the reader hit
        # first (b6-opus R25: summary/iteration/getitem/draw/receptive_fields
        # each leaked a different private name).
        from .._errors import TraceCleanedUpError

        raise TraceCleanedUpError(
            f"this Trace was husked by cleanup(), so {name!r} (like every "
            "logged field) has been deleted",
            remedy="re-capture with tl.trace(...); cleanup() permanently empties a Trace",
            attribute=name,
        )
    hint = _MISSING_ATTR_HINTS.get(name)
    if hint is not None:
        raise AttributeError(f"{type(trace).__name__!s} object has no attribute {name!r}; {hint}")
    raise AttributeError(f"{type(trace).__name__!s} object has no attribute {name!r}")


_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
"""Matches CPython object/function ``repr`` heap addresses (``at 0x7f...``)."""


def _scrubbed_transform_repr(fn: Any) -> str | None:
    """Return a persistence-safe repr of an activation-transform callable.

    The repr is stored (``_activation_transform_repr``) at every save level. A plain
    function repr is inert, but ``functools.partial`` reprs embed the BOUND ARGUMENT
    VALUES (B8-20: a probe recovered a planted token from a saved artifact). Redact a
    partial's positional and keyword arguments to ``<scrubbed>`` while keeping the
    wrapped function's own (recursively scrubbed) repr, so the string stays useful
    without leaking captured values.

    Parameters
    ----------
    fn:
        Activation-transform callable, or ``None``.

    Returns
    -------
    str | None
        Scrubbed repr, or ``None`` when ``fn`` is ``None``.
    """

    import functools

    if fn is None:
        return None
    if isinstance(fn, functools.partial):
        inner = _scrubbed_transform_repr(fn.func)
        parts = [inner] if inner is not None else []
        parts.extend("<scrubbed>" for _ in fn.args)
        parts.extend(f"{key}=<scrubbed>" for key in (fn.keywords or {}))
        return f"functools.partial({', '.join(parts)})"
    # A plain callable's ``repr`` embeds a live heap address for anything using
    # the default object/function repr (``<function f at 0x7f...>``,
    # ``<Foo object at 0x...>``) -- a non-deterministic value (breaks
    # byte-identical artifacts) and an ASLR heap-layout leak spliced into a
    # persisted KEEP field (b3-opus, completing the B8-20 scrub). Redact any
    # ``0x<hex>`` address before it is stored.
    return _ADDRESS_RE.sub("0x<scrubbed>", repr(fn))


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

    # Declared so the scrubber walks this record field-by-field instead of
    # persisting it verbatim: ``source_file`` is the ABSOLUTE path of the user's
    # forward-defining module, and without a spec the source-path relativizer never
    # ran, leaking host paths at every save level including ``include_source=False``
    # (B8-18). The scrubber applies the source-privacy policy to ``source_file`` via
    # ``_apply_conditional_source_policy``; every other field is portable structural
    # metadata. Declared ``ClassVar`` so ``@dataclass`` does not treat it as a field.
    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
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

    # Declared so the scrubber walks this record and applies the source-privacy
    # policy to ``source_file`` (the absolute forward-module path) instead of
    # persisting it verbatim (B8-18). ``ClassVar`` so ``@dataclass`` ignores it.
    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
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

    # Declared so the scrubber descends into ``_list``/``_dict`` and reaches each
    # ``Conditional`` (whose ``source_file`` must be privacy-scrubbed) rather than
    # persisting the whole accessor subtree verbatim (B8-18). ``_list`` and ``_dict``
    # share the same ``Conditional`` objects, so the scrub memo keeps them identical.
    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
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
    TraceStackMixin,
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
    def runnable_descriptor(self) -> "SparseRunDescriptor | None":
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

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        """Strictly validate and atomically stage sparse-run state.

        Parameters
        ----------
        sd:
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

        from .._runnable_state import load_trace_state_dict

        load_trace_state_dict(self, sd)

    def find_nan(self) -> Any:
        """Return the first NaN or Inf among saved outputs in execution order.

        Selectively saved traces identify this as the first finding among
        saved tensors and report unsaved ancestor operations as an uncertainty
        zone rather than claiming the true birth operation.

        Returns
        -------
        FindNanResult
            Structured non-finite diagnostic. Source locations are surfaced
            from the operation's existing recorded code context when available.
        """

        from ..debug._nan import find_nan_in_trace

        return find_nan_in_trace(self)

    def audit(self) -> "TraceAudit":
        """Return an honest one-call health report for this completed trace.

        Payload-dependent and gradient-dependent diagnostics are listed as
        skipped when this capture cannot support them, rather than being used
        to claim a clean result.

        Returns
        -------
        torchlens.debug.TraceAudit
            Prioritized findings plus checks run and skipped-check reasons.
        """

        from ..debug._audit import audit_trace

        return audit_trace(self)

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

        from ..captured_run import forget_event_stream

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
    backward_pass_logs: dict[int, BackwardPass]
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
    _module_call_accessor: Any
    _op_accessor_cache: Any
    _layer_accessor_cache: Any
    _receptive_field_solution: Any
    _rf_directional_solutions: Any
    _tl_rf_probe_active: Any

    #: Legacy persisted spellings ``__setstate__`` still adapts, published by
    #: the writer contract as the record's alias rules (alias-or-fail,
    #: ecosystem MEMO 3.3). The reasoned ledger lives in
    #: ``_trace_legacy_state.py``.
    PORTABLE_STATE_ALIASES: ClassVar[frozenset[str]] = TRACE_PORTABLE_STATE_ALIASES

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "trace_label": FieldPolicy.KEEP,
        "model_class_name": FieldPolicy.KEEP,
        "model_label": FieldPolicy.KEEP,
        "backend": FieldPolicy.KEEP,
        "backend_runtime_config": FieldPolicy.KEEP,
        "backend_runtime_device_summary": FieldPolicy.KEEP,
        "backend_runtime_version": FieldPolicy.KEEP,
        # Provenance marker for traces cooked from a Recording: postprocess
        # runs exhaustive-style (capture_mode stays "exhaustive" for behavior
        # compatibility), but the marker records the true origin so gates and
        # diagnostics never mistake a cooked projection for a live exhaustive
        # capture. Session-only; not a portable fact.
        "_cooked_from": FieldPolicy.DROP,
        # Settlement-authority state (torchlens/capture/outcome.py). The
        # settled outcome is a PERSISTED portable fact (tlspec v7): saved as
        # its string-only payload, parsed against closed vocabularies and the
        # status coherence matrix at load. The phase marker and stop-request
        # latch are strictly session-transient.
        "_capture_outcome": FieldPolicy.KEEP,
        "_capture_phase": FieldPolicy.DROP,
        "_settlement_ops_committed": FieldPolicy.DROP,
        "_stop_requested": FieldPolicy.DROP,
        "_paddle_capture_depth": FieldPolicy.DROP,
        "_paddle_op_captures": FieldPolicy.DROP,
        "_paddle_alias_annotations": FieldPolicy.DROP,
        "_paddle_capture_gap_markers": FieldPolicy.DROP,
        "_tf_unresolved_producers": FieldPolicy.DROP,
        "_tf_init_op_labels": FieldPolicy.DROP,
        "_tf_op_captures": FieldPolicy.DROP,
        "_tf_validation_result": FieldPolicy.DROP,
        "_tl_save_selector_fire_count": FieldPolicy.DROP,
        "_module_call_accessor": FieldPolicy.DROP,
        "_op_accessor_cache": FieldPolicy.DROP,
        "_layer_accessor_cache": FieldPolicy.DROP,
        "_receptive_field_solution": FieldPolicy.DROP,
        "_rf_directional_solutions": FieldPolicy.DROP,
        "module_identity_mode": FieldPolicy.KEEP,
        "param_source": FieldPolicy.KEEP,
        "derived_grads": FieldPolicy.KEEP,
        "num_context_lines": FieldPolicy.KEEP,
        "_optimizer": FieldPolicy.DROP,
        "tlspec_version": FieldPolicy.KEEP,
        "_tracing_finished": FieldPolicy.KEEP,
        "capture_mode": FieldPolicy.KEEP,
        # L7a structure-only mode marker: persists as of tlspec v8 (the
        # coordinated bump) with the marker-coherence load-validation rows in
        # torchlens/_io/forgery_validation.py (M-C2/M-C3; M-C1's form-(a)
        # stripped-marker case is a documented scope statement there).
        "structure_only": FieldPolicy.KEEP,
        # L6 resolved-intervention audit record: persists as of tlspec v8
        # (reprs/identities/digests only, never raw values) with its load
        # validation in torchlens/_io/forgery_validation.py.
        "intervention_audit": FieldPolicy.KEEP,
        "_runnable": FieldPolicy.DROP,
        "_fast_run_session": FieldPolicy.DROP,
        "_distributed_plane_p": FieldPolicy.DROP,
        "escape_detector_mode": FieldPolicy.DROP,
        "escape_detector_verified": FieldPolicy.DROP,
        "escape_diagnostics": FieldPolicy.DROP,
        "escape_detector_event_count": FieldPolicy.DROP,
        "escape_detector_callback_ns": FieldPolicy.DROP,
        "escape_detector_backward_coverage": FieldPolicy.DROP,
        "completeness_witness_mode": FieldPolicy.DROP,
        "completeness_witness_verified": FieldPolicy.DROP,
        "completeness_diagnostics": FieldPolicy.DROP,
        "completeness_decompositions": FieldPolicy.DROP,
        "completeness_witness_event_count": FieldPolicy.DROP,
        "completeness_witness_accounted_count": FieldPolicy.DROP,
        "completeness_witness_expected_opaque_count": FieldPolicy.DROP,
        "completeness_witness_unaccounted_count": FieldPolicy.DROP,
        "completeness_witness_callback_ns": FieldPolicy.DROP,
        # grind-r5 P7: the escape-disclosure VERDICT persists (negative claims
        # only -- __getstate__/__setstate__ degrade anything else to None), so
        # a capture TorchLens refused to bless is no longer byte-
        # indistinguishable from a clean one after save/load. rescue_rerun
        # stays session-time per docs/migration/scoped_detached_patching.md.
        "capture_verified": FieldPolicy.KEEP,
        "capture_verification_reason": FieldPolicy.KEEP,
        "rescue_rerun": FieldPolicy.DROP,
        "capture_owner_thread_id": FieldPolicy.DROP,
        "capture_owner_thread_qualified": FieldPolicy.DROP,
        "capture_thread_count_start": FieldPolicy.DROP,
        "capture_thread_count_end": FieldPolicy.DROP,
        "capture_thread_activity_detected": FieldPolicy.DROP,
        "capture_guard_passes": FieldPolicy.DROP,
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
        # Session-time measurement knob (like ``backward_ready``): it selects how
        # ``forward_peak_memory`` was measured during this capture and has no
        # meaning for a loaded artifact, which never re-measures. Portable load
        # restores the default ``False``, so it stays out of
        # ``MODEL_LOG_FIELD_ORDER`` and out of the portable schema.
        "measure_python_peak_memory": FieldPolicy.DROP,
        # Session-time witness knob for collective boundary records: it selects
        # what capture PAID FOR (digests or nothing), not what a trace means;
        # the per-boundary witness.policy_resolved field is the portable
        # evidence. Portable load restores the default "none".
        "distributed_witness": FieldPolicy.DROP,
        # Session-time resource ceiling: it bounds what THIS process was willing
        # to retain and has no meaning for a loaded artifact, which retains
        # nothing. Portable load restores the default, so it stays out of
        # ``MODEL_LOG_FIELD_ORDER`` and out of the portable schema.
        "save_budget": FieldPolicy.DROP,
        "raise_on_nan": FieldPolicy.KEEP,
        # Session-time recording knob (same class as measure_python_peak_memory):
        # it selects what capture PAID FOR (per-op finiteness checks), not what a
        # trace means. Portable load restores the default False, so it stays out
        # of MODEL_LOG_FIELD_ORDER and out of the portable schema.
        "track_nonfinite": FieldPolicy.DROP,
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
        # L1 grouping surface: persists as of tlspec v8; the grouping-policy
        # stamp validates at load (C1-C8, postprocess/_grouping_stamp.py).
        "grouping": FieldPolicy.KEEP,
        "grouping_policy": FieldPolicy.KEEP,
        # L8/F6 shard-local capture marker (merge-ranks C2 substrate):
        # persists as of tlspec v8 with the closed-vocabulary load validation
        # in torchlens/_io/forgery_validation.py. Value vocabulary
        # ("rank_local_shard") is DOCUMENTED-UNSTABLE pending its S2 amendment.
        "distributed_scope": FieldPolicy.KEEP,
        # L9 backward-residuals surface: both persist as of tlspec v8 with
        # closed-vocabulary load validation in _io/forgery_validation.py.
        # Value vocabularies provisional (E-L9-4 routing).
        "grad_fn_timing_provenance": FieldPolicy.KEEP,
        "checkpoint_invocation_witness": FieldPolicy.KEEP,
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
        "_warned_once": FieldPolicy.DROP,
        "_spec_revision": FieldPolicy.KEEP,
        "_out_recipe_revision": FieldPolicy.KEEP,
        "_append_sequence_id": FieldPolicy.KEEP,
        "_last_hook_handle_ids": FieldPolicy.DROP,
        "_predicate_save_options": FieldPolicy.DROP,
        "_predicate_history_size": FieldPolicy.DROP,
        "_predicate_history": FieldPolicy.DROP,
        "_predicate_lookback": FieldPolicy.DROP,
        "_predicate_lookback_payload_policy": FieldPolicy.DROP,
        "_capture_config": FieldPolicy.DROP,
        "_stop_directive": FieldPolicy.DROP,
        "_halt_returns_partial_trace": FieldPolicy.DROP,
        "_predicate_save_decisions": FieldPolicy.DROP,
        "_predicate_current_contexts": FieldPolicy.DROP,
        "_predicate_lookback_candidates": FieldPolicy.DROP,
        "_postprocessing_active": FieldPolicy.DROP,
        "_raw_transform_escape_detected": FieldPolicy.DROP,
        "_raw_dynamo_region_detected": FieldPolicy.DROP,
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
        "_buffer_write_tracker": FieldPolicy.DROP,
        "_param_storage_addresses": FieldPolicy.DROP,
        "_buffer_initial_values": FieldPolicy.BLOB_RECURSIVE,
        "_buffer_persistence": FieldPolicy.KEEP,
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
        "ops_with_params": FieldPolicy.DROP,
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
        # r83 S4: live-capture scratch reinstated by ``__setstate__`` alongside
        # ``_tl_backward_hooked_tensor_keys``, but never registered here. Any
        # trace that went through ``__setstate__`` -- which ``cache=True`` makes
        # routine -- therefore tripped the catalog tripwire and failed
        # ``save(level="runnable")`` outright on the SECOND capture. DROP, like
        # its sibling: pending live-fire records are session state and are
        # already reset on rehydrate.
        "_pending_live_fire_records": FieldPolicy.DROP,
        "_module_logs": FieldPolicy.DROP,
        "_param_logs_by_module": FieldPolicy.DROP,
        "_raw_graph_ws": FieldPolicy.DROP,
        "_module_capture_ws": FieldPolicy.DROP,
        "_wrapper_runtime_ws": FieldPolicy.DROP,
        # The per-trace columnar Op row store (torchlens._trace_core). Never
        # portable: plain pickle re-materializes each Op as a detached row
        # from its own state, and .tlspec artifacts stay object-shaped until
        # the M11 direct semantic serialization.
        "_trace_core": FieldPolicy.DROP,
        # L3 primitive-op profile: persists as of tlspec v8; loaded profiles
        # validate through validate_loaded_primitive_profile whenever present.
        "_primitive_op_profile": FieldPolicy.KEEP,
        "_pre_forward_rng_states": FieldPolicy.DROP,
        # r63 C1: pre-clone per-slot state metadata signatures (producer-side only,
        # never portable) and the buffer storage-pointer attribution index.
        # r77 F2: capture-time persistent-buffer name universe + geometry
        # (producer-side only, never portable).
        "_buffer_storage_addresses": FieldPolicy.DROP,
        "_mlx_saved_payloads": FieldPolicy.DROP,
        "_mlx_capture_depth": FieldPolicy.DROP,
        "_out_writer": FieldPolicy.DROP,
        # Runtime-only: the live per-device accountant that enforces
        # ``save_budget`` while payloads are being retained. It describes what
        # THIS process was willing to allocate, so it is never portable; a loaded
        # artifact retains nothing and rebuilds it from the restored option.
        "_save_budget_accountant": FieldPolicy.DROP,
        "_keep_outs_in_memory": FieldPolicy.DROP,
        "_grad_stream_retain_in_memory": FieldPolicy.DROP,
        "_defer_streaming_bundle_finalization": FieldPolicy.DROP,
        "_out_sink": FieldPolicy.DROP,
        # Runtime-only: the set of dispatchable op func-call-ids the orphan-removal
        # pass pruned, read by the validation dispatch-count backstop. Never portable.
        "_orphan_pruned_func_call_ids": FieldPolicy.DROP,
        # Runtime-only (r29 F3b): the capture-time (slot -> producer) parent-edge
        # truth keyed by raw label, read by the capture_edge_survival metadata
        # invariant. Registered in _io/scrub.py's runtime-only list; declared here
        # so the portable-state cover stays exhaustive. Never portable.
        "_capture_parent_edge_truth": FieldPolicy.DROP,
        # track_nonfinite's runtime store (events/unchecked/pending), set lazily
        # by the torch op-finalize hook; same runtime-bookkeeping class as
        # _capture_parent_edge_truth: never persisted, absent on loaded traces.
        "_nonfinite_capture": FieldPolicy.DROP,
        "_capture_events": FieldPolicy.DROP,
        "_capture_session": FieldPolicy.DROP,
        "_tl_backward_hooked_tensor_keys": FieldPolicy.DROP,
        "_tl_grad_hook_owner_by_label": FieldPolicy.DROP,
        # RF probe suppression flag: declared statically so a Trace serialized
        # BEFORE any receptive-field probe has the same class spec as one
        # serialized after (the probe used to setdefault this at call time).
        "_tl_rf_probe_active": FieldPolicy.DROP,
        "_active_backward_pass_index": FieldPolicy.DROP,
        "_backward_roots_by_pass": FieldPolicy.DROP,
        "_backward_projection_event_count": FieldPolicy.DROP,
        "_backward_projection_revision": FieldPolicy.DROP,
        "_backward_projection_fold_state": FieldPolicy.DROP,
        "_implicit_backward_pass_open": FieldPolicy.DROP,
        "_tl_backward_triggers_disarmed": FieldPolicy.DROP,
        # Idempotent-cleanup sentinel (b6-opus R25): stamped by cleanup() so a
        # second cleanup() early-returns and every reader of a husked trace
        # funnels into the one typed TraceCleanedUpError. Session-time by
        # construction -- a load rebinds a live trace, never a husk.
        "_tl_cleaned_up": FieldPolicy.DROP,
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
        "_session_param_inventory": FieldPolicy.DROP,
        "_session_buffer_inventory": FieldPolicy.DROP,
        "_session_buffer_identity": FieldPolicy.DROP,
        "backward_root_grad_fn_object_ids": FieldPolicy.KEEP,
        "backward_durations": FieldPolicy.KEEP,
        "num_backward_passes": FieldPolicy.KEEP,
        "backward_peak_memory": FieldPolicy.KEEP,
        "backward_memory_backend": FieldPolicy.KEEP,
        "_backward_gradfn_refs": FieldPolicy.DROP,
        # B1-17: the remaining undeclared runtime Trace attrs the widened
        # lockstep gate finds across the 30-axis postprocess matrix. All are
        # session-time by construction; declaring them makes the gate's
        # authority the policy table rather than the pre-spec allowance in
        # `_io/scrub.py`.
        #
        # Predicate-intervention dedup caches (INTERVENED axis).
        # `..._target_keys` pins the live intervention spec, so it is also
        # dropped at the postprocess seam.
        "_tl_predicate_intervention_spec_keys": FieldPolicy.DROP,
        "_tl_predicate_intervention_target_keys": FieldPolicy.DROP,
        # Streaming-bundle provenance (STREAMING axes). Already popped and
        # restored around the scrub by `_io/bundle.py` and cleared by
        # `data_classes/cleanup.py`; a load rebinds them fresh, so DROP is the
        # existing behavior made declarative.
        "_source_bundle_path": FieldPolicy.DROP,
        "_source_bundle_manifest_sha256": FieldPolicy.DROP,
        # Two-pass selective-save retention flag (DEFERRED_RETENTION axis).
        "_retain_layers_to_save_output_parents": FieldPolicy.DROP,
        # Validation side-channel state (B1-04). `validate_forward_pass` is
        # a public method on a user-held Trace, and
        # validation ENTRY unconditionally sets `_last_validation_failure`
        # (`reset_validation_failure` writes None on every run), so a plain
        # validate-then-save sequence hit
        # `TorchLensIOError: Trace._last_validation_failure is missing from
        # PORTABLE_STATE_SPEC` -- a hard refusal on a completely ordinary
        # workflow. Both attrs are session-time diagnostics carried on the
        # trace as a side channel, exactly the `_fast_run_session` class, and
        # `ValidationFailure`/`ValidationDiagnostic` are not portable records.
        "_last_validation_failure": FieldPolicy.DROP,
        "_validation_diagnostics": FieldPolicy.DROP,
        # Session-time semantic-output scratch (B1-02). Written at capture
        # entry, consumed only by ``decode_outputs_for_trace``, and dropped on
        # every settlement path (``capture/trace.py``'s
        # ``_drop_semantic_output_transients`` plus the postprocess seam).
        # Declared here so the runtime-declaration lockstep gate can SEE them
        # on the halted axis and so the portable scrub is policy-driven rather
        # than a hand-maintained name list in ``_io/scrub.py``:
        # ``_output_tokenizer`` holds a LIVE user tokenizer and
        # ``_semantic_output_metadata`` a model-derived key, so a surviving
        # copy leaks vocab/merges into a plain pickle of the Trace.
        "_output_style": FieldPolicy.DROP,
        "_output_head": FieldPolicy.DROP,
        "_output_tokenizer": FieldPolicy.DROP,
        "_semantic_output_metadata": FieldPolicy.DROP,
    }
    FIELD_POLICY = build_record_field_policy_table(
        MODEL_LOG_FIELD_ORDER,
        PORTABLE_STATE_SPEC,
        fork_policy=MODEL_LOG_FIELD_FORK_POLICY,
        default_fill_state=_MODEL_LOG_DEFAULT_FILL,
        schema_key="trace",
    )
    PORTABLE_STATE_SPEC = portable_state_spec_from_policy(FIELD_POLICY)

    def __init__(
        self,
        model_class_name: str,
        output_device: str = "same",
        activation_transform: ActivationPostfunc | None = None,
        grad_transform: GradientPostfunc | None = None,
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
        measure_python_peak_memory: bool = False,
        distributed_witness: str = "none",
        save_budget: SaveBudgetOption = "auto",
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
        self._tracing_finished = False
        self._raw_graph_ws = RawGraphWorkspace()
        self._module_capture_ws = ModuleCaptureWorkspace()
        self._wrapper_runtime_ws = WrapperRuntimeWorkspace()
        self._primitive_op_profile = None
        self._module_capture_ws.module_build_data = _init_module_hierarchy_data()
        self.capture_mode: Literal["exhaustive", "predicate"] = "exhaustive"
        # L7a: True marks a structure-only capture (the flag declares the
        # mode; every value-bearing claim is a hypothesis and value consumers
        # gate through torchlens.capture.structure_only). DOCUMENTED-UNSTABLE
        # spelling pending naming-session ratification.
        self.structure_only: bool = False
        # L6: session-time audit records for resolved-selection interventions
        # (query repr + resolve digest + per-site relations; patch_from adds
        # source identity + value digests). DROP under v7; pre-release-
        # registered, flips to KEEP at the wave-3 bump. DOCUMENTED-UNSTABLE.
        self.intervention_audit: list[dict[str, Any]] = []
        self._runnable = RunnableTraceState()
        self._fast_run_session: Any | None = None
        # Merge-ranks C2 plane-P: session-time physical dispatch observation
        # journal for armed captures (census criteria 2-4 evidence). DROP
        # under its private name; never survives save/load. DOCUMENTED-UNSTABLE.
        self._distributed_plane_p: Any | None = None
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
        self._activation_transform_repr = _scrubbed_transform_repr(activation_transform)
        self.save_raw_activations = save_raw_activations
        self.input_annotations: dict[str, Any] = {}
        self.grad_transform = grad_transform
        self.grad_transform_repr = repr(grad_transform) if grad_transform is not None else None
        self.save_raw_gradients = save_raw_gradients
        self.save_mode = save_mode
        self._source_code_blob: dict[str, str] = {}
        self._source_model_ref: weakref.ReferenceType[nn.Module] | None = None
        self.parent_run: weakref.ReferenceType[Trace] | None = None
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
        self.measure_python_peak_memory = measure_python_peak_memory
        self.distributed_witness = distributed_witness
        self.save_budget = save_budget
        # Built once per capture; ``None`` when budgeting is disabled. Charged on
        # the hot path by the activation-save paths in the torch backend.
        self._save_budget_accountant = SaveBudget.from_option(save_budget)
        self.facet_registry_snapshot = facet_registry_snapshot
        self.raise_on_nan: bool = False
        self.track_nonfinite: bool = False
        self.annotations: dict[str, Any] = {}
        self.code_context: list[FuncCallLocation] = []
        self.manual_tensor_connections: list[tuple[str, str]] = []
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
        self._out_identity_cache: dict[
            int, tuple[torch.Tensor, str, torch.Tensor, int | None, int]
        ] = {}
        self._out_hash_cache: dict[str, tuple[str, torch.Tensor]] = {}
        self._code_context_cache: dict[Any, tuple[Any, ...]] = {}
        self._halt_returns_partial_trace = False
        self._replay_arg_version_data_complete = True
        self.save_arg_values = save_arg_values
        self.save_grads = "all" if save_grads is True else save_grads
        self.capture_tensor_grad_hooks = capture_tensor_grad_hooks
        self.save_code_context = save_code_context
        self.save_rng_states = save_rng_states
        self.recurrence_detection = recurrence_detection
        # L1 grouping surface: "structural" is the only entry-legal knob
        # value in wave 0 (others refuse typed at trace entry); the stamp is
        # written by each producer once step-7 grouping settles.
        self.grouping = "structural"
        self.grouping_policy: dict[str, Any] | None = None
        # L8/F6: set to "rank_local_shard" by C2 capture when any param/input
        # carries a non-Replicate DTensor placement (the sharded predicate in
        # torchlens.distributed._dtensor). Unreachable until the D-L8-CAP
        # capture relaxation: every sharded capture still refuses at entry.
        self.distributed_scope: str | None = None
        # L9 backward residuals: per-fire timing clock provenance
        # ("unmeasured" until a timing prehook arms) and the checkpoint-
        # invocation witness summary (None until torch capture builds it).
        # Both DROP-gated pre-bump; spellings DOCUMENTED-UNSTABLE.
        self.grad_fn_timing_provenance: str = "unmeasured"
        self.checkpoint_invocation_witness: dict[str, Any] | None = None
        self.verbose = verbose
        self.profile_enabled = False
        self.has_gradients = False
        self.mark_layer_depths = mark_layer_depths
        self.graph_shape_hash: str | None = None
        self._intervention_spec: InterventionSpec | None = InterventionSpec()
        self.state_history: list[Any] = []
        self.observer_spans: list[dict[str, Any]] = list(_state._active_record_spans.get())
        self.last_run: Any | None = None
        self.append_history: list[dict[str, Any]] = []
        self._has_direct_writes = False
        self._warned_direct_write = False
        self._warned_mutate_in_place = False
        self._warned_once: set[str] = set()
        self._raw_transform_escape_detected = False
        self._raw_dynamo_region_detected = False
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
        self._out_writer: BundleStreamWriter | None = None
        self._keep_outs_in_memory: bool = True
        self._defer_streaming_bundle_finalization: bool = False
        self._out_sink: Callable[[str, torch.Tensor], None] | None = None
        # Model structure info (computed @properties: is_recurrent,
        # max_layer_op_count, is_branching, has_conditional_branching)

        # Tensor Tracking - post-processed (populated after _tracing_finished=True):
        self.layer_list: list[Op] = []  # ordered list of all layer ops
        self.layer_dict_main_keys: dict[str, Op] = OrderedDict()  # primary label -> entry
        self.layer_dict_all_keys: dict[str, Op] = OrderedDict()  # all lookup keys -> entry
        self.layer_logs: dict[str, Layer] = OrderedDict()  # no-pass label -> aggregate Layer
        self.op_labels: list[str] = []  # pass-qualified labels (e.g. "conv2d_1_1:1")
        self.layer_labels: list[str] = []  # pass-stripped labels (e.g. "conv2d_1_1")
        self.layer_num_calls: dict[str, int] = OrderedDict()  # no-pass label -> pass count
        self.by_pass: dict[int, list[int]] = {}
        self._layer_nums_to_save: list[int] = []  # ordinal positions of layers to save
        self._grad_op_nums_to_save: list[int] | str = []
        self.num_ops: int = 0  # total operations after postprocessing

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
        self.buffer_num_calls: dict[str, int] = {}
        self._buffer_accessor = None
        self._buffer_write_tracker: Any | None = None
        self._buffer_initial_values: dict[str, Any] = {}
        self.internal_source_ops: list[str] = []
        self.internal_sink_ops: list[str] = []
        self.internally_terminated_bool_ops: list[str] = []
        self.conditional_branch_edges: list[tuple[str, str]] = []
        self.conditional_records: list[ConditionalEvent] = []
        self.conditional_arm_entry_edges: dict[tuple[int, str], list[tuple[str, str]]] = {}
        self.conditional_edge_call_indices: dict[tuple[str, str, int, str], list[int]] = {}
        self.conditionals = ConditionalAccessor()
        self._orphan_labels: list[str] = []
        self._orphan_logs: tuple[Op, ...] = ()
        self.orphan_records: list[dict[str, Any]] = []
        self._saved_grad_labels: set[str] = set()
        self.layers_with_params: dict[str, list[Any]] = defaultdict(list)
        # Maps equivalence_class -> set of layer labels that share
        # that equivalence type (populated by loop_detection.py).
        self.op_equivalence_classes: dict[str, set[str]] = defaultdict(set)

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
        self.param_logs: ParamAccessor = ParamAccessor({})
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
        self.grad_fn_logs: dict[int, GradFn] = OrderedDict()
        self.grad_fn_order: list[int] = []
        self.backward_pass_logs: dict[int, BackwardPass] = OrderedDict()
        self._grad_fn_param_refs: dict[str, str] = {}
        self._param_log_by_pid: dict[int, str] = {}
        # r79 session-leak fix: the RECORDED prep inventories of stamped
        # parameters/buffers. Session-scoped strong refs -- authoritative source
        # for stamp cleanup (a param/buffer popped from the live module tree
        # mid-forward escapes re-traversal but never this list); emptied by
        # ``_cleanup_model_session``. Never portable.
        self._session_param_inventory: list[Any] = []
        self._session_buffer_inventory: list[Any] = []
        # r81: id(tensor) -> _SessionBufferStamp identity records for every
        # buffer stamp written this session -- the buffer-rung storage-identity
        # belt (param-rung ``_param_ref is value`` parity). Session-scoped
        # strong refs (tensor + stamp-time storage keeper); emptied by
        # ``_cleanup_model_session``. Never portable.
        self._session_buffer_identity: dict[int, Any] = {}
        self.backward_root_grad_fn_object_ids: list[int] = []
        self.backward_durations: list[Duration] = []
        self.num_backward_passes: int = 0
        self.backward_peak_memory: Bytes = Bytes(0)
        self.backward_memory_backend: str = "unknown"
        _state.register_log(self)

    # ********************************************
    # ************ Built-in Methods **************
    # ********************************************

    def __len__(self) -> int:
        """Number of layer-pass entries. Uses final list after postprocessing, raw dict during logging."""
        if self._tracing_finished:
            return len(self.layer_list)
        else:
            return len(self._raw_graph_ws.raw_layer_dict)

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
            raise InvalidArgumentError(
                "annotate() requires data=, image=, or both",
                code="annotation_payload_missing",
                remedy="pass data=, image=, or both to annotate()",
            )
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
            raise InvalidArgumentError(
                "annotate(data=torch.Tensor) is supported only for torch traces in this "
                f"release; this trace uses backend={backend_name!r}",
                code="annotation_backend_unsupported",
                remedy="pass JSON-serializable annotation data for non-torch traces",
                backend=backend_name,
            )
        from .._io.payload_codec import get_payload_codec

        decision = get_payload_codec(backend_name).validate_for_save(tensor, strict=True)
        if decision.__class__.__name__ != "Ok":
            reason = getattr(decision, "text", "unsupported tensor payload")
            raise InvalidArgumentError(
                f"Annotation tensor is not portable for backend {backend_name!r}: {reason}",
                code="annotation_tensor_not_portable",
                remedy="pass a dense, codec-supported tensor as annotation data",
                backend=backend_name,
                reason=reason,
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
            raise InvalidArgumentError(
                "annotate(data=...) must be JSON-serializable or a torch.Tensor; "
                "values are never silently stringified",
                code="annotation_not_json_serializable",
                remedy="convert arrays to torch.Tensor for blob persistence",
                argument="data",
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

        blobs = self._annotation_blobs
        if not isinstance(blobs, dict):
            return None
        kept = {
            key: value
            for key, value in blobs.items()
            if not key.startswith(self._DERIVED_ANNOTATION_BLOB_PREFIXES)
        }
        return kept or None

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
            raise InvalidArgumentError(
                'annotations["user"] must be a dict to store user annotations; '
                f"found {type(user_annotations).__name__}",
                code="annotation_namespace_invalid",
                remedy='restore annotations["user"] to a dict before annotating',
            )
        return user_annotations

    def _mark_annotations_mutated(self) -> None:
        """Invalidate render-only caches after an annotation mutation.

        Returns
        -------
        None
            The cached sibling-ordering decision is discarded.
        """

        self.__dict__.pop("_last_sibling_ordering_decision", None)
        self.__dict__.pop("_last_encoding_state", None)

    def find_layers(self, query: str, *, limit: int = 10) -> list[str]:
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

        from ..capture.outcome import CaptureOutcome, CaptureStatus, outcome_for

        if self.__dict__.get("_tl_cleaned_up", False):
            # cleanup() deletes the settled sidecar with every other field.
            # The frozen vocabulary designates UNKNOWN as the most-restrictive
            # value and the save gate already lands there; ``None`` is outside
            # the vocabulary and was the one unsettled read (b6-opus R25).
            return CaptureOutcome(status=CaptureStatus.UNKNOWN, derived=True)
        return outcome_for(self)

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
        nonfinite_summary = nonfinite
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
            return iter(list(self._raw_graph_ws.raw_layer_dict.values()))

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Call :func:`torchlens.save` for this model log.

        Parameters
        ----------
        path:
            Destination bundle directory.
        **kwargs:
            Portable save options. For runnable saves, ``include_weights=True``
            bundles the full capture-time ``state_dict``: all named parameters
            and persistent buffers, as state rather than a reconstructed model.
            Independently, ``include_activations=True`` archives exactly the
            capture-time ``save=``-selected payloads for inspection and eligible
            byte-exact attestation; those payloads never seed execution.
            ``include_source`` (default ``True``) controls whether the captured
            model source code is embedded; set it ``False`` to strip verbatim
            source, docstrings, and source-file references from a shared
            ``.tlspec``. Absolute source paths are always relativized to a bare
            basename, so no ``$HOME`` / username is ever embedded. See
            :func:`torchlens.save` for the full option list.

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

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle state with non-picklable weakref-backed accessors stripped."""
        state = self.__dict__.copy()
        state["_pickle_module_accessor_state"] = self.__dict__.get("_module_logs")
        # Event streams never serialize (FieldPolicy.DROP): strip the stream
        # AND the projection guard derived from it, or a restored trace would
        # claim a source-process revision/fold-state over a fresh empty stream
        # (silently dropped passes / stale-id partial folds on the advertised
        # restored-trace backward-capture path).
        state.pop("_capture_events", None)
        state.pop("_backward_projection_event_count", None)
        state.pop("_backward_projection_revision", None)
        state.pop("_backward_projection_fold_state", None)
        state.pop("_tl_materializing_backward_projection", None)
        state["_module_logs"] = None
        state["_buffer_accessor"] = None
        state["_source_model_ref"] = None
        state["parent_run"] = None
        state["last_run"] = None
        state["_out_identity_cache"] = {}
        state["_out_hash_cache"] = {}
        state["_code_context_cache"] = {}
        # Lazy per-instance accessor caches (FieldPolicy.DROP) rebuild on
        # demand after restore; carrying them would make pickle bytes depend
        # on which accessors were touched (access-is-pure contract).
        state.pop("_op_accessor_cache", None)
        state.pop("_layer_accessor_cache", None)
        state.pop("_module_call_accessor", None)
        # Lazy influence-geometry caches (FieldPolicy.DROP) hold
        # mappingproxy-bearing solution objects after any receptive-field
        # access, so carrying them made pickle.dumps/copy.deepcopy crash with
        # "cannot pickle 'mappingproxy' object" while tl.save succeeded on
        # the same trace. They rebuild on first access after restore.
        state.pop("_receptive_field_solution", None)
        state.pop("_rf_directional_solutions", None)
        # Render diagnostic from the last draw(): its EncodingChannelSpec can
        # hold the RAW user callable a `color_by=lambda ...` passed in, so
        # carrying it made pickle.dumps crash after an ordinary draw while
        # tl.save succeeded on the same trace (the R10-7 raw-callable class).
        # Runtime-only either way; it rebuilds on the next draw.
        state.pop("_last_encoding_state", None)
        state.pop("_container_ordinals_by_output_op_label", None)
        state.pop("_container_ordinals_by_input_func_call_id", None)
        # B1-02: the semantic-output scratch never serializes. Plain pickle
        # legitimately carries most session-time DROP state (in-process
        # round-trips need `backward_ready`, `save_budget`, ...), but these two
        # hold LIVE USER OBJECTS -- an HF tokenizer and a model-derived
        # metadata key. A surviving copy reconstructs the tokenizer on load and
        # bakes its vocab/merges into an artifact the user believes is a graph
        # (measured 47KB -> 238KB). Capture drops them at every settlement
        # path; this is the serialization boundary's own belt.
        state.pop("_output_style", None)
        state.pop("_output_head", None)
        state.pop("_output_tokenizer", None)
        state.pop("_semantic_output_metadata", None)
        # Capture-session predicate carriers (FieldPolicy.DROP) hold LIVE USER
        # CLOSURES when predicate capture/intervention/halt was used (a
        # ``tl.when(...)`` conditional is a local function), so pickling a
        # trace that ``tl.save`` accepted failed on exactly these fields.
        # Serialize to the loaded-artifact form instead (absent / None), which
        # every post-capture consumer already tolerates -- same belt as the
        # semantic-output scratch above.
        state.pop("_stop_directive", None)
        state.pop("_capture_config", None)
        state["_predicate_save_options"] = None
        # R10-6: run(fast=True) binds a session holding weakrefs/compiled
        # binders; every other session workspace pops here but this one did
        # not, so pickle.dumps/copy.deepcopy after a fast run crashed with
        # "cannot pickle 'weakref.ReferenceType'". Session-time (FieldPolicy
        # DROP under its private name); a restored trace re-runs verified.
        state.pop("_fast_run_session", None)
        # Session-time plane-P dispatch journal (merge-ranks C2): tuples of
        # per-dispatch facts, meaningless outside the capture session.
        state.pop("_distributed_plane_p", None)
        # R10-7: the REPR is scrubbed below, but the RAW user callables stayed
        # in state, so a lambda transform= made pickle.dumps crash while
        # tl.save succeeded on the same trace. Serialize to the loaded-artifact
        # form (None), which every post-load consumer already tolerates.
        state["activation_transform"] = None
        state["grad_transform"] = None
        state["_output_transform"] = None
        # R10-7b: the MODERN tl.trace(..., transform=...) kwarg populates
        # `_transform`, a fourth raw-callable holder the original fix missed
        # (it covered only the deprecated activation_transform= spelling).
        state["_transform"] = None
        state.pop("_raw_graph_ws", None)
        state.pop("_module_capture_ws", None)
        state.pop("_wrapper_runtime_ws", None)
        state.pop("_trace_core", None)
        state["_backward_gradfn_refs"] = []
        state["_tl_backward_hooked_tensor_keys"] = set()
        state.pop("_tl_grad_hook_owner_by_label", None)
        state["_pending_live_fire_records"] = []
        state["_last_hook_handle_ids"] = ()
        state["_activation_transform_repr"] = _scrubbed_transform_repr(self.activation_transform)
        # Runnable traces bind state as immutable MappingProxyType views, which
        # cannot be pickled/deepcopied. Preserve tensor identity while replacing
        # only those mapping proxies with ordinary dictionaries.
        state["_runnable"] = runnable_trace_state(self).pickle_safe_copy()
        # The settled capture outcome persists as its STRING-ONLY payload dict
        # (tlspec v7): loads parse it against the closed vocabularies and the
        # status coherence matrix, so a stale or forged record can only ever
        # degrade to UNKNOWN, never upgrade. A pre-settlement live object
        # persists None and loads derive from the structural lattice.
        outcome = state.get("_capture_outcome")
        state["_capture_outcome"] = (
            outcome.to_payload() if outcome is not None and hasattr(outcome, "to_payload") else None
        )
        # grind-r5 P7 (b3 opus R10-1 / b6 opus R16-1): the producer's NEGATIVE
        # verification claim persists -- a capture TorchLens itself refused to
        # bless must not round-trip into "no claim". A POSITIVE claim never
        # persists (a tampered artifact could otherwise forge verified=True;
        # verdicts may only degrade across persistence). ``rescue_rerun``
        # stays session-time per the migration doc.
        if state.get("capture_verified") is False:
            reason = state.get("capture_verification_reason")
            state["capture_verification_reason"] = reason if isinstance(reason, str) else None
        else:
            state["capture_verified"] = None
            state["capture_verification_reason"] = None
        state["tlspec_version"] = TLSPEC_VERSION
        return state

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
        # refuses to persist a positive claim (grind-r5 P7).
        cloned.capture_verified = self.capture_verified
        cloned.capture_verification_reason = self.capture_verification_reason
        memo[id(self)] = cloned
        return cloned

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore pickle state and rebuild weakref-backed links."""
        pickle_module_accessor_state = state.pop("_pickle_module_accessor_state", None)
        # P7/R10: restore the persisted NEGATIVE verification disclosure. The
        # row is string-only and can only ever WORSEN a verdict: anything but
        # the exact {"verified": False} shape is ignored (stays "no claim"), so
        # a forged row cannot bless a capture and a stripped row merely reverts
        # to the historical launder this closes for honest artifacts.
        verification_row = state.pop("_capture_verification", None)
        if (
            isinstance(verification_row, dict)
            and verification_row.get("verified") is False
            and isinstance(verification_row.get("reason"), (str, type(None)))
        ):
            state["capture_verified"] = False
            state["capture_verification_reason"] = verification_row.get("reason")
        for field_name in (
            *LEGACY_TRACE_BUILD_STATE_KEYS,
            # "_build_state" is the pre-M10 flat scratchpad key; the three
            # workspace keys are its dissolved successors. All transient,
            # all dropped on restore.
            "_build_state",
            "_raw_graph_ws",
            "_module_capture_ws",
            "_wrapper_runtime_ws",
            "_trace_core",
        ):
            state.pop(field_name, None)
        serialized_tlspec_version = read_tlspec_version(
            state, cls_name=type(self).__name__, cls=type(self)
        )
        containers_were_serialized = "_containers" in state and state["_containers"] is not None
        setstate_defaults = {
            **_MODEL_LOG_DEFAULT_FILL,
            "tlspec_version": serialized_tlspec_version,
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
            "save_grads": None,
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
            "_out_writer": None,
            "_keep_outs_in_memory": True,
            "_defer_streaming_bundle_finalization": False,
            "_out_sink": None,
            "append_history": [],
            "_source_code_blob": {},
            "_source_model_ref": None,
            "backward_ready": False,
            "inference_only": False,
            "chunked_forward": False,
            "module_filter": None,
            "raise_on_nan": False,
            "track_nonfinite": False,
            "structure_only": False,
            "intervention_audit": [],
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
        runnable_state = normalize_runnable_trace_state(state)
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
        state.pop("_keep_grads_in_memory", None)
        state.pop("_grad_stream_retain_in_memory", None)
        pop_retired_legacy_keys(state)
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
        if state.get("measure_python_peak_memory") is None:
            state["measure_python_peak_memory"] = False
        # ``track_nonfinite`` is FieldPolicy.DROP for the same reason: a loaded
        # artifact never re-records, so load restores the default and the
        # queryable record falls back to the saved-payload basis.
        if state.get("track_nonfinite") is None:
            state["track_nonfinite"] = False
        if state.get("distributed_witness") is None:
            state["distributed_witness"] = "none"
        # ``save_budget`` is FieldPolicy.DROP, so a portable artifact never
        # carries a real value; it arrives absent or None and is restored to the
        # default. A loaded trace retains nothing, so there is no ceiling to honor.
        if state.get("save_budget") is None:
            state["save_budget"] = "auto"
        if state["inference_only"] is None:
            state["inference_only"] = False
        if state["chunked_forward"] is None:
            state["chunked_forward"] = False
        if runnable_state.poisoned is None:
            runnable_state.poisoned = False
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
        from .._io.state_keys import refuse_callable_shadowing_state_keys

        refuse_callable_shadowing_state_keys(type(self), state)
        # Restore is keyed off the INCOMING state, never the reused object's
        # dict: a legacy artifact that baked in a stream is discarded, and a
        # reused object's prior stream must not survive a re-restore (it would
        # retain the previous trace's events and re-serialize them later).
        state.pop("_capture_events", None)
        self.__dict__.update(state)
        # A persisted primitive-op profile (tlspec v8) is hostile artifact
        # input: validate its foreign keys whenever one is present.
        if self.__dict__.get("_primitive_op_profile") is not None:
            from ..validation._invariants_primitive_ops import validate_loaded_primitive_profile

            validate_loaded_primitive_profile(self)
        # Event streams never serialize (FieldPolicy.DROP), but a restored
        # trace remains a supported backward-capture target within the live
        # process, so restore installs a fresh stream EXPLICITLY here rather
        # than letting backward capture fabricate one silently on demand.
        # The stream is DETACHED, not blank: it records the restored
        # projection's pass-index base and cumulative baselines so a new
        # backward numbers itself after the preserved passes and a full
        # rebuild keeps (never silently erases) the pre-pickle projection.
        from ..ir import CaptureEvents

        self.__dict__["_capture_events"] = CaptureEvents.detached_from(self)
        # The projection guard is derived from the stream that just got
        # replaced; stale values from the source process (or a reused object)
        # would silently drop new passes or fold onto dead-process fold state.
        self.__dict__.pop("_backward_projection_event_count", None)
        self.__dict__.pop("_backward_projection_revision", None)
        self.__dict__.pop("_backward_projection_fold_state", None)
        self.__dict__.pop("_tl_materializing_backward_projection", None)
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
        # Persisted DROP-gated claim families become hostile artifact input at
        # this boundary. Validate them before any consumer can observe or bind
        # the restored values; wholly absent legacy families remain legal.
        from .._io.forgery_validation import validate_persisted_forgery_surfaces

        validate_persisted_forgery_surfaces(self)
        # grind-r5 P7: load-side twin of the __getstate__ sanitation -- a
        # tampered artifact claiming capture_verified=True (or any non-bool)
        # degrades to None/no-claim; only the producer's NEGATIVE claim (with
        # a string-only reason) is adopted. Verdicts never improve across a
        # save/load cycle.
        if self.__dict__.get("capture_verified") is not False:
            self.__dict__["capture_verified"] = None
            self.__dict__["capture_verification_reason"] = None
        elif not isinstance(self.__dict__.get("capture_verification_reason"), (str, type(None))):
            self.__dict__["capture_verification_reason"] = None
        # Resolve the settled capture outcome BEFORE core rehydration: adopt a
        # coherent persisted attestation, else derive from the structural
        # lattice (fail-closed to UNKNOWN on parse/coherence violations). The
        # helper reads only the restored __dict__ and never consults
        # rehydration side effects; rehydration keys on the same truthiness
        # convention independently.
        from ..capture.outcome import resolve_loaded_outcome

        self.__dict__["_capture_outcome"] = resolve_loaded_outcome(self.__dict__)
        # Grouping-policy stamp: adopt-or-degrade (L1). An absent stamp
        # (every pre-stamp v7 artifact) settles silently to the canonical
        # legacy settlement; an invalid one warns once and settles with the
        # violated rule's name. Monotonic: verdicts only worsen across
        # persistence, and the settled payload round-trips byte-stable. The
        # knob mirror normalizes first (a DROP-scrubbed field restores as an
        # explicit None, bypassing default fill).
        if self.__dict__.get("grouping") is None:
            self.__dict__["grouping"] = "structural"
        # L9: a DROP-scrubbed timing-provenance field restores as an explicit
        # None; normalize to the honest "unmeasured" state (a loaded pre-bump
        # artifact carries no timing evidence). The witness stays None.
        if self.__dict__.get("grad_fn_timing_provenance") is None:
            self.__dict__["grad_fn_timing_provenance"] = "unmeasured"
        from ..postprocess._grouping_stamp import settle_loaded_grouping_policy

        self.__dict__["grouping_policy"] = settle_loaded_grouping_policy(self.__dict__)
        # F9: adopt the restored detached records into a fresh sealed core so
        # loaded traces rejoin the single-truth store (best-effort — an abort
        # preserves the coreless-island behavior; backward records stay
        # detached by design). See data_classes/_trace_rehydrate.py.
        from ._trace_rehydrate import rehydrate_trace_core

        rehydrate_trace_core(self)
        if pickle_module_accessor_state is not None:
            from .._io.accessor_rebuild import rebuild_trace_accessors

            rebuild_trace_accessors(
                self,
                pickle_module_accessor_state._dict,
                pickle_module_accessor_state._list,
                pickle_module_accessor_state._pass_dict,
            )
        # Episode-ledger load validation (S7): an episode payload in the
        # restored annotations is validated fail-closed -- illegal attachment
        # refuses typed, incoherent geometry quarantines with one warning.
        if isinstance(self.__dict__.get("annotations"), dict) and (
            "episode" in self.__dict__["annotations"]
        ):
            from ..capture._episode_ledger import validate_loaded_episode_annotations

            validate_loaded_episode_annotations(self)
        _state.register_log(self)

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
        )
        current_state = dict(state_items(self))
        preserved_trace_user_annotations = self._copy_user_annotations(
            current_state.get("annotations")
        )
        preserved_state = {
            field_name: current_state.get(field_name) for field_name in preserved_fields
        }
        # DERIVED annotation blobs (feature maps / RDM / MDS / scree, the
        # ``_DERIVED_ANNOTATION_BLOB_PREFIXES`` namespaces) are NOT preserved:
        # they were computed from the replaced run's activations and would
        # render stale data over the new run's stimuli. USER
        # ``annotate(data=...)`` blobs (bare ``layer:``/``op:`` keys) are the
        # user's own attachments and survive like the user annotations above.
        preserved_state["_annotation_blobs"] = self._user_annotation_blobs()
        replacement_state = dict(state_items(new_log))
        replacement_state.update(preserved_state)
        replacement_state["annotations"] = self._merge_user_annotations(
            self._copy_rerun_value(getattr(new_log, "annotations", {})),
            preserved_trace_user_annotations,
        )
        state_restore(self, replacement_state)
        self.__dict__.pop("_validation_replay_status", None)
        _invalidate_trace_op_layer_accessor_caches(self)
        _invalidate_trace_module_call_accessor_cache(self)
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

        # Pair the two op sequences POSITIONALLY, never through a label -> op map.
        # Neither `_layer_label_raw` nor `layer_label` is pass-qualified, so every
        # pass of a multi-pass (recurrent) layer shares both keys: a dict keyed by
        # either collapses an N-pass layer to its last pass and then refreshes all
        # N existing passes from that one op, silently overwriting the earlier
        # passes' activations and pass labels. The label-sequence equality checked
        # just above is exactly the precondition that makes index i of one list the
        # same op as index i of the other.
        for layer, new_layer in zip(self.layer_list, new_log.layer_list):
            self._refresh_rerun_op_from(layer, new_layer)
        self._refresh_rerun_layer_logs_from(new_log)
        self._refresh_rerun_trace_fields_from(new_log)
        self.__dict__.pop("_validation_replay_status", None)
        _invalidate_trace_op_layer_accessor_caches(self)
        _invalidate_trace_module_call_accessor_cache(self)
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
        # DERIVED annotation blobs (feature-map/RDM/MDS/scree namespaces) were
        # computed from the replaced activations and are invalidated; USER
        # ``annotate(data=...)`` blobs survive the same-shape refresh.
        self._annotation_blobs = self._user_annotation_blobs()
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

    def discharge_against(self, real_trace: "Trace") -> Any:
        """Discharge this structure-only trace's hypotheses against a real run.

        DOCUMENTED-UNSTABLE surface (L7a; no deprecation shim owed on
        rename). Only defined on a structure-only capture; ``real_trace``
        must be an ordinary settled COMPLETE capture of the same graph.
        Returns a frozen ``StructureDischarge`` record (per-claim table +
        overall verdict); neither trace is mutated — the verdict registers in
        a weak-keyed side table consulted by the structure-only capability
        chokepoint (a REFUTED discharge flips hypothesis consumers to typed
        refusals). See ``torchlens.capture.structure_only.discharge_against``
        for the full join/precondition contract.
        """

        from ..capture.structure_only import discharge_against as _discharge

        return _discharge(self, real_trace)


Trace.FIELD_FORK_POLICY = fork_policy_from_policy(Trace.FIELD_POLICY)  # type: ignore[attr-defined]
Trace.DEFAULT_FILL_STATE = default_fill_state_from_policy(Trace.FIELD_POLICY)  # type: ignore[attr-defined]

# The tlspec v8 coordinated bump retired this class's S3 pre-release
# registrations: every formerly gated Trace field above now declares its
# persisting policy directly. The registrar mechanism itself stays for
# future sprints (torchlens/_io/prerelease.py).
