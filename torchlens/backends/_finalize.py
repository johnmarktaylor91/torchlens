"""Shared preview-backend trace finalization helpers."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeAlias, cast

from ..data_classes.layer import Layer
from ..data_classes.module import ModuleAccessor
from ..data_classes.trace import Trace, _init_module_hierarchy_data
from ..postprocess.finalization import _build_module_logs, _build_root_module_log
from ..quantities import Bytes
from .registry import BackendName

OpHook: TypeAlias = Callable[[Any, Trace, set[str]], None]
OpEnrichmentHook: TypeAlias = Callable[[Any], None]
LayerEnrichmentHook: TypeAlias = Callable[[Any, Any], None]
ModuleCallNormalizer: TypeAlias = Callable[[Any], tuple[tuple[str, int], ...]]
MetadataTopLevelPredicate: TypeAlias = Callable[
    [str, dict[str, Any], dict[str, dict[str, Any]]], bool
]
OpTopLevelPredicate: TypeAlias = Callable[[str], bool]
TrainingModeResolver: TypeAlias = Callable[[dict[str, Any]], bool]


def finalize_single_pass_trace(
    trace: Trace,
    *,
    backend_name: str,
    module_tree: Any | None,
    attach_function_root_module: Callable[[Trace], None],
    attach_object_module_logs: Callable[[Trace, Any], None],
    attach_op_params: OpHook | None = None,
    enrich_op: OpEnrichmentHook | None = None,
    enrich_layer: LayerEnrichmentHook | None = None,
    update_param_usage: bool = True,
    update_param_totals_from_layers: bool = False,
    count_layers_with_attached_params: bool = False,
    finish_before_module_logs: bool = True,
) -> None:
    """Finalize raw single-pass op logs into public trace accessors.

    Parameters
    ----------
    trace:
        Trace whose ``_raw_layer_dict`` contains backend-created op logs.
    backend_name:
        Canonical backend name written to ``trace.backend``.
    module_tree:
        Optional object-module discovery tree for object-module traces.
    attach_function_root_module:
        Callback used when no object module tree is active.
    attach_object_module_logs:
        Callback used when object module attribution is active.
    attach_op_params:
        Optional hook that attaches backend parameter logs to one op.
    enrich_op:
        Optional hook for backend-specific op metadata before layer creation.
    enrich_layer:
        Optional hook for backend-specific layer metadata copied from an op.
    update_param_usage:
        Whether attached op params should receive ``used_by_*`` cross-links.
    update_param_totals_from_layers:
        Whether trace parameter counters should be recomputed from finalized layers.
    count_layers_with_attached_params:
        Whether to compute ``trace.num_layers_with_params`` from attached params.
    finish_before_module_logs:
        Whether to set ``_tracing_finished`` before module logs are attached.

    Returns
    -------
    None
        The trace is updated in place.
    """

    seen_param_barcodes: set[str] = set()
    layers_with_params_seen: set[str] = set()
    for raw_index, (label, op_log) in enumerate(trace._raw_layer_dict.items()):
        _finalize_single_op(trace, op_log, label, raw_index)
        if enrich_op is not None:
            enrich_op(op_log)
        if attach_op_params is not None:
            attach_op_params(op_log, trace, seen_param_barcodes)
            if not isinstance(op_log.param_memory, Bytes):
                op_log.param_memory = Bytes(int(op_log.param_memory))
        if getattr(op_log, "_param_logs", []):
            layers_with_params_seen.add(op_log.layer_label)
        if update_param_usage:
            _attach_param_usage(trace, op_log)
        layer_log = Layer(op_log)
        layer_log.ops[1] = op_log
        layer_log.call_labels.append(op_log.label)
        if enrich_layer is not None:
            enrich_layer(layer_log, op_log)
        trace.layer_logs[label] = layer_log

    trace.num_ops = sum(
        1
        for op_log in trace.layer_list
        if not (op_log.is_input or op_log.is_output or op_log.is_buffer)
    )
    if update_param_totals_from_layers:
        _update_param_totals_from_layers(trace)
    if count_layers_with_attached_params:
        trace.num_layers_with_params = len(layers_with_params_seen)
    trace._layers_logged = True
    trace._layers_saved = True
    trace.has_backward_pass = False
    trace.capture_end_time = time.time()
    trace.backend = cast(BackendName, backend_name)
    if finish_before_module_logs:
        trace._tracing_finished = True
        _set_per_op_tracing_finished(trace)
    if module_tree is None:
        trace.module_identity_mode = "function_root"
        attach_function_root_module(trace)
    else:
        trace.module_identity_mode = "object_module"
        attach_object_module_logs(trace, module_tree)
    if not finish_before_module_logs:
        trace._tracing_finished = True
        _set_per_op_tracing_finished(trace)


def _set_per_op_tracing_finished(trace: Trace) -> None:
    """Flip ``_tracing_finished`` on every retained op, mirroring the torch path.

    Parameters
    ----------
    trace:
        Trace whose ``_tracing_finished`` flag was just set at the trace level.

    Returns
    -------
    None
        Every op reachable via ``trace.layer_dict_main_keys`` is mutated in place.

    Notes
    -----
    Mirrors ``torchlens.postprocess.finalization._set_tracing_finished``, which
    also flips this flag on every retained ``OpLog`` (not just the trace). Without
    this, preview-backend ops stay stuck on the "mid-capture" ``__str__``/repr
    branch, which assumes torch-only attributes (e.g. ``grad_fn``) and crashes
    on non-torch tensors.
    """

    for layer_label in trace.layer_dict_main_keys:
        op_log = trace.layer_dict_main_keys[layer_label]
        op_log._tracing_finished = True


def attach_function_root_module(trace: Trace) -> None:
    """Attach the shared ``self`` module log for function-root traces.

    Parameters
    ----------
    trace:
        Trace receiving root-module metadata.

    Returns
    -------
    None
        ``trace._module_logs`` is populated with a single ``self`` module.
    """

    mbd = trace._module_build_data
    mbd["top_level_modules"] = ["self"]
    mbd["top_level_module_ops"] = ["self:1"]
    trace._module_metadata = {
        "self": {
            "cls": None,
            "class_name": trace.model_class_name,
            "class_qualname": trace.model_class_qualname,
            "all_addresses": ["self"],
            "training": False,
        }
    }
    root = _build_root_module_log(trace, {}, mbd)
    trace._module_logs = ModuleAccessor({"self": root})


def attach_object_module_logs(
    trace: Trace,
    tree: Any,
    *,
    normalize_module_calls: ModuleCallNormalizer,
    metadata_top_level: MetadataTopLevelPredicate,
    op_top_level: OpTopLevelPredicate,
    training_mode: TrainingModeResolver,
) -> None:
    """Build module logs for preview backends with object-module attribution.

    Parameters
    ----------
    trace:
        Trace receiving module build-data and public module logs.
    tree:
        Backend module tree exposing ``metadata``, ``forward_args_by_call``, and
        ``call_counts`` attributes.
    normalize_module_calls:
        Backend normalizer for raw per-op module-call records.
    metadata_top_level:
        Predicate for top-level modules discovered from module metadata.
    op_top_level:
        Predicate for top-level modules observed in finalized op call stacks.
    training_mode:
        Resolver for backend-specific module training metadata.

    Returns
    -------
    None
        ``trace.modules`` and transient module build-data are populated.
    """

    trace._module_build_data = _init_module_hierarchy_data()
    trace._module_forward_args = dict(tree.forward_args_by_call)
    trace._module_metadata = tree.metadata
    mbd = trace._module_build_data
    metadata_by_address = cast(dict[str, dict[str, Any]], tree.metadata)
    call_counts = cast(dict[str, int], tree.call_counts)
    for address, metadata in metadata_by_address.items():
        if address not in mbd["addresses"]:
            mbd["addresses"].append(address)
        mbd["module_types"][address] = str(metadata.get("class_name", ""))
        mbd["module_training_modes"][address] = training_mode(metadata)
        mbd["module_num_calls"][address] = max(1, call_counts.get(address, 1))
        for child_address in metadata.get("address_children", []):
            if child_address not in mbd["module_children"][address]:
                mbd["module_children"][address].append(child_address)
        if metadata_top_level(address, metadata, metadata_by_address):
            mbd["top_level_modules"].append(address)

    for param in trace.param_logs:
        owner = param.module_address
        mbd["module_nparams"][owner] += param.num_params
        if param.is_trainable:
            mbd["module_nparams_trainable"][owner] += param.num_params
        else:
            mbd["module_nparams_frozen"][owner] += param.num_params

    populate_object_module_build_data(
        trace,
        normalize_module_calls=normalize_module_calls,
        op_top_level=op_top_level,
    )
    _build_module_logs(trace)


def populate_object_module_build_data(
    trace: Trace,
    *,
    normalize_module_calls: ModuleCallNormalizer,
    op_top_level: OpTopLevelPredicate,
) -> None:
    """Populate module hierarchy side channels from attributed op logs.

    Parameters
    ----------
    trace:
        Trace whose finalized ops carry module-call records.
    normalize_module_calls:
        Backend normalizer for raw per-op module-call records.
    op_top_level:
        Predicate for top-level modules observed in op call stacks.

    Returns
    -------
    None
        ``trace._module_build_data`` is updated in place.
    """

    mbd = trace._module_build_data
    seen_layers: dict[str, set[str]] = defaultdict(set)
    seen_pass_layers: dict[str, set[str]] = defaultdict(set)
    seen_module_ops: set[str] = set()
    seen_top_level_ops: set[str] = set()
    seen_pass_children: dict[str, set[str]] = defaultdict(set)
    seen_addresses = set(mbd["addresses"])
    for op_log in trace.layer_list:
        normalized_calls = normalize_module_calls(op_log.modules)
        op_log.modules = [f"{address}:{call_index}" for address, call_index in normalized_calls]
        op_log.module = op_log.modules[-1] if op_log.modules else None
        parent_call_label: str | None = None
        for module_index, (address, call_index) in enumerate(normalized_calls):
            call_label = f"{address}:{call_index}"
            if mbd["module_num_calls"][address] < call_index:
                mbd["module_num_calls"][address] = call_index
            mbd["module_num_tensors"][address] += 1
            mbd["module_call_index_tensors"][call_label] += 1
            if op_log.layer_label not in seen_layers[address]:
                seen_layers[address].add(op_log.layer_label)
                mbd["module_layers"][address].append(op_log.layer_label)
            if op_log.label not in seen_pass_layers[call_label]:
                seen_pass_layers[call_label].add(op_log.label)
                mbd["module_pass_layers"][call_label].append(op_log.label)
            if address not in seen_addresses:
                seen_addresses.add(address)
                mbd["addresses"].append(address)
            if call_label not in seen_module_ops:
                seen_module_ops.add(call_label)
                mbd["module_ops"].append(call_label)
            if module_index == 0:
                if call_label not in seen_top_level_ops:
                    seen_top_level_ops.add(call_label)
                    mbd["top_level_module_ops"].append(call_label)
                if op_top_level(address) and address not in mbd["top_level_modules"]:
                    mbd["top_level_modules"].append(address)
            elif (
                parent_call_label is not None
                and call_label not in seen_pass_children[parent_call_label]
            ):
                seen_pass_children[parent_call_label].add(call_label)
                mbd["module_pass_children"][parent_call_label].append(call_label)
            parent_call_label = call_label


def _finalize_single_op(trace: Trace, op_log: Any, label: str, raw_index: int) -> None:
    """Attach standard single-pass labels and lookup keys to one op log.

    Parameters
    ----------
    trace:
        Trace receiving lookup indexes.
    op_log:
        Operation log being finalized.
    label:
        Raw backend label.
    raw_index:
        Zero-based raw op index.

    Returns
    -------
    None
        ``op_log`` and trace lookup dictionaries are mutated in place.
    """

    pass_label = f"{label}:1"
    op_log._label_raw = label
    op_log._layer_label_raw = label
    op_log.label = pass_label
    op_log.label_short = pass_label
    op_log.layer_label = label
    op_log.layer_label_short = label
    op_log.lookup_keys = [label, pass_label]
    op_log.pass_index = 1
    op_log.num_passes = 1
    trace.layer_list.append(op_log)
    trace.layer_dict_main_keys[label] = op_log
    trace.layer_dict_all_keys[label] = op_log
    trace.layer_dict_all_keys[pass_label] = op_log
    trace.op_labels.append(pass_label)
    trace.layer_labels.append(label)
    trace.layer_num_calls[label] = 1
    trace._lookup_keys_to_layer_num_dict[label] = raw_index
    trace._layer_num_to_lookup_keys_dict[raw_index].append(label)


def _attach_param_usage(trace: Trace, op_log: Any) -> None:
    """Update parameter usage cross-links for params attached to one op.

    Parameters
    ----------
    trace:
        Trace receiving ``layers_with_params`` entries.
    op_log:
        Finalized op log that may carry ``_param_logs``.

    Returns
    -------
    None
        Parameter logs are mutated in place.
    """

    for param in getattr(op_log, "_param_logs", []):
        if op_log.label not in param.used_by_ops:
            param.used_by_ops.append(op_log.label)
        if op_log.layer_label not in param.used_by_layers:
            param.used_by_layers.append(op_log.layer_label)
        if op_log.layer_label not in trace.layers_with_params[param.barcode]:
            trace.layers_with_params[param.barcode].append(op_log.layer_label)


def _update_param_totals_from_layers(trace: Trace) -> None:
    """Recompute trace parameter totals from finalized single-pass layer logs.

    Parameters
    ----------
    trace:
        Trace whose layer logs already carry per-layer parameter counters.

    Returns
    -------
    None
        Trace-level parameter counters are updated when parameters are present.
    """

    seen_layers: set[str] = set()
    num_param_tensors = 0
    num_params = 0
    num_params_trainable = 0
    for op_log in trace.layer_list:
        if op_log.layer_label in seen_layers:
            continue
        seen_layers.add(op_log.layer_label)
        num_param_tensors += op_log.num_param_tensors
        num_params += op_log.num_params
        num_params_trainable += op_log.num_params_trainable
    if trace.param_source != "none":
        trace.num_param_tensors = num_param_tensors
        trace.num_params = num_params
        trace.num_params_trainable = num_params_trainable
        trace.num_params_frozen = num_params - num_params_trainable
        trace.num_layers_with_params = len(
            {op.layer_label for op in trace.layer_list if op.uses_params}
        )
