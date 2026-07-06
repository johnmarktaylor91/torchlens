"""Mutable trace-build state for capture and postprocessing."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .container import ContainerSpec
from .container_registry import ContainerRegistry
from .events import (
    ConditionalEvent,
    ModuleEnterEvent,
    ModuleEvent,
    ModuleExitEvent,
    ModulePrepEvent,
)


@dataclass(slots=True)
class TraceBuildState:
    """Transient capture/postprocess state discarded before returning a Trace."""

    raw_layer_dict: dict[str, object] = field(default_factory=OrderedDict)
    raw_layer_labels_list: list[str] = field(default_factory=list)
    mod_entered: dict[int, list[str]] = field(default_factory=dict)
    mod_exited: dict[int, list[str]] = field(default_factory=dict)
    mod_call_index: dict[int, int] = field(default_factory=dict)
    mod_call_labels: dict[int, list[tuple[str, int]]] = field(default_factory=dict)
    exhaustive_module_stack: list[object] = field(default_factory=list)
    module_build_data: dict[str, Any] = field(default_factory=dict)
    module_metadata: dict[Any, Any] = field(default_factory=dict)
    module_forward_args: dict[Any, Any] = field(default_factory=dict)
    module_containment_engine: str = "hook_stack"
    current_func_barcode: object | None = None
    grad_fn_strong_refs: list[Any] = field(default_factory=list)
    in_exhaustive_pass: bool = True
    layer_counter: int = 0
    raw_layer_type_counter: dict[str, int] = field(default_factory=lambda: defaultdict(lambda: 0))
    output_container_specs_by_raw_label: dict[str, ContainerSpec] = field(default_factory=dict)
    output_container_specs: tuple[ContainerSpec, ...] = ()
    container_registry: ContainerRegistry = field(default_factory=ContainerRegistry)
    input_tensor_addresses: list[int] = field(default_factory=list)
    module_events: list[ModuleEvent] = field(default_factory=list)
    module_prep_events: list[ModulePrepEvent] = field(default_factory=list)
    module_enter_events: list[ModuleEnterEvent] = field(default_factory=list)
    module_exit_events: list[ModuleExitEvent] = field(default_factory=list)
    conditional_events: list[ConditionalEvent] = field(default_factory=list)
