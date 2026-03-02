"""Structured per-module metadata: ModulePassLog, ModuleLog, ModuleAccessor."""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .param_log import ParamAccessor


class ModulePassLog:
    """Per-(module, pass_num) data. Lightweight container for one invocation of a module."""

    def __init__(
        self,
        module_address: str,
        pass_num: int,
        pass_label: str,
        layers: List[str],
        input_layers: List[str],
        output_layers: List[str],
        forward_args: Optional[tuple] = None,
        forward_kwargs: Optional[dict] = None,
        call_parent: Optional[str] = None,
        call_children: Optional[List[str]] = None,
    ):
        self.module_address = module_address
        self.pass_num = pass_num
        self.pass_label = pass_label
        self.layers = layers
        self.num_layers = len(layers)
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []

    def __repr__(self) -> str:
        lines = [
            f"ModulePassLog: {self.pass_label}",
            f"  layers: {self.num_layers}",
        ]
        if self.input_layers:
            lines.append(f"  input_layers: {self.input_layers}")
        if self.output_layers:
            lines.append(f"  output_layers: {self.output_layers}")
        if self.call_children:
            lines.append(f"  call_children: {self.call_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.num_layers


class ModuleLog:
    """Per-module-object metadata. The primary user-facing class for module inspection."""

    def __init__(
        self,
        # Identity
        address: str,
        all_addresses: Optional[List[str]] = None,
        name: str = "",
        module_class_name: str = "",
        # Source info
        source_file: Optional[str] = None,
        source_line: Optional[int] = None,
        class_docstring: Optional[str] = None,
        init_signature: Optional[str] = None,
        init_docstring: Optional[str] = None,
        forward_signature: Optional[str] = None,
        forward_docstring: Optional[str] = None,
        # Hierarchy — address-based (static)
        address_parent: Optional[str] = None,
        address_children: Optional[List[str]] = None,
        address_depth: int = 0,
        # Hierarchy — call-based (dynamic)
        call_parent: Optional[str] = None,
        call_children: Optional[List[str]] = None,
        nesting_depth: int = 0,
        # Pass info
        num_passes: int = 1,
        passes: Optional[Dict[int, "ModulePassLog"]] = None,
        pass_labels: Optional[List[str]] = None,
        # Layers (aggregate)
        all_layers: Optional[List[str]] = None,
        # Parameters
        params: Optional["ParamAccessor"] = None,
        num_params: int = 0,
        num_params_trainable: int = 0,
        num_params_frozen: int = 0,
        params_fsize: int = 0,
        params_fsize_nice: str = "",
        requires_grad: bool = False,
        # Buffers
        buffer_layers: Optional[List[str]] = None,
        # Module state
        training_mode: bool = True,
        has_forward_hooks: bool = False,
        has_backward_hooks: bool = False,
        extra_attributes: Optional[Dict] = None,
        methods: Optional[List[str]] = None,
        # Back-reference
        _source_model_log=None,
    ):
        self.address = address
        self.all_addresses = all_addresses if all_addresses is not None else [address]
        self.name = name
        self.module_class_name = module_class_name

        self.source_file = source_file
        self.source_line = source_line
        self.class_docstring = class_docstring
        self.init_signature = init_signature
        self.init_docstring = init_docstring
        self.forward_signature = forward_signature
        self.forward_docstring = forward_docstring

        self.address_parent = address_parent
        self.address_children = address_children if address_children is not None else []
        self.address_depth = address_depth

        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []
        self.nesting_depth = nesting_depth

        self.num_passes = num_passes
        self.passes = passes if passes is not None else {}
        self.pass_labels = pass_labels if pass_labels is not None else []

        self.all_layers = all_layers if all_layers is not None else []
        self.num_layers = len(self.all_layers)

        from .param_log import ParamAccessor

        self.params = params if params is not None else ParamAccessor({})
        self.num_params = num_params
        self.num_params_trainable = num_params_trainable
        self.num_params_frozen = num_params_frozen
        self.params_fsize = params_fsize
        self.params_fsize_nice = params_fsize_nice
        self.requires_grad = requires_grad

        self.buffer_layers = buffer_layers if buffer_layers is not None else []

        self.training_mode = training_mode
        self.has_forward_hooks = has_forward_hooks
        self.has_backward_hooks = has_backward_hooks
        self.extra_attributes = extra_attributes if extra_attributes is not None else {}
        self.methods = methods if methods is not None else []

        self._source_model_log = _source_model_log

    # --- Per-call delegating properties ---

    def _single_pass_or_error(self, field_name: str):
        if self.num_passes > 1:
            raise AttributeError(
                f"Module '{self.address}' has {self.num_passes} passes. "
                f"Access '{field_name}' on a specific pass: "
                f"module.passes[1].{field_name}, module.passes[2].{field_name}, etc."
            )
        if 1 not in self.passes:
            return None
        return getattr(self.passes[1], field_name)

    @property
    def layers(self) -> List[str]:
        result = self._single_pass_or_error("layers")
        return result if result is not None else []

    @property
    def input_layers(self) -> List[str]:
        result = self._single_pass_or_error("input_layers")
        return result if result is not None else []

    @property
    def output_layers(self) -> List[str]:
        result = self._single_pass_or_error("output_layers")
        return result if result is not None else []

    @property
    def forward_args(self) -> Optional[tuple]:
        return self._single_pass_or_error("forward_args")

    @property
    def forward_kwargs(self) -> Optional[dict]:
        return self._single_pass_or_error("forward_kwargs")

    def __repr__(self) -> str:
        lines = [
            f"ModuleLog: {self.address} ({self.module_class_name})",
            f"  nesting_depth: {self.nesting_depth}, address_depth: {self.address_depth}",
            f"  num_params: {self.num_params}",
            f"  num_layers: {self.num_layers}",
            f"  num_passes: {self.num_passes}",
        ]
        if self.address_children:
            lines.append(f"  children: {self.address_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.num_layers

    def __getitem__(self, ix):
        if self._source_model_log is None:
            raise RuntimeError("No source ModelLog reference; cannot index into layers.")
        return self._source_model_log[self.all_layers[ix]]

    def __iter__(self):
        if self._source_model_log is None:
            return iter(self.all_layers)
        return iter(self._source_model_log[label] for label in self.all_layers)

    def to_pandas(self) -> "pd.DataFrame":
        if self._source_model_log is None:
            raise RuntimeError("No source ModelLog reference; cannot build DataFrame.")
        rows = []
        for label in self.all_layers:
            entry = self._source_model_log[label]
            rows.append(
                {
                    "layer_label": entry.layer_label,
                    "layer_type": entry.layer_type,
                    "tensor_shape": entry.tensor_shape,
                    "tensor_dtype": entry.tensor_dtype,
                    "pass_num": entry.pass_num,
                    "func_applied_name": entry.func_applied_name,
                }
            )
        return pd.DataFrame(rows)


class ModuleAccessor:
    """Dict-like accessor for ModuleLog objects. Supports indexing by address, index, or pass notation."""

    def __init__(
        self,
        module_dict: Dict[str, "ModuleLog"],
        module_list: Optional[List["ModuleLog"]] = None,
        pass_dict: Optional[Dict[str, "ModulePassLog"]] = None,
    ):
        self._dict = module_dict
        self._list = module_list if module_list is not None else list(module_dict.values())
        self._pass_dict = pass_dict if pass_dict is not None else {}

    def __getitem__(self, key: Union[int, str]) -> Union["ModuleLog", "ModulePassLog"]:
        if isinstance(key, int):
            return self._list[key]
        if key == "":
            key = "self"
        if key in self._dict:
            return self._dict[key]
        if key in self._pass_dict:
            return self._pass_dict[key]
        raise KeyError(
            f"Module '{key}' not found. Valid addresses: {list(self._dict.keys())[:10]}..."
        )

    def __contains__(self, key) -> bool:
        if key == "":
            key = "self"
        return key in self._dict or key in self._pass_dict

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._list)

    def __repr__(self) -> str:
        if len(self) == 0:
            return "ModuleAccessor({})"
        items = []
        for ml in self._list:
            items.append(
                f"  '{ml.address}': {ml.module_class_name} "
                f"(depth={ml.nesting_depth}, params={ml.num_params}, "
                f"layers={ml.num_layers}, passes={ml.num_passes})"
            )
        inner = "\n".join(items)
        return f"ModuleAccessor({len(self)} modules):\n{inner}"

    def to_pandas(self) -> "pd.DataFrame":
        rows = []
        for ml in self._list:
            rows.append(
                {
                    "address": ml.address,
                    "module_class_name": ml.module_class_name,
                    "nesting_depth": ml.nesting_depth,
                    "address_depth": ml.address_depth,
                    "num_params": ml.num_params,
                    "num_layers": ml.num_layers,
                    "num_passes": ml.num_passes,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        if len(self) == 0:
            return "No modules."
        lines = [
            f"{'Address':<40} {'Class':<20} {'Depth':>5} {'Params':>10} {'Layers':>7} {'Passes':>7}"
        ]
        lines.append("-" * 92)
        for ml in self._list:
            lines.append(
                f"{ml.address:<40} {ml.module_class_name:<20} "
                f"{ml.nesting_depth:>5} {ml.num_params:>10} "
                f"{ml.num_layers:>7} {ml.num_passes:>7}"
            )
        return "\n".join(lines)
