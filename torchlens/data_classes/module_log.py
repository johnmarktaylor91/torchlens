"""Structured per-module metadata: ModulePassLog, ModuleLog, ModuleAccessor.

Three-level hierarchy:

* **ModulePassLog** -- one invocation of one module (lightweight container).
  Stores ``layers`` as pass-qualified labels (e.g. ``"conv2d_1_1:1"``),
  pointing to individual LayerPassLog entries.

* **ModuleLog** -- aggregate metadata for one ``nn.Module`` across all its
  invocations.  Stores ``all_layers`` as no-pass labels (e.g.
  ``"conv2d_1_1"``), pointing to aggregate LayerLog entries.
  For single-pass modules, per-pass fields are delegated to ``passes[1]``
  via ``_single_pass_or_error()``.

* **ModuleAccessor** -- dict-like accessor returned by ``model_log.modules``.
  Supports lookup by module address, alias (shared modules), pass label,
  or ordinal index.

ModuleLog vs ModulePassLog label convention:
  - ModuleLog.all_layers stores **no-pass** labels -> LayerLog
  - ModulePassLog.layers stores **pass-qualified** labels -> LayerPassLog
This matches each accessor's natural granularity.
"""

import weakref
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from ..utils.display import human_readable_size

import pandas as pd

if TYPE_CHECKING:
    from .param_log import ParamAccessor


class ModulePassLog:
    """Per-(module, pass_num) data for one invocation of a module.

    Lightweight container holding the list of layers computed during
    this particular invocation, the captured forward arguments, and
    the call-graph edges (parent/children in the module invocation tree).

    ``layers`` stores **pass-qualified** labels (e.g. ``"conv2d_1_1:1"``)
    that resolve to individual LayerPassLog entries.
    """

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
        all_module_addresses: Optional[List[str]] = None,
    ):
        self.module_address = module_address
        self.pass_num = pass_num
        self.pass_label = pass_label  # e.g. "features.0:1"
        self.layers = layers  # pass-qualified layer labels
        self.num_layers = len(layers)
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.call_parent = call_parent
        self.call_children = call_children if call_children is not None else []
        self.all_module_addresses = (
            all_module_addresses if all_module_addresses is not None else [module_address]
        )
        self.is_shared_module = len(self.all_module_addresses) > 1

    def __repr__(self) -> str:
        """Show pass label, layer count, and children."""
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
        """Return the number of layers in this pass."""
        return self.num_layers


class ModuleLog:
    """Aggregate metadata for one nn.Module across all its invocations.

    The primary user-facing class for module inspection.  Provides both
    static metadata (source file, class name, hierarchy) and dynamic
    data (layers computed, parameter usage, pass-level detail).

    ``all_layers`` stores **no-pass** labels (e.g. ``"conv2d_1_1"``),
    pointing to aggregate LayerLog entries.  For per-pass detail, access
    ``self.passes[pass_num].layers`` which stores pass-qualified labels.

    For single-pass modules, per-pass fields (``layers``, ``input_layers``,
    ``output_layers``, ``forward_args``, ``forward_kwargs``) are accessible
    directly via ``_single_pass_or_error()`` delegation to ``passes[1]``.
    """

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
        self.is_shared = len(self.all_addresses) > 1
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

        # all_layers stores NO-PASS labels (e.g. "conv2d_1_1") -> LayerLog.
        # Contrast with ModulePassLog.layers which stores pass-qualified labels.
        self.all_layers = all_layers if all_layers is not None else []
        self.num_layers = len(self.all_layers)

        from .param_log import ParamAccessor

        self.params = params if params is not None else ParamAccessor({})
        self.num_params = num_params
        self.num_params_trainable = num_params_trainable
        self.num_params_frozen = num_params_frozen
        self.params_fsize = params_fsize
        self.requires_grad = requires_grad

        self.buffer_layers = buffer_layers if buffer_layers is not None else []
        self._buffer_accessor = None  # populated by _build_module_logs

        self.training_mode = training_mode
        self.has_forward_hooks = has_forward_hooks
        self.has_backward_hooks = has_backward_hooks
        self.extra_attributes = extra_attributes if extra_attributes is not None else {}
        self.methods = methods if methods is not None else []

        # Store as weakref to break circular reference (ModelLog -> _module_logs -> ModuleLog -> ModelLog).
        self._source_model_log_ref = (
            weakref.ref(_source_model_log) if _source_model_log is not None else None
        )

    @property
    def params_fsize_nice(self) -> str:
        return human_readable_size(self.params_fsize)

    @property
    def _source_model_log(self):
        """Back-reference to the owning ModelLog (stored as weakref)."""
        ref = self.__dict__.get("_source_model_log_ref")
        if ref is None:
            return None
        return ref()

    @_source_model_log.setter
    def _source_model_log(self, value):
        self._source_model_log_ref = weakref.ref(value) if value is not None else None

    # --- Per-call delegating properties ---
    # Mirror the LayerLog delegation pattern: single-pass modules transparently
    # expose per-pass fields; multi-pass modules raise with guidance.

    def _single_pass_or_error(self, field_name: str):
        """Return a field from the single pass, or raise if the module has multiple passes.

        For modules invoked once, transparently delegates to passes[1].
        For multi-pass modules, raises AttributeError directing the user to
        access the field on a specific pass.
        """
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

    @property
    def buffers(self):
        """Scoped BufferAccessor for buffers belonging to this module."""
        if self._buffer_accessor is not None:
            return self._buffer_accessor
        # Build on first access from the source ModelLog
        from .buffer_log import BufferAccessor

        if self._source_model_log is None or self._source_model_log._buffer_accessor is None:
            self._buffer_accessor = BufferAccessor({})
            return self._buffer_accessor
        parent_accessor = self._source_model_log._buffer_accessor
        scoped = {
            addr: bl
            for addr, bl in parent_accessor._dict.items()
            if bl.module_address == self.address
        }
        self._buffer_accessor = BufferAccessor(scoped, source_model_log=self._source_model_log)
        return self._buffer_accessor

    def __repr__(self) -> str:
        """Show address, class, depth, param count, layer count, and pass count."""
        lines = [
            f"ModuleLog: {self.address} ({self.module_class_name})",
            f"  nesting_depth: {self.nesting_depth}, address_depth: {self.address_depth}",
            f"  num_params: {self.num_params}",
            f"  num_layers: {self.num_layers}",
            f"  num_passes: {self.num_passes}",
        ]
        if self.is_shared:
            lines.append(f"  aliases: {self.all_addresses}")
        if self.address_children:
            lines.append(f"  children: {self.address_children}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the total number of layers across all passes of this module."""
        return self.num_layers

    def __getitem__(self, ix):
        """Return the LayerPassLog at position ix within this module's layer list.

        Supports int indexing and string label lookup (#120).
        """
        if self._source_model_log is None:
            raise RuntimeError("No source ModelLog reference; cannot index into layers.")
        if isinstance(ix, str):
            # String label lookup within this module's layers
            if ix in self.all_layers:
                return self._source_model_log[ix]
            # Try substring match within module layers
            matches = [lbl for lbl in self.all_layers if ix in lbl]
            if len(matches) == 1:
                return self._source_model_log[matches[0]]
            elif len(matches) > 1:
                raise ValueError(
                    f"Ambiguous lookup: '{ix}' matches {len(matches)} layers in module "
                    f"'{self.address}': {', '.join(matches[:5])}"
                )
            raise KeyError(f"'{ix}' not found in module '{self.address}' layers")
        return self._source_model_log[self.all_layers[ix]]

    def __iter__(self):
        """Iterate over LayerPassLog entries for all layers in this module."""
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
    """Dict-like accessor for ModuleLog objects.

    Supports indexing by:
    * **module address** (str) -- e.g. ``"features.0"``, ``"self"``.
    * **alias** (str) -- for shared modules (same nn.Module registered at
      multiple addresses), any alias resolves to the same ModuleLog.
    * **pass label** (str) -- e.g. ``"features.0:2"`` returns a ModulePassLog.
    * **ordinal index** (int) -- position in address order.

    Available as ``model_log.modules``.
    """

    def __init__(
        self,
        module_dict: Dict[str, "ModuleLog"],
        module_list: Optional[List["ModuleLog"]] = None,
        pass_dict: Optional[Dict[str, "ModulePassLog"]] = None,
    ):
        self._dict = module_dict  # primary address -> ModuleLog
        self._list = module_list if module_list is not None else list(module_dict.values())
        self._pass_dict = pass_dict if pass_dict is not None else {}  # pass label -> ModulePassLog
        # Alias map: for shared modules (same nn.Module at multiple addresses),
        # every non-primary address also resolves to the same ModuleLog.
        self._alias_dict: Dict[str, "ModuleLog"] = {}
        for ml in self._dict.values():
            for alias in ml.all_addresses:
                if alias not in self._dict:
                    self._alias_dict[alias] = ml

    def __getitem__(self, key: Union[int, str]) -> Union["ModuleLog", "ModulePassLog"]:
        """Return a ModuleLog by address string or ordinal index, or a ModulePassLog by pass label.

        Shared modules (same nn.Module registered under multiple addresses) can
        be looked up by any of their aliases.
        """
        if isinstance(key, int):
            return self._list[key]
        if key == "":
            key = "self"
        if key in self._dict:
            return self._dict[key]
        if key in self._alias_dict:
            return self._alias_dict[key]
        if key in self._pass_dict:
            return self._pass_dict[key]
        raise KeyError(
            f"Module '{key}' not found. Valid addresses: {list(self._dict.keys())[:10]}..."
        )

    def __contains__(self, key) -> bool:
        """Return True if key is a known module address, alias, or pass label."""
        if key == "":
            key = "self"
        return key in self._dict or key in self._alias_dict or key in self._pass_dict

    def __len__(self) -> int:
        """Return the number of modules in this accessor."""
        return len(self._dict)

    def __iter__(self):
        """Iterate over ModuleLog objects in address order."""
        return iter(self._list)

    def __repr__(self) -> str:
        """Show a table of all modules with class, depth, param, layer, and pass counts."""
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
