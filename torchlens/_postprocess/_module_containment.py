"""Op containment facts: tally each op's module calls into the module build data.

Every op records the module calls it ran inside (``Op.modules``, outermost
first). This pass tallies those calls into the module build data (call counts,
layers per module, top-level calls, call children) and then normalizes the
op's own containment facts so they read as under a plain trace: the root
module's own calls (``self:N``, pushed only on a bound-method root) stay in
the tally but leave the op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace


def _log_module_hierarchy_info_for_layer(
    self: Trace, layer_entry: Op, _shadow_sets: dict[str, Any]
) -> None:
    """Tally one op's module calls into the module build data, then drop root frames.

    Records per-module and per-call op counts, module-to-layer mappings, top-level
    module calls and parent-child call links, addresses and call labels. The shadow
    sets give O(1) dedup for the insertion-ordered lists.
    """
    _module_labels_seen = _shadow_sets["module_layers"]
    _module_call_labels_seen = _shadow_sets["module_pass_layers"]
    _top_level_module_ops_seen = _shadow_sets["top_level_module_ops"]
    _module_pass_children_seen = _shadow_sets["module_pass_children"]
    _addresses_seen = _shadow_sets["addresses"]
    _module_ops_seen = _shadow_sets["module_ops"]
    mbd = self._module_capture_ws.module_build_data

    parent_call_label = None
    layer_label = layer_entry.layer_label
    op_label = layer_entry.label
    for module_index, raw_module_call_label in enumerate(layer_entry.modules):
        if isinstance(raw_module_call_label, str):
            module_name, module_pass = raw_module_call_label.rsplit(":", 1)
            module_pass = int(module_pass)  # type: ignore[assignment]
        else:
            module_name, module_pass = raw_module_call_label
        module_pass_nice_label = f"{module_name}:{module_pass}"
        mbd["module_num_tensors"][module_name] += 1
        mbd["module_call_index_tensors"][module_pass_nice_label] += 1
        if layer_label not in _module_labels_seen[module_name]:
            _module_labels_seen[module_name].add(layer_label)
            mbd["module_layers"][module_name].append(layer_label)
        if op_label not in _module_call_labels_seen[module_pass_nice_label]:
            _module_call_labels_seen[module_pass_nice_label].add(op_label)
            mbd["module_pass_layers"][module_pass_nice_label].append(op_label)
        if (module_index == 0) and (module_pass_nice_label not in _top_level_module_ops_seen):
            _top_level_module_ops_seen.add(module_pass_nice_label)
            mbd["top_level_module_ops"].append(module_pass_nice_label)
        else:
            if (parent_call_label is not None) and (
                module_pass_nice_label not in _module_pass_children_seen[parent_call_label]
            ):
                _module_pass_children_seen[parent_call_label].add(module_pass_nice_label)
                mbd["module_pass_children"][parent_call_label].append(module_pass_nice_label)
        parent_call_label = module_pass_nice_label
        if mbd["module_num_calls"][module_name] < module_pass:
            mbd["module_num_calls"][module_name] = module_pass
        if module_name not in _addresses_seen:
            _addresses_seen.add(module_name)
            mbd["addresses"].append(module_name)
        if module_pass_nice_label not in _module_ops_seen:
            _module_ops_seen.add(module_pass_nice_label)
            mbd["module_ops"].append(module_pass_nice_label)
    _drop_root_call_frames(layer_entry)


def _module_call_address(module_call: Any) -> str:
    """Return the address of a raw module call, ``(address, pass)`` or ``"address:pass"``."""
    if isinstance(module_call, str):
        return module_call.rsplit(":", 1)[0]
    return str(module_call[0])


def _drop_root_call_frames(layer_entry: Op) -> None:
    """Remove the root module's own calls from an op's containment facts.

    A plain root is never pushed on the module stack, so ``Op.modules`` and
    ``Op.module_call_stack`` list submodule calls only. A bound-method root's
    owner IS pushed (its forward runs inside the bound method's call, at the
    root address ``self``), and its calls were just tallied into the module
    build data (``self:1`` .. ``self:N`` keep their ops, children and call
    counts there). Dropping them from the op afterwards makes the op read
    exactly as under a plain trace of the owner. ``output_of_module_calls``
    and ``input_to_module_calls`` keep the root calls: they are the boundary
    facts the root's ModuleCalls are built from.
    """
    if not any(_module_call_address(call) == "self" for call in layer_entry.modules):
        return
    layer_entry.modules = [
        call for call in layer_entry.modules if _module_call_address(call) != "self"
    ]
    layer_entry.module_call_stack = [
        call for call in layer_entry.module_call_stack if _module_call_address(call) != "self"
    ]
    if layer_entry.module is not None and _module_call_address(layer_entry.module) == "self":
        layer_entry.module = layer_entry.modules[-1] if layer_entry.modules else None
