"""Module-build-data readers for the root module's calls.

A plain module root's forward runs undecorated, so its one call ``self:1`` is
synthesized to span the whole capture. A bound-method root's owner is
instrumented at the root address, so its calls ``self:1`` .. ``self:N`` are
tallied like a submodule's; these helpers read that tally.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _recorded_root_calls(mbd: dict[str, Any]) -> int:
    """Number of recorded calls of the root module itself, ``0`` when none.

    A plain root's forward runs undecorated (the capture calls it once, so the
    root call is synthesized). A bound-method root's owner is instrumented at
    the root address, so its calls are tallied like a submodule's.
    """
    if "self" not in mbd.get("addresses", []):
        return 0
    return int(mbd["module_num_calls"].get("self", 0))


def _root_call_build_inputs(
    self: Trace, mbd: dict[str, Any]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Child-to-parent call map and input layers per call, as ``_build_module_logs`` derives them."""

    child_to_parent = {
        child: parent
        for parent, children in mbd["module_pass_children"].items()
        for child in children
    }
    input_layers_by_call: dict[str, list[str]] = defaultdict(list)
    for op in self._layer_list:
        for call_label in op.input_to_module_calls:
            input_layers_by_call[call_label].append(op.layer_label)
    return child_to_parent, input_layers_by_call


def _root_call_children(mbd: dict[str, Any]) -> list[str]:
    """Resolve the root ``self:1`` ModuleCall's direct call children.

    Two capture shapes feed this:

    * Function-root traces (``self`` is synthetic and never appears in an op's
      module stack): the direct children are the outermost real module calls,
      recorded in ``top_level_module_ops`` while ``module_pass_children['self:1']``
      stays empty.
    * Object-module / explicit-self traces (``self`` is itself a traced module,
      so ops carry ``['self:1', child:1, ...]`` stacks): the outermost call IS
      ``self:1`` itself, so ``top_level_module_ops`` collapses to ``['self:1']``
      and the real direct children land in ``module_pass_children['self:1']``.

    Unioning both sources (preserving order, dropping the self-reference) yields
    the correct children in both shapes. Previously only ``top_level_module_ops``
    was consulted, so the explicit-self shape silently dropped every child of
    ``self:1`` -- an asymmetric call-tree the module-hierarchy invariant rightly
    flags (parent lists no children while children point back to ``self:1``).
    """

    children: list[str] = []
    for label in mbd.get("top_level_module_ops", []):
        if label.rsplit(":", 1)[0] != "self" and label not in children:
            children.append(label)
    for label in mbd.get("module_pass_children", {}).get("self:1", []):
        if label != "self:1" and label not in children:
            children.append(label)
    return children
