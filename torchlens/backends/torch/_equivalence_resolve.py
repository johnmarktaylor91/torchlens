"""Equivalence class and call-site class as resolvers over recorded inputs.

Each class is split into what must be observed at the call and a resolver that turns
those observations into the class string:

- The parameter-free equivalence class: the observation is the argument record
  (:func:`~torchlens.backends.torch._arg_hash.observe_arg_inputs`), the output slot of
  a multi-output call and the innermost module frame; :func:`resolve_equivalence_class`
  is a pure function of them and may run after the call.
- The call-site class: the observation is the call-site key
  (:func:`observe_call_site_key`, read while the call's pending chain is current);
  the resolver is ``torchlens._utils._call_site.call_site_class``, which numbers
  keys by first appearance in the capture's index, so resolves must run in op order.

The parameterized equivalence class (``tensor_tracking._make_raw_param_group_barcode``)
is already a pure function of recorded values: the layer type, the parameter
barcodes and the output slot.
"""

from collections.abc import Sequence
from typing import Any

from ..._utils._call_site import call_site_scope
from ._arg_hash import resolve_arg_hash

__all__ = ("observe_call_site_key", "resolve_equivalence_class")


def resolve_equivalence_class(
    layer_type: str,
    arg_record: tuple[Any, ...],
    output_index: int | None,
    module: tuple[str, int] | None,
) -> str:
    """Return the parameter-free equivalence class from its recorded inputs.

    Args:
        layer_type: The normalized operation name (with ``_inplace`` for an in-place
            output).
        arg_record: The call's argument record from ``observe_arg_inputs``.
        output_index: The output slot of a multi-output call, else ``None``.
        module: The innermost ``(address, pass)`` module frame, else ``None``.

    Returns:
        The class string: layer type and argument hash, then the output slot and the
        module address when present.
    """

    equivalence_class = f"{layer_type}_{resolve_arg_hash(arg_record)}"
    if output_index is not None:
        equivalence_class += f"_outindex{output_index}"
    if module is not None:
        equivalence_class += f"_module{module[0]}"
    return equivalence_class


def observe_call_site_key(
    chain: Any,
    modules: Sequence[tuple[str, int]],
    stateful: frozenset[str] | None,
    func_name: str,
    slot: int | None,
) -> tuple[Any, ...] | None:
    """Return the call-site key ``(scope, chain, func_name, slot)`` of one output.

    ``chain`` is the chain the raw-graph workspace holds pending for this call (a
    chain left by another call is not this call's); the key is None when it is
    empty. The scope is the innermost stateful module on the op's module
    stack (``stateful`` is the session's frozen set). Every part is immutable once read.
    """

    if not chain:
        return None
    scope = call_site_scope((address for address, _pass in modules), stateful)
    return (scope, chain, func_name, slot)
