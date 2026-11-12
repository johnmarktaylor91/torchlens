"""Equivalence class and call-site class as resolvers over recorded inputs (lane FQ).

Each class is split into what must be observed at the call and a resolver that turns
those observations into the class string:

- The parameter-free equivalence class: the observation is the argument record
  (:func:`~torchlens.backends.torch._arg_hash.observe_arg_inputs`), the output slot of
  a multi-output call and the innermost module frame; :func:`resolve_equivalence_class`
  is a pure function of them and may run after the call.
- The call-site class: the observation is the call-site key (:func:`pending_chain`
  and :func:`observe_call_site_key`, read while the call's pending chain is
  current); :func:`resolve_call_site_class` numbers keys by first appearance in the
  capture's index, so resolves must run in op order.

The parameterized equivalence class (``tensor_tracking._make_raw_param_group_barcode``)
is already a pure function of recorded values: the layer type, the parameter
barcodes and the output slot.
"""

from collections.abc import Sequence
from typing import Any

from ..._utils._call_site import call_site_class, call_site_scope
from ._arg_hash import resolve_arg_hash

__all__ = (
    "observe_call_site_key",
    "pending_chain",
    "resolve_call_site_class",
    "resolve_equivalence_class",
)


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


def pending_chain(pending: tuple[Any, Any] | None, func_call_id: Any) -> Any:
    """Return the call-site chain pending for ``func_call_id``, else ``None``.

    ``pending`` is the raw-graph workspace's ``(func_call_id, chain)`` of the wrapped
    call being logged; a chain left by another call is not this call's.
    """

    if pending is None or pending[0] != func_call_id:
        return None
    return pending[1]


def observe_call_site_key(
    chain: Any,
    modules: Sequence[tuple[str, int]],
    stateful: frozenset[str] | None,
    func_name: str,
    slot: int | None,
) -> tuple[Any, ...] | None:
    """Return the call-site key ``(scope, chain, func_name, slot)`` of one output.

    ``chain`` is the call's chain from :func:`pending_chain`; the key is None when it
    is missing or empty. The scope is the innermost stateful module on the op's module
    stack (``stateful`` is the session's frozen set). Every part is immutable once read.
    """

    if not chain:
        return None
    scope = call_site_scope((address for address, _pass in modules), stateful)
    return (scope, chain, func_name, slot)


def resolve_call_site_class(index: dict[Any, int], key: tuple[Any, ...], layer_type: str) -> str:
    """Return the call-site class ``<type>@cs<N>`` of a key, numbering it in ``index``.

    ``N`` is the key's first-appearance number in the capture, so keys must be
    resolved in op order.
    """

    return call_site_class(index, key, layer_type, key[3])
