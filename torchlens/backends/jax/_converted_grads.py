"""Derived gradients for module roots and closure constants, over the captured jaxpr.

The raw-function gradient path differentiates ``fn(params, *inputs)`` and
refuses anything it cannot see as an explicit leaf. Two common shapes hold
their arrays as closed-jaxpr constants instead:

* a module root (Equinox or Flax NNX): the module's parameter arrays are
  closure constants of the jaxpr traced from ``model(*inputs)``;
* a function root that closes over arrays (recorded as ``constant`` ops).

Both are served the same way: the captured closed jaxpr is evaluated as a pure
function whose constants are explicit arguments. Module parameters (matched to
their module address by value identity, the channel the capture already uses)
become the differentiated ``params`` dict, so records land at
``params.<address>`` and mirror onto ``trace.params``; any other constant is a
non-differentiated trailing leaf. Differentiating the captured jaxpr itself
means the gradient belongs to exactly the graph the trace recorded.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ..._data_classes.derived_grad import (
    IntermediateDerivedGradAccessor,
    set_intermediate_derived_grads,
)
from ...backends.registry import BackendUnsupportedError
from ._host_state import _reject_closed_over_host_state


def _param_const_positions(closed_jaxpr: Any, module_tree: Any) -> dict[int, str]:
    """Map closed-jaxpr const positions to module param addresses by identity."""

    if module_tree is None:
        return {}
    by_id = module_tree.param_address_by_value_id
    return {
        index: by_id[id(const)]
        for index, const in enumerate(closed_jaxpr.consts)
        if id(const) in by_id
    }


def _jaxpr_function(
    closed_jaxpr: Any,
    param_positions: dict[int, str],
    static_argnums: tuple[int, ...],
    output_treedef: Any,
    module_root: bool,
) -> Callable[..., Any]:
    """Return ``fn(params, *args, other_consts)`` evaluating the captured jaxpr.

    For a function root ``params`` is the jaxpr's first dynamic argument; for a
    module root it fills the constants matched to module parameters.
    """

    import jax
    from jax import core as jax_core

    def converted(*call_args: Any) -> Any:
        """Evaluate the captured jaxpr with explicit constant arguments."""

        params, *positional, other_consts = call_args
        others = iter(other_consts)
        consts = [
            params[param_positions[index]] if index in param_positions else next(others)
            for index in range(len(closed_jaxpr.consts))
        ]
        dynamic = tuple(
            value for index, value in enumerate(positional) if index + 1 not in static_argnums
        )
        leaves = jax.tree.leaves(dynamic if module_root else (params, *dynamic))
        flat = jax_core.eval_jaxpr(closed_jaxpr.jaxpr, consts, *leaves)
        return jax.tree.unflatten(output_treedef, flat)

    return converted


def converted_gradient_view(
    *,
    model: Any,
    args: Sequence[Any],
    closed_jaxpr: Any,
    module_tree: Any,
    grad_options: Any,
    static_argnums: tuple[int, ...],
    captured_output: Any,
) -> tuple[Callable[..., Any], list[Any], Any, tuple[int, ...]] | None:
    """Return the explicit-leaf gradient view, or ``None`` for the plain raw path.

    ``closed_jaxpr`` is the captured jaxpr (module returns stripped), ``module_tree``
    the attributed tree of a module root (else ``None``) and ``captured_output``
    the raw output whose pytree structure the view returns. The result is
    ``(fn, [params, *inputs, other_consts], grad_options, fn_static_argnums)``.
    """

    consts = tuple(getattr(closed_jaxpr, "consts", ()))
    if module_tree is None and not consts:
        return None
    param_positions = _param_const_positions(closed_jaxpr, module_tree)
    if module_tree is not None:
        if grad_options.params is not None and grad_options.params is not model:
            raise BackendUnsupportedError(
                "JAX module-root derived gradients read the parameters from the module "
                "itself; pass GradOptions(params=model) or params=None.",
                code="trace_option_unsupported",
                option="grad_options",
            )
        params: Any = {address: consts[index] for index, address in param_positions.items()}
        positional = list(args)
        shifted_static = tuple(index + 1 for index in static_argnums)
    else:
        params, positional = args[0], list(args[1:])
        shifted_static = static_argnums
    others = tuple(const for index, const in enumerate(consts) if index not in param_positions)
    import jax

    treedef = jax.tree.structure(captured_output)
    fn = _jaxpr_function(
        closed_jaxpr, param_positions, shifted_static, treedef, module_tree is not None
    )
    view_options = replace(grad_options, params=params, intermediate_grads=False)
    return fn, [params, *positional, others], view_options, shifted_static


def attach_jax_derived_grads(
    backend: Any,
    *,
    trace: Any,
    call: tuple[Any, Callable[..., Any], Sequence[Any], tuple[int, ...]],
    closed_jaxpr: Any,
    module_tree: Any,
    captured_output: Any,
    grad_options: Any,
    control_flow: tuple[str, int],
    derive: Callable[..., Any],
) -> None:
    """Attach derived gradients, through the explicit-leaf view when it applies.

    ``backend`` owns the raw-path gradient methods; ``call`` is
    ``(model, traced_callable, args, static_argnums)`` of the capture and
    ``control_flow`` its ``(jax_control_flow, jax_max_control_flow_unroll)``;
    ``derive`` is ``jaxpr.derive_closed_jaxpr`` (passed in to keep this module
    out of the backend import cycle). A
    module root or a function root with closure constants takes the
    explicit-leaf view; every other root takes the raw path unchanged.
    """

    model, traced, args, static_argnums = call
    view = converted_gradient_view(
        model=model,
        args=args,
        closed_jaxpr=closed_jaxpr,
        module_tree=module_tree,
        grad_options=grad_options,
        static_argnums=static_argnums,
        captured_output=captured_output,
    )
    if view is None and traced is not model and inspect.isfunction(model):
        # The raw path inspects ``traced``; also inspect the keyword-bound user callable.
        _reject_closed_over_host_state(model)
    fn, fn_args, view_options, fn_static = view or (
        traced,
        list(args),
        grad_options,
        static_argnums,
    )
    backend._attach_derived_grads(
        trace=trace,
        model=fn,
        args=fn_args,
        static_argnums=fn_static,
        closed_jaxpr=closed_jaxpr if view is None else derive(fn, fn_args, fn_static),
        captured_output=captured_output,
        grad_options=view_options,
        jax_control_flow=control_flow[0],
        jax_max_control_flow_unroll=control_flow[1],
    )
    if view is not None and grad_options.intermediate_grads:
        captured = (closed_jaxpr, args, static_argnums)
        _attach_intermediate(backend, trace, captured, captured_output, grad_options, control_flow)


def _attach_intermediate(
    backend: Any,
    trace: Any,
    capture: tuple[Any, Sequence[Any], tuple[int, ...]],
    captured_output: Any,
    grad_options: Any,
    control_flow: tuple[str, int],
) -> None:
    """Attach zero-tap intermediate gradients over the captured jaxpr and inputs."""

    closed_jaxpr, args, static_argnums = capture
    records = getattr(trace, "_autograd_derived_grads", None)
    fingerprint = next(
        (
            record.provenance.get("fingerprint")
            for record in getattr(records, "values", lambda: ())()
            if isinstance(getattr(record, "provenance", None), dict)
        ),
        None,
    )
    try:
        accessor = backend._derive_intermediate_grads_zero_tap(
            trace=trace,
            closed_jaxpr=closed_jaxpr,
            args=args,
            static_argnums=static_argnums,
            captured_output=captured_output,
            grad_options=grad_options,
            fingerprint=fingerprint,
            jax_control_flow=control_flow[0],
            jax_max_control_flow_unroll=control_flow[1],
        )
    except BackendUnsupportedError:
        raise
    except Exception as exc:  # noqa: BLE001 - same degraded contract as the raw path
        set_intermediate_derived_grads(trace, IntermediateDerivedGradAccessor())
        trace.jax_intermediate_derived_grad_status = {
            "status": "degraded",
            "reason": f"producer_error:{type(exc).__name__}",
        }
        return
    set_intermediate_derived_grads(trace, accessor)
