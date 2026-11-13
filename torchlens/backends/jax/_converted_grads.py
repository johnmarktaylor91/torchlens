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
from typing import Any, NamedTuple

from ..._data_classes.derived_grad import (
    IntermediateDerivedGradAccessor,
    set_intermediate_derived_grads,
)
from ...backends.registry import BackendUnsupportedError
from ._edited_grads import (
    EditedProgram,
    edited_closed_jaxpr,
    edited_grads_refusal,
    edited_program_of,
)
from ._host_state import _reject_closed_over_host_state


class DerivedGradCapture(NamedTuple):
    """What a derived-gradient attach reads from one JAX capture.

    ``traced`` is the callable the capture traced (``model`` itself, or the
    keyword-bound wrapper); ``closed_jaxpr`` is the captured jaxpr with module
    returns stripped, ``module_tree`` the attributed tree of a module root (else
    ``None``), ``captured_output`` the raw output, and ``control_flow`` is
    ``(jax_control_flow, jax_max_control_flow_unroll)``.
    """

    model: Any
    traced: Callable[..., Any]
    args: Sequence[Any]
    static_argnums: tuple[int, ...]
    closed_jaxpr: Any
    module_tree: Any
    captured_output: Any
    grad_options: Any
    control_flow: tuple[str, int]


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


def _module_root_inputs(params: Any, dynamic: tuple[Any, ...]) -> Any:
    """Return a module root's jaxpr inputs: params fill constants, not arguments."""

    del params
    return dynamic


def _function_root_inputs(params: Any, dynamic: tuple[Any, ...]) -> Any:
    """Return a function root's jaxpr inputs: params are the first dynamic argument."""

    return (params, *dynamic)


def _jaxpr_function(
    closed_jaxpr: Any,
    param_positions: dict[int, str],
    static_argnums: tuple[int, ...],
    output_treedef: Any,
    module_root: bool,
    edited: Any = None,
) -> Callable[..., Any]:
    """Return ``fn(params, *args, other_consts)`` evaluating the captured jaxpr.

    For a function root ``params`` is the jaxpr's first dynamic argument; for a
    module root it fills the constants matched to module parameters. ``edited``
    is the edited program of an edited capture (see ``edited_closed_jaxpr``): it
    is evaluated instead, with its own constants as the tail of ``other_consts``.
    """

    import jax
    from jax import core as jax_core

    num_others = len(closed_jaxpr.consts) - len(param_positions)

    # A function, not the bool: the host-state check rejects closed-over scalars.
    jaxpr_inputs = _module_root_inputs if module_root else _function_root_inputs

    def converted(*call_args: Any) -> Any:
        """Evaluate the captured jaxpr with explicit constant arguments."""

        params, *positional, other_consts = call_args
        others = iter(other_consts[:num_others])
        consts = [
            params[param_positions[index]] if index in param_positions else next(others)
            for index in range(len(closed_jaxpr.consts))
        ]
        dynamic = tuple(
            value for index, value in enumerate(positional) if index + 1 not in static_argnums
        )
        leaves = jax.tree.leaves(jaxpr_inputs(params, dynamic))
        if edited is None:
            flat = jax_core.eval_jaxpr(closed_jaxpr.jaxpr, consts, *leaves)
        else:
            flat = jax_core.eval_jaxpr(edited.jaxpr, other_consts[num_others:], *consts, *leaves)
        return jax.tree.unflatten(output_treedef, flat)

    return converted


def converted_gradient_view(
    capture: DerivedGradCapture,
    edits: EditedProgram | None = None,
) -> tuple[Callable[..., Any], list[Any], Any, tuple[int, ...]] | None:
    """Return the explicit-leaf gradient view, or ``None`` for the plain raw path.

    The view returns the pytree structure of ``capture.captured_output``. The result is
    ``(fn, [params, *inputs, other_consts], grad_options, fn_static_argnums)``. An
    edited capture always takes the view, evaluated with its edits re-applied, so
    the gradient belongs to the edited program and never reruns the unedited model.
    """

    model, _, args, static_argnums, closed_jaxpr, module_tree, captured_output = capture[:7]
    grad_options = capture.grad_options
    consts = tuple(getattr(closed_jaxpr, "consts", ()))
    if module_tree is None and not consts and edits is None:
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
    edited = None if edits is None else edited_closed_jaxpr(closed_jaxpr, edits)
    if edited is not None:
        others = (*others, *edited.consts)
    fn = _jaxpr_function(
        closed_jaxpr, param_positions, shifted_static, treedef, module_tree is not None, edited
    )
    view_options = replace(grad_options, params=params, intermediate_grads=False)
    return fn, [params, *positional, others], view_options, shifted_static


def attach_jax_derived_grads(
    backend: Any,
    *,
    trace: Any,
    capture: DerivedGradCapture,
    derive: Callable[..., Any],
) -> None:
    """Attach derived gradients, through the explicit-leaf view when it applies.

    ``backend`` owns the raw-path gradient methods; ``derive`` is
    ``jaxpr.derive_closed_jaxpr`` (passed in to keep this module out of the
    backend import cycle). A module root or a function root with closure
    constants takes the explicit-leaf view, and so does any capture whose
    ``intervene=`` edits fired (evaluated with the edits); every other root takes
    the raw path unchanged.
    """

    if getattr(trace, "is_stopped", False) is True:
        raise edited_grads_refusal(
            "a capture halted by stop_after= has no gradient program for its halted "
            "boundary yet; differentiating the whole jaxpr would rerun past the stop. "
            "Drop stop_after= or grad_options=.",
            code="intervention_with_derived_grads_unsupported",
        )
    model, traced, args, static_argnums = capture[:4]
    grad_options, control_flow = capture.grad_options, capture.control_flow
    edits = edited_program_of(backend, trace)
    if edits is not None:
        edits.check_positions()
    view = converted_gradient_view(capture, edits)
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
        closed_jaxpr=capture.closed_jaxpr if view is None else derive(fn, fn_args, fn_static),
        captured_output=capture.captured_output,
        grad_options=view_options,
        jax_control_flow=control_flow[0],
        jax_max_control_flow_unroll=control_flow[1],
    )
    if view is not None and grad_options.intermediate_grads:
        _attach_intermediate(backend, trace, capture)


def _attach_intermediate(backend: Any, trace: Any, capture: DerivedGradCapture) -> None:
    """Attach zero-tap intermediate gradients over the captured jaxpr and inputs."""

    _, _, args, static_argnums, closed_jaxpr, _, captured_output, grad_options = capture[:8]
    control_flow = capture.control_flow
    records = trace._autograd_derived_grads
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
