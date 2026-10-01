"""Guided backpropagation and deconvolution: strict exact-ReLU, op-level.

Attrib memo D11-D13. Both methods are DEFINED in the literature for exact
ReLU only, so the coverage rule is strict: ``torch.relu``/``torch.relu_``,
``F.relu`` (either ``inplace``), ``Tensor.relu``/``Tensor.relu_``, and module
``nn.ReLU`` dispatch (which reaches ``F.relu``) are rewritten; GELU, SiLU,
LeakyReLU, ReLU6, Hardtanh, fused customs, and softmax are excluded -- a
"relu family" kwarg would borrow a published method's name while changing the
method. A future generalized positive-gradient rule ships under a DIFFERENT
name, reusing this module's activation-matching predicate as plumbing.

Coverage is OP-LEVEL via a ``TorchFunctionMode``: module-dispatched,
functional, in-place, and reused firings are all rewritten -- this is exactly
where module-hook implementations diverge (captum's GuidedBackprop cannot see
the one functional ``F.relu`` in torchvision's ``DenseNet.forward``; the
three-way identity row pins that divergence bit-exactly, memo D13).

In-place spellings are served OUT-OF-PLACE by the rewrite (the returned
tensor carries the rule; the input is never mutated). A forward that DISCARDS
the in-place return value would silently change semantics, so every call
verifies FORWARD FIDELITY: the rewritten forward must reproduce the plain
forward bit-exactly, or the call refuses typed.

Rules (D12): guided ReLU passes ``clamp(grad, min=0)`` through the native
ReLU backward (positive upstream gradient times the forward-positive mask);
deconvolution passes ``clamp(grad, min=0)`` WITHOUT the forward mask. Results
are SIGNED by default (``absolute=False``); shipped ``saliency`` still
absolute-values unconditionally, which is why no mode kwarg ships under a
classic name in v1.

There is deliberately NO spelling that mounts these backward rules on IG,
layer IG, conductance, or GradientShap: modified backward rules invalidate
path and completeness semantics, so the refusal is the absence of the knob.
``noise_tunnel(method=guided_backprop)`` works (D12).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils.hooks import RemovableHandle

from torchlens.attribution._core import (
    InputKwargs,
    _AttributionTarget,
    _call_model,
    _make_input_leaves,
    _normalize_model_inputs,
    _scalarize_output,
    _target_repr,
    _temporarily_eval,
)
from torchlens.attribution._result import AttributionError, AttributionResult

_RELU_NAMES = frozenset({"relu", "relu_"})

# Non-ReLU activation spellings recorded during the forward so a zero-match
# refusal can NAME what the model actually uses (teaching refusal).
_ACTIVATION_CENSUS_NAMES = frozenset(
    {
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "softmax",
        "log_softmax",
        "leaky_relu",
        "leaky_relu_",
        "relu6",
        "hardtanh",
        "hardtanh_",
        "elu",
        "elu_",
        "selu",
        "celu",
        "mish",
        "hardswish",
        "hardsigmoid",
        "glu",
        "prelu",
        "rrelu",
        "rrelu_",
        "softplus",
    }
)

_SITES_VOCABULARY = ("all", "module")


class _GuidedReluFunction(torch.autograd.Function):
    """Exact ReLU forward; backward passes positive gradient times the mask."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        """Compute the exact ReLU and save the input for the backward mask."""

        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        """Positive upstream gradient times the forward-positive mask."""

        (x,) = ctx.saved_tensors
        return grad_output.clamp(min=0) * (x > 0).to(grad_output.dtype)


class _DeconvReluFunction(torch.autograd.Function):
    """Exact ReLU forward; backward passes positive gradient WITHOUT the mask."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:
        """Compute the exact ReLU (no saved state; the mask is not applied)."""

        return torch.relu(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        """Positive upstream gradient, unmasked (deconvolution rule)."""

        return grad_output.clamp(min=0)


_RULE_FUNCTIONS: dict[str, type[torch.autograd.Function]] = {
    "guided_backprop": _GuidedReluFunction,
    "deconvolution": _DeconvReluFunction,
}


class _ReluRewriteMode(TorchFunctionMode):
    """Torch-function mode rewriting exact-ReLU calls with a backward rule.

    Parameters
    ----------
    rule
        The autograd Function implementing the modified backward.
    restrict_to_module
        When ``True`` only firings dispatched inside an ``nn.ReLU`` module are
        rewritten (the captum-coverage restriction of the three-way identity
        row); functional/method firings pass through with native backward.
    module_depth
        Shared one-element counter maintained by the ``nn.ReLU`` hooks.
    """

    def __init__(
        self,
        rule: type[torch.autograd.Function],
        restrict_to_module: bool,
        module_depth: list[int],
    ) -> None:
        """Initialize counters and the observed-activation census."""

        super().__init__()
        self.rule = rule
        self.restrict_to_module = restrict_to_module
        self.module_depth = module_depth
        self.total_relu_firings = 0
        self.module_dispatched_firings = 0
        self.rewritten_firings = 0
        self.observed_activations: set[str] = set()

    def __torch_function__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Rewrite exact-ReLU calls; record the activation census otherwise."""

        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if (
            name in _RELU_NAMES
            and args
            and isinstance(args[0], Tensor)
            and args[0].is_floating_point()
        ):
            self.total_relu_firings += 1
            in_module = self.module_depth[0] > 0
            if in_module:
                self.module_dispatched_firings += 1
            if in_module or not self.restrict_to_module:
                self.rewritten_firings += 1
                # In-place spellings are served out-of-place: the returned
                # tensor carries the rule and the input is never mutated; the
                # forward-fidelity check refuses if that changed semantics.
                return self.rule.apply(args[0])
            return func(*args, **kwargs)
        if name in _ACTIVATION_CENSUS_NAMES:
            self.observed_activations.add(name)
        return func(*args, **kwargs)


def _install_relu_module_hooks(model: Module, depth: list[int]) -> list[RemovableHandle]:
    """Install depth-counting hooks on every ``nn.ReLU`` module.

    Parameters
    ----------
    model
        Model whose ``nn.ReLU`` children should be tracked.
    depth
        Shared one-element counter incremented while inside an ``nn.ReLU``.

    Returns
    -------
    list[RemovableHandle]
        Handles to remove in the caller's ``finally``.
    """

    handles: list[RemovableHandle] = []
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):

            def _enter(_module: Module, _args: tuple[Any, ...]) -> None:
                """Mark entry into an nn.ReLU dispatch."""

                depth[0] += 1

            def _exit(_module: Module, _args: tuple[Any, ...], _output: Any) -> None:
                """Mark exit from an nn.ReLU dispatch."""

                depth[0] -= 1

            handles.append(module.register_forward_pre_hook(_enter))
            handles.append(module.register_forward_hook(_exit))
    return handles


def _modified_backward_attribution(
    rule_name: str,
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs,
    *,
    target: _AttributionTarget,
    sites: str,
    absolute: bool,
) -> AttributionResult:
    """Shared engine for the two named modified-backward methods.

    Parameters
    ----------
    rule_name
        ``"guided_backprop"`` or ``"deconvolution"``.
    model
        PyTorch module to attribute.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index or callable ``output -> scalar tensor``.
    sites
        ``"all"`` (every executed exact-ReLU op -- the op-level default) or
        ``"module"`` (only ``nn.ReLU``-dispatched firings -- the module-hook
        coverage class the three-way identity row compares against captum).
    absolute
        Whether to absolute-value the signed result (default signed, D12).

    Returns
    -------
    AttributionResult
        Signed (or absolute) input gradients under the modified rule, with
        the site census and observed-activation disclosure.

    Raises
    ------
    AttributionError
        On unknown ``sites``, zero matched ReLU sites (naming the observed
        activation kinds), or a forward-fidelity violation.
    """

    if sites not in _SITES_VOCABULARY:
        raise AttributionError(
            f"sites must be one of {_SITES_VOCABULARY}; got {sites!r}. "
            "Remedy: use 'all' for op-level coverage or 'module' for "
            "nn.ReLU-dispatched firings only.",
            code="guided_sites_invalid",
        )
    prepared = _normalize_model_inputs(inputs, input_kwargs)
    rule = _RULE_FUNCTIONS[rule_name]
    depth = [0]
    hook_handles: list[RemovableHandle] = []
    try:
        with _temporarily_eval(model):
            input_leaves = _make_input_leaves(prepared)
            # Plain reference forward FIRST: the rewritten forward must
            # reproduce it bit-exactly (the rewrite is an identity on
            # forward values), or an in-place-discarding forward changed
            # semantics under the out-of-place rewrite.
            with torch.no_grad():
                reference_output = _call_model(model, prepared, input_leaves)
                reference_scalar = _scalarize_output(reference_output, target)
            hook_handles = _install_relu_module_hooks(model, depth)
            mode = _ReluRewriteMode(rule, sites == "module", depth)
            with mode:
                output = _call_model(model, prepared, input_leaves)
            scalar = _scalarize_output(output, target)
            if not torch.equal(scalar.detach(), reference_scalar):
                raise AttributionError(
                    f"{rule_name} forward under the ReLU rewrite did not "
                    "reproduce the plain forward bit-exactly. The rewrite "
                    "serves in-place ReLU out-of-place, so a forward that "
                    "DISCARDS the in-place return value (e.g. `relu_(x)` "
                    "followed by reads of `x`) is outside the coverage "
                    "claim. Remedy: use the returned tensor of in-place "
                    "ReLU calls in the model's forward.",
                    code="guided_forward_fidelity_violated",
                )
            if mode.rewritten_firings == 0:
                observed = sorted(mode.observed_activations)
                observed_note = (
                    f"observed activation kinds: {', '.join(observed)}"
                    if observed
                    else "no census-recognized activation calls were observed"
                )
                raise AttributionError(
                    f"{rule_name} matched ZERO exact-ReLU sites under "
                    f"sites={sites!r} ({observed_note}). Guided rules are "
                    "defined for exact ReLU only; GELU, SiLU, LeakyReLU, "
                    "ReLU6, Hardtanh, fused customs, and softmax are "
                    "excluded by definition. Remedy: use these methods on "
                    "ReLU models; for other activations use saliency, "
                    "input_x_grad, or integrated_gradients.",
                    code="guided_no_relu_sites",
                )
            gradients, _scalar_detached = _gradient_for_scalar(scalar, prepared, input_leaves)
    finally:
        for handle in hook_handles:
            handle.remove()
    values = tuple(gradient.abs() if absolute else gradient for gradient in gradients)
    from torchlens.attribution._core import _value_tree_from_leaves

    return AttributionResult(
        method=rule_name,
        values=_value_tree_from_leaves(prepared, values),
        target_repr=_target_repr(target),
        extra={
            "sites": sites,
            "absolute": absolute,
            "site_census": {
                "total_relu_firings": mode.total_relu_firings,
                "module_dispatched_firings": mode.module_dispatched_firings,
                "functional_or_method_firings": (
                    mode.total_relu_firings - mode.module_dispatched_firings
                ),
                "rewritten_firings": mode.rewritten_firings,
            },
            "observed_activations": sorted(mode.observed_activations),
            "inplace_note": (
                "in-place ReLU spellings are served out-of-place; forward "
                "fidelity is verified bit-exactly against the plain forward"
            ),
        },
    )


def _gradient_for_scalar(
    scalar: Tensor,
    prepared: Any,
    input_leaves: tuple[Tensor, ...],
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Differentiate an already-scalarized target with the shared dedup rules.

    Parameters
    ----------
    scalar
        Scalar target value connected to the rewritten graph.
    prepared
        Normalized attribution inputs.
    input_leaves
        Identity-interned differentiable leaves.

    Returns
    -------
    tuple[tuple[Tensor, ...], Tensor]
        Detached per-slot gradients and the detached scalar.

    Raises
    ------
    AttributionError
        If the target is not differentiable with respect to the inputs.
    """

    del prepared  # Reserved for future per-slot diagnostics.
    unique_index_by_id: dict[int, int] = {}
    unique_leaves: list[Tensor] = []
    for leaf in input_leaves:
        if id(leaf) not in unique_index_by_id:
            unique_index_by_id[id(leaf)] = len(unique_leaves)
            unique_leaves.append(leaf)
    try:
        raw = torch.autograd.grad(scalar, unique_leaves, allow_unused=True)
    except RuntimeError as exc:
        raise AttributionError(
            "target scalar is not differentiable with respect to the attributed "
            "inputs. Remedy: choose a target built from the model output.",
            code="attribution_target_not_differentiable",
        ) from exc
    gradients = tuple(
        torch.zeros_like(leaf)
        if raw[unique_index_by_id[id(leaf)]] is None
        else raw[unique_index_by_id[id(leaf)]].detach()
        for leaf in input_leaves
    )
    return gradients, scalar.detach()


def guided_backprop(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    sites: str = "all",
    absolute: bool = False,
) -> AttributionResult:
    """Guided backpropagation: positive gradients through exact-ReLU sites.

    Every executed exact-ReLU op is rewritten (module-dispatched, functional,
    in-place, and reused firings); the backward passes
    ``clamp(grad, min=0) * (input > 0)`` at each site. Results are SIGNED by
    default.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index or callable ``output -> scalar tensor``.
    sites
        ``"all"`` (op-level coverage, default) or ``"module"``
        (``nn.ReLU``-dispatched firings only -- the module-hook coverage class
        for reference-implementation parity).
    absolute
        Whether to absolute-value the result (default ``False``).

    Returns
    -------
    AttributionResult
        Guided gradients with the site census disclosure.
    """

    return _modified_backward_attribution(
        "guided_backprop",
        model,
        inputs,
        input_kwargs,
        target=target,
        sites=sites,
        absolute=absolute,
    )


def deconvolution(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    sites: str = "all",
    absolute: bool = False,
) -> AttributionResult:
    """Deconvolution: positive gradients WITHOUT the forward-ReLU mask.

    Every executed exact-ReLU op is rewritten; the backward passes
    ``clamp(grad, min=0)`` at each site (no forward mask -- the defining
    difference from guided backpropagation). Results are SIGNED by default.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor or tuple/list of positional model arguments.
    input_kwargs
        Optional keyword arguments for ``model``.
    target
        Integer class index or callable ``output -> scalar tensor``.
    sites
        ``"all"`` (op-level coverage, default) or ``"module"``
        (``nn.ReLU``-dispatched firings only).
    absolute
        Whether to absolute-value the result (default ``False``).

    Returns
    -------
    AttributionResult
        Deconvolution gradients with the site census disclosure.
    """

    return _modified_backward_attribution(
        "deconvolution",
        model,
        inputs,
        input_kwargs,
        target=target,
        sites=sites,
        absolute=absolute,
    )


__all__ = ["deconvolution", "guided_backprop"]
