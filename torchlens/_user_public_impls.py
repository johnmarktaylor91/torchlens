"""Private implementations for public user-facing utility commands."""

from __future__ import annotations

import contextlib
import os
import random
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from torch import nn
from tqdm import tqdm

from . import user_funcs as _user_funcs
from ._capture_state_helpers import (
    _clone_state_dict_with_metadata,
    _model_for_ground_truth_validation,
    _model_for_validation_replay,
    _ModuleTreePlainAttrSnapshot,
    _move_tensors_to_device,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ._deploy_env import restore_state_dict_resilient
from ._deprecations import MISSING, MissingType
from ._input_coerce import _coerce_input_args
from ._literals import (
    BufferVisibilityLiteral,
    CollapseLiteral,
    FoldRepeatsLiteral,
    VisDirectionLiteral,
    VisModeLiteral,
    VisNodeModeLiteral,
    VisNodePlacementLiteral,
    VisRendererLiteral,
)
from ._robustness import check_model_and_input_variants
from .backends import BackendName, resolve_backend_spec
from .data_classes.trace import Trace
from .errors import TraceNotReproducibleWarning
from .options import (
    CaptureOptions,
    VisualizationOptions,
    merge_visualization_options,
    visualization_to_render_kwargs,
)
from .utils.arg_handling import normalize_input_args, safe_copy_input_tree
from .utils.display import warn_parallel
from .utils.hashing import compute_graph_shape_hash
from .utils.introspection import get_vars_of_type_from_obj
from .utils.rng import set_random_seed
from .visualization.code_panel import CodePanelOption

if TYPE_CHECKING:
    import pandas as pd

    from .data_classes.module import Module


def release_model(model: nn.Module) -> None:
    """Release a traced PyTorch model from persistent TorchLens preparation.

    Parameters
    ----------
    model:
        Model whose full module tree should be restored. The operation is safe
        for never-traced and already-released models.

    Returns
    -------
    None
        The model is restored in place and may be pickled or traced again.

    Notes
    -----
    TorchLens installs persistent, toggle-gated wrappers on non-root module
    ``forward`` methods. Call ``release_model`` after the final trace when the
    complete model object must be serialized with :func:`torch.save` or
    :mod:`pickle`. Saving ``model.state_dict()`` is unaffected by preparation
    and does not require release.

    Plain module attributes holding torch function references captured in the
    other wrap state (``self.act = F.relu`` grabbed before wrapping, pickled
    while wrapped -- or the reverse) fail pickle's by-reference identity
    check. ``release_model`` normalizes such attributes (including one level
    of builtin list/tuple/dict/set nesting) to the values currently live at
    their public torch names, so serialization succeeds at release time and a
    fresh-process load resolves the pristine torch function. Call it again
    after any later wrap-state change before re-serializing. References held
    inside closures, ``functools.partial`` objects, or custom containers --
    and bare references held outside the model -- remain outside the sweep.
    """
    from .backends.torch.model_prep import release_model as release_torch_model

    release_torch_model(model)


def log_model_metadata(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
) -> Trace:
    """Return model metadata without saving any outs.

    Equivalent to ``trace(model, input_args, input_kwargs,
    capture=CaptureOptions(layers_to_save=None,
    compute_input_output_distances=True))``.

    Parameters
    ----------
    model:
        PyTorch model to inspect.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.

    Returns
    -------
    Trace
        Trace with full metadata but no saved outs.
    """
    model = unwrap_compiled_model(model)
    model_trace = _user_funcs.trace(
        model,
        input_args,
        input_kwargs,
        capture=CaptureOptions(
            layers_to_save=None,
            compute_input_output_distances=True,
        ),
    )
    return model_trace


def summary(  # noqa: PLR0913 -- ladder-conjugated public verb: the three input rungs plus the two execution dials ARE the spec'd surface (F17 B11)
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    input_kwargs: dict[Any, Any] | None = None,
    *,
    input_size: Any | None = None,
    execution_mode: Literal["eval", "train", "same"] = "eval",
    grad_mode: Literal["off", "same"] = "off",
    **summary_kwargs: Any,
) -> str:
    """Run a metadata-only forward pass and return a rendered summary report.

    The one-call door is SAFE by default (A4): the captured forward runs in
    eval mode under ``torch.no_grad()``, and every module training flag plus
    the host/device RNG state is restored bit-identically afterwards -- a
    summary call never mutates the model (BatchNorm running stats included)
    or advances the caller's RNG streams. The function RETURNS the result
    and never auto-prints (the repr self-displays).

    Input precedence follows the quickstart ladder (F17, memo D2): real
    ``input_args`` XOR ``input_size=`` XOR nothing. ``input_size=``
    synthesizes seeded tensors per the quickstart grammar and the synthesis
    is disclosed in the render; a zero-input call infers a verified input
    and REUSES its verification trace (no second capture). Mixing the
    spellings refuses typed (``input_rung_conflict``).

    Parameters
    ----------
    model:
        PyTorch model to inspect.
    input_args:
        Positional args for ``model.forward()``. The quickstart input ladder
        applies (F17): a real input is the gold rung; omit it and pass
        ``input_size=`` for a declared shape with synthesized values; omit
        both for an inferred shape. Non-gold rungs are disclosed in the
        trace's persistent provenance record and in the summary text.
    input_kwargs:
        Keyword args for ``model.forward()``.
    input_size:
        Declared input shape(s) per the quickstart grammar (one flat tuple,
        a sequence of tuples, or a forward-keyword-to-shape mapping).
    execution_mode:
        ``"eval"`` (default) runs the captured forward in eval mode and
        restores every module training flag afterwards. ``"train"`` opts into
        train-mode execution (stateful layers such as BatchNorm WILL update
        their buffers). ``"same"`` leaves modes exactly as the caller set
        them.
    grad_mode:
        ``"off"`` (default) runs the captured forward under
        ``torch.no_grad()``. ``"same"`` keeps the caller's grad context.
    **summary_kwargs:
        Forwarded to ``Trace.summary`` (rebuilt grammar or legacy presets).

    Returns
    -------
    str
        A ``SummaryReport`` (``str`` subclass): canonical ASCII payload
        plus the typed result API.
    """
    from .utils.rng import log_current_rng_states, set_rng_from_saved_states

    _validate_summary_modes(execution_mode, grad_mode)
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if input_kwargs is None:
        input_kwargs = {}

    # Quickstart input ladder (F17 B11): the summary verb conjugates exactly
    # like trace and render. Non-gold rungs synthesize concrete tensors here
    # (disclosed via the provenance record attached after capture AND the
    # rebuilt renderer's synthetic-input banner line); the inferred rung
    # reuses the inference search's exact verified trace when the default
    # eval/no-grad policy is requested (memo D9 in-call reuse).
    ladder_provenance = None
    input_synthesis: str | None = None
    facade_report = _maybe_weightsfree_summary(
        model, input_args, input_kwargs, input_size, summary_kwargs
    )
    if facade_report is not None:
        return facade_report
    if input_size is not None or (input_args is None and not input_kwargs):
        from .quickstart._resolve import attach_provenance, resolve_inputs

        resolved = resolve_inputs(model, input_args, input_kwargs, input_size, verb="summary")
        ladder_provenance = resolved.provenance
        input_synthesis = _ladder_synthesis_note(input_size, resolved)
        if (
            resolved.plan.verified_trace is not None
            and execution_mode == "eval"
            and grad_mode == "off"
        ):
            reused = resolved.plan.verified_trace
            attach_provenance(reused, ladder_provenance)
            return _summary_report_from_trace(
                reused,
                summary_kwargs,
                execution_mode="eval",
                grad_mode="off",
                input_synthesis=input_synthesis,
            )
        input_args = list(resolved.plan.input_args)
        input_kwargs = dict(resolved.plan.input_kwargs)

    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)

    saved_training_flags = [(module, module.training) for module in model.modules()]
    rng_snapshot = log_current_rng_states()
    grad_context: contextlib.AbstractContextManager[Any]
    grad_context = torch.no_grad() if grad_mode == "off" else contextlib.nullcontext()
    try:
        if execution_mode == "eval":
            model.eval()
        elif execution_mode == "train":
            model.train()
        with grad_context:
            trace = _user_funcs._run_model_and_save_specified_outs(
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                layers_to_save=None,
                recurrence_detection=True,
            )
    finally:
        for module, was_training in saved_training_flags:
            module.training = was_training
        set_rng_from_saved_states(rng_snapshot)

    if ladder_provenance is not None:
        from .quickstart._resolve import attach_provenance

        attach_provenance(trace, ladder_provenance)
    return _summary_report_from_trace(
        trace,
        summary_kwargs,
        execution_mode=execution_mode,
        grad_mode=grad_mode,
        input_synthesis=input_synthesis,
    )


def _format_input_size(input_size: Any) -> str:
    """Render the caller's ``input_size=`` spelling compactly for disclosure."""

    try:
        if isinstance(input_size, dict):
            inner = ", ".join(
                f"{key!r}: {_format_input_size(value)}" for key, value in input_size.items()
            )
            return "{" + inner + "}"
        items = list(input_size)
        if items and all(isinstance(dim, int) and not isinstance(dim, bool) for dim in items):
            return repr(tuple(items))
        return "(" + ", ".join(_format_input_size(item) for item in items) + ")"
    except TypeError:
        return repr(input_size)


def _ladder_synthesis_note(input_size: Any, resolved: Any) -> str | None:
    """Derive the one-line synthesis disclosure from ladder provenance.

    Non-gold rungs feed the rebuilt renderer's ``synthetic input:`` banner
    line (F08) from the ONE quickstart provenance record (F17); the gold
    rung returns ``None`` (real values, nothing to disclose).
    """

    provenance = resolved.provenance
    if provenance.values_semantic:
        return None
    recipes = provenance.recipes or ()
    recipe_names = sorted({str(row["recipe"]) for row in recipes if row.get("recipe")})
    dtypes = sorted({str(row["dtype"]) for row in recipes if row.get("dtype")})
    recipe_part = "/".join(recipe_names) if recipe_names else "synthesized"
    if dtypes:
        recipe_part += " " + "/".join(dtype.removeprefix("torch.") for dtype in dtypes)
    if provenance.origin == "declared":
        seeds = sorted({row["seed"] for row in recipes if row.get("seed") is not None})
        seed_part = f", seed {seeds[0]}" if len(seeds) == 1 else ""
        return f"input_size={_format_input_size(input_size)}, {recipe_part}{seed_part}"
    shapes = ", ".join(str(tuple(facts.shape)) for facts in provenance.tensors)
    note = f"inferred shape {shapes or 'unknown'}, {recipe_part}"
    if provenance.strategy:
        note += f", strategy {provenance.strategy}"
    if provenance.flexible_dims:
        note += f", flexible dims {tuple(provenance.flexible_dims)}"
    return note


def _summary_report_from_trace(
    trace: Any,
    summary_kwargs: dict[str, Any],
    *,
    execution_mode: str,
    grad_mode: str,
    input_synthesis: str | None,
) -> str:
    """Route one captured trace through the summary grammars and clean up."""

    from ._errors import InvalidArgumentError
    from .report._summary_config import route_summary_call

    route_kwargs = dict(summary_kwargs)
    level = route_kwargs.pop("level", None)
    route = route_summary_call(level, route_kwargs)
    if input_synthesis is not None and isinstance(level, str) and level == "output":
        raise InvalidArgumentError(
            "decoded-output views refuse synthetic inputs: a label table computed "
            "from noise is the most misleading thing this surface could print.",
            code="summary_synthetic_output_refused",
            remedy="pass a real input for output views, or drop level='output'",
        )
    try:
        if route == "rebuilt":
            return trace.summary(
                **summary_kwargs,
                _execution_note=_summary_execution_note(execution_mode, grad_mode, short=True),
                _input_synthesis=input_synthesis,
            )
        report = trace.summary(**summary_kwargs)
    finally:
        trace.cleanup()
    finalized = _finalize_summary_report(report, execution_mode, grad_mode)
    if input_synthesis is None:
        return finalized
    from .report._summary_report import SummaryReport

    full_text = str(finalized) + f"\nSynthetic input: {input_synthesis}."
    if isinstance(finalized, SummaryReport):
        return SummaryReport(
            full_text, rows=finalized.rows, totals=finalized.totals, capture=finalized.capture
        )
    return full_text


def _validate_summary_modes(execution_mode: str, grad_mode: str) -> None:
    """Refuse unknown one-call summary execution/grad mode tokens typed."""

    from ._errors import InvalidArgumentError

    if execution_mode not in ("eval", "train", "same"):
        raise InvalidArgumentError(
            f"execution_mode must be 'eval', 'train', or 'same'; got {execution_mode!r}.",
            code="summary_execution_mode_invalid",
            remedy=(
                "Use 'eval' (the default, safe reporting mode -- modes restored after the "
                "one captured forward), 'train' (explicit opt-in; stateful layers update), "
                "or 'same' (keep the caller's module modes)."
            ),
        )
    if grad_mode not in ("off", "same"):
        raise InvalidArgumentError(
            f"grad_mode must be 'off' or 'same'; got {grad_mode!r}.",
            code="summary_grad_mode_invalid",
            remedy=(
                "Use 'off' (the default; the captured forward runs under torch.no_grad()) "
                "or 'same' (keep the caller's grad context)."
            ),
        )


def _maybe_weightsfree_summary(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    input_size: Any,
    summary_kwargs: dict[str, Any],
) -> str | None:
    """Rung-4 gate (weightsfree memo D12/face 2): auto-select or refuse.

    ``None`` on every non-meta model (the ordinary summary path proceeds).
    For an ALL-meta model: with input evidence (declared ``input_size`` or
    explicit inputs) the facade auto-selects the exact structure-only option
    state — same marker, admission record, and evidence envelope as the
    explicit power path, auto-set recorded in provenance; a bare call with
    NO input evidence refuses rather than guessing (the inference rung runs
    real probing forwards a weights-free model cannot serve).
    """

    if not _all_meta_model(model):
        return None
    if input_args is None and not input_kwargs and input_size is None:
        from ._errors import InvalidArgumentError

        raise InvalidArgumentError(
            "summary() on a meta-built model needs input evidence: the "
            "zero-argument inference rung runs real probing forwards, which "
            "a weights-free model cannot serve.",
            code="weightsfree_facade_input_evidence_required",
            remedy=(
                "pass input_size= (declared shapes; e.g. input_size=(1, 8)) "
                "or explicit meta inputs (torch.zeros(..., device='meta'))"
            ),
            argument="input_size",
        )
    return _weightsfree_summary_facade(model, input_args, input_kwargs, input_size, summary_kwargs)


def _all_meta_model(model: nn.Module) -> bool:
    """Whether every registered parameter/buffer sits on the meta device.

    The rung-4 facade trigger (weightsfree memo D12): only an ALL-meta model
    may auto-select structure-only; a mixed or parameterless-real model takes
    the ordinary path (and mixed substrates refuse at the entry gate).
    """

    saw_state = False
    for tensor in list(model.parameters()) + list(model.buffers()):
        saw_state = True
        if not tensor.is_meta:
            return False
    return saw_state


def _weightsfree_summary_facade(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    input_size: Any,
    summary_kwargs: dict[str, Any],
) -> str:
    """Rung 4: the zero-payload-storage summary of a meta-built model.

    Resolves the input plan (declared ``input_size`` synthesizes shapes and
    converts the leaves to meta empties — shape/dtype without values; user
    inputs pass through unchanged), captures ONE structure-only trace via the
    same primitive as the explicit power path, records the auto-selection in
    the evidence envelope's input plan, and renders the summary under the
    hypothesis banner.
    """

    from .options import CaptureOptions as _CaptureOptions

    if input_size is not None:
        from .quickstart._resolve import resolve_inputs

        resolved = resolve_inputs(model, None, None, input_size, verb="summary")
        input_args = [
            torch.empty_like(leaf, device="meta") if isinstance(leaf, torch.Tensor) else leaf
            for leaf in resolved.plan.input_args
        ]
        input_kwargs = {
            key: (
                torch.empty_like(value, device="meta") if isinstance(value, torch.Tensor) else value
            )
            for key, value in dict(resolved.plan.input_kwargs).items()
        }
    trace = _user_funcs.trace(
        model,
        input_args,
        input_kwargs=input_kwargs or None,
        capture=_CaptureOptions(structure_only=True),
    )
    try:
        envelope = trace.structure_evidence
        if isinstance(envelope, dict) and isinstance(envelope.get("input_plan"), dict):
            envelope["input_plan"]["selection_source"] = "summary_facade_auto"
            if input_size is not None:
                envelope["input_plan"]["source"] = "declared_input_size"
                envelope["input_plan"]["synthesized"] = ["meta_input_leaves"]
        report = trace.summary(**summary_kwargs)
    finally:
        trace.cleanup()
    return _finalize_summary_report(report, "eval", "off")


def _finalize_summary_report(report: Any, execution_mode: str, grad_mode: str) -> str:
    """Suffix the execution disclosure, keeping the typed detached report.

    The report survives its Trace's cleanup by construction (C02, summary
    item 10: it retains neither the model nor the Trace).
    """

    from .report._summary_report import SummaryReport

    full_text = str(report) + "\n" + _summary_execution_note(execution_mode, grad_mode)
    if isinstance(report, SummaryReport):
        return SummaryReport(
            full_text, rows=report.rows, totals=report.totals, capture=report.capture
        )
    return full_text


def _summary_execution_note(execution_mode: str, grad_mode: str, *, short: bool = False) -> str:
    """Return the execution disclosure for one-call summaries.

    ``short=True`` yields the compact header form the rebuilt renderer
    hoists; the default is the historical trailing line, byte-stable.
    """

    if execution_mode == "eval":
        mode_part = "eval mode"
    elif execution_mode == "train":
        mode_part = "train mode (explicit; stateful layers may have updated buffers)"
    else:
        mode_part = "caller's module modes"
    grad_part = "no_grad" if grad_mode == "off" else "caller's grad context"
    if short:
        return f"{mode_part}, {grad_part}, state restored"
    return (
        f"Execution: one-call capture ran in {mode_part} under {grad_part}; "
        "module training flags and RNG state restored."
    )


def show_model_graph(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    view: VisModeLiteral | MissingType = MISSING,
    depth: int | MissingType = MISSING,
    renderer: VisRendererLiteral | MissingType = MISSING,
    layout: VisNodePlacementLiteral | MissingType = MISSING,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    module: Module | str | None = None,
    collapse: CollapseLiteral | MissingType = MISSING,
    fold_repeats: FoldRepeatsLiteral | MissingType = MISSING,
    order_siblings: bool | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    random_seed: int | None = None,
    verbose: bool = False,
    recurrence_detection: bool | MissingType = MISSING,
    visualization: VisualizationOptions | None = None,
) -> None:
    """Convenience wrapper: visualize the computational graph without saving outs.

    Runs an exhaustive forward pass (no outs saved) to discover the graph
    structure, renders the visualization, then cleans up the Trace.  For more
    control, use ``trace`` and call ``Trace.draw`` on the result directly.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    module:
        Optional module focus. Pass a Module or module address string to render
        only layers that ran inside that module.
    order_siblings:
        Whether Graphviz ``dot`` renders should verify and apply
        execution-order placement for true parallel siblings.
    code_panel:
        Optional source-code panel mode. ``True`` is equivalent to
        ``"forward"``. Built-in modes use source captured at log time; callable
        modes receive the live model object and are only available while that
        object is still alive.
    collapse:
        Smart module-collapse mode: ``"none"``, ``"auto"``, ``"max"``, or a
        float in ``[0.0, 1.0]``. Float levels follow the public monotone
        schedule.
    fold_repeats:
        Repeat-fold policy. ``None`` preserves the default policy. ``True`` folds
        every eligible repeated run. ``False`` disables run folding.
    random_seed:
        Fixed RNG seed for stochastic models. Reseeds the process-global RNG
        engines without restoring them; see ``capture.random_seed`` on ``tl.trace``.
    recurrence_detection:
        If True, run full isomorphic subgraph expansion. Set this to False when
        the forward pass has more than about 1M operations and postprocessing
        speed matters.
    visualization:
        Grouped visualization options. When omitted, ``show_model_graph``
        defaults to ``VisualizationOptions(view="unrolled")``.

    Returns
    -------
    None
        The graph is rendered for side effects.
    """
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if not input_kwargs:
        input_kwargs = {}
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)

    if recurrence_detection is MISSING:
        recurrence_detection = True
    recurrence_detection_enabled = bool(recurrence_detection)
    visualization_options = merge_visualization_options(
        function_default_mode="unrolled",
        visualization=visualization,
        view=view,
        depth=depth,
        renderer=renderer,
        layout=layout,
        node_style=node_style,
        collapse=collapse,
        fold_repeats=fold_repeats,
        order_siblings=order_siblings,
    )

    if visualization_options.view not in ["none", "rolled", "unrolled"]:
        raise ValueError("Visualization option must be either 'none', 'rolled', or 'unrolled'.")

    trace = _user_funcs._run_model_and_save_specified_outs(
        model=model,
        input_args=input_args,
        input_kwargs=input_kwargs,
        layers_to_save=None,
        activation_transform=None,
        mark_layer_depths=False,
        detach_saved_activations=False,
        save_grads=False,
        random_seed=random_seed,
        recurrence_detection=recurrence_detection_enabled,
        verbose=verbose,
    )
    # Render in a try/finally so temporary TorchLens metadata on the model is
    # always cleaned up, even if Graphviz rendering raises.
    try:
        render_kwargs = visualization_to_render_kwargs(visualization_options)
        if module is not None:
            from .data_classes.module import Module

            render_kwargs["module"] = module.address if isinstance(module, Module) else module
        if code_panel is not False:
            render_kwargs["code_panel"] = code_panel
        trace.draw(**render_kwargs)
    finally:
        trace.cleanup()


def draw_backward(
    trace: Trace,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    collapsed_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    node_style: VisNodeModeLiteral | MissingType = MISSING,
    code_panel: CodePanelOption = False,
    vis_mode: VisModeLiteral = "rolled",
    bwd: int | Iterable[int] | None = None,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing Trace's captured backward grad_fn_handle graph.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.recording_backward()``.
    node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    collapsed_node_spec_fn:
        Accepted for forward-visualization API symmetry. Not applied because
        backward graphs do not render collapsed module nodes.
    node_style:
        Node-style preset applied to grad_fn nodes.
    code_panel:
        Optional source-code panel mode.
    vis_mode:
        ``"rolled"`` renders one node per GradFn; ``"unrolled"`` renders one
        node per GradFnCall grouped by backward pass.
    bwd:
        Optional one-based backward pass number or numbers to render.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, graph overrides, and edge overrides are used.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    if visualization is None:
        container_path = "backward_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "topdown"
        graph_overrides = None
        edge_overrides = None
        node_mode: VisNodeModeLiteral = "default"
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides
        node_mode = visualization.node_style

    if node_style is not MISSING:
        node_mode = cast(VisNodeModeLiteral, node_style)

    return trace.draw_backward(
        vis_outpath=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        collapsed_node_spec_fn=collapsed_node_spec_fn,
        vis_node_mode=node_mode,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        vis_fileformat=file_format,
        vis_direction=direction,
        code_panel=code_panel,
        vis_mode=vis_mode,
        bwd=bwd,
    )


def draw_combined(
    trace: Trace,
    node_spec_fn: Callable[[Any, Any], Any] | None = None,
    backward_node_spec_fn: Callable[[Any, Any], Any] | None = None,
    vis_mode: VisModeLiteral = "unrolled",
    intervening_cluster: Literal["upstream", "outside", "downstream", "own"] = "upstream",
    show_buffer_layers: BufferVisibilityLiteral = "meaningful",
    bwd: int | Iterable[int] | None = None,
    visualization: VisualizationOptions | None = None,
) -> str:
    """Render an existing Trace's forward ops and backward grad_fns together.

    Parameters
    ----------
    trace:
        Trace with backward metadata captured by ``trace.log_backward(loss)``
        or ``trace.recording_backward()``.
    node_spec_fn:
        Optional callback receiving ``(layer_log, default_spec)``.
    backward_node_spec_fn:
        Optional callback receiving ``(grad_fn_handle, default_spec)``.
    vis_mode:
        Combined rendering currently supports only ``"unrolled"``.
    intervening_cluster:
        Placement mode for grad_fns without a corresponding forward op.
    show_buffer_layers:
        Buffer visibility mode for the forward side.
    bwd:
        Optional one-based backward pass number or numbers to render.
    visualization:
        Grouped visualization options. Only output path, save behavior, file
        format, direction, graph overrides, and edge overrides are used.

    Returns
    -------
    str
        Graphviz DOT source.
    """

    if visualization is None:
        container_path = "combined_modelgraph"
        save_only = False
        file_format = "pdf"
        direction: VisDirectionLiteral = "leftright"
        graph_overrides = None
        edge_overrides = None
    else:
        container_path = visualization.container_path
        save_only = visualization.save_only
        file_format = visualization.file_format
        direction = visualization.direction
        graph_overrides = visualization.graph_overrides
        edge_overrides = visualization.edge_overrides

    return trace.draw_combined(
        vis_outpath=container_path,
        vis_graph_overrides=graph_overrides,
        node_spec_fn=node_spec_fn,
        backward_node_spec_fn=backward_node_spec_fn,
        vis_edge_overrides=edge_overrides,
        vis_save_only=save_only,
        vis_fileformat=file_format,
        vis_direction=direction,
        vis_mode=vis_mode,
        intervening_cluster=intervening_cluster,
        show_buffer_layers=show_buffer_layers,
        bwd=bwd,
    )


def show_bundle_graph(
    bundle: Any,
    vis_outpath: str = "bundle_modelgraph",
    vis_mode: VisModeLiteral = "unrolled",
    direction: str = "forward",
    vis_direction: VisDirectionLiteral = "bottomup",
    vis_graph_overrides: dict[str, Any] | None = None,
    vis_node_overrides: dict[str, Any] | None = None,
    vis_edge_overrides: dict[tuple[str, str], Any] | None = None,
    vis_save_only: bool = False,
    vis_fileformat: str = "pdf",
) -> str | None:
    """Render a multi-trace bundle graph.

    Parameters
    ----------
    bundle:
        ``torchlens.Bundle`` instance.
    vis_outpath:
        Output path for Graphviz rendering.
    vis_mode:
        ``"rolled"``, ``"unrolled"``, or ``"none"``.
    direction:
        Graph content direction: ``"forward"``, ``"backward"``, ``"both"``, or
        ``"overlay"``.
    vis_direction:
        Graphviz layout direction.
    vis_graph_overrides:
        Graph-level Graphviz overrides.
    vis_node_overrides:
        Per-node Graphviz style overrides.
    vis_edge_overrides:
        Per-edge Graphviz style overrides keyed by ``(source, target)``.
    vis_save_only:
        If True, save without opening a viewer.
    vis_fileformat:
        Output file format.

    Returns
    -------
    str | None
        DOT source, or ``None`` when ``vis_mode='none'``.
    """

    if vis_mode == "none":
        return None
    if vis_mode not in {"rolled", "unrolled"}:
        raise ValueError("vis_mode must be 'rolled', 'unrolled', or 'none'.")
    if direction not in {"forward", "backward", "both", "overlay"}:
        raise ValueError("direction must be 'forward', 'backward', 'both', or 'overlay'.")

    import graphviz

    from .visualization._bundle_graph import (
        _add_bundle_backward_graph,
        _add_bundle_forward_edges,
        _add_bundle_forward_nodes,
    )
    from .visualization._render_utils import (
        direction_to_rankdir,
        render_dot_to_file,
        strip_known_extension,
    )

    dot = graphviz.Digraph(
        name="TorchLens_Bundle",
        comment="TorchLens bundle graph",
        format=vis_fileformat,
    )
    graph_attrs = {
        "rankdir": direction_to_rankdir(vis_direction),
        "label": f"TorchLens bundle graph ({vis_mode}, {direction})",
        "labelloc": "t",
        "labeljust": "left",
        "compound": "true",
    }
    graph_attrs.update({key: str(value) for key, value in (vis_graph_overrides or {}).items()})
    dot.graph_attr.update(graph_attrs)

    if direction in {"forward", "both", "overlay"}:
        _add_bundle_forward_nodes(dot, bundle, vis_mode, vis_node_overrides)
        _add_bundle_forward_edges(dot, bundle, vis_edge_overrides)
    if direction in {"backward", "both", "overlay"}:
        _add_bundle_backward_graph(dot, bundle)
    return render_dot_to_file(
        dot,
        strip_known_extension(vis_outpath),
        vis_fileformat,
        vis_save_only,
    )


def validate_forward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    backend: BackendName | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    Parameters
    ----------
    model:
        Model or callable to validate.
    input_args:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    random_seed:
        Fixed RNG seed for reproducibility. Reseeds the process-global RNG
        engines without restoring them; see ``tl.trace``'s ``random_seed``.
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True, also run metadata invariant checks.
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.

    Returns
    -------
    bool
        True if all validation checks pass, False otherwise.
    """

    spec = resolve_backend_spec(backend, model, input_args, input_kwargs)
    return spec.validate_entry(
        model,
        input_args,
        input_kwargs=input_kwargs,
        random_seed=random_seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
    )


def _restore_validation_replay_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    plain_attr_snapshot: _ModuleTreePlainAttrSnapshot | None,
) -> None:
    """Restore a validation model to its pre-capture replay state.

    Parameters
    ----------
    model:
        Model used for validation capture and replay.
    state_dict:
        Cloned state dictionary from before the traced validation forward.
    plain_attr_snapshot:
        Optional snapshot of plain Python attributes to restore.
    """

    # R07: both restores always run; a raising load_state_dict must not skip
    # the plain-attribute restore (first failure re-raises, later ones chain).
    with contextlib.ExitStack() as restores:
        if plain_attr_snapshot is not None:
            restores.callback(plain_attr_snapshot.restore_changed_attrs)
        # Wrapped-parameter modules (bitsandbytes) refuse their OWN state
        # dict through load_state_dict on CPU; the resilient restore falls
        # back to proven in-place restoration (lane F37).
        restores.callback(restore_state_dict_resilient, model, state_dict)


def _first_reproducibility_divergence(left: Trace, right: Trace) -> str | None:
    """Return a concise first-difference hint for two validation traces.

    Parameters
    ----------
    left:
        First validation trace.
    right:
        Second validation trace.

    Returns
    -------
    str | None
        Human-readable first divergence hint, or ``None`` when no cheap hint is
        available.
    """

    for index, (left_layer, right_layer) in enumerate(zip(left.layer_list, right.layer_list)):
        left_sig = (
            getattr(left_layer, "layer_type", None),
            getattr(left_layer, "func_name", None),
            len(getattr(left_layer, "parents", ()) or ()),
            bool(getattr(left_layer, "is_output", False)),
            bool(getattr(left_layer, "is_buffer", False)),
        )
        right_sig = (
            getattr(right_layer, "layer_type", None),
            getattr(right_layer, "func_name", None),
            len(getattr(right_layer, "parents", ()) or ()),
            bool(getattr(right_layer, "is_output", False)),
            bool(getattr(right_layer, "is_buffer", False)),
        )
        if left_sig != right_sig:
            return (
                f"first divergence at op index {index}: "
                f"{getattr(left_layer, 'func_name', None)!r} vs "
                f"{getattr(right_layer, 'func_name', None)!r}"
            )
    if len(left.layer_list) != len(right.layer_list):
        return f"op count changed: {len(left.layer_list)} vs {len(right.layer_list)}"
    return None


def _warn_if_validation_trace_not_reproducible(
    first_trace: Trace,
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any],
    random_seed: int,
) -> Literal["matched", "mismatch", "unavailable"]:
    """Warn when a validation trace changes after one fresh re-trace.

    Parameters
    ----------
    first_trace:
        Trace captured from the model's pre-validation state.
    model:
        Validation model after the first traced forward has run.
    input_args:
        Positional inputs for the second capture.
    input_kwargs:
        Keyword inputs for the second capture.
    random_seed:
        Seed reused for the second capture to avoid RNG-only graph drift.

    Returns
    -------
    Literal["matched", "mismatch", "unavailable"]
        ``"matched"`` when the fresh re-trace is structurally identical,
        ``"mismatch"`` when the graphs diverge, and ``"unavailable"`` when the
        fresh re-trace check itself cannot be completed.
    """

    second_trace: Trace | None = None
    try:
        # Buffer-source identity is assigned during postprocessing only when the
        # relevant activations are saved. Match the validation trace's "all"
        # selection so the structural comparison does not compare two capture modes.
        # r33 F-2: the first trace is captured under a FORCED "shadow"
        # completeness-witness mode (validate_forward_pass), so the re-trace
        # must run under the same mode -- comparing a shadow capture against an
        # ambient-mode capture is exactly the two-capture-modes comparison this
        # check must not make, and it deterministically false-FAILED every
        # witness-mode-sensitive model as "stateful/non-reproducible".
        from . import _state

        prior_witness_mode = _state._completeness_witness_mode
        _state._completeness_witness_mode = "shadow"
        try:
            second_trace = _user_funcs._run_model_and_save_specified_outs(
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                layers_to_save="all",
                activation_transform=None,
                mark_layer_depths=False,
                detach_saved_activations=False,
                save_grads=False,
                save_arg_values=False,
                random_seed=random_seed,
                save_rng_states=False,
            )
        finally:
            _state._completeness_witness_mode = prior_witness_mode
        first_hash = compute_graph_shape_hash(first_trace, include_module_address=False)
        second_hash = compute_graph_shape_hash(second_trace, include_module_address=False)
        if first_hash == second_hash:
            return "matched"
        hint = _first_reproducibility_divergence(first_trace, second_trace)
        hint_suffix = f" ({hint})" if hint is not None else ""
        message = (
            "TorchLens validation detected a stateful/non-reproducible model: "
            "fresh re-trace produced a structurally different graph. The trace "
            "may represent a one-time execution. Re-run from fresh model state or "
            f"make the forward path state-independent{hint_suffix}."
        )
        from .validation.diagnostics import ValidationDiagnostic, record_validation_diagnostic

        record_validation_diagnostic(
            first_trace,
            ValidationDiagnostic(
                check="trace_retrace_structure_mismatch",
                message=message,
                extra={
                    "first_graph_hash": first_hash,
                    "retrace_graph_hash": second_hash,
                    "first_op_count": len(first_trace.layer_list),
                    "retrace_op_count": len(second_trace.layer_list),
                    "first_divergence": hint,
                },
            ),
        )
        warnings.warn(
            TraceNotReproducibleWarning(
                message,
                first_graph_hash=first_hash,
                retrace_graph_hash=second_hash,
                first_op_count=len(first_trace.layer_list),
                retrace_op_count=len(second_trace.layer_list),
                first_divergence=hint,
            ),
            stacklevel=2,
        )
        return "mismatch"
    except Exception as exc:
        message = (
            "TorchLens validation could not run the fresh re-trace reproducibility "
            f"check ({type(exc).__name__}: {exc})."
        )
        from .validation.diagnostics import ValidationDiagnostic, record_validation_diagnostic

        record_validation_diagnostic(
            first_trace,
            ValidationDiagnostic(
                check="trace_retrace_unavailable",
                message=message,
                extra={"exception_type": type(exc).__name__},
            ),
        )
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=2,
        )
        return "unavailable"
    finally:
        if second_trace is not None:
            second_trace.cleanup()


def _downgrade_retrace_mismatch_to_unverified(trace: Trace) -> None:
    """Convert a replay pass into an honest unverified status after retrace drift.

    Parameters
    ----------
    trace:
        Validation trace whose pristine re-trace diverged structurally.
    """

    from .validation.status import ValidationReplayStatus

    current_status = trace.validation_replay_status
    unverified_reason_counts = dict(current_status.unverified_reason_counts)
    unverified_reason_counts["trace_retrace_structure_mismatch"] = (
        unverified_reason_counts.get("trace_retrace_structure_mismatch", 0) + 1
    )
    setattr(
        trace,
        "_validation_replay_status",
        ValidationReplayStatus.unverified(
            backend=current_status.backend,
            source=current_status.source,
            reason="trace_retrace_structure_mismatch",
            message=(
                "Replay validation matched the captured trace, but the pristine "
                "re-trace diverged structurally, so the overall validation "
                "result is unverified."
            ),
            replayed_node_count=current_status.replayed_node_count,
            unverified_node_count=max(1, current_status.unverified_node_count),
            payload_load_status=current_status.payload_load_status,
            pure_unverified_node_count=current_status.pure_unverified_node_count,
            effect_region_node_count=current_status.effect_region_node_count,
            failed_node_count=current_status.failed_node_count,
            unverified_reason_counts=unverified_reason_counts,
            exempted_reason_counts=current_status.exempted_reason_counts,
            decisions=current_status.decisions,
        ),
    )


def _refuse_validation_precondition(
    message: str,
    trace_observer: Callable[[Trace | None], None] | None = None,
) -> Literal[False]:
    """Warn, record a structured precondition-refusal diagnostic, and refuse.

    ``_validate_forward_pass_torch`` has several early-exit sites (an
    unreproducible input topology, an unsnapshotable plain attribute, a
    non-pristine ground truth, a dropped-output enumeration defect) that
    return bare ``False`` before Step 2 builds a ``Trace``. A ``Trace``-scoped
    :class:`~.validation.diagnostics.ValidationFailure` (the mechanism every
    later mismatch uses) has nothing to attach to at these sites, so a caller
    reading only ``get_validation_failure(trace)`` saw nothing and fell back to
    an uninformative ``repr(False)`` (menagerie's ``convit_*`` "replay failed
    (False)" rows). Recording with ``trace=None`` makes the reason reachable
    through ``get_validation_failure(None)`` / ``last_validation_failure()``;
    invoking the caller's ``_trace_observer`` (with ``trace=None``, which every
    known observer already treats as "nothing to summarize yet", the same as
    a trace that never ran) at the SAME early-exit site it would otherwise
    never see makes that reason reach an observer-based caller too, not just
    one that happens to poll ``get_validation_failure`` directly.

    Parameters
    ----------
    message:
        Human-readable refusal reason, also emitted as a ``RuntimeWarning``.
    trace_observer:
        The caller's optional ``_trace_observer``, forwarded ``None`` in
        place of a ``Trace`` so it still runs at this early exit.

    Returns
    -------
    Literal[False]
        Always ``False``, for ``return _refuse_validation_precondition(...)``.
    """

    warnings.warn(message, RuntimeWarning, stacklevel=3)
    from .validation.diagnostics import (
        CHECK_PRECONDITION,
        ValidationFailure,
        record_validation_failure,
    )

    record_validation_failure(None, ValidationFailure(check=CHECK_PRECONDITION, message=message))
    if trace_observer is not None:
        trace_observer(None)
    return False


def _validate_forward_pass_torch(
    model: nn.Module | Callable[..., Any],
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    random_seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    num_threads: int | None = None,
    _trace_observer: Callable[[Trace | None], None] | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() on PRISTINE torch to get ground-truth output
       tensors: if torchlens wrappers are installed, they are removed for
       this one forward (restored from the pre-decoration originals ledger)
       and reinstalled afterwards, so the ground truth never observes
       through the wrapper layer it is meant to check (R75-1). If they
       cannot be removed (a capture is active), validation refuses with
       ``False`` rather than blessing a wrap-state-dependent ground truth.
    2. Run ``trace`` with ``save_arg_values=True`` and ``layers_to_save='all'``
       to capture every out and its creating function's arguments.
    3. Call ``Trace.validate_forward_pass`` which replays the forward pass
       layer-by-layer from saved outs, checking that the output matches
       ground truth.  It also injects random outs and verifies the output
       changes (proving the saved outs are actually used, not just ignored).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_arg_values=True is required:**  The validation replay re-executes
    each function using its saved non-tensor arguments (e.g., stride, padding for
    conv2d).  Without them, replay cannot reconstruct the correct computation.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    random_seed:
        Fixed RNG seed for reproducibility (auto-generated if None). Reseeds
        the process-global RNG engines without restoring them; see
        ``tl.trace``'s ``random_seed``.
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True (default), also run metadata invariant checks.
    num_threads:
        Optional intra-op thread count for the validation forwards. ``None``
        preserves the process default; an integer pins for this harness call and
        restores the previous thread count afterward.
    _trace_observer:
        Optional private callback invoked with the completed validation trace
        after replay validation and before cleanup. Also invoked with
        ``None`` at an earlier CHECK_PRECONDITION refusal site (an
        unreproducible input topology, an unsnapshotable plain attribute, a
        non-pristine ground truth, a dropped-output enumeration defect) that
        returns before any ``Trace`` exists, so a caller is never left with
        no observer call at all and an uninformative bare ``False``.

    Returns
    -------
    bool
        True if all validation checks pass, False otherwise.
    """
    warn_parallel()
    from .validation.diagnostics import reset_validation_failure

    # Clear any stale precondition-refusal failure from a prior call on this
    # process (CHECK_PRECONDITION sites below have no Trace to reset against,
    # unlike the Trace-scoped reset in core.validate_saved_outs).
    reset_validation_failure(None)
    # F41 bound-method roots: validation captures through the same ruled root
    # contract as tl.trace -- a bound method of an nn.Module wraps into the
    # TL-authored synthetic root before the module-shaped preflights run;
    # other callables refuse typed (previously a bare AttributeError leak).
    from ._errors import InvalidArgumentError
    from .backends.torch.bound_root import TLBoundMethodRoot, is_bound_method_of_module

    if not isinstance(model, nn.Module):
        if not is_bound_method_of_module(model):
            raise InvalidArgumentError(
                f"Unsupported model type for validation: received {type(model).__name__}, "
                "not a torch.nn.Module or a bound method of one",
                code="model_type_unsupported",
                remedy=(
                    "pass a torch.nn.Module, or a bound method of an nn.Module "
                    "(owner resolved via method.__self__)"
                ),
                argument="model",
                received_type=type(model).__name__,
            )
        model = TLBoundMethodRoot(model)
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    set_random_seed(random_seed)
    input_args = normalize_input_args(input_args, model)
    if not input_kwargs:
        input_kwargs = {}
    # Deep-copy inputs so the ground-truth forward pass doesn't mutate the
    # originals (some models modify inputs in-place).
    input_args_copy, input_kwargs_copy, input_copy_gaps = safe_copy_input_tree(
        input_args,
        input_kwargs,
    )
    if input_copy_gaps:
        return _refuse_validation_precondition(
            "TorchLens validation cannot reproduce the caller's input topology: "
            f"{input_copy_gaps!r}. Returning False rather than validating altered semantics.",
            _trace_observer,
        )

    # A META first parameter never pins the input device: offload-hooked
    # models (accelerate device_map / cpu/disk offload, lane F37) hold meta
    # params between forwards and their hooks place inputs themselves; the
    # first NON-meta param still pins for mixed dispatch.
    model_device = next((p.device for p in model.parameters() if p.device.type != "meta"), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward.
    state_dict = _clone_state_dict_with_metadata(model)
    trace: Trace | None = None
    outs_are_valid = False
    # Determinism stabilizer for the capture + replay + perturbation region.
    #
    # The replay-validation drift between an op's value captured INLINE in the
    # full forward and the value recomputed by ISOLATED per-op replay is, for
    # most ops, float-reduction-ORDER non-determinism in parallel CPU kernels
    # (multi-threaded conv/matmul, atomic-add scatter) -- NOT randomness (RNG is
    # already seeded above). Forcing deterministic algorithms makes the
    # ground-truth forward, the TorchLens capture, and the per-op replay use the
    # same reduction order, which removes the nondeterministic in-place-scatter
    # PERTURBATION flake (the GNN/molecular "regression" class: a wrong value
    # injected into a scatter destination sometimes produced an output
    # indistinguishable from the original under a thread race, spuriously failing
    # the sensitivity check). warn_only=True so no op raises if it lacks a
    # deterministic impl (CPU scatter_add IS deterministic in torch 2.8, so this
    # is not even exercised there, but it keeps the stabilizer safe on any op).
    #
    # In addition to deterministic algorithms, callers may pass ``num_threads``
    # to PIN an intra-op thread count (save/restore, scoped to this harness only)
    # for the duration of the validation forwards. Multi-threaded float
    # reduction-ORDER is non-deterministic ACROSS RUNS even with deterministic
    # algorithms enabled: two CLEAN forwards of the SAME nn.Module can disagree
    # by ~3e-7 at the output. That straddles the strict phase-0 ground-truth bar
    # (GROUND_TRUTH_OUTPUT_RTOL=1e-6), making the spectral-GCN family (MSTGCN /
    # TGT-MSTGCN Chebyshev sparse aggregation) FLAKY, and the same thread
    # non-determinism in MoE masked-gate routing makes the perturbation
    # sensitivity check FLAKY (minimax / nllb-moe). Pinning one thread makes both
    # bit-exact (abs diff -> exactly 0.0), so the strict bar becomes DETERMINISTIC
    # -- this does NOT loosen any tolerance, it removes the inter-run thread
    # non-determinism the bar was never meant to police.
    #
    # Downstream catalog validators (the Model Menagerie) run this harness at the
    # worker process default thread count for throughput, then retry exactly the failed forward
    # validation once with ``num_threads=1``. A genuine capture/replay bug still
    # fails the single-thread retry; the known reduction-order flakes are rescued
    # by a strict bit-exact rerun instead of by loosening any tolerance.
    #
    # LOAD-BEARING: when a deterministic retry is needed, pinning at PROCESS start
    # does NOT reliably fix the validation path (the capture forward and/or model
    # internals re-parallelize); the pin must wrap the forwards INSIDE the
    # harness, which is what ``num_threads=1`` does.
    #
    # Deterministic algorithms remain on as well: the optional thread pin removes
    # inter-run reduction-order drift, while deterministic algorithms remove
    # intra-run scatter non-determinism. (The single-thread pin alone covers most
    # of it, but keeping deterministic algorithms is strictly safer and free
    # here.) The thread pin does NOT make deep CPU conv inline-vs-isolated replay
    # bit-exact (that residual is oneDNN kernel selection, not threading) -- that
    # is what the band-C reduction-depth tolerance in validation/core.py covers;
    # the changes are complementary.
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_num_threads = torch.get_num_threads()
    try:
        # R07-2 (install-move half): both process-global installs run INSIDE
        # the restoring try -- a raising set_num_threads used to strand the
        # already-flipped deterministic stance for the life of the process
        # (the finally below never ran). Restoring to the just-snapshotted
        # priors is idempotent when an install never landed.
        #
        # Skip forcing it on torch builds whose CPU fill_empty_deterministic_
        # kernel does not cover Float8 (HAS_CPU_FLOAT8_DETERMINISTIC_FILL):
        # the kernel is missing outright there, so warn_only=True cannot help,
        # and forcing crashed any validation that allocates a fresh Float8
        # CPU tensor.
        from .utils._torch_compat import get_cpu_float8_deterministic_fill_support

        if get_cpu_float8_deterministic_fill_support():
            torch.use_deterministic_algorithms(True, warn_only=True)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        ground_truth_model, plain_attr_snapshot = _model_for_ground_truth_validation(model)
        if plain_attr_snapshot is not None and not plain_attr_snapshot.is_complete:
            return _refuse_validation_precondition(
                "TorchLens validation cannot prove model-state restoration after deepcopy "
                "failed because these plain attributes are unsupported: "
                f"{plain_attr_snapshot.unsupported_attr_paths!r}. Returning False rather "
                "than reporting unverified success.",
                _trace_observer,
            )
        from ._errors import CaptureContextError
        from .backends.torch.ops import _walk_output_tensors_with_paths
        from .validation._pristine import pristine_torch_oracle

        # R75-1: the phase-0 ground truth must observe through PRISTINE
        # torch. In a wrapped process this forward used to run through the
        # installed pass-through wrapper shells -- the same closures capture
        # and replay observe through -- so a wrapper-layer numeric
        # distortion validated clean whenever any capture had run earlier
        # in the process (probe-proven). The wrappers are removed for this
        # one forward and reinstalled after; if they cannot be removed (a
        # capture is active), validation REFUSES rather than blessing a
        # wrap-state-dependent ground truth.
        try:
            with pristine_torch_oracle():
                ground_truth_output = ground_truth_model(*input_args_copy, **input_kwargs_copy)
        except CaptureContextError:
            return _refuse_validation_precondition(
                "TorchLens validation could not compute a pristine-torch ground "
                "truth (a capture is active in this process, or the unwrap "
                "ledger is poisoned); the verdict would depend on the wrapper "
                "installation it is meant to check. Returning False rather "
                "than reporting unverified success.",
                _trace_observer,
            )
        ground_truth_output_all = [
            (tensor, tuple(path))
            for tensor, path, _container_spec in _walk_output_tensors_with_paths(
                ground_truth_output
            )
        ]
        if not ground_truth_output_all:
            ground_truth_output_all = get_vars_of_type_from_obj(
                ground_truth_output,
                torch.Tensor,
                search_depth=5,
                return_addresses=True,
                allow_repeats=True,
            )
        # b9 R74/75-1: ground-truth enumeration SHARED ITS ROOT with capture
        # (both resolve through the one backends walker), so a walker defect
        # dropped the same output leaf from both sides and validation blessed
        # a missing output. Cross-check the adapter's enumeration against the
        # validation-owned independent traversal; a disagreement is a capture
        # (or adapter) bug and must FAIL validation, never pass silently.
        from .validation._output_walk import independent_output_tensor_ids

        adapter_leaf_ids = {id(entry[0]) for entry in ground_truth_output_all}
        independent_leaf_ids = set(independent_output_tensor_ids(ground_truth_output))
        missed_by_adapter = independent_leaf_ids - adapter_leaf_ids
        # Direction matters: the adapter legitimately sees MORE than the
        # generic walk (registered custom containers, opaque structseq
        # internals), and more-than can never hide a dropped output. Leaves
        # the independent walk found that the adapter MISSED are exactly the
        # dropped-output defect class.
        if missed_by_adapter:
            return _refuse_validation_precondition(
                "TorchLens validation found a ground-truth output-enumeration "
                f"defect: {len(missed_by_adapter)} tensor leaf(ves) reachable in "
                "the model output are missing from the capture-side walker's "
                "enumeration. Validation fails rather than validating against "
                "the same defective enumeration.",
                _trace_observer,
            )
        # Deduplicate by structural address to match how capture/trace.py extracts
        # outputs (same tensor returned in multiple positions is counted once).
        addresses_used = []
        ground_truth_output_tensors = []
        for entry in ground_truth_output_all:
            if entry[1] in addresses_used:
                continue
            # Clone/detach the ground-truth output BEFORE restoring state_dict below.
            # When the model returns a registered buffer directly (e.g. `return self.h`
            # after `self.h = ...`), the output tensor IS the live buffer object;
            # `model.load_state_dict` writes buffers in-place, which would clobber this
            # saved ground-truth reference back to its initial value and produce a
            # validation FALSE-NEGATIVE. (Inputs are already deep-copied above; outputs
            # were not.) Snapshotting the value here corrects the ground truth fed to the
            # tripwire — it does NOT weaken any check.
            ground_truth_output_tensors.append(entry[0].detach().clone())
            addresses_used.append(entry[1])
        restore_state_dict_resilient(model, state_dict)
        if plain_attr_snapshot is not None:
            plain_attr_snapshot.restore_changed_attrs()

        validation_model, validation_plain_attr_snapshot, validation_model_copied = (
            _model_for_validation_replay(model)
        )
        if (
            validation_plain_attr_snapshot is not None
            and not validation_plain_attr_snapshot.is_complete
        ):
            return _refuse_validation_precondition(
                "TorchLens validation cannot prove replay-state restoration after deepcopy "
                "failed because these plain attributes are unsupported: "
                f"{validation_plain_attr_snapshot.unsupported_attr_paths!r}. Returning False "
                "rather than reporting unverified success.",
                _trace_observer,
            )
        validation_state_dict = _clone_state_dict_with_metadata(validation_model)
        (
            validation_input_args,
            validation_input_kwargs,
            validation_input_gaps,
        ) = safe_copy_input_tree(input_args, input_kwargs)
        (
            reproducibility_input_args,
            reproducibility_input_kwargs,
            reproducibility_input_gaps,
        ) = safe_copy_input_tree(input_args, input_kwargs)
        if validation_input_gaps or reproducibility_input_gaps:
            copy_gaps = validation_input_gaps + reproducibility_input_gaps
            return _refuse_validation_precondition(
                "TorchLens validation cannot reproduce the caller's input topology: "
                f"{copy_gaps!r}. Returning False rather than validating altered semantics.",
                _trace_observer,
            )
        if model_device is not None:
            validation_input_args = _move_tensors_to_device(validation_input_args, model_device)
            validation_input_kwargs = _move_tensors_to_device(validation_input_kwargs, model_device)
            reproducibility_input_args = _move_tensors_to_device(
                reproducibility_input_args, model_device
            )
            reproducibility_input_kwargs = _move_tensors_to_device(
                reproducibility_input_kwargs, model_device
            )

        # Step 2: Run the model *through* TorchLens, saving all outs.
        # save_arg_values=True is essential - the replay needs each function's
        # non-tensor arguments to re-execute the computation from saved outs.
        from . import _state

        prior_witness_mode = _state._completeness_witness_mode
        _state._completeness_witness_mode = "shadow"
        try:
            trace = _user_funcs._run_model_and_save_specified_outs(
                model=validation_model,
                input_args=validation_input_args,
                input_kwargs=validation_input_kwargs,
                layers_to_save="all",
                activation_transform=None,
                mark_layer_depths=False,
                detach_saved_activations=False,
                save_grads=False,
                save_arg_values=True,
                random_seed=random_seed,
                save_rng_states=True,
            )
        finally:
            _state._completeness_witness_mode = prior_witness_mode
        from .validation._completeness_backstop import completeness_backstop_counts

        (
            trace._validation_dispatch_op_count,
            trace._validation_captured_dispatchable_op_count,
        ) = completeness_backstop_counts(trace)
        retrace_outcome: Literal["matched", "mismatch", "unavailable"] = "unavailable"
        if validation_model_copied:
            retrace_outcome = _warn_if_validation_trace_not_reproducible(
                trace,
                validation_model,
                reproducibility_input_args,
                reproducibility_input_kwargs,
                random_seed,
            )
        else:
            from .validation.diagnostics import ValidationDiagnostic, record_validation_diagnostic

            record_validation_diagnostic(
                trace,
                ValidationDiagnostic(
                    check="trace_retrace_pristine_copy_unavailable",
                    message=(
                        "TorchLens validation skipped the pristine trace-vs-retrace check "
                        "because the model could not be deep-copied."
                    ),
                    extra={"model_type": f"{type(model).__module__}.{type(model).__qualname__}"},
                ),
            )
        _restore_validation_replay_state(
            validation_model,
            validation_state_dict,
            validation_plain_attr_snapshot,
        )
        # Step 3: Validate by replaying the forward pass from saved outs.
        # validate_saved_outs resets the diagnostics ledger at entry so a
        # DIRECT repeat call reports this-run evidence only (B8-43); the
        # retrace diagnostics recorded above belong to THIS flow, so they are
        # snapshotted and re-prepended after the replay run.
        from .validation.diagnostics import (
            MAX_VALIDATION_DIAGNOSTICS,
            TRACE_DIAGNOSTICS_ATTR,
            get_validation_diagnostics,
        )

        flow_diagnostics = get_validation_diagnostics(trace)
        validation_result = trace.validate_forward_pass(
            ground_truth_output_tensors, verbose, validate_metadata=validate_metadata
        )
        if flow_diagnostics:
            merged = flow_diagnostics + get_validation_diagnostics(trace)
            setattr(trace, TRACE_DIAGNOSTICS_ATTR, merged[:MAX_VALIDATION_DIAGNOSTICS])
        if retrace_outcome == "mismatch":
            _downgrade_retrace_mismatch_to_unverified(trace)
            outs_are_valid = False
            from .validation.diagnostics import (
                CHECK_RETRACE_MISMATCH,
                ValidationFailure,
                record_validation_failure,
            )

            retrace_diagnostic = next(
                (
                    diagnostic
                    for diagnostic in get_validation_diagnostics(trace)
                    if diagnostic.check == "trace_retrace_structure_mismatch"
                ),
                None,
            )
            record_validation_failure(
                trace,
                ValidationFailure(
                    check=CHECK_RETRACE_MISMATCH,
                    message=(
                        retrace_diagnostic.message
                        if retrace_diagnostic is not None
                        else (
                            "pristine re-trace diverged structurally after a "
                            "replay that otherwise passed"
                        )
                    ),
                    extra=dict(retrace_diagnostic.extra) if retrace_diagnostic is not None else {},
                ),
            )
        elif isinstance(validation_result, bool):
            outs_are_valid = validation_result
        else:
            outs_are_valid = bool(getattr(validation_result, "passed", False))
        if _trace_observer is not None:
            _trace_observer(trace)
    finally:
        # R07: per-step fenced teardown. One raising restore (determinism
        # flag, thread count, state_dict, plain attrs, trace cleanup) must not
        # skip the later steps -- the pre-fix straight-line block left user
        # model params unrestored and wrapper-session state uncleaned when an
        # early restore raised. ExitStack runs EVERY callback and re-raises
        # the first failure (later failures chain); callbacks are pushed in
        # reverse so execution keeps the original step order.
        with contextlib.ExitStack() as teardown:
            if trace is not None:
                teardown.callback(trace.cleanup)
            if "plain_attr_snapshot" in locals() and plain_attr_snapshot is not None:
                teardown.callback(plain_attr_snapshot.restore_changed_attrs)
            teardown.callback(restore_state_dict_resilient, model, state_dict)
            if num_threads is not None:
                teardown.callback(torch.set_num_threads, prior_num_threads)
            teardown.callback(
                torch.use_deterministic_algorithms,
                prior_deterministic,
                warn_only=prior_warn_only,
            )
    return outs_are_valid


def validate_backward_pass(
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    validate_metadata: bool = True,
    random_seed: int | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    validate_layer_grads: bool = True,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
) -> bool:
    """Validate first-class backward capture against stock autograd.

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Positional args for ``model.forward()``.
    input_kwargs:
        Keyword args for ``model.forward()``.
    loss_fn:
        Optional callable mapping model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    random_seed:
        Fixed RNG seed for stock and candidate passes. Reseeds the
        process-global RNG engines without restoring them; see
        ``tl.trace``'s ``random_seed``.
    atol:
        Absolute allclose tolerance. ``None`` (default) derives the
        tolerance per gradient dtype (R13); the historical fp32 decimal
        pair applied to every dtype was ~4.5e11 fp64 ULPs loose and
        false-failed fp16 grads.
    rtol:
        Relative allclose tolerance. ``None`` (default) derives per
        gradient dtype, matching ``torchlens.validation.backward``.
    validate_layer_grads:
        If True (default), also validate captured per-module-output gradients.
    layer_grad_atol:
        Optional layer-gradient absolute tolerance.
    layer_grad_rtol:
        Optional layer-gradient relative tolerance.

    Returns
    -------
    bool
        True if backward capture matches stock autograd.
    """
    from .validation.backward import validate_backward_pass as _impl

    return _impl(
        model,
        input_args,
        input_kwargs=input_kwargs,
        loss_fn=loss_fn,
        validate_metadata=validate_metadata,
        random_seed=random_seed,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )


def validate_batch_of_models_and_inputs(
    models_and_inputs_dict: dict[str, dict[str, Any]],
    out_path: str,
    redo_model_if_already_run: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Batch-validate multiple models, writing incremental results to a CSV.

    For each model/input pair, calls ``validate_forward_pass`` and appends the
    result to a running CSV at *out_path*.  If the CSV already exists, previously
    validated models can be skipped (controlled by *redo_model_if_already_run*).

    Parameters

    ----------
        models_and_inputs_dict: Mapping of model_class_name to a dict with keys:
            - ``model_category`` (str): grouping label (e.g. 'torchvision').
            - ``model_loading_func`` (callable): zero-arg function returning an nn.Module.
            - ``model_sample_inputs`` (dict[str, input]): named sample inputs.
        out_path: File path for the results CSV (created if absent, appended otherwise).
        redo_model_if_already_run: Re-validate models already present in the CSV.
        show_progress: Show the tqdm bar and per-model status lines. Pass False
            for quiet batch runs (b8 B8-38: the bar was ungated and a bare print
            inside the loop corrupted the live bar).

    Returns

    -------
        DataFrame with columns: model_category, model_class_name, input_name, validation_success.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e

    if os.path.exists(out_path):
        current_csv = pd.read_csv(out_path)
    else:
        current_csv = pd.DataFrame.from_dict(
            {
                "model_category": [],
                "model_class_name": [],
                "input_name": [],
                "validation_success": [],
            }
        )
    models_already_run = current_csv["model_class_name"].unique()
    progress = tqdm(
        models_and_inputs_dict.items(), desc="Validating models", disable=not show_progress
    )
    for model_class_name, model_info in progress:
        # Route the status line through the bar so it never corrupts it.
        progress.set_postfix_str(model_class_name)
        if model_class_name in models_already_run and not redo_model_if_already_run:
            continue
        model_category = model_info["model_category"]
        model_loading_func = model_info["model_loading_func"]
        model = model_loading_func()
        model_sample_inputs = model_info["model_sample_inputs"]
        for input_name, input_data in model_sample_inputs.items():
            validation_success = validate_forward_pass(model, input_data)
            current_csv = pd.concat(
                [
                    current_csv,
                    pd.DataFrame(
                        [
                            {
                                "model_category": model_category,
                                "model_class_name": model_class_name,
                                "input_name": input_name,
                                "validation_success": validation_success,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        current_csv.to_csv(out_path, index=False)
        del model
    return current_csv
