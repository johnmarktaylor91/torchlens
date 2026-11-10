"""Private implementations for public user-facing utility commands.

Modules below the public facade that need a public verb (``trace``,
``release_model``, ``summary``, the ``validate_*`` family) import it from
here, never from ``torchlens`` or ``torchlens._user_funcs``: this module is
the body of those verbs, so it is the one settle-layer door to them.
"""

from __future__ import annotations

import contextlib
import random
import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from torch import nn

from . import _user_funcs as _user_funcs, _validation_argument_references as _arg_references
from ._capture_state_helpers import (
    _model_for_ground_truth_validation,
    _model_for_validation_replay,
    _ModuleTreePlainAttrSnapshot,
    _move_tensors_to_device,
    _reject_opaque_wrappers,
    _unwrap_data_parallel,
    unwrap_compiled_model,
)
from ._data_classes.trace import Trace
from ._deploy_env import restore_state_dict_resilient
from ._errors import CaptureNotActiveError, InvalidArgumentError
from ._input_coerce import _coerce_inputs
from ._literals import (
    OutputDeviceLiteral,
)
from ._robustness import check_model_and_input_variants
from ._save_budget import SaveBudgetOption
from ._utils.arg_handling import normalize_input_args, safe_copy_input_tree
from ._utils.display import warn_parallel
from ._utils.input_leaves import leaf_preserving_input_copies
from ._utils.introspection import get_vars_of_type_from_obj
from ._utils.rng import restores_rng_streams, set_random_seed
from ._validation_entry_options import (
    ForwardValidationEntryOptions,
    resolve_forward_validation_entry,
)
from ._validation_retrace_saves import compare_retrace_graphs, retrace_save_indices
from ._validation_state_snapshot import (
    COPY_FAILED_CODE,
    StateSnapshot,
    restore_refusal_message,
    restore_snapshot_if_changed,
    snapshot_state_for_restore,
    state_matches_snapshot,
    with_storage_bindings_of,
)
from .backends.registry import BackendName
from .backends.torch.bound_root import TLBoundMethodRoot, is_bound_method_of_module
from .errors import TraceNotReproducibleWarning
from .options import CaptureOptions, CaptureOptions as _CaptureOptions

# FACADE by role: the body of the public ``torchlens.*`` verbs, whose upward
# calls into the renderers and reports are the facade's own (Rule F).
__tl_layer__ = "FACADE"


def trace(*args: Any, **kwargs: Any) -> Trace:
    """Capture through the public ``trace`` entry, for modules below the facade.

    The call binds late through ``torchlens._user_funcs`` (nothing is copied
    here), so a patch of ``torchlens._user_funcs.trace`` is seen while it is
    in place and gone once it is undone, exactly as for the other verbs in
    this module.

    Parameters
    ----------
    *args, **kwargs:
        Forwarded unchanged to :func:`torchlens.trace`.

    Returns
    -------
    Trace
        The captured trace.
    """

    return _user_funcs.trace(*args, **kwargs)


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

    Equivalent to ``trace(model, input_args, input_kwargs=input_kwargs,
    save=None, capture=CaptureOptions(compute_input_output_distances=True))``.

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
        input_kwargs=input_kwargs,
        capture=CaptureOptions(
            compute_input_output_distances=True,
        ),
        save=None,
    )
    return model_trace


def summary(  # noqa: PLR0913 -- ladder-conjugated public verb: the three input rungs plus the two execution dials ARE the spec'd surface (F17 B11)
    model: nn.Module | str,
    inputs: torch.Tensor | list[Any] | tuple[Any, ...] | None = None,
    input_kwargs: dict[Any, Any] | None = None,
    *,
    input_shape: Any | None = None,
    execution_mode: Literal["eval", "train", "same"] = "eval",
    grad_mode: Literal["off", "same"] = "off",
    **summary_kwargs: Any,
) -> str:
    """Run a metadata-only forward pass and return a rendered summary report.

    The one-call summary is SAFE by default: the captured forward runs in
    eval mode under ``torch.no_grad()``, and every module training flag plus
    the host/device RNG state is restored bit-identically afterwards -- a
    summary call never mutates the model (BatchNorm running stats included)
    or advances the caller's RNG streams. The function RETURNS the result
    and never auto-prints (the repr self-displays).

    Give exactly one of: a real ``inputs``, ``input_shape=``, or
    nothing. ``input_shape=``
    synthesizes seeded tensors per the quickstart grammar and the synthesis
    is disclosed in the render; a zero-input call infers a verified input
    and REUSES its verification trace (no second capture). Passing a real
    input and ``input_shape=`` together raises ``input_rung_conflict``.

    Parameters
    ----------
    model:
        PyTorch model to inspect, or a Hugging Face Hub model id (``"gpt2"``):
        a string fetches the config only, builds the architecture on the meta
        device (no weight download) and takes the weights-free path, so it
        needs ``input_shape=`` or meta inputs. Requires ``transformers``; the
        string form is DOCUMENTED-UNSTABLE.
    inputs:
        Positional args for ``model.forward()``. A real input is best; omit
        it and pass ``input_shape=`` for a declared shape with synthesized
        values, or omit both for an inferred shape. A synthesized input is
        disclosed in the trace's persistent provenance record and in the
        summary text.
    input_kwargs:
        Keyword args for ``model.forward()``.
    input_shape:
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
        Forwarded to ``Trace.summary`` (the summary grammar). They are
        validated BEFORE the capture runs: a removed legacy spelling or an
        unknown option raises a typed error without executing the model.

    Returns
    -------
    str
        A ``SummaryReport`` (``str`` subclass): canonical ASCII payload
        plus the typed result API.
    """
    from ._utils.rng import log_current_rng_states, set_rng_from_saved_states

    _validate_summary_modes(execution_mode, grad_mode)
    _validate_summary_grammar(summary_kwargs)
    if isinstance(model, str):
        from .report._summary_hub import build_hub_meta_model  # deferred: startup cost

        model = build_hub_meta_model(model)
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    if input_kwargs is None:
        input_kwargs = {}

    # Quickstart input ladder: the summary verb conjugates exactly
    # like trace and render. Non-gold rungs synthesize concrete tensors here
    # (disclosed via the provenance record attached after capture AND the
    # rebuilt renderer's synthetic-input banner line); the inferred rung
    # reuses the inference search's exact verified trace when the default
    # eval/no-grad policy is requested (memo in-call reuse).
    ladder_provenance = None
    input_synthesis: str | None = None
    facade_report = _maybe_weightsfree_summary(
        model, inputs, input_kwargs, input_shape, summary_kwargs
    )
    if facade_report is not None:
        return facade_report
    if input_shape is not None or (inputs is None and not input_kwargs):
        from ._quickstart._resolve import attach_provenance, resolve_inputs

        resolved = resolve_inputs(model, inputs, input_kwargs, input_shape, verb="summary")
        ladder_provenance = resolved.provenance
        input_synthesis = _ladder_synthesis_note(input_shape, resolved)
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
        inputs = list(resolved.plan.input_args)
        input_kwargs = dict(resolved.plan.input_kwargs)

    inputs, input_kwargs = _coerce_inputs(model, inputs, input_kwargs)
    check_model_and_input_variants(model, inputs, input_kwargs)

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
                input_args=inputs,
                input_kwargs=input_kwargs,
                layers_to_save=None,
                recurrence_detection=True,
            )
    finally:
        for module, was_training in saved_training_flags:
            module.training = was_training
        set_rng_from_saved_states(rng_snapshot)

    if ladder_provenance is not None:
        from ._quickstart._resolve import attach_provenance

        attach_provenance(trace, ladder_provenance)
    return _summary_report_from_trace(
        trace,
        summary_kwargs,
        execution_mode=execution_mode,
        grad_mode=grad_mode,
        input_synthesis=input_synthesis,
    )


def _format_input_size(input_size: Any) -> str:
    """Render the caller's ``input_shape=`` spelling compactly for disclosure."""

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
    line from the ONE quickstart provenance record; the gold
    rung returns ``None`` (real values, nothing to disclose).
    """

    provenance = resolved.provenance
    if provenance.has_semantic_values:
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
        return f"input_shape={_format_input_size(input_size)}, {recipe_part}{seed_part}"
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
    """Render one captured trace's summary, then clean the trace up."""

    try:
        return trace.summary(
            **summary_kwargs,
            _execution_note=_summary_execution_note(execution_mode, grad_mode),
            _input_synthesis=input_synthesis,
        )
    finally:
        trace.cleanup()


def _validate_summary_grammar(summary_kwargs: dict[str, Any]) -> None:
    """Resolve the summary grammar up front so a bad option never runs a capture.

    ``Trace.summary`` resolves the same configuration again after capture;
    resolving here first makes removed legacy spellings and unknown options
    refuse typed before the model executes. ``None`` means "default" only
    for grammar options; any other name is checked by presence, matching
    ``Trace.summary``, so ``preset=None`` refuses here too.
    """

    from .report._summary_config import GRAMMAR_OPTIONS, resolve_config  # deferred: startup cost

    resolve_config(
        **{
            name: value
            for name, value in summary_kwargs.items()
            if value is not None or name not in GRAMMAR_OPTIONS
        }
    )


def _validate_summary_modes(execution_mode: str, grad_mode: str) -> None:
    """Refuse unknown one-call summary execution/grad mode tokens typed."""

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
    """Rung-4 gate: auto-select or refuse.

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
        raise InvalidArgumentError(
            "summary() on a meta-built model needs input evidence: the "
            "zero-argument inference rung runs real probing forwards, which "
            "a weights-free model cannot serve.",
            code="weightsfree_facade_input_evidence_required",
            remedy=(
                "pass input_shape= (declared shapes; e.g. input_shape=(1, 8)) "
                "or explicit meta inputs (torch.zeros(..., device='meta'))"
            ),
            argument="input_shape",
        )
    return _weightsfree_summary_facade(model, input_args, input_kwargs, input_size, summary_kwargs)


def _all_meta_model(model: nn.Module) -> bool:
    """Whether every registered parameter/buffer sits on the meta device.

    The rung-4 facade trigger: only an ALL-meta model
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

    if input_size is not None:
        from ._quickstart._resolve import resolve_inputs

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
        envelope = trace._graph_structure_evidence
        if isinstance(envelope, dict) and isinstance(envelope.get("input_plan"), dict):
            envelope["input_plan"]["selection_source"] = "summary_facade_auto"
            if input_size is not None:
                envelope["input_plan"]["source"] = "declared_input_size"
                envelope["input_plan"]["synthesized"] = ["meta_input_leaves"]
        return trace.summary(
            **summary_kwargs,
            _execution_note=_summary_execution_note("eval", "off"),
        )
    finally:
        trace.cleanup()


def _summary_execution_note(execution_mode: str, grad_mode: str) -> str:
    """Return the compact execution disclosure the summary header hoists.

    Both one-call doors pass it into ``trace.summary(_execution_note=...)``,
    so the text, ``details()``, ``render()`` and HTML all carry it.
    """

    if execution_mode == "eval":
        mode_part = "eval mode"
    elif execution_mode == "train":
        mode_part = "train mode (explicit; stateful layers may have updated buffers)"
    else:
        mode_part = "caller's module modes"
    grad_part = "no_grad" if grad_mode == "off" else "caller's grad context"
    return f"{mode_part}, {grad_part}, state restored"


_GRAPH_VERBS = frozenset({"show_model_graph"})


def __getattr__(name: str) -> Any:
    """Resolve the graph verbs from their own module on first use."""
    if name in _GRAPH_VERBS:
        from . import _user_public_graph_impls  # deferred: startup cost

        return getattr(_user_public_graph_impls, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@restores_rng_streams
def validate_forward_pass(
    model: nn.Module,
    inputs: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    backend: BackendName | None = None,
    output_device: OutputDeviceLiteral = "same",
    save_budget: SaveBudgetOption = "auto",
    num_threads: int | None = None,
    trace_observer: Callable[[Trace | None], None] | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    Parameters
    ----------
    model:
        Model or callable to validate.
    inputs:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    seed:
        Fixed RNG seed; like ``tl.trace``, the caller's streams are restored on return.
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True, also run metadata invariant checks.
    backend:
        Explicit backend name. ``None`` preserves legacy auto-resolution.
    output_device:
        Where the validator capture keeps saved activations (same values and
        default as ``CaptureOptions.output_device``). Saved function
        arguments (the replay inputs) other than parameters are kept there
        too and replay moves each back to its op's device, so under ``"cpu"``
        the GPU keeps only parameters and one forward's transients; the
        capture's ``save_budget`` charges each copy where it lives.
    save_budget:
        Per-device retained-bytes ceiling for the validator capture (same values and
        default as ``CaptureOptions.save_budget``); parameters kept by reference are not charged.
    num_threads:
        Torch intra-op threads for the validation forwards, restored afterwards.
    trace_observer:
        Called once per run with the validation trace after replay and before its
        cleanup, or with ``None`` when a precondition fails; see ``tl.validate``.

    Returns
    -------
    bool
        True if all validation checks pass, False otherwise.

    Raises
    ------
    BackendUnsupportedError
        If a non-default save option, ``num_threads`` or ``trace_observer`` is
        requested for a non-torch backend.

    Examples
    --------
    >>> ok = tl.validation.validate_forward_pass(model, x)
    >>> type(ok).__name__, ok
    ('bool', True)
    """

    spec, forwarded = resolve_forward_validation_entry(
        backend,
        model,
        inputs,
        input_kwargs,
        ForwardValidationEntryOptions(output_device, save_budget, num_threads, trace_observer),
    )
    return spec.validate_entry(
        model,
        inputs,
        input_kwargs=input_kwargs,
        seed=seed,
        verbose=verbose,
        validate_metadata=validate_metadata,
        **forwarded,
    )


def _restore_validation_replay_state(
    model: nn.Module,
    state_dict: StateSnapshot,
    plain_attr_snapshot: _ModuleTreePlainAttrSnapshot | None,
) -> None:
    """Restore a validation model to its pre-capture replay state.

    Parameters
    ----------
    model:
        Model used for validation capture and replay.
    state_dict:
        State snapshot from before the traced validation forward.
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
        # back to proven in-place restoration.
        restores.callback(
            restore_snapshot_if_changed, model, state_dict, restore_state_dict_resilient
        )


def _warn_if_validation_trace_not_reproducible(
    first_trace: Trace,
    model: nn.Module,
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any],
    random_seed: int,
) -> Literal["matched", "mismatch", "unavailable"]:
    """Warn when a validation trace changes after one fresh re-trace.

    The re-trace reuses ``first_trace``'s own ``output_device`` and
    ``save_budget``, so it runs under the same capture mode by construction
    (and on a GPU model with ``output_device="cpu"`` does not exceed the device
    budget the first capture avoided, which would silently skip this check as
    ``"unavailable"``).

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

    # Read outside the guard below: a trace without its save options must fail
    # loudly, never read as an "unavailable" re-trace.
    retrace_output_device = cast(OutputDeviceLiteral, first_trace._options_output_device)
    retrace_save_budget = first_trace.save_budget
    second_trace: Trace | None = None
    try:
        # Save only the outputs the structure hash reads (buffer nodes and sources;
        # ``_validation_retrace_saves``), failing closed if the save set missed one.
        # The first trace is captured under a FORCED "shadow" completeness-witness
        # mode (validate_forward_pass), so the re-trace must run under the same
        # mode -- comparing a shadow capture against an ambient-mode capture is
        # exactly the two-capture-modes comparison this check must not make (it
        # false-FAILED every witness-mode-sensitive model as non-reproducible).
        from . import _state

        retrace_saved_indices = retrace_save_indices(first_trace)
        prior_witness_mode = _state._completeness_witness_mode
        _state._completeness_witness_mode = "shadow"
        try:
            second_trace = _user_funcs._run_model_and_save_specified_outs(
                model=model,
                input_args=input_args,
                input_kwargs=input_kwargs,
                layers_to_save="none",
                activation_transform=None,
                mark_layer_depths=False,
                detach_saved_activations=False,
                save_grads=False,
                save_args=False,
                seed=random_seed,
                save_rng=False,
                output_device=retrace_output_device,
                save_budget=retrace_save_budget,
                _resolved_layer_nums_to_save=retrace_saved_indices,
            )
        finally:
            _state._completeness_witness_mode = prior_witness_mode
        verdict = compare_retrace_graphs(first_trace, second_trace, retrace_saved_indices)
        if verdict is None:
            return "matched"
        first_hash, second_hash, hint = verdict
        hint_suffix = f" ({hint})" if hint is not None else ""
        message = (
            "TorchLens validation detected a stateful/non-reproducible model: "
            "fresh re-trace produced a structurally different graph. The trace "
            "may represent a one-time execution. Re-run from fresh model state or "
            f"make the forward path state-independent{hint_suffix}."
        )
        from .validation._diagnostics import ValidationDiagnostic, record_validation_diagnostic

        record_validation_diagnostic(
            first_trace,
            ValidationDiagnostic(
                check_name="trace_retrace_structure_mismatch",
                message=message,
                extra={
                    "first_graph_hash": first_hash,
                    "retrace_graph_hash": second_hash,
                    "first_op_count": len(first_trace._layer_list),
                    "retrace_op_count": len(second_trace._layer_list),
                    "first_divergence": hint,
                },
            ),
        )
        warnings.warn(
            TraceNotReproducibleWarning(
                message,
                first_graph_hash=first_hash,
                retrace_graph_hash=second_hash,
                first_op_count=len(first_trace._layer_list),
                retrace_op_count=len(second_trace._layer_list),
                first_divergence=hint,
            ),
            stacklevel=2,
        )
        return "mismatch"
    except Warning:
        # A filter promoted a warning (this check's own verdict warning, or one
        # the re-trace emitted) to an error: let it propagate, never read it as
        # "unavailable", which would silently skip the mismatch downgrade.
        raise
    except Exception as exc:
        message = (
            "TorchLens validation could not run the fresh re-trace reproducibility "
            f"check ({type(exc).__name__}: {exc})."
        )
        from .validation._diagnostics import ValidationDiagnostic, record_validation_diagnostic

        record_validation_diagnostic(
            first_trace,
            ValidationDiagnostic(
                check_name="trace_retrace_unavailable",
                message=message,
                extra={"exception_type": type(exc).__name__},
            ),
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
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

    from ._validation_status import ValidationStatus

    current_status = trace.validation_replay_status
    unverified_reason_counts = dict(current_status.unverified_reason_counts)
    unverified_reason_counts["trace_retrace_structure_mismatch"] = (
        unverified_reason_counts.get("trace_retrace_structure_mismatch", 0) + 1
    )
    setattr(
        trace,
        "_validation_replay_status",
        ValidationStatus.unverified(
            backend=current_status.backend,
            source=current_status.source,
            reason="trace_retrace_structure_mismatch",
            message=(
                "Replay validation matched the captured trace, but the pristine "
                "re-trace diverged structurally, so the overall validation "
                "result is unverified."
            ),
            num_replayed_ops=current_status.num_replayed_ops,
            num_unverified_ops=max(1, current_status.num_unverified_ops),
            payload_load_status=current_status.payload_load_status,
            num_pure_unverified_ops=current_status.num_pure_unverified_ops,
            num_effect_region_ops=current_status.num_effect_region_ops,
            num_failed_ops=current_status.num_failed_ops,
            unverified_reason_counts=unverified_reason_counts,
            exempted_reason_counts=current_status.exempted_reason_counts,
            decisions=current_status.decisions,
        ),
    )


def _refuse_validation_precondition(
    message: str,
    trace_observer: Callable[[Trace | None], None] | None = None,
    extra: dict[str, Any] | None = None,
) -> Literal[False]:
    """Warn, record a structured precondition-refusal diagnostic, and refuse.

    ``_validate_forward_pass_torch`` has several early-exit sites (an
    unreproducible input topology, an unsnapshotable plain attribute, a
    non-pristine ground truth, a dropped-output enumeration defect) that
    return bare ``False`` before Step 2 builds a ``Trace``. A ``Trace``-scoped
    :class:`~.validation._diagnostics.ValidationFailure` (the mechanism every
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
    extra:
        Optional structured payload for the recorded failure (``code``).

    Returns
    -------
    Literal[False]
        Always ``False``, for ``return _refuse_validation_precondition(...)``.
    """

    warnings.warn(message, RuntimeWarning, stacklevel=3)
    from .validation._diagnostics import (  # deferred: cycle validation
        CHECK_PRECONDITION,
        ValidationFailure,
        record_validation_failure,
    )

    record_validation_failure(
        None,
        ValidationFailure(check_name=CHECK_PRECONDITION, message=message, extra=dict(extra or {})),
    )
    if trace_observer is not None:
        trace_observer(None)
    return False


@_arg_references.with_parameter_reference_fallback
@leaf_preserving_input_copies()
def _validate_forward_pass_torch(
    model: nn.Module | Callable[..., Any],
    input_args: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    seed: int | None = None,
    verbose: bool = False,
    validate_metadata: bool = True,
    *,
    num_threads: int | None = None,
    output_device: OutputDeviceLiteral = "same",
    save_budget: SaveBudgetOption = "auto",
    _trace_observer: Callable[[Trace | None], None] | None = None,
) -> bool:
    """Validate that saved outs faithfully reproduce the model's output.

    **How it works:**

    1. Run model.forward() on PRISTINE torch to get ground-truth output
       tensors: if torchlens wrappers are installed, they are removed for
       this one forward (restored from the pre-decoration originals ledger)
       and reinstalled afterwards, so the ground truth never observes
       through the wrapper layer it is meant to check. If they
       cannot be removed (a capture is active), validation refuses with
       ``False`` rather than blessing a wrap-state-dependent ground truth.
    2. Run ``trace`` with ``save_args=True`` and ``save='all'``
       to capture every out and its creating function's arguments.
    3. Call ``Trace.validate_forward_pass`` which replays the forward pass
       layer-by-layer from saved outs, checking that the output matches ground
       truth, and injects random outs to verify the output changes (the saved outs are used).
    4. If ``validate_metadata=True``, run comprehensive invariant checks on all
       metadata cross-references (graph edges, module containment, labels, etc.).

    **Why save_args=True is required:**  The validation replay re-executes each
    function using its saved non-tensor arguments (e.g., stride, padding for conv2d).

    Parameters
    ----------
    model:
        PyTorch model.
    input_args:
        Input for which to validate the saved outs.
    input_kwargs:
        Keyword arguments for model forward pass.
    seed:
        Fixed RNG seed (auto-generated if None). Seeds the global RNG engines for the
        ground-truth and logged runs; the public entries restore the caller's streams.
    verbose:
        If True, print detailed error messages on validation failure.
    validate_metadata:
        If True (default), also run metadata invariant checks.
    num_threads:
        Optional intra-op thread count for the validation forwards. ``None``
        preserves the process default; an integer pins for this harness call and
        restores the previous thread count afterward.
    output_device:
        Where both validation captures keep saved activations (same values
        and default as ``CaptureOptions.output_device``). Ground-truth outputs
        are snapshotted onto the same device so the output comparison stays
        exact; replay moves saved inputs back to each op's original device.
        Saved function arguments other than parameters are kept there too,
        so the GPU keeps only parameters and one forward's transients.
    save_budget:
        Per-device retained-bytes ceiling for both validation captures (same
        values and default as ``CaptureOptions.save_budget``); exceeding it
        raises ``SaveBudgetExceededError``. Parameters kept by reference are not charged.
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
    from .validation._diagnostics import reset_validation_failure  # deferred: cycle validation

    # Clear any stale precondition-refusal failure from a prior call on this
    # process (CHECK_PRECONDITION sites below have no Trace to reset against,
    # unlike the Trace-scoped reset in core.validate_saved_outs).
    reset_validation_failure(None)
    # bound-method roots: validation captures through the same ruled root
    # contract as tl.trace -- a bound method of an nn.Module wraps into the
    # TL-authored synthetic root before the module-shaped preflights run;
    # other callables refuse typed (previously a bare AttributeError leak).

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
    input_args, input_kwargs = _coerce_inputs(model, input_args, input_kwargs)
    check_model_and_input_variants(model, input_args, input_kwargs)
    # Fix a random seed so both the ground-truth run and the logged run see
    # identical randomness (critical for models with dropout, etc.).
    if seed is None:
        seed = random.randint(1, 4294967294)
    set_random_seed(seed)
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
    # models (accelerate device_map / cpu/disk offload) hold meta
    # params between forwards and their hooks place inputs themselves; the
    # first NON-meta param still pins for mixed dispatch.
    model_device = next((p.device for p in model.parameters() if p.device.type != "meta"), None)
    if model_device is not None:
        input_args_copy = _move_tensors_to_device(input_args_copy, model_device)
        input_kwargs_copy = _move_tensors_to_device(input_kwargs_copy, model_device)

    # Step 1: Get ground-truth outputs by running the model *outside* TorchLens.
    # Save state_dict first because requires_grad forcing during logging can
    # alter parameter metadata; we restore it afterward. Accelerator tensors are
    # snapshotted to host memory, so this snapshot adds no copy of the weights
    # on the device beside the deepcopy validation runs.
    state_dict = snapshot_state_for_restore(model)
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
    # is what the band-C reduction-depth tolerance in validation/_core.py covers;
    # the changes are complementary.
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prior_num_threads = torch.get_num_threads()
    try:
        # (install-move half): both process-global installs run INSIDE
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
        from ._utils._torch_compat import get_cpu_float8_deterministic_fill_support

        if get_cpu_float8_deterministic_fill_support():
            torch.use_deterministic_algorithms(True, warn_only=True)
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        ground_truth_model, plain_attr_snapshot = _model_for_ground_truth_validation(model)
        if plain_attr_snapshot is not None and not plain_attr_snapshot.is_complete:
            return _refuse_validation_precondition(
                restore_refusal_message(
                    "model-state",
                    plain_attr_snapshot.unsupported_attr_paths,
                    plain_attr_snapshot.copy_failure,
                ),
                _trace_observer,
                {"code": COPY_FAILED_CODE} if plain_attr_snapshot.copy_failure else None,
            )
        from .backends.torch._ops_containers import _walk_output_tensors_with_paths
        from .validation._pristine import pristine_ground_truth_oracle  # deferred: cycle validation

        # the phase-0 ground truth must observe through PRISTINE
        # torch. In a wrapped process this forward used to run through the
        # installed pass-through wrapper shells -- the same closures capture
        # and replay observe through -- so a wrapper-layer numeric
        # distortion validated clean whenever any capture had run earlier
        # in the process (probe-proven). The wrappers are removed for this
        # one forward and reinstalled after; if they cannot be removed (a
        # capture is active), validation REFUSES rather than blessing a
        # wrap-state-dependent ground truth.
        try:
            with pristine_ground_truth_oracle(ground_truth_model):
                ground_truth_output = ground_truth_model(*input_args_copy, **input_kwargs_copy)
        except CaptureNotActiveError:
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
        # Ground-truth enumeration SHARED ITS ROOT with capture
        # (both resolve through the one backends walker), so a walker defect
        # dropped the same output leaf from both sides and validation blessed
        # a missing output. Cross-check the adapter's enumeration against the
        # validation-owned independent traversal; a disagreement is a capture
        # (or adapter) bug and must FAIL validation, never pass silently.
        from .validation._output_walk import independent_output_tensor_ids  # deferred: cycle

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
        # Deduplicate by structural address to match how _capture/trace.py extracts
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
            ground_truth_snapshot = entry[0].detach().clone()
            if output_device != "same":
                # Saved outputs live on ``output_device``; an exact copy of
                # the ground truth there keeps the comparison device-local.
                ground_truth_snapshot = ground_truth_snapshot.to(output_device)
            ground_truth_output_tensors.append(ground_truth_snapshot)
            addresses_used.append(entry[1])
        # Release the ground-truth copy (and the outputs and autograd graph that
        # pin its weights) before the replay copy is made.
        if ground_truth_output_all:
            del entry
        del ground_truth_model, ground_truth_output, ground_truth_output_all
        # Restore only if the ground-truth run changed the live model (it runs on
        # a deepcopy when one can be made): an unconditional in-place restore
        # moves every tensor's version counter and breaks a graph the caller
        # holds across validate.
        restore_snapshot_if_changed(model, state_dict, restore_state_dict_resilient)
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
                restore_refusal_message(
                    "replay-state",
                    validation_plain_attr_snapshot.unsupported_attr_paths,
                    validation_plain_attr_snapshot.copy_failure,
                ),
                _trace_observer,
                {"code": COPY_FAILED_CODE} if validation_plain_attr_snapshot.copy_failure else None,
            )
        # The replay model was just copied from (or is) the restored live model,
        # so its state normally equals the live snapshot bit for bit: reuse its
        # values rather than hold a second copy of the weights, with the
        # storage bindings of the replay model's own tensors.
        validation_state_dict = (
            with_storage_bindings_of(state_dict, validation_model)
            if state_matches_snapshot(validation_model, state_dict)
            else snapshot_state_for_restore(validation_model)
        )
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
        # save_args=True is essential - the replay needs each function's
        # non-tensor arguments to re-execute the computation from saved outs.
        from . import _state

        prior_witness_mode = _state._completeness_witness_mode
        _state._completeness_witness_mode = "shadow"
        try:
            with _arg_references.parameter_argument_references(
                validation_model, validation_model_copied, output_device
            ) as arg_references:
                trace = _user_funcs._run_model_and_save_specified_outs(
                    model=validation_model,
                    input_args=validation_input_args,
                    input_kwargs=validation_input_kwargs,
                    layers_to_save="all",
                    activation_transform=None,
                    mark_layer_depths=False,
                    detach_saved_activations=False,
                    save_grads=False,
                    save_args=True,
                    seed=seed,
                    save_rng=True,
                    output_device=output_device,
                    save_budget=save_budget,
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
                seed,
            )
        else:
            from .validation._diagnostics import ValidationDiagnostic, record_validation_diagnostic

            record_validation_diagnostic(
                trace,
                ValidationDiagnostic(
                    check_name="trace_retrace_pristine_copy_unavailable",
                    message=(
                        "TorchLens validation skipped the pristine trace-vs-retrace check "
                        "because the model could not be deep-copied."
                    ),
                    extra={"model_type": f"{type(model).__module__}.{type(model).__qualname__}"},
                ),
            )
        arg_references.raise_if_changed(trace, validation_state_dict)
        _restore_validation_replay_state(
            validation_model,
            validation_state_dict,
            validation_plain_attr_snapshot,
        )
        # Step 3: Validate by replaying the forward pass from saved outs.
        # validate_saved_outs resets the diagnostics ledger at entry so a
        # DIRECT repeat call reports this-run evidence only; the
        # retrace diagnostics recorded above belong to THIS flow, so they are
        # snapshotted and re-prepended after the replay run.
        from .validation._diagnostics import (  # deferred: cycle validation
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
            from .validation._diagnostics import (  # deferred: cycle validation
                CHECK_RETRACE_MISMATCH,
                ValidationFailure,
                record_validation_failure,
            )

            retrace_diagnostic = next(
                (
                    diagnostic
                    for diagnostic in get_validation_diagnostics(trace)
                    if diagnostic.check_name == "trace_retrace_structure_mismatch"
                ),
                None,
            )
            record_validation_failure(
                trace,
                ValidationFailure(
                    check_name=CHECK_RETRACE_MISMATCH,
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
            teardown.callback(
                restore_snapshot_if_changed, model, state_dict, restore_state_dict_resilient
            )
            if num_threads is not None:
                teardown.callback(torch.set_num_threads, prior_num_threads)
            teardown.callback(
                torch.use_deterministic_algorithms,
                prior_deterministic,
                warn_only=prior_warn_only,
            )
    return outs_are_valid


@restores_rng_streams
def validate_backward_pass(
    model: nn.Module,
    inputs: torch.Tensor | list[Any] | tuple[Any, ...],
    input_kwargs: dict[Any, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    validate_metadata: bool = True,
    seed: int | None = None,
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
    inputs:
        Positional args for ``model.forward()``; for a Hugging Face model also a string, PIL image or modality dict, resolved as ``tl.trace`` resolves it.
    input_kwargs:
        Keyword args for ``model.forward()``.
    loss_fn:
        Optional callable mapping model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    seed:
        Fixed RNG seed for stock and candidate passes. Like ``tl.trace``, the call
        seeds the process-global RNG engines for its runs and restores the caller's
        streams on return and on error.
    atol:
        Absolute allclose tolerance. ``None`` (default) derives the
        tolerance per gradient dtype; the historical fp32 decimal
        pair applied to every dtype was ~4.5e11 fp64 ULPs loose and
        false-failed fp16 grads.
    rtol:
        Relative allclose tolerance. ``None`` (default) derives per
        gradient dtype, matching ``torchlens.validation._backward``.
    validate_layer_grads:
        If True (default), also validate captured per-module-output gradients.
        The root module call is never compared, so a model whose forward calls
        no child module returns False with a coded ``TorchLensWarning``
        (``backward_validation_no_module_output_evidence``).
    layer_grad_atol:
        Optional layer-gradient absolute tolerance.
    layer_grad_rtol:
        Optional layer-gradient relative tolerance.

    Returns
    -------
    bool
        True if backward capture matches stock autograd.

    Examples
    --------
    >>> ok = tl.validation.validate_backward_pass(model, x)
    >>> type(ok).__name__, ok
    ('bool', True)
    """
    from .validation._backward import validate_backward_pass as _impl  # deferred: cycle validation

    return _impl(
        model,
        inputs,
        input_kwargs=input_kwargs,
        loss_fn=loss_fn,
        validate_metadata=validate_metadata,
        seed=seed,
        atol=atol,
        rtol=rtol,
        validate_layer_grads=validate_layer_grads,
        layer_grad_atol=layer_grad_atol,
        layer_grad_rtol=layer_grad_rtol,
    )
