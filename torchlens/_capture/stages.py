"""Private hand-off records between the four capture stages.

The torch capture path runs PREPARE (request resolution), WRAP (model
instrumentation), RUN (the forward pass and its arms) and FINALIZE (outputs,
settlement and the door epilogue). These frozen records are the only values
that cross a stage boundary. This module is a leaf: it imports nothing from the
package at runtime (annotations only, under ``TYPE_CHECKING``), so it never
joins the import cycle. Nothing here is public.

``InternalCaptureConfig`` moved here verbatim from ``torchlens._capture.config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torch import nn

    from torchlens._capture.session import CaptureSession
    from torchlens._capture.stop import StopDirective
    from torchlens._capture_state_helpers import CompiledCapturePrep
    from torchlens._data_classes.trace import Trace
    from torchlens.options import CaptureOptions

ForwardArm = Literal["completed", "halted", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class InternalCaptureConfig:
    """Grouped internal capture configuration for one forward run.

    Parameters
    ----------
    capture_mode
        Active producer mode: exhaustive, fast, or predicate.
    layers_to_save
        Public layer-save selector after entry-boundary normalization.
    grad_layers_to_save
        Gradient layer selector after entry-boundary normalization.
    random_seed
        Forward RNG seed.
    postprocess
        Whether the run should materialize the final ``Trace`` immediately.
    stop
        Compiled stop directive for halt/non-finite/forward-error behavior.
    """

    capture_mode: str
    layers_to_save: Any
    grad_layers_to_save: Any
    random_seed: int | None
    postprocess: bool
    stop: StopDirective


@dataclass(frozen=True, slots=True)
class CaptureRequest:
    """One resolved door call, handed from PREPARE to RUN and FINALIZE.

    Parameters
    ----------
    model
        The user's model.
    input_args
        Positional forward inputs.
    input_kwargs
        Keyword forward inputs.
    input_arg_names
        Names bound to the positional inputs, when known.
    backend
        Resolved backend name.
    options
        The resolved capture options.
    plan
        Resolved selectors, stop directive, streaming options, intervention
        spec and hook plan, episode, facet recipes, save transforms, gradient
        selectors and seed, keyed by today's keyword names.
    cache
        Capture-cache key, path, secret and store callable, or ``None``.
    admission
        Meta admission, rescue eligibility, weightsfree and structure-only flags.
    postprocess
        Whether the run materializes the final ``Trace`` immediately.
    reservation_resume
        Capture-slot reservation handed back by a resumed recorder, if any.
    """

    model: nn.Module
    input_args: Any
    input_kwargs: dict[str, Any] | None = None
    input_arg_names: tuple[str, ...] | None = None
    backend: str = "torch"
    options: CaptureOptions | None = None
    plan: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    admission: dict[str, Any] | None = None
    postprocess: bool = True
    reservation_resume: Any = None


@dataclass(frozen=True, slots=True)
class InstrumentedModel:
    """The instrumented model, handed from WRAP to RUN.

    Release is the exit of the ``instrumented()`` context, not a method.

    Parameters
    ----------
    model
        The prepared, possibly unwrapped module.
    compiled_prep
        Compiled-capture preparation, when the model was compiled.
    observer
        Handle of the capture observer bracket.
    label_session
        Label-session token.
    unwrap_when_done
        Whether torch wrappers are removed after the capture.
    """

    model: nn.Module
    compiled_prep: CompiledCapturePrep | None = None
    observer: Any = None
    label_session: Any = None
    unwrap_when_done: bool = False


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """The outcome of one forward pass, handed from RUN to FINALIZE.

    Parameters
    ----------
    trace
        The trace the forward pass populated.
    request
        The request that started the run.
    arm
        Which exit arm the forward pass took.
    raw_output
        The model's raw output, when it returned.
    output_tensors
        Tensors extracted from the output.
    output_tensor_addresses
        Addresses of the extracted output tensors.
    session
        The capture session.
    exc
        The exception the forward pass raised, if any.
    committed_ops
        Number of operations committed before the arm was taken.
    raw_graph_ws
        Raw graph workspace handed to postprocess.
    """

    trace: Trace
    request: CaptureRequest | None = None
    arm: ForwardArm = "completed"
    raw_output: Any = None
    output_tensors: tuple[Any, ...] = ()
    output_tensor_addresses: tuple[str, ...] = ()
    session: CaptureSession | None = None
    exc: BaseException | None = None
    committed_ops: int = 0
    raw_graph_ws: Any = None


# The class moved here verbatim from ``torchlens._capture.config``, which re-exports it.
# Pickles keep naming the old module, so saved artifacts and the safe-unpickle allowlist
# stay byte-for-byte unchanged.
InternalCaptureConfig.__module__ = "torchlens._capture.config"
