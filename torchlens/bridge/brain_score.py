"""Brain-Score bridge helpers.

Besides the offline :func:`per_layer` scorer, this module adapts TorchLens
capture to Brain-Score's ``ActivationsExtractorHelper`` seam
(``brainscore_vision.model_helpers.activations.core``): the helper owns
stimulus loading, preprocessing, batching, and assembly packaging, and calls a
``get_activations(images, layer_names) -> OrderedDict[str, np.ndarray]``
callable per batch. :func:`get_activations_fn` builds that callable over
:func:`torchlens.extract` (so layer names may be module dotted paths — the
spelling Brain-Score users already write — OR any TorchLens lookup, including
functional-op labels module hooks cannot address), and
:func:`activations_extractor` wires it into a ready
``ActivationsExtractorHelper``.

The wiring matches brainscore-vision 2.3.22's ``PytorchWrapper`` source
(read directly from the published wheel): images arrive as a list of
preprocessed numpy arrays or tensors, are stacked, moved to the model's
device, dtype-matched to the model parameters, and run under ``eval()`` +
``no_grad()``; the special layer name ``"logits"`` resolves to the model
output. VERIFIED END TO END (the neuro-memo 4.3 live gate, 2026-08-29): a
real resnet18 IMAGENET1K_V1 checkpoint ran through a real
``ActivationsExtractorHelper`` and ``StimulusSet`` against a live
brainscore-vision 2.3.22 install on Python 3.12 -- values, presentation
order, layer coordinates, logits, dotted module paths, a functional-op
mapping, a short final batch, and CPU behavior all matched direct torchlens
extraction, and ``per_layer`` default sites ran on a PARTIALLY saved trace.
CUDA is not claimed (no covered runner). brainscore-vision requires
Python >= 3.11; the verified adapters are also exposed as
``torchlens.neuro.activations_extractor`` / ``get_activations_fn``. Adapter
spellings are DOCUMENTED-UNSTABLE pending the naming/UI sprint.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import nn

from ._utils import tensor_layers

#: TorchLens lookup used for Brain-Score's special ``"logits"`` layer name.
_LOGITS_LOOKUP = "output_1"


def per_layer(
    log: Any,
    benchmark: Any,
    *,
    sites: Iterable[Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a Brain-Score-style benchmark independently for saved layers.

    This offline adapter accepts a callable benchmark fixture so tests can run
    without network access or GUI resources.

    Parameters
    ----------
    log:
        TorchLens ``Trace`` containing saved outs.
    benchmark:
        Callable benchmark accepting ``(out, layer=..., **kwargs)``.
    sites:
        Optional iterable of sites to score: layer labels, selectors, or
        module dotted paths (the spelling Brain-Score users already write).
        Defaults to every STIMULUS-INDEXED saved tensor layer except input
        placeholders, through the core eligibility gate -- a partially
        saved trace scores its saved sites instead of crashing, and buffer
        overwrites are skipped with one summarized disclosure instead of
        being scored as stimulus responses.
    **kwargs:
        Additional benchmark keyword arguments.

    Returns
    -------
    dict[str, Any]
        Mapping from TorchLens layer label to benchmark score.

    Raises
    ------
    TypeError
        If ``benchmark`` is not callable.
    ValueError
        If an explicitly requested site has no saved tensor out or is not
        stimulus-indexed.
    """

    if not callable(benchmark):
        raise TypeError("Brain-Score bridge currently requires a callable offline benchmark.")

    scores: dict[str, Any] = {}
    for layer in tensor_layers(log, sites, verb="bridge.brain_score.per_layer"):
        out = getattr(layer, "out")
        label = str(getattr(layer, "layer_label", "layer"))
        # TODO: connect the real Brain-Score Benchmark API once the offline
        # vendored fixtures are available in the launch extras matrix.
        scores[label] = benchmark(out, layer=label, **kwargs)
    return scores


def _stack_images(
    images: Any, device: torch.device | str, dtype: torch.dtype | None
) -> torch.Tensor:
    """Stack Brain-Score batch images into one model input tensor.

    Parameters
    ----------
    images:
        List/tuple of per-image numpy arrays or tensors (Brain-Score's
        preprocessed batch), or an already-stacked array/tensor.
    device:
        Device the model lives on.
    dtype:
        Model parameter dtype to match for floating-point inputs, or ``None``.

    Returns
    -------
    torch.Tensor
        Batched input on ``device``.
    """

    if isinstance(images, (list, tuple)):
        tensors = [
            image if isinstance(image, torch.Tensor) else torch.from_numpy(image)
            for image in images
        ]
        batch = torch.stack(tensors)
    elif isinstance(images, torch.Tensor):
        batch = images
    else:
        batch = torch.from_numpy(images)
    batch = batch.to(device)
    if dtype is not None and batch.is_floating_point() and batch.dtype != dtype:
        batch = batch.to(dtype)
    return batch


def get_activations_fn(
    model: nn.Module,
    *,
    layer_map: Mapping[str, str] | None = None,
    device: torch.device | str | None = None,
    logits_lookup: str = _LOGITS_LOOKUP,
) -> Callable[[Any, Iterable[str]], OrderedDict[str, Any]]:
    """Build a Brain-Score-shaped ``get_activations`` callable over TorchLens.

    The returned callable has the exact contract Brain-Score's
    ``ActivationsExtractorHelper`` expects: it receives one preprocessed image
    batch and the benchmark's layer names, and returns an ``OrderedDict``
    mapping each requested name to a CPU numpy array whose leading axis is the
    stimulus axis. It runs standalone — no Brain-Score import required — so it
    is also directly testable offline.

    Parameters
    ----------
    model:
        PyTorch model to capture. Run in ``eval()`` mode under ``no_grad()``
        for each batch, matching Brain-Score's own ``PytorchWrapper``.
    layer_map:
        Optional mapping from Brain-Score layer names to TorchLens lookups.
        Unmapped names are used verbatim (module dotted paths work as-is, and
        any TorchLens lookup — including functional-op labels that module
        hooks cannot address — is accepted).
    device:
        Optional device override; defaults to the model's first parameter's
        device (CPU for parameterless models). The model is moved there once.
    logits_lookup:
        TorchLens lookup used for Brain-Score's special ``"logits"`` name.

    Returns
    -------
    Callable[[Any, Iterable[str]], OrderedDict[str, Any]]
        The per-batch activations callable.
    """

    first_param = next(model.parameters(), None)
    resolved_device: torch.device | str = (
        device if device is not None else (first_param.device if first_param is not None else "cpu")
    )
    model_dtype = first_param.dtype if first_param is not None else None
    model = model.to(resolved_device)
    mapping = dict(layer_map or {})

    def get_activations(images: Any, layer_names: Iterable[str]) -> OrderedDict[str, Any]:
        """Extract one preprocessed batch's activations for Brain-Score.

        Parameters
        ----------
        images:
            Preprocessed image batch (list of arrays/tensors, or stacked).
        layer_names:
            Brain-Score layer names to serve, in the benchmark's order.

        Returns
        -------
        OrderedDict[str, Any]
            Requested layer names to CPU numpy activations, batch axis first.
        """

        import torchlens as tl

        names = [str(name) for name in layer_names]
        plan = {
            name: mapping.get(name, logits_lookup if name == "logits" else name) for name in names
        }
        batch = _stack_images(images, resolved_device, model_dtype)
        model.eval()
        with torch.no_grad():
            outputs = tl.extract(model, batch, plan)
        return OrderedDict((name, outputs[name].detach().cpu().numpy()) for name in names)

    return get_activations


def activations_extractor(
    model: nn.Module,
    preprocessing: Callable[[Any], Any] | None,
    *,
    identifier: str | None = None,
    layer_map: Mapping[str, str] | None = None,
    device: torch.device | str | None = None,
    **extractor_kwargs: Any,
) -> Any:
    """Build a Brain-Score ``ActivationsExtractorHelper`` over TorchLens capture.

    Drop-in for the extractor a Brain-Score ``PytorchWrapper`` builds
    internally: call it with stimuli and layers, or hand it to model-commitment
    machinery. Capture runs through TorchLens instead of forward hooks, so
    benchmark layer lists may address functional ops, not just modules.

    Parameters
    ----------
    model:
        PyTorch model to capture.
    preprocessing:
        Brain-Score preprocessing callable mapping stimulus paths to model
        inputs (for images, typically ``load_preprocess_images``); ``None``
        passes stimuli through unchanged.
    identifier:
        Brain-Score activations identifier; defaults to the model class name,
        matching ``PytorchWrapper``.
    layer_map:
        Optional mapping from Brain-Score layer names to TorchLens lookups.
        The special name ``"logits"`` defaults to the model output; map it
        here to override.
    device:
        Optional device override for model and stimuli.
    **extractor_kwargs:
        Forwarded to ``ActivationsExtractorHelper`` (e.g. ``batch_size=``).

    Returns
    -------
    Any
        A configured ``brainscore_vision.model_helpers.activations.core.
        ActivationsExtractorHelper``.

    Raises
    ------
    ImportError
        If ``brainscore_vision`` is not installed (requires Python >= 3.11).
    """

    try:
        from brainscore_vision.model_helpers.activations.core import (
            ActivationsExtractorHelper,
        )
    except ImportError as exc:
        raise ImportError(
            "The Brain-Score extractor adapter requires brainscore-vision "
            "(Python >= 3.11): pip install brainscore-vision."
        ) from exc

    get_activations = get_activations_fn(model, layer_map=layer_map, device=device)
    return ActivationsExtractorHelper(
        identifier=identifier or model.__class__.__name__,
        get_activations=get_activations,
        preprocessing=preprocessing,
        **extractor_kwargs,
    )


__all__ = ["activations_extractor", "get_activations_fn", "per_layer"]
