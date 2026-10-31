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
``ActivationsExtractorHelper`` held by a ``PytorchWrapper``-shaped
activations model, the object a model plugin's ``ModelCommitment`` takes.

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
END TO END THROUGH A PLUGIN (2026-10-14): :func:`activations_extractor`
returns a ``PytorchWrapper``-shaped activations model, and a Brain-Score
model plugin registering ``ModelCommitment(activations_model=...)`` over a
resnet18 was scored by ``brainscore_vision.score`` on
MajajHong2015public.IT-pls (neural) and Geirhos2021colour-top1 (behavioral,
through the logits leg) against brainscore-vision 2.4.1 / brainscore-core
2.5.0 on Python 3.11, with every committed layer and the logits equal to
Brain-Score's hook-based ``PytorchWrapper`` on 24 stimuli.
CUDA is not claimed (no covered runner). brainscore-vision requires
Python >= 3.11; the verified adapters are also exposed as
``torchlens.neuro.activations_extractor`` / ``get_activations_fn``. Adapter
spellings are DOCUMENTED-UNSTABLE pending the naming/UI sprint.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._utils import tensor_layers

#: TorchLens lookup used for Brain-Score's special ``"logits"`` layer name.
_LOGITS_LOOKUP = "output_1"


@dataclass(frozen=True)
class ScoreParts:
    """A benchmark score with its raw and ceiling-normalised values named apart.

    DOCUMENTED-UNSTABLE. Brain-Score reports a
    ceiling-normalised score and keeps the raw score and the ceiling in its
    attributes; this record puts them side by side so the normalised value
    never stands in for the raw one. TorchLens computes none of these
    numbers: they are read from the benchmark's result.

    Attributes
    ----------
    raw:
        The raw (un-normalised) score. For a result that carries no
        ceiling, this is the result itself. ``None`` only when the result
        carries a ceiling but its raw attribute is not a single number
        (``note`` says so).
    ceiling_normalised:
        The ceiling-normalised score, or ``None`` when the result carries
        no ceiling (nothing was normalised).
    ceiling:
        The ceiling the benchmark normalised by, or ``None`` (when the
        result carries a ceiling attribute that is not one number, ``note``
        says so).
    error:
        The benchmark's reported error of the score, or ``None``.
    note:
        Why a field is ``None`` when the result does not explain itself.
    """

    raw: float | None
    ceiling_normalised: float | None = None
    ceiling: float | None = None
    error: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as a plain dictionary.

        Returns
        -------
        dict[str, Any]
            ``raw``, ``ceiling_normalised``, ``ceiling``, ``error``, ``note``.
        """

        return {
            "raw": self.raw,
            "ceiling_normalised": self.ceiling_normalised,
            "ceiling": self.ceiling,
            "error": self.error,
            "note": self.note,
        }


def _aggregation_part(value: Any, part: str) -> Any:
    """Select ``part`` of an older Brain-Score ``aggregation`` dimension, if present.

    Brain-Score 1.x scores carried ``aggregation=["center", "error"]``;
    2.x scores are scalar with the error in ``attrs``. Either reads here.
    """

    dims = getattr(value, "dims", ())
    if "aggregation" in dims and hasattr(value, "sel"):
        return value.sel(aggregation=part)
    return value


def _scalar_or_none(value: Any) -> float | None:
    """Return ``value`` as a float when it holds exactly one number, else ``None``."""

    if value is None:
        return None
    raw_values = getattr(_aggregation_part(value, "center"), "values", value)
    try:
        array = torch.as_tensor(raw_values, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError):
        return None
    if array.numel() != 1:
        return None
    return float(array.reshape(()).item())


def score_parts(score: Any) -> ScoreParts:
    """Split a benchmark score into raw, ceiling-normalised, ceiling and error.

    DOCUMENTED-UNSTABLE. A Brain-Score ``Score`` whose
    attributes hold ``"ceiling"`` is ceiling-normalised: its own value goes
    to ``ceiling_normalised`` and its ``"raw"`` attribute to ``raw``. A
    result without a ceiling (a behavioral engineering score, or a plain
    number from an offline callable) is raw: ``raw`` holds it and
    ``ceiling_normalised`` is ``None``. Nothing is recomputed.

    Parameters
    ----------
    score:
        A Brain-Score ``Score`` (any object with ``values`` and ``attrs``)
        or a plain number.

    Returns
    -------
    ScoreParts
        The named parts.

    Raises
    ------
    TypeError
        If ``score`` is neither a single number nor a ``Score``-shaped
        object whose value is a single number.
    """

    attrs = getattr(score, "attrs", None)
    value = _scalar_or_none(score)
    if value is None:
        raise TypeError(
            f"score_parts needs a single-number score; got {type(score).__name__} "
            "without one value."
        )
    attrs = attrs if isinstance(attrs, Mapping) else {}
    error = _scalar_or_none(attrs.get("error"))
    if error is None and "aggregation" in getattr(score, "dims", ()):
        error = _scalar_or_none(_aggregation_part(score, "error"))
    if "ceiling" not in attrs:
        return ScoreParts(
            raw=value,
            error=error,
            note="the result carries no 'ceiling' attribute, so it is reported as raw",
        )
    raw = _scalar_or_none(attrs.get("raw"))
    ceiling = _scalar_or_none(attrs.get("ceiling"))
    gaps = [
        f"its {name!r} attribute is not a single number; read score.attrs[{name!r}]"
        for name, part in (("raw", raw), ("ceiling", ceiling))
        if part is None
    ]
    return ScoreParts(
        raw=raw,
        ceiling_normalised=value,
        ceiling=ceiling,
        error=error,
        note="the result carries a ceiling but " + "; ".join(gaps) if gaps else None,
    )


def per_layer(
    trace: Any,
    benchmark: Any,
    *,
    sites: Iterable[Any] | None = None,
    **kwargs: Any,
) -> dict[str, ScoreParts]:
    """Run a Brain-Score-style benchmark independently for saved layers.

    This offline adapter accepts a callable benchmark fixture so tests can run
    without network access or GUI resources.

    Parameters
    ----------
    trace:
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
    dict[str, ScoreParts]
        Mapping from TorchLens layer label to the benchmark's score split
        by :func:`score_parts`: the raw score is always named, and a
        ceiling-normalised score sits beside it, never in its place.

    Raises
    ------
    TypeError
        If ``benchmark`` is not callable, or returns something
        :func:`score_parts` cannot read as one number.
    ValueError
        If an explicitly requested site has no saved tensor out or is not
        stimulus-indexed.
    """

    if not callable(benchmark):
        raise TypeError("Brain-Score bridge currently requires a callable offline benchmark.")

    scores: dict[str, ScoreParts] = {}
    for layer in tensor_layers(trace, sites, verb="bridge.brain_score.per_layer"):
        out = getattr(layer, "out")
        label = str(getattr(layer, "layer_label", "layer"))
        # TODO: connect the real Brain-Score Benchmark API once the offline
        # vendored fixtures are available in the launch extras matrix.
        scores[label] = score_parts(benchmark(out, layer=label, **kwargs))
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


class _ActivationsModel:
    """Brain-Score activations model over a TorchLens-backed extractor helper.

    Mirrors only the ``PytorchWrapper`` surface Brain-Score's plugin path
    reads (brainscore-vision 2.4.1): ``ModelCommitment.__init__`` calls
    ``activations_model._extractor.set_visual_degrees`` (
    ``model_helpers/brain_transformation/__init__.py:36``) and hands the
    object to ``LayerMappedModel`` / ``LabelBehavior``, which read
    ``.identifier`` (``brain_transformation/neural.py``) and call it with
    stimuli and layers. ``from_paths``, ``from_stimulus_set`` and the hook
    registrars are installed by the helper's own ``insert_attrs``, exactly as
    ``PytorchWrapper.__init__`` does.

    Parameters
    ----------
    extractor:
        The ``ActivationsExtractorHelper`` serving TorchLens activations.
    """

    def __init__(self, extractor: Any) -> None:
        """Hold the helper and install its delegating attributes."""

        self._extractor = extractor
        insert_attrs = getattr(extractor, "insert_attrs", None)
        if callable(insert_attrs):
            insert_attrs(self)

    @property
    def identifier(self) -> Any:
        """Brain-Score activations identifier (stored on the helper)."""

        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value: Any) -> None:
        """Set the helper's activations identifier."""

        self._extractor.identifier = value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Extract activations through the helper (stimuli, ``layers=``)."""

        return self._extractor(*args, **kwargs)


def activations_extractor(
    model: nn.Module,
    preprocessing: Callable[[Any], Any] | None,
    *,
    identifier: str | None = None,
    layer_map: Mapping[str, str] | None = None,
    device: torch.device | str | None = None,
    **extractor_kwargs: Any,
) -> Any:
    """Build a Brain-Score activations model over TorchLens capture.

    Drop-in for a Brain-Score ``PytorchWrapper``: call it with stimuli and
    layers, or hand it to model-commitment machinery
    (``ModelCommitment(activations_model=...)`` in a model plugin). The
    configured ``ActivationsExtractorHelper`` is its ``_extractor``, as on
    ``PytorchWrapper``. Capture runs through TorchLens instead of forward
    hooks, so benchmark layer lists may address functional ops, not just
    modules.

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
        A ``PytorchWrapper``-shaped activations model whose ``_extractor`` is
        a configured ``brainscore_vision.model_helpers.activations.core.
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
    return _ActivationsModel(
        ActivationsExtractorHelper(
            identifier=identifier or model.__class__.__name__,
            get_activations=get_activations,
            preprocessing=preprocessing,
            **extractor_kwargs,
        )
    )


__all__ = [
    "ScoreParts",
    "activations_extractor",
    "get_activations_fn",
    "per_layer",
    "score_parts",
]
