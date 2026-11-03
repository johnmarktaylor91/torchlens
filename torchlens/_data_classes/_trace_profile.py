"""Trace model-profile helpers."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from .trace import Trace


@dataclass(frozen=True)
class ModelProfile:
    """Computed descriptor for recognized semantic I/O profiles.

    Attributes
    ----------
    input_modality:
        Conservative modality inferred from raw input and preprocessing
        provenance.
    input_preprocessing_source:
        ``ResolvedPreprocessing.source`` when present.
    output_postprocessing_source:
        ``ResolvedPostprocessing.source`` when present.
    output_label_count:
        Number of known output labels/classes, if available.
    has_output_labels:
        Whether a label space is available from ``output_id2label``.
    num_stimuli:
        Inferred leading batch/stimulus count from ``input``.
    has_raw_images:
        Whether raw input appears to contain image objects.
    keystone_applicable:
        Whether the image-classifier keystone cascade has the required input
        and output metadata.
    """

    input_modality: str
    input_preprocessing_source: str | None
    output_postprocessing_source: str | None
    output_label_count: int | None
    has_output_labels: bool
    num_stimuli: int | None
    has_raw_images: bool
    keystone_applicable: bool


_IMAGE_PREPROCESSING_SOURCES = frozenset(
    {
        "hf_auto_image_processor",
        "torchvision_weights",
        "imagenet_default",
        "timm",
    }
)
_TEXT_PREPROCESSING_SOURCES = frozenset({"hf_auto_tokenizer"})


def _is_pil_image_like(value: Any) -> bool:
    """Return whether ``value`` looks like a PIL image instance.

    Parameters
    ----------
    value:
        Candidate raw input value.

    Returns
    -------
    bool
        Whether the value has the minimal PIL image surface used by TorchLens
        raw-input rendering.
    """

    return hasattr(value, "mode") and hasattr(value, "size") and hasattr(value, "copy")


def _raw_input_contains_images(value: Any) -> bool:
    """Return whether raw input contains one or more image-like objects.

    Parameters
    ----------
    value:
        Raw input value captured on a trace.

    Returns
    -------
    bool
        Whether an image-like object is present.
    """

    if _is_pil_image_like(value):
        return True
    if isinstance(value, Mapping):
        return any(_raw_input_contains_images(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_raw_input_contains_images(item) for item in value)
    return False


def _raw_input_num_stimuli(value: Any) -> int | None:
    """Infer a leading stimulus count from raw input.

    Parameters
    ----------
    value:
        Raw input value captured on a trace.

    Returns
    -------
    int | None
        Inferred stimulus count, or ``None`` when unavailable.
    """

    if value is None:
        return None
    if _is_pil_image_like(value) or isinstance(value, str):
        return 1
    if isinstance(value, (torch.Tensor, np.ndarray)):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, Mapping):
        counts = {
            count for item in value.values() if (count := _raw_input_num_stimuli(item)) is not None
        }
        return counts.pop() if len(counts) == 1 else None
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return len(value)
    return None


def _infer_input_modality(raw_input: Any, preprocessing_source: str | None) -> str:
    """Infer a conservative input modality descriptor.

    Parameters
    ----------
    raw_input:
        Trace raw input value.
    preprocessing_source:
        Resolved preprocessing source string, if any.

    Returns
    -------
    str
        ``"image"``, ``"text"``, ``"tensor"``, or ``"unknown"``.
    """

    if preprocessing_source in _IMAGE_PREPROCESSING_SOURCES or _raw_input_contains_images(
        raw_input
    ):
        return "image"
    if preprocessing_source in _TEXT_PREPROCESSING_SOURCES or isinstance(raw_input, str):
        return "text"
    if isinstance(raw_input, (torch.Tensor, np.ndarray)):
        return "tensor"
    return "unknown"


def model_profile(trace: "Trace") -> ModelProfile:
    """Body of ``Trace.model_profile``; the record keeps the thin spelling and docstring."""

    input_preprocessor = trace._provenance_input_preprocessor
    output_postprocessor = trace._provenance_output_postprocessor
    input_source = input_preprocessor.source if input_preprocessor is not None else None
    output_source = output_postprocessor.source if output_postprocessor is not None else None
    has_output_labels = bool(trace.output_id2label)
    output_label_count = (
        len(trace.output_id2label)
        if trace.output_id2label is not None
        else trace.output_num_classes
    )
    has_raw_images = _raw_input_contains_images(trace.input)
    input_modality = _infer_input_modality(trace.input, input_source)
    return ModelProfile(
        input_modality=input_modality,
        input_preprocessing_source=input_source,
        output_postprocessing_source=output_source,
        output_label_count=output_label_count,
        has_output_labels=has_output_labels,
        num_stimuli=_raw_input_num_stimuli(trace.input),
        has_raw_images=has_raw_images,
        keystone_applicable=bool(
            input_modality == "image"
            and has_raw_images
            and has_output_labels
            and output_label_count is not None
        ),
    )
