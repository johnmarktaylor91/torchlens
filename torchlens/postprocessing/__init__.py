"""Output decoding provenance: the public home of custom output detectors.

Output decoding turns a model's raw output into something a person reads
(class labels from logits, text from token ids). TorchLens resolves it from
the model's own metadata (a Hugging Face ``config.id2label``, verified
ImageNet-width logits, an attached tokenizer) through priority-ordered OUTPUT
DETECTORS, and a project registers its own label bank the same way:

.. code-block:: python

    import torchlens as tl
    from torchlens.postprocessing import ResolvedPostprocessing


    @tl.postprocessing.register(name="my_labels", priority=5)
    def my_labels(outputs, meta):
        if meta.get("style") != "my_labels":
            return None
        return ResolvedPostprocessing(
            source="project",
            resolved_for="my_labels",
            is_model_specific=True,
            config={"id2label": {0: "cat", 1: "dog"}},
            description="project label bank",
            style="classification",
            selected_output_head=meta.get("head"),
            label_source="project-local",
            label_source_version="v1",
            confidence=1.0,
        )

A detector runs automatically at the end of every capture (``meta["style"]``
is ``None``) and on demand from ``trace.decode_output(style="my_labels")``;
it returns a :class:`ResolvedPostprocessing` record or ``None``. Every
spelling is DOCUMENTED-UNSTABLE pending the naming sprint; import as
``import torchlens.postprocessing``.
"""

from __future__ import annotations

from torchlens._autoroute.output import (
    Detector,
    info,
    list,
    register,
    snapshot,
    unregister,
)
from torchlens._data_classes.trace import ResolvedPostprocessing

__tl_layer__ = "L5"

__all__ = [
    "Detector",
    "ResolvedPostprocessing",
    "info",
    "list",
    "register",
    "snapshot",
    "unregister",
]
