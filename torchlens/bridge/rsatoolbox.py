"""rsatoolbox bridge: the single-Dataset special case (neuro memo D2).

The full per-site handoff lives at :func:`torchlens.neuro.datasets`; this
legacy spelling survives as a DELEGATION to the same flattening and
descriptor code path (``torchlens.neuro._datasets.single_dataset``), so one
implementation owns the identity story. What changed with the delegation:
the historical silent flatten is now the disclosed ``pool="flatten"``
descriptor; the ``"neuroid"`` channel label is retired for a neutral
``feature_index`` (an artificial unit is not a biological one); descriptors
carry the full identity story (site, site_key, versions, dtype, casts)
instead of ``{'source': 'torchlens'}``; and the always-written
``tl_presentation_index`` observation descriptor makes rsatoolbox's sorted
``calc_rdm(descriptor=...)`` outputs recoverable. The accidental integer
``presentation`` column is still written for existing readers.

``site=None`` keeps the historical final-output behavior for existing
callers (Trace route only).
"""

from __future__ import annotations

from typing import Any

__tl_layer__ = "L8"


def dataset(source: Any, site: str | None = None) -> Any:
    """Convert one TorchLens site into an ``rsatoolbox`` Dataset.

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` with saved activations (in-memory route), a
        ``LoadedExtraction``, or an extraction-artifact directory path
        (file route).
    site:
        Site selector: Trace lookup (qualified module address,
        pass-qualified label) or extraction output key. ``None`` selects
        the final tensor output (the historical behavior; Trace route
        only).

    Returns
    -------
    Any
        ``rsatoolbox.data.Dataset``: batch items/stimuli as observations,
        flattened units as channels (``feature_index``); ``obs_descriptors``
        carry recorded stimulus ids when the artifact has them plus the
        always-written ``tl_presentation_index``; ``descriptors`` carry the
        full identity story including the disclosed ``pool="flatten"``.

    Raises
    ------
    ImportError
        If rsatoolbox is unavailable.
    ValueError
        If ``site=None`` and the log holds no tensor output activation, or
        an explicitly requested site is not stimulus-indexed (the core
        eligibility gate; a buffer overwrite must never masquerade as a
        stimulus response).
    torchlens.features.FeatureShapingError
        When the site holds no payload or row/id cardinality disagrees.
    """

    from torchlens.neuro._datasets import single_dataset

    return single_dataset(source, site)


__all__ = ["dataset"]
