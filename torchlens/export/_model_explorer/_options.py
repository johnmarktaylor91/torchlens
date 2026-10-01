"""The frozen collection-option record for the Model Explorer writers.

The public doors (``to_model_explorer_dict``, ``tl.export.model_explorer``)
keep the flat keyword spellings as their API and construct ONE frozen record
from them; the collection/episode/file builders thread the record instead of
tunneling nine scalars per call (the C06 WatchSettings precedent).
"""

from __future__ import annotations

from dataclasses import dataclass

__tl_layer__ = "L8"


@dataclass(frozen=True)
class ModelExplorerOptions:
    """The closed user-option vocabulary shared by every collection writer.

    Field semantics are documented once, on
    :func:`torchlens.export._model_explorer._collection.to_model_explorer_dict`;
    an unknown flat keyword refuses at construction with the standard
    ``TypeError`` naming the offending argument (byte-compatible with the
    former flat signatures).
    """

    label: str | None = None
    privacy_profile: str = "local"
    strict_namespace: bool = False
    include_source: bool = False
    include_rolled: bool | None = None
    per_step: bool | None = None
    boundary_proxies: bool = True
    step_budget_bytes: int = 12_000_000
    max_step_graphs: int = 128
