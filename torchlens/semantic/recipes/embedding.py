"""Built-in semantic recipes for embedding modules."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..facets import register
from ._helpers import add_if_present, first_input, module_output

_EMBEDDING_FACETS = ("lookup", "weight", "indices")


@register(
    class_name="Embedding",
    class_qualname="torch.nn.modules.sparse.Embedding",
    facets=_EMBEDDING_FACETS,
)
def embedding(module: Any) -> dict[str, Any]:
    """Return facets for ``torch.nn.Embedding`` modules."""

    result: dict[str, Any] = {}
    add_if_present(result, "lookup", module_output(module))
    add_if_present(result, "indices", first_input(module))
    cls = getattr(module, "cls", None)
    if isinstance(cls, nn.Embedding):
        result["weight"] = cls.weight
    return result
