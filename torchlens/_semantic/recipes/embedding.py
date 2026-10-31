"""Built-in semantic recipes for embedding modules."""

from __future__ import annotations

from typing import Any

from ..facets import register
from ._helpers import add_if_present, first_input_spec, module_output_spec, parameter_spec

_EMBEDDING_FACETS = ("lookup", "weight", "indices")


@register(
    class_name="Embedding",
    class_qualname="torch.nn.modules.sparse.Embedding",
    facets=_EMBEDDING_FACETS,
)
def embedding(module: Any) -> dict[str, Any]:
    """Return facets for ``torch.nn.Embedding`` modules."""

    result: dict[str, Any] = {}
    add_if_present(result, "lookup", module_output_spec(module, "embedding"))
    add_if_present(result, "indices", first_input_spec(module, "embedding"))
    add_if_present(result, "weight", parameter_spec(module, "weight", "embedding"))
    return result
