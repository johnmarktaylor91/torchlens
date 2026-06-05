"""Built-in TorchLens semantic facet recipes."""

from . import attention, embedding, mlp, norm
from ..facets import mark_current_registry_as_builtins

mark_current_registry_as_builtins()

__all__ = ["attention", "embedding", "mlp", "norm"]
