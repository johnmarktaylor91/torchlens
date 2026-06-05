"""Built-in TorchLens semantic facet recipes."""

from . import attention, embedding, mlp, norm
from ..facets import mark_current_registry_as_builtins

BUILTIN_FACET_CAPABILITY_INVENTORY: dict[str, dict[str, str]] = {
    "attention": {
        "q": "op_structural",
        "k": "op_structural",
        "v": "op_structural",
        "attn_out": "op_structural",
        "input": "module_input",
        "n_heads": "computed_read_only",
        "n_q_heads": "computed_read_only",
        "n_kv_heads": "computed_read_only",
        "d_head": "computed_read_only",
        "head": "computed_read_only",
        "pattern": "missing",
    },
    "mlp": {
        "gated_out": "computed_read_only",
        "up_out": "op_structural",
        "down_out": "op_structural",
        "intermediate": "computed_read_only",
        "input": "module_input",
        "output": "op_structural",
    },
    "norm": {
        "normalized": "op_structural",
        "gamma": "parameter",
        "beta": "parameter",
        "input": "module_input",
    },
    "embedding": {
        "lookup": "op_structural",
        "weight": "parameter",
        "indices": "module_input",
    },
}

mark_current_registry_as_builtins()

__all__ = [
    "BUILTIN_FACET_CAPABILITY_INVENTORY",
    "attention",
    "embedding",
    "mlp",
    "norm",
]
