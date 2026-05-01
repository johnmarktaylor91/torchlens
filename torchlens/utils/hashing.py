"""Barcode generation and argument hashing for tensor identity tracking.

Barcodes are short opaque identifiers attached to tensors during logging.
They serve two purposes:

1. **Random barcodes** (``make_random_barcode``): assigned to each tensor as
   it is created during the forward pass. These act as globally unique IDs
   so that the logging pipeline can track tensor identity across operations,
   even when the same ``torch.Tensor`` object is reused by in-place ops.

2. **Deterministic barcodes** (``make_short_barcode_from_input``): derived
   from the *content* of a tensor's creation arguments (param data pointers,
   buffer shapes, etc.). Two tensors with the same deterministic barcode
   originated from the same parameter/buffer and are candidates for
   *same-layer grouping* in loop detection — the barcode is the key signal
   that separate forward-pass operations actually reference the same weight.
"""

import hashlib
import json
import re
import random
import string
from typing import Any, List

_BARCODE_ALPHABET = string.ascii_letters + string.digits


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generate a random alphanumeric identifier for internal tensor tracking.

    These barcodes are invisible to the user and are used as unique
    internal keys for tensor entries in ``ModelLog``.

    Args:
        barcode_len: Length of the identifier string.

    Returns:
        Random alphanumeric string of the requested length.
    """
    return "".join(random.choices(_BARCODE_ALPHABET, k=barcode_len))


def make_short_barcode_from_input(things_to_hash: List[Any], barcode_len: int = 16) -> str:
    """Produce a deterministic short hash from a list of values.

    Used to create content-based barcodes for parameters and buffers so
    that loop detection can identify operations that share the same weights.
    The inputs are stringified, joined with a null-byte separator (to avoid
    accidental collisions from concatenation), and hashed with SHA-256.  This
    avoids Python's process-randomized ``hash()`` and the collision-prone decimal
    truncation used by older TorchLens releases.

    Args:
        things_to_hash: Values to hash (must be stringifiable).
        barcode_len: Maximum length of the returned barcode.

    Returns:
        A deterministic hexadecimal SHA-256 prefix of ``barcode_len`` characters.
    """
    # Null-byte separator prevents "ab" + "c" from colliding with "a" + "bc".
    joined = "\x00".join([str(x) for x in things_to_hash])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:barcode_len]


_MODULE_PASS_SUFFIX_RE = re.compile(r":\d+(?=\.|$)")
_ROLLED_LOOP_INDEX_RE = re.compile(r"\[\d+\]")


def normalize_module_address_for_hash(module_address: Any) -> str | None:
    """Normalize a module address for graph-shape hashing.

    Parameters
    ----------
    module_address:
        Module address in either tuple/list form, pass-qualified string form, or
        ``None``.

    Returns
    -------
    str | None
        Pass- and rolled-loop-normalized module address, or ``None`` when no
        module address is available.
    """

    if module_address is None:
        return None
    if isinstance(module_address, (tuple, list)):
        module_address = ".".join(str(part) for part in module_address)
    normalized = str(module_address)
    normalized = _MODULE_PASS_SUFFIX_RE.sub("", normalized)
    normalized = _ROLLED_LOOP_INDEX_RE.sub("", normalized)
    normalized = " ".join(normalized.split())
    return normalized or None


def _hashable_path_component(component: Any) -> Any:
    """Return a JSON-stable representation of one output-path component.

    Parameters
    ----------
    component:
        Output path component.

    Returns
    -------
    Any
        JSON-serializable representation.
    """

    if hasattr(component, "index"):
        return {"type": type(component).__name__, "index": component.index}
    if hasattr(component, "key"):
        return {"type": type(component).__name__, "key": repr(component.key)}
    if hasattr(component, "name"):
        return {"type": type(component).__name__, "name": component.name}
    return {"type": type(component).__name__, "value": repr(component)}


def _container_cardinality(container_spec: Any) -> Any:
    """Return stable cardinality metadata from a container spec.

    Parameters
    ----------
    container_spec:
        Optional intervention ``ContainerSpec``.

    Returns
    -------
    Any
        JSON-serializable cardinality payload.
    """

    if container_spec is None:
        return None
    return {
        "kind": getattr(container_spec, "kind", None),
        "length": getattr(container_spec, "length", None),
        "num_keys": len(getattr(container_spec, "keys", ()) or ()),
        "num_fields": len(getattr(container_spec, "fields", ()) or ()),
        "num_children": len(getattr(container_spec, "child_specs", ()) or ()),
    }


def compute_graph_shape_hash(model_log: Any) -> str:
    """Compute a deterministic shape hash for a postprocessed graph.

    The hash intentionally excludes run-specific activation values and raw/final
    labels. It includes operation order, normalized function names, parent-edge
    positions, normalized module addresses, output paths, and output container
    cardinality.

    Parameters
    ----------
    model_log:
        Postprocessed ``ModelLog`` with populated ``layer_list``.

    Returns
    -------
    str
        SHA-256 hex digest over the canonical graph-shape payload.
    """

    order_by_label = {layer.layer_label: index for index, layer in enumerate(model_log.layer_list)}
    records = []
    for index, layer in enumerate(model_log.layer_list):
        module_address = normalize_module_address_for_hash(
            getattr(layer, "containing_module", None)
        )
        layer.module_address_normalized = module_address
        parent_indices = sorted(
            order_by_label[parent_label]
            for parent_label in getattr(layer, "parent_layers", ())
            if parent_label in order_by_label
        )
        records.append(
            {
                "index": index,
                "layer_type": getattr(layer, "layer_type", None),
                "func_name": str(getattr(layer, "func_name", None)),
                "parent_indices": parent_indices,
                "module_address_normalized": module_address,
                "output_path": [
                    _hashable_path_component(component)
                    for component in (getattr(layer, "output_path", None) or ())
                ],
                "container_cardinality": _container_cardinality(
                    getattr(layer, "container_spec", None)
                ),
                "is_input": bool(getattr(layer, "is_input_layer", False)),
                "is_output": bool(getattr(layer, "is_output_layer", False)),
                "is_buffer": bool(getattr(layer, "is_buffer_layer", False)),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
