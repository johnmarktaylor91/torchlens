"""Barcode generation and graph hash helpers.

Barcodes are short opaque identifiers attached to tensors during logging.
They serve two purposes:

1. **Random barcodes** (``make_random_barcode``): assigned to each tensor as
   it is created during the forward pass. These act as globally unique IDs
   so that the logging pipeline can track tensor identity across operations,
   even when the same ``torch.Tensor`` object is reused by in-place ops.

2. **Deterministic barcodes** (``make_short_barcode_from_input``): derived
   from the *structure* of a tensor's creation arguments (shape/dtype/scalar
   tokens; Parameters and tensor values are excluded). Two tensors with the
   same deterministic barcode
   originated from the same parameter/buffer and are candidates for
   *same-layer grouping* in loop detection — the barcode is the key signal
   that separate forward-pass operations actually reference the same weight.

The graph hash family answers compatibility questions about captured structure:

* ``compute_graph_shape_hash`` hashes a postprocessed graph's operation order,
  layer/function kind, parent topology, output container paths/cardinality, and
  boundary flags. It is shape- and dtype-blind. By default it is sensitive to
  normalized module addresses; pass ``include_module_address=False`` for an
  address-free topology hash used to compare distinct designs independent of
  where modules live in a model.
* ``compute_raw_event_shape_hash`` hashes raw capture events before
  postprocessing. It includes raw op order, function identity, parent topology,
  normalized module addresses, output container metadata, and output shape/dtype.
  A shape- and dtype-preserving output-replacement node (an edit that fired or
  a raw forward hook that swapped a module output) is folded into its parent:
  it is a value edit, not a model operation, so a staged edit never reads as
  control-flow divergence while any op, edge or shape change still does.

Both graph hashes are deterministic within a TorchLens version for equivalent
capture structure. They are not security hashes and are not guaranteed stable
across intentional schema/policy changes.
"""

import hashlib
import json
import random
import re
import string
from typing import Any

_BARCODE_ALPHABET = string.ascii_letters + string.digits

# Dedicated, process-private RNG for internal tensor barcodes. Drawing from a
# private ``random.Random`` (auto-seeded from OS entropy) -- instead of the global
# ``random`` module -- keeps capture from perturbing the user's global Python RNG
# stream, and lets runnable capture honestly detect *user* Python/NumPy RNG use by
# bracketing the forward with host-RNG snapshots (torchlens's own per-tensor barcode
# draws would otherwise masquerade as user host-RNG consumption).
_BARCODE_RNG = random.Random()


def seed_barcode_rng(seed: int) -> None:
    """Seed the process-private barcode RNG for reproducible tensor barcodes.

    The barcode RNG is a torchlens-internal engine kept off the global ``random``
    stream (so barcode draws never masquerade as user host-RNG consumption during a
    forward). It must still track :func:`torchlens.utils.rng.set_random_seed`: a fixed
    capture seed has to produce a reproducible barcode sequence so that a fork replay
    of the same graph (``save_new_outs`` reuses the original capture seed) assigns
    matching barcodes -- otherwise tensor<->op<->param cross-references diverge between
    the original and replayed captures. Seeding a *private* ``random.Random`` keeps the
    host-RNG honesty bracketing intact while restoring that determinism.
    """
    _BARCODE_RNG.seed(seed)


def make_random_barcode(barcode_len: int = 8) -> str:
    """Generate a random alphanumeric identifier for internal tensor tracking.

    These barcodes are invisible to the user and are used as unique
    internal keys for tensor entries in ``Trace``.

    Args:
        barcode_len: Length of the identifier string.

    Returns:
        Random alphanumeric string of the requested length.
    """
    return "".join(_BARCODE_RNG.choices(_BARCODE_ALPHABET, k=barcode_len))


def make_short_barcode_from_input(things_to_hash: list[Any], barcode_len: int = 16) -> str:
    """Produce a deterministic short hash from a list of values.

    Used to create content-based barcodes for parameters and buffers so
    that loop detection can identify operations that share the same weights.
    Each value is encoded as a ``[type_name, repr]`` pair inside a JSON list and
    hashed with SHA-256.  The type tag distinguishes values whose ``str()``
    coincides (``1`` vs ``"1"``), and the JSON list structure -- with its escaped
    string quoting -- prevents both concatenation collisions and adversarial
    forging of the element separator (a value containing the raw separator byte
    can no longer masquerade as two elements, e.g. ``["a\\x00b"]`` vs
    ``["a", "b"]``).  This avoids Python's process-randomized ``hash()`` and the
    collision-prone decimal truncation used by older TorchLens releases.

    Args:
        things_to_hash: Values to hash. Each must be ``repr``-able (the common
            case: shape/dtype/scalar tokens; Parameters and tensor values are
            excluded upstream).
        barcode_len: Maximum length of the returned barcode.

    Returns:
        A deterministic hexadecimal SHA-256 prefix of ``barcode_len`` characters.
    """
    # Type-tagged, structurally-delimited encoding: the enclosing JSON list makes
    # element boundaries unforgeable and the type name disambiguates values whose
    # ``str()`` collides. ``ensure_ascii`` keeps the digest byte-stable regardless
    # of locale/encoding.
    payload = json.dumps(
        [[type(x).__name__, repr(x)] for x in things_to_hash],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:barcode_len]


_MODULE_PASS_SUFFIX_RE = re.compile(r":\d+(?=\.|$)")
_ROLLED_LOOP_INDEX_RE = re.compile(r"\[\d+\]")


def normalize_address_for_hash(address: Any) -> str | None:
    """Normalize a module address for graph-shape hashing.

    Parameters
    ----------
    address:
        Module address in either tuple/list form, pass-qualified string form, or
        ``None``.

    Returns
    -------
    str | None
        Pass- and rolled-loop-normalized module address, or ``None`` when no
        module address is available.
    """

    if address is None:
        return None
    if isinstance(address, (tuple, list)):
        address = ".".join(str(part) for part in address)
    normalized = str(address)
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

    component_type = type(component).__name__
    if component_type == "TupleIndex" and hasattr(component, "index"):
        return {"type": type(component).__name__, "index": component.index}
    if component_type in {"DictKey", "HFKey"} and hasattr(component, "key"):
        return {"type": type(component).__name__, "key": repr(component.key)}
    if component_type in {"NamedField", "DataclassField"} and hasattr(component, "name"):
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


def populate_normalized_layer_addresses(trace: Any) -> None:
    """Populate normalized module addresses on postprocessed layers.

    Parameters
    ----------
    trace:
        Postprocessed ``Trace`` with populated ``layer_list``.

    Returns
    -------
    None
        Each layer's ``_address_normalized`` field is updated in place.
    """

    for layer in trace.layer_list:
        layer._address_normalized = normalize_address_for_hash(getattr(layer, "module", None))


def compute_graph_shape_hash(
    trace: Any,
    *,
    include_module_address: bool = True,
    rerun_signature: bool = False,
) -> str:
    """Compute a deterministic shape hash for a postprocessed graph.

    The hash intentionally excludes run-specific out values and raw/final
    labels. It includes operation order, normalized function names, parent-edge
    positions, output paths, output container cardinality, and, by default,
    normalized module addresses.

    Parameters
    ----------
    trace:
        Postprocessed ``Trace`` with populated ``layer_list``.
    include_module_address:
        Whether normalized module addresses contribute to the digest. The
        default preserves the historical address-sensitive hash.
    rerun_signature:
        Rerun-divergence form, used when a trace carries no raw-event hash (a
        loaded trace): layer output shapes and dtypes join the payload and a
        value-only replacement layer folds into its parent, mirroring
        ``compute_raw_event_shape_hash``. The default payload is unchanged.

    Returns
    -------
    str
        SHA-256 hex digest over the canonical graph-shape payload.
    """

    # ``layer.parents`` references each parent by its FINAL lookup label: the
    # non-pass-qualified ``layer_label`` for single-pass parents but the
    # pass-qualified ``label`` (e.g. ``linear_1_1:2``) for multi-pass/recurrent
    # parents (see ``postprocess/labeling.py`` ``final_lookup_label``). The
    # ordering map must be keyed by that same injective label space: keying by
    # ``layer_label`` alone silently dropped every multi-pass parent edge (the
    # pass-qualified reference never matched) and collapsed all passes of a
    # recurrent layer onto one index, letting structurally different recurrent
    # graphs hash identically and defeating ``tl.hash.assert_unchanged``.
    order_by_label: dict[Any, int] = {}
    layer_by_label: dict[Any, Any] = {}
    kept_layers = []
    for layer in trace.layer_list:
        reference_label = (
            layer.layer_label
            if getattr(layer, "num_passes", 1) == 1
            else getattr(layer, "label", None) or layer.layer_label
        )
        folded_into = (
            _replacement_layer_fold_target(layer, layer_by_label, order_by_label)
            if rerun_signature
            else None
        )
        layer_by_label[reference_label] = layer
        if folded_into is not None:
            order_by_label[reference_label] = folded_into
            continue
        order_by_label[reference_label] = len(kept_layers)
        kept_layers.append(layer)
    records = []
    for index, layer in enumerate(kept_layers):
        address = normalize_address_for_hash(getattr(layer, "module", None))
        hash_address = address if include_module_address else None
        # Preserve parent EDGE ORDER: ``layer.parents`` is an ordered list whose
        # position encodes operand routing. Sorting would make a noncommutative
        # op's ``(a, b)`` and ``(b, a)`` parents hash identically, silently
        # accepting operand-order drift. This mirrors the operand-order-sensitive
        # refresh graph signature (commit 74898ada); the shape hash must not be
        # blind to a distinction the refresh tripwire enforces.
        #
        # EVERY parent edge contributes to the hash input: an unresolvable
        # reference maps to a position-preserving ``None`` sentinel instead of
        # being skipped, so any future label-scheme drift changes the digest
        # loudly rather than silently reintroducing dropped-edge false matches.
        parent_indices = [
            order_by_label.get(parent_label) for parent_label in getattr(layer, "parents", ())
        ]
        records.append(
            {
                "index": index,
                "layer_type": getattr(layer, "layer_type", None),
                "func_name": str(getattr(layer, "func_name", None)),
                "parent_indices": parent_indices,
                "_address_normalized": hash_address,
                "container_path": [
                    _hashable_path_component(component)
                    for component in (getattr(layer, "container_path", None) or ())
                ],
                "container_cardinality": _container_cardinality(
                    getattr(layer, "container_spec", None)
                ),
                "is_input": bool(getattr(layer, "is_input", False)),
                "is_output": bool(getattr(layer, "is_output", False)),
                "is_buffer": bool(getattr(layer, "is_buffer", False)),
            }
        )
        if rerun_signature:
            # Normalized so a live and a loaded layer render alike.
            shape = getattr(layer, "shape", None)
            records[-1]["shape"] = None if shape is None else [int(dim) for dim in shape]
            records[-1]["dtype"] = str(getattr(layer, "dtype", None))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replacement_layer_fold_target(
    layer: Any, layer_by_label: dict[Any, Any], order_by_label: dict[Any, int]
) -> int | None:
    """Return the parent index a value-only replacement layer folds into, if any.

    The postprocessed-graph twin of ``_replacement_fold_target``: only a
    single-parent ``interventionreplacement`` layer whose output keeps its
    parent's shape and dtype folds.
    """

    if getattr(layer, "layer_type", None) != "interventionreplacement":
        return None
    parents = list(getattr(layer, "parents", ()) or ())
    if len(parents) != 1 or parents[0] not in order_by_label:
        return None
    parent = layer_by_label[parents[0]]
    if getattr(layer, "shape", None) != getattr(parent, "shape", None):
        return None
    if getattr(layer, "dtype", None) != getattr(parent, "dtype", None):
        return None
    return order_by_label[parents[0]]


def compute_raw_event_shape_hash(capture_events: Any) -> str:
    """Compute a deterministic topology hash from raw capture events.

    Parameters
    ----------
    capture_events:
        In-flight or remembered ``CaptureEvents`` buffer.

    Returns
    -------
    str
        SHA-256 hex digest over operation order, function names, output shapes,
        normalized parent-edge order indices, and normalized module addresses.
    """

    # Read through the canonical reducer view, never the raw list: the hash is
    # a persisted change-detection key and must keep seeing AMENDED parents
    # (register_tensor_connection) when the raw list becomes append-only (P4).
    # Today the reducer is a no-op passthrough, so this is byte-identical.
    folded_events = (
        capture_events.amended_op_records()
        if hasattr(capture_events, "amended_op_records")
        else capture_events.op_events
    )
    events_by_raw_label = {event.label_raw: event for event in folded_events}
    order_by_raw_label: dict[Any, int] = {}
    kept_events = []
    for event in folded_events:
        folded_into = _replacement_fold_target(event, events_by_raw_label, order_by_raw_label)
        if folded_into is not None:
            order_by_raw_label[event.label_raw] = folded_into
            continue
        order_by_raw_label[event.label_raw] = len(kept_events)
        kept_events.append(event)
    records = []
    for index, event in enumerate(kept_events):
        function = event.function
        output = event.output
        tensor = output.tensor
        # Preserve parent EDGE ORDER (see ``compute_graph_shape_hash``):
        # ``event.parents`` is ordered by operand position, so sorting would
        # discard the very order this hash claims to include.
        parent_indices = [
            order_by_raw_label[parent.parent_label_raw]
            for parent in event.parents
            if parent.parent_label_raw in order_by_raw_label
        ]
        module_addresses = [
            normalize_address_for_hash(address)
            for address, _call_index in getattr(event, "modules", ()) or ()
        ]
        records.append(
            {
                "index": index,
                "kind": event.kind,
                "layer_type": event.layer_type,
                "func_name": getattr(function, "func_name", None),
                "func_qualname": getattr(function, "func_qualname", None),
                "output_shape": getattr(tensor, "shape", None),
                "output_dtype": getattr(tensor, "dtype", None),
                "parent_indices": parent_indices,
                "module_addresses": module_addresses,
                "container_path": [
                    _hashable_path_component(component)
                    for component in (getattr(output, "container_path", None) or ())
                ],
                "container_cardinality": _container_cardinality(
                    getattr(output, "container_spec", None)
                ),
            }
        )
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _replacement_fold_target(
    event: Any, events_by_raw_label: dict[Any, Any], order_by_raw_label: dict[Any, int]
) -> int | None:
    """Return the parent index an output-replacement event folds into, if any.

    Only a single-parent ``interventionreplacement`` event whose output keeps its
    parent's shape and dtype folds: it swaps a value without changing the
    model's operations. Anything else (internal sources, shape or dtype changes,
    unresolved parents) stays in the hash.
    """

    if getattr(event, "layer_type", None) != "interventionreplacement":
        return None
    parents = list(getattr(event, "parents", ()) or ())
    if len(parents) != 1:
        return None
    parent_label = parents[0].parent_label_raw
    parent_event = events_by_raw_label.get(parent_label)
    if parent_event is None or parent_label not in order_by_raw_label:
        return None
    tensor = getattr(event.output, "tensor", None)
    parent_tensor = getattr(parent_event.output, "tensor", None)
    if getattr(tensor, "shape", None) != getattr(parent_tensor, "shape", None):
        return None
    if getattr(tensor, "dtype", None) != getattr(parent_tensor, "dtype", None):
        return None
    return order_by_raw_label[parent_label]
