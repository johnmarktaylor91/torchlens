"""Parameter indexes and buffer ownership invariants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _module_claim_address,
        _module_claims,
    )

__all__ = (
    "_param_log_list_contains_param",
    "_param_address_index",
    "_check_buffer_xrefs",
    "_check_buffer_static_versions",
    "_buffer_address_has_module_ancestor",
    "_check_buffer_semantic_ownership",
    "_owner_module_address_for_buffer",
    "_module_addresses_for_buffer_version",
    "_active_buffer_consumer_module_addresses",
    "_check_buffer_write_versions",
    "_resolve_trace_label",
    "_check_buffer_replay_validated_versions",
)


def _param_log_list_contains_param(param_logs: object, param: object) -> bool:
    """Return whether a parameter log collection contains ``param`` by identity.

    Parameters
    ----------
    param_logs:
        Iterable of Param records.
    param:
        Param record to look for.

    Returns
    -------
    bool
        ``True`` when any log has the same address or barcode as ``param``.
    """

    address = getattr(param, "address", None)
    barcode = getattr(param, "barcode", None)
    if not isinstance(param_logs, Iterable):
        return False
    return any(
        candidate is param
        or (
            address is not None
            and getattr(candidate, "address", None) == address
            and getattr(candidate, "barcode", None) == barcode
        )
        for candidate in param_logs or ()
    )


def _param_address_index(ml: Trace) -> dict[object, object]:
    """Build a one-pass primary/alias address -> Param index.

    Replaces a per-lookup linear scan over ``ml._param_logs``. Co-parent
    resolution runs once per param, so the scan made ``param_xrefs`` quadratic in
    the parameter count: every weight-tied or bias/weight sibling link re-walked
    the whole parameter list (800 lookups x 800 params on an 800-layer chain).

    First writer wins, which reproduces the scan's short-circuit order exactly:
    the scan returned the FIRST param matching on either its primary address or
    one of its aliases, and within one param it tested the primary before the
    aliases. Iterating params in the same order and refusing to overwrite an
    existing key therefore resolves every address to the same Param the scan
    picked.

    The ``address`` key is inserted even when it is ``None`` (missing or unset
    attribute). The scan compared with ``getattr(param, "address", None) ==
    address``, so a ``None`` query legitimately matched an addressless param;
    dropping such keys would silently turn that into an unresolved-co-parent
    failure. Indexing requires hashable keys where the scan only needed ``==``.
    That is satisfied: primary addresses already key ``ParamAccessor._dict``, and
    ``all_addresses`` is seeded with the primary address and only ever grows by
    appending module-path alias strings (weight tying), so every key is a ``str``.

    Parameters
    ----------
    ml:
        Trace containing parameter metadata.

    Returns
    -------
    dict[object, object]
        Mapping from every primary and alias address to its owning Param.
    """

    # Iterate the accessor rather than its backing list: `ParamAccessor.__iter__`
    # may re-resolve released live-param references, and the scan this replaces
    # went through the same path. `_check_param_xrefs` has already iterated every
    # param before any co-parent lookup happens, so this pass adds no resolution
    # the check did not already perform.
    index: dict[object, object] = {}
    for param in ml._param_logs:
        address = getattr(param, "address", None)
        if address not in index:
            index[address] = param
        for alias in getattr(param, "all_addresses", None) or ():
            if alias not in index:
                index[alias] = param
    return index


def _check_buffer_xrefs(ml: Trace) -> None:
    """Check K: buffer layer and Buffer cross-references.

    Precondition contract: torch registered buffers are represented by
    ``Buffer`` entities whose versions are buffer Op nodes. Static read
    versions and write versions share the same ``Buffer.versions`` list, so
    write-only fields are only required for versions with
    ``buffer_write_kind is not None``. Buffer addresses may live under an
    uncalled child module, a container, or a top-level attribute; resolving any
    ancestor module is sufficient. ``buffer_source`` is required to resolve
    only when it is populated, because selective materialization may null it
    when the producer raw label did not survive. ``buffer_replay_validated`` is
    asserted as backed only for write versions that explicitly set it to
    ``True``; static init-only buffers and write versions with ``None``/``False``
    do not claim successful identity replay.

    Validates:
    - buffer_layers list entries are valid layer labels.
    - static Buffer version nodes resolve to buffer Ops at the same address.
    - write versions have valid write-kind domains and dense pass sets per address.
    - populated source and replay-validation metadata are backed by resolvable evidence.
    """
    name = "buffer_xrefs"
    label_set = set(ml.layer_labels)

    for lbl in ml.buffer_layers:
        if lbl not in label_set:
            raise MetadataInvariantError(
                name, f"buffer_layers contains '{lbl}' not in layer_labels"
            )

    # Check Buffer objects via buffer accessor
    if hasattr(ml, "_buffer_accessor") and ml._buffer_accessor is not None:
        for buf in ml.buffers:
            _check_buffer_static_versions(ml, buf, name)
            _check_buffer_semantic_ownership(ml, buf, name)
            _check_buffer_write_versions(ml, buf, name)
            _check_buffer_replay_validated_versions(ml, buf, name)


def _check_buffer_static_versions(ml: Trace, buf: object, name: str) -> None:
    """Check static Buffer entity/version structure.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If the Buffer entity has no address, no versions, non-buffer versions,
        mismatched version addresses, or no acceptable ancestor module.
    """

    address = getattr(buf, "address", None)
    layer_label = getattr(buf, "layer_label", None)
    if not address:
        raise MetadataInvariantError(
            name,
            f"Buffer '{layer_label}' has empty address",
        )
    versions = list(getattr(buf, "versions", ()) or ())
    if not versions:
        raise MetadataInvariantError(
            name,
            f"Buffer '{address}' has no version nodes",
        )
    for version in versions:
        label = getattr(version, "layer_label", type(version).__name__)
        if not getattr(version, "is_buffer", False):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' is not a buffer Op",
            )
        if getattr(version, "address", None) != address:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has address "
                f"{getattr(version, 'address', None)!r}",
            )
        if label not in getattr(ml, "_layer_dict_all_keys", {}):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' does not resolve in trace",
            )

    if not _buffer_address_has_module_ancestor(ml, address):
        raise MetadataInvariantError(
            name,
            f"Buffer '{layer_label}' address='{address}' — no ancestor found in module accessor",
        )


def _buffer_address_has_module_ancestor(ml: Trace, address: str) -> bool:
    """Return whether ``address`` or an ancestor is in the module accessor.

    Parameters
    ----------
    ml:
        Trace containing module metadata.
    address:
        Registered buffer address.

    Returns
    -------
    bool
        ``True`` when the buffer address satisfies the loose ancestry rule.
    """

    addr = address
    found_ancestor = addr in ml.modules
    while not found_ancestor and "." in addr:
        addr = addr.rsplit(".", 1)[0]
        found_ancestor = addr in ml.modules
    return found_ancestor or "" in ml.modules


def _check_buffer_semantic_ownership(ml: Trace, buf: object, name: str) -> None:
    """Check buffer source versions are owned by their module or a consumer.

    Parameters
    ----------
    ml:
        Trace containing module and buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a buffer source version claims a module stack unrelated to the
        registered buffer owner and unrelated to any active consumer op.
    """

    address = getattr(buf, "address", None)
    if not isinstance(address, str) or not address:
        return
    owner_address = _owner_module_address_for_buffer(address)
    for version in list(getattr(buf, "versions", ()) or ()):
        version_label = getattr(version, "layer_label", type(version).__name__)
        module_claims = _module_addresses_for_buffer_version(version)
        if not module_claims:
            continue
        valid_addresses = {owner_address}
        valid_addresses.update(_active_buffer_consumer_module_addresses(ml, version))
        if module_claims.isdisjoint(valid_addresses):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{version_label}' has module stack "
                f"{sorted(module_claims)!r}, expected owner/consumer in "
                f"{sorted(valid_addresses)!r}",
            )


def _owner_module_address_for_buffer(address: str) -> str:
    """Return the owning module address for a registered buffer address.

    Parameters
    ----------
    address:
        Dotted registered buffer address.

    Returns
    -------
    str
        Owning module address, or ``"self"`` for top-level buffers.
    """

    if "." not in address:
        return "self"
    return address.rsplit(".", 1)[0]


def _module_addresses_for_buffer_version(version: object) -> set[str]:
    """Return module addresses claimed by a buffer version node.

    Parameters
    ----------
    version:
        Buffer op version.

    Returns
    -------
    set[str]
        Pass-stripped module addresses.
    """

    claims: set[str] = set()
    for claim in _module_claims(cast("Op", version)):
        address = _module_claim_address(claim)
        if address:
            claims.add(address)
    return claims


def _active_buffer_consumer_module_addresses(ml: Trace, version: object) -> set[str]:
    """Return module addresses for real active consumers of a buffer version.

    Parameters
    ----------
    ml:
        Trace containing the graph.
    version:
        Buffer op version whose children should be inspected.

    Returns
    -------
    set[str]
        Pass-stripped module addresses claimed by child ops consuming the
        buffer version.
    """

    addresses: set[str] = set()
    for child_label in getattr(version, "children", ()) or ():
        try:
            child = ml[child_label]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if getattr(child, "is_output", False):
            continue
        for claim in _module_claims(child):
            address = _module_claim_address(claim)
            if address:
                addresses.add(address)
    return addresses


def _check_buffer_write_versions(ml: Trace, buf: object, name: str) -> None:
    """Check write-version buffer metadata domains and resolvable populated fields.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If write-kind domains are invalid, buffer passes are not dense as a
        set, or populated source labels fail to resolve.
    """

    valid_write_kinds = {"reassign", "inplace", "fused", "data_reassign"}
    address = getattr(buf, "address", None)
    versions = list(getattr(buf, "versions", ()) or ())
    write_versions = [
        version for version in versions if getattr(version, "buffer_write_kind", None) is not None
    ]
    if not write_versions:
        return

    passes: list[int] = [
        buffer_pass
        for version in versions
        if isinstance(buffer_pass := getattr(version, "buffer_pass", None), int)
    ]
    for version in write_versions:
        label = getattr(version, "layer_label", type(version).__name__)
        write_kind = getattr(version, "buffer_write_kind", None)
        if write_kind not in valid_write_kinds:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has invalid buffer_write_kind "
                f"{write_kind!r}",
            )
        buffer_pass = getattr(version, "buffer_pass", None)
        if not isinstance(buffer_pass, int) or buffer_pass < 1:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has invalid buffer_pass {buffer_pass!r}",
            )

        source = getattr(version, "buffer_source", None)
        if source is not None and _resolve_trace_label(ml, source) is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has unresolved buffer_source {source!r}",
            )

    expected_passes = set(range(1, max(passes) + 1))
    if set(passes) != expected_passes:
        raise MetadataInvariantError(
            name,
            f"Buffer '{address}' write buffer_pass values must be dense as a set, got "
            f"{sorted(passes)!r}",
        )


def _resolve_trace_label(ml: Trace, label: str) -> str | None:
    """Resolve a final or raw layer label to a known trace label.

    Parameters
    ----------
    ml:
        Trace containing raw/final lookup maps.
    label:
        Raw or final label to resolve.

    Returns
    -------
    str | None
        Resolved label when present in the trace, otherwise ``None``.
    """

    # Membership is tested against the lookup mapping directly rather than a
    # `set(...)` copy: `x in mapping` is an O(1) key probe with exactly the
    # hash/`__eq__` semantics a set copy of those same keys would have, whereas
    # rebuilding the set made every resolution O(n_labels). This helper is
    # called once per edge-use record and per module boundary, so the copy made
    # `check_metadata_invariants` quadratic in trace size.
    all_keys = getattr(ml, "_layer_dict_all_keys", {})
    final_label = getattr(ml, "_raw_to_final_layer_labels", {}).get(label)
    # `.get(label)` returns None both when `label` genuinely has no raw-label
    # mapping (the common case for an already-final label) and, in principle,
    # if a raw label were ever mapped to a literal ``None`` value. Either way,
    # a `None` `final_label` must never be treated as "resolved": some traces
    # legitimately register a `None` lookup key in `layer_dict_all_keys`
    # (e.g. a layer with an unset `io_role`), so `None in all_keys` can be
    # True even though no real label resolved. Without this guard that
    # collision made every already-final label whose raw form does not
    # remap (i.e. almost all of them) silently resolve to `None` instead of
    # falling through to the `label in all_keys` check below, so a
    # perfectly valid parent reference like "relu_1_2" was reported as
    # "missing" by check_metadata_invariants.
    if final_label is not None and final_label in all_keys:
        return final_label
    if label in all_keys:
        return label
    return None


def _check_buffer_replay_validated_versions(ml: Trace, buf: object, name: str) -> None:
    """Check explicit successful buffer replay claims have identity-replay evidence.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a write version asserts ``buffer_replay_validated=True`` without the
        source-parent and saved-argument evidence created by buffer replay
        postprocessing.
    """

    address = getattr(buf, "address", None)
    for version in getattr(buf, "versions", ()) or ():
        if getattr(version, "buffer_write_kind", None) is None:
            continue
        if getattr(version, "buffer_replay_validated", None) is not True:
            continue
        label = getattr(version, "layer_label", type(version).__name__)
        source = getattr(version, "buffer_source", None)
        if source is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "buffer_source",
            )
        resolved_source = _resolve_trace_label(ml, source)
        if resolved_source is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source {source!r} does not resolve",
            )
        parents = set(getattr(version, "parents", ()) or ())
        if resolved_source not in parents:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source is not a parent",
            )
        parent_args = getattr(version, "parent_arg_positions", {}) or {}
        arg_positions = parent_args.get("args", {}) if isinstance(parent_args, Mapping) else {}
        resolved_arg0 = _resolve_trace_label(ml, arg_positions.get(0, ""))
        if resolved_arg0 != resolved_source:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source is not args[0]",
            )
        if not getattr(version, "saved_args", None):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "saved source argument",
            )
        if (
            getattr(version, "out", None) is None
            or getattr(ml[resolved_source], "out", None) is None
        ):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "comparable payloads",
            )
