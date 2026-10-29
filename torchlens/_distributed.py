"""Honest detection of distributed (DTensor / device-mesh / TP / PP) model state.

TorchLens captures a *rank-local eager* forward pass. Several distributed
execution modes leave the model looking like an ordinary ``nn.Module`` while
replacing its parameters with tensor subclasses whose real work happens under
``__torch_dispatch__``, below the ``__torch_function__`` layer TorchLens wraps.
The result is not a crash but something worse: a capture that *succeeds* and is
silently wrong.

Concretely, on a ``parallelize_module``-wrapped ``nn.Linear`` (tensor parallel),
capture used to report ``0 total modules`` and ``0 params total; 0 B``, with the
``linear`` op itself absent from the graph and replaced by bare ``viewas``
nodes, while ``tl.compat.report`` reported all-pass for the same model.
All-pass-then-silently-wrong is the worst possible ordering, so this module
centralises the detection used by *both* surfaces:

* :func:`torchlens.compat.report` gains ``dtensor``, ``device_mesh``,
  ``tensor_parallel``, and ``pipeline_parallel`` rows, and
* capture entry (:func:`torchlens._robustness.check_model_and_input_variants`)
  refuses with :class:`DistributedCaptureUnsupportedError` rather than returning
  a wrong trace.

Detection is capability-probed through :mod:`torchlens.utils._torch_compat`
(never a ``torch.__version__`` parse) and costs a handful of ``sys.modules``
lookups when no distributed namespace has been imported, which is the
overwhelmingly common case.

Reporting is deliberately wider than refusal. A bare ``DeviceMesh`` attached to
a model whose parameters are ordinary dense tensors does not make capture wrong,
so it is *reported* but not refused. DTensor/ShardedTensor state, active tensor-
parallel hooks/styles, and pipeline-stage fragments refuse.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .errors._base import CompatibilityError
from .utils._torch_compat import (
    get_device_mesh_type,
    get_dtensor_type,
    get_pipelining_module_types,
)

__all__ = [
    "DistributedCaptureUnsupportedError",
    "DistributedFinding",
    "MAX_REPORTED_SITES",
    "REFUSING_KINDS",
    "check_distributed_capture",
    "detect_distributed_state",
]

MAX_REPORTED_SITES = 5
"""Number of named sites listed in a finding before eliding the remainder."""

REFUSING_KINDS: frozenset[str] = frozenset(
    {"dtensor", "tensor_parallel", "pipeline_parallel", "scan_incomplete"}
)
"""Finding kinds that make a capture provably wrong and therefore refuse it.

``device_mesh`` alone is informational. ``scan_incomplete`` refuses when an
inspection boundary could hide refusing state. Active tensor-parallel styles
refuse even when parameters remain dense: ``PrepareModuleInput`` installs hooks
whose rank redistribution runs below TorchLens' capture layer and would otherwise
be omitted.
"""

# Namespaces whose presence in ``sys.modules`` is a precondition for any live
# distributed object of the corresponding kind existing in this process.
_DISTRIBUTED_MODULE_SENTINELS: tuple[str, ...] = (
    "torch.distributed.tensor",
    "torch.distributed._tensor",
    "torch.distributed.device_mesh",
    "torch.distributed.pipelining",
    "torch.distributed.pipeline",
    "torch.distributed._shard",
)

# Module-path prefixes for the tensor-parallel and pipeline-parallel APIs, used
# to detect wrapper *modules* (as opposed to wrapper tensors) and as the
# structural fallback when an exact type probe is unavailable.
_TENSOR_PARALLEL_NAMESPACES: tuple[str, ...] = (
    "torch.distributed.tensor.parallel",
    "torch.distributed._tensor.parallel",
)
_PIPELINE_PARALLEL_NAMESPACES: tuple[str, ...] = (
    "torch.distributed.pipelining",
    "torch.distributed.pipeline",
)

# Namespaces that must be imported for a *tensor-level* distributed object
# (DTensor, ShardedTensor) to exist. Deliberately excludes
# ``torch.distributed.device_mesh``, which plain ``import torch`` already pulls
# in on modern torch and therefore carries no information.
_SHARDED_TENSOR_MODULE_SENTINELS: tuple[str, ...] = (
    "torch.distributed.tensor",
    "torch.distributed._tensor",
    "torch.distributed._shard",
)

_DOCS_POINTER = (
    "See 'Distributed and Sharded Execution' in LIMITATIONS.md; run "
    "torchlens.compat.report(model, x) for the full compatibility table."
)

# ``nn.Module`` owns these instance attributes itself; none can hold a
# user-supplied device mesh or pipeline object, and together they are 17 of the
# ~25 entries in a typical module ``__dict__``. Probed from a live bare module
# rather than hardcoded, so the skip set tracks torch's own bookkeeping across
# versions instead of going stale.
_NN_MODULE_INTERNAL_ATTRS: frozenset[str] = frozenset(vars(nn.Module()))

# Attribute value types that can never *be* a mesh/pipeline object. Containers
# are absent on purpose: a user may hold a mesh in a small list or dict, and
# :func:`_iter_candidate_attribute_values` performs a bounded recursive walk.
_SKIP_ATTRIBUTE_TYPES: frozenset[type] = frozenset(
    {bool, int, float, complex, str, bytes, bytearray, type(None), torch.dtype, torch.device}
)

_MAX_OBJECT_WALK_DEPTH = 12
_MAX_OBJECT_WALK_NODES = 4096


class DistributedCaptureUnsupportedError(CompatibilityError, RuntimeError):
    """Raised when capture is asked to trace sharded/distributed model state.

    The structured findings are retained on ``fields["findings"]`` so callers
    branch on :attr:`DistributedFinding.kind` rather than parsing message text.
    """


@dataclass(frozen=True)
class DistributedFinding:
    """One detected distributed-execution condition.

    Parameters
    ----------
    kind:
        Stable machine-readable condition key: ``"dtensor"``,
        ``"device_mesh"``, ``"tensor_parallel"``, ``"pipeline_parallel"``, or
        ``"scan_incomplete"``.
    detail:
        Explanation of what was detected and why capture cannot be trusted.
    suggestion:
        Concrete next action for the user.
    sites:
        Named model sites (parameter/buffer/input positions, or input
        positions) where the condition was observed.
    exact:
        Whether detection matched an exact probed torch type (``True``) or a
        structural fallback (``False``), so the compat row can be honest about
        the strength of the evidence.
    geometry:
        Per-site dual-geometry records aligned with ``sites`` (``None`` for a
        site without one). For DTensor state each record declares BOTH the
        logical shape and the locally held shard
        (:func:`torchlens.distributed._dtensor.dtensor_dual_geometry`), so a
        refused sharded model is refused with its parameters precisely
        identified rather than silently mis-counted.
    """

    kind: str
    detail: str
    suggestion: str
    sites: tuple[str, ...] = ()
    exact: bool = True
    geometry: tuple[dict | None, ...] = ()

    @property
    def refuses_capture(self) -> bool:
        """Whether this finding makes capture refuse.

        Returns
        -------
        bool
            True when :attr:`kind` is in :data:`REFUSING_KINDS`.
        """

        return self.kind in REFUSING_KINDS

    def describe_sites(self) -> str:
        """Return a bounded, human-readable rendering of ``sites``.

        Returns
        -------
        str
            Comma-separated site list truncated to :data:`MAX_REPORTED_SITES`
            entries with an explicit remainder count, or ``""`` when no sites
            were recorded.
        """

        if not self.sites:
            return ""
        shown = self.sites[:MAX_REPORTED_SITES]
        rendered = ", ".join(shown)
        remaining = len(self.sites) - len(shown)
        if remaining > 0:
            rendered = f"{rendered} (+{remaining} more)"
        return rendered


@dataclass
class _Evidence:
    """Mutable per-kind site collector used while walking a model and inputs."""

    dtensor_sites: list[str] = field(default_factory=list)
    dtensor_geometry: dict[str, dict | None] = field(default_factory=dict)
    dtensor_exact: bool = True
    dtensor_sharded: bool = False
    shard_sites: list[str] = field(default_factory=list)
    mesh_sites: list[str] = field(default_factory=list)
    mesh_exact: bool = True
    mesh_descriptions: list[str] = field(default_factory=list)
    tp_module_sites: list[str] = field(default_factory=list)
    tp_hook_sites: list[str] = field(default_factory=list)
    tp_module_exact: bool = True
    pp_sites: list[str] = field(default_factory=list)
    pp_exact: bool = True
    scan_incomplete_sites: list[str] = field(default_factory=list)


def _distributed_namespace_imported() -> bool:
    """Return whether any distributed namespace has been imported.

    Returns
    -------
    bool
        True when at least one namespace that could define a live DTensor,
        ``DeviceMesh``, sharded tensor, or pipeline stage is in ``sys.modules``.

    Notes
    -----
    This is the cheap gate that keeps the guard off the cost budget for ordinary
    single-process models: Python registers a module before any class from it can
    be instantiated, so an un-imported namespace cannot have live instances.
    """

    return any(name in sys.modules for name in _DISTRIBUTED_MODULE_SENTINELS)


def _sharded_tensor_namespace_imported() -> bool:
    """Return whether a namespace defining sharded tensor subclasses is imported.

    Returns
    -------
    bool
        True when ``DTensor`` or ``ShardedTensor`` could have live instances in
        this process.
    """

    return any(name in sys.modules for name in _SHARDED_TENSOR_MODULE_SENTINELS)


def _distributed_initialized() -> bool | None:
    """Return whether a torch distributed process group is initialized.

    Returns
    -------
    bool | None
        True when ``torch.distributed`` is available and initialized. False
        when distributed support is genuinely absent (unavailable build, or no
        ``torch.distributed`` namespace at all). None when the capability
        probe ITSELF fails -- unknown state, which the caller must treat as a
        scan-completeness gap, never as "not initialized" (a silent False
        here disabled the device-mesh scan fail-open).
    """

    try:
        if not torch.distributed.is_available():
            return False
        return bool(torch.distributed.is_initialized())
    except AttributeError:
        # Builds without distributed support lack the namespace entirely: a
        # genuine capability absence, not a failed probe.
        return False
    except Exception:
        return None


def _type_in_namespace(value: Any, prefixes: Sequence[str]) -> bool:
    """Return whether ``value``'s type or a base lives under a listed namespace.

    Parameters
    ----------
    value:
        Object whose type MRO is inspected.
    prefixes:
        Dotted module-path prefixes. A prefix matches a module that equals it or
        is a dotted descendant, so ``"torch.distributed.pipelining"`` matches
        ``torch.distributed.pipelining.stage`` but not
        ``torch.distributed.pipeliningx``.

    Returns
    -------
    bool
        True when any MRO entry is defined under a listed namespace.
    """

    try:
        mro = type(value).__mro__
    except Exception:
        return False
    for klass in mro:
        module = getattr(klass, "__module__", "") or ""
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes):
            return True
    return False


def _classify_tensor(value: Any) -> tuple[str | None, bool]:
    """Classify a tensor as DTensor, sharded, or ordinary.

    Parameters
    ----------
    value:
        Candidate tensor.

    Returns
    -------
    tuple[str | None, bool]
        ``(kind, exact)`` where ``kind`` is ``"dtensor"``, ``"sharded"``, or
        ``None``. ``exact`` is True when an exact probed torch type matched, and
        False when only the structural type-name fallback did.
    """

    # Fast exit for the overwhelmingly common case: an exact dense tensor can
    # never be a distributed subclass, and this check costs one identity compare.
    if type(value) is torch.Tensor:
        return None, True

    dtensor_type = get_dtensor_type()
    if dtensor_type is not None and isinstance(value, dtensor_type):
        return "dtensor", True

    value_type = type(value)
    type_name = value_type.__name__.lower()
    module_name = (value_type.__module__ or "").lower()
    # ShardedTensor is tested first, and the DTensor name test is an equality
    # rather than a substring: "dtensor" is a substring of "shardedtensor"
    # ("shar-dtensor"), so a substring test here reported ShardedTensor state as
    # DTensor state. Both refuse capture, but the reported variant must be true.
    if "shardedtensor" in type_name or ("distributed" in module_name and "shard" in module_name):
        return "sharded", False
    if type_name == "dtensor" and "distributed" in module_name:
        return "dtensor", False
    return None, True


def _dtensor_is_sharded(value: Any) -> bool:
    """Return whether a DTensor carries at least one non-replicated placement.

    Parameters
    ----------
    value:
        DTensor to inspect.

    Returns
    -------
    bool
        True when any placement is not a plain ``Replicate``. A fully replicated
        DTensor still breaks capture, but only a *sharded* placement means the
        rank-local tensor holds a fraction of the logical parameter, which is the
        distinction the tensor-parallel row reports on.
    """

    placements = getattr(value, "placements", None)
    if not placements:
        return False
    return any(type(placement).__name__ != "Replicate" for placement in placements)


def _describe_mesh(mesh: Any) -> str:
    """Return a short, exception-safe description of a device mesh.

    Parameters
    ----------
    mesh:
        Device-mesh-like object.

    Returns
    -------
    str
        Bounded description, or the type name when introspection fails.
    """

    try:
        shape = tuple(getattr(mesh, "shape", ()) or ())
        device_type = getattr(mesh, "device_type", None)
        names = getattr(mesh, "mesh_dim_names", None)
        parts = [f"shape={shape}"]
        if device_type:
            parts.append(f"device_type={device_type!r}")
        if names:
            parts.append(f"dim_names={tuple(names)}")
        return f"DeviceMesh({', '.join(parts)})"
    except Exception:
        return type(mesh).__name__


def _is_device_mesh(value: Any) -> tuple[bool, bool]:
    """Return whether ``value`` is a device mesh, and whether the match is exact.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    tuple[bool, bool]
        ``(detected, exact)``.
    """

    mesh_type = get_device_mesh_type()
    if mesh_type is not None and isinstance(value, mesh_type):
        return True, True
    value_type = type(value)
    if value_type.__name__ == "DeviceMesh" and "distributed" in (value_type.__module__ or ""):
        return True, False
    return False, True


def _iter_named_state(
    model: nn.Module,
    evidence: _Evidence,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` for every parameter and buffer without dedupe.

    Parameters
    ----------
    model:
        Model to inspect.
    evidence:
        Collector that receives a fail-closed scan-incomplete site.

    Yields
    ------
    tuple[str, torch.Tensor]
        Qualified state name and tensor. Duplicates are preserved so a shared
        DTensor is reported under every name that exposes it.

    Notes
    -----
    ``remove_duplicate=False`` is requested when supported so tied parameters
    are not hidden; any failure degrades to the default generators rather than
    aborting detection.
    """

    for accessor in ("named_parameters", "named_buffers"):
        getter = getattr(model, accessor, None)
        if getter is None:
            continue
        try:
            items = list(getter(remove_duplicate=False))
        except Exception:
            try:
                items = list(getter())
            except Exception:
                evidence.scan_incomplete_sites.append(f"model.{accessor}()")
                continue
        for name, tensor in items:
            if isinstance(tensor, torch.Tensor):
                yield (name or "<unnamed>"), tensor


def _iter_input_tensors(payload: Any) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(path, tensor)`` through builtin and inspectable user containers.

    Parameters
    ----------
    payload:
        Input tree to walk. Tensors, builtin containers, and inspectable user
        containers are traversed; ``nn.Module`` instances are not descended into.

    Yields
    ------
    tuple[str, torch.Tensor]
        Dotted access path and the tensor found there.
    """

    yield from _walk_inputs(payload, "input", set())


def _walk_inputs(payload: Any, path: str, seen: set[int]) -> Iterator[tuple[str, torch.Tensor]]:
    """Recursive worker for :func:`_iter_input_tensors`.

    Parameters
    ----------
    payload:
        Current node in the input tree.
    path:
        Access path rendered so far.
    seen:
        Visited object ids, so shared and cyclic containers terminate.

    Yields
    ------
    tuple[str, torch.Tensor]
        Dotted access path and the tensor found there.
    """

    yield from _walk_tensors(payload, path, seen, depth=0, nodes=[0])


def _walk_tensors(
    payload: Any,
    path: str,
    seen: set[int],
    *,
    depth: int,
    nodes: list[int],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Walk tensors through builtin and inspectable user containers.

    Parameters
    ----------
    payload:
        Current object.
    path:
        Rendered access path.
    seen:
        Object identities already visited.
    depth:
        Current recursion depth.
    nodes:
        Mutable one-item node counter shared by the traversal.

    Yields
    ------
    tuple[str, torch.Tensor]
        Tensor path and value.
    """

    if depth > _MAX_OBJECT_WALK_DEPTH or nodes[0] >= _MAX_OBJECT_WALK_NODES:
        return
    payload_id = id(payload)
    if payload_id in seen:
        return
    seen.add(payload_id)
    nodes[0] += 1
    if isinstance(payload, torch.Tensor):
        yield path, payload
        return
    if isinstance(payload, nn.Module):
        return
    if isinstance(payload, (list, tuple)):
        for index, item in enumerate(payload):
            yield from _walk_tensors(item, f"{path}[{index}]", seen, depth=depth + 1, nodes=nodes)
        return
    if isinstance(payload, (set, frozenset)):
        for index, item in enumerate(payload):
            yield from _walk_tensors(item, f"{path}{{{index}}}", seen, depth=depth + 1, nodes=nodes)
        return
    if isinstance(payload, dict):
        for key, item in payload.items():
            yield from _walk_tensors(item, f"{path}[{key!r}]", seen, depth=depth + 1, nodes=nodes)
        return
    try:
        attributes = vars(payload)
    except (TypeError, AttributeError):
        return
    for attr_name, item in attributes.items():
        yield from _walk_tensors(item, f"{path}.{attr_name}", seen, depth=depth + 1, nodes=nodes)


def _collect_module_evidence(model: nn.Module, evidence: _Evidence) -> None:
    """Record tensor-parallel, pipeline, and device-mesh module-level evidence.

    Parameters
    ----------
    model:
        Model to inspect.
    evidence:
        Collector mutated in place.

    Notes
    -----
    The type probes are hoisted out of the per-module loop, and the two
    namespace-based structural checks are skipped entirely unless the matching
    namespace has actually been imported. Both matter: this runs at capture
    entry on every trace, and an unhoisted MRO walk per module attribute cost
    ~44 ms on resnet50.
    """

    pipelining_types = get_pipelining_module_types()
    check_tp_namespace = any(name in sys.modules for name in _TENSOR_PARALLEL_NAMESPACES)
    check_pp_namespace = any(name in sys.modules for name in _PIPELINE_PARALLEL_NAMESPACES)
    # The device-mesh attribute scan is the only part of this walk that runs on an
    # ordinary single-process model, because plain ``import torch`` imports
    # ``torch.distributed.device_mesh``. Gate it on distributed actually being
    # live: a mesh cannot be built without an initialized process group in any
    # supported flow, and a mesh with no sharded tensors is an informational row
    # that never refuses capture, so the refusing path stays fully sound while an
    # ordinary trace pays nothing. See FORKS.md ("device-mesh scan gate").
    scan_tensors = _sharded_tensor_namespace_imported()
    initialized = _distributed_initialized()
    if initialized is None:
        # The capability probe failed: absence of live distributed state
        # cannot be established, so disclose the gap (a refusing
        # scan_incomplete finding) and scan meshes conservatively anyway.
        evidence.scan_incomplete_sites.append("torch.distributed.is_initialized()")
    scan_meshes = scan_tensors or initialized is not False
    mesh_type = get_device_mesh_type() if scan_meshes else None
    if (
        not scan_meshes
        and not pipelining_types
        and not check_tp_namespace
        and not check_pp_namespace
    ):
        return

    try:
        named_modules = list(model.named_modules())
    except Exception:
        evidence.scan_incomplete_sites.append("model.named_modules()")
        named_modules = [("", model)]

    for name, module in named_modules:
        label = name or "<root>"
        if check_tp_namespace and _type_in_namespace(module, _TENSOR_PARALLEL_NAMESPACES):
            evidence.tp_module_sites.append(label)
        if check_tp_namespace:
            _collect_tp_hook_evidence(module, label, evidence)
        if pipelining_types and isinstance(module, pipelining_types):
            evidence.pp_sites.append(label)
        elif check_pp_namespace and _type_in_namespace(module, _PIPELINE_PARALLEL_NAMESPACES):
            evidence.pp_sites.append(label)
            evidence.pp_exact = False
        _collect_attribute_evidence(
            module,
            label,
            evidence,
            scan_meshes,
            mesh_type,
            pipelining_types,
            check_pp_namespace,
            scan_tensors,
        )


def _collect_tp_hook_evidence(module: nn.Module, label: str, evidence: _Evidence) -> None:
    """Record active tensor-parallel forward hooks on one module.

    Parameters
    ----------
    module:
        Module whose hook registries are inspected directly.
    label:
        Module path used in findings.
    evidence:
        Mutable evidence collector.
    """

    for registry_name in ("_forward_pre_hooks", "_forward_hooks"):
        registry = vars(module).get(registry_name)
        if not isinstance(registry, dict):
            continue
        for hook_id, hook in registry.items():
            module_name = str(getattr(hook, "__module__", "") or "")
            qualname = str(getattr(hook, "__qualname__", "") or "")
            if any(
                module_name == prefix or module_name.startswith(f"{prefix}.")
                for prefix in _TENSOR_PARALLEL_NAMESPACES
            ):
                evidence.tp_hook_sites.append(
                    f"{label}.{registry_name}[{hook_id!r}] ({qualname or type(hook).__name__})"
                )


def _iter_candidate_attribute_values(module: nn.Module) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, value)`` for module attributes that could be mesh/pipeline objects.

    Parameters
    ----------
    module:
        Module whose instance ``__dict__`` is scanned.

    Yields
    ------
    tuple[str, Any]
        Attribute access path and candidate value.

    Notes
    -----
    Only instance ``__dict__`` mappings are read, never descriptors. Builtin and
    user-defined containers are traversed to a documented global node/depth cap.
    """

    try:
        attributes = vars(module)
    except (TypeError, AttributeError):
        return
    seen: set[int] = {id(module)}
    nodes = [0]
    for attr_name, value in attributes.items():
        if attr_name in _NN_MODULE_INTERNAL_ATTRS:
            continue
        yield from _walk_attribute_values(value, attr_name, seen, depth=0, nodes=nodes)


def _walk_attribute_values(
    value: Any,
    path: str,
    seen: set[int],
    *,
    depth: int,
    nodes: list[int],
) -> Iterator[tuple[str, Any]]:
    """Yield inspectable nested module-attribute values within bounded work.

    Parameters
    ----------
    value:
        Current value.
    path:
        Rendered module-relative path.
    seen:
        Object identities already visited.
    depth:
        Current recursion depth.
    nodes:
        Shared node counter.

    Yields
    ------
    tuple[str, Any]
        Candidate path and value.
    """

    if depth > _MAX_OBJECT_WALK_DEPTH or nodes[0] >= _MAX_OBJECT_WALK_NODES:
        return
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    nodes[0] += 1
    if type(value) in _SKIP_ATTRIBUTE_TYPES:
        return
    yield path, value
    if isinstance(value, (torch.Tensor, nn.Module)):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_attribute_values(
                item, f"{path}[{index}]", seen, depth=depth + 1, nodes=nodes
            )
        return
    if isinstance(value, (set, frozenset)):
        for index, item in enumerate(value):
            yield from _walk_attribute_values(
                item, f"{path}{{{index}}}", seen, depth=depth + 1, nodes=nodes
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_attribute_values(
                item, f"{path}[{key!r}]", seen, depth=depth + 1, nodes=nodes
            )
        return
    try:
        attributes = vars(value)
    except (TypeError, AttributeError):
        return
    for attr_name, item in attributes.items():
        yield from _walk_attribute_values(
            item, f"{path}.{attr_name}", seen, depth=depth + 1, nodes=nodes
        )


def _collect_attribute_evidence(
    module: nn.Module,
    label: str,
    evidence: _Evidence,
    scan_meshes: bool,
    mesh_type: type[Any] | None,
    pipelining_types: tuple[type[Any], ...],
    check_pp_namespace: bool,
    scan_tensors: bool,
) -> None:
    """Record device-mesh and pipeline objects held as plain module attributes.

    Parameters
    ----------
    module:
        Module whose ``__dict__`` is scanned.
    label:
        Module path used to name findings.
    evidence:
        Collector mutated in place.
    scan_meshes:
        Whether device-mesh detection is enabled for this walk.
    mesh_type:
        Probed ``DeviceMesh`` type, or ``None`` when unavailable or not scanned.
    pipelining_types:
        Probed pipeline-parallel stage types, empty when unavailable.
    check_pp_namespace:
        Whether the structural pipeline-namespace fallback is worth running.
    scan_tensors:
        Whether distributed tensor subclasses can exist in this process.
    """

    for attr_path, value in _iter_candidate_attribute_values(module):
        if isinstance(value, torch.Tensor):
            if scan_tensors:
                _record_tensor(f"{label}.{attr_path}", value, evidence)
            continue
        if isinstance(value, nn.Module):
            continue
        if scan_meshes:
            if mesh_type is not None:
                if isinstance(value, mesh_type):
                    evidence.mesh_sites.append(f"{label}.{attr_path}")
                    evidence.mesh_descriptions.append(_describe_mesh(value))
                    continue
            else:
                detected, exact = _is_device_mesh(value)
                if detected:
                    evidence.mesh_sites.append(f"{label}.{attr_path}")
                    evidence.mesh_descriptions.append(_describe_mesh(value))
                    if not exact:
                        evidence.mesh_exact = False
                    continue
        if pipelining_types and isinstance(value, pipelining_types):
            evidence.pp_sites.append(f"{label}.{attr_path}")
        elif check_pp_namespace and _type_in_namespace(value, _PIPELINE_PARALLEL_NAMESPACES):
            evidence.pp_sites.append(f"{label}.{attr_path}")
            evidence.pp_exact = False


def _record_tensor(site: str, tensor: torch.Tensor, evidence: _Evidence) -> None:
    """Classify one tensor and record any distributed evidence it carries.

    Parameters
    ----------
    site:
        Name to report for this tensor.
    tensor:
        Tensor to classify.
    evidence:
        Collector mutated in place.
    """

    kind, exact = _classify_tensor(tensor)
    if kind is None:
        return
    if kind == "dtensor":
        evidence.dtensor_sites.append(site)
        try:
            from .distributed._dtensor import dtensor_dual_geometry

            evidence.dtensor_geometry[site] = dtensor_dual_geometry(tensor)
        except Exception:
            evidence.dtensor_geometry[site] = None
        if not exact:
            evidence.dtensor_exact = False
        if _dtensor_is_sharded(tensor):
            evidence.dtensor_sharded = True
        mesh = getattr(tensor, "device_mesh", None)
        if mesh is not None:
            detected, mesh_exact = _is_device_mesh(mesh)
            if detected:
                evidence.mesh_sites.append(f"{site}.device_mesh")
                evidence.mesh_descriptions.append(_describe_mesh(mesh))
                if not mesh_exact:
                    evidence.mesh_exact = False
        return
    evidence.shard_sites.append(site)


def detect_distributed_state(
    model: Any,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
) -> tuple[DistributedFinding, ...]:
    """Detect distributed/sharded state on a model and its inputs.

    Parameters
    ----------
    model:
        Model about to be inspected or captured.
    input_args:
        Positional arguments destined for ``model.forward``. May be a bare
        tensor or a nested container.
    input_kwargs:
        Keyword arguments destined for ``model.forward``.

    Returns
    -------
    tuple[DistributedFinding, ...]
        Findings in stable order: ``scan_incomplete``, ``dtensor``,
        ``tensor_parallel``, ``pipeline_parallel``, ``device_mesh``. Empty when
        nothing distributed was detected and the scan completed.

    Notes
    -----
    Returns immediately when no distributed namespace has been imported, so the
    ordinary single-process path pays only a handful of ``sys.modules`` lookups.
    """

    if not _distributed_namespace_imported():
        return ()

    evidence = _Evidence()
    # A DTensor/ShardedTensor instance cannot exist unless its defining module is
    # imported, so when none is, skip the tensor walk entirely. The module-level
    # walk still runs: ``torch.distributed.device_mesh`` is imported by plain
    # ``import torch``, so a mesh can be present with no sharded tensors at all.
    scan_tensors = _sharded_tensor_namespace_imported()

    if isinstance(model, nn.Module):
        if scan_tensors:
            for name, tensor in _iter_named_state(model, evidence):
                _record_tensor(name, tensor, evidence)
        _collect_module_evidence(model, evidence)

    if scan_tensors:
        for path, tensor in _iter_input_tensors(input_args):
            _record_tensor(path, tensor, evidence)
        if input_kwargs:
            for path, tensor in _iter_input_tensors(dict(input_kwargs)):
                _record_tensor(path, tensor, evidence)

    return _build_findings(evidence)


def _build_findings(evidence: _Evidence) -> tuple[DistributedFinding, ...]:
    """Assemble ordered findings from collected evidence.

    Parameters
    ----------
    evidence:
        Collector populated by the model/input walk.

    Returns
    -------
    tuple[DistributedFinding, ...]
        Findings in stable reporting order.
    """

    findings: list[DistributedFinding] = []

    incomplete_sites = tuple(dict.fromkeys(evidence.scan_incomplete_sites))
    if incomplete_sites:
        findings.append(
            DistributedFinding(
                kind="scan_incomplete",
                detail=(
                    "The bounded distributed-state entry scan could not read one or more "
                    "model enumeration surfaces, so absence of sharded or parallel state "
                    "cannot be established."
                ),
                suggestion=(
                    "Enter the wrapper's state-materialization context or expose standard "
                    "named_parameters/named_buffers/named_modules accessors, then retry."
                ),
                sites=incomplete_sites,
                exact=True,
            )
        )

    sharded_tensor_sites = tuple(dict.fromkeys(evidence.dtensor_sites + evidence.shard_sites))
    if sharded_tensor_sites:
        variants: list[str] = []
        if evidence.dtensor_sites:
            variants.append("DTensor")
        if evidence.shard_sites:
            variants.append("ShardedTensor")
        geometry = tuple(evidence.dtensor_geometry.get(site) for site in sharded_tensor_sites)
        identity_summary = ""
        logical_total = sum(
            record["logical_numel"]
            for record in geometry
            if record and record.get("logical_numel") is not None
        )
        local_total = sum(
            record["local_numel"]
            for record in geometry
            if record and record.get("local_numel") is not None
        )
        if logical_total:
            identity_summary = (
                f" Identified logical state: {logical_total} logical element(s) "
                f"across the DTensor site(s), of which this rank physically "
                f"holds {local_total}."
            )
        findings.append(
            DistributedFinding(
                kind="dtensor",
                detail=(
                    f"{'/'.join(variants)} state detected on "
                    f"{len(sharded_tensor_sites)} parameter/buffer/input site(s). "
                    "These are tensor subclasses whose real work happens under "
                    "__torch_dispatch__, below the __torch_function__ layer TorchLens "
                    "wraps, so capture records rank-local view/reshape shims instead of "
                    f"the real ops and reports zero parameters.{identity_summary}"
                ),
                suggestion=(
                    "Capture a rank-local dense module instead: build the unsharded "
                    "module and trace it before distributing it, or materialize the "
                    "logical tensors (DTensor.full_tensor()) into a plain module first."
                ),
                sites=sharded_tensor_sites,
                exact=evidence.dtensor_exact,
                geometry=geometry,
            )
        )

    tp_sites = tuple(dict.fromkeys(evidence.tp_module_sites + evidence.tp_hook_sites))
    if tp_sites or evidence.dtensor_sharded:
        if evidence.tp_hook_sites:
            detail = (
                f"Active torch.distributed.tensor.parallel forward hook(s) detected at "
                f"{len(evidence.tp_hook_sites)} module site(s). Dense parameters do not make "
                "this safe: input/output redistribution and collectives execute below "
                "TorchLens' wrapped layer."
            )
        elif tp_sites:
            detail = (
                f"torch.distributed.tensor.parallel style/wrapper classes detected at "
                f"{len(tp_sites)} module site(s)."
            )
        else:
            detail = (
                "Parameters are DTensors with at least one non-replicated placement, "
                "the signature left by tensor-parallel parallelize_module (FSDP2 "
                "fully_shard produces the same shape of state). The rank-local tensor "
                "holds only a fraction of each logical parameter."
            )
        findings.append(
            DistributedFinding(
                kind="tensor_parallel",
                detail=(
                    f"{detail} TorchLens captures a single rank, so per-rank shapes and "
                    "parameter counts are fractions of the logical model and collectives "
                    "are invisible to capture."
                ),
                suggestion=(
                    "Trace the undistributed module. Cross-rank merging "
                    "(tl.merge_ranks) covers dense-parameter captures with "
                    "explicit collectives; DTensor/TP capture stays refused "
                    "until its capture fidelity is proven (planned design: "
                    "docs/reference/tensor_parallel_design.md)."
                ),
                sites=tp_sites,
                exact=evidence.tp_module_exact,
            )
        )

    pp_sites = tuple(dict.fromkeys(evidence.pp_sites))
    if pp_sites:
        findings.append(
            DistributedFinding(
                kind="pipeline_parallel",
                detail=(
                    f"torch.distributed pipeline-parallel stage/schedule objects detected "
                    f"at {len(pp_sites)} site(s). A pipeline stage holds only its own slice "
                    "of the model and its execution is driven by a schedule that runs "
                    "microbatches outside the traced forward, so a capture here is a "
                    "fragment, not the model."
                ),
                suggestion=(
                    "Trace the unsplit module before building pipeline stages, or trace a "
                    "single stage's underlying submodule directly and interpret it as a "
                    "fragment."
                ),
                sites=pp_sites,
                exact=evidence.pp_exact,
            )
        )

    mesh_sites = tuple(dict.fromkeys(evidence.mesh_sites))
    if mesh_sites:
        descriptions = tuple(dict.fromkeys(evidence.mesh_descriptions))
        rendered = "; ".join(descriptions[:MAX_REPORTED_SITES])
        findings.append(
            DistributedFinding(
                kind="device_mesh",
                detail=(
                    f"Device mesh detected at {len(mesh_sites)} site(s): {rendered}. "
                    "A mesh by itself does not corrupt capture; it is reported because it "
                    "identifies the distributed topology whose sharded tensors do."
                ),
                suggestion=(
                    "No action needed if the traced parameters are dense; see the dtensor "
                    "row when they are not."
                ),
                sites=mesh_sites,
                exact=evidence.mesh_exact,
            )
        )

    return tuple(findings)


def check_distributed_capture(
    model: Any,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
) -> tuple[DistributedFinding, ...]:
    """Refuse capture of distributed state that would produce a wrong trace.

    Parameters
    ----------
    model:
        Model about to be captured.
    input_args:
        Positional arguments destined for ``model.forward``.
    input_kwargs:
        Keyword arguments destined for ``model.forward``.

    Returns
    -------
    tuple[DistributedFinding, ...]
        All findings detected, including non-refusing ones, so callers can
        surface them without re-running detection.

    Raises
    ------
    DistributedCaptureUnsupportedError
        When any finding's kind is in :data:`REFUSING_KINDS`.
    """

    findings = detect_distributed_state(model, input_args, input_kwargs)
    refusing = tuple(finding for finding in findings if finding.refuses_capture)
    if not refusing:
        return findings

    bullets = []
    for finding in refusing:
        sites = finding.describe_sites()
        bullet = f"  - [{finding.kind}] {finding.detail}"
        if sites:
            bullet = f"{bullet}\n    sites: {sites}"
        bullet = f"{bullet}\n    suggestion: {finding.suggestion}"
        bullets.append(bullet)
    raise DistributedCaptureUnsupportedError(
        "torchlens cannot capture this model: it holds distributed/sharded state that "
        "TorchLens would record incorrectly rather than fail on.\n"
        + "\n".join(bullets)
        + f"\n\n{_DOCS_POINTER}",
        findings=refusing,
    )
