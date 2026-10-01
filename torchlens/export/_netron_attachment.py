"""Netron ``netron:attachment`` sidecar writer (lane F14, memo D-15).

Netron 9.2.2 implements a first-class metrics sidecar: a second JSON file
whose ``metadata`` and ``metrics`` containers render in dedicated sidebar
sections for models, graphs, nodes, values, and tensors -- the maintainer's
own answer to netron #1240/#1234. The vendor parser is silently lossy, so
this writer enforces every observed drop rule as a typed refusal instead:

- entity names truncate at the first newline before matching -> no emitted
  target may contain a newline;
- within one target, duplicate metric names silently last-write-wins -> the
  writer asserts per-target uniqueness;
- malformed items (missing kind/target/name) are silently dropped -> the
  writer refuses them;
- model/graph-scope rows require ``target: ""`` or they vanish, and unknown
  targets vanish silently -> every emitted target must resolve to a real
  node or value in the selected projection.

The ``key_fn`` seam is the plumbing for the deferred real-ONNX overlay
(attach TorchLens metrics onto a user's own ``torch.onnx.export`` file);
``stats_fn`` is the deferred saved-tensor-statistics provider seam. Both
default to identity/off.
"""

from __future__ import annotations

import json as _json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from ..utils.display import atomic_write_text
from ._netron_fields import _quantity, module_path_of
from ._netron_records import NetronProjection

__tl_layer__ = "L8"

#: Vendor magic marker; netron only merges files carrying this signature.
_ATTACHMENT_SIGNATURE = "netron:attachment"

#: Closed vendor kind vocabulary (view.js Attachment.Container consumers).
_ATTACHMENT_KINDS = frozenset({"model", "graph", "node", "value", "tensor"})


def _attachment_refusal(
    problem: str, remedy: str, *, code: str, **fields: Any
) -> ConfigurationError:
    """Build the typed attachment refusal (every site passes its stable code)."""

    return ConfigurationError(
        f"Netron attachment sidecar refused: {problem}. Remedy: {remedy}.",
        code=code,
        **fields,
    )


def _validate_item(item: dict[str, Any], known_targets: set[str]) -> None:
    """Enforce the four vendor drop rules on one sidecar item, fail-closed."""

    kind = item.get("kind")
    if kind not in _ATTACHMENT_KINDS:
        raise _attachment_refusal(
            f"item kind {kind!r} is not one of {sorted(_ATTACHMENT_KINDS)}",
            "emit only vendor-known kinds",
            code="netron_attachment_invalid",
            kind=str(kind),
        )
    if "target" not in item or not isinstance(item["target"], str):
        raise _attachment_refusal(
            "item is missing its string target (netron drops it silently)",
            "give every item a target string; model/graph scope uses target=''",
            code="netron_attachment_invalid",
            kind=str(kind),
        )
    target = item["target"]
    if "\n" in target or "\n" in str(item.get("name", "")):
        raise _attachment_refusal(
            "target or name contains a newline (netron truncates names at the "
            "first newline before matching)",
            "strip newlines from entity names",
            code="netron_attachment_invalid",
            target=target,
        )
    if kind in ("model", "graph"):
        if target != "":
            raise _attachment_refusal(
                f"{kind}-scope rows require target='' or they vanish silently",
                "set target='' for model/graph rows",
                code="netron_attachment_invalid",
                target=target,
            )
    elif target not in known_targets:
        raise _attachment_refusal(
            f"target {target!r} does not resolve to any node or value in the "
            "selected projection (netron drops unknown targets silently)",
            "emit targets from the exported projection only",
            code="netron_attachment_invalid",
            target=target,
        )
    if not str(item.get("name", "")):
        raise _attachment_refusal(
            "item has no name (netron drops malformed items silently)",
            "name every metadata/metric row",
            code="netron_attachment_invalid",
            target=target,
        )


def _default_rows(
    projection: NetronProjection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the default metadata and metric rows from projection sources."""

    metadata: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    total_duration = 0.0
    total_flops = 0.0
    for name, source in projection.node_sources.items():
        path = module_path_of(source)
        if path:
            metadata.append({"kind": "node", "target": name, "name": "module path", "value": path})
        duration = _quantity(getattr(source, "func_duration", None)) or _quantity(
            getattr(source, "forward_duration", None)
        )
        if duration:
            total_duration += duration
            metrics.append(
                {
                    "kind": "node",
                    "target": name,
                    "name": "observed duration",
                    "value": f"{duration * 1e6:.0f}",
                    "type": "us",
                }
            )
        flops = _quantity(getattr(source, "flops_forward", None))
        if flops:
            total_flops += flops
            metrics.append(
                {
                    "kind": "node",
                    "target": name,
                    "name": "flops (estimated)",
                    "value": f"{flops:.0f}",
                }
            )
        memory = _quantity(getattr(source, "activation_memory", None))
        if memory:
            metrics.append(
                {
                    "kind": "node",
                    "target": name,
                    "name": "output tensor bytes",
                    "value": f"{int(memory)}",
                }
            )
    if total_duration:
        metrics.append(
            {
                "kind": "model",
                "target": "",
                "name": "observed forward duration",
                "value": f"{total_duration * 1e6:.0f}",
                "type": "us",
            }
        )
    if total_flops:
        metrics.append(
            {
                "kind": "model",
                "target": "",
                "name": "total flops (estimated)",
                "value": f"{total_flops:.0f}",
            }
        )
    return metadata, metrics


def write_attachment(
    projection: NetronProjection,
    path: Path,
    *,
    key_fn: Callable[[str], str] | None = None,
    stats_fn: Callable[[NetronProjection], list[dict[str, Any]]] | None = None,
) -> Path:
    """Write the ``<name>.attachment.json`` companion next to the artifact.

    Parameters
    ----------
    projection:
        The projection the main artifact was emitted from (target names must
        resolve against exactly this projection).
    path:
        The main artifact path; the companion lands beside it as
        ``<stem>.attachment.json``.
    key_fn:
        Optional target-name mapper -- the pluggable seam for the deferred
        real-ONNX overlay. Defaults to identity.
    stats_fn:
        Optional extra-rows provider (the deferred saved-tensor statistics
        seam; rows must satisfy the same guards). Defaults to none.

    Returns
    -------
    Path
        The written companion path.
    """

    mapper = key_fn or (lambda name: name)
    known_targets = {name for node in projection.nodes for name in ([node.name] + node.outputs)}
    for function in projection.functions:
        for node in function.nodes:
            known_targets.add(node.name)
            known_targets.update(node.outputs)
    known_targets.update(info.name for info in projection.graph_inputs)
    known_targets.update(info.name for info in projection.graph_outputs)
    known_targets = {mapper(name) for name in known_targets}
    metadata, metrics = _default_rows(projection)
    if stats_fn is not None:
        metrics.extend(stats_fn(projection))
    seen: set[tuple[str, str, str, str]] = set()
    for container_name, rows in (("metadata", metadata), ("metrics", metrics)):
        for item in rows:
            if item.get("kind") not in ("model", "graph") and "target" in item:
                item["target"] = mapper(str(item["target"]))
            _validate_item(item, known_targets)
            key = (container_name, str(item["kind"]), item["target"], str(item["name"]))
            if key in seen:
                raise _attachment_refusal(
                    f"duplicate {container_name} name {item['name']!r} for target "
                    f"{item['target']!r} (netron keeps only the last row silently)",
                    "emit per-target-unique names",
                    code="netron_attachment_invalid",
                    target=item["target"],
                )
            seen.add(key)
    companion = path.with_name(path.stem + ".attachment.json")
    payload = {
        "signature": _ATTACHMENT_SIGNATURE,
        "metadata": metadata,
        "metrics": metrics,
    }
    atomic_write_text(companion, _json.dumps(payload, indent=2))
    return companion
