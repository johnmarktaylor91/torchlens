"""The TB graph with real StepStats: executed DAG, truthful fields (memo 3.14).

Two layers, deliberately split:

- :func:`build_graph_ir` -- a DEPENDENCY-FREE intermediate representation
  built from a TorchLens ``Trace``: module-namespaced node names (one
  deterministic encoder shared with the site-token grammar), dataflow
  inputs, output shapes/dtypes, and evidence-qualified host-bracket
  timings. Testable with zero optional packages, including the ingest
  oracle's core join check (every stats row must name a graph node).
- :func:`graph_ir_to_tensorboard` -- serializes the IR into a
  ``GraphDef`` + ``RunMetadata(StepStats)`` through tensorboard's own
  protos and writes via torch's ``FileWriter.add_graph`` path (the exact
  path the 151/151 resnet18 binding was measured through in TB 2.21).

Truthful fields ONLY: CPU host-bracket duration ships with evidence
``instrumented_host_bracket`` (relative comparison only); CUDA compute time
is OMITTED until the kernel-telemetry join supplies device durations
(``func_duration`` has no CUDA sync -- painting dispatch as kernel cost is a
beautiful lie); allocator-memory coloring is OFF (TB's frontend never reads
``allocator_name``; filling an allocator-labeled widget with output bytes
violates the no-fake-colors ruling). Output shapes/bytes go where the
frontend actually reads them. Missing evidence omits a field; it never
writes zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._errors import SinkProtocolError

__tl_layer__ = "L8"

#: The one timing-evidence token launch ships (memo 3.14 / section 9).
TIMING_EVIDENCE = "instrumented_host_bracket"


@dataclass(frozen=True)
class GraphNodeIR:
    """One executed-op node: name, op kind, inputs, shapes, evidence."""

    name: str
    op: str
    inputs: tuple[str, ...]
    label: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    duration_us: int | None = None
    output_bytes: int | None = None


@dataclass(frozen=True)
class GraphIR:
    """The dep-free executed-DAG graph payload (sink capability ``graph``)."""

    nodes: tuple[GraphNodeIR, ...]
    device: str = "/device:CPU:0"
    timing_evidence: str | None = None
    coverage: str = "executed_paths_only"
    disclosures: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """Render the IR as plain JSON-serializable data (JSONL rows)."""

        return {
            "kind": "torchlens.trackers.graph_ir",
            "device": self.device,
            "timing_evidence": self.timing_evidence,
            "coverage": self.coverage,
            "disclosures": list(self.disclosures),
            "nodes": [
                {
                    "name": node.name,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "label": node.label,
                    "shape": list(node.shape) if node.shape is not None else None,
                    "dtype": node.dtype,
                    "duration_us": node.duration_us,
                    "output_bytes": node.output_bytes,
                }
                for node in self.nodes
            ],
        }

    def stats_names(self) -> tuple[str, ...]:
        """Names of every node that will emit a StepStats row (timed nodes)."""

        return tuple(node.name for node in self.nodes if node.duration_us is not None)

    def join_oracle(self, stats_names: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """The independent ingest check: stats rows without a graph node.

        The TB frontend joins ``NodeExecStats.node_name`` on graph node
        names; an unmatched row silently drops its coloring (never an
        error). Returns the unmatched stats names -- empty means every
        timed row binds. Pass externally harvested ``stats_names`` to check
        a serialized event file against this IR (the corruption test).
        """

        graph_names = {node.name for node in self.nodes}
        rows = self.stats_names() if stats_names is None else stats_names
        return tuple(sorted(name for name in rows if name not in graph_names))


def node_name_for(layer: Any) -> str:
    """The ONE deterministic node-name encoder (memo 3.14).

    Namespaces from ``module_call_stack`` (``module_address`` is verified
    empty on real models), with ``:`` replaced -- the same slash-grouped
    spelling the TB Graphs dashboard collapses on and the site tokens use.
    """

    stack = getattr(layer, "module_call_stack", None) or ()
    parts: list[str] = []
    for entry in stack:
        text = str(entry).replace(":", "_")
        if text and text != "None":
            parts.append(text)
    parts.append(str(getattr(layer, "layer_label", "op")).replace(":", "_"))
    return "/".join(parts)


def build_graph_ir(trace: Any, *, include_timing: bool = True) -> GraphIR:
    """Build the executed-DAG graph IR from one TorchLens trace.

    Never ``jit.trace``: the DAG is what actually ran, so dict inputs,
    data-dependent branches, and functional calls are all present (the
    stock ``add_graph`` failure classes). Dynamic-path coverage is marked
    ``executed_paths_only`` -- one capture shows one taken path, disclosed,
    never guessed.
    """

    layers = list(getattr(trace, "layer_list", ()) or ())
    names: dict[str, str] = {}
    for layer in layers:
        names[str(layer.layer_label)] = node_name_for(layer)
    nodes: list[GraphNodeIR] = []
    disclosures: list[str] = []
    timed_any = False
    for layer in layers:
        label = str(layer.layer_label)
        shape_raw = getattr(layer, "shape", None) or ()
        shape = (
            tuple(int(dim) for dim in shape_raw)
            if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape_raw)
            and shape_raw
            else None
        )
        duration_us: int | None = None
        if include_timing:
            raw_duration = getattr(layer, "func_duration", None)
            if raw_duration:
                duration_us = max(1, int(float(raw_duration) * 1_000_000))
                timed_any = True
        output_bytes_raw = getattr(layer, "activation_memory", None)
        nodes.append(
            GraphNodeIR(
                name=names[label],
                op=str(getattr(layer, "func_name", "") or getattr(layer, "layer_type", "op")),
                inputs=tuple(
                    names[str(parent)]
                    for parent in (getattr(layer, "parents", ()) or ())
                    if str(parent) in names
                ),
                label=label,
                shape=shape,
                dtype=str(getattr(layer, "dtype", "") or "") or None,
                duration_us=duration_us,
                output_bytes=int(output_bytes_raw) if output_bytes_raw else None,
            )
        )
    disclosures.append("memory_coloring: unavailable (no allocator source)")
    device = "/device:CPU:0"
    if timed_any:
        disclosures.append(
            "timing basis: instrumented host bracket around the wrapped call; "
            "relative comparison only, includes TorchLens overhead"
        )
    return GraphIR(
        nodes=tuple(nodes),
        device=device,
        timing_evidence=TIMING_EVIDENCE if timed_any else None,
        disclosures=tuple(disclosures),
    )


def graph_ir_to_tensorboard(ir: GraphIR, logdir: str) -> str:
    """Serialize one GraphIR into a TB event file; returns the logdir.

    Writes GraphDef + RunMetadata(StepStats) through torch's own
    ``FileWriter.add_graph`` entry point -- the exact path the 151/151
    real-resnet18 StepStats binding was measured through. Requires the
    tensorboard package (typed refusal naming the extra otherwise).
    """

    try:
        from tensorboard.compat.proto.attr_value_pb2 import AttrValue
        from tensorboard.compat.proto.config_pb2 import RunMetadata
        from tensorboard.compat.proto.graph_pb2 import GraphDef
        from tensorboard.compat.proto.node_def_pb2 import NodeDef
        from tensorboard.compat.proto.step_stats_pb2 import (
            DeviceStepStats,
            NodeExecStats,
            StepStats,
        )
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.compat.proto.versions_pb2 import VersionDef
        from torch.utils.tensorboard.writer import FileWriter
    except ImportError as exc:
        raise SinkProtocolError(
            f"Serializing the TB graph needs the tensorboard package: {exc}.",
            code="tracker_sink_unavailable",
            sink="graph_ir_to_tensorboard",
            remedy='pip install "torchlens[tensorboard]" (tensorboard>=2.21).',
        ) from exc

    proto_nodes = []
    node_stats = []
    cursor = 1_000_000
    for node in ir.nodes:
        attrs: dict[str, Any] = {
            "torchlens_label": AttrValue(s=node.label.encode()),
        }
        if node.shape is not None:
            shape_proto = TensorShapeProto(
                dim=[TensorShapeProto.Dim(size=dim) for dim in node.shape]
            )
            attrs["_output_shapes"] = AttrValue(list=AttrValue.ListValue(shape=[shape_proto]))
        if ir.timing_evidence is not None:
            attrs["torchlens_timing_evidence"] = AttrValue(s=ir.timing_evidence.encode())
        proto_nodes.append(
            NodeDef(
                name=node.name,
                op=node.op,
                input=list(node.inputs),
                device=ir.device,
                attr=attrs,
            )
        )
        if node.duration_us is not None:
            node_stats.append(
                NodeExecStats(
                    node_name=node.name,
                    all_start_micros=cursor,
                    op_start_rel_micros=0,
                    op_end_rel_micros=node.duration_us,
                    all_end_rel_micros=node.duration_us,
                )
            )
            cursor += node.duration_us
    graph_def = GraphDef(node=proto_nodes, versions=VersionDef(producer=22))
    run_metadata = None
    if node_stats:
        run_metadata = RunMetadata(
            step_stats=StepStats(
                dev_stats=[DeviceStepStats(device=ir.device, node_stats=node_stats)]
            )
        )
    writer = FileWriter(logdir)
    try:
        writer.add_graph((graph_def, run_metadata))
    finally:
        writer.flush()
        writer.close()
    return logdir


__all__ = [
    "TIMING_EVIDENCE",
    "GraphIR",
    "GraphNodeIR",
    "build_graph_ir",
    "graph_ir_to_tensorboard",
    "node_name_for",
]
