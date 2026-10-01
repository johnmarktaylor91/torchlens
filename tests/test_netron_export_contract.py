"""Netron export acceptance contract, tier T1 (lane F14, netron memo s6).

Strict ``google.protobuf.json_format`` parse into ``onnx.ModelProto`` plus
``onnx.checker.check_model(full_check=True)`` on every fixture -- the gate
that caught four real structural bugs (SSA violations, unsorted function
bodies, a missing scalar shape, the killer function cycle) netron's
permissive reader swallowed. Structural invariants ride beside it: SSA,
topological order (root AND function bodies), typed-value coverage, exact
graph I/O including duplicate-reference aliases, dtype/shape fidelity,
no false zeros, attribute sort order, deterministic bytes, and the two
NEGATIVE fixtures proving the checker actually rejects the failure classes
the gate exists to catch.

The ``onnx``/``protobuf`` dependencies are declared through
sprint/packaging_requests.tsv (netron memo B0: the historical importorskip
was skipped in every CI leg, making the "contract-tested" claim false).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx", reason="netron acceptance contract needs onnx (test extra)")
json_format = pytest.importorskip("google.protobuf.json_format")

import torchlens as tl  # noqa: E402

pytestmark = pytest.mark.smoke


class _Inner(nn.Module):
    """Two-op child module (linear + relu)."""

    def __init__(self) -> None:
        """Build the child linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then relu."""

        return torch.relu(self.fc(x))


class _Nested(nn.Module):
    """Two children plus a root-level op and an eval BatchNorm tail."""

    def __init__(self) -> None:
        """Build children a, b and the norm tail."""

        super().__init__()
        self.a = _Inner()
        self.b = _Inner()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine both children at the root, then normalize."""

        return self.bn(self.a(x) + self.b(x))


@pytest.fixture(scope="module")
def nested_log() -> Any:
    """One shared trace of the nested fixture model (eval mode)."""

    log = tl.trace(_Nested().eval(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def _parse_and_check(path: Path) -> Any:
    """Strict-parse one artifact and run the official structural validator."""

    model = json_format.Parse(path.read_text(encoding="utf-8"), onnx.ModelProto())
    onnx.checker.check_model(model, full_check=True)
    return model


@pytest.mark.parametrize("granularity", ["op", "module", "rolled"])
def test_every_granularity_parses_strict_and_passes_check_model(
    nested_log: Any, tmp_path: Path, granularity: str
) -> None:
    """All three projections satisfy the strict parse + checker contract."""

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity=granularity)
    model = _parse_and_check(path)
    assert model.ir_version == 10
    assert model.producer_name == "torchlens"
    assert len(model.graph.input) == 1
    assert len(model.graph.output) == 1


def test_ssa_and_topological_order_root_and_bodies(nested_log: Any, tmp_path: Path) -> None:
    """Every value is produced once, before every consumption, everywhere."""

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity="module")
    payload = json.loads(path.read_text(encoding="utf-8"))

    def _assert_sorted(nodes: list[dict[str, Any]], formals: list[str]) -> None:
        produced: set[str] = set(formals)
        outputs: list[str] = []
        for node in nodes:
            outputs.extend(node["output"])
        assert len(outputs) == len(set(outputs)), "SSA: duplicate producer"
        available = produced | {row["name"] for row in payload["graph"].get("input", [])}
        for node in nodes:
            for value in node["input"]:
                assert value in available, f"{node['name']} consumes {value} before production"
            available.update(node["output"])

    _assert_sorted(payload["graph"]["node"], [])
    for function in payload.get("functions", []):
        _assert_sorted(function["node"], list(function["input"]))


def test_typed_value_coverage_and_io_promotion(nested_log: Any, tmp_path: Path) -> None:
    """Every tensor-producing op is typed; inputs/outputs are native graph I/O."""

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = payload["graph"]
    node_names = {node["name"] for node in graph["node"]}
    assert "input_1" not in node_names, "input layers PROMOTE to graph.input (SSA rule)"
    assert graph["input"][0]["name"] == "input_1"
    typed = {row["name"] for row in graph.get("valueInfo", [])}
    io_names = {row["name"] for row in graph["input"]} | {row["name"] for row in graph["output"]}
    for node in graph["node"]:
        value = node["output"][0]
        assert value in typed or value in io_names, f"untyped value {value}"
    for row in graph["input"] + graph["output"]:
        tensor_type = row["type"]["tensorType"]
        assert tensor_type["elemType"] == 1
        assert [d["dimValue"] for d in tensor_type["shape"]["dim"]] == ["2", "4"]


def test_aliased_outputs_emit_duplicate_references(tmp_path: Path) -> None:
    """Aliased graph outputs reference the SAME produced value once per alias."""

    class _TwoOut(nn.Module):
        """Returns the same tensor twice."""

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """One relu, aliased into both outputs."""

            y = torch.relu(x)
            return y, y

    log = tl.trace(_TwoOut(), torch.randn(2, 4))
    path = tl.export.netron(log, tmp_path / "alias.json", granularity="op")
    _parse_and_check(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    outputs = [row["name"] for row in payload["graph"]["output"]]
    assert outputs == ["relu_1_1", "relu_1_1"], "output cardinality and aliasing preserved"


def test_scalar_and_zero_dim_shapes(tmp_path: Path) -> None:
    """Rank-0 values carry an explicit EMPTY shape, never an absent one."""

    class _Scalar(nn.Module):
        """Produces a genuine rank-0 tensor."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Sum to a scalar and keep one op after it."""

            return x.sum() * 2.0

    log = tl.trace(_Scalar(), torch.randn(2, 4))
    path = tl.export.netron(log, tmp_path / "scalar.json", granularity="op")
    _parse_and_check(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out_type = payload["graph"]["output"][0]["type"]["tensorType"]
    assert out_type["elemType"] == 1
    assert out_type.get("shape", {}) == {} or out_type["shape"].get("dim", []) == []


def test_no_false_zeros_in_curated_attributes(nested_log: Any, tmp_path: Path) -> None:
    """Not-applicable facts are ABSENT, never 0 (memo D-17)."""

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for node in payload["graph"]["node"]:
        attrs = {attr["name"]: attr for attr in node.get("attribute", [])}
        if "params" in attrs:
            assert int(attrs["params"]["i"]) > 0
        if "flops_estimated" in attrs:
            assert int(attrs["flops_estimated"]["i"]) > 0
        if "observed_duration_us" in attrs:
            assert int(attrs["observed_duration_us"]["i"]) >= 0
        assert "pass" not in attrs, "single-pass entities never carry a pass attr"


def test_attribute_names_are_alphabetically_sorted(nested_log: Any, tmp_path: Path) -> None:
    """Attribute order IS display order: netron sorts case-insensitively (D-17)."""

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity="op")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for node in payload["graph"]["node"]:
        names = [attr["name"] for attr in node.get("attribute", [])]
        assert names == sorted(names, key=str.lower), f"{node['name']}: {names}"
        assert "module" not in names, "netron already spends 'module' on the domain"


def test_deterministic_bytes(nested_log: Any, tmp_path: Path) -> None:
    """Exporting the same trace twice yields byte-identical artifacts."""

    first = tl.export.netron(nested_log, tmp_path / "a.json", granularity="module")
    second = tl.export.netron(nested_log, tmp_path / "b.json", granularity="module")
    assert first.read_bytes() == second.read_bytes()


def test_docstrings_are_one_line_and_privacy_normalized(nested_log: Any, tmp_path: Path) -> None:
    """Node docStrings carry basename+line only -- never absolute paths (D-18)."""

    import os

    path = tl.export.netron(nested_log, tmp_path / "g.json", granularity="op")
    text = path.read_text(encoding="utf-8")
    home = os.path.expanduser("~")
    assert home not in text
    assert os.environ.get("USER", "\x00nouser") not in text
    payload = json.loads(text)
    for node in payload["graph"]["node"]:
        assert "\n" not in node.get("docString", "")


def test_unknown_granularity_refuses_teaching() -> None:
    """The closed granularity vocabulary refuses with the stable code."""

    log = tl.trace(nn.Sequential(nn.ReLU()), torch.randn(1, 2))
    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log, "unused.json", granularity="modules")
    assert excinfo.value.fields["code"] == "netron_granularity_invalid"
    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log, "unused.json", depth=0)
    assert excinfo.value.fields["code"] == "netron_granularity_invalid"


def test_path_none_without_open_refuses_teaching() -> None:
    """path=None without open=True is a typed nothing-to-do refusal (D-20)."""

    log = tl.trace(nn.Sequential(nn.ReLU()), torch.randn(1, 2))
    with pytest.raises(tl.errors.ConfigurationError) as excinfo:
        tl.export.netron(log)
    assert excinfo.value.fields["code"] == "netron_path_or_open_required"


def test_negative_function_cycle_fails_check_model(tmp_path: Path) -> None:
    """The checker NAMES the function-cycle class netron only dies on (D-07)."""

    payload = {
        "irVersion": 10,
        "producerName": "negative",
        "opsetImport": [{"domain": "ai.torchlens.module", "version": 1}],
        "graph": {
            "name": "g",
            "node": [
                {
                    "name": "call_a",
                    "opType": "f_a",
                    "domain": "ai.torchlens.module",
                    "input": ["x"],
                    "output": ["y"],
                }
            ],
            "input": [
                {
                    "name": "x",
                    "type": {"tensorType": {"elemType": 1, "shape": {"dim": [{"dimValue": "2"}]}}},
                }
            ],
            "output": [
                {
                    "name": "y",
                    "type": {"tensorType": {"elemType": 1, "shape": {"dim": [{"dimValue": "2"}]}}},
                }
            ],
        },
        "functions": [
            {
                "name": "f_a",
                "domain": "ai.torchlens.module",
                "input": ["x"],
                "output": ["y"],
                "node": [
                    {
                        "name": "inner_b",
                        "opType": "f_b",
                        "domain": "ai.torchlens.module",
                        "input": ["x"],
                        "output": ["y"],
                    }
                ],
                "opsetImport": [{"domain": "ai.torchlens.module", "version": 1}],
            },
            {
                "name": "f_b",
                "domain": "ai.torchlens.module",
                "input": ["x"],
                "output": ["y"],
                "node": [
                    {
                        "name": "inner_a",
                        "opType": "f_a",
                        "domain": "ai.torchlens.module",
                        "input": ["x"],
                        "output": ["y"],
                    }
                ],
                "opsetImport": [{"domain": "ai.torchlens.module", "version": 1}],
            },
        ],
    }
    model = json_format.Parse(json.dumps(payload), onnx.ModelProto())
    with pytest.raises(onnx.checker.ValidationError, match="[Cc]ycle"):
        onnx.checker.check_model(model, full_check=True)


def test_negative_missing_shape_scalar_graph_output_fails_check_model() -> None:
    """A graph output with NO shape (unknown rank) fails the boundary check."""

    payload = {
        "irVersion": 10,
        "producerName": "negative",
        "opsetImport": [{"domain": "ai.torchlens.lossy", "version": 1}],
        "graph": {
            "name": "g",
            "node": [
                {
                    "name": "s",
                    "opType": "sum",
                    "domain": "ai.torchlens.lossy",
                    "input": ["x"],
                    "output": ["y"],
                }
            ],
            "input": [
                {
                    "name": "x",
                    "type": {"tensorType": {"elemType": 1, "shape": {"dim": [{"dimValue": "2"}]}}},
                }
            ],
            "output": [{"name": "y", "type": {"tensorType": {"elemType": 1}}}],
        },
    }
    model = json_format.Parse(json.dumps(payload), onnx.ModelProto())
    with pytest.raises(onnx.checker.ValidationError, match="shape.*required"):
        onnx.checker.check_model(model, full_check=True)
