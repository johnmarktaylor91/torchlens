"""Output-tree reconstruction honesty for the unified ``Trace.run`` surface.

Round-6 hardening. A run reporting ``VERIFIED`` MUST return the exact output
object -- correct container TYPE, all non-tensor literal leaves byte-exact, all
fields. A lossy or wrong output must be ``UNVERIFIABLE`` (or diverged), never a
silent substitution blessed as ``VERIFIED``/``ATTESTED``.

Findings covered here:

* F1 -- ``live`` ``Trace.run()`` ignored the captured ``ContainerSpec`` and
  rebuilt a dict keyed by TorchLens path dataclasses (dropping literal leaves and
  the real container type) while still reporting ``VERIFIED``.
* F2 -- a ``bytes`` output literal was scrubbed to ``"<scrubbed:bytes>"`` on
  save yet the loaded run reported ``VERIFIED``/``ATTESTED``.
* F3 -- a root custom ``Mapping`` / ``OrderedDict`` / ``defaultdict`` output
  collapsed to a bare tensor / plain ``dict`` and reported ``VERIFIED``.
* F5 -- a ``memoryview`` output leaf passed save but crashed the loaded run.
* F6 -- a tuple / float output-dict key raised a raw ``ValueError`` during save.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


def _capture(model: nn.Module, x: Any) -> Any:
    """Capture an intervention-ready trace with container structure retained."""

    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


class _Pair(NamedTuple):
    """Named output pair with a tensor and a literal field."""

    value: torch.Tensor
    meta: int


class _R7NamedOutput(NamedTuple):
    """Namedtuple output used by the round-7 completeness regressions."""

    value: torch.Tensor
    meta: Any


@dataclass
class _R7DataclassOutput:
    """Dataclass output used by the round-7 completeness regressions."""

    value: torch.Tensor
    meta: Any


class _R7ModelOutput(dict[str, Any]):
    """HuggingFace-style output used by the round-7 completeness regressions."""

    def __init__(self, **kwargs: Any) -> None:
        """Store output fields with HuggingFace-style attribute access."""

        super().__init__(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _R7ContainerOutputModel(nn.Module):
    """Return one namedtuple, dataclass, or HuggingFace-style output contract."""

    def __init__(self, kind: str, metadata: Any) -> None:
        """Store the requested output kind and metadata leaf."""

        super().__init__()
        self.kind = kind
        self.metadata = metadata

    def forward(self, x: torch.Tensor) -> Any:
        """Return a tensor plus the configured metadata leaf."""

        value = x + 1
        if self.kind == "namedtuple":
            return _R7NamedOutput(value, self.metadata)
        if self.kind == "dataclass":
            return _R7DataclassOutput(value, self.metadata)
        if self.kind == "hf_model_output":
            return _R7ModelOutput(value=value, meta=self.metadata)
        raise AssertionError(self.kind)


class _F1Model(nn.Module):
    """Return the same tensor under several public output container contracts."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x: torch.Tensor) -> Any:
        """Return a structured value selected by ``kind``."""

        y = x + 1
        if self.kind == "list":
            return [y, y * 2]
        if self.kind == "namedtuple":
            return _Pair(y, 7)
        if self.kind == "structseq":
            return torch.max(x, dim=0)
        if self.kind == "bytes":
            return (y, b"exact")
        if self.kind == "dict_literal":
            return {"value": y, "meta": 7}
        raise AssertionError(self.kind)


@pytest.mark.parametrize("kind", ["list", "namedtuple", "structseq", "bytes", "dict_literal"])
def test_f1_live_run_returns_exact_output_object(kind: str) -> None:
    """F1: a VERIFIED live run returns the same object type + literal leaves."""

    model = _F1Model(kind)
    x = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    trace = _capture(model, x)
    result = trace.run(inputs=x)
    live = model(x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is type(live)
    if kind == "namedtuple":
        assert result.output.meta == 7
    if kind == "bytes":
        assert result.output[1] == b"exact"
        assert isinstance(result.output[1], bytes)
    if kind == "dict_literal":
        assert result.output["meta"] == 7
    if kind == "list":
        assert torch.equal(result.output[1], live[1])
    if kind == "structseq":
        assert torch.equal(result.output.indices, live.indices)


class _CustomMapping(Mapping[str, Any]):
    """Minimal custom Mapping output container."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class _F3Model(nn.Module):
    """Return a mapping-subtype output selected by ``kind``."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x: torch.Tensor) -> Any:
        """Return a mapping-subtype output."""

        y = x + 1
        if self.kind == "ordered":
            return OrderedDict([("value", y), ("meta", 3)])
        if self.kind == "defaultdict":
            return defaultdict(list, {"value": y, "meta": 3})
        if self.kind == "custom":
            return _CustomMapping({"value": y, "meta": 3})
        raise AssertionError(self.kind)


def _roundtrip(model: nn.Module, x: Any, path: Path, *, include_activations: bool = False) -> Any:
    """Capture, runnable-save, load, and run on the identical input."""

    trace = _capture(model, x)
    trace.save(path, level="runnable", include_activations=include_activations)
    return tl.load(path).run(inputs=x)


def test_f2_bytes_output_literal_round_trips_byte_exact(tmp_path: Path) -> None:
    """F2: a bytes output leaf survives save/load byte-exact under a VERIFIED run."""

    class Model(nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            return (x + 1, b"exact-user-bytes")

    result = _roundtrip(
        Model(), torch.tensor([1.0, 2.0]), tmp_path / "bytes.tlspec", include_activations=True
    )
    assert isinstance(result.output, tuple)
    assert isinstance(result.output[1], bytes)
    assert result.output[1] == b"exact-user-bytes"
    # A byte-exact literal may be VERIFIED; a scrubbed/changed one never is.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.parametrize("kind", ["ordered", "defaultdict"])
def test_f3_reconstructable_mapping_output_preserved_exact(kind: str, tmp_path: Path) -> None:
    """F3: OrderedDict/defaultdict outputs reconstruct to the EXACT type + fields."""

    model = _F3Model(kind)
    x = torch.tensor([1.0, 2.0])
    result = _roundtrip(model, x, tmp_path / f"{kind}.tlspec")
    live = model(x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is type(live)
    assert set(result.output.keys()) == set(live.keys())
    assert result.output["meta"] == 3
    if kind == "defaultdict":
        assert result.output.default_factory is live.default_factory


def test_f3_custom_mapping_output_never_bare_tensor_verified(tmp_path: Path) -> None:
    """F3: a custom Mapping output honestly rejects at save OR is UNVERIFIABLE.

    It must NEVER collapse to a bare tensor while reporting VERIFIED.
    """

    model = _F3Model("custom")
    x = torch.tensor([1.0, 2.0])
    trace = _capture(model, x)
    path = tmp_path / "custom.tlspec"
    try:
        trace.save(path, level="runnable")
    except tl.errors.TorchLensError:
        # Honest producer-preflight rejection at save is the chosen contract.
        return
    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation in {
        NumericAttestationStatus.NOT_APPLICABLE,
        NumericAttestationStatus.NOT_PRESENT,
    }


def test_f5_memoryview_output_round_trips_or_rejects(tmp_path: Path) -> None:
    """F5: a memoryview output leaf must round-trip or honestly reject, never crash."""

    class Model(nn.Module):
        def forward(self, x: torch.Tensor) -> Any:
            return (x + 1, memoryview(b"abc"))

    model = Model()
    x = torch.tensor([1.0, 2.0])
    trace = _capture(model, x)
    path = tmp_path / "memoryview.tlspec"
    try:
        trace.save(path, level="runnable")
    except (ValueError, tl.errors.TorchLensError):
        # Honest reject at save is acceptable.
        return
    # If save advertised runnable, the loaded run must NOT crash.
    result = tl.load(path).run(inputs=x)
    if result.report.path_faithfulness is PathFaithfulness.VERIFIED:
        assert bytes(result.output[1]) == b"abc"
    else:
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


@pytest.mark.parametrize("kind", ["namedtuple", "dataclass", "hf_model_output"])
def test_r7_nonreconstructable_class_output_is_unverifiable_or_rejected(
    kind: str, tmp_path: Path
) -> None:
    """Class outputs with opaque metadata never advertise then crash."""

    model = _R7ContainerOutputModel(kind, memoryview(b"opaque"))
    x = torch.tensor([1.0, 2.0])
    trace = _capture(model, x)

    live_result = trace.run(inputs=x)
    assert live_result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE

    with pytest.raises(tl.errors.TorchLensError):
        trace.save(tmp_path / f"{kind}_opaque.tlspec", level="runnable")


@pytest.mark.parametrize("kind", ["namedtuple", "dataclass", "hf_model_output"])
def test_r7_reconstructable_class_output_round_trips_verified(kind: str, tmp_path: Path) -> None:
    """Class outputs with replay-safe metadata rebuild exactly without divergence."""

    model = _R7ContainerOutputModel(kind, 7)
    x = torch.tensor([1.0, 2.0])
    result = _roundtrip(model, x, tmp_path / f"{kind}_faithful.tlspec")
    live = model(x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is type(live)
    assert result.output.meta == 7
    assert torch.equal(result.output.value, live.value)


def test_f7_custom_mapping_input_preflight_accepted(tmp_path: Path) -> None:
    """F7: a custom Mapping model INPUT with tensor leaves passes producer preflight."""

    from torchlens._io.runnable import build_sparse_run_descriptor

    class Model(nn.Module):
        def forward(self, cfg: Mapping[str, Any]) -> torch.Tensor:
            return cfg["value"] + 1 if cfg["mode"] == 0 else cfg["value"] * 10

    cfg = _CustomMapping({"value": torch.tensor([1.0, 2.0]), "mode": 0})
    trace = _capture(Model(), cfg)
    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.preflight.passed
    # And it must actually save runnable without a false rejection.
    trace.save(tmp_path / "custom_input.tlspec", level="runnable")


@pytest.mark.parametrize("key", [("value", 1), 1.5])
def test_f6_exotic_output_dict_key_typed_reject_not_raw_valueerror(
    key: Any, tmp_path: Path
) -> None:
    """F6: a tuple/float output-dict key gives a typed reject or faithful save."""

    class Model(nn.Module):
        def __init__(self, k: Any) -> None:
            super().__init__()
            self.k = k

        def forward(self, x: torch.Tensor) -> Any:
            return {self.k: x + 1}

    model = Model(key)
    x = torch.tensor([1.0, 2.0])
    trace = _capture(model, x)
    path = tmp_path / "exotic_key.tlspec"
    try:
        trace.save(path, level="runnable")
    except tl.errors.TorchLensError:
        # A typed TorchLens error (honest reject) is acceptable.
        return
    # If it saved, the identical loaded run must return the mapping faithfully.
    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is not PathFaithfulness.DIVERGED
    assert key in result.output
