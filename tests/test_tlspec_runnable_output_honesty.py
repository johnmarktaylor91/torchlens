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
    """Class outputs with replay-safe metadata rebuild exactly without divergence.

    r27-B2: a custom-``__init__`` ``hf_model_output`` (``_R7ModelOutput`` defines its OWN
    ``__init__``) is honest fail-closed to UNVERIFIABLE -- its constructor cannot be proven
    side-effect-free from the type without INVOKING it (the RCE surface). The output still
    RECONSTRUCTS faithfully; only the honesty verdict is downgraded. ``namedtuple`` / ``dataclass``
    outputs (no custom constructor to distrust) stay VERIFIED.
    """

    model = _R7ContainerOutputModel(kind, 7)
    x = torch.tensor([1.0, 2.0])
    result = _roundtrip(model, x, tmp_path / f"{kind}_faithful.tlspec")
    live = model(x)

    if kind == "hf_model_output":
        assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    else:
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


# --------------------------------------------------------------------------- #
# r26-C1: a host-escaped non-tensor scalar OUTPUT the sparse DAG cannot reproduce
# must be UNVERIFIABLE, never a false VERIFIED with a dropped ``None`` output.
# --------------------------------------------------------------------------- #


class _IntScalarOutput(nn.Module):
    """Return an ``int(...)`` host-escaped scalar (no output tensor slot)."""

    def forward(self, x: torch.Tensor) -> Any:
        return int(x.argmax().item())


class _FloatScalarOutput(nn.Module):
    """Return a ``float(...)`` host-escaped scalar derived from a param op."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        return float(self.lin(x).sum().detach())


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_factory", "x"),
    [
        (_IntScalarOutput, torch.tensor([0.1, 0.9, 0.3, 0.2])),
        (_FloatScalarOutput, torch.randn(2, 4)),
    ],
)
def test_r26_host_escaped_scalar_output_refuses_at_save(
    model_factory: Any, x: torch.Tensor, tmp_path: Path
) -> None:
    """A model returning a host-escaped Python scalar has no reproducible output.

    ``trace.output_layers == []`` (no output tensor slot), so the v2 descriptor has
    no carrier for the output contract. r37 corr1_1/INV-3 hardening: the runnable
    save REFUSES uniformly at preflight (missing_output_container_contract) --
    accept-then-``None`` (the pre-r37 posture this test used to pin as a run-time
    UNVERIFIABLE ceiling) is gone; the artifact is never produced.
    """

    from torchlens.errors import RunnablePreflightError

    model = model_factory()
    trace = _capture(model, x)
    assert trace.output_layers == []
    path = tmp_path / "scalar_out.tlspec"
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(path, level="runnable", include_activations=True)
    assert "missing_output_container_contract" in str(excinfo.value.fields.get("diagnostics"))


@pytest.mark.smoke
def test_r26_normal_tensor_output_still_verified(tmp_path: Path) -> None:
    """A normal tensor-output model must still VERIFY (no C1 over-trigger)."""

    class _TensorOut(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.lin(x)

    x = torch.randn(2, 4)
    result = _roundtrip(_TensorOut(), x, tmp_path / "tensor_out.tlspec", include_activations=True)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert isinstance(result.output, torch.Tensor)


# --------------------------------------------------------------------------- #
# r27-H4 (R27-C1/C2): a run whose path ceils at UNVERIFIABLE because the OUTPUT
# was never reproduced (host-escaped scalar) or reconstructs LOSSILY (a computed
# __post_init__ field) must NEVER simultaneously report numeric_attestation =
# ATTESTED -- an internally inconsistent false positive. The two lossy/dropped
# flags are now threaded into the numeric-attestation gate.
# --------------------------------------------------------------------------- #


@dataclass
class _H4LossyPostInitOutput:
    """Dataclass output whose ``__post_init__`` computes dropped non-field state.

    The non-invoking rebuild restores only ``value`` / ``meta`` and cannot recompute
    ``derived``, so ``_container_spec_reconstruction_lossy`` flags it and the run ceils
    at UNVERIFIABLE. Defined at MODULE scope so ``resolve_container_type`` finds it and
    the lossiness comes specifically from the ``__post_init__`` signal (not "type not
    loaded"), exercising the exact H4 gate.
    """

    value: torch.Tensor
    meta: int

    def __post_init__(self) -> None:
        """Compute a dropped non-field attribute."""

        self.derived = int(self.meta) * 2


class _H4LossyDataclassModel(nn.Module):
    """Return a lossy-reconstruction dataclass output."""

    def forward(self, x: torch.Tensor) -> Any:
        """Return the lossy ``__post_init__`` dataclass output."""

        return _H4LossyPostInitOutput(value=x + 1, meta=7)


class _H4FloatScalarModel(nn.Module):
    """Return a host-escaped ``float(...)`` scalar (no output tensor slot)."""

    def __init__(self) -> None:
        """Build a linear op so activations exist to (falsely) attest."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> Any:
        """Return a host-escaped scalar derived from a param op."""

        return float(self.lin(x).sum().detach())


@pytest.mark.smoke
def test_h4_host_escaped_scalar_output_refuses_at_save(tmp_path: Path) -> None:
    """A dropped (host-escaped) output can never reach attestation: since r37 the
    zero-tensor-slot artifact is refused at SAVE (missing_output_container_contract),
    so no run -- attested or otherwise -- exists to mis-report."""

    from torchlens.errors import RunnablePreflightError

    x = torch.randn(2, 4)
    trace = _capture(_H4FloatScalarModel(), x)
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(tmp_path / "h4_scalar.tlspec", level="runnable", include_activations=True)
    assert "missing_output_container_contract" in str(excinfo.value.fields.get("diagnostics"))


@pytest.mark.smoke
def test_h4_lossy_container_output_not_attested(tmp_path: Path) -> None:
    """A lossy __post_init__ dataclass output must not be ATTESTED (path UNVERIFIABLE)."""

    x = torch.tensor([1.0, 2.0])
    result = _roundtrip(
        _H4LossyDataclassModel(), x, tmp_path / "h4_lossy.tlspec", include_activations=True
    )

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


# ---------------------------------------------------------------------------
# r35 I1 (hon1_1 == corr1_1, hon1_2, decision B): uniform save-time refusal of
# every lossy/unaddressable model output -- any cardinality, any depth.
# ---------------------------------------------------------------------------


class _IntSet(set):  # type: ignore[type-arg]
    """Set subclass output (must refuse exactly like a bare set)."""


class _LambdaOutputModel(nn.Module):
    """Return an arbitrary lossy/unaddressable output shape."""

    def __init__(self, builder: Any) -> None:
        super().__init__()
        self._builder = builder

    def forward(self, x: torch.Tensor) -> Any:
        return self._builder(x)


class _OpaqueTensorHolder:
    """Opaque object (DynamicCache-style) holding a tensor."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value


@dataclass
class _SetFieldOutput:
    """Dataclass output holding a set field."""

    value: torch.Tensor
    extras: Any


_R35_LOSSY_OUTPUT_BUILDERS = {
    "bare_one_tensor_set": lambda x: {x + 1},
    "one_tensor_set_with_flag": lambda x: {x + 1, "flag"},
    "bare_two_tensor_set": lambda x: {x + 1, x * 2},
    "bare_frozenset": lambda x: frozenset({x + 1}),
    "empty_set_in_tuple": lambda x: (x + 1, set()),
    "empty_frozenset_in_tuple": lambda x: (x + 1, frozenset()),
    "set_subclass": lambda x: _IntSet({x + 1}),
    "tuple_nested_one_tensor_set": lambda x: (x + 1, {x * 2}),
    "tuple_nested_two_tensor_set": lambda x: (x + 1, {x * 2, x * 3}),
    "dict_nested_set": lambda x: {"a": {x + 1}},
    "deep_list_dict_nested_set": lambda x: [{"inner": (x + 1, {x * 2})}],
    "dataclass_with_set_field": lambda x: _SetFieldOutput(value=x + 1, extras={x * 2}),
    "opaque_tensor_holder_in_tuple": lambda x: (x + 1, _OpaqueTensorHolder(x * 2)),
}


@pytest.mark.smoke
@pytest.mark.filterwarnings("ignore:TorchLens intervention-ready output traversal:UserWarning")
@pytest.mark.parametrize("kind", sorted(_R35_LOSSY_OUTPUT_BUILDERS))
def test_r35_lossy_output_refuses_runnable_save_uniformly(tmp_path: Path, kind: str) -> None:
    """Every lossy/unaddressable output refuses at save with the typed code.

    r34 hon1_1/corr1_1 (bare one-tensor set: false VERIFIED+ATTESTED) and
    hon1_2 (nested sets: false DIVERGED / advertise-then-crash) close as ONE
    class: refuse-unless-proved at save. Analysis capture stays available.
    """

    from torchlens.errors import RunnablePreflightError

    model = _LambdaOutputModel(_R35_LOSSY_OUTPUT_BUILDERS[kind]).eval()
    x = torch.arange(4.0) + 1.0
    trace = _capture(model, x)
    with pytest.raises(RunnablePreflightError) as excinfo:
        tl.save(trace, str(tmp_path / f"{kind}.tlspec"), level="runnable")
    codes = {diag.code.value for diag in excinfo.value.fields["diagnostics"]}
    assert "missing_output_container_contract" in codes
    # No runnable artifact was written.
    assert not (tmp_path / f"{kind}.tlspec").exists()
    # Ordinary analysis save remains available for the same capture.
    tl.save(trace, str(tmp_path / f"{kind}_analysis.tlspec"), level="portable")


_R35_LOSSLESS_OUTPUT_BUILDERS = {
    "bare_tensor": lambda x: x + 1,
    "tuple": lambda x: (x + 1, x * 2),
    "one_tensor_list": lambda x: [x + 1],
    "dict": lambda x: {"a": x + 1, "b": x * 2},
    "namedtuple": lambda x: _Pair(value=x + 1, meta=3),
    "tuple_with_literal": lambda x: (x + 1, "ok", 7),
}


@pytest.mark.smoke
@pytest.mark.parametrize("kind", sorted(_R35_LOSSLESS_OUTPUT_BUILDERS))
def test_r35_lossless_outputs_still_save_and_run_verified(tmp_path: Path, kind: str) -> None:
    """Positive controls: proved-lossless outputs save, run, and match kinds."""

    model = _LambdaOutputModel(_R35_LOSSLESS_OUTPUT_BUILDERS[kind]).eval()
    x = torch.arange(4.0) + 1.0
    expected = model(x)
    trace = _capture(model, x)
    path = tmp_path / f"{kind}.tlspec"
    tl.save(trace, str(path), level="runnable")
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert type(result.output) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(result.output, expected)
