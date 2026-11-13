"""Regression tests for portable artifact determinism and round-trip fidelity."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

import torchlens as tl
from torchlens._utils._record_group_table import op_attr


class _TiedReprKey:
    """Hashable key whose repr intentionally collides with peer keys."""

    def __init__(self, token: str) -> None:
        """Store deterministic content hidden from repr.

        Parameters
        ----------
        token:
            Stable key content.
        """

        self.token = token

    def __repr__(self) -> str:
        """Return an intentionally non-unique representation."""

        return "tied"


class _TiedReprKeyA(_TiedReprKey):
    """First distinct type sharing the tied representation."""


class _TiedReprKeyB(_TiedReprKey):
    """Second distinct type sharing the tied representation."""


class _LinearModel(torch.nn.Module):
    """Tiny parameterized model used by deterministic artifact tests."""

    def __init__(self) -> None:
        """Initialize one deterministic linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the test layer.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Layer output.
        """

        return self.linear(value)


class _ComplexModel(torch.nn.Module):
    """Tiny complex-valued model used by payload transport tests."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a supported complex64 activation.

        Parameters
        ----------
        value:
            Complex input tensor.

        Returns
        -------
        torch.Tensor
            Complex activation.
        """

        return value * (1 + 0j)


class _BufferedModel(torch.nn.Module):
    """Model with a child module and persistent buffer for pickle tests."""

    def __init__(self) -> None:
        """Initialize the child module and buffer."""

        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.register_buffer("offset", torch.ones(2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply the child module and buffer.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Buffered output.
        """

        return self.linear(value) + self.offset


def _save_seeded_trace(path: Path, *, seed: int) -> tl.Trace:
    """Capture, save, and reload one trace with a chosen capture seed.

    Parameters
    ----------
    path:
        Destination bundle path.
    seed:
        Capture seed controlling the live parameter barcode.

    Returns
    -------
    tl.Trace
        Reloaded portable trace.
    """

    torch.manual_seed(0)
    trace = tl.trace(
        _LinearModel(),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(seed=seed),
    )
    trace.save(path)
    return tl.load(path)


def test_save_scrub_remaps_process_local_identity_tokens(tmp_path: Path) -> None:
    """Equivalent traces persist dense identities independent of live ids and barcodes."""

    first_path = tmp_path / "first.tlspec"
    second_path = tmp_path / "second.tlspec"
    first = _save_seeded_trace(first_path, seed=1)
    second = _save_seeded_trace(second_path, seed=2)

    first_param_ops = [op for op in first._layer_list if op.has_params]
    second_param_ops = [op for op in second._layer_list if op.has_params]
    assert [op._param_barcodes for op in first_param_ops] == [
        op._param_barcodes for op in second_param_ops
    ]
    assert [op.graph.equivalence_class for op in first_param_ops] == [
        op.graph.equivalence_class for op in second_param_ops
    ]
    assert first.provenance.model_object_id == second.provenance.model_object_id == 1
    assert first.provenance.input_object_id == second.provenance.input_object_id == 1

    first_manifest = json.loads((first_path / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_path / "manifest.json").read_text(encoding="utf-8"))
    assert [site["op_kind"] for site in first_manifest["sites"]] == [
        site["op_kind"] for site in second_manifest["sites"]
    ]


def test_save_scrub_remaps_autograd_identity_joins(tmp_path: Path) -> None:
    """Autograd ids become dense while all persisted graph joins remain valid."""

    value = torch.ones(1, 2, requires_grad=True)
    trace = tl.trace(_LinearModel(), value, capture=tl.options.CaptureOptions(backward_ready=True))
    trace.backward(trace.output_ops[0].out.sum())
    path = tmp_path / "backward.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    expected_ids = list(range(1, len(loaded.autograd.grad_fn_order) + 1))
    assert loaded.autograd.grad_fn_order == expected_ids
    assert list(loaded._grad_fn_logs) == expected_ids
    assert set(loaded.autograd.backward_root_grad_fn_object_ids) <= set(expected_ids)
    assert all(
        grad_fn.grad_fn_object_id == grad_fn_id
        and set(grad_fn.next_grad_fn_ids) <= set(expected_ids)
        for grad_fn_id, grad_fn in loaded._grad_fn_logs.items()
    )


def test_bundle_writer_resolves_lazy_conjugate_payloads(tmp_path: Path) -> None:
    """Saved tensor bytes represent the logical value, not lazy physical storage."""

    trace = tl.trace(
        _ComplexModel(),
        torch.tensor([1 + 2j], dtype=torch.complex64),
        capture=tl.options.CaptureOptions(activation_transform=lambda value: value.conj()),
        save="all",
    )
    expected = trace.output_ops[0].payload.transformed_out.clone()
    path = tmp_path / "conjugate.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    assert torch.equal(loaded.output_ops[0].payload.transformed_out, expected)


def test_frozenset_round_trips_recursive_blob_payloads(tmp_path: Path) -> None:
    """Frozenset containers preserve type and materialize nested tensor blobs."""

    trace = tl.trace(_LinearModel(), torch.ones(1, 2))
    target = trace.output_ops[0]
    target._call_config = {"frozen": frozenset({torch.tensor(3)})}
    path = tmp_path / "frozenset.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    frozen = loaded.output_ops[0].call.config["frozen"]
    assert isinstance(frozen, frozenset)
    assert len(frozen) == 1
    assert torch.equal(next(iter(frozen)), torch.tensor(3))


def test_tuple_subclasses_preserve_type_across_bundle_round_trip(tmp_path: Path) -> None:
    """Torch Size and structseq metadata do not flatten to builtin tuple."""

    trace = tl.trace(_LinearModel(), torch.ones(1, 2))
    target = trace.output_ops[0]
    maximum = torch.max(torch.tensor([1, 3]), dim=0)
    target._call_config = {"size": torch.Size([2, 3]), "maximum": maximum}
    path = tmp_path / "tuple-subclasses.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    config = loaded.output_ops[0].call.config
    assert isinstance(config["size"], torch.Size)
    assert type(config["maximum"]) is type(maximum)
    assert torch.equal(config["maximum"].values, maximum.values)


@pytest.mark.smoke
def test_backup_cleanup_failure_does_not_fail_completed_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-swap backup cleanup error cannot turn a successful save into failure."""

    path = tmp_path / "overwrite.tlspec"
    first = tl.trace(_LinearModel(), torch.ones(1, 2))
    first.save(path)
    second = tl.trace(_LinearModel(), torch.full((1, 2), 2.0))

    def fail_cleanup(_path: Path) -> None:
        """Simulate a filesystem refusal while deleting the stale backup."""

        raise OSError("simulated backup cleanup refusal")

    monkeypatch.setattr("torchlens._io.bundle._remove_path", fail_cleanup)
    second.save(path, overwrite=True)

    loaded = tl.load(path)
    assert torch.equal(loaded.output_ops[0].out, second.output_ops[0].out)


def test_intervention_overwrite_rename_failure_restores_previous_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed intervention swap restores the old artifact from its backup."""

    trace = tl.trace(
        _LinearModel(),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    path = tmp_path / "intervention.tlspec"
    trace.save_intervention(path, level="audit")
    original_spec = (path / "intervention.json").read_bytes()
    original_rename = os.rename

    def fail_replacement(source: str | Path, target: str | Path) -> None:
        """Fail only the staged replacement's final installation."""

        if Path(source).name.startswith(f"{path.name}.tmp.") and Path(target) == path:
            raise OSError("simulated intervention swap refusal")
        original_rename(source, target)

    monkeypatch.setattr("torchlens.intervention._save.os.rename", fail_replacement)
    with pytest.raises(OSError, match="swap refusal"):
        trace.save_intervention(path, level="audit", overwrite=True)

    assert (path / "intervention.json").read_bytes() == original_spec


def test_content_hash_is_address_and_insertion_order_independent() -> None:
    """Object fallback and tied mapping keys never ingest address or source order."""

    left_a = _TiedReprKey("a")
    left_b = _TiedReprKey("b")
    right_a = _TiedReprKey("a")
    right_b = _TiedReprKey("b")

    assert tl.fingerprint.content(left_a) == tl.fingerprint.content(right_a)
    assert tl.fingerprint.content({left_a: 1, left_b: 2}) == tl.fingerprint.content(
        {right_b: 2, right_a: 1}
    )


def test_loop_signature_sorts_sets_by_emitted_tokens() -> None:
    """Set signature order is total even when distinct elements share repr."""

    from torchlens._postprocess.loop_detection import _append_signature_tokens

    tokens: list[str] = []
    _append_signature_tokens(
        {_TiedReprKeyB("b"), _TiedReprKeyA("a")},
        "arg",
        tokens,
        0,
    )

    assert "_TiedReprKeyA" in tokens[1]
    assert "_TiedReprKeyB" in tokens[2]


def test_merged_tree_hash_frames_paths_by_length() -> None:
    """Newlines and NUL-like boundaries in member paths cannot forge entries."""

    from torchlens._merged._artifact import _tree_hash_entry

    path = "nested/name\nwith-newline"
    framed = _tree_hash_entry(path, 3, "00" * 32)
    path_size = int.from_bytes(framed[:8], "big")

    assert path_size == len(path.encode("utf-8"))
    assert framed[8 : 8 + path_size].decode("utf-8") == path


def test_resave_preserves_capture_provenance_certificate(tmp_path: Path) -> None:
    """Loaded artifact provenance is carried forward instead of re-derived weaker."""

    first_path = tmp_path / "first.tlspec"
    second_path = tmp_path / "second.tlspec"
    trace = tl.trace(_LinearModel(), torch.ones(1, 2))
    trace.save(first_path)
    tl.load(first_path).save(second_path)

    first_manifest = json.loads((first_path / "manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second_path / "manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["provenance"] == first_manifest["provenance"]


def test_provenance_set_values_are_canonically_ordered() -> None:
    """Set-like provenance values normalize independently of hash iteration order."""

    from torchlens._io.bundle import _json_ready_provenance_value

    assert _json_ready_provenance_value({"cuda", "cpu", "mps"}) == ["cpu", "cuda", "mps"]


def test_current_manifest_refuses_unparseable_torchlens_version(tmp_path: Path) -> None:
    """Current-schema artifacts cannot bypass the producer-version consistency belt."""

    from torchlens._io import ArtifactError

    path = tmp_path / "bad-version.tlspec"
    tl.trace(_LinearModel(), torch.ones(1, 2)).save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["torchlens_version"] = "not-a-version"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ArtifactError, match="torchlens_version must be a PEP 440 version string"
    ) as info:
        tl.load(path)
    assert info.value.fields.get("code") == "manifest_schema_invalid"


def test_bounded_json_ceiling_counts_utf8_bytes() -> None:
    """Multibyte JSON text cannot exceed a byte ceiling via character counting."""

    from torchlens._io._json import loads_bounded

    with pytest.raises(json.JSONDecodeError, match="maximum size"):
        loads_bounded('"éé"', max_bytes=5)


def test_tlspec_writer_refuses_nonstandard_nan_tokens(tmp_path: Path) -> None:
    """Manifest writers fail before creating JSON containing NaN or Infinity."""

    from torchlens._io.tlspec import _TlSpecWriter

    path = tmp_path / "manifest.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        _TlSpecWriter.write_json(path, {"bad": float("nan")})
    assert not path.exists()


@pytest.mark.rare
@pytest.mark.backend_jax
def test_jax_codec_refuses_unknown_logical_dtype() -> None:
    """JAX decode fails closed instead of silently retaining the transport dtype."""

    pytest.importorskip("jax")
    from torchlens._io.payload_codec import JaxPayloadCodec
    from torchlens.backends.registry import BackendRuntimeCompatibilityError

    entry = {"logical_dtype": "not_a_real_jax_dtype", "codec_metadata": {}}
    with pytest.raises(BackendRuntimeCompatibilityError, match="logical_dtype"):
        JaxPayloadCodec().from_numpy(np.ones(1, dtype=np.float32), entry, map_location=None)


@pytest.mark.rare
@pytest.mark.backend_jax
def test_jax_codec_restores_weak_type() -> None:
    """JAX weak scalar semantics survive codec encode/decode when JAX is available."""

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from torchlens._io.payload_codec import JaxPayloadCodec

    codec = JaxPayloadCodec()
    value = jnp.asarray(1)
    encoded = codec.to_numpy(value)
    entry = {
        "logical_dtype": encoded.logical_dtype,
        "codec_metadata": encoded.codec_metadata,
    }
    restored = codec.from_numpy(encoded.array, entry, map_location=None)

    assert value.weak_type is True
    assert restored.weak_type is True
    assert jax.device_get(restored) == jax.device_get(value)


def test_codec_metadata_preserves_explicit_none_values() -> None:
    """Codec audit metadata distinguishes explicit null from absent fields."""

    from torchlens._io.payload_codec import _json_ready_mapping

    assert _json_ready_mapping({"outer": None, "nested": {"inner": None}}) == {
        "outer": None,
        "nested": {"inner": None},
    }


@pytest.mark.smoke
def test_codec_metadata_round_trip_preserves_tuples() -> None:
    """Tagged codec metadata distinguishes tuples from JSON lists."""

    from torchlens._io.manifest import TensorEntry
    from torchlens._io.payload_codec import _json_ready_mapping

    metadata = _json_ready_mapping({"tuple": (1, [2, (3,)]), "list": [1, 2]})
    entry = TensorEntry.from_dict(
        {
            "blob_id": "00000000",
            "kind": "out",
            "label": "output_1",
            "relative_path": "tensors/00000000.safetensors",
            "backend": "safetensors",
            "shape": [1],
            "dtype": "torch.float32",
            "device_at_save": "cpu",
            "layout": "torch.strided",
            "bytes": 4,
            "sha256": "0" * 64,
            "codec_metadata": metadata,
        }
    )

    assert entry.codec_metadata == {"tuple": (1, [2, (3,)]), "list": [1, 2]}
    reparsed = TensorEntry.from_dict(entry.to_dict())
    assert reparsed.codec_metadata == entry.codec_metadata


def test_save_succeeds_for_multi_pass_recurrent_trace(tmp_path: Path) -> None:
    """Saving a trace with a multi-pass Layer must not trip the scrub probe.

    Regression: the identity scrub probed ``record.parent_param_ops`` before
    its _PORTABLE_STATE_SPEC gate; a multi-pass Layer's per-pass delegation
    raises InvalidArgumentError (a ValueError getattr does not swallow), so
    EVERY ``Trace.save`` of a recurrent model died at the bundle writer.
    """

    class _Recurrent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.relu(self.linear(x))
            return x

    trace = tl.trace(_Recurrent(), torch.ones(1, 2))
    assert any(layer.num_passes > 1 for layer in trace.layers.values())
    path = tmp_path / "recurrent.tlspec"
    trace.save(path)
    assert tl.load(path)._num_ops == trace._num_ops


def test_canonical_metadata_relation_sets_round_trip(tmp_path: Path) -> None:
    """Relation frozensets survive the canonical (sorted-REDUCE) metadata dump.

    B3R4-R21-2 rewrites exact set/frozenset pickling in ``metadata.pkl`` as
    ``cls(sorted_members)`` — a REDUCE on ``builtins.frozenset``/``set`` —
    so this pins that the default-deny safe unpickler admits the spelling
    and the loaded relation sets are logically unchanged.
    """

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 2))
    trace = tl.trace(model, torch.ones(1, 2))
    path = tmp_path / "canonical.tlspec"
    trace.save(path)

    loaded = tl.load(path)
    for label in ("linear_1_1", "relu_1_2", "linear_2_3"):
        loaded_layer = loaded[label]
        assert isinstance(loaded_layer.input_ancestors, frozenset)
        assert set(loaded_layer.input_ancestors) == set(trace[label].input_ancestors)


def test_foreign_defaultdict_factory_keeps_data_and_discloses_the_factory(tmp_path: Path) -> None:
    """An unsupported factory is never run, and its mapping contents survive load."""

    def factory() -> list:
        raise AssertionError("a serialized factory must never execute")

    trace = tl.trace(_LinearModel(), torch.ones(1, 2))
    setattr(
        trace.output_ops[0], op_attr("func_config"), {"lookup": defaultdict(factory, {"seen": [3]})}
    )
    path = tmp_path / "factory-disclosure.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    assert any(
        row["kind"] == "unencodable_values_stringified"
        for row in loaded.annotations["capture_advisories"]
    )
    mapping = loaded.output_ops[0].call.config["lookup"]
    assert type(mapping) is dict
    assert mapping == {"seen": [3]}
    manifest = json.loads((path / "manifest.json").read_text())
    assert any("default_factory" in at for at in manifest["disclosures"]["first"])
