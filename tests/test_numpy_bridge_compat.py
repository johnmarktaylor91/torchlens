"""HAS_NUMPY_BRIDGE: torch builds compiled against NumPy 1.x running under NumPy 2.

torch 2.1 and 2.2 wheels import under NumPy 2 but every bridge call
(``torch.from_numpy``, ``Tensor.numpy()``, ``np.asarray(tensor)``) raises
``RuntimeError: Numpy is not available``. TorchLens names that state through a
capability flag in the doctor snapshot and keeps content digests, exports and
NumPy inputs working through byte-copy fallbacks instead of failing deep inside.

On a healthy install the broken bridge is simulated by making the bridge calls
raise exactly what torch raises on the broken pair. On a real broken pair
(``HAS_NUMPY_BRIDGE`` False) the live-environment tests run unpatched.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._transport as transport
from torchlens._utils import _numpy_bridge as bridge, _torch_compat as tc
from torchlens._utils.tensor_leaves import tensor_payload_bytes

_BROKEN = RuntimeError("Numpy is not available")


def _sample_tensors() -> list[torch.Tensor]:
    """Return tensors covering dtypes, 0-dim, empty and noncontiguous layouts."""

    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return [
        base,
        base.t(),
        base[:, ::2],
        torch.tensor(3.5),
        torch.zeros(0, 3),
        torch.arange(10, dtype=torch.int64),
        torch.tensor([True, False, True]),
        torch.arange(6, dtype=torch.float16),
        torch.arange(6, dtype=torch.float32).to(torch.bfloat16),
        torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex64),
    ]


def _reference_bytes(tensor: torch.Tensor) -> bytes:
    """Return the tensor's logical bytes, computed without NumPy (``uint8`` then ``tolist``)."""

    flat = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bytes(flat.tolist())


def _break_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make torch's NumPy bridge raise like a NumPy-1.x torch build under NumPy 2."""

    def _raise(*_args: object, **_kwargs: object) -> object:
        """Raise torch's broken-bridge error."""

        raise _BROKEN

    monkeypatch.setattr(torch.Tensor, "numpy", _raise)
    monkeypatch.setattr(torch, "from_numpy", _raise)
    monkeypatch.setattr(tc, "_warned_missing_capabilities", set())
    monkeypatch.setattr(tc, "HAS_NUMPY_BRIDGE", True)
    monkeypatch.setattr(tc, "_NUMPY_BRIDGE_PROBED", False)
    monkeypatch.setenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", "1")


@pytest.fixture
def broken_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the bridge for one test (fixture form of :func:`_break_bridge`)."""

    _break_bridge(monkeypatch)


def test_payload_bytes_match_the_logical_payload() -> None:
    """``tensor_payload_bytes`` and ``digest_byte_view`` return the logical payload bytes.

    On a healthy bridge this equals the historical ``.numpy().tobytes()`` idiom, so
    every digest that moved onto the helper keeps its value.
    """

    for tensor in _sample_tensors():
        expected = _reference_bytes(tensor)
        if tc.get_numpy_bridge_support() and tensor.dtype != torch.bfloat16:
            assert tensor.detach().cpu().contiguous().numpy().tobytes() == expected
        assert tensor_payload_bytes(tensor) == expected
        assert transport.digest_byte_view(tensor).tobytes() == expected


def test_digests_survive_a_broken_bridge_byte_identically(broken_bridge: None) -> None:
    """With the bridge broken, digests still produce the exact same bytes.

    Before the fallback, ``digest_byte_view`` and ``tensor_payload_bytes`` called
    ``.numpy()`` unconditionally, so every fingerprint, op-dedup and stimulus digest
    died with torch's bare ``RuntimeError: Numpy is not available``.
    """

    assert tc.get_numpy_bridge_support() is False
    for tensor in _sample_tensors():
        want = _reference_bytes(tensor)
        assert tensor_payload_bytes(tensor) == want
        assert transport.digest_byte_view(tensor).tobytes() == want


def test_probe_flags_a_broken_bridge_in_the_snapshot(
    broken_bridge: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe flips HAS_NUMPY_BRIDGE, warns once, and the snapshot reports it."""

    monkeypatch.delenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert tc.get_numpy_bridge_support(force_probe=True) is False
        assert tc.get_numpy_bridge_support(force_probe=True) is False
    messages = [str(w.message) for w in caught if issubclass(w.category, tc.TorchCapabilityWarning)]
    assert len(messages) == 1
    assert "HAS_NUMPY_BRIDGE" in messages[0]
    assert "numpy<2" in messages[0]
    assert tc.HAS_NUMPY_BRIDGE is False
    assert "HAS_NUMPY_BRIDGE" not in tc.OPTIONAL_CAPABILITY_FLAGS
    assert tc.get_torch_capability_snapshot()["HAS_NUMPY_BRIDGE"] is False


def test_bridge_probe_latches_after_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real-op bridge probe runs once per latch (import-hygiene budget)."""

    from torchlens._utils import _kernel_probes

    calls = 0
    real_probe = _kernel_probes.probe_numpy_bridge

    def _counting_probe() -> bool:
        """Count probe runs and defer to the real probe."""

        nonlocal calls
        calls += 1
        return real_probe()

    monkeypatch.setattr(_kernel_probes, "probe_numpy_bridge", _counting_probe)
    monkeypatch.setattr(tc, "_warned_missing_capabilities", set())
    monkeypatch.setattr(tc, "_NUMPY_BRIDGE_PROBED", False)
    monkeypatch.setenv("TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS", "1")
    first = tc.get_numpy_bridge_support()
    assert tc.get_numpy_bridge_support() is first
    assert calls == 1


def test_tensor_numpy_bytes_probes_only_after_a_failed_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RNG digest's state read never probes on a working bridge; it falls back on a broken one.

    A probe inside a capture adds its torch calls to the user's trace and to the
    per-op Python-call census, so the working-bridge read must be the plain
    ``.cpu().numpy().tobytes()`` with no probe at all.
    """

    from torchlens._utils import _kernel_probes

    state = torch.Generator().get_state()
    if _kernel_probes.probe_numpy_bridge():

        def _no_probe() -> bool:
            """Fail when the probe runs."""

            raise AssertionError("bridge probe ran although .numpy() works")

        monkeypatch.setattr(_kernel_probes, "probe_numpy_bridge", _no_probe)
        monkeypatch.setattr(tc, "_NUMPY_BRIDGE_PROBED", False)
        assert bridge.tensor_numpy_bytes(state) == state.numpy().tobytes()
        monkeypatch.undo()
    _break_bridge(monkeypatch)
    assert bridge.tensor_numpy_bytes(state) == _reference_bytes(state)
    assert tc.HAS_NUMPY_BRIDGE is False


def test_live_environment_flag_matches_the_real_bridge() -> None:
    """On any install, the flag agrees with what the bridge really does."""

    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
        torch.zeros(1).numpy()
        works = True
    except RuntimeError:
        works = False
    assert tc.get_numpy_bridge_support(force_probe=True) is works
    tensor = torch.arange(5, dtype=torch.float32)
    assert tensor_payload_bytes(tensor) == bytes(transport.digest_byte_view(tensor))
    assert len(tensor_payload_bytes(tensor)) == 20


def test_tensor_to_numpy_rebuilds_arrays_on_a_broken_bridge(broken_bridge: None) -> None:
    """``tensor_to_numpy`` returns the same array NumPy-side; bf16 still refuses."""

    for tensor in _sample_tensors():
        if tensor.dtype == torch.bfloat16:
            with pytest.raises(TypeError, match="bfloat16"):
                bridge.tensor_to_numpy(tensor)
            continue
        array = bridge.tensor_to_numpy(tensor)
        assert isinstance(array, np.ndarray)
        assert array.shape == tuple(tensor.shape)
        assert str(array.dtype) == str(tensor.dtype).removeprefix("torch.")
        assert np.ascontiguousarray(array).tobytes() == _reference_bytes(tensor)
        array.reshape(-1)[:0] = 0  # writable, like Tensor.numpy()


def test_extraction_npy_export_survives_a_broken_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``.npy`` exporter writes the same values when the bridge is broken.

    Before ``tensor_to_numpy`` every exporter called ``tensor.cpu().numpy()`` and
    died with torch's bare ``RuntimeError: Numpy is not available``.
    """

    from torchlens.dataset_extraction import export_extraction, open_extraction

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    out = tmp_path / "artifact"
    tl.extract_dataset(
        model,
        torch.arange(18, dtype=torch.float32).reshape(6, 3),
        {"relu": "relu"},
        batch_size=4,
        path=out,
        progress=False,
    )
    expected = open_extraction(out).materialize(["relu"])["relu"]
    expected_bytes = _reference_bytes(expected)

    _break_bridge(monkeypatch)
    dest = tmp_path / "export"
    export_extraction(out, dest, file_format="npy")
    contract = json.loads((dest / "export_contract.json").read_text())
    stem = contract["keys"]["relu"]["sanitized_stem"]
    values = np.load(dest / f"{stem}.npy")
    assert values.shape == tuple(expected.shape)
    assert np.ascontiguousarray(values).tobytes() == expected_bytes


def test_tensor_from_numpy_copies_through_the_buffer_protocol(broken_bridge: None) -> None:
    """``tensor_from_numpy`` rebuilds tensors without the bridge, values and dtype intact."""

    for array in (
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(12, dtype=np.int64).reshape(4, 3).T,
        np.array([True, False]),
        np.zeros((0, 2), dtype=np.float64),
        np.array(2.5, dtype=np.float16),
    ):
        tensor = bridge.tensor_from_numpy(array)
        assert tuple(tensor.shape) == array.shape
        assert str(tensor.dtype) == f"torch.{array.dtype.name}"
        assert tensor.tolist() == array.tolist()
    with pytest.raises(TypeError, match="no matching dtype"):
        bridge.tensor_from_numpy(np.array(["a"], dtype=object))


def test_numpy_input_traces_on_a_broken_bridge(broken_bridge: None) -> None:
    """A NumPy input array still traces when torch cannot read NumPy arrays.

    Before the fix the input coercion called ``torch.from_numpy`` and the capture
    died with torch's bare ``RuntimeError: Numpy is not available``.
    """

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    array = np.arange(8, dtype=np.float32).reshape(2, 4)
    trace = tl.trace(model, array)
    expected = model(torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert torch.equal(trace.output_ops[0].out, expected.detach())


def test_notebook_grid_slices_convert_on_a_broken_bridge(broken_bridge: None) -> None:
    """The notebook grid's ``numpy(force=True)`` port keeps working without the bridge."""

    from torchlens.report._notebook._truncation import _to_numpy

    source = torch.arange(6, dtype=torch.float16, requires_grad=True).reshape(2, 3)
    array = _to_numpy(source)
    assert array.dtype.name == "float32"
    assert array.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_host_array_converts_tensors_and_array_likes_on_a_broken_bridge(
    broken_bridge: None,
) -> None:
    """``host_array`` sends tensors through the fallback and array-likes to NumPy."""

    tensor = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)
    assert bridge.host_array(tensor).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert bridge.host_array([1, 2]).tolist() == [1, 2]
