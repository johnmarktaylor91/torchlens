"""Paddle ``storage=tl.to_disk(path)``: payloads stream during capture and load back as Paddle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens._errors import InvalidArgumentError  # noqa: E402
from torchlens._io.streaming import BundleStreamWriter  # noqa: E402
from torchlens.backends.paddle import _stream_capture  # noqa: E402
from torchlens.backends.paddle._retention import STREAM_KEY  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


def _chain(x: Any) -> Any:
    """Run a short chain of distinct Paddle ops."""

    h = paddle.multiply(x, x)
    h = paddle.nn.functional.relu(h - 0.25)
    h = paddle.tanh(h)
    h = paddle.add(h, x)
    return paddle.exp(h * 0.5)


def _input() -> Any:
    """Return a deterministic 4 x 6 float32 input."""

    return paddle.arange(24, dtype="float32").reshape([4, 6]) / 12.0 - 0.5


def _assert_payloads_equal(actual: Any, expected: Any) -> None:
    """Assert both traces have the same ops and equal Paddle payloads, op by op."""

    assert [op.label for op in actual._layer_list] == [op.label for op in expected._layer_list]
    for got, want in zip(actual._layer_list, expected._layer_list, strict=True):
        assert type(got.out).__module__.startswith("paddle"), (got.label, type(got.out))
        assert str(got.out.dtype) == str(want.out.dtype), got.label
        np.testing.assert_array_equal(got.out.numpy(), want.out.numpy())


def test_streamed_bundle_loads_payloads_equal_to_plain_capture(tmp_path: Path) -> None:
    """tl.load of the streamed bundle gives the plain capture's payloads as Paddle tensors."""

    plain = tl.trace(_chain, _input(), backend="paddle")
    path = tmp_path / "streamed"
    streamed = tl.trace(_chain, _input(), backend="paddle", storage=tl.to_disk(path))

    assert path.is_dir()
    assert Path(streamed._source_bundle_path) == path
    loaded = tl.load(path)
    assert loaded.provenance.backend == "paddle"
    _assert_payloads_equal(loaded, plain)


def test_streamed_payloads_leave_ram_by_default(tmp_path: Path) -> None:
    """Without retain_in_memory no op keeps its payload; each is saved and in the bundle."""

    streamed = tl.trace(_chain, _input(), backend="paddle", storage=tl.to_disk(tmp_path / "b"))

    assert streamed._layer_list
    assert all(op.has_saved_activation for op in streamed._layer_list)
    assert all(op._slot("out") is None for op in streamed._layer_list)
    assert all(op.out_ref is not None for op in streamed._layer_list)
    assert streamed._out_writer is None
    assert STREAM_KEY not in streamed.__dict__


def test_retain_in_memory_keeps_payloads(tmp_path: Path) -> None:
    """retain_in_memory=True keeps each payload in RAM and still publishes the bundle."""

    plain = tl.trace(_chain, _input(), backend="paddle")
    path = tmp_path / "kept"
    storage = tl.to_disk(path, retain_in_memory=True)
    streamed = tl.trace(_chain, _input(), backend="paddle", storage=storage)

    _assert_payloads_equal(streamed, plain)
    _assert_payloads_equal(tl.load(path), plain)


def test_synchronous_writes_publish_the_same_bundle(tmp_path: Path) -> None:
    """async_writes=False writes each payload inline and loads back equal."""

    plain = tl.trace(_chain, _input(), backend="paddle")
    path = tmp_path / "sync"
    tl.trace(_chain, _input(), backend="paddle", storage=tl.to_disk(path, async_writes=False))

    _assert_payloads_equal(tl.load(path), plain)


def test_pending_bytes_stay_within_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The async pipeline gets max_pending_bytes and never holds more than one payload over it."""

    armed: list[tuple[int | None, Any]] = []
    original = BundleStreamWriter.arm_async_writes

    def spy(self: Any, max_pending_bytes: int | None = None) -> None:
        original(self, max_pending_bytes)
        armed.append((max_pending_bytes, self._async_engine))

    monkeypatch.setattr(BundleStreamWriter, "arm_async_writes", spy)
    plain = tl.trace(_chain, _input(), backend="paddle")
    path = tmp_path / "budget"
    storage = tl.to_disk(path, max_pending_bytes=64)
    tl.trace(_chain, _input(), backend="paddle", storage=storage)

    assert [budget for budget, _engine in armed] == [64]
    largest = max(op.out.numpy().nbytes for op in plain._layer_list)
    assert 0 < armed[0][1]._peak_pending_bytes <= max(64, largest)
    _assert_payloads_equal(tl.load(path), plain)


@pytest.mark.parametrize("dtype_name", ["float32", "int64", "bool", "float16", "bfloat16"])
def test_streamed_dtypes_round_trip(tmp_path: Path, dtype_name: str) -> None:
    """Each saveable Paddle dtype streams and loads back with its dtype and bits."""

    base = paddle.to_tensor([[-1.5, -0.25], [0.125, 3.5]], dtype="float32")
    value = base.cast(dtype_name) if dtype_name != "bool" else base > 0
    path = tmp_path / f"stream_{dtype_name}"
    tl.trace(lambda t: paddle.assign(t), value, backend="paddle", storage=tl.to_disk(path))

    loaded = tl.load(path)
    restored = loaded[loaded.output_layers[0]].out
    assert str(restored.dtype) == str(value.dtype)
    np.testing.assert_array_equal(restored.numpy(), value.numpy())


def test_failed_capture_publishes_nothing(tmp_path: Path) -> None:
    """A capture that raises leaves no bundle at the path, only a PARTIAL temp directory."""

    def boom(x: Any) -> Any:
        _chain(x)
        raise RuntimeError("model failed")

    path = tmp_path / "failed"
    with pytest.raises(Exception, match="model failed"):
        tl.trace(boom, _input(), backend="paddle", storage=tl.to_disk(path))

    assert not path.exists()
    temps = list(tmp_path.glob("failed.tmp.*"))
    assert temps and all((temp / "PARTIAL").exists() for temp in temps)


def test_existing_bundle_path_is_refused(tmp_path: Path) -> None:
    """An existing target is refused before capture, as for torch streaming."""

    path = tmp_path / "taken"
    path.mkdir()
    with pytest.raises(Exception) as info:
        tl.trace(_chain, _input(), backend="paddle", storage=tl.to_disk(path))
    assert info.value.fields.get("code") == "bundle_path_exists"


def test_storage_with_save_selector_is_refused(tmp_path: Path) -> None:
    """storage= with save= is refused typed until streamed selection is built."""

    path = tmp_path / "sparse"
    with pytest.raises(InvalidArgumentError) as info:
        tl.trace(
            _chain,
            _input(),
            backend="paddle",
            storage=tl.to_disk(path),
            save=tl.func("tanh"),
        )
    assert info.value.fields.get("code") == "trace_option_unsupported"
    assert not path.exists()


def test_plain_capture_is_unchanged_and_saves(tmp_path: Path) -> None:
    """Without storage= payloads stay in RAM, no capture-mode state remains, and save works."""

    trace = tl.trace(_chain, _input(), backend="paddle")

    assert all(op.out is not None for op in trace._layer_list)
    assert getattr(trace, "_source_bundle_path", None) is None
    assert STREAM_KEY not in trace.__dict__
    assert _stream_capture.MODE_KEY not in trace.__dict__
    path = tmp_path / "saved"
    trace.save(path)
    _assert_payloads_equal(tl.load(path), trace)
