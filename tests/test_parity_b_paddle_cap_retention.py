"""Paddle ``save=`` decided at capture: sparse payloads, full validation, lower peak."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

paddle = pytest.importorskip("paddle")

import torchlens as tl  # noqa: E402
from torchlens.backends.paddle import PaddleBackend  # noqa: E402
from torchlens.backends.paddle._retention import STATE_KEY, SpilledPayloads  # noqa: E402

pytestmark = [pytest.mark.backend_paddle, pytest.mark.rare]


def _chain(x: Any) -> Any:
    """Run a short chain of distinct Paddle ops.

    Parameters
    ----------
    x
        Input tensor.

    Returns
    -------
    Any
        Chain output.
    """

    h = paddle.multiply(x, x)
    h = paddle.nn.functional.relu(h - 0.25)
    h = paddle.tanh(h)
    h = paddle.add(h, x)
    h = paddle.nn.functional.relu(h)
    return paddle.exp(h * 0.5)


def _input() -> Any:
    """Return a deterministic input tensor."""

    return paddle.arange(24, dtype="float32").reshape([4, 6]) / 12.0 - 0.5


def _ops(trace: Any) -> list[Any]:
    """Return the computed (non-input, non-output) ops of ``trace``."""

    return [op for op in trace._layer_list if op.type not in ("input", "output")]


def _validates(trace: Any) -> None:
    """Assert replay validation of ``trace`` passes."""

    assert PaddleBackend().validate_trace(trace) is True, trace.validation_replay_status
    assert trace.validation_replay_status.state == "passed"


@pytest.mark.parametrize(
    "selector",
    [tl.func("functional.relu"), tl.func("tanh"), tl.func("tanh") | tl.func("exp")],
    ids=["relu", "tanh", "union"],
)
def test_sparse_save_equals_full_capture_on_selected_ops(selector: Any) -> None:
    """Selected payloads equal a full capture's; both traces validate."""

    full = tl.trace(_chain, _input(), backend="paddle")
    sparse = tl.trace(_chain, _input(), backend="paddle", save=selector)

    _validates(full)
    _validates(sparse)
    hidden = sparse._selective_save_hidden_payloads
    assert isinstance(hidden, SpilledPayloads)
    full_by_label = {op.label: op for op in _ops(full)}
    selected = [op for op in _ops(sparse) if bool(selector(op))]
    assert selected
    for op in _ops(sparse):
        if op in selected:
            assert op.has_saved_activation is True
            np.testing.assert_array_equal(op.out.numpy(), full_by_label[op.label].out.numpy())
        else:
            assert op.has_saved_activation is False
            assert op.out is None
            replay = hidden[op.label]
            np.testing.assert_array_equal(replay.numpy(), full_by_label[op.label].out.numpy())


def test_label_selector_reloads_the_selected_op() -> None:
    """A final-label selector keeps exactly that op, whatever capture predicted."""

    full = tl.trace(_chain, _input(), backend="paddle")
    target = _ops(full)[-1].label
    sparse = tl.trace(_chain, _input(), backend="paddle", save=tl.label(target))

    _validates(sparse)
    kept = [op.label for op in _ops(sparse) if op.has_saved_activation]
    assert kept == [target]
    spool = Path(sparse._selective_save_hidden_payloads._spool)
    assert len(list(spool.glob("*.npy"))) >= len(_ops(full)) - 1  # the rest left RAM at capture
    expected = {op.label: op.out.numpy() for op in _ops(full)}
    np.testing.assert_array_equal(
        next(op for op in _ops(sparse) if op.label == target).out.numpy(), expected[target]
    )


def test_sparse_save_matches_post_hoc_save_surface() -> None:
    """Capture-time retention exposes the same public payloads as the post-hoc filter."""

    sparse = tl.trace(_chain, _input(), backend="paddle", save=tl.func("functional.relu"))
    kept = {op.label for op in sparse._layer_list if op.has_saved_activation}
    full = tl.trace(_chain, _input(), backend="paddle")
    expected = {op.label for op in full._layer_list if bool(tl.func("functional.relu")(op))}
    assert kept == expected


def test_plain_capture_is_unchanged() -> None:
    """Without save= every payload stays in RAM and no spool exists."""

    trace = tl.trace(_chain, _input(), backend="paddle")

    assert STATE_KEY not in trace.__dict__
    assert not isinstance(getattr(trace, "_selective_save_hidden_payloads", None), SpilledPayloads)
    assert all(op.has_saved_activation and op.out is not None for op in _ops(trace))
    _validates(trace)


def test_spool_is_removed_with_the_trace() -> None:
    """The spool directory lives exactly as long as the trace's payload table."""

    trace = tl.trace(_chain, _input(), backend="paddle", save=tl.func("tanh"))
    spool = Path(trace._selective_save_hidden_payloads._spool)
    assert spool.is_dir() and any(spool.iterdir())

    del trace
    gc.collect()
    assert not spool.exists()


def test_failed_capture_leaves_no_spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that raises removes its spool."""

    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def boom(x: Any) -> Any:
        _chain(x)
        raise RuntimeError("model failed")

    with pytest.raises(Exception, match="model failed"):
        tl.trace(boom, _input(), backend="paddle", save=tl.func("tanh"))
    gc.collect()
    assert not any(tmp_path.glob("torchlens-paddle-retention-*"))


_PEAK_SCRIPT = textwrap.dedent(
    """
    import gc, json, sys
    import paddle
    import torchlens as tl

    def status_kib(field):
        for line in open("/proc/self/status"):
            if line.startswith(field + ":"):
                return int(line.split()[1])
        raise KeyError(field)

    def chain(v):
        for _ in range(12):
            v = paddle.nn.functional.relu(paddle.multiply(paddle.add(v, 1.0), 0.5))
        return paddle.exp(v * 0.01)

    config = sys.argv[1]
    small = paddle.ones([64], dtype="float32")
    warm = tl.trace(chain, small, backend="paddle")  # warm-up; names the last op's final label
    last = [op for op in warm._layer_list if op.type not in ("input", "output")][-1].label
    tl.trace(chain, small, backend="paddle", save=tl.func("exp"))
    del warm
    x = paddle.ones([1 << 20], dtype="float32")  # 4 MiB per activation
    gc.collect()
    base = status_kib("VmRSS")
    open("/proc/self/clear_refs", "w").write("5")
    save = {"func": tl.func("exp"), "label": tl.label(last)}.get(config)
    kw = {} if save is None else {"save": save}
    trace = tl.trace(chain, x, backend="paddle", **kw)
    print(json.dumps({"peak_bytes": (status_kib("VmHWM") - base) * 1024}))
    """
)


def _peak(config: str) -> int:
    """Return a fresh process's capture peak RSS delta for ``config``."""

    done = subprocess.run(
        [sys.executable, "-c", _PEAK_SCRIPT, config],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return int(json.loads(done.stdout.strip().splitlines()[-1])["peak_bytes"])


@pytest.mark.heavy
@pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc")
@pytest.mark.parametrize("config", ["func", "label"])
def test_capture_time_save_lowers_peak_memory(config: str) -> None:
    """save= of the last op (by function or final label) peaks well below a full capture."""

    full = _peak("full")
    sparse = _peak(config)
    assert full > 100 * 2**20, full
    assert sparse < full // 2, (full, sparse)
