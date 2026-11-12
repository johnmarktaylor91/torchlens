"""Validation after reload, per non-torch backend, in a fresh process without the model.

Each roster row is captured and saved at the conformance adapter's
``replay_save_level`` together with its reference outputs (as NumPy files).
A fresh interpreter then loads the artifact, rebuilds the reference outputs
from the files and validates: the row must pass exactly as the conformance
cell ``c2_validation_after_load`` judges it; a payload changed after load must
fail naming the op or its consumer; an artifact written without replay recipes
(a default-level save, or a bundle from before the recipe section) must stay
unavailable. Rows run only where the framework and its replay adapter exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests.conformance._adapters import resolve_adapter
from tests.conformance._validation_cells import capture_for_validation
from torchlens._io import _trace_sidecars
from torchlens.backends._replay_adapters import has_adapter

_REPO = Path(__file__).resolve().parents[1]

#: Runs in the fresh interpreter: never imports or builds a roster model.
_FRESH = textwrap.dedent(
    """
    import json, sys, warnings
    import numpy as np
    from tests.conformance._adapters import resolve_adapter
    from tests.conformance._protocol_cases import adapter_protocol
    from tests.conformance._validation_cells import (
        names_reason_problems, tamper_target, tampered_payload, validation_status_problems,
    )

    backend, root = sys.argv[1], sys.argv[2]
    adapter = resolve_adapter(backend)
    protocol = adapter_protocol(adapter)
    meta = json.loads(open(f"{root}/meta.json").read())

    def framework_array(value):
        if backend == "jax":
            import jax.numpy as jnp
            return jnp.asarray(value)
        if backend == "tf":
            import tensorflow as tf
            return tf.convert_to_tensor(value)
        if backend == "mlx":
            import mlx.core as mx
            return mx.array(value)
        if backend == "paddle":
            import paddle
            return paddle.to_tensor(value)
        from tinygrad import Tensor
        return Tensor(value)

    reference = [framework_array(np.load(f"{root}/ref_{i}.npy")) for i in range(meta["n_ref"])]

    def summary(status):
        return {
            "state": getattr(status, "state", None),
            "reason": getattr(status, "reason", None),
            "message": str(getattr(status, "message", ""))[:400],
        }

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = protocol.validate(adapter.load(f"{root}/replay.tlspec"), reference)
        out["clean"] = summary(status)
        out["clean_problems"] = list(validation_status_problems(status))
        loaded = adapter.load(f"{root}/replay.tlspec")
        target = tamper_target(loaded)
        target.out = tampered_payload(target.out)
        tampered = protocol.validate(loaded, reference)
        out["tampered"] = summary(tampered)
        out["tampered_problems"] = list(names_reason_problems(tampered, target, loaded))
        out["portable"] = summary(protocol.validate(adapter.load(f"{root}/portable.tlspec"), reference))
        out["old"] = summary(protocol.validate(adapter.load(f"{root}/old.tlspec"), reference))
    print("RESULT " + json.dumps(out))
    """
)


def _save_row(adapter: Any, model: Any, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture one row and save it three ways, with its reference outputs."""

    import numpy as np

    net, example = model.builder()
    trace = capture_for_validation(adapter, net, example)
    reference = adapter.reference_output(net, example)
    outputs = list(reference) if isinstance(reference, (tuple, list)) else [reference]
    for index, value in enumerate(outputs):
        np.save(root / f"ref_{index}.npy", adapter.as_array(value))
    (root / "meta.json").write_text(json.dumps({"n_ref": len(outputs)}))
    adapter.save(trace, root / "replay.tlspec", level=adapter.replay_save_level)
    adapter.save(trace, root / "portable.tlspec")
    with monkeypatch.context() as patch:
        # A bundle written before the replay recipe section existed.
        patch.setattr(_trace_sidecars, "write_replay_recipes", lambda *a, **k: False)
        adapter.save(trace, root / "old.tlspec", level=adapter.replay_save_level)


def _fresh_validate(backend: str, root: Path) -> dict[str, Any]:
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH, backend, str(root)],
        capture_output=True,
        text=True,
        cwd=_REPO,
        env=env,
        timeout=600,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def _run_backend(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = resolve_adapter(backend)
    reason = adapter.unavailable_reason()
    if reason is not None:
        pytest.skip(f"{backend} framework unavailable: {reason}")
    if not has_adapter(backend):
        pytest.skip(f"{backend} replay adapter has not landed on this branch")
    for model in adapter.default_roster():
        root = tmp_path / model.family
        root.mkdir()
        _save_row(adapter, model, root, monkeypatch)
        result = _fresh_validate(backend, root)
        row = f"{backend}/{model.family}"
        assert result["clean_problems"] == [], (row, result["clean"])
        assert result["tampered_problems"] == [], (row, result["tampered"])
        for kind in ("portable", "old"):
            assert result[kind]["state"] == "unavailable", (row, kind, result[kind])
            assert result[kind]["reason"] == "loaded_trace_runtime_capture_stripped", result[kind]


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_jax
def test_jax_validates_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_backend("jax", tmp_path, monkeypatch)


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.tf_backend
def test_tf_validates_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_backend("tf", tmp_path, monkeypatch)


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_mlx
def test_mlx_validates_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_backend("mlx", tmp_path, monkeypatch)


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_paddle
def test_paddle_validates_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_backend("paddle", tmp_path, monkeypatch)


@pytest.mark.rare
@pytest.mark.heavy
@pytest.mark.backend_tinygrad
def test_tinygrad_validates_after_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_backend("tinygrad", tmp_path, monkeypatch)
