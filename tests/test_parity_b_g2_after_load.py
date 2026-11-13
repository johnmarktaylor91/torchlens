"""Validation after reload, per non-torch backend, in a fresh process without the model.

Each roster row is captured and saved at the conformance adapter's
``replay_save_level`` together with its reference outputs (as NumPy files).
A fresh interpreter then loads the artifact, rebuilds the reference outputs
from the files and validates. The verdict must equal the live one: a row
that passes live passes after load as the conformance cell
``c2_validation_after_load`` judges it, any other live verdict recurs with its
state and reason, and a payload changed after load fails naming the op or its
consumer wherever the live validator names it. An artifact written without replay recipes
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
from tests.conformance._protocol_cases import adapter_protocol
from tests.conformance._validation_cells import (
    capture_for_validation,
    names_reason_problems,
    tamper_target,
    tampered_payload,
)
from torchlens._io import _trace_sidecars
from torchlens.backends._replay_adapters import has_adapter

_REPO = Path(__file__).resolve().parents[1]
_STRIPPED = "loaded_trace_runtime_capture_stripped"

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


def _live_verdicts(adapter: Any, model: Any, outputs: list[Any]) -> dict[str, Any]:
    """The live trace's clean verdict and whether a live tamper is named."""

    import warnings

    net, example = model.builder()
    protocol = adapter_protocol(adapter)
    clean = protocol.validate(capture_for_validation(adapter, net, example), outputs)
    trace = capture_for_validation(adapter, net, example)
    target = tamper_target(trace)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        target.out = tampered_payload(target.out)
    tampered = protocol.validate(trace, outputs)
    return {
        "state": clean.state,
        "reason": clean.reason,
        "tamper_named": not names_reason_problems(tampered, target, trace),
    }


def _save_row(
    adapter: Any, model: Any, root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Capture one row, save it three ways with its reference outputs; return live verdicts."""

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
    return _live_verdicts(adapter, model, outputs)


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


def _row_problems(row: str, live: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Where the fresh-process verdicts fall short of the live ones for one row.

    A row that passes live must pass after load with registered reasons; any
    other live verdict must recur after load with the same state and reason. A
    tamper the live validator names must be named after load. Artifacts saved
    without replay recipes must stay unavailable.
    """

    problems: list[str] = []
    clean = result["clean"]
    if live["state"] == "passed":
        problems += [f"{row}: {item}" for item in result["clean_problems"]]
    elif (clean["state"], clean["reason"]) != (live["state"], live["reason"]):
        problems.append(f"{row}: after load {clean}, live {live}")
    if live["tamper_named"]:
        problems += [f"{row} tampered: {item}" for item in result["tampered_problems"]]
    for kind in ("portable", "old"):
        verdict = result[kind]
        if (verdict["state"], verdict["reason"]) != ("unavailable", _STRIPPED):
            problems.append(f"{row} {kind}: {verdict}")
    return problems


def _run_backend(backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = resolve_adapter(backend)
    reason = adapter.unavailable_reason()
    if reason is not None:
        pytest.skip(f"{backend} framework unavailable: {reason}")
    if not has_adapter(backend):
        pytest.skip(f"{backend} replay adapter has not landed on this branch")
    problems: list[str] = []
    for model in adapter.default_roster():
        root = tmp_path / model.family
        root.mkdir()
        live = _save_row(adapter, model, root, monkeypatch)
        result = _fresh_validate(backend, root)
        problems += _row_problems(f"{backend}/{model.family}", live, result)
    assert not problems, "\n".join(problems)


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
