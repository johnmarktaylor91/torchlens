"""One-model execution worker that emits observations, never ``runs`` authority."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from menagerie.crawler.constants import RunMode
from menagerie.crawler.frameworks import NativeForwardAdapter
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.modes import (
    classify_train_eval_divergence,
    detect_meaningful_modes,
    output_signature,
)
from menagerie.crawler.policy import ExecutionPolicy, PolicyObservation, PolicyViolation
from menagerie.crawler.recipe import LoadedRecipe, load_recipe
from menagerie.crawler.standard_inputs import (
    InputSpec,
    MaterializedInput,
    materialize_standard_input,
)


@dataclass(frozen=True)
class WorkerRequest:
    """Complete typed request for one isolated model execution.

    Parameters
    ----------
    stable_id:
        Durable model identity used only for receipt association.
    recipe:
        Closed declarative or typed-adapter recipe.
    modality, input_spec:
        Source-gated modality and concrete single-tensor call contract.
    scratch_root, receipt_path:
        Writable worker and atomic result locations.
    seed, device, framework:
        Deterministic native execution settings.
    meaningful_modes:
        Explicit meaningful modes. ``None`` conservatively defaults to both.
    source_identity, execution_identity:
        Parent-computed identities echoed for receipt binding, never awarded.
    """

    stable_id: str
    recipe: Mapping[str, Any]
    modality: Union[str, tuple[str, ...], None]
    input_spec: InputSpec
    scratch_root: Path
    receipt_path: Path
    seed: int = 0
    device: str = "cpu"
    framework: str = "pytorch"
    meaningful_modes: Optional[tuple[RunMode, ...]] = None
    source_identity: str = "unbound"
    execution_identity: str = "unbound"

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, receipt_path: Optional[Path] = None
    ) -> "WorkerRequest":
        """Validate and normalize a JSON worker request.

        Parameters
        ----------
        value:
            JSON-compatible request mapping.
        receipt_path:
            Optional argv-owned receipt path override.

        Returns
        -------
        WorkerRequest
            Typed request.
        """

        recipe = value.get("recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError("worker recipe must be an object")
        input_value = value.get("input_spec")
        if not isinstance(input_value, Mapping):
            raise ValueError("worker input_spec must be an object")
        modality_value = value.get("modality")
        if isinstance(modality_value, list):
            modality: Union[str, tuple[str, ...], None] = tuple(
                str(item) for item in modality_value
            )
        elif modality_value is None or isinstance(modality_value, str):
            modality = modality_value
        else:
            raise ValueError("worker modality must be a string, string list, or null")
        modes_value = value.get("meaningful_modes")
        modes: Optional[tuple[RunMode, ...]] = None
        if modes_value is not None:
            if not isinstance(modes_value, list) or not modes_value:
                raise ValueError("meaningful_modes must be a non-empty list when supplied")
            modes = tuple(RunMode(str(mode)) for mode in modes_value)
        if receipt_path is None:
            raw_receipt = value.get("receipt_path")
            if not isinstance(raw_receipt, str) or not raw_receipt:
                raise ValueError("worker receipt_path must be provided")
            receipt_value = Path(raw_receipt)
        else:
            receipt_value = receipt_path
        return cls(
            stable_id=str(value.get("stable_id", "")),
            recipe=dict(recipe),
            modality=modality,
            input_spec=InputSpec.from_value(input_value),
            scratch_root=Path(str(value["scratch_root"])),
            receipt_path=receipt_value,
            seed=int(value.get("seed", 0)),
            device=str(value.get("device", "cpu")),
            framework=str(value.get("framework", "pytorch")),
            meaningful_modes=modes,
            source_identity=str(value.get("source_identity", "unbound")),
            execution_identity=str(value.get("execution_identity", "unbound")),
        )


def _seed_frameworks(seed: int, framework: str) -> None:
    """Seed Python, NumPy, and the selected native framework.

    Parameters
    ----------
    seed:
        Deterministic execution seed.
    framework:
        Selected native execution framework.
    """

    random.seed(seed)
    np.random.seed(seed)
    normalized = framework.lower()
    if normalized in {"torch", "pytorch"}:
        import torch

        torch.manual_seed(seed)
    elif normalized in {"tensorflow", "tf"}:
        import tensorflow as tf

        tf.random.set_seed(seed)
    elif normalized in {"paddle", "paddlepaddle"}:
        import paddle

        paddle.seed(seed)


def _parameter_counts(model: object) -> tuple[Optional[int], Optional[int]]:
    """Count native model parameters when an iterator is available.

    Parameters
    ----------
    model:
        Native model or transparent adapter.

    Returns
    -------
    tuple[int | None, int | None]
        Total and trainable counts.
    """

    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return (None, None)
    try:
        values = tuple(parameters())
        total = sum(int(parameter.numel()) for parameter in values)
        trainable = sum(
            int(parameter.numel())
            for parameter in values
            if bool(getattr(parameter, "requires_grad", False))
        )
    except (AttributeError, TypeError, RuntimeError):
        return (None, None)
    return (total, trainable)


def _input_signature(materialized: MaterializedInput) -> dict[str, Any]:
    """Describe the standard input without retaining payload values.

    Parameters
    ----------
    materialized:
        Built standard or fallback input.

    Returns
    -------
    dict[str, Any]
        Shape, dtype, device, and pytree location.
    """

    value = materialized.value
    return {
        "tree": {"tuple": [{"leaf": 0}]},
        "leaves": [
            {
                "path": "args[0]",
                "kind": "tensor",
                "shape": list(materialized.spec.shape),
                "dtype": str(getattr(value, "dtype", materialized.spec.dtype)),
                "device": str(getattr(value, "device", "")) or None,
                "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        ],
    }


def _set_mode(model: object, mode: RunMode) -> None:
    """Explicitly select native train or eval mode.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    mode:
        Requested runtime mode.
    """

    method = getattr(model, mode.value, None)
    if not callable(method):
        if mode is RunMode.TRAIN:
            train_method = getattr(model, "train", None)
            if callable(train_method):
                train_method(True)
                return
        raise TypeError(f"model does not expose explicit {mode.value}()")
    method()


def _inference_context(framework: str) -> Any:
    """Return a no-grad context when the native framework supports one.

    Parameters
    ----------
    framework:
        Native execution framework.

    Returns
    -------
    Any
        Context manager.
    """

    if framework.lower() in {"torch", "pytorch"}:
        import torch

        return torch.no_grad()
    return nullcontext()


def _native_metadata(model: object, framework: str) -> tuple[str, str]:
    """Return native framework and explicitly delegated method.

    Parameters
    ----------
    model:
        Native model or transparent adapter.
    framework:
        Requested execution framework.

    Returns
    -------
    tuple[str, str]
        Framework and delegated method.
    """

    if isinstance(model, NativeForwardAdapter):
        return (model.metadata.run_framework, model.metadata.native_call_method)
    return (framework, "forward")


def _mode_receipt(
    model: object,
    materialized: MaterializedInput,
    mode: RunMode,
    framework: str,
    constructor_seconds: float,
) -> tuple[dict[str, Any], Optional[object]]:
    """Run one explicit forward and produce an honest per-mode receipt.

    Parameters
    ----------
    model:
        Constructed native model.
    materialized:
        Shared standard input.
    mode:
        Explicit runtime mode.
    framework:
        Native execution framework.
    constructor_seconds:
        Already observed construction duration.

    Returns
    -------
    tuple[dict[str, Any], object | None]
        Per-mode receipt and captured output retained only in memory.
    """

    total, trainable = _parameter_counts(model)
    native_framework, delegated_method = _native_metadata(model, framework)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "mode": mode.value,
        "constructor_started": True,
        "constructor_completed": True,
        "constructor_seconds": constructor_seconds,
        "input_completed": True,
        "input_signature": _input_signature(materialized),
        "input_kind": materialized.input_kind,
        "input_asset": materialized.input_asset,
        "input_note": materialized.input_note,
        "forward_started": False,
        "forward_completed": False,
        "forward_seconds": None,
        "output_signature": None,
        "parameter_count_total": total,
        "parameter_count_trainable": trainable,
        "native_framework": native_framework,
        "delegated_method": delegated_method,
        "error": None,
    }
    try:
        _set_mode(model, mode)
        forward = getattr(model, "forward", None)
        if not callable(forward):
            raise TypeError("constructed model does not expose a callable forward attribute")
        receipt["forward_started"] = True
        with _inference_context(framework):
            output = forward(*materialized.args, **materialized.kwargs)
        receipt["forward_seconds"] = time.monotonic() - started
        receipt["forward_completed"] = True
        receipt["output_signature"] = output_signature(output)
        return receipt, output
    except Exception as exc:  # noqa: BLE001 -- receipt must retain native failures
        receipt["forward_seconds"] = time.monotonic() - started
        receipt["error"] = {
            "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return receipt, None


def _atomic_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically fsync and rename one complete worker receipt.

    Parameters
    ----------
    path:
        Final receipt path.
    payload:
        Receipt without its self hash.

    Returns
    -------
    dict[str, Any]
        Persisted payload including ``receipt_sha256``.
    """

    record = dict(payload)
    record["receipt_sha256"] = stable_hash(payload)
    data = canonical_json_bytes(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return record


def _execute(request: WorkerRequest) -> tuple[dict[str, Any], PolicyObservation]:
    """Execute one model under active in-child policy tripwires.

    Parameters
    ----------
    request:
        Typed worker request.

    Returns
    -------
    tuple[dict[str, Any], PolicyObservation]
        Receipt payload and policy observation.
    """

    request.scratch_root.mkdir(parents=True, exist_ok=True)
    policy = ExecutionPolicy(request.scratch_root, request.receipt_path.parent)
    base: dict[str, Any] = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": request.stable_id,
        "source_identity": request.source_identity,
        "execution_identity": request.execution_identity,
        "seed": request.seed,
        "device": request.device,
        "framework": request.framework,
        "awards_runs": False,
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "meaningful_modes": [],
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "error": None,
    }
    with policy:
        try:
            _seed_frameworks(request.seed, request.framework)
            loaded: LoadedRecipe = load_recipe(
                request.recipe, source_identity=request.source_identity
            )
            base["recipe_revision"] = loaded.recipe_revision
            base["constructor_started"] = True
            constructor_started = time.monotonic()
            model = loaded.build_model()
            constructor_seconds = time.monotonic() - constructor_started
            base["constructor_completed"] = True
            materialized = materialize_standard_input(
                request.modality,
                request.input_spec,
                framework=request.framework,
                device=request.device,
                seed=request.seed,
            )
            base["input_completed"] = True
            modes = request.meaningful_modes or (RunMode.TRAIN, RunMode.EVAL)
            if not modes:
                modes = detect_meaningful_modes(model)
            base["meaningful_modes"] = [mode.value for mode in modes]
            outputs: dict[str, object] = {}
            for mode in modes:
                receipt, output = _mode_receipt(
                    model, materialized, mode, request.framework, constructor_seconds
                )
                base["per_mode"][mode.value] = receipt
                if output is not None:
                    outputs[mode.value] = output
            if {RunMode.TRAIN.value, RunMode.EVAL.value}.issubset(outputs):
                divergence = classify_train_eval_divergence(outputs["train"], outputs["eval"])
                base["train_eval_divergence"] = divergence.classification
                base["divergence_evidence"] = divergence.evidence
            elif len(outputs) == 1:
                base["train_eval_divergence"] = "none"
                base["divergence_evidence"] = "only one meaningful mode"
        except Exception as exc:  # noqa: BLE001 -- constructor/input/policy failures are receipts
            reason_code = exc.reason_code if isinstance(exc, PolicyViolation) else None
            base["error"] = {
                "reason_code": reason_code,
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    base["policy_observation"] = policy.observation.to_dict()
    return base, policy.observation


def run_worker(request: WorkerRequest) -> dict[str, Any]:
    """Execute one model and atomically persist its observation receipt.

    Parameters
    ----------
    request:
        Complete typed worker request.

    Returns
    -------
    dict[str, Any]
        Persisted receipt. This function never awards the canonical ``runs`` status.
    """

    payload, _observation = _execute(request)
    return _atomic_receipt(request.receipt_path, payload)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the private worker subprocess CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    argparse.Namespace
        Parsed request and receipt paths.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the argv-only worker entry point.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Zero when every requested forward completed; one otherwise.
    """

    args = _parse_args(argv)
    request_value = json.loads(args.request.read_text(encoding="utf-8"))
    if not isinstance(request_value, dict):
        raise ValueError("worker request must be a JSON object")
    request = WorkerRequest.from_mapping(request_value, receipt_path=args.receipt)
    receipt = run_worker(request)
    per_mode = receipt.get("per_mode", {})
    succeeded = bool(per_mode) and all(
        bool(item.get("forward_completed"))
        for item in per_mode.values()
        if isinstance(item, Mapping)
    )
    return 0 if receipt.get("error") is None and succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
