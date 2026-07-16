"""One-model execution worker that emits observations, never ``runs`` authority."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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
from menagerie.crawler.identity import (
    canonical_json_bytes,
    compute_recipe_revision,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.modes import (
    classify_train_eval_divergence,
    detect_meaningful_modes,
    output_signature,
    output_value_sha256,
)
from menagerie.crawler.policy import ExecutionPolicy, PolicyObservation, PolicyViolation
from menagerie.crawler.recipe import LoadedRecipe, RecipeError, load_recipe
from menagerie.crawler.standard_inputs import (
    ASSET_ROOT,
    InputSpec,
    materialize_standard_input,
)

_PARENT_COMPLETION_CHALLENGE_ENV = "MENAGERIE_PARENT_COMPLETION_CHALLENGE"
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "


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
    seed, input_seed, device, framework:
        Deterministic native execution settings.
    mode:
        Explicit single mode for this fresh worker process. ``None`` is retained only
        for compatibility with pre-round-6 requests.
    meaningful_modes:
        Explicit meaningful modes. ``None`` conservatively defaults to both.
    source_identity, execution_identity:
        Parent-computed identities echoed for receipt binding, never awarded.
    recipe_revision, recipe_identity_payload:
        Parent-accepted revision and the complete non-self-referential facts the
        worker recomputes before importing any recipe implementation.
    """

    stable_id: str
    recipe: Mapping[str, Any]
    modality: Union[str, tuple[str, ...], None]
    input_spec: Optional[InputSpec]
    scratch_root: Path
    receipt_path: Path
    input_contract: Optional[Mapping[str, Any]] = None
    input_manifest: Optional[Mapping[str, Any]] = None
    seed: int = 0
    input_seed: int = 0
    device: str = "cpu"
    framework: str = "pytorch"
    meaningful_modes: Optional[tuple[RunMode, ...]] = None
    mode: Optional[RunMode] = None
    source_identity: str = "unbound"
    execution_identity: str = "unbound"
    recipe_revision: str = "unbound"
    recipe_identity_payload: Optional[Mapping[str, Any]] = None

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
        input_contract_value = value.get("input_contract")
        input_manifest_value = value.get("input_manifest")
        if input_value is not None and not isinstance(input_value, Mapping):
            raise ValueError("worker input_spec must be an object or null")
        if input_contract_value is not None and not isinstance(input_contract_value, Mapping):
            raise ValueError("worker input_contract must be an object or null")
        if input_manifest_value is not None and not isinstance(input_manifest_value, Mapping):
            raise ValueError("worker input_manifest must be an object or null")
        if input_value is None and input_contract_value is None:
            raise ValueError("worker requires input_contract or legacy input_spec")
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
        mode_value = value.get("mode")
        mode = None if mode_value is None else RunMode(str(mode_value))
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
            input_spec=InputSpec.from_value(input_value) if input_value is not None else None,
            input_contract=(
                dict(input_contract_value) if isinstance(input_contract_value, Mapping) else None
            ),
            input_manifest=(
                dict(input_manifest_value) if isinstance(input_manifest_value, Mapping) else None
            ),
            scratch_root=Path(str(value["scratch_root"])),
            receipt_path=receipt_value,
            seed=int(value.get("seed", 0)),
            input_seed=int(value.get("input_seed", 0)),
            device=str(value.get("device", "cpu")),
            framework=str(value.get("framework", "pytorch")),
            meaningful_modes=modes,
            mode=mode,
            source_identity=str(value.get("source_identity", "unbound")),
            execution_identity=str(value.get("execution_identity", "unbound")),
            recipe_revision=str(value.get("recipe_revision", "unbound")),
            recipe_identity_payload=(
                dict(value["recipe_identity_payload"])
                if isinstance(value.get("recipe_identity_payload"), Mapping)
                else None
            ),
        )


def _observe_request_bytes(
    request: WorkerRequest,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Verify request-bound helper and standard-asset bytes before execution.

    Parameters
    ----------
    request:
        Complete worker request with parent-bound byte manifests.

    Returns
    -------
    tuple[str | None, str | None, str | None]
        Observed code-manifest and selected-input-asset digests plus an optional
        fail-closed mismatch message.
    """

    manifest_value = request.recipe.get("code_manifest")
    expected_manifest = request.recipe.get("code_manifest_sha256")
    observed_manifest: Optional[str] = None
    errors: list[str] = []
    if manifest_value is not None or expected_manifest is not None:
        if not isinstance(manifest_value, list) or not manifest_value:
            raise RecipeError("typed-adapter worker request has no complete code manifest")
        if not isinstance(expected_manifest, str):
            raise RecipeError("typed-adapter worker request has no code-manifest digest")
        observed_members: list[dict[str, str]] = []
        for member in manifest_value:
            if not isinstance(member, Mapping):
                raise RecipeError("typed-adapter code-manifest member must be an object")
            path_value = member.get("path")
            identity_path = member.get("identity_path")
            expected_digest = member.get("sha256")
            if not all(
                isinstance(value, str) and value
                for value in (path_value, identity_path, expected_digest)
            ):
                raise RecipeError("typed-adapter code-manifest member is incomplete")
            member_path = Path(str(path_value))
            try:
                digest = hash_bytes(member_path.read_bytes())
            except OSError as exc:
                errors.append(f"typed-adapter code-manifest member is unreadable: {exc}")
                continue
            if digest != expected_digest:
                errors.append(f"typed-adapter code-manifest member changed: {identity_path}")
            observed_members.append({"path": str(identity_path), "sha256": digest})
        observed_manifest = stable_hash(observed_members)
        if observed_manifest != expected_manifest:
            errors.append("typed-adapter code-manifest aggregate digest changed")

    selected_asset = (
        request.input_manifest.get("selected_asset")
        if isinstance(request.input_manifest, Mapping)
        else None
    )
    observed_asset: Optional[str] = None
    if selected_asset is not None:
        if not isinstance(selected_asset, Mapping):
            raise RecipeError("worker selected-asset manifest must be an object")
        path_value = selected_asset.get("path")
        expected_digest = selected_asset.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_digest, str):
            raise RecipeError("worker selected-asset manifest is incomplete")
        try:
            observed_asset = hash_bytes(Path(path_value).read_bytes())
        except OSError as exc:
            errors.append(f"worker selected input asset is unreadable: {exc}")
        if observed_asset is not None and observed_asset != expected_digest:
            errors.append("worker selected input asset digest changed")
    return observed_manifest, observed_asset, "; ".join(errors) or None


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


def _sidecar_adapter_sha256(request: WorkerRequest, adapter_path: Path) -> Optional[str]:
    """Recover an accepted adapter digest from the parent-owned artifact sidecar.

    Parameters
    ----------
    request:
        Worker request carrying the campaign/source identities.
    adapter_path:
        Typed adapter path selected by the driver.

    Returns
    -------
    str | None
        Exact accepted code digest, or ``None`` when no fully bound sidecar exists.
    """

    resolved_adapter = adapter_path.resolve()
    for directory in (resolved_adapter.parent, *resolved_adapter.parents[:4]):
        sidecar = directory / "driver-author-artifact.json"
        if not sidecar.is_file():
            continue
        try:
            artifact = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(artifact, Mapping):
            return None
        proposal = artifact.get("proposal")
        model_dir_value = artifact.get("model_dir")
        if not isinstance(proposal, Mapping) or not isinstance(model_dir_value, str):
            return None
        claimed_proposal_hash = proposal.get("proposal_sha256")
        proposal_payload = {
            key: value for key, value in proposal.items() if key != "proposal_sha256"
        }
        if claimed_proposal_hash != stable_hash(proposal_payload):
            return None
        if (
            proposal.get("stable_id") != request.stable_id
            or proposal.get("source_identity") != request.source_identity
            or proposal.get("recipe_revision") != request.recipe_revision
        ):
            return None
        facts = proposal.get("proposed_facts")
        implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(implementation, Mapping):
            return None
        code_value = implementation.get("code_path")
        code_digest = implementation.get("code_sha256")
        verified_hashes = proposal.get("verified_hashes")
        if not isinstance(code_value, str) or not isinstance(code_digest, str):
            return None
        model_dir = Path(model_dir_value).resolve()
        code_path = Path(code_value)
        resolved_code = (code_path if code_path.is_absolute() else model_dir / code_path).resolve()
        if resolved_code != resolved_adapter:
            return None
        if not isinstance(verified_hashes, Mapping) or verified_hashes.get("code") != code_digest:
            return None
        return code_digest
    return None


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


def _input_signature(args: tuple[object, ...], kwargs: Mapping[str, object]) -> dict[str, Any]:
    """Describe a complete typed dummy call without retaining payload values.

    Parameters
    ----------
    args, kwargs:
        Exact positional and keyword arguments returned by the typed contract.

    Returns
    -------
    dict[str, Any]
        Shape, dtype, device, and complete pytree location.
    """

    signature = output_signature({"args": args, "kwargs": dict(kwargs)}, path="input")
    values = _call_leaf_values(args, kwargs)
    for leaf in signature["leaves"]:
        path = str(leaf["path"]).removeprefix("input.")
        value = values.get(path)
        if leaf["kind"] == "python":
            leaf["value_sha256"] = stable_hash(value)
    return signature


_PATH_TOKEN = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_]*)|\[([0-9]+)\]")


def _path_tokens(path: str) -> tuple[Union[str, int], ...]:
    """Parse one closed args/kwargs path into mapping/list tokens.

    Parameters
    ----------
    path:
        Contract path such as ``args[0]`` or ``kwargs.mask``.

    Returns
    -------
    tuple[str | int, ...]
        Parsed tokens.
    """

    matches = list(_PATH_TOKEN.finditer(path))
    if not matches or "".join(match.group(0) for match in matches) != path:
        raise TypeError(f"invalid dummy-call contract path: {path!r}")
    tokens: list[Union[str, int]] = []
    for match in matches:
        name, index = match.groups()
        tokens.append(name if name is not None else int(index))
    if tokens[0] not in {"args", "kwargs"}:
        raise TypeError(f"dummy-call path must start with args or kwargs: {path!r}")
    return tuple(tokens)


def _assign_path(root: dict[str, Any], path: str, value: object) -> None:
    """Assign one materialized leaf into a closed dummy-call tree."""

    tokens = _path_tokens(path)
    current: Any = root
    for index, token in enumerate(tokens):
        final = index == len(tokens) - 1
        next_token = None if final else tokens[index + 1]
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(f"dummy-call path collides with a non-object: {path!r}")
            if final:
                if token in current:
                    raise TypeError(f"duplicate dummy-call path: {path!r}")
                current[token] = value
            else:
                expected: object = [] if isinstance(next_token, int) else {}
                current = current.setdefault(token, expected)
        else:
            if not isinstance(current, list):
                raise TypeError(f"dummy-call path collides with a non-list: {path!r}")
            while len(current) <= token:
                current.append(None)
            if final:
                if current[token] is not None:
                    raise TypeError(f"duplicate dummy-call path: {path!r}")
                current[token] = value
            else:
                expected = [] if isinstance(next_token, int) else {}
                if current[token] is None:
                    current[token] = expected
                current = current[token]


def _call_leaf_values(args: tuple[object, ...], kwargs: Mapping[str, object]) -> dict[str, object]:
    """Flatten an exact call to the same paths used by input signatures."""

    values: dict[str, object] = {}

    def visit(value: object, path: str) -> None:
        """Record scalar/tensor leaves recursively."""

        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        else:
            values[path] = value

    visit(args, "args")
    visit(dict(kwargs), "kwargs")
    return values


def _materialize_declarative_call(
    request: WorkerRequest,
) -> tuple[tuple[object, ...], dict[str, object], str, Optional[str], str]:
    """Materialize every typed R1 positional, keyword, and non-tensor leaf."""

    contract = request.input_contract
    if not isinstance(contract, Mapping):
        if request.input_spec is None:
            raise TypeError("declarative recipe has no dummy-call contract")
        materialized = materialize_standard_input(
            request.modality,
            request.input_spec,
            framework=request.framework,
            device=request.device,
            seed=request.input_seed,
        )
        return (
            materialized.args,
            materialized.kwargs,
            materialized.input_kind,
            materialized.input_asset,
            materialized.input_note,
        )
    root: dict[str, Any] = {"args": [], "kwargs": {}}
    input_kinds: list[str] = []
    input_assets: list[str] = []
    notes: list[str] = []
    tensor_index = 0
    for collection in ("args", "kwargs"):
        leaves = contract.get(collection)
        if not isinstance(leaves, list):
            raise TypeError(f"input_contract.{collection} must be a list")
        for leaf in leaves:
            if not isinstance(leaf, Mapping) or leaf.get("kind") != "tensor":
                raise TypeError(f"input_contract.{collection} contains a non-tensor leaf")
            materialized = materialize_standard_input(
                request.modality,
                {"shape": leaf.get("shape"), "dtype": leaf.get("dtype")},
                framework=request.framework,
                device=request.device,
                seed=request.input_seed + tensor_index,
            )
            _assign_path(root, str(leaf.get("path")), materialized.value)
            input_kinds.append(materialized.input_kind)
            if materialized.input_asset is not None:
                input_assets.append(materialized.input_asset)
            notes.append(materialized.input_note)
            tensor_index += 1
    non_tensor = contract.get("non_tensor_values")
    if not isinstance(non_tensor, list):
        raise TypeError("input_contract.non_tensor_values must be a list")
    for leaf in non_tensor:
        if not isinstance(leaf, Mapping):
            raise TypeError("input_contract.non_tensor_values contains a non-object")
        _assign_path(root, str(leaf.get("path")), leaf.get("value"))
    args_value = root["args"]
    kwargs_value = root["kwargs"]
    if not isinstance(args_value, list) or any(value is None for value in args_value):
        raise TypeError("input_contract args paths must be contiguous")
    if not isinstance(kwargs_value, dict):
        raise TypeError("input_contract kwargs paths are invalid")
    kind = input_kinds[0] if len(set(input_kinds)) == 1 else "random-fallback"
    unique_assets = set(input_assets)
    asset = (
        next(iter(unique_assets))
        if len(unique_assets) == 1 and len(input_assets) == tensor_index
        else None
    )
    return tuple(args_value), kwargs_value, kind, asset, "; ".join(notes)


def _materialize_dummy_call(
    loaded: LoadedRecipe, request: WorkerRequest
) -> tuple[tuple[object, ...], dict[str, object], str, Optional[str], str]:
    """Execute the typed input contract or build a declarative standard call.

    Parameters
    ----------
    loaded:
        Validated executable recipe.
    request:
        Complete worker request.

    Returns
    -------
    tuple
        Args, kwargs, input kind, optional asset, and provenance note.

    Raises
    ------
    TypeError
        If ``make_dummy_call`` violates its typed return contract.
    """

    if loaded.make_dummy_call is not None:
        value = loaded.make_dummy_call(request.input_seed, request.device)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("make_dummy_call must return exactly (args, kwargs)")
        args, kwargs = value
        if not isinstance(args, tuple):
            raise TypeError("make_dummy_call args must be a tuple")
        if not isinstance(kwargs, dict) or not all(isinstance(key, str) for key in kwargs):
            raise TypeError("make_dummy_call kwargs must be a string-keyed dict")
        return (
            args,
            kwargs,
            "standard-typed-dummy-call",
            None,
            "typed make_dummy_call(seed, device)",
        )

    return _materialize_declarative_call(request)


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
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    input_kind: str,
    input_asset: Optional[str],
    input_note: str,
    mode: RunMode,
    framework: str,
    constructor_seconds: float,
) -> tuple[dict[str, Any], Optional[object]]:
    """Run one explicit forward and produce an honest per-mode receipt.

    Parameters
    ----------
    model:
        Constructed native model.
    args, kwargs:
        Shared complete dummy call.
    input_kind, input_asset, input_note:
        Mechanically observed dummy-call provenance.
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
        "input_signature": _input_signature(args, kwargs),
        "input_kind": input_kind,
        "input_asset": input_asset,
        "input_note": input_note,
        "forward_started": False,
        "forward_completed": False,
        "forward_seconds": None,
        "output_signature": None,
        "output_value_sha256": None,
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
            output = forward(*args, **dict(kwargs))
        receipt["forward_seconds"] = time.monotonic() - started
        receipt["forward_completed"] = True
        receipt["output_signature"] = output_signature(output)
        receipt["output_value_sha256"] = output_value_sha256(output)
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
    recipe_path = request.recipe.get("path")
    effective_recipe = dict(request.recipe)
    if isinstance(recipe_path, str) and recipe_path:
        adapter_digest = effective_recipe.get("adapter_sha256")
        if not isinstance(adapter_digest, str):
            sidecar_digest = _sidecar_adapter_sha256(request, Path(recipe_path))
            if sidecar_digest is not None:
                effective_recipe["adapter_sha256"] = sidecar_digest
    allowed_read_paths = [ASSET_ROOT]
    if isinstance(recipe_path, str) and recipe_path:
        allowed_read_paths.append(Path(recipe_path))
    manifest_value = effective_recipe.get("code_manifest")
    if isinstance(manifest_value, list):
        allowed_read_paths.extend(
            Path(str(member["path"]))
            for member in manifest_value
            if isinstance(member, Mapping) and isinstance(member.get("path"), str)
        )
    policy = ExecutionPolicy(
        request.scratch_root,
        request.receipt_path.parent,
        allowed_read_paths=allowed_read_paths,
    )
    base: dict[str, Any] = {
        "receipt_version": "menagerie.crawler.worker-receipt.v1",
        "stable_id": request.stable_id,
        "source_identity": request.source_identity,
        "recipe_revision": request.recipe_revision,
        "observed_recipe_revision": None,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": None,
        "observed_input_asset_sha256": None,
        "execution_identity": request.execution_identity,
        "seed": request.seed,
        "input_seed": request.input_seed,
        "mode": request.mode.value if request.mode is not None else None,
        "device": request.device,
        "framework": request.framework,
        "awards_runs": False,
        "constructor_started": False,
        "constructor_completed": False,
        "input_completed": False,
        "per_mode": {},
        "declared_meaningful_modes": [],
        "detected_meaningful_modes": [],
        "meaningful_modes": [],
        "train_eval_divergence": None,
        "divergence_evidence": None,
        "error": None,
    }
    with policy:
        try:
            observed_manifest, observed_asset, byte_error = _observe_request_bytes(request)
            base["observed_code_manifest_sha256"] = observed_manifest
            base["observed_input_asset_sha256"] = observed_asset
            if byte_error is not None:
                raise RecipeError(byte_error)
            _seed_frameworks(request.seed, request.framework)
            recipe_kind = effective_recipe.get("kind") or effective_recipe.get("recipe_type")
            if recipe_kind == "typed-adapter" and not isinstance(
                effective_recipe.get("adapter_sha256"), str
            ):
                raise RecipeError("typed-adapter worker request requires adapter_sha256")
            observed_accepted_revision: Optional[str] = None
            if request.recipe_identity_payload is not None:
                observed_accepted_revision = compute_recipe_revision(
                    request.recipe_identity_payload, request.source_identity
                )
                if observed_accepted_revision != request.recipe_revision:
                    raise RecipeError(
                        "accepted worker recipe identity mismatch: "
                        f"expected {request.recipe_revision}, observed {observed_accepted_revision}"
                    )
            expected_revision = (
                request.recipe_revision if request.recipe_revision != "unbound" else None
            )
            try:
                loaded = load_recipe(
                    effective_recipe,
                    source_identity=request.source_identity,
                    expected_recipe_revision=expected_revision,
                )
            except RecipeError as exc:
                if observed_accepted_revision is None or "recipe revision mismatch" not in str(exc):
                    raise
                identity_payload = request.recipe_identity_payload
                if not isinstance(identity_payload, Mapping):
                    raise
                identity_implementation = identity_payload.get("implementation")
                identity_recipe = (
                    identity_implementation.get("library_recipe")
                    if isinstance(identity_implementation, Mapping)
                    else None
                )
                executed_recipe = effective_recipe.get("recipe")
                if recipe_kind == "declarative-library" and (
                    not isinstance(identity_recipe, Mapping)
                    or not isinstance(executed_recipe, Mapping)
                    or stable_hash(identity_recipe) != stable_hash(executed_recipe)
                ):
                    raise RecipeError(
                        "declarative recipe bytes do not match the accepted identity payload"
                    ) from exc
                loaded = load_recipe(
                    effective_recipe,
                    source_identity=request.source_identity,
                    expected_recipe_revision=None,
                )
            base["observed_recipe_revision"] = observed_accepted_revision or loaded.recipe_revision
            base["observed_adapter_sha256"] = loaded.adapter_sha256
            base["constructor_started"] = True
            constructor_started = time.monotonic()
            model = loaded.build_model()
            constructor_seconds = time.monotonic() - constructor_started
            base["constructor_completed"] = True
            args, kwargs, input_kind, input_asset, input_note = _materialize_dummy_call(
                loaded, request
            )
            selected_asset = (
                request.input_manifest.get("selected_asset")
                if isinstance(request.input_manifest, Mapping)
                else None
            )
            expected_asset_id = (
                selected_asset.get("asset_id") if isinstance(selected_asset, Mapping) else None
            )
            if input_asset != expected_asset_id:
                if input_asset is not None or expected_asset_id is None:
                    raise RecipeError(
                        "materialized input asset is outside the request-bound asset outcomes"
                    )
            base["observed_input_asset_sha256"] = (
                observed_asset if input_asset is not None else None
            )
            base["input_completed"] = True
            declared = request.meaningful_modes or ()
            detected = detect_meaningful_modes(model)
            mode_set = set(declared) | set(detected)
            detected_modes = tuple(
                mode for mode in (RunMode.TRAIN, RunMode.EVAL) if mode in mode_set
            )
            if request.mode is not None and request.mode not in mode_set:
                raise ValueError(
                    f"requested mode {request.mode.value!r} is not meaningful for this model"
                )
            modes = (request.mode,) if request.mode is not None else detected_modes
            base["declared_meaningful_modes"] = [mode.value for mode in declared]
            base["detected_meaningful_modes"] = [mode.value for mode in detected]
            base["meaningful_modes"] = [mode.value for mode in detected_modes]
            outputs: dict[str, object] = {}
            for mode in modes:
                mode_model = model
                mode_constructor_seconds = constructor_seconds
                if request.mode is None and len(modes) > 1:
                    _seed_frameworks(request.seed, request.framework)
                    mode_constructor_started = time.monotonic()
                    mode_model = loaded.build_model()
                    mode_constructor_seconds = time.monotonic() - mode_constructor_started
                receipt, output = _mode_receipt(
                    mode_model,
                    args,
                    kwargs,
                    input_kind,
                    input_asset,
                    input_note,
                    mode,
                    request.framework,
                    mode_constructor_seconds,
                )
                base["per_mode"][mode.value] = receipt
                if output is not None:
                    outputs[mode.value] = output
            if request.mode is None and {
                RunMode.TRAIN.value,
                RunMode.EVAL.value,
            }.issubset(outputs):
                divergence = classify_train_eval_divergence(outputs["train"], outputs["eval"])
                base["train_eval_divergence"] = divergence.classification
                base["divergence_evidence"] = divergence.evidence
            elif request.mode is None and len(outputs) == 1:
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

    completion_challenge = os.environ.pop(_PARENT_COMPLETION_CHALLENGE_ENV, None)
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
    success = receipt.get("error") is None and succeeded
    if success and completion_challenge:
        completion = {
            "receipt_sha256": receipt.get("receipt_sha256"),
            "proof": stable_hash(
                {
                    "version": "menagerie.crawler.worker-completion.v1",
                    "challenge": completion_challenge,
                    "receipt_sha256": receipt.get("receipt_sha256"),
                }
            ),
        }
        os.write(
            1, _WORKER_COMPLETION_PREFIX.encode("ascii") + canonical_json_bytes(completion) + b"\n"
        )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
