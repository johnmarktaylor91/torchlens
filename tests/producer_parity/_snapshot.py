"""Three-layer capture snapshots for the parity harness.

A snapshot canonicalizes one capture into JSON-able structures across the
three comparison layers the design-of-record names:

* ``journal`` — the op-event stream exactly as postprocess step 0 consumes it
  (intercepted at ``materialize_from_events`` entry, the same seam the capture
  oracle uses), canonicalized per field path.
* ``store`` — the finished trace's per-op semantic cells in ``_OP_SLOT_NAMES``
  layout order, with ``_MISSING`` for absent cells, never ``repr``.
* ``artifact`` — a ``.tlspec`` save: canonicalized manifest plus per-blob
  SHA256 digests.

Volatile identity tokens are NOT compared by value: each occurrence becomes a
``TokenSite`` row (bucket, layer, anchor, path, token). Non-token volatile
values (timing, memory, RNG state) are canonicalized per named path to
presence + type. Every canonicalizer entry is a reviewed row in this module,
seeded empirically by the P0 legacy-vs-legacy control (a control diff is
either a genuine nondeterminism to canonicalize BY NAME here, or a bug). The
two-check cross-leg comparator and its attestation shims died with the P7
deletion (transient campaign tooling); the snapshot machinery survives for
the remaining harness consumers.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

_MISSING = "__MISSING__"

# ---------------------------------------------------------------------------
# Token buckets (journal paths use [*]-normalized specs).
# ---------------------------------------------------------------------------

JOURNAL_TOKEN_PATHS: dict[str, str] = {
    # path spec -> bucket
    "grad_fn_handle": "grad_fn_object_id",
    "output.tensor.backend_handle_id": "backend_handle_id",
    "output.transformed_tensor.backend_handle_id": "backend_handle_id",
    # dedup annotations reference the SOURCE tensor by live object id: same
    # id-space as backend_handle_id, so it joins that bucket's partition.
    "transform_config._tl_annotations.dedup_source_id": "backend_handle_id",
    "params[*].barcode": "tl_barcode",
    "equivalence_class": "tl_barcode_derived",
}

# Journal paths canonicalized to presence + type (never value-compared).
JOURNAL_VOLATILE_PATHS: frozenset[str] = frozenset(
    {
        "function.func_duration",
        "function.func_rng_states",
        "function.func_autocast_state",
        "backend_semantics.autograd_memory",
        "backend_semantics.num_autograd_tensors",
        "backend_semantics.bytes_delta_at_call",
        "backend_semantics.bytes_peak_at_call",
        "backend_semantics.backend_grad_handle",
        "function.func",
        "function.code_context[*]",
        "source_trace",
        "record_context",
        "capture_spec",
        "transform_fn_source",
        "output.activation_transform",
        "fire_results[*].fire_record",
        "policy.stream",
    }
)

# Store cells: per-field canonicalization rules. Unlisted fields compare
# strictly through the generic canonicalizer.
STORE_FIELD_RULES: dict[str, str] = {
    # transient caches / runtime handles: excluded from comparison
    "_facets_cache": "skip",
    "_receptive_field_cache": "skip",
    "_projective_field_cache": "skip",
    "_arg_expressions_cache": "skip",
    "_source_trace_ref": "skip",
    "grad_fn": "skip",
    "func": "opaque",
    "activation_transform": "opaque",
    "func_rng_states": "volatile",
    "func_autocast_state": "volatile",
    "func_duration": "volatile",
    "bytes_delta_at_call": "volatile",
    "bytes_peak_at_call": "volatile",
    "autograd_memory": "volatile",
    "num_autograd_tensors": "volatile",
    # identity tokens
    "grad_fn_object_id": "token:grad_fn_object_id",
    "grad_fn_handle": "token_obj:grad_fn_object_id",
    "_param_barcodes": "token_list:tl_barcode",
    "equivalence_class": "token:tl_barcode_derived",
    "parent_param_ops": "token_keyed_dict:tl_barcode",
    "param_shapes": "maybe_token_keyed_dict:tl_barcode",
    "equivalent_ops": "strict",
    "recurrent_ops": "strict",
    # tensors
    "out": "tensor",
    "transformed_out": "tensor",
    "grad": "tensor",
    "transformed_grad": "tensor",
    "saved_args": "generic",
    "saved_kwargs": "generic",
    "args_template": "generic",
    "kwargs_template": "generic",
    "out_versions_by_child": "generic",
    # pending blob ids are per-save minted identifiers
    "_pending_blob_id": "volatile",
    "_pending_transformed_out_blob_id": "volatile",
    "_pending_grad_blob_id": "volatile",
    "_pending_transformed_grad_blob_id": "volatile",
    "_grad_records": "generic",
}

_INDEX_RE = re.compile(r"\[\d+\]")
_HEX_ID_RE = re.compile(r"0x[0-9a-fA-F]{6,}")


def _normalize_path(path: str) -> str:
    """Normalize concrete list indices to the [*] spec form."""

    return _INDEX_RE.sub("[*]", path)


@dataclass(frozen=True)
class TokenSite:
    """One anchored occurrence of a volatile identity token."""

    bucket: str
    layer: str  # journal | store | artifact
    anchor: str  # label_raw for op rows; path-derived for artifacts
    path: str
    token: str


@dataclass
class Snapshot:
    """Canonicalized three-layer view of one capture run."""

    scenario: str
    journal: dict[str, dict[str, Any]] = field(default_factory=dict)
    store: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact: dict[str, Any] = field(default_factory=dict)
    token_sites: list[TokenSite] = field(default_factory=list)
    # backend_handle_id coherence rows: (layer, anchor, path, ok, detail)
    coherence: list[tuple[str, str, str, bool, str]] = field(default_factory=list)
    # anchors with no retained payload for the coherence bucket (named residual)
    presence_only: list[tuple[str, str, str]] = field(default_factory=list)


class _Canonicalizer:
    """Stateful canonicalization for one layer of one snapshot."""

    def __init__(self, snapshot: Snapshot, layer: str) -> None:
        self.snapshot = snapshot
        self.layer = layer
        self.anchor = ""

    def token(self, bucket: str, path: str, token: str) -> dict[str, str]:
        self.snapshot.token_sites.append(
            TokenSite(bucket=bucket, layer=self.layer, anchor=self.anchor, path=path, token=token)
        )
        return {"__token__": bucket}

    def canon(self, value: Any, path: str) -> Any:
        spec = _normalize_path(path)
        segment = spec.rsplit(".", 1)[-1].split("[", 1)[0]
        if segment in _ARTIFACT_TIMING_KEYS:
            return _presence_marker(value)
        if segment in _TOKEN_SEGMENTS and value is not None:
            return self._canon_token_path(value, path, _TOKEN_SEGMENTS[segment])
        if segment in _TOKEN_LIST_SEGMENTS:
            bucket = _TOKEN_LIST_SEGMENTS[segment]
            if isinstance(value, (list, tuple)):
                return [
                    self._canon_token_path(item, f"{path}[{index}]", bucket)
                    for index, item in enumerate(value)
                ]
            if path.endswith("]") and value is not None:
                # a single element reached through an outer list recursion
                return self._canon_token_path(value, path, bucket)
        if self.layer == "journal":
            bucket = JOURNAL_TOKEN_PATHS.get(spec)
            if bucket is not None:
                return self._canon_token_path(value, path, bucket)
            if spec in JOURNAL_VOLATILE_PATHS:
                return _presence_marker(value)
        return self._canon_generic(value, path)

    def _canon_token_path(self, value: Any, path: str, bucket: str) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return self.token(bucket, path, value)
        if isinstance(value, int):
            return self.token(bucket, path, str(value))
        # object-valued token (grad_fn handle): the token is its id()
        return self.token(bucket, path, str(id(value)))

    def _canon_generic(self, value: Any, path: str) -> Any:
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            # exact float equality is intended: same-leg determinism is the claim
            return value
        if isinstance(value, str):
            if _HEX_ID_RE.search(value):
                # id-embedded name: token bucket with the embedded id extracted
                embedded = _HEX_ID_RE.findall(value)
                self.snapshot.token_sites.append(
                    TokenSite(
                        bucket="id_embedded_names",
                        layer=self.layer,
                        anchor=self.anchor,
                        path=path,
                        token="|".join(embedded),
                    )
                )
                return {"__id_embedded__": _HEX_ID_RE.sub("0xID", value)}
            return value
        if isinstance(value, torch.Tensor):
            return _tensor_fingerprint(value)
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        import enum

        if isinstance(value, enum.Enum):
            return f"{type(value).__name__}.{value.name}"
        if is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: self.canon(getattr(value, f.name), f"{path}.{f.name}" if path else f.name)
                for f in dataclass_fields(value)
            }
        if isinstance(value, dict):
            out = {}
            for key in sorted(value, key=_stable_key):
                rendered = _stable_key(key)
                if rendered in _ARTIFACT_TIMING_KEYS:
                    out[rendered] = _presence_marker(value[key])
                else:
                    out[rendered] = self.canon(value[key], f"{path}.{rendered}")
            return out
        if isinstance(value, (list, tuple)):
            return [self.canon(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            items = [self.canon(item, f"{path}[.]") for item in value]
            return {"__set__": sorted(items, key=json.dumps)}
        if callable(value) or hasattr(value, "__dict__") or hasattr(value, "__slots__"):
            return {"__opaque__": type(value).__qualname__}
        return {"__opaque__": type(value).__qualname__}


def _stable_key(key: Any) -> str:
    """Render a dict key deterministically without embedding object ids."""

    if isinstance(key, str):
        return key
    if isinstance(key, (int, bool, float)):
        return repr(key)
    if isinstance(key, tuple):
        return "(" + ",".join(_stable_key(item) for item in key) + ")"
    return f"<{type(key).__qualname__}>"


def _presence_marker(value: Any) -> dict[str, Any]:
    """Canonicalize a volatile value to presence + type."""

    return {"__volatile__": type(value).__qualname__, "present": value is not None}


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    """Shape/dtype/content digest for a tensor (never object identity)."""

    detached = tensor.detach().cpu().contiguous()
    try:
        data = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    except Exception:
        data = detached.numpy().tobytes()
    return {
        "__tensor__": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Journal layer
# ---------------------------------------------------------------------------


def journal_snapshot(snapshot: Snapshot, events: Any) -> None:
    """Canonicalize the op lane as consumed by step 0.

    Decomposed ``OpRecord`` journals project through the inverse oracle
    adapter (with the side-index grad-fn handle rebound), so both legs
    snapshot in the SAME flat shape and stay cross-comparable.
    """

    from ._oracle_adapter import op_event_from_record

    handles = getattr(events, "grad_fn_handles_by_label_raw", {}) or {}
    canonizer = _Canonicalizer(snapshot, "journal")
    for entry in events.op_events:
        event = op_event_from_record(entry, grad_fn_handle=handles.get(entry.label_raw))
        canonizer.anchor = event.label_raw
        row: dict[str, Any] = {}
        for f in dataclass_fields(event):
            row[f.name] = canonizer.canon(getattr(event, f.name), f.name)
        snapshot.journal[event.label_raw] = row


# ---------------------------------------------------------------------------
# Store layer
# ---------------------------------------------------------------------------


def store_snapshot(snapshot: Snapshot, trace: Any) -> None:
    """Canonicalize per-op store cells in layout order."""

    from torchlens._data_classes.op import _OP_SLOT_NAMES

    canonizer = _Canonicalizer(snapshot, "store")
    for op in trace.ops:
        anchor = getattr(op, "raw_label", None) or getattr(op, "_label_raw", None)
        if anchor is None:
            continue
        canonizer.anchor = anchor
        row: dict[str, Any] = {}
        for name in _OP_SLOT_NAMES:
            rule = STORE_FIELD_RULES.get(name, "strict")
            if rule == "skip":
                continue
            try:
                value = getattr(op, name, _MISSING)
            except Exception as error:
                # A typed read refusal (unsaved payload, mutated reference) is
                # itself a comparable fact: same capture -> same refusal.
                row[name] = {"__unreadable__": type(error).__qualname__}
                continue
            if value is _MISSING:
                row[name] = _MISSING
                continue
            row[name] = _apply_store_rule(canonizer, name, rule, value)
        snapshot.store[anchor] = row


def _apply_store_rule(canonizer: _Canonicalizer, name: str, rule: str, value: Any) -> Any:
    if rule == "volatile":
        return _presence_marker(value)
    if rule == "opaque":
        return None if value is None else {"__opaque__": type(value).__qualname__}
    if rule == "tensor":
        return None if value is None else canonizer._canon_generic(value, name)
    if rule.startswith("token:"):
        bucket = rule.split(":", 1)[1]
        return None if value is None else canonizer.token(bucket, name, str(value))
    if rule.startswith("token_obj:"):
        bucket = rule.split(":", 1)[1]
        return None if value is None else canonizer.token(bucket, name, str(id(value)))
    if rule.startswith("token_list:"):
        bucket = rule.split(":", 1)[1]
        return [canonizer.token(bucket, f"{name}[{i}]", str(v)) for i, v in enumerate(value or ())]
    if rule.startswith("token_keyed_dict:"):
        # Insertion order is the params-encounter order and is leg-stable;
        # sorting by the (random) token would scramble site identity.
        bucket = rule.split(":", 1)[1]
        result = {}
        for index, (key, entry) in enumerate((value or {}).items()):
            marker = canonizer.token(bucket, f"{name}.entry{index}", str(key))
            result[f"entry{index}"] = [marker, canonizer.canon(entry, f"{name}.val{index}")]
        return result
    if rule.startswith("maybe_token_keyed_dict:"):
        bucket = rule.split(":", 1)[1]
        if isinstance(value, dict) and value and all(isinstance(k, str) for k in value):
            return _apply_store_rule(canonizer, name, f"token_keyed_dict:{bucket}", value)
        return canonizer.canon(value, name)
    return canonizer.canon(value, name)


# ---------------------------------------------------------------------------
# Artifact layer
# ---------------------------------------------------------------------------

# Manifest/metadata keys canonicalized to presence (named, reviewed):
# created_at* (wall clock), save_duration (timing), rng_state_digests (RNG
# state is presence+type per the DoR), random_seed (trace mints a random seed
# when the user passes none), model_object_id / input_object_id (live ids).
ARTIFACT_VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "created_at",
        "created_at_utc",
        "save_duration",
        "rng_state_digests",
        "git_commit_hash",  # artifact provenance: moves every commit by design
        "random_seed",
        "model_object_id",
        "input_object_id",
        # per-phase wall-clock totals inside _phase_timings rows
        "total_s",
    }
)

# Manifest keys whose values are barcode-derived equivalence classes.
ARTIFACT_TOKEN_KEYS: dict[str, str] = {"op_kind": "tl_barcode_derived"}

# Metadata keys whose dict KEYS are volatile tokens.
ARTIFACT_TOKEN_KEYED_DICTS: dict[str, str] = {
    "op_equivalence_classes": "tl_barcode_derived",
    "grad_fn_logs": "grad_fn_object_id",
}

# Field/key segments that hold a single token value wherever they appear.
_TOKEN_SEGMENTS: dict[str, str] = {
    "dedup_source_id": "backend_handle_id",
    "object_id": "object_id_generic",
    "creator_object_id": "object_id_generic",
    "grad_fn_object_id": "grad_fn_object_id",
}

# Field/key segments holding a LIST of tokens.
_TOKEN_LIST_SEGMENTS: dict[str, str] = {
    "root_grad_fn_ids": "grad_fn_object_id",
    "backward_root_grad_fn_object_ids": "grad_fn_object_id",
    "next_grad_fn_ids": "grad_fn_object_id",
    "grad_fn_order": "grad_fn_object_id",
    "topology": "object_id_generic",
}

# Runtime measurements by key name, canonicalized to presence + type on every
# layer (design-of-record: timing/memory are never portable facts).
_ARTIFACT_TIMING_KEYS = frozenset(
    {
        "capture_start_time",
        "capture_end_time",
        "cleanup_duration",
        "forward_duration",
        "func_calls_duration",
        "forward_peak_memory",
        "postprocess_duration",
        "backward_duration",
        "backward_durations",
        "setup_duration",
        "timestamp",
        "duration",
        "_time_started",
        "_time_finished",
        "peak_memory",
        "backward_peak_memory",
    }
)


def artifact_snapshot(snapshot: Snapshot, trace: Any, tmp_path: Path) -> None:
    """Save a .tlspec and canonicalize manifest + blob digests."""

    target = tmp_path / "parity_artifact.tlspec"
    trace.save(str(target))
    canonizer = _Canonicalizer(snapshot, "artifact")
    canonizer.anchor = "<artifact>"
    result: dict[str, Any] = {}
    base = Path(target)
    for file_path in sorted(base.rglob("*")):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(base))
        if file_path.name.endswith(".json"):
            payload = json.loads(file_path.read_text())
            result[rel] = _canon_manifest(canonizer, payload, rel)
        elif file_path.name.endswith(".pkl"):
            # Structured comparison: a byte digest of a pickle embeds live
            # object ids and the minted seed; load and canonicalize instead.
            import pickle

            payload = pickle.loads(file_path.read_bytes())
            result[rel] = {"__pickle__": _canon_manifest(canonizer, payload, rel)}
        else:
            result[rel] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    snapshot.artifact = result


def _canon_manifest(canonizer: _Canonicalizer, value: Any, path: str) -> Any:
    if isinstance(value, dict):
        out = {}
        for key in sorted(value, key=_stable_key):
            rendered = _stable_key(key)
            if rendered in ARTIFACT_VOLATILE_KEYS or rendered in _ARTIFACT_TIMING_KEYS:
                out[rendered] = _presence_marker(value[key])
            elif rendered == "name" and isinstance(value[key], str):
                # per-process model-instance counter suffix; the semantic stem
                # still compares strictly
                out[rendered] = re.sub(r"_\d+$", "_<n>", value[key])
            elif rendered in ARTIFACT_TOKEN_KEYS and isinstance(value[key], str):
                out[rendered] = canonizer.token(
                    ARTIFACT_TOKEN_KEYS[rendered], f"{path}.{rendered}", value[key]
                )
            elif rendered in ARTIFACT_TOKEN_KEYED_DICTS and isinstance(value[key], dict):
                bucket = ARTIFACT_TOKEN_KEYED_DICTS[rendered]
                inner = {}
                for index, (token_key, entry) in enumerate(value[key].items()):
                    marker = canonizer.token(
                        bucket, f"{path}.{rendered}.entry{index}", str(token_key)
                    )
                    inner[f"entry{index}"] = [
                        marker,
                        _canon_manifest(canonizer, entry, f"{path}.{rendered}.val{index}"),
                    ]
                out[rendered] = inner
            else:
                out[rendered] = _canon_manifest(canonizer, value[key], f"{path}.{rendered}")
        return out
    if isinstance(value, list):
        return [_canon_manifest(canonizer, item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, str):
        return canonizer.canon(value, path)
    return canonizer.canon(value, path)


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _journal_interception(
    snapshot: Snapshot,
    journal_mutator: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Snapshot the op lane at materialize entry (the step-0 seam).

    ``journal_mutator`` (planted-proof hook) runs against the REAL journal
    after commit and before the snapshot — the faithful "journal mutated
    post-commit" plant the a1 proof requires.
    """

    postprocess_module = importlib.import_module("torchlens._postprocess")
    materialize_module = importlib.import_module("torchlens._postprocess._materialize")
    original_public = postprocess_module.materialize_from_events
    original_direct = materialize_module.materialize_from_events

    def observing(trace: Any, events: Any) -> None:
        if not snapshot.journal:
            if journal_mutator is not None:
                journal_mutator(events)
            journal_snapshot(snapshot, events)
        original_direct(trace, events)

    postprocess_module.materialize_from_events = observing
    materialize_module.materialize_from_events = observing
    try:
        yield
    finally:
        postprocess_module.materialize_from_events = original_public
        materialize_module.materialize_from_events = original_direct


@dataclass
class RunResult:
    """Snapshot plus live objects for one scenario run."""

    snapshot: Snapshot
    model: Any
    trace: Any


def run_scenario(
    scenario: Any,
    tmp_path: Path,
    *,
    with_artifact: bool = True,
    post_capture: Callable[[Any], None] | None = None,
    journal_mutator: Callable[[Any], None] | None = None,
) -> RunResult:
    """Run one scenario and produce its three-layer snapshot."""

    snapshot = Snapshot(scenario=scenario.name)
    model, inputs = scenario.build()

    with _journal_interception(snapshot, journal_mutator):
        captured = scenario.capture(model, inputs)
        if scenario.kind == "record":
            captured = captured.to_trace()
    if post_capture is not None:
        post_capture(captured)

    store_snapshot(snapshot, captured)
    if with_artifact:
        try:
            artifact_snapshot(snapshot, captured, tmp_path)
        except Exception as error:  # artifact layer optional per scenario
            snapshot.artifact = {"__unsupported__": type(error).__qualname__}
    return RunResult(snapshot=snapshot, model=model, trace=captured)
