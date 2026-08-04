"""Pre-admission host-capacity refusal for models this machine cannot instantiate.

Some catalog models cannot be built on a small host at all. Full-config
Mixtral-8x7B declares roughly 46.5B parameters; at fp32 random init that is about
186 GB of parameter tensors alone, and on a 16 GiB machine the macOS OOM killer
takes the worker down after minutes of futile allocation -- after an agentic
author session and an environment solve have already been spent.

This module refuses those models EARLY, CHEAPLY, and HONESTLY:

* :func:`assess_model_capacity` derives a *lower bound* on the model's parameter
  count from the authored recipe's own declared configuration, without importing
  torch, without importing the model's library, and without allocating anything.
* The bound is compared against a ceiling derived from the host's real physical
  memory, read at run time. On a bigger machine the same catalog admits the same
  models with no code change.
* A refusal is recorded as a durable, append-only row carrying the estimate AND
  the threshold that produced it, so a future reviewer can tell whether the call
  was right.

Three properties are deliberate:

**It is a refusal, not a relaxation.** Nothing here weakens a validation check.
A deferred model simply never reaches the environment lane, so no attempt, gate,
or record is asserted about it.

**It is biased toward attempting.** The agentic authoring stage runs ONCE per
model, so a model wrongly refused becomes a permanent dead record, while a model
wrongly admitted costs one OOM'd run. Every lever is therefore set the generous
way: the parameter estimate is a strict LOWER bound (it omits biases, norms,
positional tables, and the third gated feed-forward projection), and the host
ceiling is deliberately set ABOVE what the machine can really hold
(:data:`OVERCOMMIT_ALLOWANCE`). Only a model whose own configuration puts it
beyond that generous ceiling is refused.

**It is honest about coverage.** The estimate is derivable only when the recipe
declares a recognizable layer-stack configuration. When it is not derivable the
verdict is :data:`CapacityVerdict.NOT_DERIVABLE` and the model is ADMITTED, and
the row-level basis says so rather than pretending to completeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import fcntl
import json
import os
from pathlib import Path
import subprocess  # noqa: S404 -- fixed-argv sysctl probe, no shell, no user input
import sys
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    fsync_directory,
    stable_hash,
)
from menagerie.crawler.models import JsonObject

CAPACITY_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.host-capacity-deferral.v1"

CAPACITY_DEFERRAL_DISPOSITION = "deferred:needs-more-memory"
"""Greppable disposition for a model refused because this host is too small.

It follows the established ``deferred:needs-<capability>`` shape
(``needs-cuda``, ``needs-x86``, ``needs-opus-tier``, ``needs-source-access``):
a real, located model that this campaign cannot execute, naming the exact
capability that would recover it. Host memory is such a capability.

It is deliberately NOT ``failed:*``. A ``failed:`` code asserts that our pipeline
broke on a model; nothing broke here. The model is fine, the proposal is fine,
and the machine is simply too small -- a fact about the HOST, recoverable by
running the identical campaign somewhere bigger.

It is also deliberately not minted as a canonical model terminal. A canonical
``deferred:*`` record must carry a checker-adjudicated terminal gate over frozen
sources (see ``authority._derive_deferral`` and
``reducer._derive_blocked_capability_deferral_proof``); this refusal is
machine-derived and has no such adjudication, so claiming that terminal would
mean fabricating a proof. The model instead stays UNCOMPLETED for this campaign
and is recorded here, which is the truthful statement: not attempted, not
failed, recoverable on bigger hardware.
"""

CAPACITY_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "host-capacity-deferrals.jsonl"

BYTES_PER_PARAMETER = 4
"""Worst-case parameter width for a random-init forward-only capture.

The crawler instantiates every model from a constructor at random init, on CPU,
in the default dtype. That is fp32: four bytes per parameter, no sharding, no
quantization, and no pretrained checkpoint to stream. Autograd state and
optimizer state are not involved -- the capture is a single forward pass -- so
parameter bytes, not training footprint, is the right basis.
"""

OVERCOMMIT_ALLOWANCE = 2.0
"""Multiple of physical memory a model's parameters may nominally claim.

This is the margin, and it is set generously ON PURPOSE. A host does not die the
instant a process's parameter bytes reach physical RAM: macOS compresses and
swaps, and the enforced metric the OOM killer watches
(``max(ri_resident_size, ri_phys_footprint)``, see ``worker_supervisor``) climbs
well past ``hw.memsize`` before a signal 9 arrives. The practical instantiation
ceiling measured on the 16 GiB host is nearer 4B parameters; this allowance puts
the REFUSAL threshold at roughly 8.6B parameters there, more than double the
practical ceiling.

So the band between "probably will not fit" and "is refused without trying" is
wide, and everything in it is attempted. That is the intended asymmetry: an OOM
costs one wasted run, a false refusal costs a catalog entry forever.
"""

HOST_MEMORY_ENV_VAR = "MENAGERIE_HOST_MEMORY_BYTES"
"""Operator override for the host physical-memory figure, in bytes.

Present so the check is testable and so an operator can model a larger machine
before moving a campaign to it. Raising it admits strictly more models (the
biased-toward-attempting direction); lowering it defers strictly more. It cannot
weaken any validation check either way, because this gate only decides whether a
model is attempted at all.
"""

_HIDDEN_SIZE_KEYS = ("hidden_size", "d_model", "n_embd", "embed_dim", "model_dim")
_DEPTH_KEYS = (
    "num_hidden_layers",
    "num_layers",
    "n_layer",
    "n_layers",
    "num_encoder_layers",
    "depth",
)
_INTERMEDIATE_KEYS = ("intermediate_size", "ffn_dim", "d_ff", "n_inner", "mlp_dim")
_VOCAB_KEYS = ("vocab_size", "n_vocab")
_HEADS_KEYS = ("num_attention_heads", "n_head", "num_heads")
_KV_HEADS_KEYS = ("num_key_value_heads", "num_kv_heads")
_EXPERT_KEYS = ("num_local_experts", "num_experts", "moe_num_experts", "n_routed_experts")


class CapacityVerdict(StrEnum):
    """Closed pre-admission capacity outcomes."""

    ADMIT = "admit"
    """The model fits the host ceiling, or its size could not be bounded."""

    DEFER = "defer"
    """The model's own declared configuration exceeds the host ceiling."""

    NOT_DERIVABLE = "not-derivable"
    """No parameter bound could be derived; the model is admitted regardless."""


class CapacityDeferralError(ValueError):
    """Raised when a host-capacity deferral row is malformed or cannot persist."""


@dataclass(frozen=True)
class ParameterEstimate:
    """A strict lower bound on one model's instantiated parameter count."""

    parameter_count_lower_bound: int
    basis: str
    config_path: str
    terms: JsonObject

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this estimate."""

        return {
            "parameter_count_lower_bound": self.parameter_count_lower_bound,
            "basis": self.basis,
            "config_path": self.config_path,
            "terms": dict(self.terms),
            "bytes_per_parameter": BYTES_PER_PARAMETER,
            "estimated_parameter_bytes": (
                self.parameter_count_lower_bound * BYTES_PER_PARAMETER
            ),
        }


@dataclass(frozen=True)
class HostCapacity:
    """The host memory figure and the parameter ceiling derived from it."""

    physical_memory_bytes: int
    memory_source: str

    @property
    def admissible_parameter_bytes(self) -> int:
        """Return the generous parameter-byte budget this host will attempt."""

        return int(self.physical_memory_bytes * OVERCOMMIT_ALLOWANCE)

    @property
    def admissible_parameter_ceiling(self) -> int:
        """Return the parameter count above which a model is refused outright."""

        return self.admissible_parameter_bytes // BYTES_PER_PARAMETER

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this threshold."""

        return {
            "host_physical_memory_bytes": self.physical_memory_bytes,
            "host_memory_source": self.memory_source,
            "overcommit_allowance": OVERCOMMIT_ALLOWANCE,
            "admissible_parameter_bytes": self.admissible_parameter_bytes,
            "admissible_parameter_ceiling": self.admissible_parameter_ceiling,
        }


@dataclass(frozen=True)
class CapacityAssessment:
    """One complete pre-admission capacity decision with its full justification."""

    verdict: CapacityVerdict
    host: HostCapacity
    estimate: Optional[ParameterEstimate]
    explanation: str

    @property
    def deferred(self) -> bool:
        """Return whether this assessment refuses the model on this host."""

        return self.verdict is CapacityVerdict.DEFER

    def to_json(self) -> JsonObject:
        """Return the immutable JSON projection of this decision."""

        payload: JsonObject = {
            "verdict": self.verdict.value,
            "threshold": self.host.to_json(),
            "explanation": self.explanation,
            "estimate": self.estimate.to_json() if self.estimate is not None else None,
        }
        if self.estimate is not None and self.host.admissible_parameter_ceiling > 0:
            payload["overage_ratio"] = round(
                self.estimate.parameter_count_lower_bound
                / self.host.admissible_parameter_ceiling,
                4,
            )
        return payload


def host_capacity(*, environ: Optional[Mapping[str, str]] = None) -> HostCapacity:
    """Read this host's real physical memory and derive its parameter ceiling.

    The figure is TOTAL physical memory rather than momentarily-free memory. Free
    memory fluctuates with whatever else the machine is doing, and a fluctuating
    threshold would make the same model refuse on one run and admit on the next
    -- unacceptable when the authoring stage runs once per model. Total physical
    memory is stable, machine-derived, and moves automatically when the campaign
    is re-run on bigger hardware, which is the whole point.

    Parameters
    ----------
    environ:
        Environment mapping to consult for the operator override. Defaults to
        the live process environment.

    Returns
    -------
    HostCapacity
        Host memory figure, its provenance, and the derived ceiling.

    Raises
    ------
    CapacityDeferralError
        If the override is present but not a positive integer.
    """

    source_environ = os.environ if environ is None else environ
    override = source_environ.get(HOST_MEMORY_ENV_VAR)
    if override is not None:
        try:
            override_bytes = int(override)
        except ValueError as exc:
            raise CapacityDeferralError(
                f"{HOST_MEMORY_ENV_VAR} must be an integer byte count, got {override!r}"
            ) from exc
        if override_bytes <= 0:
            raise CapacityDeferralError(f"{HOST_MEMORY_ENV_VAR} must be positive")
        return HostCapacity(
            physical_memory_bytes=override_bytes,
            memory_source=f"env:{HOST_MEMORY_ENV_VAR}",
        )
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        pages = -1
        page_size = -1
    if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
        return HostCapacity(
            physical_memory_bytes=pages * page_size,
            memory_source="sysconf:SC_PHYS_PAGES*SC_PAGE_SIZE",
        )
    if sys.platform == "darwin":
        try:
            observed = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            memsize = int(observed)
        except (OSError, subprocess.SubprocessError, ValueError):
            memsize = 0
        if memsize > 0:
            return HostCapacity(
                physical_memory_bytes=memsize,
                memory_source="sysctl:hw.memsize",
            )
    raise CapacityDeferralError(
        "host physical memory is unreadable; set "
        f"{HOST_MEMORY_ENV_VAR} to declare it explicitly"
    )


def _positive_int(candidate: Any) -> Optional[int]:
    """Return a strictly positive integer value, or ``None`` for anything else."""

    if isinstance(candidate, bool) or not isinstance(candidate, int):
        return None
    return candidate if candidate > 0 else None


def _first_key(config: Mapping[str, Any], keys: Sequence[str]) -> tuple[Optional[str], Optional[int]]:
    """Return the first declared positive integer among ``keys``."""

    for key in keys:
        value = _positive_int(config.get(key))
        if value is not None:
            return key, value
    return None, None


def _candidate_configs(node: Any, path: str) -> list[tuple[str, Mapping[str, Any]]]:
    """Collect every nested keyword mapping in a recipe, with its dotted path.

    Recipes nest constructor payloads (``{"__construct__": {"kwargs": {...}}}``)
    to arbitrary depth, and different rungs put the configuration in different
    places. Rather than hardcode one shape, every mapping in the recipe is a
    candidate and the caller picks the one that actually declares a layer stack.

    Parameters
    ----------
    node:
        Recipe fragment being walked.
    path:
        Dotted path of ``node`` within the recipe, for provenance.

    Returns
    -------
    list[tuple[str, Mapping[str, Any]]]
        Every nested mapping, outermost first, paired with its dotted path.
    """

    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(node, Mapping):
        found.append((path, node))
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_candidate_configs(value, child))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found.extend(_candidate_configs(value, f"{path}[{index}]"))
    return found


def estimate_parameter_count(recipe: Any) -> Optional[ParameterEstimate]:
    """Derive a strict lower bound on a recipe's instantiated parameter count.

    The bound is assembled from the configuration the AUTHOR already declared --
    no import, no construction, no allocation. Every term is deliberately
    understated so the result can only ever be smaller than the real model:

    * Embeddings count ``vocab_size * hidden`` once, doubled only when the config
      explicitly says the output head is untied.
    * Attention counts the four projections at their declared head geometry, and
      falls back to square projections when head counts are absent.
    * The feed-forward block counts TWO projections per expert, never three, even
      though gated architectures (Mixtral, Llama, and every SwiGLU descendant)
      have three. For Mixtral this alone understates the model by about a third.
    * Biases, normalization scales, positional tables, and every auxiliary head
      are omitted entirely.

    Parameters
    ----------
    recipe:
        Authored recipe fragment, typically
        ``proposal["proposed_facts"]["implementation"]["library_recipe"]``.

    Returns
    -------
    ParameterEstimate | None
        Lower bound and its provenance, or ``None`` when the recipe declares no
        recognizable layer stack. ``None`` means "unknown", never "small".
    """

    for path, config in _candidate_configs(recipe, ""):
        hidden_key, hidden = _first_key(config, _HIDDEN_SIZE_KEYS)
        depth_key, depth = _first_key(config, _DEPTH_KEYS)
        if hidden is None or depth is None:
            continue
        terms: JsonObject = {
            "hidden_size": {"key": hidden_key, "value": hidden},
            "depth": {"key": depth_key, "value": depth},
        }

        embedding = 0
        vocab_key, vocab = _first_key(config, _VOCAB_KEYS)
        if vocab is not None:
            # Untied output heads double the embedding cost. `tie_word_embeddings`
            # defaults to True across the config families this recognizes, so an
            # absent flag counts the table ONCE -- the smaller, safer reading.
            untied = config.get("tie_word_embeddings") is False
            embedding = vocab * hidden * (2 if untied else 1)
            terms["embedding"] = {
                "key": vocab_key,
                "vocab_size": vocab,
                "untied_output_head": untied,
                "parameters": embedding,
            }

        _heads_key, heads = _first_key(config, _HEADS_KEYS)
        _kv_key, kv_heads = _first_key(config, _KV_HEADS_KEYS)
        head_dim = _positive_int(config.get("head_dim"))
        if head_dim is None and heads is not None and hidden % heads == 0:
            head_dim = hidden // heads
        if head_dim is not None and heads is not None:
            kv_dim = (kv_heads if kv_heads is not None else heads) * head_dim
            attention = hidden * (heads * head_dim) + (heads * head_dim) * hidden + 2 * (
                hidden * kv_dim
            )
        else:
            # No head geometry declared: four square projections is the standard
            # shape and remains a lower bound for grouped-query variants.
            attention = 4 * hidden * hidden
        terms["attention_per_layer"] = {
            "heads": heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "parameters": attention,
        }

        intermediate_key, intermediate = _first_key(config, _INTERMEDIATE_KEYS)
        if intermediate is None:
            # Transformer feed-forward blocks are conventionally 4x the model
            # width; using that ratio keeps a stack without a declared
            # intermediate size in scope while staying a lower bound for the
            # wider designs that omit the field.
            intermediate = 4 * hidden
        _expert_key, experts = _first_key(config, _EXPERT_KEYS)
        expert_count = experts if experts is not None else 1
        feedforward = expert_count * 2 * hidden * intermediate
        terms["feedforward_per_layer"] = {
            "key": intermediate_key,
            "intermediate_size": intermediate,
            "intermediate_declared": intermediate_key is not None,
            "experts": expert_count,
            "projections_counted": 2,
            "parameters": feedforward,
        }

        total = embedding + depth * (attention + feedforward)
        terms["per_layer_parameters"] = attention + feedforward
        return ParameterEstimate(
            parameter_count_lower_bound=total,
            basis="declared-config-layer-stack-lower-bound",
            config_path=path or "<recipe>",
            terms=terms,
        )
    return None


def assess_model_capacity(
    recipe: Any,
    *,
    host: Optional[HostCapacity] = None,
) -> CapacityAssessment:
    """Decide whether this host should attempt to instantiate one recipe.

    Parameters
    ----------
    recipe:
        Authored recipe fragment carrying the model's declared configuration.
    host:
        Host capacity to compare against. Read from the machine when omitted.

    Returns
    -------
    CapacityAssessment
        Verdict, the host threshold, the estimate, and a plain-language reason.
    """

    resolved_host = host_capacity() if host is None else host
    estimate = estimate_parameter_count(recipe)
    if estimate is None:
        return CapacityAssessment(
            verdict=CapacityVerdict.NOT_DERIVABLE,
            host=resolved_host,
            estimate=None,
            explanation=(
                "the recipe declares no recognizable layer stack, so no parameter "
                "bound could be derived; the model is admitted and will be attempted"
            ),
        )
    ceiling = resolved_host.admissible_parameter_ceiling
    if estimate.parameter_count_lower_bound > ceiling:
        return CapacityAssessment(
            verdict=CapacityVerdict.DEFER,
            host=resolved_host,
            estimate=estimate,
            explanation=(
                f"declared configuration bounds the model below by "
                f"{estimate.parameter_count_lower_bound:,} parameters "
                f"({estimate.parameter_count_lower_bound * BYTES_PER_PARAMETER:,} bytes at "
                f"fp32 random init), above this host's ceiling of {ceiling:,} parameters "
                f"({resolved_host.physical_memory_bytes:,} bytes of physical memory "
                f"times an allowance of {OVERCOMMIT_ALLOWANCE})"
            ),
        )
    return CapacityAssessment(
        verdict=CapacityVerdict.ADMIT,
        host=resolved_host,
        estimate=estimate,
        explanation=(
            f"declared configuration bounds the model below by "
            f"{estimate.parameter_count_lower_bound:,} parameters, within this host's "
            f"ceiling of {ceiling:,}"
        ),
    )


def capacity_deferral_path(records_root: Path) -> Path:
    """Return the canonical host-capacity deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only host-capacity deferral JSONL path.
    """

    return records_root / CAPACITY_DEFERRAL_RELATIVE_PATH


def build_capacity_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    assessment: CapacityAssessment,
) -> JsonObject:
    """Assemble one complete, self-describing host-capacity deferral row.

    The row states BOTH the estimate and the threshold that produced it, so a
    later reviewer can re-derive the decision and judge whether it was right,
    without needing the code that made it.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the refusal.
    created_at:
        UTC timestamp of the decision.
    assessment:
        The refusing capacity decision.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    CapacityDeferralError
        If the assessment does not actually refuse the model.
    """

    if not assessment.deferred:
        raise CapacityDeferralError(
            "only a refusing capacity assessment may be recorded as a deferral"
        )
    assert assessment.estimate is not None  # guaranteed by CapacityVerdict.DEFER
    required_bytes = assessment.estimate.parameter_count_lower_bound * BYTES_PER_PARAMETER
    row: JsonObject = {
        "schema_version": CAPACITY_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": CAPACITY_DEFERRAL_DISPOSITION,
        "capacity": assessment.to_json(),
        "recheck_hint": (
            "re-run this campaign on a host with at least "
            f"{required_bytes / OVERCOMMIT_ALLOWANCE / 2**30:.1f} GiB of physical memory, "
            "or set "
            f"{HOST_MEMORY_ENV_VAR} to that machine's byte count"
        ),
    }
    row["row_sha256"] = stable_hash(row)
    return row


def validate_capacity_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one host-capacity deferral row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate deferral row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    CapacityDeferralError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "capacity",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise CapacityDeferralError(
            f"host-capacity deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != CAPACITY_DEFERRAL_SCHEMA_VERSION:
        raise CapacityDeferralError(
            f"host-capacity deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != CAPACITY_DEFERRAL_DISPOSITION:
        raise CapacityDeferralError(
            f"host-capacity deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise CapacityDeferralError(f"host-capacity deferral row has an empty {field}")
    capacity = row["capacity"]
    if not isinstance(capacity, Mapping) or capacity.get("verdict") != CapacityVerdict.DEFER.value:
        raise CapacityDeferralError("host-capacity deferral row does not record a refusal")
    estimate = capacity.get("estimate")
    threshold = capacity.get("threshold")
    if not isinstance(estimate, Mapping) or not isinstance(threshold, Mapping):
        raise CapacityDeferralError(
            "host-capacity deferral row must state both its estimate and its threshold"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    digest = validated.pop("row_sha256")
    if stable_hash(validated) != digest:
        raise CapacityDeferralError("host-capacity deferral row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def append_capacity_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one host-capacity deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination host-capacity deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CapacityDeferralError
        If the same work generation already carries a different deferral.
    """

    validated = validate_capacity_deferral_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = _decode_capacity_lines(handle.read(), path)
        matching = [
            item for item in existing if (item["stable_id"], item["work_id"]) == key
        ]
        if matching:
            # A repeated refusal of the same work generation is expected on every
            # resume: the host has not grown, so the identical row is re-derived.
            # Only a CONFLICTING row for the same generation is an error.
            if len(matching) != 1 or matching[0] != validated:
                raise CapacityDeferralError(
                    f"conflicting host-capacity deferral for {key[0]} at {key[1]}"
                )
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _decode_capacity_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CapacityDeferralError(
                f"host-capacity deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise CapacityDeferralError(
                f"host-capacity deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_capacity_deferral_row(decoded))
    return tuple(rows)


def load_capacity_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load host-capacity deferrals from one or more campaign roots.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Rows ordered by stable ID then work ID.
    """

    by_key: dict[tuple[str, str], JsonObject] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in _decode_capacity_lines(path.read_bytes(), path):
            by_key[(row["stable_id"], row["work_id"])] = row
    return tuple(by_key[key] for key in sorted(by_key))
