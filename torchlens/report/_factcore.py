"""FactCore: the ONE numbers substrate (C02; sumfam item 7, megaplan D6).

Single-source law (sumfam D1): no family surface computes a headline fact
(count, total, coverage figure, label) from raw trace attributes --
everything reads FactCore. IdentityIndex owns identities and join rules
(D2); the counts record names its grains explicitly (D3: the bare word
"operations" cannot denote both 69 and 151); every fact carries evidence,
reason, and coverage; payload figures are tagged ``at_capture`` vs
``retained_now`` (D8); the capture fingerprint lets envelope composition
refuse mismatched captures.

Everything here is a READ-TIME derivation over persisted facts (live-only,
declared in sprint/field_intent.tsv); spellings DOCUMENTED-UNSTABLE
pending naming-session ratification. Renderers are projections; A07 ->
C02 -> F08/F09/F10 is the fence chain.
"""

from __future__ import annotations

import hashlib
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ._compute_truth import ComputeAggregation, compute_aggregation

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

#: FactCore schema version (bump on any section/field change).
FACTCORE_SCHEMA_VERSION = 1

#: The grain menu (D3): one vocabulary used identically by every level=
#: argument. op = pass-qualified executed operations; layer = recurrence-
#: grouped layers; site = structural L1 sites; module = declared modules;
#: call = module call instances.
GRAINS: tuple[str, ...] = ("op", "layer", "site", "module", "call")

#: Payload-scope vocabulary (D8): an immutable capture fact vs what THIS
#: object holds right now.
PAYLOAD_SCOPES: tuple[str, ...] = ("at_capture", "retained_now")


@dataclass(frozen=True)
class CountsRecord:
    """The explicit counts vocabulary (D3).

    ``compute_ops`` counts real executed compute operations (the identity
    partition's op rows: alias/boundary/buffer rows excluded);
    ``tracked_tensor_rows`` counts EVERY layer-list row including alias and
    boundary pseudo-rows -- the two numbers a bare "operations" used to
    conflate.
    """

    compute_ops: int
    tracked_tensor_rows: int
    alias_rows: int
    layers: int
    sites: int
    modules: int
    module_calls: int

    def for_grain(self, grain: str) -> int:
        """Return the count for one grain-menu entry, refusing unknowns."""

        mapping = {
            "op": self.compute_ops,
            "layer": self.layers,
            "site": self.sites,
            "module": self.modules,
            "call": self.module_calls,
        }
        if grain not in mapping:
            raise InvalidArgumentError(
                f"unknown grain {grain!r}; the grain menu is {GRAINS}.",
                code="factcore_grain_invalid",
                remedy=f"pass one of {', '.join(GRAINS)}",
            )
        return mapping[grain]


@dataclass(frozen=True)
class IdentityIndex:
    """Shared identities and join rules (D2): joins refuse ambiguity.

    One identity spine, typed projections -- never one physical row table.
    ``op_labels`` are pass-qualified and execution-ordered; ``op_sites``
    aligns with them (``None`` where no site key was derivable).
    """

    op_labels: tuple[str, ...]
    layer_labels: tuple[str, ...]
    op_layers: tuple[str, ...]
    op_sites: tuple[str | None, ...]
    module_addresses: tuple[str, ...]
    module_call_labels: tuple[str, ...]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): the spine points, never dumps."""

        return (
            f"IdentityIndex({len(self.op_labels)} ops, "
            f"{len(self.layer_labels)} layers, "
            f"{len(self.module_addresses)} modules; read .op_labels ...)"
        )

    def ops_of_layer(self, layer_label: str) -> tuple[str, ...]:
        """All pass-qualified op labels of one layer (never a silent pick)."""

        found = tuple(
            op_label
            for op_label, owner in zip(self.op_labels, self.op_layers, strict=True)
            if owner == layer_label
        )
        if not found:
            raise InvalidArgumentError(
                f"unknown layer label {layer_label!r}.",
                code="factcore_identity_unknown",
                remedy="use a label from identity.layer_labels",
            )
        return found

    def layer_of_op(self, op_label: str) -> str:
        """The owning layer of one pass-qualified op label."""

        try:
            return self.op_layers[self.op_labels.index(op_label)]
        except ValueError:
            raise InvalidArgumentError(
                f"unknown op label {op_label!r}.",
                code="factcore_identity_unknown",
                remedy="use a pass-qualified label from identity.op_labels",
            ) from None

    def ops_of_site(self, site_key: str) -> tuple[str, ...]:
        """All op labels minted at one structural site."""

        found = tuple(
            op_label
            for op_label, site in zip(self.op_labels, self.op_sites, strict=True)
            if site == site_key
        )
        if not found:
            raise InvalidArgumentError(
                f"unknown site key {site_key!r}.",
                code="factcore_identity_unknown",
                remedy="use a key from identity.op_sites",
            )
        return found


@dataclass(frozen=True)
class ParamFacts:
    """Parameter truth (A07's identity partition, read as facts).

    ``total`` is torch's own dedup identity rule; ``per_path_total`` is the
    remove_duplicate=False sum DISCLOSED BESIDE it (torchinfo's number),
    never printed under a "unique" label.
    """

    total: int | None
    per_path_total: int | None
    executed: int | None
    unexecuted: int | None
    trainable: int | None
    frozen: int | None
    tied_groups: tuple[tuple[str, ...], ...]
    evidence: str = "measured"


@dataclass(frozen=True)
class MemoryFacts:
    """Payload byte figures under the D8 scope law.

    ``at_capture_bytes`` is the immutable capture fact (what the capture
    retained when it ran); ``retained_now_bytes`` is what THIS object holds
    (present or lazily materializable payloads) -- zero on a
    payload-stripped artifact even though the capture once held 64 MB.
    """

    at_capture_bytes: int
    at_capture_saved_ops: int
    retained_now_bytes: int
    retained_now_present_ops: int
    retained_now_lazy_ops: int
    scope_note: str = "at_capture is a capture fact; retained_now describes THIS object"


@dataclass(frozen=True)
class FactCore:
    """The one numbers substrate for a finished trace.

    Sections: counts (D3 grains), identity (D2 joins), params (A07 truth),
    compute (the costreport-D1 aggregation face), memory (D8 scopes).
    Health facts ride ``trace.health_facts`` (sumfam item 8) and are
    deliberately NOT duplicated here -- one observation store.
    """

    schema_version: int
    capture_fingerprint: str
    counts: CountsRecord
    identity: IdentityIndex
    params: ParamFacts
    compute: ComputeAggregation
    memory: MemoryFacts

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): sections point, never dump.

        The generated repr chained per-op compute rows -- 9.4k chars on a
        ten-layer toy and unbounded on real models.
        """

        return (
            f"FactCore(v{self.schema_version} {self.capture_fingerprint}, "
            f"sections={'/'.join(self.section_ids)}; read .counts .params "
            f".compute .memory .identity)"
        )

    @property
    def section_ids(self) -> tuple[str, ...]:
        """Stable section IDs (row IDs live on each section's rows)."""

        return ("counts", "identity", "params", "compute", "memory")


def capture_fingerprint(trace: Any) -> str:
    """Stable digest of one capture's identity (sumfam item 7).

    Survives save/load of the SAME capture (derived from persisted facts
    only); distinguishes different captures so envelope composition can
    refuse a mismatched pairing (composition row 11). Not a byte hash of
    payloads -- a cheap identity fingerprint.
    """

    ops = list(getattr(trace, "layer_list", ()) or ())
    label_digest = hashlib.sha256()
    for op in ops:
        label_digest.update(str(getattr(op, "layer_label", "?")).encode())
        label_digest.update(b"|")
    parts = (
        str(getattr(trace, "backend", None)),
        str(getattr(trace, "model_class_name", None)),
        str(len(ops)),
        str(getattr(trace, "num_params", None)),
        label_digest.hexdigest()[:16],
    )
    return "fc1-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _counts(trace: Any) -> CountsRecord:
    """Derive the counts record from the identity partition."""

    ops = list(getattr(trace, "layer_list", ()) or ())
    alias_rows = sum(
        1
        for op in ops
        if getattr(op, "is_input", False)
        or getattr(op, "is_output", False)
        or getattr(op, "is_buffer", False)
    )
    compute_ops = len(ops) - alias_rows
    site_keys = {site for op in ops if (site := getattr(op, "site_key", None)) is not None}
    return CountsRecord(
        compute_ops=compute_ops,
        tracked_tensor_rows=len(ops),
        alias_rows=alias_rows,
        layers=len(getattr(trace, "layer_labels", ()) or ()),
        sites=len(site_keys),
        modules=len(getattr(trace, "modules", {}) or {}),
        module_calls=len(getattr(trace, "module_calls", {}) or {}),
    )


def _identity(trace: Any) -> IdentityIndex:
    """Derive the identity spine."""

    ops = list(getattr(trace, "layer_list", ()) or ())
    op_labels = tuple(str(getattr(op, "label", getattr(op, "layer_label", "?"))) for op in ops)
    op_layers = tuple(str(getattr(op, "layer_label", "?")) for op in ops)
    op_sites = tuple(getattr(op, "site_key", None) for op in ops)
    modules = getattr(trace, "modules", {}) or {}
    module_calls = getattr(trace, "module_calls", {}) or {}
    return IdentityIndex(
        op_labels=op_labels,
        layer_labels=tuple(getattr(trace, "layer_labels", ()) or ()),
        op_layers=op_layers,
        op_sites=op_sites,
        module_addresses=tuple(str(key) for key in modules),
        module_call_labels=tuple(str(key) for key in module_calls),
    )


def _params(trace: Any) -> ParamFacts:
    """Read A07's parameter truth as facts."""

    def read(name: str) -> int | None:
        """Read one optional integer trace attribute."""

        value = getattr(trace, name, None)
        return None if value is None else int(value)

    tied = getattr(trace, "tied_param_groups", ()) or ()
    return ParamFacts(
        total=read("num_params"),
        per_path_total=read("num_params_by_path"),
        executed=read("num_params_executed"),
        unexecuted=read("num_params_unexecuted"),
        trainable=read("num_params_trainable"),
        frozen=read("num_params_frozen"),
        tied_groups=tuple(tuple(str(name) for name in group) for group in tied),
    )


def _memory(trace: Any) -> MemoryFacts:
    """Derive the D8 payload-scope figures."""

    from ._agent_json import _payload_state

    at_capture_bytes = 0
    at_capture_saved = 0
    retained_bytes = 0
    present = 0
    lazy = 0
    for op in getattr(trace, "layer_list", ()) or ():
        if not getattr(op, "has_saved_activation", False):
            continue
        memory = getattr(op, "activation_memory", None)
        size = 0 if memory is None else int(memory)
        at_capture_saved += 1
        at_capture_bytes += size
        state = _payload_state(op)
        if state == "present":
            present += 1
            retained_bytes += size
        elif state == "lazy":
            lazy += 1
            retained_bytes += size
    return MemoryFacts(
        at_capture_bytes=at_capture_bytes,
        at_capture_saved_ops=at_capture_saved,
        retained_now_bytes=retained_bytes,
        retained_now_present_ops=present,
        retained_now_lazy_ops=lazy,
    )


#: Session cache: FactCore is a pure derivation over a FINISHED trace;
#: id-keyed with weakref eviction (same pattern as the TensorStats cache).
_FACTCORE_CACHE: dict[int, tuple[weakref.ref, FactCore]] = {}


def factcore(trace: Trace) -> FactCore:
    """Build (or serve cached) the FactCore for one finished trace."""

    cache_id = id(trace)
    cached = _FACTCORE_CACHE.get(cache_id)
    if cached is not None and cached[0]() is trace:
        return cached[1]
    core = FactCore(
        schema_version=FACTCORE_SCHEMA_VERSION,
        capture_fingerprint=capture_fingerprint(trace),
        counts=_counts(trace),
        identity=_identity(trace),
        params=_params(trace),
        compute=compute_aggregation(trace),
        memory=_memory(trace),
    )

    def _evict(_reference: weakref.ref, _id: int = cache_id) -> None:
        """Drop the cache row when the trace is collected."""

        _FACTCORE_CACHE.pop(_id, None)

    try:
        reference = weakref.ref(trace, _evict)
    except TypeError:
        return core
    _FACTCORE_CACHE[cache_id] = (reference, core)
    return core
