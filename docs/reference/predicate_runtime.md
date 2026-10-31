# Predicate runtime extension point (S4 contract)

Status: DOCUMENTED-UNSTABLE (provisional-name protocol) — every
public spelling on this page may rename without a deprecation shim until its
naming-session ratification. The SEMANTICS below are the frozen S4 seam
contract; consumers build against them now.

Module: `torchlens.ir.predicate_registry`. Public surface: exactly
`PredicateProtocol`, `coerce_predicate`, `register_predicate`. There is NO
public bare-name resolver: the only way to obtain a callable from the
registry is `coerce_predicate(name, slot=...)`, which returns the enforcing
wrapper.

The registry is INERT until consuming lanes adopt name acceptance in their
own merges: no shipped capture surface accepts a registered NAME today.
`@register_predicate` prepares a predicate for those consumers; it changes
no current behavior.

## The frozen callable signature

```python
class PredicateProtocol(Protocol):
    def __call__(self, ctx: RecordContext) -> PredicateDecision: ...
```

ONE positional `RecordContext` — the concrete class, not a generic subject —
no kwargs. The protocol covers the CAPTURE-LIFECYCLE `save`/`halt`/`until`
slots only.

## Per-slot acceptance table (normative)

| slot | subject | accepted returns (post-normalization) |
|---|---|---|
| `save=` | `RecordContext` | `CaptureDecision` (`True`/`False`/`None`/`CaptureSpec`); `RetroactiveCaptureDecision` RAW-CALLABLE-ONLY (see below) |
| run-time `save=` | `RecordContext` | SAME ROW as `save=` — run-time save maps onto `slot="save"` with identical normalization |
| `halt=` | `RecordContext` | `bool` ONLY; EXCLUSIVE stop — the record satisfying the predicate is NOT captured (shipped semantics) |
| `until=` | `RecordContext` | `bool` ONLY; INCLUSIVE stop — the record satisfying the predicate IS captured, then the run stops. This row pins subject + return domain + inclusivity; the consumer owns its dispatch |
| grad `save=` | `GradRecordContext` | `CaptureDecision` — NOT a `PredicateProtocol` member (different subject); documented per-slot |
| `intervene=` | live proxy (`Any`) | `InterventionPredicateDecision` — OUTSIDE `PredicateProtocol` and the registry; registered names are NOT accepted by `intervene=` |

The slot vocabulary `save | halt | until` is a CLOSED public vocabulary
(S2-routed). Unknown slots refuse with an ordinary typed `ValueError`; this is
also the enforcement point for the capture-lifecycle-only non-promise.

## Decision normalization table (public contract; shipped behavior)

| predicate return | meaning |
|---|---|
| `True` | full `CaptureSpec` (save out + metadata; slot defaults for the rest) |
| `False` | save nothing |
| `None` | slot default |
| `CaptureSpec` | passthrough |
| anything else | `PredicateError(code="predicate_return_invalid")` |

Halt and until slots: `bool` only (inclusivity difference per the table
above).

RETROACTIVE RETURNS: `RetroactiveCaptureDecision` passthrough remains
RAW-CALLABLE behavior exactly as shipped. A REGISTERED predicate resolved via
`coerce_predicate` returns through the slot wrapper, which refuses
`RetroactiveCaptureDecision` typed with the existing code
`predicate_return_invalid` plus `reason="registered_retroactive_unsupported"`
(the `followed_by` machinery that gives retroactive decisions meaning is
builtin-gated, so silent passthrough would be undefined behavior).

## The single coercion point

```text
coerce_predicate(value, *, slot: Literal["save", "halt", "until"])
```

VALUE DOMAIN, closed: exactly `{raw callable, registered-name str}`. A
non-callable non-str refuses with a house-style typed `ValueError`.

* RAW CALLABLES ARE RETURNED UNWRAPPED — identity, not equivalence. Zero
  behavior change is structural: all three capture-layer introspection
  points (`followed_by` support detection, `BaseSelector` alias-retry
  detection, predicate cache keys) see the user's original object.
* `BaseSelector` instances ARE raw callables and are ACCEPTED on the
  raw-callable branch, identity-returned, never wrapped. A selector that
  reaches `coerce_predicate` gets the DOCUMENTED outcome: shipped
  capture-predicate `keep_op` semantics for that same object — never silent
  misbehavior. Consumer dispatch of selector/label forms to their own
  selector paths happens BEFORE this door (semantic routing, consumer-owned).
  Coercion and registration deliberately DIFFER: coercion is identity
  (introspection survives), registration wraps (introspection would silently
  vanish) — which is why registration refuses, by structural introspection,
  the same shapes coercion passes through unchanged.
* Label-strings-as-selectors keep their shipped per-lifecycle selector
  meanings on the CONSUMERS' own selector paths, never inside
  `coerce_predicate`. A str reaching `coerce_predicate` is ALWAYS read as a
  registry name (miss -> `predicate_unregistered`).
* REGISTERED NAMES return the slot-aware WRAPPER, where the narrowed return
  domain is ENFORCED.

WRAPPER TRANSPARENCY CONTRACT: the wrapper carries
`__torchlens_cache_key__ = ("registered", name, version)` (read unmodified by
the shipped cache helper) and otherwise deliberately presents as a PLAIN
callable — it forwards no `.selector` and is not a `BaseSelector`. That is
safe by design: registration refuses introspection-bearing objects, so a
wrapped registered predicate has no retroactive path to lose, and a plain
callable's conservative alias-retry treatment is intended. Binding is
COERCE-TIME: the wrapper closes over the resolved `(callable, version)`;
`replace=True` invalidates caches because the NEXT coercion mints a wrapper
with the bumped key, while an already-bound wrapper keeps its consistent
(old callable, old key) pair for the run.

## Registration

```text
@register_predicate(name, *, replace=False)
```

* The decorator returns the function TRULY UNCHANGED — no attribute is
  written onto the user's object. The registry stores an internal
  `name -> (callable, version)` table. Total over every callable shape
  (plain function, bound method, `functools.partial`, `__slots__`/frozen
  callable instance); the same callable under two names yields two
  registrations with DISTINCT cache keys.
* REGISTRY STATE MACHINE: builtin names (empty set at S4) are NEVER
  replaceable — `replace=True` on a builtin refuses typed. User
  re-registration without `replace=True` refuses typed
  (`predicate_name_conflict`); with `replace=True` it replaces and bumps the
  version counter. Registration is setup-time; the lock guards the registry
  dict only.
* REGISTRABLE SHAPES: any plain callable satisfying `PredicateProtocol`.
  Objects the capture layer introspects structurally — `BaseSelector`
  instances, `followed_by` composites, anything carrying a `.selector`
  attribute — REFUSE at registration with an ordinary typed `ValueError`:
  their capture-time meaning depends on introspection the registry
  deliberately does not forward. Register the underlying plain predicate
  instead.
* NO LOADER MARKER: registration sets no attribute consulted by the
  restricted loader (`utils._callable_safety`, default-deny). A pickled or
  persisted reference to a registered predicate still refuses typed
  (`UntrustedCallableError` path unchanged). Registered predicates are never
  persistence-resolution targets.

## Four enforced caller-facing clauses (shipped behavior, now promised)

1. IDEMPOTENT UNDER ALIAS-RETRY: a predicate may run TWICE per event with
   `ctx.label` rewritten to the prefix-alias form. Predicates must be pure
   w.r.t. observable effects.
2. SHORT-CIRCUIT: `and`/`or` composition short-circuits per subject;
   invocation is NEVER guaranteed.
3. `ctx.label` IS THE INVOCATION-SPECIFIC IN-FLIGHT SPELLING: on first
   invocation the raw spelling (e.g. `"relu_1_2_raw"`); on a compatibility
   retry it may instead be the prefix alias (e.g. `"relu_1"`). Predicates
   needing stable raw identity use `ctx.raw_label` (may be `None` where a
   backend cannot supply one). Finalized public labels do not exist yet at
   capture time either way.
4. VALUE-DEPENDENT FIELDS MAY RAISE: MLX deferred-value sentinel fields raise
   `MLXValueUnavailableError` on consumption. Predicates touching
   `tensor_requires_grad` / `is_scalar_bool` / `bool_value` must tolerate
   typed refusal.

ERROR POLICY (inherited): user-predicate exceptions route through the
accumulate/fail-fast machinery (`max_predicate_failures` /
`on_predicate_error`). Registration does not change error routing.

## What S4 explicitly does NOT promise

* NO `Selection` / `ResolvedSelection` types, NO `__selection__` protocol, NO
  new operator dunders (separate merges).
* NO selector-kind extension: the `SelectorKind` closed Literal does not
  widen here, and there is NO kind-string / `TargetSpec` round-trip for
  registered predicates.
* NO non-torch backend promise (backends keep their static-kind gates).
* NO `RetroactiveCaptureDecision` support for registered predicates —
  ENFORCED typed at the coerce wrapper.
* NO `lifecycle="site"/"live"` evaluation of registered predicates —
  ENFORCED at the closed slot vocabulary; a registered callable manually
  handed to `where()` is just a raw callable, outside this contract.
* NO `RecordContext` field additions and no field-registration mechanism (the
  schema stays CLOSED; additions are ordinary coordinated PRs).
* NO persistence of registered callables (opaque callables stay audit-only);
  persist-by-name is a named FUTURE option, not promised.
* NO thread-safety beyond registration-time locking; registration is
  setup-time, not per-event.
