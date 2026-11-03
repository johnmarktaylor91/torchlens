"""Frozen wire version discriminators shared by runnable writers and readers."""

from typing import Final

__tl_layer__ = "L1"

RUNNABLE_TLSPEC_SCHEMA_VERSION: Final = "sparse_recorded_taken_path_v2"
"""Frozen sparse runnable descriptor capability/version string.

v2 requires EXPLICIT execution-context records (per-call ``execution_context``
plus the capture-scoped ``ambient_context``); the parser REJECTS their absence,
so an absent context can only ever mean a legacy v1 artifact, which is
analysis-only (fail-closed), never "assume disabled".
"""

RUNNABLE_CALL_RECIPE_VERSION: Final = "non_tensor_args_tensor_slots_context_and_obligations_v3"
"""Frozen sparse call-recipe version string.

v3 adds REQUIRED per-call obligation fields to every call record:
``control_obligations`` (owner-record scalar-bool/loop-predicate obligations) and
``control_dependencies`` (owner-record conditional arm-entry edges on the CHILD
call). Absence of either field is a parse failure -- the explicit-schema-decision
doctrine: obligation fields are never optional or defaulted.
"""

RUNNABLE_CALLABLE_REF_SCHEMA_VERSION: Final = 1
"""Frozen ``FunctionRegistryKey`` schema version accepted by rung 1."""

RUNNABLE_INITIALIZER_POLICY_VERSION: Final = "torchlens_role_init_v2"
"""Frozen canonical role-based random-initializer policy version.

v2 semantics are the v1 role table plus degenerate totality: a legal
``numel() == 0`` slot allocates and returns without sampling or consuming any
generator state, and every nonempty Kaiming slot requires finite positive
``fan_in`` (producer preflight and the runtime validator share the check).
"""

RUNNABLE_ACTIVATION_PAYLOAD_SCHEMA_VERSION: Final = "selected_activation_v2"
"""Frozen selected-activation payload schema (v2: physical input fingerprints)."""

LEGACY_RUNNABLE_TLSPEC_SCHEMA_VERSIONS: Final[frozenset[str]] = frozenset(
    {"sparse_recorded_taken_path_v1"}
)
"""Known legacy capability strings: analysis-only load, typed run refusal."""

WITNESS_FAMILY_REGISTRY_VERSION: Final = "witness_family_registry_v2"
"""Closed registry/version discriminator persisted in every required-witness inventory.

A parser only accepts an inventory authored against ITS OWN closed registry; any other
discriminator refuses at parse (``context_field_invalid``), so a future registry change
is an explicit schema decision, never a silent vocabulary drift. v2 extends the
registry from the seven ``SHAPE_STRUCTURE_FACT`` prefixes to EVERY verdict-steering
witness family -- the four direct control kinds included -- plus the two explicit
claim-only families, each row a typed :class:`FamilySpec` naming its independent
replay-structural anchor. A v1 inventory refuses at parse (analysis-only load); there
is no legacy fallback and no silent compatibility default.

``WITNESS_FAMILY_REGISTRY`` itself is defined after :class:`ControlWitnessKind` below
(the typed rows reference the kind enum).
"""
