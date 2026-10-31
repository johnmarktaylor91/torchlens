"""God-object oracle: aliases-v1 identity/mutation matrix + artifact fixtures.

Companion to ``tests/surface_oracle`` (surface-v1 byte-identity goldens).
This package freezes the OBSERVABLE IDENTITY AND MUTATION-EFFECT contract of
the public record objects — the behaviors that value snapshots cannot prove:
repeated-lookup identity, copy-on-read alias barriers, mutable-container
observability, ``Op._copy_with_shallow_fields()`` selective depth, fork isolation, the direct-write
warning/dirty-state transition, weakref lifetime, and payload identity.

Any behavior change caught here during the columnar re-plumbing is a bug in
the migration (docs/reference/trace_core_design.md, LOCKED tripwire).
"""
