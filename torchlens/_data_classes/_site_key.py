"""Structural site keys (``site_key_v1``): the portable bridging relation.

A site key is a process-stable STRUCTURAL-POSITION identity for every
retained op, minted from raw (pre-grouping) records inside the one neutral
grouper. It is policy-independent (identical bytes whether the capture
folded, recurrence-grouped, or ran ``recurrence_detection=False``), carries
no parameter barcodes, object ids, execution-global counters, or argument
hashes, and therefore survives save/load and process boundaries. Two
captures of the same program agree on site keys even when their layer
labels disagree -- the key proves POSITION, never source identity (joins
are guarded and verdict-tiered in :mod:`._site_join`).

Serialized form (the ``"s1"`` prefix makes any future re-keying a visible
schema event)::

    "s1|" + "/".join(esc(entry) for entry in module_site)
          + "|" + esc(layer_type)
          + "|" + ("" if output_slot is None else str(output_slot))
          + "|" + str(call_ordinal)

* ``module_site`` -- the op's module ADDRESS stack with pass suffixes
  stripped, outer-to-inner. One canonical normalizer (:func:`site_axis`)
  is defined on BOTH live representations -- build-time ``(address, pass)``
  pairs (``Op.modules``) and serialized ``"address:pass"`` strings
  (``Op.module_call_stack``) -- with pinned byte parity between the two.
* ``esc`` -- percent-escape of the UTF-8 bytes of ``%``, ``|``, ``/`` and
  control characters (< U+0020). Both separators are in the escape set, so
  the top-level ``split("|")`` and the site-level ``split("/")`` are exact
  inverses. Dots need no escaping: torch forbids ``.`` inside a single
  module NAME, so dots in an entry are exactly the address hierarchy.
* ``output_slot`` -- ``multi_output_index``; rendered ``""`` when ``None``
  (a present slot is all digits, so ``""`` is unambiguous).
* ``call_ordinal`` -- 1-based occurrence ordinal of this ``(module_site,
  layer_type, output_slot)`` tuple WITHIN one innermost module CALL
  instance (pass-qualified stack entry), in execution order.

The serialization is pinned byte-for-byte against the reference encoder of
the L1 design memo (``sitekey_ref.py``); any deviation is a schema change,
never a refactor.
"""

from __future__ import annotations

from typing import Any

#: Version prefix of the current site-key schema.
SITE_KEY_PREFIX = "s1"

#: Characters percent-escaped inside key components (both separators, so the
#: two-level split is an exact inverse; ``%`` so escaping self-delimits).
_ESCAPE_CHARS = frozenset({"%", "|", "/"})

#: Call-instance identity of ops running directly in the root forward.
ROOT_CALL_INSTANCE = "<root>"


def escape_site_component(component: str) -> str:
    """Percent-escape one key component (UTF-8 bytes of ``%|/`` + controls)."""

    out: list[str] = []
    for ch in component:
        if ch in _ESCAPE_CHARS or ord(ch) < 0x20:
            out.append("".join(f"%{b:02X}" for b in ch.encode("utf-8")))
        else:
            out.append(ch)
    return "".join(out)


def unescape_site_component(component: str) -> str:
    """Exact inverse of :func:`escape_site_component`."""

    out = bytearray()
    index = 0
    while index < len(component):
        if component[index] == "%":
            out.extend(bytes([int(component[index + 1 : index + 3], 16)]))
            index += 3
        else:
            out.extend(component[index].encode("utf-8"))
            index += 1
    return out.decode("utf-8")


def _entry_address(entry: Any) -> str:
    """Return the pass-suffix-free address of one module-stack entry.

    Accepts BOTH live representations: a serialized ``"address:pass"``
    string (rsplit-once on ``":"`` -- correct even for names containing
    ``":"`` because the pass suffix is always the last appended segment) and
    a build-time ``(address, pass)`` pair.
    """

    if isinstance(entry, str):
        return entry.rsplit(":", 1)[0]
    return str(entry[0])


def _entry_call_instance(entry: Any) -> str:
    """Return the pass-QUALIFIED identity of one module-stack entry.

    The serialized string IS the identity; a build-time pair renders to the
    identical ``"address:pass"`` spelling, so the two representations mint
    identical ordinals (rsplit-once recovers the pair uniquely).
    """

    if isinstance(entry, str):
        return entry
    return f"{entry[0]}:{entry[1]}"


def site_axis(module_entries: Any) -> tuple[str, ...]:
    """Canonical site-axis normalizer: address tuple, pass indexes stripped.

    Defined on both live representations (build-time pairs and serialized
    strings) so build-time minting and load-side recompute share ONE
    definition; byte parity between the two is a pinned test obligation.
    """

    return tuple(_entry_address(entry) for entry in module_entries or ())


def call_instance_id(module_entries: Any) -> str:
    """Return the pass-qualified innermost call-instance identity.

    Ops running directly in the root forward share the single
    :data:`ROOT_CALL_INSTANCE` pseudo-instance.
    """

    entries = tuple(module_entries or ())
    if not entries:
        return ROOT_CALL_INSTANCE
    return _entry_call_instance(entries[-1])


def render_site_key(
    module_site: tuple[str, ...],
    layer_type: str,
    output_slot: int | None,
    call_ordinal: int,
) -> str:
    """Render one ``site_key_v1`` string (reference-encoder byte parity)."""

    return (
        SITE_KEY_PREFIX
        + "|"
        + "/".join(escape_site_component(entry) for entry in module_site)
        + "|"
        + escape_site_component(layer_type)
        + "|"
        + ("" if output_slot is None else str(output_slot))
        + "|"
        + str(call_ordinal)
    )


def parse_site_key(key: str) -> tuple[tuple[str, ...], str, int | None, int]:
    """Parse one rendered key back to ``(site, type, slot, ordinal)``.

    Raises
    ------
    ValueError
        If the key is not a well-formed ``site_key_v1`` string.
    """

    parts = key.split("|")
    if len(parts) != 5 or parts[0] != SITE_KEY_PREFIX:
        raise ValueError(f"not a site_key_v1 string: {key!r}")
    prefix, site, layer_type, slot, ordinal = parts
    module_site = tuple(unescape_site_component(e) for e in site.split("/")) if site else ()
    if slot and not slot.isdigit():
        raise ValueError(f"malformed output slot in site key: {key!r}")
    if not ordinal.isdigit() or int(ordinal) < 1:
        raise ValueError(f"malformed call ordinal in site key: {key!r}")
    return (
        module_site,
        unescape_site_component(layer_type),
        None if slot == "" else int(slot),
        int(ordinal),
    )


class SiteKeyMinter:
    """Per-capture ordinal counter minting ``site_key_v1`` strings.

    One minter per capture, fed retained ops in EXECUTION ORDER: the
    ordinal is the 1-based occurrence count of ``(module_site, layer_type,
    output_slot)`` within one pass-qualified innermost call instance.
    Pruned/orphan ops never reach a minter (they consume no ordinals and
    carry no keys -- the SF-63 ruling).
    """

    __slots__ = ("_seen",)

    def __init__(self) -> None:
        self._seen: dict[tuple[str, tuple[str, ...], str, int | None], int] = {}

    def mint(
        self,
        module_entries: Any,
        layer_type: str,
        output_slot: int | None,
    ) -> str:
        """Mint the site key for the next retained op at this position."""

        return self.mint_at(
            site_axis(module_entries),
            call_instance_id(module_entries),
            layer_type,
            output_slot,
        )

    def mint_at(
        self,
        module_site: tuple[str, ...],
        call_instance: str,
        layer_type: str,
        output_slot: int | None,
    ) -> str:
        """Mint from a pre-normalized site axis (backend dialects, e.g. JAX).

        ``module_site`` must already be iteration/pass-free and
        ``call_instance`` iteration/pass-QUALIFIED; the torch/preview module
        representations go through :meth:`mint` instead.
        """

        cohort = (call_instance, module_site, layer_type, output_slot)
        ordinal = self._seen.get(cohort, 0) + 1
        self._seen[cohort] = ordinal
        return render_site_key(module_site, layer_type, output_slot, ordinal)


def operation_witness(op: Any) -> tuple[str | None, int | None] | None:
    """Return the op's source-location witness ``(file, line)``, or ``None``.

    The witness is the deepest OPERATION frame of the persisted
    ``code_context``: live contexts are ordered shallow-to-deep and END
    with the trace CALL-SITE entry -- the first surviving frame above the
    outermost forward, appended after the operation frames
    (``utils/introspection.py``) -- so the operation frame is
    ``code_context[-2]``. NEVER ``[0]``: that is the outermost forward
    line, identical across captures of a nested model whose INNER branch
    sites differ -- reading it re-opens exactly the false-join class this
    witness exists to refuse. A degenerate single-entry context (not
    producible by ``tl.trace``, whose user call-site frame always survives
    the internal-frame filter) falls back to its only frame. The selector
    is pinned byte-for-byte against the reference implementation.
    """

    code_context = getattr(op, "code_context", None)
    if not code_context:
        return None
    frame = code_context[-2] if len(code_context) >= 2 else code_context[0]
    return (getattr(frame, "file", None), getattr(frame, "line_number", None))


__tl_layer__ = "L1"
