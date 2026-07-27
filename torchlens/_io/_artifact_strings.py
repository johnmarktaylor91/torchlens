"""Shared closed-grammar sanitizer for attacker-controlled portable device strings.

A ``.tlspec`` manifest is *data*, not authority: a hostile artifact may carry any
bytes in the plaintext ``manifest.json`` ``logical_device`` / ``device_at_save``
fields. Round 54 ``sec_2`` (HIGH) showed the tinygrad payload codec passing such a
raw string straight into ``tinygrad.Tensor(device=<attacker str>)``; tinygrad's
``disk:<path>`` backend then writes an attacker-named file on a default
``tl.load(path)`` -- arbitrary file write, no ``.run()``, no trust flag.

This module owns the ONE device-token gate every non-torch payload codec routes
an *artifact*-supplied device string through, so no codec can smuggle a
filesystem path / URL / scheme into a materialization sink. Admission is layered
and deliberately contains NO hand-maintained accelerator vocabulary -- both
static-list directions were falsified in practice (round-56 ``corr_1``: the r55
positive allowlist omitted legit ``NV``/``AMD``/``WEBGPU``/``QCOM``/``DSP``/
``NPY`` and silently relocated their payloads to the runtime default, commonly
CPU; round-57 probe D: a static ``{disk, ext}`` denylist named a base tinygrad
0.13 no longer ships (``ext``) while wrongly admitting its NEW ``tinyfs`` socket
sink). Either static list drifts from the installed runtime; the gate below
cannot:

1. **Closed grammar** (every backend): a token is exactly an alphabetic base
   with an optional ``:<pure-int>`` index, so every path / URL / scheme /
   relative spelling is inexpressible before any vocabulary question arises.
2. **I/O-sink exclude** (tinygrad): the small, security-motivated set of
   construction-time file/socket/fabric sink bases
   (:data:`_TINYGRAD_IO_SINK_BASES`) is refused BEFORE the runtime-membership
   check, so a sink base is refused even though the installed runtime names it
   (``DISK``/``TINYFS``/``RDMA`` all appear in ``Device._devices``).
3. **Runtime-derived vocabulary** (tinygrad, fail-closed): the base must be
   named by the *installed* ``tinygrad.Device._devices`` inventory
   (:func:`_tinygrad_runtime_device_bases`). A grammar-valid but unknown base
   refuses to the runtime default, and when tinygrad is not importable the
   inventory is empty so EVERY artifact token refuses to the default -- never a
   crash, never an open-ended admission. Because the vocabulary is read from
   the runtime itself, a future tinygrad accelerator is admitted automatically
   and a removed one stops being admitted: no drift by construction.

The other non-torch backends (mlx/paddle/jax/tf) expose no filesystem/network
device, so grammar-only admission is sufficient there -- zero security
regression, and it also stops the shared gate from over-refusing any
accelerator those runtimes accept.

Caller-supplied ``map_location`` stays trusted input -- the victim owns their own
load call -- and passes through unchanged; only the artifact token is gated.
"""

from __future__ import annotations

import re
from typing import Any

# Construction-time I/O sink bases in tinygrad, refused unconditionally and
# BEFORE the runtime-membership check. This is the one small hand-maintained
# set; it is security-motivated (each entry names a base whose device *open*
# performs filesystem/network I/O), NOT an accelerator vocabulary:
#   disk   -> os.open(<token filename>, O_CREAT) on device open (r54 sec_2);
#   tinyfs -> socket.connect(TINYFS_ENDPOINT) on device open;
#   rdma   -> NIC / RDMA fabric setup on device open;
#   ext    -> historical file-backed base (gone in tinygrad 0.13; denied
#             defensively for older installed runtimes);
#   shm    -> shared-memory sub-scheme spelling, denied defensively.
_TINYGRAD_IO_SINK_BASES: frozenset[str] = frozenset({"disk", "tinyfs", "rdma", "ext", "shm"})

# A safe device token is exactly a bare alphabetic base name with an optional
# ``:<pure-int index>`` suffix. Anything else -- a path separator, ``..``, a URL
# scheme, a non-integer suffix -- fails the full match and is refused.
_DEVICE_TOKEN_RE = re.compile(r"^([A-Za-z]+)(?::([0-9]+))?$")


def _tinygrad_runtime_device_bases() -> frozenset[str]:
    """Return the installed tinygrad runtime's device bases, lowercased.

    Fail-closed by construction: when tinygrad is absent (or its inventory is
    unreadable) the vocabulary is EMPTY, so every artifact token resolves to the
    runtime default rather than crashing or being admitted unchecked.
    """

    try:
        from tinygrad.device import Device

        return frozenset(str(name).lower() for name in Device._devices)
    except Exception:
        return frozenset()


def _strip_device_wrapper(text: str) -> str:
    """Unwrap the common ``Device(...)`` / ``Place(...)`` / ``DeviceType....`` reprs.

    These are the wrapper spellings the backends emit from ``str(tensor.device)``
    / ``str(tensor.place)``; the inner token is then validated against the closed
    grammar. Unwrapping is purely structural (no eval, no import) -- a hostile
    ``device(disk:/tmp/x)`` unwraps to ``disk:/tmp/x`` which then fails the token
    grammar and is refused.
    """

    lowered = text.lower()
    for prefix in ("device(", "place("):
        if lowered.startswith(prefix) and text.endswith(")"):
            return text[len(prefix) : -1].strip().strip("'\"")
    if "devicetype." in lowered:
        return text.rsplit(".", maxsplit=1)[-1].strip()
    return text


def _sanitize_artifact_device_token(raw_device: Any, backend_name: str) -> str | None:
    """Return an artifact device token IFF the closed gate admits it.

    ``raw_device`` is attacker-controllable ``manifest.json`` data. Returns the
    ORIGINAL token unchanged when its (wrapper-stripped) base passes the closed
    grammar plus, for tinygrad, the sink-exclude-then-runtime-membership check --
    preserving the exact string the backend's own resolver already accepts, so
    admission is zero-regression for legitimate tokens. Returns ``None``
    (meaning "use the runtime default") for every path / URL / scheme /
    relative / malformed token, for every tinygrad I/O-sink base, and for every
    grammar-valid base the installed tinygrad runtime does not name (fail-closed
    -- including the tinygrad-absent case, where the runtime vocabulary is
    empty), so a materialization sink never sees a raw filesystem string and an
    unknown token never silently relocates elsewhere than the default.
    """

    if not isinstance(raw_device, str):
        return None
    text = raw_device.strip()
    if not text or text.lower() == "unknown":
        return None
    inner = _strip_device_wrapper(text)
    match = _DEVICE_TOKEN_RE.match(inner)
    if match is None:
        return None
    if backend_name.lower() == "tinygrad":
        base = match.group(1).lower()
        # Sink exclude FIRST: a file/socket/fabric sink base must be refused
        # even though the installed runtime names it in ``Device._devices``.
        if base in _TINYGRAD_IO_SINK_BASES:
            return None
        if base not in _tinygrad_runtime_device_bases():
            return None
    return text


def _resolve_portable_device(backend_name: str, raw_device: Any, map_location: Any) -> Any | None:
    """Resolve the device a non-torch payload codec should materialize onto.

    ``map_location`` is trusted caller input (the victim owns their own load call)
    and passes through unchanged. ``raw_device`` is artifact ``manifest.json`` data
    and is admitted ONLY through the closed device gate
    (:func:`_sanitize_artifact_device_token`). ``backend_name`` selects the
    backend-aware layer of that gate: tinygrad tokens additionally pass the
    I/O-sink exclude and the runtime-derived fail-closed vocabulary; the other
    non-torch backends are grammar-only (they expose no filesystem/network
    device).

    Returns the trusted ``map_location`` when supplied, else the sanitized artifact
    token, else ``None`` (runtime default / refused).
    """

    if map_location is not None:
        return map_location
    return _sanitize_artifact_device_token(raw_device, backend_name)
