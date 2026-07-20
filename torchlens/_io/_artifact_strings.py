"""Shared closed-grammar sanitizer for attacker-controlled portable device strings.

A ``.tlspec`` manifest is *data*, not authority: a hostile artifact may carry any
bytes in the plaintext ``manifest.json`` ``logical_device`` / ``device_at_save``
fields. Round 54 ``sec_2`` (HIGH) showed the tinygrad payload codec passing such a
raw string straight into ``tinygrad.Tensor(device=<attacker str>)``; tinygrad's
``disk:<path>`` backend then writes an attacker-named file on a default
``tl.load(path)`` -- arbitrary file write, no ``.run()``, no trust flag.

This module owns the ONE closed device-token allowlist every non-torch payload
codec routes an *artifact*-supplied device string through, so no codec can smuggle
a filesystem path / URL / scheme into a materialization sink. Unified over all
five non-torch backends (tinygrad/mlx/paddle/jax/tf) so a future codec inherits
the refusal by construction (round-55 CLASS 1, probe D: 0 legit device strings
over-refused, 0 attacker path tokens admitted).

Caller-supplied ``map_location`` stays trusted input -- the victim owns their own
load call -- and passes through unchanged; only the artifact token is gated.
"""

from __future__ import annotations

import re
from typing import Any

# Closed set of device *base* tokens across every supported eager backend
# (torch/tinygrad/mlx/paddle/jax/tf), case-insensitive. This is the exhaustive
# real device-name vocabulary; it deliberately EXCLUDES the path-bearing sinks
# ``disk`` and ``ext`` (tinygrad's file/mmap backends -- the sec_2 hole) and every
# scheme (``file``/``http``/...). Probe D confirmed this admits every legitimate
# cpu/gpu/cuda/... token these backends emit and refuses every path/url token.
_ALLOWED_DEVICE_BASES: frozenset[str] = frozenset(
    {
        "cpu",
        "gpu",
        "cuda",
        "mps",
        "clang",
        "metal",
        "npu",
        "xpu",
        "xla",
        "tpu",
        "rocm",
        "hip",
        "llvm",
        "python",
    }
)

# A safe device token is exactly a bare alphabetic base name with an optional
# ``:<pure-int index>`` suffix. Anything else -- a path separator, ``..``, a URL
# scheme, a non-integer suffix -- fails the full match and is refused.
_DEVICE_TOKEN_RE = re.compile(r"^([A-Za-z]+)(?::([0-9]+))?$")


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


def _sanitize_artifact_device_token(raw_device: Any) -> str | None:
    """Return an artifact device token IFF it matches the closed backend grammar.

    ``raw_device`` is attacker-controllable ``manifest.json`` data. Returns the
    ORIGINAL token unchanged when its (wrapper-stripped) base is an allowlisted
    device name with an optional pure-integer index -- preserving the exact string
    each backend's own resolver already accepts, so this is zero-regression for
    legitimate tokens. Returns ``None`` (meaning "use the runtime default") for
    every path / URL / scheme / relative / malformed token, so a materialization
    sink never sees a raw filesystem string.
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
    if match.group(1).lower() not in _ALLOWED_DEVICE_BASES:
        return None
    return text


def _resolve_portable_device(backend_name: str, raw_device: Any, map_location: Any) -> Any | None:
    """Resolve the device a non-torch payload codec should materialize onto.

    ``map_location`` is trusted caller input (the victim owns their own load call)
    and passes through unchanged. ``raw_device`` is artifact ``manifest.json`` data
    and is admitted ONLY through the closed device grammar
    (:func:`_sanitize_artifact_device_token`). ``backend_name`` is accepted for a
    stable per-codec signature; the grammar is unified across every backend
    (probe D), so no per-backend branch is required.

    Returns the trusted ``map_location`` when supplied, else the sanitized artifact
    token, else ``None`` (runtime default / refused).
    """

    del backend_name
    if map_location is not None:
        return map_location
    return _sanitize_artifact_device_token(raw_device)
