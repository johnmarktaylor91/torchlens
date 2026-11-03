"""Bounded JSON reads for attacker-controlled ``.tlspec`` manifest boundaries.

A hostile ``.tlspec`` can carry a deeply-nested ``manifest.json`` whose stdlib
``json.load`` blows the C recursion stack -- an uncaught ``RecursionError`` that
escapes ``tl.load(path)`` before any descriptor-parse graceful-degradation net
runs (round 54 ``free_2``). This module owns the ONE bounded reader every manifest
/ metadata / format-detection JSON boundary routes through:

1. a byte ceiling (rejects an absurd manifest before it is read into memory), and
2. a single-pass, string-aware bracket-depth *prescan* that fails BEFORE the
   recursive ``json.loads`` ever runs.

Both violations surface as ``json.JSONDecodeError`` so every existing
``except (OSError, json.JSONDecodeError)`` handler at the call sites catches them
unchanged and applies its existing disposition (``None`` / typed
``ArtifactError``) -- an over-nested artifact degrades typed, never crashes.
``RecursionError`` is also caught as a belt in case the ambient recursion limit is
below the depth ceiling.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import IO, Any

# Manifests for large models carry many tensor entries but are shallow (nesting is
# a handful of levels); the deepest legitimate structure is a nested literal-list
# constant, still far below this. 200 sits comfortably above any real manifest and
# comfortably below the ~496-frame depth at which stdlib ``json.load`` overflows
# the C stack at the default recursion limit, so the prescan always fires first.
_MAX_JSON_DEPTH = 200

# 512 MiB: no legitimate ``manifest.json`` approaches this; it bounds the prescan's
# worst-case cost and rejects an absurd artifact before it is read into memory.
_MAX_JSON_BYTES = 512 * 1024 * 1024

# Object-count ceiling (R60-1). The byte + depth ceilings do NOT bound the number
# of Python objects ``json.loads`` allocates: a manifest well under the byte cap
# but packed with tiny values (``[0,0,0,...]`` / ``{}`` runs) expands ~16x into RSS
# BEFORE any downstream body-index cross-check can refuse it (an 81 MiB such
# manifest measured 1.24 GiB RSS; ~8 GiB projected at the byte cap). The prescan
# already walks every structural character, so it counts value-producing tokens
# (container opens + value separators, an upper bound on the node count) and
# refuses BEFORE the recursive decoder allocates anything. 64M nodes caps the
# worst case near ~2 GiB while sitting far above any real body-index-bounded
# manifest (which is orders of magnitude smaller).
_MAX_JSON_NODES = 64_000_000

_OPEN_BRACKETS = frozenset("[{")
_CLOSE_BRACKETS = frozenset("]}")

# The prescan's state machine only ever transitions on a quote, a backslash, a
# bracket, or a value-separating comma; every other character is inert. Both
# patterns below let the C regex engine skip the inert bulk (a multi-MB manifest is
# >90% inert) instead of paying one Python loop iteration per character. Commas are
# scanned only for the node-count bound; a comma inside a string never counts (the
# state machine is quote-aware).
_QUOTE_OR_BRACKET = re.compile(r'["\[\]{},]')
_ESCAPE_RELEVANT = re.compile(r'["\\\[\]{},]')

# Reduce/scan in bounded slices so a nesting bomb is still refused after roughly
# one chunk rather than after a full-payload reduction (the original per-character
# loop bailed within ~``max_depth`` characters).
_PRESCAN_CHUNK_CHARS = 1 << 20


def _refuse(detail: str, text: str) -> json.JSONDecodeError:
    """Build a ``JSONDecodeError`` so existing JSON-boundary handlers catch it."""

    return json.JSONDecodeError(detail, text if text else " ", 0)


def _fd_size(handle: IO[Any]) -> int | None:
    """Return the size of an open handle's fd, or ``None`` if it cannot be stat'd.

    ``fstat`` on the fd we are about to read (rather than a separate ``path.stat``)
    keeps the size measurement bound to the exact bytes the read will consume.
    """

    try:
        return os.fstat(handle.fileno()).st_size
    except (OSError, ValueError, AttributeError):  # non-seekable / pipe / no fileno
        return None


def _bounded_read_bytes(handle: IO[bytes], max_bytes: int) -> bytes:
    """Read up to ``max_bytes`` bytes, allocating the FILE size, not the ceiling.

    ``handle.read(max_bytes + 1)`` pre-allocates a ``max_bytes + 1`` buffer whatever
    the real file size is, so a tiny manifest under the 512-MiB ceiling still
    transiently requested ~512 MiB (R33-1). We stat the open fd first: a file already
    over the ceiling is refused before any allocation, and a file within it reads
    exactly its own size plus one sentinel byte (to still catch a file that grew after
    the stat). When the size cannot be determined (a pipe/non-seekable handle) we fall
    back to the ceiling read, since there is nothing else to bound it by.
    """

    size = _fd_size(handle)
    if size is not None and size > max_bytes:
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
    read_count = (size + 1) if size is not None else (max_bytes + 1)
    data = handle.read(read_count)
    if len(data) > max_bytes:  # grew after the stat, or unbounded fallback
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
    return data


def _prescan_depth(text: str, *, max_depth: int, max_nodes: int = _MAX_JSON_NODES) -> None:
    """Refuse over-nested OR over-populated JSON via a single-pass, string-aware scan.

    Zero recursion: a running bracket depth that never enters ``json``'s recursive
    decoder. Quote/escape aware so brackets and commas inside string literals do
    not count. The scan bails as soon as the running depth exceeds the depth
    ceiling OR the running node count (container opens + value separators, an upper
    bound on the number of Python objects the decoder would allocate) exceeds the
    object-count ceiling, so a nesting bomb whose brackets are front-loaded and an
    object-count bomb whose commas are front-loaded are both refused within the
    FIRST chunk rather than after scanning the whole payload. Note the bail is not
    per-character: ``findall`` first materializes the matches for the current
    ``_PRESCAN_CHUNK_CHARS`` slice, so up to one chunk's structural characters are
    collected before either check can fire.

    The scan is the same state machine as a naive per-character loop, but the
    inert characters are skipped by the regex engine. When the payload carries no
    backslash at all the escape branch is unreachable, so each chunk is reduced to
    just its structural characters before the Python loop runs; otherwise the scan
    walks only the escape-relevant character positions (a backslash's effect on the
    very next character is recovered from the match offsets).
    """

    depth = 0
    nodes = 0
    in_string = False

    def _too_many_nodes() -> json.JSONDecodeError:
        """Build the typed refusal for a manifest exceeding the node budget."""
        return _refuse(f"manifest JSON node count exceeds the maximum of {max_nodes}", text)

    if "\\" not in text:
        # R60 fast refusal (b8): establish the node count at C speed FIRST --
        # ``str.split('"')`` alternates outside/inside-string segments (no
        # backslash means no escaped quotes), so counting separators/opens over
        # the outside segments is exactly the loop's node accounting, at ~GB/s
        # instead of ~1.6 MB/s of per-character Python. A 130 MB comma bomb is
        # refused in ~0.2 s instead of ~80 s. Chunked so a front-loaded bomb
        # refuses early and transient memory stays one chunk. The stateful scan
        # below still owns the (order-dependent) depth ceiling. Disclosed
        # residual: a payload carrying a backslash routes to the exact
        # escape-aware scan, whose refusal stays CPU-slow near the ceiling.
        fast_in_string = False
        fast_nodes = 0
        for start in range(0, len(text) or 1, _PRESCAN_CHUNK_CHARS):
            segments = text[start : start + _PRESCAN_CHUNK_CHARS].split('"')
            outside_segments = segments[1::2] if fast_in_string else segments[0::2]
            for segment in outside_segments:
                fast_nodes += segment.count(",") + segment.count("[") + segment.count("{")
            if fast_nodes > max_nodes:
                raise _too_many_nodes()
            if (len(segments) - 1) % 2:
                fast_in_string = not fast_in_string
        for start in range(0, len(text) or 1, _PRESCAN_CHUNK_CHARS):
            for char in _QUOTE_OR_BRACKET.findall(text[start : start + _PRESCAN_CHUNK_CHARS]):
                if in_string:
                    if char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char in _OPEN_BRACKETS:
                    depth += 1
                    nodes += 1
                    if depth > max_depth:
                        raise _refuse(
                            f"manifest JSON nesting exceeds the maximum depth of {max_depth}",
                            text,
                        )
                    if nodes > max_nodes:
                        raise _too_many_nodes()
                elif char == ",":
                    nodes += 1
                    if nodes > max_nodes:
                        raise _too_many_nodes()
                elif char in _CLOSE_BRACKETS and depth > 0:
                    depth -= 1
        return

    # ``escaped_at`` is the absolute offset of the character a backslash escapes.
    # Reaching an escape-relevant character at that exact offset consumes the
    # escape; reaching a later one means the escaped character was inert and the
    # escape has already been consumed. Either way the flag clears, which is
    # exactly what a per-character loop does.
    escaped_at = -1
    for match in _ESCAPE_RELEVANT.finditer(text):
        char = match[0]
        if in_string:
            position = match.start()
            if position == escaped_at:
                escaped_at = -1
                continue
            escaped_at = -1
            if char == "\\":
                escaped_at = position + 1
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPEN_BRACKETS:
            depth += 1
            nodes += 1
            if depth > max_depth:
                raise _refuse(
                    f"manifest JSON nesting exceeds the maximum depth of {max_depth}", text
                )
            if nodes > max_nodes:
                raise _too_many_nodes()
        elif char == ",":
            nodes += 1
            if nodes > max_nodes:
                raise _too_many_nodes()
        elif char in _CLOSE_BRACKETS and depth > 0:
            depth -= 1


def loads_bounded(
    text: str,
    *,
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
    max_nodes: int = _MAX_JSON_NODES,
) -> Any:
    """Parse a JSON string after byte-size, depth, and object-count prescans.

    Raises ``json.JSONDecodeError`` on an over-size, over-nested, or over-populated
    payload (so the existing boundary handlers catch it), and re-raises a
    ``RecursionError`` from the stdlib decoder as a ``json.JSONDecodeError`` belt.
    """

    # ``len(text.encode("utf-8"))`` allocates a full extra copy of the payload just
    # to measure it -- ~1.5x the 512-MiB ceiling before json even parses (B8-14).
    # UTF-8 uses >= 1 byte per code point, so ``len(text) <= byte length`` always:
    # a character count over the ceiling is a byte count over the ceiling (refuse
    # without encoding), and only when the character count is within the ceiling can
    # multibyte inflation push the byte count over, so the encode runs only there.
    if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
        raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", text)
    _prescan_depth(text, max_depth=max_depth, max_nodes=max_nodes)

    def _reject_constant(name: str) -> Any:
        """Refuse the non-finite JSON constant ``name`` at parse time.

        The writers use ``allow_nan=False``, so NaN/Infinity in an artifact
        is a forgery -- and one that loads fine but makes every re-save
        raise (a stillborn artifact). Refuse at parse instead.
        """

        raise _refuse(f"manifest JSON contains the non-finite constant {name}", text)

    try:
        return json.loads(text, parse_constant=_reject_constant)
    except RecursionError as exc:  # pragma: no cover - prescan normally fires first
        raise _refuse(
            "manifest JSON nesting exceeded the interpreter recursion limit", text
        ) from exc


def load_bounded(
    handle: IO[str],
    *,
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
    max_nodes: int = _MAX_JSON_NODES,
) -> Any:
    """Read a JSON object from a text handle with bounded size, nesting, and count.

    Reads at most ``max_bytes + 1`` bytes so an oversize file is rejected without
    being fully loaded into memory, then delegates to :func:`loads_bounded`.
    """

    raw_handle = getattr(handle, "buffer", None)
    if raw_handle is not None:
        raw = _bounded_read_bytes(raw_handle, max_bytes)
        encoding = getattr(handle, "encoding", None) or "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            # A non-decodable byte is the same does-not-parse class as broken
            # JSON; the raw UnicodeDecodeError used to leak through every
            # JSON-boundary handler untyped (R73 fuzz find, 2026-08-15).
            raise _refuse(f"manifest bytes are not valid {encoding}: {exc}", "") from exc
    else:
        # Text handle with no binary buffer: read one char past the fd size (or the
        # ceiling if it cannot be stat'd) rather than the whole ceiling.
        size = _fd_size(handle)
        if size is not None and size > max_bytes:
            raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", "")
        text = handle.read((size + 1) if size is not None else (max_bytes + 1))
        if len(text.encode("utf-8")) > max_bytes:
            raise _refuse(f"manifest JSON exceeds the maximum size of {max_bytes} bytes", text)
    return loads_bounded(text, max_depth=max_depth, max_bytes=max_bytes, max_nodes=max_nodes)


def read_bounded(
    path: Path,
    *,
    encoding: str = "utf-8",
    max_depth: int = _MAX_JSON_DEPTH,
    max_bytes: int = _MAX_JSON_BYTES,
    max_nodes: int = _MAX_JSON_NODES,
) -> Any:
    """Read and parse a JSON file without ever allocating the whole file first.

    ``Path.read_text()`` materializes the ENTIRE attacker-controlled file before
    any ceiling can be applied, so a multi-GiB ``manifest.json`` is an allocation
    DoS even though :func:`loads_bounded` would have refused it a microsecond
    later. This helper opens the file and reads at most ``max_bytes + 1`` bytes, so
    the ceiling is enforced BEFORE the allocation.

    Parameters
    ----------
    path:
        JSON file to read.
    encoding:
        Text encoding of the file.
    max_depth:
        Maximum permitted bracket nesting depth.
    max_bytes:
        Maximum permitted payload size.
    max_nodes:
        Maximum permitted container-node count.

    Returns
    -------
    Any
        The decoded JSON value.

    Raises
    ------
    json.JSONDecodeError
        On an over-size or over-nested payload, so the existing JSON-boundary
        handlers at each call site catch it unchanged.
    """

    with path.open("rb") as handle:
        data = _bounded_read_bytes(handle, max_bytes)
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        # Same does-not-parse class as broken JSON (R73): route through the
        # one channel every JSON-boundary handler already catches.
        raise _refuse(f"manifest bytes are not valid {encoding}: {exc}", "") from exc
    return loads_bounded(text, max_depth=max_depth, max_bytes=max_bytes, max_nodes=max_nodes)


def read_bytes_bounded(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> bytes:
    """Read a JSON file's RAW bytes under the same ceiling as :func:`read_bounded`.

    For the boundaries that must hash the exact on-disk bytes (the merged-artifact
    descriptor checksum) before parsing them. ``Path.read_bytes()`` would allocate
    the whole attacker-sized file first; this reads at most ``max_bytes + 1`` and
    refuses over-size payloads through the same ``JSONDecodeError`` channel.

    Parameters
    ----------
    path:
        JSON file to read.
    max_bytes:
        Maximum permitted payload size.

    Returns
    -------
    bytes
        The file's raw bytes.

    Raises
    ------
    json.JSONDecodeError
        When the payload exceeds ``max_bytes``.
    """

    with path.open("rb") as handle:
        return _bounded_read_bytes(handle, max_bytes)
