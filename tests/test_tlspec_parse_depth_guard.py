"""r55 CLASS 4 immunizer -- bounded JSON/literal parse at every manifest boundary.

r54 ``free_2`` (LOW-MED): a deeply-nested ``manifest.json`` blew the C recursion
stack in stdlib ``json.load`` at load / format-detection -- an uncaught
``RecursionError`` that escaped ``tl.load(path)`` before the descriptor-parse
graceful-degradation net ran. The class is closed by ONE bounded reader
(``_io/_json``: byte ceiling + string-aware depth prescan BEFORE ``json.loads``)
routed at every manifest boundary, plus an independent depth counter through
``runnable_load._parse_literal``. Over-limit degrades typed; never crashes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import _json
from torchlens._io.runnable_load import _MAX_LITERAL_NESTING_DEPTH, _parse_literal

pytestmark = pytest.mark.smoke


# --------------------------------------------------------------------------- #
# (a) bounded JSON reader                                                      #
# --------------------------------------------------------------------------- #


def test_bounded_json_refuses_deep_nesting_without_recursion() -> None:
    """A depth far past the ceiling is a typed ``JSONDecodeError``, never a crash."""

    deep = "[" * 5000 + "]" * 5000
    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded(deep)


def test_bounded_json_refuses_oversize_payload() -> None:
    """An over-size payload is a typed ``JSONDecodeError`` before parsing."""

    with pytest.raises(json.JSONDecodeError):
        _json.loads_bounded("[]", max_bytes=1)


def test_bounded_json_parses_normal_manifest() -> None:
    """A normal shallow object parses identically to stdlib ``json``."""

    obj = {"a": 1, "b": [1, 2, {"c": [3, 4]}], "d": "text with [brackets] {inside}"}
    text = json.dumps(obj)
    assert _json.loads_bounded(text) == obj


def test_bounded_json_string_aware_depth() -> None:
    """Brackets inside string literals do not inflate the measured depth."""

    text = json.dumps({"k": "[[[[[[[[[[ not real nesting ]]]]]]]]]]"})
    assert _json.loads_bounded(text, max_depth=3) == {"k": "[[[[[[[[[[ not real nesting ]]]]]]]]]]"}


def test_bounded_json_refuses_flat_array_object_count_bomb() -> None:
    """A shallow-but-huge flat array is refused BEFORE the decoder allocates it.

    R60-1: the byte + depth ceilings do not bound the number of Python objects
    ``json.loads`` allocates. A flat ``[0,0,0,...]`` array is depth-1 and small on
    disk yet expands ~16x into RSS. The object-count prescan (commas as an upper
    bound on nodes) refuses it before any allocation.
    """

    text = "[" + ",".join("0" for _ in range(1000)) + "]"
    with pytest.raises(json.JSONDecodeError, match="node count"):
        _json.loads_bounded(text, max_nodes=100)
    # Under the ceiling it still parses.
    assert _json.loads_bounded(text, max_nodes=10_000) == [0] * 1000


def test_bounded_json_refuses_object_count_bomb_via_nested_containers() -> None:
    """Many tiny empty containers (opens, no commas) also count as nodes."""

    text = "[" + ",".join("{}" for _ in range(500)) + "]"
    with pytest.raises(json.JSONDecodeError, match="node count"):
        _json.loads_bounded(text, max_nodes=100)


def test_object_count_prescan_is_string_aware() -> None:
    """Commas inside string literals never count toward the node ceiling."""

    text = json.dumps({"k": "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w"})
    # One object + one key/value = 2 nodes; the 22 in-string commas must not count.
    assert _json.loads_bounded(text, max_nodes=5) == json.loads(text)


def test_read_bounded_threads_max_nodes(tmp_path: Path) -> None:
    """The file readers enforce the object-count ceiling too (refuse-before-alloc)."""

    payload = tmp_path / "bomb.json"
    payload.write_text("[" + ",".join("0" for _ in range(1000)) + "]")
    with pytest.raises(json.JSONDecodeError, match="node count"):
        _json.read_bounded(payload, max_nodes=100)
    with (
        payload.open("r", encoding="utf-8") as handle,
        pytest.raises(json.JSONDecodeError, match="node count"),
    ):
        _json.load_bounded(handle, max_nodes=100)


# --------------------------------------------------------------------------- #
# (b) independent literal-depth counter                                       #
# --------------------------------------------------------------------------- #


def test_parse_literal_refuses_over_depth_without_recursion() -> None:
    """A nested literal past the ceiling raises ``ValueError``, not ``RecursionError``."""

    node: dict = {"kind": "int", "value": 0}
    for _ in range(_MAX_LITERAL_NESTING_DEPTH + 50):
        node = {"kind": "list", "items": [node]}
    with pytest.raises(ValueError):
        _parse_literal(node)


def test_parse_literal_accepts_shallow_nesting() -> None:
    """A shallow nested literal parses cleanly (no over-refusal)."""

    node: dict = {"kind": "int", "value": 0}
    for _ in range(20):
        node = {"kind": "list", "items": [node]}
    parsed = _parse_literal(node)
    assert parsed is not None


# --------------------------------------------------------------------------- #
# (c) end-to-end: deep manifest degrades typed, never RecursionError           #
# --------------------------------------------------------------------------- #


class _M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc(x))


def test_deeply_nested_manifest_does_not_crash_load(tmp_path: Path) -> None:
    """A depth-900 nested ``manifest.json`` never escapes ``tl.load`` as a crash."""

    trace = tl.trace(
        _M().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    bundle = tmp_path / "deep.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)

    manifest_path = bundle / "manifest.json"
    text = manifest_path.read_text()
    # Splice a depth-900 nested array as a new top-level key WITHOUT recursing in
    # the builder (raw string construction, mirroring the free_2 repro).
    depth = 900
    raw = "[" * depth + "0" + "]" * depth
    injected = '{"__deep__": ' + raw + ", " + text[1:]
    manifest_path.write_text(injected)

    try:
        tl.load(str(bundle))
    except RecursionError:  # pragma: no cover - the exact failure we forbid
        pytest.fail("uncaught RecursionError escaped tl.load() on a deep manifest")
    except Exception:
        pass  # any typed disposition (TorchLensIOError / analysis-only) is acceptable


def test_normal_bundle_still_loads(tmp_path: Path) -> None:
    """The bounded reader does not perturb a legitimate load."""

    trace = tl.trace(
        _M().eval(),
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    bundle = tmp_path / "clean.tlspec"
    tl.save(trace, str(bundle), level="runnable", include_weights=True)
    loaded = tl.load(str(bundle))
    assert loaded is not None


# --------------------------------------------------------------------------- #
# (d) the OTHER doors into the same artifacts (r58 A3b/A3c)                     #
# --------------------------------------------------------------------------- #
#
# The bounded reader was routed at ``tl.load``'s manifest boundary only. Three other
# entries read the SAME untrusted artifacts with stdlib ``json``, so a hostile payload
# escaped them as an untyped RecursionError with no ceiling at all.


def _deep_json_object(depth: int, extra: str = "") -> str:
    """Build a JSON object carrying a raw depth-``depth`` nested array."""

    nest = "[" * depth + "0" + "]" * depth
    return '{"__deep__": ' + nest + (", " + extra if extra else "") + "}"


def test_intervention_spec_deep_json_refuses_typed(tmp_path: Path) -> None:
    """``load_intervention_spec`` refuses an over-nested spec.json typed.

    Fail-before: stdlib ``json.load`` in ``_read_json_file`` answered 10,000 nested
    arrays with an untyped ``RecursionError`` escaping the public loader.
    """

    from torchlens.intervention.errors import ReplayPreconditionError
    from torchlens.intervention.save import load_intervention_spec

    spec_dir = tmp_path / "deep_spec"
    spec_dir.mkdir()
    (spec_dir / "spec.json").write_text(_deep_json_object(10_000, '"format_version": 1'))
    (spec_dir / "manifest.json").write_text("{}")

    with pytest.raises(ReplayPreconditionError, match="not parsable JSON"):
        load_intervention_spec(spec_dir)


def test_validate_tlspec_deep_manifest_refuses_typed(tmp_path: Path) -> None:
    """``validate_tlspec`` refuses an over-nested manifest.json typed.

    Same artifact as ``test_deeply_nested_manifest_does_not_crash_load``, different
    door: the public validator read the manifest with a bare ``json.load``.
    """

    bundle = tmp_path / "deep.tlspec"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(_deep_json_object(900))

    with pytest.raises(ValueError, match="Failed to parse .tlspec manifest JSON"):
        tl.validation.validate_tlspec(bundle)


def test_profiler_kineto_trace_deep_json_refuses_typed(tmp_path: Path) -> None:
    """The profiler bridge refuses an over-nested external Kineto trace typed."""

    from torchlens.bridge.profiler import _load_trace

    kineto = tmp_path / "kineto.json"
    kineto.write_text(_deep_json_object(10_000))

    with pytest.raises(json.JSONDecodeError):
        _load_trace(kineto)


# --------------------------------------------------------------------------- #
# (e) the ceiling must apply BEFORE the allocation (r58 A3d)                    #
# --------------------------------------------------------------------------- #


def test_read_bounded_refuses_before_reading_the_whole_file(tmp_path: Path) -> None:
    """The path reader stops at ``max_bytes + 1``; it never slurps the file first.

    Fail-before: ``Path.read_text()`` materialized the ENTIRE attacker-sized file and
    only then handed it to a byte ceiling that could no longer prevent anything.
    """

    payload = tmp_path / "big.json"
    payload.write_text("[" + ",".join("0" for _ in range(50_000)) + "]")
    assert payload.stat().st_size > 64
    with pytest.raises(json.JSONDecodeError, match="maximum size"):
        _json.read_bounded(payload, max_bytes=64)
    assert _json.read_bounded(payload) == [0] * 50_000


def test_read_bytes_bounded_enforces_the_same_ceiling(tmp_path: Path) -> None:
    """The raw-bytes reader (checksum subjects) honours the same ceiling."""

    payload = tmp_path / "descriptor.json"
    payload.write_text('{"descriptor_kind": "x"}')
    with pytest.raises(json.JSONDecodeError, match="maximum size"):
        _json.read_bytes_bounded(payload, max_bytes=4)
    assert _json.read_bytes_bounded(payload) == payload.read_bytes()


# --------------------------------------------------------------------------- #
# (e) R33-1: allocation tracks the file, not the ceiling                        #
# --------------------------------------------------------------------------- #


def _peak_bytes_of(fn) -> int:
    """Return the tracemalloc peak (bytes) of a single call."""

    import tracemalloc

    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@pytest.mark.parametrize(
    "reader",
    [
        lambda p, mb: _json.read_bytes_bounded(p, max_bytes=mb),
        lambda p, mb: _json.read_bounded(p, max_bytes=mb),
    ],
    ids=["read_bytes_bounded", "read_bounded"],
)
def test_bounded_readers_allocate_the_file_not_the_ceiling(tmp_path: Path, reader) -> None:
    """A tiny file under a huge ceiling must not transiently allocate the ceiling.

    Fail-before (R33-1): the readers did ``handle.read(max_bytes + 1)``, which
    pre-allocates a ``max_bytes + 1`` buffer regardless of the real file size, so
    every ``.tlspec`` load transiently requested ~512 MiB no matter how small the
    manifest -- an allocation DoS under ``RLIMIT_AS``/strict overcommit, invisible to
    RSS-only audits. The fixed reader stats the open fd and reads only what is there.
    """

    ceiling = 256 * 1024 * 1024  # 256 MiB, far above the payload
    payload = tmp_path / "small.json"
    payload.write_text("[" + ",".join("0" for _ in range(4_000)) + "]")
    file_size = payload.stat().st_size
    assert file_size < 1 * 1024 * 1024, "payload must be far below the ceiling"

    peak = _peak_bytes_of(lambda: reader(payload, ceiling))

    # Allow generous headroom for decode/parse overhead but stay FAR below the
    # ceiling: a ceiling-sized allocation would blow this by orders of magnitude.
    assert peak < 32 * 1024 * 1024, (
        f"bounded reader peaked at {peak} bytes on a {file_size}-byte file under a "
        f"{ceiling}-byte ceiling; it is allocating the ceiling, not the payload"
    )


def test_read_bounded_still_refuses_a_file_that_exceeds_the_ceiling(tmp_path: Path) -> None:
    """Stat-then-allocate must not weaken the over-size refusal (tripwire intact)."""

    payload = tmp_path / "oversize.json"
    payload.write_text("[" + ",".join("0" for _ in range(50_000)) + "]")
    with pytest.raises(json.JSONDecodeError, match="maximum size"):
        _json.read_bounded(payload, max_bytes=64)
    with pytest.raises(json.JSONDecodeError, match="maximum size"):
        _json.read_bytes_bounded(payload, max_bytes=64)


# --------------------------------------------------------------------------- #
# (f) source-scan: no bare json.load(s) anywhere in the package (r58 A3f)       #
# --------------------------------------------------------------------------- #

# Reason-bearing ledger. The lint previously rooted at ``_io`` + ``io`` only, which
# made the intervention-spec, validation, and bridge readers structurally unreachable
# by it -- three live findings the gate was supposed to be holding. Rooting it at the
# whole package converts them into a standing gate; the rows below are the remaining
# offenders, each with why it is exempt or still pending.
BARE_JSON_READ_LEDGER = {
    # PERMANENT: reads a first-party file shipped inside the wheel
    # (torchlens/autoroute/data/imagenet1k_labels.json), not an artifact boundary.
    "autoroute/_builtin_output.py": "package-shipped label table, not untrusted input",
}


def test_no_bare_json_read_in_the_package() -> None:
    """Every JSON READ in torchlens routes through the bounded helper or is ledgered."""

    package_root = Path(tl.__file__).resolve().parent
    offenders: list[str] = []
    for source_path in sorted(package_root.rglob("*.py")):
        relative = source_path.relative_to(package_root).as_posix()
        if relative == "_io/_json.py" or relative in BARE_JSON_READ_LEDGER:
            continue
        for lineno, line in enumerate(source_path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "json.load(" in line or "json.loads(" in line:
                offenders.append(f"{relative}:{lineno}: {stripped}")
    assert not offenders, "bare json.load(s) at an artifact boundary (use _json.*_bounded): " + str(
        offenders
    )


def test_bare_json_read_ledger_has_no_stale_rows() -> None:
    """A ledgered file that no longer reads JSON bare must lose its row."""

    package_root = Path(tl.__file__).resolve().parent
    stale = [
        relative
        for relative in BARE_JSON_READ_LEDGER
        if not any(
            "json.load(" in line or "json.loads(" in line
            for line in (package_root / relative).read_text().splitlines()
            if not line.strip().startswith("#")
        )
    ]
    assert not stale, f"delete these BARE_JSON_READ_LEDGER rows: {stale}"
