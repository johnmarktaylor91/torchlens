"""Soundness of the fixed-floor cuts: each fast path returns what the path it replaced returned."""

from __future__ import annotations

import ast
import hashlib
import io
import os
from typing import Any

import pytest

from torchlens import _memory_probes
from torchlens._capture import peak_memory
from torchlens._postprocess import _branch_source
from torchlens._semantic import _facet_provenance, facets
from torchlens._semantic.facets import FacetFunction, RecordScope

_SAMPLES = (
    "",
    "x = 1\n",
    "a = 1\r\nb = 2\rc = 3\n",
    "no newline at end",
    "s = 'é dash'\nt = 2\n",
    "café = 1\n\n\n",
)


def _with_char(char: str) -> list[str]:
    """Return sources holding ``char`` at the start, middle and end, ASCII and not."""

    return [
        f"{char}x = 1\n",
        f"x = 1  # a{char}b\ny = 2\n",
        f"x = 1\n{char}",
        f"s = 'é'\n# {char}\n",
    ]


def _break_class_members() -> set[str]:
    """Return the characters of the ``_NON_PARSER_BREAKS`` regex class."""

    pattern = _branch_source._NON_PARSER_BREAKS.pattern
    assert pattern.startswith("[") and pattern.endswith("]")
    return set(pattern[1:-1])


def test_break_tuples_match_the_regex_class() -> None:
    """The substring tuples hold exactly the regex class, ASCII members in the ASCII tuple."""

    members = _break_class_members()
    assert set(_branch_source._ALL_NON_PARSER_BREAKS) == members
    assert len(_branch_source._ALL_NON_PARSER_BREAKS) == len(members)
    assert set(_branch_source._ASCII_NON_PARSER_BREAKS) == {c for c in members if c.isascii()}


@pytest.mark.parametrize("char", sorted(_break_class_members()))
def test_break_scan_agrees_with_regex_for_each_break(char: str) -> None:
    """Every break character is found, in ASCII and non-ASCII sources alike."""

    for source in _with_char(char):
        expected = _branch_source._NON_PARSER_BREAKS.search(source) is None
        assert _branch_source._has_no_non_parser_break(source) is expected is False


@pytest.mark.parametrize("source", _SAMPLES)
def test_break_scan_agrees_with_regex_without_breaks(source: str) -> None:
    """Sources with no break character take the fast path, as the regex said."""

    expected = _branch_source._NON_PARSER_BREAKS.search(source) is None
    assert _branch_source._has_no_non_parser_break(source) is expected


@pytest.mark.parametrize(
    "source",
    [*_SAMPLES, *(s for c in sorted(_break_class_members()) for s in _with_char(c))],
)
def test_split_source_lines_matches_the_parser_split(source: str) -> None:
    """The split equals CPython's own parser-style split on every sample."""

    assert _branch_source.split_source_lines(source) == ast._splitlines_no_ff(source)  # type: ignore[attr-defined]


_MEMINFO = (
    b"MemTotal:       16314660 kB\n"
    b"MemFree:         1234567 kB\n"
    b"MemAvailable:    7654321 kB\n"
    b"Buffers:          123456 kB\n"
)


def _old_meminfo_parse(raw: bytes) -> int | None:
    """The text-mode line loop the binary read replaced."""

    for line in io.StringIO(raw.decode("ascii")):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _fake_open(payloads: dict[str, bytes]) -> Any:
    """Return an ``open`` that serves ``payloads`` by path and records the mode."""

    def fake(path: str, mode: str = "r", *args: Any, **kwargs: Any) -> io.BytesIO:
        assert "b" in mode, "the fast path reads procfs in binary"
        return io.BytesIO(payloads[path])

    return fake


def test_meminfo_binary_read_matches_text_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_available_host_bytes`` returns the value the old text parse returned."""

    monkeypatch.setattr(
        _memory_probes, "open", _fake_open({"/proc/meminfo": _MEMINFO}), raising=False
    )
    assert _memory_probes._available_host_bytes() == _old_meminfo_parse(_MEMINFO) == 7654321 * 1024


@pytest.mark.skipif(not os.path.exists("/proc/meminfo"), reason="needs procfs")
def test_meminfo_binary_read_matches_text_parse_on_live_procfs() -> None:
    """On a live procfs both parses see the same MemAvailable line (to within drift)."""

    with open("/proc/meminfo", "rb") as handle:
        old = _old_meminfo_parse(handle.read())
    new = _memory_probes._available_host_bytes()
    assert old is not None and new is not None
    assert abs(new - old) < 512 * 1024 * 1024


def test_statm_binary_read_matches_text_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """``process_rss_bytes`` without psutil returns the old text parse's value."""

    statm = b"123456 7890 1234 56 0 7890 0\n"
    monkeypatch.setattr(peak_memory, "optional_psutil", lambda: None)
    monkeypatch.setattr(peak_memory, "open", _fake_open({"/proc/self/statm": statm}), raising=False)
    expected = int(statm.decode().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    assert peak_memory.process_rss_bytes() == expected


def _full_hash(publics: tuple[FacetFunction, ...]) -> str:
    """The per-snapshot digest the reuse replaced, recomputed from scratch."""

    digest = hashlib.sha256()
    for public in publics:
        digest.update(public.name.encode("utf-8"))
        digest.update(repr(public.class_names).encode("utf-8"))
        digest.update(repr(public.qualnames).encode("utf-8"))
        digest.update(str(public.has_predicate).encode("utf-8"))
        digest.update(public.target_scope.encode("utf-8"))
        digest.update(public.source.encode("utf-8"))
    return digest.hexdigest()[:16]


def _public(name: str, scope: RecordScope = "op") -> FacetFunction:
    return FacetFunction(
        name=name,
        class_names=("Linear",),
        qualnames=(f"m.{name}",),
        has_predicate=False,
        target_scope=scope,
        source="user",
    )


def test_provenance_reuse_equals_a_fresh_hash() -> None:
    """Reused, recomputed and changed recipe tuples all give the from-scratch digest."""

    a, b = _public("a"), _public("b")
    first = _facet_provenance.provenance_id((a, b))
    assert first == _full_hash((a, b))
    assert _facet_provenance.provenance_id((a, b)) == first
    assert _facet_provenance.provenance_id((b, a)) == _full_hash((b, a)) != first
    assert _facet_provenance.provenance_id((a,)) == _full_hash((a,))
    equal_copy = (_public("a"), _public("b"))
    assert _facet_provenance.provenance_id(equal_copy) == first
    assert _facet_provenance.provenance_id((a, _public("b", "module"))) == _full_hash(
        (a, _public("b", "module"))
    )
    assert _facet_provenance.provenance_id(()) == _full_hash(())


def test_snapshot_provenance_tracks_registry_changes() -> None:
    """``facets.snapshot`` keeps the from-scratch digest across register and reset."""

    def _check() -> str:
        snap = facets.snapshot()
        assert snap.provenance_id == _full_hash(tuple(r.public for r in snap.functions))
        return snap.provenance_id

    before = _check()
    assert _check() == before

    def ff_soundness_probe(record: Any) -> dict[str, int]:
        return {"ff_soundness_probe": 1}

    with facets.adding(ff_soundness_probe):
        during = _check()
        assert during != before
        assert facets.snapshot(extra_facets=[ff_soundness_probe]).provenance_id == _full_hash(
            tuple(r.public for r in facets.snapshot(extra_facets=[ff_soundness_probe]).functions)
        )
    assert _check() == before


def test_split_regex_is_still_the_documented_definition() -> None:
    """The regex the tuples mirror is unchanged: the CPython non-parser breaks."""

    assert _break_class_members() == set("\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029")
