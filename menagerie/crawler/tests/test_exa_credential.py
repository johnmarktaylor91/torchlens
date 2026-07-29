"""The Exa credential must lift the quota without ever reaching an artifact.

Two independent requirements, and the second is the one with permanent consequences.

**Availability.** The anonymous Exa MCP endpoint is rate limited at roughly 1,400 searches per
month -- about 230 models -- so it was never viable for a 28,482-model campaign. It exhausted
mid-run on 2026-07-28 and cost six of six first attempts to ``research-tools-unavailable``. An
authenticated key lifts that ceiling; an absent key must still fall back rather than refusing to
start, because a missing credential should never be why a campaign cannot run.

**Secrecy.** ``johnmarktaylor91/torchlens`` is a PUBLIC repository and the crawler serializes a
great deal -- envelopes, receipts, manifests, effort records, attempt records. A key leaking into
any of those would be permanent and public. The credential is therefore read from the environment
at spawn time and injected only into the child process's ``--mcp-config``.
"""

from __future__ import annotations

import json

import pytest

from menagerie.crawler.author_executor import (
    EXA_API_KEY_ENV,
    EXA_MCP_URL,
    RESEARCH_TOOLS,
    exa_mcp_config,
)

_SENTINEL = "exa-key-sentinel-2f8badce"


def _url(config: str) -> str:
    return json.loads(config)["mcpServers"]["exa"]["url"]


def test_absent_key_falls_back_to_the_anonymous_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine without the key still runs, degraded exactly as before."""

    monkeypatch.delenv(EXA_API_KEY_ENV, raising=False)

    assert _url(exa_mcp_config()) == EXA_MCP_URL


def test_blank_key_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty or whitespace value must not produce ``?exaApiKey=``."""

    for blank in ("", "   ", "\n"):
        monkeypatch.setenv(EXA_API_KEY_ENV, blank)
        assert _url(exa_mcp_config()) == EXA_MCP_URL


def test_present_key_authenticates_the_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is what lifts the quota ceiling, so it must actually reach the URL."""

    monkeypatch.setenv(EXA_API_KEY_ENV, _SENTINEL)
    url = _url(exa_mcp_config())

    assert url.startswith(f"{EXA_MCP_URL}?exaApiKey=")
    assert _SENTINEL in url


def test_a_key_needing_escaping_survives_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key with URL-significant characters must not silently corrupt the endpoint."""

    monkeypatch.setenv(EXA_API_KEY_ENV, "a b&c=d/e?f")
    url = _url(exa_mcp_config())

    assert "a b&c=d/e?f" not in url  # raw, unescaped -- would break the query string
    assert "a%20b%26c%3Dd%2Fe%3Ff" in url


def test_the_key_is_not_baked_in_at_import_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution happens per call, so a module imported before the key still picks it up.

    A module-level constant captured at import would silently keep using the anonymous
    endpoint for the life of the process -- exactly the failure this replaced.
    """

    monkeypatch.delenv(EXA_API_KEY_ENV, raising=False)
    assert _url(exa_mcp_config()) == EXA_MCP_URL

    monkeypatch.setenv(EXA_API_KEY_ENV, _SENTINEL)
    assert _SENTINEL in _url(exa_mcp_config())


def test_webfetch_is_available_so_a_ref_can_be_confirmed_without_exa() -> None:
    """Without a fetch tool an author cannot open any page, so no ref is confirmable.

    Its absence was the sharpest finding of the Exa dependency assessment: WebSearch returns a
    synthesised answer, not a page, so "a ref you confirmed" became unconfirmable and the
    fabricated-ref class returned. WebFetch is the resilience layer for an Exa outage.
    """

    assert "WebFetch" in RESEARCH_TOOLS
    assert "WebSearch" in RESEARCH_TOOLS


def test_the_credential_never_appears_in_the_tool_rules(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The rules are rendered into briefs and recorded; the key must not ride along."""

    from menagerie.crawler.author_executor import stage_tool_rules

    monkeypatch.setenv(EXA_API_KEY_ENV, _SENTINEL)
    rules = stage_tool_rules(write_root=tmp_path / "attempt", read_roots=[tmp_path])

    assert _SENTINEL not in " ".join(rules)
