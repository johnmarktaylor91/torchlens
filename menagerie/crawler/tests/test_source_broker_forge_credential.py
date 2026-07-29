"""The forge credential must lift the API ceiling without ever reaching an artifact,
and being throttled must never be recorded as the author's bad reference.

Two defects, found together in a live rung on 2026-07-29 while resolving
``github.com/yuanzhi-zhu/SlimFlow`` at ``main``:

**The ceiling.** The broker called ``api.github.com`` unauthenticated, which is capped at 60
requests per hour. A campaign of 28,482 models resolving at least one ref each cannot run
inside that. This is the same shape as the anonymous Exa tier that stopped the previous rung:
a tier that works for a handful and dies at scale. Authenticated, the same endpoint allows
5,000 per hour.

**The lie.** The resulting ``403`` was recorded as ``bad-ref`` -- a durable claim that the
author supplied a reference that does not exist. The repository was up, the ref was correct,
and we were throttled. It is the same defect class this project has already fixed twice
(``failed:source`` for models whose source *was* found, ``R5_SKIP`` where no rung was
selected): a record that names the wrong party. It also sends retries to repair a reference
that was never broken, and would mislead any later triage of what actually needs work.

``johnmarktaylor91/torchlens`` is a PUBLIC repository and broker receipts are durable, so the
credential is attached as a request header on the API host only -- never a URL, never a
receipt, never an outcome record.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.source_broker import (
    GITHUB_API_VERSION,
    OUTCOME_BAD_REF,
    OUTCOME_FETCHED,
    OUTCOME_RATE_LIMITED,
    OUTCOME_UNREACHABLE,
    OUTCOME_UNSUPPORTED_FORGE,
    SourceBrokerError,
    TransportResponse,
    UrllibTransport,
    broker_source_pack,
    rate_limit_signal,
    resolve_github_token,
    write_broker_outputs,
)

_SENTINEL = "ghp-sentinel-4b17c2de9f0a"

RESOLVED_SHA = "8379e338134bd33e53340b47f95c13028a4f9dbf"
REPO = "github.com/yuanzhi-zhu/SlimFlow"
REF = "main"
COMMITS_URL = f"https://api.github.com/repos/yuanzhi-zhu/SlimFlow/commits/{REF}"
RAW_URL = f"https://raw.githubusercontent.com/yuanzhi-zhu/SlimFlow/{RESOLVED_SHA}/model.py"

# Verbatim shape of what GitHub returns once the anonymous hourly budget is gone.
EXHAUSTED_HEADERS = {
    "x-ratelimit-limit": "60",
    "x-ratelimit-remaining": "0",
    "x-ratelimit-reset": "1785000000",
    "x-ratelimit-resource": "core",
    "x-ratelimit-used": "60",
}
RATE_LIMIT_BODY = json.dumps({"message": "API rate limit exceeded for 203.0.113.7."}).encode()


class HeaderTransport:
    """In-memory transport carrying response headers as well as bodies."""

    def __init__(self, responses: dict[str, tuple[int, bytes, dict[str, str]]]) -> None:
        self.responses = responses
        self.requested: list[str] = []

    def __call__(self, url: str, *, max_bytes: int, timeout: float) -> TransportResponse:
        self.requested.append(url)
        status, body, headers = self.responses.get(url, (0, b"", {}))
        return TransportResponse(
            status=status,
            final_url=url,
            redirect_chain=(url,),
            body=body[:max_bytes],
            truncated=False,
            error=None if status else "no route",
            headers=headers,
        )


def _descriptor(**overrides: Any) -> dict:
    return {
        "source_id": "impl-slimflow",
        "kind": "forge-file",
        "repo": REPO,
        "path": "model.py",
        "ref": REF,
        "requested_role": "implementation",
        "basis": "The upstream repository owns the SlimFlow implementation.",
        **overrides,
    }


def _resolve(responses: dict, tmp_path: Path):
    pack = broker_source_pack(
        [_descriptor()], broker_dir=tmp_path, transport=HeaderTransport(responses)
    )
    return pack, pack.outcomes[0]


# -- 1. a throttled forge is throttling, not a bad reference ----------------


def test_rate_limited_403_is_not_recorded_as_a_bad_ref(tmp_path: Path) -> None:
    """The exact live failure: 403 with an exhausted budget, on a ref that was correct.

    Pre-fix this asserted ``bad-ref``. That record said the author named a reference that
    does not exist, when the repository was up and the reference resolved fine an hour later.
    """

    _, outcome = _resolve({COMMITS_URL: (403, RATE_LIMIT_BODY, EXHAUSTED_HEADERS)}, tmp_path)

    assert outcome.outcome == OUTCOME_RATE_LIMITED
    assert outcome.outcome != OUTCOME_BAD_REF
    assert "rate limit" in outcome.detail
    assert "not evaluated" in outcome.detail


def test_rate_limited_outcome_carries_the_forge_reset_instant(tmp_path: Path) -> None:
    """Retryability has to be actionable, so the forge's own reset is preserved."""

    pack, outcome = _resolve(
        {COMMITS_URL: (403, RATE_LIMIT_BODY, EXHAUSTED_HEADERS)}, tmp_path
    )

    assert outcome.rate_limit == {
        "status": 403,
        "remaining": 0,
        "retry_after_seconds": None,
        "reset_epoch": 1785000000,
        "resource": "core",
    }
    assert pack.blocked_by_rate_limit() is True
    assert pack.rate_limit_reset_epoch() == 1785000000
    assert pack.rows == []


def test_secondary_rate_limit_with_retry_after_is_rate_limited(tmp_path: Path) -> None:
    """GitHub's secondary limit sends ``retry-after`` and a nonzero remaining budget."""

    headers = {"retry-after": "60", "x-ratelimit-remaining": "42"}
    pack, outcome = _resolve({COMMITS_URL: (403, b"{}", headers)}, tmp_path)

    assert outcome.outcome == OUTCOME_RATE_LIMITED
    assert pack.retry_after_seconds() == 60.0


def test_429_needs_no_corroborating_header(tmp_path: Path) -> None:
    """``429`` means too many requests by definition; that is not a guess."""

    _, outcome = _resolve({COMMITS_URL: (429, b"{}", {})}, tmp_path)

    assert outcome.outcome == OUTCOME_RATE_LIMITED


def test_a_genuine_403_is_not_silently_called_rate_limiting(tmp_path: Path) -> None:
    """A 403 can also mean genuinely forbidden, so the status alone proves nothing.

    Reading every 403 as throttling would be the mirror-image lie: it would promise a
    retry that resets nothing and hide a real access refusal behind a wait.
    """

    _, outcome = _resolve({COMMITS_URL: (403, b'{"message":"Repository access blocked"}', {})}, tmp_path)

    assert outcome.outcome != OUTCOME_RATE_LIMITED
    assert outcome.outcome != OUTCOME_BAD_REF
    assert outcome.outcome == OUTCOME_UNREACHABLE


# -- 2. a genuine bad ref is still a bad ref --------------------------------


@pytest.mark.parametrize("status", [404, 422])
def test_a_nonexistent_ref_is_still_bad_ref(tmp_path: Path, status: int) -> None:
    """The forge answering "no such ref" IS a claim about what the author supplied.

    This is the half that must not be lost: fixing the false accusation must not make the
    broker unable to make a true one.
    """

    body = json.dumps({"message": "No commit found for SHA"}).encode()
    _, outcome = _resolve({COMMITS_URL: (status, body, {})}, tmp_path)

    assert outcome.outcome == OUTCOME_BAD_REF
    assert outcome.resolver_receipt is not None
    assert outcome.resolver_receipt["status"] == status


def test_a_dead_transport_is_unreachable_not_bad_ref(tmp_path: Path) -> None:
    """A network failure is ours. Nothing about the reference was ever established."""

    _, outcome = _resolve({}, tmp_path)

    assert outcome.outcome == OUTCOME_UNREACHABLE


def test_an_unsupported_forge_is_not_the_authors_fault(tmp_path: Path) -> None:
    """A GitLab URL is a real reference this resolver cannot yet dereference."""

    pack = broker_source_pack(
        [_descriptor(repo="gitlab.com/group/project")],
        broker_dir=tmp_path,
        transport=HeaderTransport({}),
    )

    assert pack.outcomes[0].outcome == OUTCOME_UNSUPPORTED_FORGE
    assert pack.outcomes[0].outcome != OUTCOME_BAD_REF


def test_the_happy_path_still_resolves_and_binds(tmp_path: Path) -> None:
    """The classification work must not disturb a normal resolution."""

    responses: dict[str, tuple[int, bytes, dict[str, str]]] = {
        COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode(), {}),
        RAW_URL: (200, b"class SlimFlow: pass\n", {}),
    }
    pack, outcome = _resolve(responses, tmp_path)

    assert outcome.outcome == OUTCOME_FETCHED
    assert pack.rows[0]["revision"] == RESOLVED_SHA
    assert pack.blocked_by_rate_limit() is False


def test_rate_limit_signal_requires_evidence_on_403() -> None:
    """The classifier itself: headers decide, status alone does not."""

    assert rate_limit_signal(403, {}) is None
    assert rate_limit_signal(403, {"x-ratelimit-remaining": "5"}) is None
    assert rate_limit_signal(200, {"x-ratelimit-remaining": "0"}) is None
    assert rate_limit_signal(403, {"X-RateLimit-Remaining": "0"}) is not None
    assert rate_limit_signal(403, {"Retry-After": "30"}) is not None
    assert rate_limit_signal(429, {}) is not None


# -- 3. the credential: used when available, absent is never fatal ----------


class _FakeResponse:
    """Minimal urllib-shaped response."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.status = status
        self._body = body
        self.headers = _message(headers)

    def read(self, size: int = -1) -> bytes:
        body, self._body = self._body, b""
        return body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _message(headers: dict[str, str]) -> Message:
    message = Message()
    for key, value in headers.items():
        message[key] = value
    return message


class _RecordingOpener:
    """Opener that records every request and serves a scripted page map."""

    def __init__(self, pages: dict[str, tuple[int, bytes, dict[str, str]]]) -> None:
        self.pages = pages
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: Optional[float] = None) -> Any:
        self.requests.append(request)
        status, body, headers = self.pages.get(request.full_url, (404, b"", {}))
        if status in (301, 302, 303, 307, 308) or status >= 400:
            raise urllib.error.HTTPError(
                request.full_url, status, "scripted", _message(headers), None
            )
        return _FakeResponse(status, body, headers)

    def authorization_for(self, url: str) -> Optional[str]:
        for request in self.requests:
            if request.full_url == url:
                return request.get_header("Authorization")
        return None


def _wired(
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[str, tuple[int, bytes, dict[str, str]]],
    token: Optional[str] = None,
) -> tuple[UrllibTransport, _RecordingOpener]:
    transport = UrllibTransport(token_resolver=lambda: token)
    opener = _RecordingOpener(pages)
    monkeypatch.setattr(transport, "_opener", opener)
    return transport, opener


def test_an_available_credential_is_actually_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The credential is the only thing that lifts 60/hour to 5,000/hour."""

    transport, opener = _wired(
        monkeypatch, {COMMITS_URL: (200, b"{}", {})}, token=_SENTINEL
    )
    transport(COMMITS_URL, max_bytes=1024, timeout=1.0)

    assert opener.authorization_for(COMMITS_URL) == f"Bearer {_SENTINEL}"
    assert opener.requests[0].get_header("X-github-api-version") == GITHUB_API_VERSION
    assert transport.github_credential_mode() == "authenticated"


def test_an_absent_credential_falls_back_instead_of_refusing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing credential must never be the reason a campaign cannot run.

    Same posture as the Exa key: degrade to the anonymous ceiling, do not raise.
    """

    transport, opener = _wired(monkeypatch, {COMMITS_URL: (200, b"{}", {})}, token=None)
    response = transport(COMMITS_URL, max_bytes=1024, timeout=1.0)

    assert response.status == 200
    assert opener.authorization_for(COMMITS_URL) is None
    assert transport.github_credential_mode() == "anonymous"


def test_the_credential_is_never_offered_to_a_non_api_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw content needs no credential, and every extra host is another escape route."""

    transport, opener = _wired(monkeypatch, {RAW_URL: (200, b"code", {})}, token=_SENTINEL)
    transport(RAW_URL, max_bytes=1024, timeout=1.0)

    assert opener.authorization_for(RAW_URL) is None


def test_the_credential_is_dropped_when_a_redirect_leaves_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attaching the header once per call would forward it through an allowlisted hop."""

    transport, opener = _wired(
        monkeypatch,
        {
            COMMITS_URL: (302, b"", {"Location": RAW_URL}),
            RAW_URL: (200, b"code", {}),
        },
        token=_SENTINEL,
    )
    transport(COMMITS_URL, max_bytes=1024, timeout=1.0)

    assert opener.authorization_for(COMMITS_URL) == f"Bearer {_SENTINEL}"
    assert opener.authorization_for(RAW_URL) is None


def test_the_transport_resolves_its_credential_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gh auth token`` is a subprocess; a pack resolving many refs pays for one."""

    calls: list[int] = []

    def resolver() -> Optional[str]:
        calls.append(1)
        return _SENTINEL

    transport = UrllibTransport(token_resolver=resolver)
    monkeypatch.setattr(transport, "_opener", _RecordingOpener({COMMITS_URL: (200, b"{}", {})}))
    transport(COMMITS_URL, max_bytes=1024, timeout=1.0)
    transport(COMMITS_URL, max_bytes=1024, timeout=1.0)

    assert calls == [1]


def test_no_forge_request_means_no_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolution is lazy, so a pack that never touches the API spawns no subprocess."""

    calls: list[int] = []

    def resolver() -> Optional[str]:
        calls.append(1)
        return None

    transport = UrllibTransport(token_resolver=resolver)
    monkeypatch.setattr(transport, "_opener", _RecordingOpener({RAW_URL: (200, b"code", {})}))
    transport(RAW_URL, max_bytes=1024, timeout=1.0)

    assert calls == []


# -- credential resolution order --------------------------------------------


def test_gh_token_wins_over_github_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gh``'s own precedence, so the broker and the debugging CLI never disagree."""

    monkeypatch.setenv("GH_TOKEN", _SENTINEL)
    monkeypatch.setenv("GITHUB_TOKEN", "other-token")

    assert resolve_github_token() == _SENTINEL


def test_github_token_is_used_when_gh_token_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CI-conventional name still works on its own."""

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", _SENTINEL)

    assert resolve_github_token() == _SENTINEL


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_a_blank_variable_is_treated_as_absent(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    """An empty export must not send ``Authorization: Bearer``."""

    monkeypatch.setenv("GH_TOKEN", blank)
    monkeypatch.setenv("GITHUB_TOKEN", blank)
    monkeypatch.setattr(
        "menagerie.crawler.source_broker.subprocess.run",
        lambda *args, **kwargs: _completed(1, ""),
    )

    assert resolve_github_token() is None


class _Completed:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def _completed(returncode: int, stdout: str) -> _Completed:
    return _Completed(returncode, stdout)


def test_gh_auth_token_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator is already logged in locally; use it rather than demanding an export."""

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    invocations: list[tuple] = []

    def fake_run(argv, **kwargs):
        invocations.append(tuple(argv))
        return _completed(0, _SENTINEL + "\n")

    monkeypatch.setattr("menagerie.crawler.source_broker.subprocess.run", fake_run)

    assert resolve_github_token() == _SENTINEL
    assert invocations == [("gh", "auth", "token")]


@pytest.mark.parametrize(
    "failure",
    [
        FileNotFoundError("gh"),
        OSError("boom"),
    ],
)
def test_a_missing_gh_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """``gh`` absent or unrunnable degrades to anonymous; it never raises."""

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fake_run(argv, **kwargs):
        raise failure

    monkeypatch.setattr("menagerie.crawler.source_broker.subprocess.run", fake_run)

    assert resolve_github_token() is None


def test_a_failing_gh_never_records_its_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gh`` failure text is not diagnostic enough to risk a credential fragment."""

    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "menagerie.crawler.source_broker.subprocess.run",
        lambda *args, **kwargs: _completed(1, ""),
    )

    assert resolve_github_token() is None


# -- 4. the credential never reaches a durable artifact ---------------------


def test_a_sentinel_credential_never_appears_in_any_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole persisted pack is scanned: receipts, manifest rows, outcomes, evidence.

    This repository is public and receipts are durable, so a leak here could not be
    unpublished. The token is attached as a request header and passed to nothing that
    builds a record -- this asserts that stays true.
    """

    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.setenv(name, _SENTINEL)
    responses = {
        COMMITS_URL: (200, json.dumps({"sha": RESOLVED_SHA}).encode(), EXHAUSTED_HEADERS),
        RAW_URL: (200, b"class SlimFlow: pass\n", {}),
    }
    pack = broker_source_pack(
        [_descriptor()], broker_dir=tmp_path, transport=HeaderTransport(responses)
    )
    path = write_broker_outputs(pack, tmp_path)

    assert _SENTINEL not in path.read_text(encoding="utf-8")
    assert _SENTINEL not in json.dumps(pack.to_dict())
    for row in pack.rows:
        assert _SENTINEL not in json.dumps(row)
    for outcome in pack.outcomes:
        assert _SENTINEL not in json.dumps(outcome.to_dict())
    for blob in (tmp_path / "evidence").iterdir():
        assert _SENTINEL.encode() not in blob.read_bytes()


def test_receipts_refuse_to_persist_a_credential_bearing_locator(tmp_path: Path) -> None:
    """A future change that puts the credential in a URL fails the write, loudly.

    The receipt already embeds the endpoint URL, so "authenticate by query parameter"
    is exactly one careless edit away from a permanent public leak.
    """

    responses: dict[str, tuple[int, bytes, dict[str, str]]] = {
        "https://api.github.com/repos/acme/w/commits/v1?access_token=" + _SENTINEL: (
            200,
            b"{}",
            {},
        ),
    }
    pack = broker_source_pack(
        [
            {
                "source_id": "doc-1",
                "kind": "raw-url",
                "url": "https://example.org/x?access_token=" + _SENTINEL,
                "requested_role": "documentation",
                "basis": "A locator that smuggles a credential.",
            }
        ],
        broker_dir=tmp_path,
        transport=HeaderTransport(responses),
    )

    with pytest.raises(SourceBrokerError, match="credential-bearing locator"):
        write_broker_outputs(pack, tmp_path)


def test_an_ordinary_registry_query_string_still_persists(tmp_path: Path) -> None:
    """The refusal is narrow: a normal query parameter must not fail the write."""

    responses: dict[str, tuple[int, bytes, dict[str, str]]] = {
        "https://example.org/x?id_list=1905.09791": (200, b"page", {})
    }
    pack = broker_source_pack(
        [
            {
                "source_id": "doc-1",
                "kind": "raw-url",
                "url": "https://example.org/x?id_list=1905.09791",
                "requested_role": "documentation",
                "basis": "An ordinary registry query.",
            }
        ],
        broker_dir=tmp_path,
        transport=HeaderTransport(responses),
    )

    assert write_broker_outputs(pack, tmp_path).exists()


def test_the_receipt_records_whether_we_were_authenticated(tmp_path: Path) -> None:
    """"Throttled" and "throttled because nobody wired a credential" are different facts."""

    _, outcome = _resolve({COMMITS_URL: (403, RATE_LIMIT_BODY, EXHAUSTED_HEADERS)}, tmp_path)

    assert outcome.resolver_receipt is not None
    # `HeaderTransport` does not report a mode, so the honest answer is "unknown"
    # rather than a fabricated "anonymous".
    assert outcome.resolver_receipt["credential_mode"] == "unknown"


def test_a_real_transport_reports_its_credential_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production transport answers the question the receipt asks."""

    authenticated = UrllibTransport(token_resolver=lambda: _SENTINEL)
    anonymous = UrllibTransport(token_resolver=lambda: None)

    assert authenticated.github_credential_mode() == "authenticated"
    assert anonymous.github_credential_mode() == "anonymous"
