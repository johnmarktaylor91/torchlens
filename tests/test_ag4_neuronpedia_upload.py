"""Uploads to Neuronpedia, tested offline against a recording fake transport.

Nothing here contacts neuronpedia.org: every test replaces the upload
module's transport factory with a fake that records each request and
answers from a script, and an autouse guard fails any test that would build
the real urllib transport. The request shapes are those of Neuronpedia's
documented API (https://www.neuronpedia.org/api-doc): ``POST
/api/graph/signed-put``, the ``PUT`` to the signed URL, ``POST
/api/graph/save-to-db``, and ``POST /api/vector/new``, all authorised by
the ``x-api-key`` header except the signed ``PUT``.

Pre-registered bars: the documented sequence and bodies (U1); zero requests
for any input refused before sending (U2); coded refusals for non-2xx
answers, an unreachable host and answers without their fields (U3); the key
and the signed URL's signature in no message, repr or log (U4); the real
transport never built under the suite (U5).
"""

from __future__ import annotations

import io
import json
import logging
import re
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

import torchlens as tl
from tests import _ag1_fixtures as fx
from torchlens.attribution.graph import _neuronpedia_upload as up
from torchlens.selection_subspace import BasisProvenance

KEY = "not-a-real-key"
BASE = "https://www.neuronpedia.org"
SIGNED = "https://bucket.s3.amazonaws.com/graphs/u/fixture-1.json?X-Amz-Signature=deadbeefcafe0123"
DOC = Path(__file__).resolve().parents[1] / "docs" / "reference" / "attribution.md"


class _Tokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(f"<{i}>" for i in ids)


@dataclass
class FakeTransport:
    """Records every request and answers from ``script`` in order."""

    script: list[object] = field(default_factory=list)
    sent: list[up._Request] = field(default_factory=list)

    def send(self, request: up._Request) -> up._Response:
        self.sent.append(request)
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        status, body = answer
        return up._Response(status, body if isinstance(body, bytes) else json.dumps(body).encode())


class _RealTransportBuilt(AssertionError):
    pass


_REAL_TRANSPORT = up._UrllibTransport


@pytest.fixture(autouse=True)
def _never_live(monkeypatch) -> None:
    """Fail any test that would build the real transport or reach a real ``urlopen``."""

    def refuse(*_args, **_kwargs):
        raise _RealTransportBuilt("the real urllib transport was built under the test suite")

    monkeypatch.setattr(up, "_UrllibTransport", refuse)
    monkeypatch.setattr(up.urllib.request, "urlopen", refuse)


@pytest.fixture
def fake(monkeypatch) -> FakeTransport:
    transport = FakeTransport()
    monkeypatch.setattr(up, "_make_transport", lambda: transport)
    return transport


def _graph_answers() -> list[object]:
    return [
        (200, {"url": SIGNED, "key": "graphs/u/fixture-1.json", "putRequestId": "put-42"}),
        (200, b""),
        (200, {"message": "Graph saved to database", "url": f"{BASE}/tiny/graph?slug=fixture"}),
    ]


@pytest.fixture(scope="module")
def exported() -> Iterator[dict]:
    model = fx.build_model(2)
    graph = tl.attribution.build_graph(
        tl.trace(model, torch.tensor([fx.TOKENS])),
        fx.exact_transcoders(model),
        embedding="embed",
        frozen=fx.frozen_names(2),
        max_n_logits=2,
        desired_logit_prob=0.99,
    )
    built = {"graph": graph.to_neuronpedia(slug="fixture", scan="tiny", tokenizer=_Tokenizer())}
    try:
        yield built["graph"]
    finally:
        built.clear()


def _copy(value: dict) -> dict:
    return json.loads(json.dumps(value))


def _direction(dim: int = 4, site: str = "model.layers.3") -> tl.intervention.Direction:
    vector = torch.randn(dim, generator=torch.Generator().manual_seed(0))
    provenance = BasisProvenance(
        origin="direction",
        method="mean_diff",
        sha256="0" * 64,
        num_vectors=1,
        space_dim=dim,
        dim=1,
        atol=0.0,
    )
    return tl.intervention.Direction(vector, site, "mean_diff", 3, -1, None, provenance)


# ---------------------------------------------------------------- graph upload


def test_graph_upload_sends_the_documented_sequence(fake, exported) -> None:
    fake.script = _graph_answers()
    url = tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
    assert url == f"{BASE}/tiny/graph?slug=fixture"
    body = json.dumps(exported, allow_nan=False).encode()
    first, second, third = fake.sent
    assert (first.method, first.url) == ("POST", f"{BASE}/api/graph/signed-put")
    assert first.headers["x-api-key"] == KEY
    assert first.headers["Content-Type"] == "application/json"
    assert json.loads(first.body) == {
        "filename": "fixture.json",
        "contentLength": len(body),
        "contentType": "application/json",
    }
    assert (second.method, second.url, second.body) == ("PUT", SIGNED, body)
    assert second.headers == {"Content-Type": "application/json", "Content-Length": str(len(body))}
    assert (third.method, third.url) == ("POST", f"{BASE}/api/graph/save-to-db")
    assert third.headers["x-api-key"] == KEY
    assert json.loads(third.body) == {"putRequestId": "put-42"}


def test_graph_upload_accepts_a_path(fake, exported, tmp_path) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(exported, indent=1), encoding="utf-8")
    fake.script = _graph_answers()
    tl.attribution.graph.upload_to_neuronpedia(path, api_key=KEY)
    assert fake.sent[1].body == path.read_bytes()
    assert json.loads(fake.sent[0].body)["contentLength"] == path.stat().st_size


def _missing_metadata(graph: dict) -> dict:
    graph = _copy(graph)
    del graph["metadata"]
    return graph


def _dangling_link(graph: dict) -> dict:
    graph = _copy(graph)
    graph["links"][0]["source"] = "nowhere"
    return graph


def _bad_slug(graph: dict) -> dict:
    graph = _copy(graph)
    graph["metadata"]["slug"] = "my graph/1"
    return graph


def _too_small(graph: dict) -> dict:
    graph = _copy(graph)
    graph["nodes"], graph["links"] = [], []
    graph["metadata"]["prompt_tokens"] = []
    return graph


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (_missing_metadata, "attribution_graph_contract_invalid"),
        (_dangling_link, "attribution_graph_contract_invalid"),
        (_bad_slug, "neuronpedia_upload_argument_invalid"),
        (_too_small, "neuronpedia_upload_argument_invalid"),
    ],
    ids=["missing_metadata", "dangling_link", "bad_slug", "too_small"],
)
def test_graph_upload_refuses_an_invalid_graph_before_any_request(
    fake, exported, change, code
) -> None:
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.graph.upload_to_neuronpedia(change(exported), api_key=KEY)
    assert excinfo.value.fields["code"] == code
    assert "Remedy:" in str(excinfo.value)
    assert fake.sent == []


def test_graph_upload_refuses_a_file_over_the_size_cap_before_any_request(
    fake, exported, monkeypatch
) -> None:
    monkeypatch.setattr(up, "_MAX_GRAPH_BYTES", 2048)
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
    assert excinfo.value.fields["code"] == "neuronpedia_upload_argument_invalid"
    assert "2,048" in str(excinfo.value)
    assert fake.sent == []


@pytest.mark.parametrize("kind", ["graph", "vector"])
def test_uploads_refuse_a_missing_api_key(fake, exported, kind) -> None:
    for key in (None, "", "   "):
        with pytest.raises(tl.attribution.AttributionError) as excinfo:
            if kind == "graph":
                tl.attribution.graph.upload_to_neuronpedia(exported, api_key=key)
            else:
                _direction().upload_to_neuronpedia(api_key=key, scan="tiny", vector_label="v")
        assert excinfo.value.fields["code"] == "neuronpedia_upload_argument_invalid"
        assert "Remedy:" in str(excinfo.value)
    assert fake.sent == []


@pytest.mark.parametrize(
    ("script", "step"),
    [
        ([(401, {"error": "unauthorized"})], "signed-URL request"),
        ([_graph_answers()[0], (403, b"<Error>AccessDenied</Error>")], "file upload"),
        ([*_graph_answers()[:2], (400, {"error": "ERR_MODEL_DOES_NOT_EXIST"})], "save request"),
        (None, "could not reach the host"),
    ],
    ids=["signed_put", "s3_put", "save_to_db", "network"],
)
def test_graph_upload_http_errors(fake, exported, monkeypatch, script, step) -> None:
    if script is None:
        # The real transport's own mapping, its urlopen stubbed (the autouse guard
        # already replaced the real one): a refused connection and an HTTP error.
        def unreachable(request, timeout):
            raise urllib.error.URLError("no route to host")

        monkeypatch.setattr(up.urllib.request, "urlopen", unreachable)
        monkeypatch.setattr(up, "_make_transport", _REAL_TRANSPORT)
        with pytest.raises(tl.attribution.AttributionError) as excinfo:
            tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
        assert excinfo.value.fields["code"] == "neuronpedia_upload_http_error"
        assert step in str(excinfo.value) and excinfo.value.__cause__ is None

        def forbidden(request, timeout):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b"no"))

        monkeypatch.setattr(up.urllib.request, "urlopen", forbidden)
        request = up._Request("POST", f"{BASE}/api/graph/signed-put", {}, b"{}")
        assert _REAL_TRANSPORT().send(request) == up._Response(403, b"no")
        return
    fake.script = list(script)
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
    assert excinfo.value.fields["code"] == "neuronpedia_upload_http_error"
    assert step in str(excinfo.value)
    assert len(fake.sent) == len(script)


def test_graph_upload_refuses_a_response_without_its_fields(fake, exported) -> None:
    for answer in ({"url": SIGNED}, {"putRequestId": "p", "url": "http://plain/x.json"}, b"[]"):
        fake.script, fake.sent = [(200, answer)], []
        with pytest.raises(tl.attribution.AttributionError) as excinfo:
            tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
        assert excinfo.value.fields["code"] == "neuronpedia_upload_response_invalid"
        assert len(fake.sent) == 1


def test_api_key_and_signature_never_appear_in_errors_or_logs(fake, exported, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    signature = re.search(r"X-Amz-Signature=(\w+)", SIGNED).group(1)
    fake.script = [
        _graph_answers()[0],
        (403, f"<Error>SignatureDoesNotMatch {signature} key={KEY}</Error>".encode()),
    ]
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
    error = excinfo.value
    texts = [str(error), repr(error), str(error.args), repr(error.fields), caplog.text]
    texts += [repr(request) for request in fake.sent]
    for text in texts:
        assert KEY not in text and signature not in text
    assert error.__cause__ is None
    fake.script, fake.sent = [(500, f"echo {KEY}".encode())], []
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        _direction().upload_to_neuronpedia(api_key=KEY, scan="tiny", vector_label="v")
    assert KEY not in str(excinfo.value) and "<api key>" in str(excinfo.value)


# ---------------------------------------------------------------- vector upload


def test_vector_upload_sends_the_documented_body(fake) -> None:
    direction = _direction()
    fake.script = [(200, {"message": "Success", "url": f"{BASE}/tiny/3-neuronpedia-resid-post/1"})]
    url = direction.upload_to_neuronpedia(
        api_key=KEY, scan="tiny", vector_label="honesty", steer_strength=4.0
    )
    assert url == f"{BASE}/tiny/3-neuronpedia-resid-post/1"
    (request,) = fake.sent
    assert (request.method, request.url) == ("POST", f"{BASE}/api/vector/new")
    assert request.headers["x-api-key"] == KEY
    assert json.loads(request.body) == {
        "modelId": "tiny",
        "layerNumber": 3,
        "hookType": "hook_resid_post",
        "vector": direction.vector.double().tolist(),
        "vectorDefaultSteerStrength": 4.0,
        "vectorLabel": "honesty",
    }
    fake.script = [(200, {"url": f"{BASE}/x"})]
    _direction(site="encoder").upload_to_neuronpedia(
        api_key=KEY, scan="tiny", vector_label="v", layer=0, hook_type="resid_pre"
    )
    assert json.loads(fake.sent[1].body)["hookType"] == "hook_resid_pre"
    assert json.loads(fake.sent[1].body)["layerNumber"] == 0


def _with_vector(vector: torch.Tensor) -> tl.intervention.Direction:
    import dataclasses

    return dataclasses.replace(_direction(), vector=vector)


@pytest.mark.parametrize(
    ("direction", "settings"),
    [
        (lambda: _with_vector(torch.tensor([1.0, float("nan")])), {}),
        (lambda: _with_vector(torch.zeros(24577)), {}),
        (_direction, {"hook_type": "attn_out"}),
        (_direction, {"steer_strength": 250.0}),
        (_direction, {"vector_label": ""}),
    ],
    ids=["non_finite", "too_long", "hook_type", "steer_strength", "vector_label"],
)
def test_vector_upload_refuses_invalid_bodies_before_any_request(fake, direction, settings) -> None:
    arguments = {"api_key": KEY, "scan": "tiny", "vector_label": "v", **settings}
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        direction().upload_to_neuronpedia(**arguments)
    assert excinfo.value.fields["code"] == "neuronpedia_upload_argument_invalid"
    assert "Remedy:" in str(excinfo.value)
    assert fake.sent == []


def test_vector_upload_http_error(fake) -> None:
    fake.script = [(400, {"message": "Vector dimension does not match model dimension"})]
    with pytest.raises(tl.attribution.AttributionError) as excinfo:
        _direction().upload_to_neuronpedia(api_key=KEY, scan="tiny", vector_label="v")
    assert excinfo.value.fields["code"] == "neuronpedia_upload_http_error"
    assert "Vector dimension does not match" in str(excinfo.value)


# ---------------------------------------------------------------- never live


def test_default_transport_is_never_built_under_tests(exported) -> None:
    """Without a fake, the upload reaches the guarded real transport and nothing else.

    The guard replaces both the transport class and ``urllib.request.urlopen``
    for every test in this module; the one case that builds the real class
    (``test_graph_upload_http_errors[network]``) stubs ``urlopen`` first.
    """

    assert up._make_transport.__module__ == up.__name__
    with pytest.raises(_RealTransportBuilt):
        tl.attribution.graph.upload_to_neuronpedia(exported, api_key=KEY)
    with pytest.raises(_RealTransportBuilt):
        _direction().upload_to_neuronpedia(api_key=KEY, scan="tiny", vector_label="v")


def test_docs_upload_example_runs_against_the_fake_transport(fake, tmp_path, monkeypatch) -> None:
    text = DOC.read_text(encoding="utf-8")
    python_fences = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
    (graph_fence,) = [f for f in python_fences if "graph.to_neuronpedia(" in f]
    assert not [f for f in python_fences if "upload_to_neuronpedia(" in f], (
        "the docs runner executes every python fence: the upload example stays a text fence"
    )
    (upload_fence,) = [
        f
        for f in re.findall(r"```text\n(.*?)\n```", text, re.DOTALL)
        if "upload_to_neuronpedia(" in f
    ]
    monkeypatch.setenv("NEURONPEDIA_API_KEY", KEY)
    fake.script = _graph_answers()
    namespace: dict[str, object] = {"DOCS_TMPDIR": str(tmp_path), "__name__": "__docs__"}
    exec(compile(graph_fence, str(DOC), "exec"), namespace)  # noqa: S102 -- the doc's own example
    exec(compile(upload_fence, str(DOC), "exec"), namespace)  # noqa: S102 -- against the fake
    assert [request.method for request in fake.sent] == ["POST", "PUT", "POST"]
    assert namespace["url"] == f"{BASE}/tiny/graph?slug=fixture"
