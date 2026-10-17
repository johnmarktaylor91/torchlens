"""Uploads to Neuronpedia through its documented API, with the standard library only.

The endpoints are those of Neuronpedia's API reference
(https://www.neuronpedia.org/api-doc), generated from the route files of
https://github.com/hijohnnylin/neuronpedia as of commit
2f0bb10893b9c63c409dbc49db61f557883e0a2c: "Upload Graph 1/2 - Get Pre-Signed
URL" (``POST /api/graph/signed-put``), the upload of the file to the URL it
returns, "Upload Graph 2/2 - Save Graph Metadata to Database"
(``POST /api/graph/save-to-db``), and "Create Vector"
(``POST /api/vector/new``). Every request is authorised by the caller's own
key in the ``x-api-key`` header; the key is never written into a message,
``repr`` or log. Every argument is checked before the first request.
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..._io._json import loads_bounded, read_bytes_bounded
from .._result import AttributionError
from ._neuronpedia import check_contract

__all__ = ["upload_to_neuronpedia", "upload_vector"]

_BASE_URL = "https://www.neuronpedia.org"
_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."
_KEY_HEADER = "x-api-key"
#: The server's own size limits for a graph file (its validator, not the looser doc range).
_MIN_GRAPH_BYTES = 1024
_MAX_GRAPH_BYTES = 100 * 1024 * 1024
#: Characters the server accepts in a graph's slug and scan.
_NAME = re.compile(r"[A-Za-z0-9_.-]+")
#: The vector endpoint's documented limits (the steer range is the doc's, inside the server's).
_MAX_VECTOR_LENGTH = 24576
_STEER_STRENGTH_RANGE = (-100.0, 100.0)
_HOOK_TYPES = (("resid_pre", "hook_resid_pre"), ("resid_post", "hook_resid_post"))
_TIMEOUT_SECONDS = 600.0
_MESSAGE_EXCERPT = 300


def _upload_refusal(message: str, *, code: str) -> AttributionError:
    """Build one coded upload refusal ending in its docs pointer."""

    return AttributionError(f"{message} {_DOCS_POINTER}", code=code)


@dataclass(frozen=True)
class _Request:
    """One HTTP request; its ``repr`` shows no header values and no query string."""

    method: str
    url: str
    headers: dict[str, str] = field(repr=False)
    body: bytes = field(repr=False)

    def __repr__(self) -> str:
        """Method and address without the query string."""

        return f"_Request({self.method} {_address(self.url)}, {len(self.body)} bytes)"


@dataclass(frozen=True)
class _Response:
    """One HTTP response: its status and body."""

    status: int
    body: bytes


class _UrllibTransport:
    """Send requests with ``urllib.request``; the one transport that reaches the network."""

    def send(self, request: _Request) -> _Response:
        """Send ``request`` and return the response, a non-2xx status included.

        Raises
        ------
        AttributionError
            ``neuronpedia_upload_http_error`` when the host cannot be reached.
        """

        prepared = urllib.request.Request(
            request.url, data=request.body, headers=request.headers, method=request.method
        )
        try:
            with urllib.request.urlopen(prepared, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 -- https only, checked by _https
                return _Response(int(response.status), response.read())
        except urllib.error.HTTPError as error:
            return _Response(int(error.code), error.read())
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise _upload_refusal(
                f"{request.method} {_address(request.url)} could not reach the host "
                f"({type(error).__name__}: {reason}). Remedy: check the network connection "
                "and retry the same call.",
                code="neuronpedia_upload_http_error",
            ) from None


def _make_transport() -> Any:
    """The transport every upload uses (tests replace this function)."""

    return _UrllibTransport()


def _address(url: str) -> str:
    """``scheme://host/path`` of ``url``, without the query string that may sign it."""

    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}{parts.path}"


def _check_key(api_key: Any) -> str:
    """The caller's key, refused (without echoing it) when missing or empty."""

    if not isinstance(api_key, str) or not api_key.strip():
        kind = "an empty string" if isinstance(api_key, str) else type(api_key).__name__
        raise _upload_refusal(
            f"api_key is {kind}; every Neuronpedia upload needs your own API key. "
            'Remedy: pass api_key=os.environ["NEURONPEDIA_API_KEY"] (the key is under '
            "Settings on neuronpedia.org).",
            code="neuronpedia_upload_argument_invalid",
        )
    return api_key


def _redact(text: str, api_key: str, url: str) -> str:
    """``text`` without the key or the URL's query values (a signature), cut short."""

    text = text.replace(api_key, "<api key>")
    for _name, value in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query):
        if value:
            text = text.replace(value, "<redacted>")
    return text[:_MESSAGE_EXCERPT]


def _send(transport: Any, request: _Request, *, step: str, api_key: str) -> bytes:
    """Send one request and return its body, refusing a non-2xx answer."""

    response = transport.send(request)
    if not 200 <= response.status < 300:
        detail = _redact(response.body.decode("utf-8", errors="replace"), api_key, request.url)
        raise _upload_refusal(
            f"Neuronpedia answered HTTP {response.status} to {step} "
            f"({request.method} {_address(request.url)}): {detail!r}. Remedy: check the key "
            "(401 or 403), the scan id and slug (400), or wait and retry the same call (429).",
            code="neuronpedia_upload_http_error",
        )
    return response.body


def _post_json(transport: Any, path: str, payload: dict, *, step: str, api_key: str) -> dict:
    """POST ``payload`` as JSON with the key and return the JSON answer."""

    request = _Request(
        "POST",
        f"{_BASE_URL}{path}",
        {"Content-Type": "application/json", _KEY_HEADER: api_key, "User-Agent": "torchlens"},
        json.dumps(payload, allow_nan=False).encode("utf-8"),
    )
    body = _send(transport, request, step=step, api_key=api_key)
    try:
        answer = loads_bounded(body.decode("utf-8"))
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise _upload_refusal(
            f"Neuronpedia's answer to {step} is not a JSON object. Remedy: retry the same "
            "call; if it persists, the API has changed from the documented one.",
            code="neuronpedia_upload_response_invalid",
        )
    return answer


def _field(answer: dict, name: str, *, step: str) -> str:
    """One string field of an answer, refused when absent."""

    value = answer.get(name)
    if not isinstance(value, str) or not value:
        raise _upload_refusal(
            f"Neuronpedia's answer to {step} has no {name!r} string. Remedy: retry the same "
            "call; if it persists, the API has changed from the documented one.",
            code="neuronpedia_upload_response_invalid",
        )
    return value


def _https(url: str, *, step: str) -> str:
    """Refuse to send the graph anywhere but an https address."""

    if urllib.parse.urlsplit(url).scheme != "https":
        raise _upload_refusal(
            f"Neuronpedia's answer to {step} gave an upload address that is not https. "
            "Remedy: retry the same call; the graph is only sent over https.",
            code="neuronpedia_upload_response_invalid",
        )
    return url


def _graph_bytes(exported: Any) -> tuple[bytes, Any]:
    """The bytes to upload and the parsed graph, from a dict or a JSON file path."""

    if isinstance(exported, dict):
        try:
            return json.dumps(exported, allow_nan=False).encode("utf-8"), exported
        except (TypeError, ValueError) as error:
            raise _upload_refusal(
                f"The graph dict cannot be written as JSON ({error}). Remedy: pass the dict "
                "AttributionGraph.to_neuronpedia(...) returned, unchanged.",
                code="attribution_graph_contract_invalid",
            ) from None
    if not isinstance(exported, (str, Path)):
        raise _upload_refusal(
            f"The graph is a {type(exported).__name__}; the upload takes the dict "
            "to_neuronpedia returns or the path of the file it wrote. Remedy: "
            "tl.attribution.graph.upload_to_neuronpedia(graph.to_neuronpedia(...), "
            "api_key=os.environ['NEURONPEDIA_API_KEY']).",
            code="neuronpedia_upload_argument_invalid",
        )
    try:
        body = read_bytes_bounded(Path(exported))
        return body, loads_bounded(body.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise _upload_refusal(
            f"The graph file {str(exported)!r} cannot be read as JSON ({type(error).__name__}). "
            "Remedy: write it with graph.to_neuronpedia(..., path=path) and pass that path.",
            code="neuronpedia_upload_argument_invalid",
        ) from None


def _check_upload_graph(graph: dict, size: int) -> str:
    """What the server refuses after the upload, refused before it; returns the slug."""

    check_contract(graph)
    meta = graph["metadata"]
    for name in ("slug", "scan"):
        if _NAME.fullmatch(meta[name]) is None:
            raise _upload_refusal(
                f"metadata.{name}={meta[name]!r} holds characters Neuronpedia refuses (it "
                "takes letters, digits, '_', '.' and '-'). Remedy: export again with "
                f'graph.to_neuronpedia({name}="my-graph-1", ...).',
                code="neuronpedia_upload_argument_invalid",
            )
    if not _MIN_GRAPH_BYTES <= size <= _MAX_GRAPH_BYTES:
        raise _upload_refusal(
            f"The graph JSON is {size:,} bytes; Neuronpedia takes {_MIN_GRAPH_BYTES:,} to "
            f"{_MAX_GRAPH_BYTES:,}. Remedy: export with stricter pruning for a smaller file, "
            "graph.to_neuronpedia(..., node_threshold=0.6, edge_threshold=0.9).",
            code="neuronpedia_upload_argument_invalid",
        )
    return str(meta["slug"])


def upload_to_neuronpedia(exported: dict[str, Any] | str | Path, *, api_key: str) -> str:
    """Upload an exported attribution graph to Neuronpedia and return its page's URL.

    Documented-unstable. This publishes the graph on neuronpedia.org under
    the account that owns ``api_key``. It follows Neuronpedia's documented
    two-step upload (https://www.neuronpedia.org/api-doc, as of commit
    2f0bb10893b9c63c409dbc49db61f557883e0a2c of
    https://github.com/hijohnnylin/neuronpedia): ``POST
    /api/graph/signed-put`` returns a signed upload address, the JSON is
    sent there with ``PUT``, and ``POST /api/graph/save-to-db`` saves it
    under its ``scan`` and ``slug``. Standard library only.

    Before any request the graph is checked against the export's structural
    contract (the required keys and value types of Neuronpedia's published
    schema), its ``slug`` and ``scan`` against the characters the server
    takes (letters, digits, ``_``, ``.``, ``-``), and its size against the
    server's 1,024 to 104,857,600 bytes.

    Parameters
    ----------
    exported
        The dict :meth:`AttributionGraph.to_neuronpedia` returned, or the path
        of the file it wrote with ``path=``.
    api_key
        Your Neuronpedia API key (under Settings on neuronpedia.org), sent in
        the ``x-api-key`` header and never written into a message.

    Returns
    -------
    str
        The URL of the graph's page on Neuronpedia.

    Raises
    ------
    AttributionError
        ``attribution_graph_contract_invalid`` for JSON that does not meet the
        contract; ``neuronpedia_upload_argument_invalid`` for a missing key,
        an unreadable file, a slug or scan the server refuses, or a size
        outside its limits; ``neuronpedia_upload_http_error`` for a non-2xx
        answer or an unreachable host; ``neuronpedia_upload_response_invalid``
        for an answer without its documented fields.
    """

    key = _check_key(api_key)
    body, graph = _graph_bytes(exported)
    slug = _check_upload_graph(graph, len(body))
    transport = _make_transport()
    step = "the signed-URL request (Upload Graph 1/2)"
    signed = _post_json(
        transport,
        "/api/graph/signed-put",
        {"filename": f"{slug}.json", "contentLength": len(body), "contentType": "application/json"},
        step=step,
        api_key=key,
    )
    put_url = _https(_field(signed, "url", step=step), step=step)
    put_request_id = _field(signed, "putRequestId", step=step)
    _send(
        transport,
        _Request(
            "PUT",
            put_url,
            {"Content-Type": "application/json", "Content-Length": str(len(body))},
            body,
        ),
        step="the file upload to the signed URL",
        api_key=key,
    )
    step = "the save request (Upload Graph 2/2)"
    saved = _post_json(
        transport,
        "/api/graph/save-to-db",
        {"putRequestId": put_request_id},
        step=step,
        api_key=key,
    )
    return _field(saved, "url", step=step)


def _vector_values(vector: Any) -> list[float]:
    """The vector as a list of finite floats within the endpoint's length limit."""

    if not isinstance(vector, torch.Tensor) or vector.dim() != 1 or vector.numel() == 0:
        raise _upload_refusal(
            "The direction's vector is not a non-empty 1-D tensor. Remedy: upload a "
            "Direction fitted by tl.intervention.direction_from_pairs(...).",
            code="neuronpedia_upload_argument_invalid",
        )
    if vector.numel() > _MAX_VECTOR_LENGTH:
        raise _upload_refusal(
            f"The vector has {vector.numel():,} values; Neuronpedia takes at most "
            f"{_MAX_VECTOR_LENGTH:,}. Remedy: upload a direction read at a site of the "
            "model's residual width.",
            code="neuronpedia_upload_argument_invalid",
        )
    values = [float(v) for v in vector.detach().cpu().double().tolist()]
    if not all(math.isfinite(v) for v in values):
        raise _upload_refusal(
            "The vector holds NaN or infinite values, which JSON cannot carry. Remedy: refit "
            "the direction on finite rows, tl.intervention.direction_from_pairs(...).",
            code="neuronpedia_upload_argument_invalid",
        )
    return values


def _check_vector_settings(  # noqa: PLR0913 -- the endpoint's fields, checked together
    scan: Any, layer: Any, hook_type: Any, vector_label: Any, steer_strength: Any
) -> None:
    """Refuse vector fields the endpoint documents as invalid."""

    for name, value in (("scan", scan), ("vector_label", vector_label)):
        if not isinstance(value, str) or not value.strip():
            raise _upload_refusal(
                f"{name}={value!r} is not a non-empty string. Remedy: "
                'direction.upload_to_neuronpedia(api_key=os.environ["NEURONPEDIA_API_KEY"], '
                'scan="gemma-2-2b", vector_label="honesty").',
                code="neuronpedia_upload_argument_invalid",
            )
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise _upload_refusal(
            f"layer={layer!r} is not a layer index of 0 or more. Remedy: pass layer=3 "
            "(the decoder block the vector belongs to).",
            code="neuronpedia_upload_argument_invalid",
        )
    if hook_type not in dict(_HOOK_TYPES):
        raise _upload_refusal(
            f"hook_type={hook_type!r} is not one of {[name for name, _ in _HOOK_TYPES]}. Remedy: "
            'pass hook_type="resid_post" for a decoder block output.',
            code="neuronpedia_upload_argument_invalid",
        )
    low, high = _STEER_STRENGTH_RANGE
    if (
        isinstance(steer_strength, bool)
        or not isinstance(steer_strength, (int, float))
        or not low <= steer_strength <= high
    ):
        raise _upload_refusal(
            f"steer_strength={steer_strength!r} is not a number from {low:g} to {high:g}. "
            "Remedy: pass steer_strength=1.0.",
            code="neuronpedia_upload_argument_invalid",
        )


def upload_vector(  # noqa: PLR0913 -- the endpoint's fields, keyword-only
    vector: torch.Tensor,
    *,
    api_key: str,
    scan: str,
    layer: int,
    hook_type: str,
    vector_label: str,
    steer_strength: float,
) -> str:
    """Implement :meth:`torchlens.intervention.Direction.upload_to_neuronpedia`."""

    key = _check_key(api_key)
    _check_vector_settings(scan, layer, hook_type, vector_label, steer_strength)
    values = _vector_values(vector)
    answer = _post_json(
        _make_transport(),
        "/api/vector/new",
        {
            "modelId": scan,
            "layerNumber": layer,
            "hookType": dict(_HOOK_TYPES)[hook_type],
            "vector": values,
            "vectorDefaultSteerStrength": float(steer_strength),
            "vectorLabel": vector_label,
        },
        step="the vector request (Create Vector)",
        api_key=key,
    )
    return _field(answer, "url", step="the vector request (Create Vector)")
