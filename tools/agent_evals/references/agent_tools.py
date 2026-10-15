"""Reference solution: the agent tool surface through the Python transport.

Docs used: docs/for-ai-agents.md (The agent surface: torchlens.agent.call_tool; torchlens_overview
mode="manifest"; torchlens_dump view="full" pages op rows under max_rows, echo data.next as
continuation; Machine-readable trace dump: rows carry layer_label under "ops").
"""

import json

from model import build_model, make_input

import torchlens as tl
from torchlens.agent import call_tool

tl.save(tl.trace(build_model(), make_input(), save="all"), "run.tlspec")
manifest = call_tool("torchlens_overview", {"path": "run.tlspec", "mode": "manifest"})

labels: list[str] = []
pages = 0
first_page_rows = None
request: dict = {"path": "run.tlspec", "view": "full", "max_rows": 2}
while True:
    page = call_tool("torchlens_dump", request)
    rows = page["data"]["ops"]
    pages += 1
    if first_page_rows is None:
        first_page_rows = len(rows)
    labels.extend(row["layer_label"] for row in rows)
    if not page["data"].get("next"):
        break
    request = {**request, "continuation": page["data"]["next"]}

with open("result.json", "w", encoding="utf-8") as handle:
    json.dump(
        {
            "manifest_schema": manifest["schema"],
            "pages": pages,
            "first_page_rows": first_page_rows,
            "row_labels": labels,
        },
        handle,
    )
