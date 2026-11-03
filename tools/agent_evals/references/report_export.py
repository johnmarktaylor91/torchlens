"""Reference solution: a budgeted report and two exports.

Docs used: docs/for-ai-agents.md (Machine-readable trace dump and budgeted reports:
trace.explain(max_tokens=N)), docs/reference/export.md (tl.export.html(log, path)),
docs/reference/glossary.md (trace.ops counting: trace.ops.to_pandas()).
"""

from pathlib import Path

from model import build_model, make_input

import torchlens as tl

trace = tl.trace(build_model(), make_input(), save="all")
Path("report.txt").write_text(trace.explain(max_tokens=300), encoding="utf-8")
tl.export.html(trace, Path("trace.html"))
trace.ops.to_pandas().to_csv("ops.csv", index=False)
