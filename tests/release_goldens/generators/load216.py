# ruff: noqa -- harvest-time provenance script, committed as run (lint header added at import)
import warnings

p = "/tmp/eco_r3/art_v2.16.0_portable"
import torchlens as tl

print("reader", tl.__version__, end=" ")
try:
    from torchlens.io import detect_tlspec_format

    print("| detect:", detect_tlspec_format(p))
except Exception as e:
    try:
        from torchlens._io.bundle import detect_tlspec_format as d2

        print("| detect(_io):", d2(p))
    except Exception:
        print("| detect unavailable:", type(e).__name__)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try:
        obj = tl.load(p)
        print("  RESULT LOADED", type(obj).__name__, "n=", len(obj))
    except Exception as e:
        code = getattr(e, "code", None) or (getattr(e, "fields", {}) or {}).get("code")
        print("  RESULT REFUSED", type(e).__name__, "code=", code)
        print("   msg:", str(e).replace("\n", " ")[:300])
    print("   warnings:", sorted({x.category.__name__ for x in w}))
