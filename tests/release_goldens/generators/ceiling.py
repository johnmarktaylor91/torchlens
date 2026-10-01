# ruff: noqa -- harvest-time provenance script, committed as run (lint header added at import)
import torchlens as tl

print("reader", tl.__version__)
try:
    tl.load("/tmp/eco_r3/art_future9")
    print("LOADED (unexpected)")
except Exception as e:
    print(
        "REFUSED",
        type(e).__name__,
        "code=",
        getattr(e, "code", None) or (getattr(e, "fields", {}) or {}).get("code"),
    )
    print("  msg:", str(e).replace("\n", " ")[:250])
# inspect on a below-floor GENUINE artifact
for name in ("art_v2.31.0_portable", "art_future9"):
    try:
        info = tl.inspect_tlspec("/tmp/eco_r3/" + name)
        print("inspect", name, "OK ->", str(info)[:160].replace("\n", " "))
    except Exception as e:
        print("inspect", name, "FAILED", type(e).__name__, str(e)[:120])
