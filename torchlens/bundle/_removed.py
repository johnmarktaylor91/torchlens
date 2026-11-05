"""The removed Bundle members and the redirect each one teaches."""

from __future__ import annotations

#: Public Bundle methods removed by the 2026-10-01 shim removal (no alias kept).
_REMOVED_BUNDLE_MEMBERS: dict[str, str] = {
    "replay": "use bundle.do(site, edit) -- do replays every member",
    "rerun": "use bundle.run(model, x) -- Bundle.rerun was renamed",
    "push": "use bundle.do(site, edit) -- do applies and replays the edit on every member",
    "attach_hooks": "use bundle.do(site, hook) -- do is the one editing verb on a bundle",
    "node": "use bundle.at(site)",
    "compare_at": "use bundle.at(site).compare()",
    "joint_metric": "call fn(bundle) directly",
    "set_capacity": "assign bundle.capacity = n",
    "traces": "use bundle.members or bundle[name]",
    "help": "print(bundle) shows the bundle card",
    "cluster": "there is no bundle clustering; compute it over bundle.map(fn)",
    "member_construction": "read bundle.history for how members entered the bundle",
    "show_diff": "use bundle.draw(graph='diff', members=(a, b), metric=...)",
    "delta_map": "use bundle.compare(a, b, metric=..., on=...) per pair, bundle.rank_changes() across members",
    "norm_delta": "use bundle.compare(baseline, member, metric='relative_l2')",
    "output_delta": "use bundle.compare(member, target, metric=...) and read the output sites",
    "aligned_pairs": "use bundle.supergraph, the structural alignment across members",
}
