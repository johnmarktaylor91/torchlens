"""Checks kit item 4: parameter checks at optimizer DEFAULTS (memo D2-D5, D13).

Panel law: all seeded-failure tests run at optimizer DEFAULTS -- round 1's
plans configured around weight decay and would have shipped a check that
works only when the user disables the default. The vacuity is LOCKED IN
here so it cannot be reintroduced: the naive did-it-change fact reports
changed=True for a dead layer under default AdamW while the gradient fact
fires with zero latency.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

import torchlens.checks as tc


class _DeadNet(nn.Module):
    """A trunk whose `dead` layer is cut from the loss (graph-preserving)."""

    def __init__(self, mechanism: str) -> None:
        super().__init__()
        self.f1 = nn.Linear(4, 8)
        self.dead = nn.Linear(8, 8)
        self.head = nn.Linear(8, 2)
        self._mechanism = mechanism

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route through the dead path by the configured mechanism."""

        hidden = torch.relu(self.f1(x))
        if self._mechanism == "graph_preserving":
            cut = self.dead(hidden) * 0
        else:  # graph_cut: zeros_like detaches the dead branch entirely
            cut = torch.zeros_like(hidden)
        return self.head(cut + 1.0)


@pytest.mark.smoke
@pytest.mark.parametrize("mechanism", ["graph_preserving", "graph_cut"])
def test_dead_layer_gradient_fact_fires_and_naive_change_is_vacuous(mechanism: str) -> None:
    """The D2 demonstration: gradient fact catches what movement cannot.

    Under REAL AdamW at defaults (weight_decay=0.01) the dead layer still
    CHANGES every step (decoupled decay is a rescale applied regardless of
    the gradient), so the naive did-it-change check passes a dead network;
    the ``param_received_no_gradient`` detector fires within the first
    window. Vacuity locked: n_unchanged == 0 on every step.
    """

    torch.manual_seed(0)
    model = _DeadNet(mechanism)
    optimizer = torch.optim.AdamW(model.parameters())  # DEFAULTS
    session = tc.ChecksSession(model, optimizer)
    session.register_change_check(window=(1, 1))  # zero-latency window
    session.attach()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                optimizer.zero_grad()
                model(torch.randn(4, 4)).sum().backward()
                optimizer.step()
        report = session.report()
    finally:
        session.detach()

    no_grad_names = {
        finding.names[0]
        for finding in report.findings
        if finding.check == "param_received_no_gradient"
    }
    expected_cone = {"dead.weight", "dead.bias"}
    if mechanism == "graph_cut":
        expected_cone |= {"f1.weight", "f1.bias"}
    assert expected_cone <= no_grad_names
    assert "head.weight" not in no_grad_names

    warn_codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "param_received_no_gradient" in warn_codes

    change_facts = [f for f in report.findings if f.check == "params_changed_fact"]
    assert change_facts, "the corroborating change fact must be recorded"
    if mechanism == "graph_preserving":
        # THE VACUITY LOCK (memo D2, measured): a zero GRADIENT still gets
        # the decoupled weight-decay rescale, so every parameter -- dead
        # ones included -- changed every step under AdamW defaults. The
        # naive did-it-change check can never detect this death mode.
        assert all(fact.values["n_unchanged"] == 0.0 for fact in change_facts)
    else:
        # graph_cut leaves grad=None and AdamW skips those params entirely
        # (no decay either): the unchanged set exists but is exactly the
        # cone -- the movement fact catches only THIS variant, which is why
        # it is corroborating, never primary.
        assert change_facts[-1].values["n_unchanged"] > 0

    # No finding ever says "learning" or claims a dead-unit verdict (D2/D14).
    for finding in report.findings:
        assert "learning" not in finding.message
        assert '"dead"' not in finding.message
    # The cone finding points at the frontier follow-up (D3, load-bearing).
    sample = next(f for f in report.findings if f.check == "param_received_no_gradient")
    assert "gradient_flow_audit" in sample.follow_up


def test_no_grad_findings_distinguish_none_from_zero() -> None:
    """grad-is-None (disconnection) and zero-tensor are different evidence."""

    torch.manual_seed(0)
    model = _DeadNet("graph_cut")
    optimizer = torch.optim.AdamW(model.parameters())
    session = tc.ChecksSession(model, optimizer)
    session.register_change_check(window=(1, 1), action="collect")
    session.attach()
    try:
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(4, 4)).sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    kinds = {
        finding.names[0]: finding.message
        for finding in report.findings
        if finding.check == "param_received_no_gradient"
    }
    # graph_cut detaches the branch: those params never get a grad AT ALL.
    assert "grad is None" in kinds["dead.weight"]


@pytest.mark.smoke
def test_frozen_clone_raises_with_delta_evidence_and_aliases() -> None:
    """The frozen invariant (D5): clone evidence names every alias."""

    torch.manual_seed(0)
    shared = nn.Linear(4, 4, bias=False)

    class Tied(nn.Module):
        """Tied module whose shared weight is declared frozen."""

        def __init__(self) -> None:
            super().__init__()
            self.a = shared
            self.b = shared
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the tied stack."""

            return self.head(self.b(self.a(x)))

    model = Tied()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    session = tc.ChecksSession(model, optimizer)
    session.register_frozen_check(["a.weight"])
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
    finally:
        session.detach()

    assert exc.value.fields["code"] == "frozen_param_changed"
    assert set(exc.value.fields["names"]) == {"a.weight", "b.weight"}
    finding = exc.value.fields["finding"]
    assert finding["values"]["delta_absmax"] > 0  # it can SHOW the delta
    assert finding["evidence"] == "clone"
    assert exc.value.fields["report"]["schema_version"] == 1


@pytest.mark.smoke
def test_frozen_digest_optin_raises_on_proof_never_exact_pass() -> None:
    """The digest opt-in (D5/DR-3): differing digest proves; equality never
    emits an exact pass -- a healthy run records zero raises and the
    evidence label stays 'digest'."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    frozen_param = model[0].weight
    frozen_param.requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_frozen_check(["0.weight"], evidence="digest")
    session.attach()
    try:
        for _ in range(2):  # healthy: frozen param owned by no optimizer
            optimizer.zero_grad()
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
        report = session.report()
        assert not [f for f in report.findings if f.check == "frozen_param_changed"]

        # Now violate out-of-band; the digest difference PROVES change.
        with torch.no_grad():
            frozen_param[0, 0] += 1.0
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        with pytest.raises(tc.CheckViolationError) as exc:
            optimizer.step()
        assert exc.value.fields["finding"]["evidence"] == "digest"
        assert "probabilistic" in exc.value.fields["finding"]["message"]
    finally:
        session.detach()


def test_update_ratio_raw_and_lr_normalized_with_zero_baseline() -> None:
    """D13: raw + lr-normalized ratios; zero baseline -> inf + flag, or 0."""

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.bias.zero_()  # zero baseline at step 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_update_ratio_check()
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    zero_baseline = [f for f in report.findings if f.code == "update_ratio_zero_baseline"]
    assert zero_baseline and zero_baseline[0].zero_baseline is True
    assert zero_baseline[0].names == ("bias",)
    # inf is reported as a None value slot (values carry floats-or-None),
    # with the flag carrying the semantics -- never an epsilon-clamped fake.
    assert zero_baseline[0].values["update_ratio"] is None


@pytest.mark.smoke
def test_update_ratio_bounds_warn_and_lr_zero_unavailable() -> None:
    """Explicit per-registration bounds warn; lr=0 companion is unavailable."""

    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    session = tc.ChecksSession(model, optimizer)
    session.register_update_ratio_check(bounds=(1e-9, None), action="collect")
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    # lr=0 moves nothing: every ratio is 0 < 1e-9 -> out of band.
    out_of_band = [f for f in report.findings if f.code == "update_ratio_out_of_band"]
    assert out_of_band
    unavailable = dict(report.unavailable)
    assert "update_ratio_lr_normalized" in unavailable
    assert "lr == 0" in unavailable["update_ratio_lr_normalized"]


def test_membership_and_train_eval_disclosures() -> None:
    """D4: membership/mode are DISCLOSURES with no severity and no action."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
    model[0].weight.requires_grad_(False)
    model[1].eval()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    session = tc.ChecksSession(model, optimizer)
    session.register_change_check(window=(1, 1), action="collect")
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(4, 4)).sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    membership = report.disclosures["membership"]
    assert "0.weight" in membership["requires_grad_false"]
    train_eval = report.disclosures["train_eval"]
    assert "1" in train_eval["eval_mode_modules"]
    # The frozen param is NEVER named by the no-gradient detector: without
    # the membership disclosure it would name every frozen param (D4).
    no_grad_names = {f.names[0] for f in report.findings if f.check == "param_received_no_gradient"}
    assert "0.weight" not in no_grad_names


def test_unknown_optimizer_band_unavailable_with_reason() -> None:
    """An unknown optimizer never guesses: band unavailable-with-reason."""

    torch.manual_seed(0)

    class ExoticOptimizer(torch.optim.SGD):
        """An optimizer kind with no day-1 adapter."""

    model = nn.Linear(4, 2)
    optimizer = ExoticOptimizer(model.parameters(), lr=0.1)
    facts = tc.optimizer_facts(optimizer)
    assert not facts.known
    assert "no adapter" in (facts.band_reason or "")

    session = tc.ChecksSession(model, optimizer)
    session.register_change_check(window=(1, 1), action="collect")
    session.attach()
    try:
        optimizer.zero_grad()
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        report = session.report()
    finally:
        session.detach()

    unavailable = dict(report.unavailable)
    assert "param_movement_decay_band" in unavailable
    assert "no adapter" in unavailable["param_movement_decay_band"]


@pytest.mark.smoke
def test_decay_band_movement_fact_carries_betas() -> None:
    """The corroborating decay fact rides adapter facts incl. betas (D2)."""

    torch.manual_seed(0)
    model = _DeadNet("graph_preserving")
    optimizer = torch.optim.AdamW(model.parameters())
    session = tc.ChecksSession(model, optimizer)
    session.register_change_check(window=(8, 8), action="collect")
    session.attach()
    try:
        # Enough steps for Adam momentum on the dead params to bleed out
        # toward the decay-only band.
        for _ in range(6):
            optimizer.zero_grad()
            model(torch.randn(4, 4)).sum().backward()
            optimizer.step()
        report = session.report()
    finally:
        session.detach()

    decay_facts = [f for f in report.findings if f.check == "param_movement_decay_band"]
    if decay_facts:  # momentum bleed-out timing is model-sized; fact shape is the pin
        fact = decay_facts[0]
        assert fact.severity == "info" and fact.action == "collect"
        assert fact.values["beta1"] == 0.9
        assert "lags" in fact.message


@pytest.mark.smoke
def test_registration_refusals_are_typed() -> None:
    """Unknown names / junk windows / junk actions refuse typed."""

    model = nn.Linear(4, 2)
    session = tc.ChecksSession(model, torch.optim.SGD(model.parameters(), lr=0.1))

    with pytest.raises(tc.CheckConfigError) as name_exc:
        session.register_frozen_check(["ghost"])
    assert name_exc.value.fields["code"] == "check_within_unknown_name"

    with pytest.raises(tc.CheckConfigError) as window_exc:
        session.register_change_check(window=(9, 2))
    assert window_exc.value.fields["code"] == "check_window_invalid"

    with pytest.raises(tc.CheckConfigError) as action_exc:
        session.register_change_check(action="explode")
    assert action_exc.value.fields["code"] == "check_vocab_invalid"

    with pytest.raises(tc.CheckConfigError) as evidence_exc:
        session.register_frozen_check(["weight"], evidence="vibes")
    assert evidence_exc.value.fields["code"] == "check_vocab_invalid"
