"""Phase 4 selector composition tests."""

from __future__ import annotations

import torch

import torchlens as tl


def test_selector_composition_in_layers_to_save() -> None:
    """Composed selectors work in capture layer selection."""

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.ReLU(),
    )
    selector = tl.in_module("1") & tl.func("relu")
    log = tl.log_forward_pass(model, torch.ones(1, 2), layers_to_save=selector)
    saved = [layer for layer in log.layer_list if layer.has_saved_activations]
    assert saved
    assert any(layer.func_name == "relu" for layer in saved)


def test_selector_composition_and_negation_in_resolve_sites() -> None:
    """Composed selectors work after capture resolution."""

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    log = tl.log_forward_pass(model, torch.ones(1, 2))
    relu_sites = log.resolve_sites(tl.func("relu") & ~tl.contains("linear"), max_fanout=100)
    assert relu_sites
    assert all(site.func_name == "relu" for site in relu_sites)
