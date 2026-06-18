"""Registry for historical TorchLens menagerie reimplementations."""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from typing import Any, Final

CLASSIC_ZOO: Final = "classics-pytorch"

_CLASSIC_MANIFEST: Final[dict[str, tuple[str, str]]] = {
    "active_inference_mdp": ("Discrete active-inference MDP agent", "E6"),
    "art1": ("ART1 (Adaptive Resonance Theory 1)", "E2"),
    "art2": ("ART2 (analog-input ART)", "E2"),
    "artmap": ("ARTMAP / Fuzzy ARTMAP", "E2"),
    "bam": ("Bidirectional Associative Memory (BAM)", "E2"),
    "bilinear_cnn": ("Bilinear-CNN (B-CNN)", "E5"),
    "binary_cmm": ("Binary CMM (Steinbuch Lernmatrix / Willshaw)", "E1"),
    "boltzmann_machine": ("Full (unrestricted) Boltzmann Machine", "E2"),
    "brain_state_in_a_box": ("Brain-State-in-a-Box (BSB)", "E1"),
    "cascade_correlation": ("Cascade-Correlation", "E3"),
    "classic_moe": ("Classic Adaptive Mixture-of-Experts (dense)", "E3"),
    "clockwork_rnn": ("Clockwork RNN", "E5"),
    "cmac": ("CMAC (Albus)", "E1"),
    "cognitron": ("Cognitron", "E1"),
    "cohen_grossberg": ("Cohen-Grossberg competitive network", "E6"),
    "compact_bilinear_pooling": ("Compact Bilinear Pooling", "E5"),
    "conditional_rbm": ("Conditional RBM / Factored CRBM", "E4"),
    "continuous_attractor_net": ("Continuous Attractor Neural Network (ring attractor)", "E6"),
    "contrastive_hebbian": ("Contrastive Hebbian Learning network", "E6"),
    "conv_dbn": ("Convolutional Deep Belief Network (conv-RBM, prob. max-pooling)", "E4"),
    "counterpropagation": ("Counterpropagation Network (CPN)", "E2"),
    "crf_as_rnn": ("CRFasRNN", "E5"),
    "deconvnet_noh": ("DeconvNet (Noh 2015)", "E5"),
    "deep_boltzmann_machine": ("Deep Boltzmann Machine (DBM)", "E4"),
    "draw": ("DRAW", "E5"),
    "dynamic_field_theory": ("Dynamic Field Theory architecture", "E6"),
    "elastic_net_tsp": ("Elastic Net (Durbin-Willshaw)", "E3"),
    "fields_of_experts": ("Product of Experts / Fields of Experts (PoE/FoE)", "E4"),
    "fitzhugh_nagumo": ("FitzHugh-Nagumo excitable neuron", "E6"),
    "friston_free_energy": ("Friston free-energy predictive-coding net", "E6"),
    "fully_recurrent_net": ("Williams-Zipser fully-recurrent net", "E3"),
    "fuzzy_art": ("Fuzzy ART", "E2"),
    "glvq": ("Generalized LVQ (GLVQ)", "E3"),
    "gmdh": ("GMDH / Ivakhnenko polynomial network", "E1"),
    "grnn_specht": ("Generalized Regression NN (GRNN, Specht)", "E3"),
    "grossberg_shunting_field": ("Grossberg recurrent competitive / shunting field", "E6"),
    "helmholtz_machine": ("Helmholtz Machine / Wake-Sleep", "E3"),
    "hme": ("Hierarchical Mixtures of Experts (HME)", "E3"),
    "hodgkin_huxley": ("Hodgkin-Huxley neuron (differentiable)", "E6"),
    "htm": ("Hierarchical Temporal Memory (HTM / CLA)", "E4"),
    "lapgan": ("LAPGAN", "E5"),
    "leabra": ("Leabra", "E6"),
    "lenet4": ("LeNet-4 / pre-LeNet-5 CNN", "E3"),
    "linear_associative_memory": ("Linear Associative Memory / Correlation Matrix Memory", "E1"),
    "little_net": ("Little persistent-state recurrent net", "E1"),
    "lvq": ("Learning Vector Quantization (LVQ)", "E3"),
    "madaline": ("MADALINE (MRI/MRII)", "E1"),
    "mark_i_perceptron": ("Mark-I / cross-coupled Perceptron (Rosenblatt 1962)", "E1"),
    "mcculloch_pitts": ("McCulloch-Pitts threshold logic net", "E1"),
    "md_lstm": ("MD-LSTM / Pyramidal-LSTM", "E5"),
    "mean_covariance_rbm": ("Mean-Covariance RBM (mcRBM / mPoT / ssRBM)", "E4"),
    "morris_lecar": ("Morris-Lecar neuron", "E6"),
    "mumford_cortical": ("Mumford predictive cortical architecture", "E6"),
    "nef": ("Neural Engineering Framework (NEF)", "E6"),
    "nettalk": ("NETtalk", "E2"),
    "neural_gpu": ("Neural GPU", "E5"),
    "olam": ("Optimal Linear Associative Memory (OLAM)", "E1"),
    "orig_lstm_1997": ("Original LSTM (1997, no forget gate)", "E3"),
    "pandemonium": ("Pandemonium (Selfridge)", "E1"),
    "peephole_lstm": ("Peephole LSTM", "E3"),
    "pi_sigma": ("Pi-Sigma network", "E2"),
    "pnn_specht": ("Probabilistic Neural Network (PNN, Specht)", "E3"),
    "predictive_sparse_decomposition": ("Predictive Sparse Decomposition (PSD)", "E4"),
    "raam": ("RAAM (Recursive Auto-Associative Memory)", "E3"),
    "rao_ballard_pcn": ("Rao-Ballard hierarchical predictive coding", "E6"),
    "recurrent_cascade_correlation": ("Recurrent Cascade-Correlation (RCC)", "E3"),
    "replicated_softmax": ("Replicated Softmax Model", "E4"),
    "sdm": ("Sparse Distributed Memory (SDM, Kanerva)", "E3"),
    "semantic_pointer_architecture": ("Semantic Pointer Architecture (SPA)", "E6"),
    "sharpmask": ("SharpMask", "E5"),
    "sigma_pi": ("Sigma-Pi / higher-order unit", "E2"),
    "sketch_a_net": ("Sketch-a-Net", "E5"),
    "stack_augmented_rnn": ("Stack/Queue-Augmented RNN", "E5"),
    "temporal_rbm": ("Temporal RBM / Recurrent Temporal RBM", "E4"),
    "tempotron": ("Tempotron", "E4"),
    "wilson_cowan": ("Wilson-Cowan population model", "E6"),
}


def _year_from_docstring(docstring: str) -> str:
    """Extract the first plausible publication year from a module docstring.

    Parameters
    ----------
    docstring:
        Module docstring text.

    Returns
    -------
    str
        Four-digit year, or an empty string when absent.
    """

    match = re.search(r"(?<!\d)(?:18|19|20)\d{2}(?!\d)", docstring)
    return match.group(0) if match else ""


def _paper_from_docstring(docstring: str) -> str:
    """Extract a compact paper/title hint from a module docstring.

    Parameters
    ----------
    docstring:
        Module docstring text.

    Returns
    -------
    str
        Paper title or historical citation hint.
    """

    lines = [line.strip() for line in docstring.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("paper:"):
            return line.split(":", 1)[1].strip()
    return lines[1] if len(lines) > 1 else ""


CLASSICS_LOAD_ERRORS: Final[list[tuple[str, str]]] = []


def _load_classics() -> dict[str, dict[str, Any]]:
    """Import classics modules and assemble the public registry.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from canonical architecture name to registry metadata.
    """

    registry: dict[str, dict[str, Any]] = {}
    for module_name, (canonical_name, era) in sorted(_CLASSIC_MANIFEST.items()):
        module_path = f"{__name__}.{module_name}"
        module = importlib.import_module(module_path)
        docstring = inspect.getdoc(module) or ""
        build = getattr(module, "build")
        example_input = getattr(module, "example_input")
        registry[canonical_name] = {
            "module_path": module_path,
            "build": build,
            "example_input": example_input,
            "year": _year_from_docstring(docstring),
            "family": canonical_name,
            "era": era,
            "paper": _paper_from_docstring(docstring),
        }

    # Batch-2+ self-declaring modules: each exposes
    # MENAGERIE_ENTRIES = [(canonical_name, build_attr, example_attr, year, code), ...].
    # Discover any classics module not in the hand-written manifest and register its entries.
    import pkgutil
    from pathlib import Path

    manifest_stems = set(_CLASSIC_MANIFEST)
    for info in pkgutil.iter_modules([str(Path(__file__).parent)]):
        stem = info.name
        if stem.startswith("_") or stem in manifest_stems:
            continue
        module_path = f"{__name__}.{stem}"
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 -- record broken module, keep loading
            CLASSICS_LOAD_ERRORS.append((stem, f"import: {type(exc).__name__}: {exc}"))
            continue
        entries = getattr(module, "MENAGERIE_ENTRIES", None)
        if not entries:
            continue
        docstring = inspect.getdoc(module) or ""
        for entry in entries:
            try:
                canonical_name, build_attr, example_attr, year, code = entry
                registry[canonical_name] = {
                    "module_path": module_path,
                    "build": getattr(module, build_attr),
                    "example_input": getattr(module, example_attr),
                    "year": str(year) or _year_from_docstring(docstring),
                    "family": canonical_name,
                    "era": code,
                    "paper": _paper_from_docstring(docstring),
                }
            except Exception as exc:  # noqa: BLE001
                CLASSICS_LOAD_ERRORS.append((stem, f"entry {entry!r}: {type(exc).__name__}: {exc}"))
    return registry


CLASSICS: Final[dict[str, dict[str, Any]]] = _load_classics()

BuildFunc = Callable[[], Any]
ExampleInputFunc = Callable[[], Any]

__all__ = ["CLASSIC_ZOO", "CLASSICS", "BuildFunc", "ExampleInputFunc"]
