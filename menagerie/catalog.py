"""Catalog build, query, and recipe tools for the TorchLens model menagerie."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from menagerie.identity import ensure_stable_ids, recipe_revision_sha256
from menagerie.schema import (
    CatalogRecord,
    VerificationExpectation,
    load_jsonl,
    recipe_is_quarantined,
    validate_records,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_TSV = DATA_DIR / "master_catalog.tsv"
SOURCE_JSONL = DATA_DIR / "master_catalog.jsonl"
DEFERRED_JSONL = DATA_DIR / "deferred.jsonl"
CANONICAL_TSV = DATA_DIR / "catalog_canonical.tsv"
CATALOG_DB = DATA_DIR / "catalog.db"
STABLE_IDS_JSONL = DATA_DIR / "stable_ids.jsonl"
SOURCE_COLUMNS = (
    "name",
    "zoo",
    "constructor_call",
    "input_shape",
    "input_dtype",
    "family",
    "domain",
    "era",
    "notes",
)
CANONICAL_COLUMNS = (
    "model_id",
    "display_index",
    "stable_id",
    "name",
    "variant",
    "family",
    "family_normalized",
    "domain",
    "zoo",
    "constructor_call",
    "input_shape",
    "input_dtype",
    "era",
    "verified",
    "notes",
    "source",
    "recipe_revision_sha256",
    "input_is_real",
    "verification_expectation",
    "quarantine",
)


PATCH_RE = re.compile(r"patch\d+", re.I)
RES_RE = re.compile(r"\b\d{3,4}(x\d{2,4})?\b")
NUMSCALE_RE = re.compile(r"\b\d+(\.\d+)?[mbg]\b", re.I)
SCALEWORD_RE = re.compile(
    r"\b(nano|tiny|small|base|large|huge|giant|gigantic|mini|micro|atto|femto|pico|"
    r"lite|light)\b",
    re.I,
)
FW_SUFFIX_RE = re.compile(
    r"\b(paddle|mmdet|mmseg|mmcls|mmpose|mmocr|mmaction|detectron2|keras|tf|jax|"
    r"torch)\b\s*$",
    re.I,
)
META_CLASS = {
    "encoder_decoder",
    "encoderdecoder",
    "cascade_encoder_decoder",
    "topdown-heatmap",
    "topdown_heatmap",
    "topdown-regression",
    "two_stage_detector",
    "single_stage_detector",
    "base_detector",
}
NEGATIVE_VERIFICATION_MARKERS = (
    "not installed",
    "source-catalog",
    "source catalog",
    "web-only",
    "web only",
    "web;",
    "source only",
    "source-only",
    "catalog-only",
    "metadata-only",
    "abi-incompat",
    "not random-init-buildable",
    "web/source",
    "strictly-local",
    "no-download",
    "blocked",
    "arch confirmed from",
    "confirmed from config",
    "build failed",
    "failed:",
    "failed locally",
    "not random-init",
)
POSITIVE_VERIFICATION_MARKERS = (
    "verified",
    "instantiat",
    "built",
    "installed",
    "smoke",
    "dry-run",
    "dryrun",
    "confirmed",
    "traced",
    "renderable",
    "random init",
    "random-init",
    "pretrained=false",
    "build succeeded",
    "live ",
    "registry",
    "[ok]",
    "recipe:",
    "recipe=",
)


@dataclass(frozen=True)
class CatalogRow:
    """A normalized menagerie catalog row.

    Parameters
    ----------
    model_id:
        Alias for ``display_index``. This is not an identity key.
    display_index:
        Human-facing integer index assigned after sorting.
    stable_id:
        Opaque durable model identity.
    name:
        Model name from the source catalog.
    variant:
        Optional natural-key discriminator for same-name/same-zoo rows.
    family:
        Source family string.
    family_normalized:
        Canonical family string.
    domain:
        Macro-domain string.
    zoo:
        Source model zoo or library.
    constructor_call:
        Python expression for random-init construction when available.
    input_shape:
        Example input tensor shape.
    input_dtype:
        Example input dtype.
    era:
        Source era/year string.
    verified:
        Whether the source metadata indicates an instantiable recipe.
    notes:
        Source notes.
    source:
        Canonical source stream: ``catalog`` for TSV rows or ``classics`` for registry rows.
    recipe_revision_sha256:
        Frozen recipe fingerprint for the row's current construction recipe.
    input_is_real:
        Whether the runtime input is a real example input rather than a sentinel ignored by a
        wrapper.
    verification_expectation:
        Expected validation posture for honest reporting.
    quarantine:
        Whether the row uses a quarantined executable recipe form.
    """

    model_id: int
    display_index: int
    stable_id: str
    name: str
    variant: str
    family: str
    family_normalized: str
    domain: str
    zoo: str
    constructor_call: str
    input_shape: str
    input_dtype: str
    era: str
    verified: bool
    notes: str
    source: str
    recipe_revision_sha256: str
    input_is_real: bool = True
    verification_expectation: str = "forward_required"
    quarantine: bool = False


def catalog_row_from_payload(payload: Mapping[str, Any]) -> CatalogRow:
    """Build a catalog row from a JSON-compatible payload.

    Parameters
    ----------
    payload:
        JSON-compatible row payload.

    Returns
    -------
    CatalogRow
        Catalog row reconstructed from every dataclass field.
    """

    values: dict[str, Any] = {}
    for field in fields(CatalogRow):
        if field.name in payload:
            values[field.name] = payload[field.name]
        elif field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:
            values[field.name] = field.default_factory()
        else:
            values[field.name] = payload[field.name]
    return CatalogRow(**values)


def macro_domain(raw: str) -> str:
    """Collapse source micro-domains into reproducible macro-domain buckets.

    Parameters
    ----------
    raw:
        Source domain string.

    Returns
    -------
    str
        Canonical macro-domain.
    """

    domain = raw.strip().lower()
    if domain in {"history", "historical", "historical-reimplementation"}:
        return "history"
    if re.search(
        r"gan|diffusion|generativ|image-generation|image_gen|text-to-image|image-to-image|"
        r"image-translation|inpaint|stylegan|super-resolution|super_resolution|restoration|"
        r"denois|deblur|deraining|matting|colorization|frame-interpolation|video-frame|"
        r"gan-inversion|autoencoding|normalizing-flow|normalizing flow|density estimation|"
        r"density_est|density|energy_based|energy-based|invertible|flow|vocoder|tts|"
        r"text-to-speech|music|svs|watermark|compression|codec|learned-compression|"
        r"tokenizer|view_synthesis|view-synthesis|world.?model|sequence_gen|"
        r"conditional generation|variational|sampling",
        domain,
    ):
        if re.search(
            r"radiance|nerf|gaussian-splat|gaussian_splat|implicit|point.?cloud|"
            r"point-voxel|sdf|mesh|surface|3d/|3d_|^3d\b|4d/|scene|sparse-view|"
            r"multi-view-stereo|rasterizer",
            domain,
        ) and not re.search(r"gan|stylegan", domain):
            return "3D/geometry/NeRF/point-cloud"
        if re.search(
            r"tts|text-to-speech|vocoder|asr|speech|audio|music|svs|sound|voice|speaker|diariz|enhancement|separation|codec|wav",
            domain,
        ):
            return "audio/speech"
        return "generative/diffusion-GAN-flow"
    if re.search(
        r"\bnerf\b|radiance|gaussian-splat|gaussian_splat|3d/|3d_|^3d\b|4d/|point.?cloud|point-voxel|point_voxel|implicit|sdf|mesh|surface-recon|neural-surface|view.?synthesis|multi-view-stereo|sparse-view|rasterizer|image-based-rendering|reflectance-field|scene",
        domain,
    ):
        return "3D/geometry/NeRF/point-cloud"
    if re.search(
        r"protein|molecul|atomistic|interatomic|crystal|materials|catalyst|quantum.?chem|quantum_ml|drug|md potential|force-field|force field|single-cell|transcriptom|genomic|dna|rna|cell-segmentation|cell segmentation",
        domain,
    ):
        return "scientific/molecular/protein/genomics"
    if re.search(
        r"weather|climate|atmosphere|ocean|air-quality|geophys|earth-system|pde|neural-operator|operator learning|operator-learning|physics-informed|physics simulation|simulation|fluid",
        domain,
    ):
        return "scientific/physics/weather/PDE"
    if re.search(
        r"\bgraph\b|gnn|knowledge-graph|knowledge graph|link-prediction|link prediction|node-class|node class|hypergraph|heterogeneous|spatiotemporal-graph|temporal-graph|geometric-dl|geometric_deep|equivariant|message passing|combinatorial-optim|program-analysis|set/readout|signed-graph|random-walk",
        domain,
    ):
        return "graph/geometric"
    if re.search(
        r"\brl\b|robotic|vla|imitation|manipulation|decision|offline-rl|actor-critic|policy|planning|embodied|reward|agent|minecraft|model-based-rl|generalist|cross-embodiment|bc\b",
        domain,
    ):
        return "RL/robotics/control"
    if re.search(
        r"recommend|recsys|collaborative|ctr|click-through|sequential-recommendation|next-basket|context-aware|tabular|deep learning.*tabular",
        domain,
    ):
        return "recsys/tabular"
    if re.search(
        r"spiking|neuromorph|snn|loihi|xylo|stdp|reservoir|plasticity|event-vision", domain
    ):
        return "spiking/neuromorphic"
    if re.search(
        r"time.?series|time series|forecast|anomaly|imputation|finance|limit-order|stochastic|differential equation|continuous-depth|continuous_flow|continuous-time|irregular time|dynamics|sde|ode|neural diff",
        domain,
    ):
        return "time-series/dynamics"
    if re.search(
        r"\bneuro\b|neuro_|neuro-|brain|eeg|fmri|spike|retina|cortex|astronom|cosmolog|galaxy|sonar|radar|rf/|micro-doppler|hep\b|physics / dynamics",
        domain,
    ):
        return "neuro/scientific-signals"
    if re.search(
        r"medical|clinical|chest-xray|pathology|polyp|brain-tumor|wsi|radiology|registration|deformable",
        domain,
    ):
        return "medical-imaging"
    if re.search(
        r"ocr|document|chart|layout|text-spotting|key-information|table-structure|pdf-to", domain
    ):
        return "OCR/document"
    if re.search(
        r"multimodal|vision-language|vision_language|image-text|video-language|video-text|vqa|clip|retrieval.*image|text-audio|speech-text",
        domain,
    ):
        return "multimodal"
    if re.search(
        r"audio|speech|asr|tts|vocoder|music|sound|voice|speaker|diariz|svs|wav|vad|keyword|alignment",
        domain,
    ):
        return "audio/speech"
    if re.search(
        r"detection|detect|tracking|reid|person-reid|mot|sot|object-as-points|lane-detection|oriented-detection|rotated-detection|salient",
        domain,
    ):
        return "vision/detection-tracking"
    if re.search(r"segment|seg\b|matting|panoptic|instance-seg|portrait", domain):
        return "vision/segmentation"
    if re.search(r"pose|keypoint|human-mesh|landmark|gait|dense-pose", domain):
        return "vision/pose"
    if re.search(
        r"depth|stereo|optical-flow|geometry|surface-normal|camera-intrinsic|monocular|multi-view",
        domain,
    ):
        return "vision/depth-geometry"
    if re.search(
        r"video|action-recognition|action|temporal-localization|skeleton-action|sign-language|lipreading",
        domain,
    ):
        return "video/action"
    if re.search(
        r"\btext\b|language|llm|nlp|sequence modeling|seq2seq|token|embedding|encoder|natural-language|question-answering|reasoning|translation|byte|retrieval",
        domain,
    ):
        return "NLP/LLM/text"
    if re.search(
        r"vision|image|classification|backbone|capsule|nas|automl|distillation|self-supervised|continual|meta.?learning|few-shot|federated|hypernetwork|face-recognition|anomaly|hashing|efficient|transformer|mlp",
        domain,
    ):
        return "vision/classification-backbone"
    if re.search(
        r"exotic|associative|memory-network|capsule|mlp_replacement|mlp-alt|general/|algorithmic",
        domain,
    ):
        return "exotic/other"
    return "exotic/other"


def _arch_from_name(name: str) -> str:
    """Recover architecture token from an mmlab-style config name.

    Parameters
    ----------
    name:
        Source model name.

    Returns
    -------
    str
        Recovered architecture token.
    """

    lowered = name.lower()
    for prefix in (
        "mmseg_encoder_decoder_",
        "mmseg_cascade_encoder_decoder_",
        "mmseg_",
        "mmpose_topdown_heatmap_",
        "mmpose_topdown_regression_",
        "mmpose_",
        "mmdet_",
        "mmcls_",
    ):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return re.split(r"[_\-]", lowered)[0]


def normalize_family(family: str, name: str, zoo: str) -> str:
    """Canonicalize a source family string.

    Parameters
    ----------
    family:
        Source family value.
    name:
        Source model name.
    zoo:
        Source model zoo.

    Returns
    -------
    str
        Canonical family value.
    """

    family_clean = family.strip()
    family_lower = family_clean.lower()
    zoo_lower = zoo.lower()

    if family_lower in META_CLASS:
        recovered = _arch_from_name(name)
        if recovered and recovered != family_lower:
            return normalize_family(recovered, name, zoo)

    if zoo_lower == "ultralytics" or family_lower.startswith("yolo_"):
        name_lower = name.lower()
        if "rtdetr" in name_lower or "rt-detr" in name_lower:
            return "RT-DETR"
        if "yoloe" in name_lower:
            match = re.search(r"yoloe-?v?(\d+)", name_lower)
            return f"YOLOE-v{match.group(1)}" if match else "YOLOE"
        match = re.search(r"yolov?(\d+)", name_lower)
        if match:
            return f"YOLOv{match.group(1)}"

    vit_distinct = {
        "vits",
        "vitgan",
        "vitgan-cips",
        "vitamin",
        "vitmatte",
        "vitpose",
        "vitdet",
        "vitstr",
        "vitstr-paddle",
        "vit-adapter",
    }
    if (
        re.fullmatch(r"(vit|vision[-_ ]?transformer|vision transformer)([-_].*)?", family_lower)
        and family_lower not in vit_distinct
        and not any(
            family_lower.startswith(item)
            for item in ("vits", "vitgan", "vitamin", "vitmatte", "vitpose", "vitdet", "vitstr")
        )
    ):
        if "deit" in family_lower:
            return "DeiT"
        return "ViT (Vision Transformer)"
    if family_lower in ("deit", "deit3") or family_lower.startswith("deit"):
        return "DeiT"
    if "swin" in family_lower:
        return "Swin Transformer"
    if (
        re.search(r"\bunet\b|u-net|unet\+\+|unetplusplus|unet3\+", family_lower)
        or family_lower.startswith("unet")
        or "unet" in family_lower.replace("-", "")
    ):
        if "transunet" in family_lower or "trans-unet" in family_lower:
            return "TransUNet"
        if "swinunet" in family_lower or "swin-unet" in family_lower:
            return "Swin-UNet"
        return "U-Net"
    if re.search(
        r"\bresnet\b|resnext|resnest|wide[-_ ]?resnet|resnet1d|3d resnet|resnet3d", family_lower
    ):
        if "resnext" in family_lower:
            return "ResNeXt"
        if "resnest" in family_lower:
            return "ResNeSt"
        return "ResNet"
    if re.search(r"efficientnet", family_lower):
        return "EfficientNetV2" if "v2" in family_lower else "EfficientNet"
    for pattern, canonical in (
        (r"mobilenet", "MobileNet"),
        (r"convnext", "ConvNeXt"),
        (r"regnet", "RegNet"),
        (r"densenet", "DenseNet"),
        (r"\bvgg\b", "VGG"),
        (r"\bdla\b", "DLA"),
        (r"hrnet", "HRNet"),
    ):
        if re.search(pattern, family_lower):
            return canonical
    if re.search(r"\byolo", family_lower):
        if "yolos" in family_lower:
            return "YOLOS (ViT detector)"
        if re.search(r"spik|ems_yolo|su_yolo", family_lower):
            pass
        elif "yolo-world" in family_lower or "yoloworld" in family_lower:
            return "YOLO-World"
        elif "bytetrack" in family_lower or "deepsort" in family_lower:
            pass
        elif "yolox" in family_lower:
            return "YOLOX"
        else:
            match = re.search(r"yolov?(\d+)", family_lower)
            if match:
                return f"YOLOv{match.group(1)}"
            return "YOLO (ultralytics)" if zoo_lower == "ultralytics" else "YOLO"
    if "detr" in family_lower:
        if "dino" in family_lower:
            return "DINO-DETR"
        if "deformable" in family_lower:
            return "Deformable DETR"
        if "dab" in family_lower:
            return "DAB-DETR"
        if "rt-detr" in family_lower or "rtdetr" in family_lower:
            return "RT-DETR"
        return "DETR"
    for pattern, canonical in (
        (r"faster.?rcnn", "Faster R-CNN"),
        (r"mask.?rcnn", "Mask R-CNN"),
        (r"cascade.?rcnn", "Cascade R-CNN"),
        (r"retinanet", "RetinaNet"),
        (r"\bfcos\b", "FCOS"),
        (r"\bssd\b", "SSD"),
        (r"deeplab", "DeepLab"),
        (r"pspnet", "PSPNet"),
        (r"segformer", "SegFormer"),
        (r"\bfpn\b", "FPN"),
        (r"mask2former", "Mask2Former"),
        (r"maskformer", "MaskFormer"),
        (r"\bsam\b|segment.?anything", "SAM (Segment Anything)"),
        (r"\bllama\b|llama\d?", "LLaMA"),
        (r"qwen", "Qwen"),
        (r"mistral|mixtral", "Mistral/Mixtral"),
        (r"\bgpt\b|gpt-?2|gpt-?neo|gptj|gpt-j", "GPT"),
        (r"\bt5\b|flan-?t5", "T5"),
        (r"\bgemma", "Gemma"),
        (r"\bphi\b|phi-?\d", "Phi"),
        (r"\bfalcon\b", "Falcon"),
        (r"\bmamba\b", "Mamba"),
        (r"\bclip\b", "CLIP"),
        (r"stable.?diffusion|^sd\b|sdxl", "Stable Diffusion"),
        (r"\bdit\b|diffusion transformer", "DiT (Diffusion Transformer)"),
        (r"\bunet\b.*diffusion|diffusion.*unet|conditional unet", "Diffusion U-Net"),
        (r"\bgcn\b", "GCN"),
        (r"\bgat\b", "GAT"),
        (r"graphsage|sage", "GraphSAGE"),
        (r"\bgin\b", "GIN"),
        (r"pointnet", "PointNet/PointNet++"),
        (r"\bwav2vec", "Wav2Vec2"),
        (r"hubert", "HuBERT"),
        (r"whisper", "Whisper"),
        (r"conformer", "Conformer"),
        (r"wavenet", "WaveNet"),
        (r"hifi.?gan", "HiFi-GAN"),
        (r"tacotron", "Tacotron"),
        (r"stylegan", "StyleGAN"),
        (r"biggan", "BigGAN"),
        (r"cyclegan", "CycleGAN"),
        (r"\bpix2pix", "pix2pix"),
        (r"\bdcgan", "DCGAN"),
        (r"esrgan|rrdb", "ESRGAN/RRDBNet"),
        (r"basicvsr", "BasicVSR"),
        (r"\bdpt\b", "DPT"),
        (r"midas", "MiDaS"),
    ):
        if re.search(pattern, family_lower):
            if pattern == r"\bssd\b" and "3dssd" in family_lower:
                continue
            if pattern == r"\bgcn\b" and "agcn" in family_lower:
                continue
            return canonical
    if re.search(r"\bbert\b|roberta|deberta|albert|electra", family_lower):
        if "roberta" in family_lower:
            return "RoBERTa"
        if "deberta" in family_lower:
            return "DeBERTa"
        return "BERT"

    base = re.split(r"[/(]", family_clean)[0].strip()
    base_lower = re.sub(r"[-_]+", " ", base.lower())
    base_lower = FW_SUFFIX_RE.sub("", base_lower)
    base_lower = PATCH_RE.sub("", base_lower)
    base_lower = RES_RE.sub("", base_lower)
    base_lower = NUMSCALE_RE.sub("", base_lower)
    base_lower = SCALEWORD_RE.sub("", base_lower)
    base_lower = re.sub(r"\s+", " ", base_lower).strip()
    return base_lower or base.lower().strip() or family_clean.lower().strip()


def normalize_family_representatives(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    """Build punctuation-insensitive final family representatives.

    Parameters
    ----------
    rows:
        Source rows with normalized family candidates.

    Returns
    -------
    dict[str, str]
        Mapping from candidate family to final representative.
    """

    representatives: dict[str, str] = {}
    candidates = {row["family_normalized"] for row in rows}
    for candidate in sorted(candidates):
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        current = representatives.get(key)
        if current is None:
            representatives[key] = candidate
            continue
        score = (sum(char.isupper() for char in candidate), -candidate.count(" "), len(candidate))
        current_score = (sum(char.isupper() for char in current), -current.count(" "), len(current))
        if score > current_score:
            representatives[key] = candidate
    return {
        candidate: representatives[re.sub(r"[^a-z0-9]", "", candidate.lower())]
        for candidate in candidates
    }


def is_verified(notes: str, zoo: str) -> bool:
    """Infer whether a catalog row has an instantiable recipe.

    Parameters
    ----------
    notes:
        Source notes.
    zoo:
        Source zoo.

    Returns
    -------
    bool
        Whether the row should be marked verified.
    """

    notes_lower = notes.lower()
    if zoo.lower() in {"timm", "torchvision", "torchvision.models"}:
        return True
    has_negative = any(marker in notes_lower for marker in NEGATIVE_VERIFICATION_MARKERS)
    has_positive = any(marker in notes_lower for marker in POSITIVE_VERIFICATION_MARKERS)
    return has_positive and not has_negative


def _source_rows(source_tsv: Path = SOURCE_TSV) -> list[dict[str, str]]:
    """Read source TSV rows with optional header detection.

    Parameters
    ----------
    source_tsv:
        Source TSV path.

    Returns
    -------
    list[dict[str, str]]
        Parsed source rows.
    """

    with source_tsv.open(newline="") as handle:
        raw_rows = list(csv.reader(handle, delimiter="\t"))
    if not raw_rows:
        return []
    first = [value.strip() for value in raw_rows[0]]
    data_rows = raw_rows[1:] if first == list(SOURCE_COLUMNS) else raw_rows
    from menagerie.classics import CLASSIC_ZOO, CLASSICS

    rows = []
    for index, row in enumerate(data_rows, start=1):
        if len(row) != len(SOURCE_COLUMNS):
            raise ValueError(f"{source_tsv}:{index} has {len(row)} columns, expected 9")
        parsed = dict(zip(SOURCE_COLUMNS, row))
        if parsed["zoo"] == CLASSIC_ZOO and parsed["name"] in CLASSICS:
            continue
        rows.append(parsed)
    return rows


def _constructor_from_record(record: CatalogRecord) -> str:
    """Return the legacy constructor text preserved in a typed record.

    Parameters
    ----------
    record:
        Typed catalog record.

    Returns
    -------
    str
        Constructor text used by legacy row consumers.
    """

    legacy_constructor = record.legacy.get("constructor_call")
    if isinstance(legacy_constructor, str):
        return legacy_constructor
    recipe = record.recipe
    if recipe.type in {"import-callable", "expression"}:
        return recipe.expr
    if recipe.type == "statement":
        return recipe.code
    if recipe.type == "exec-string":
        return recipe.body
    raise ValueError(f"unsupported recipe type={recipe.type!r}")


def _row_from_record(record: CatalogRecord, source: str) -> dict[str, str | bool]:
    """Convert one typed JSONL record to the canonical build row shape.

    Parameters
    ----------
    record:
        Typed catalog record.
    source:
        Source stream label for the canonical row.

    Returns
    -------
    dict[str, str | bool]
        Row dictionary ready for normalization and identity assignment.
    """

    notes = record.notes
    verified = record.verification_expectation == VerificationExpectation.forward_required
    if record.verification_expectation == VerificationExpectation.deferred:
        reason = record.deferral.reason if record.deferral is not None else "deferred"
        notes = (
            f"{notes}; verification_expectation=deferred; deferral_reason={reason}"
            if notes
            else f"verification_expectation=deferred; deferral_reason={reason}"
        )
    return {
        "name": record.name,
        "variant": record.variant,
        "zoo": record.zoo,
        "constructor_call": _constructor_from_record(record),
        "input_shape": record.input_shape or "",
        "input_dtype": record.input_dtype or "",
        "family": record.family,
        "domain": record.domain,
        "era": record.era,
        "notes": notes,
        "verified": verified,
        "source": source,
        "input_is_real": record.input_is_real,
        "verification_expectation": record.verification_expectation.value,
        "quarantine": recipe_is_quarantined(record.recipe),
    }


def _jsonl_source_rows(
    source_jsonl: Path = SOURCE_JSONL, deferred_jsonl: Path = DEFERRED_JSONL
) -> list[dict[str, str | bool]]:
    """Load typed non-classics rows from committed JSONL sources.

    Parameters
    ----------
    source_jsonl:
        Forward-required non-classics JSONL source.
    deferred_jsonl:
        Honestly deferred non-classics JSONL source.

    Returns
    -------
    list[dict[str, str | bool]]
        Source rows converted from typed records.
    """

    records = load_jsonl(source_jsonl)
    deferred_records = load_jsonl(deferred_jsonl) if deferred_jsonl.exists() else []
    validate_records([*records, *deferred_records])
    return [
        *[_row_from_record(record, "catalog") for record in records],
        *[_row_from_record(record, "catalog") for record in deferred_records],
    ]


def _shape_dtype_for_input(example: Any) -> tuple[str, str]:
    """Describe a classics example input for catalog metadata.

    Parameters
    ----------
    example:
        Example input object returned by a classics module.

    Returns
    -------
    tuple[str, str]
        Catalog shape string and dtype string.
    """

    inputs = example if isinstance(example, tuple) else (example,)
    shapes = []
    dtypes = []
    for item in inputs:
        shape = getattr(item, "shape", None)
        dtype = getattr(item, "dtype", None)
        if shape is None:
            shapes.append(type(item).__name__)
        else:
            shapes.append(tuple(int(dim) for dim in shape))
        if dtype is not None:
            dtypes.append(str(dtype).replace("torch.", ""))
    if len(shapes) == 1:
        shape_text = str(shapes[0])
    else:
        shape_text = str(shapes)
    dtype_text = dtypes[0] if len(set(dtypes)) == 1 else ";".join(dtypes)
    return shape_text, dtype_text or "unknown"


def _classics_source_rows() -> list[dict[str, str]]:
    """Build virtual source rows for local historical reimplementations.

    Returns
    -------
    list[dict[str, str]]
        Source-schema rows derived from ``menagerie.classics.CLASSICS``.
    """

    from menagerie.classics import CLASSIC_ZOO, CLASSICS

    rows = []
    for name, entry in CLASSICS.items():
        module_path = str(entry["module_path"])
        module_name = module_path.rsplit(".", maxsplit=1)[-1]
        # Self-declaring modules expose one builder per family (e.g.
        # ``build_zoed_n``/``build_zoed_k``); hand-manifest modules expose a
        # single zero-arg ``build``. Recover the actual builder name from the
        # registered build callable so the constructor_call is faithful per
        # variant rather than always pointing at ``build()`` (which may not even
        # exist for multi-variant modules such as espnet_speech).
        build_name = getattr(entry["build"], "__name__", "build")
        example = entry["example_input"]()
        input_shape, input_dtype = _shape_dtype_for_input(example)
        paper = str(entry["paper"])
        zoo = str(entry.get("zoo") or CLASSIC_ZOO)
        origin = {
            CLASSIC_ZOO: "historical-reimplementation",
            "vendored-pytorch": "vendored-real-repo-code",
            "ported-pytorch": "faithful-port-of-real-code",
            "reimpl-pytorch": "faithful-reimplementation-from-description",
        }.get(zoo, zoo)
        notes = f"verified; traced; source={origin}"
        if paper:
            notes = f"{notes}; paper={paper}"
        rows.append(
            {
                "name": name,
                "zoo": zoo,
                "constructor_call": f"menagerie.classics.{module_name}.{build_name}()",
                "input_shape": input_shape,
                "input_dtype": input_dtype,
                "family": str(entry["family"]),
                "domain": "history",
                "era": str(entry["era"]),
                "notes": notes,
            }
        )
    return rows


def build_canonical_rows(
    source_jsonl: Path = SOURCE_JSONL, deferred_jsonl: Path = DEFERRED_JSONL
) -> list[CatalogRow]:
    """Build normalized, deduplicated catalog rows.

    Parameters
    ----------
    source_jsonl:
        Forward-required non-classics JSONL source.
    deferred_jsonl:
        Honestly deferred non-classics JSONL source.

    Returns
    -------
    list[CatalogRow]
        Canonical rows sorted by model name, zoo, and constructor.
    """

    intermediate = []
    for row in _jsonl_source_rows(source_jsonl, deferred_jsonl):
        normalized = normalize_family(row["family"], row["name"], row["zoo"])
        intermediate.append(
            {
                **row,
                "family_normalized": normalized,
                "domain": macro_domain(row["domain"]),
                "verified": is_verified(row["notes"], row["zoo"]),
                "source": "catalog",
            }
        )
    for row in _classics_source_rows():
        normalized = normalize_family(row["family"], row["name"], row["zoo"])
        intermediate.append(
            {
                **row,
                "family_normalized": normalized,
                "domain": macro_domain(row["domain"]),
                "verified": True,
                "source": "classics",
                "input_is_real": True,
                "verification_expectation": VerificationExpectation.forward_required.value,
                "quarantine": False,
            }
        )
    representative_map = normalize_family_representatives(intermediate)
    for row in intermediate:
        row["family_normalized"] = representative_map[row["family_normalized"]]
    sorted_rows = sorted(
        intermediate,
        key=lambda item: (
            str(item["family_normalized"]).lower(),
            str(item["name"]).lower(),
            str(item["zoo"]).lower(),
        ),
    )
    variants = [str(row.get("variant", "")) for row in sorted_rows]

    stable_keys = {
        (str(row["name"]), str(row["zoo"]), variant) for row, variant in zip(sorted_rows, variants)
    }
    stable_ids = ensure_stable_ids(stable_keys, STABLE_IDS_JSONL)

    canonical_rows = []
    for index, (row, variant) in enumerate(zip(sorted_rows, variants), start=1):
        draft = CatalogRow(
            model_id=index,
            display_index=index,
            stable_id=stable_ids[(str(row["name"]), str(row["zoo"]), variant)],
            name=str(row["name"]),
            variant=variant,
            family=str(row["family"]),
            family_normalized=str(row["family_normalized"]),
            domain=str(row["domain"]),
            zoo=str(row["zoo"]),
            constructor_call=str(row["constructor_call"]),
            input_shape=str(row["input_shape"]),
            input_dtype=str(row["input_dtype"]),
            era=str(row["era"]),
            verified=bool(row["verified"]),
            notes=str(row["notes"]),
            source=str(row["source"]),
            recipe_revision_sha256="",
            input_is_real=bool(row.get("input_is_real", True)),
            verification_expectation=str(
                row.get("verification_expectation", VerificationExpectation.forward_required.value)
            ),
            quarantine=bool(row.get("quarantine", False)),
        )
        canonical_rows.append(
            CatalogRow(
                **{
                    **draft.__dict__,
                    "recipe_revision_sha256": recipe_revision_sha256(draft),
                }
            )
        )
    assert_catalog_identity(canonical_rows)
    return canonical_rows


def assert_catalog_identity(rows: Sequence[CatalogRow]) -> None:
    """Assert canonical row identity invariants.

    Parameters
    ----------
    rows:
        Canonical rows to validate.

    Raises
    ------
    AssertionError
        If stable IDs or natural keys are duplicated.
    """

    stable_ids = [row.stable_id for row in rows]
    natural_keys = [(row.name, row.zoo, row.variant) for row in rows]
    duplicate_stable_ids = [
        stable_id for stable_id, count in Counter(stable_ids).items() if count > 1
    ]
    duplicate_natural_keys = [key for key, count in Counter(natural_keys).items() if count > 1]
    if duplicate_stable_ids:
        raise AssertionError(f"duplicate stable_id values: {duplicate_stable_ids[:10]!r}")
    if duplicate_natural_keys:
        raise AssertionError(f"duplicate natural keys: {duplicate_natural_keys[:10]!r}")


def write_catalog(
    rows: Sequence[CatalogRow],
    canonical_tsv: Path = CANONICAL_TSV,
    db_path: Path = CATALOG_DB,
) -> None:
    """Persist canonical rows to TSV and SQLite.

    Parameters
    ----------
    rows:
        Canonical rows.
    canonical_tsv:
        Output TSV path.
    db_path:
        Output SQLite database path.
    """

    canonical_tsv.parent.mkdir(parents=True, exist_ok=True)
    with canonical_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(CANONICAL_COLUMNS)
        for row in rows:
            writer.writerow([*row.__dict__.values()])

    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE models (
                model_id INTEGER PRIMARY KEY,
                display_index INTEGER NOT NULL,
                stable_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                variant TEXT NOT NULL DEFAULT '',
                family TEXT NOT NULL,
                family_normalized TEXT NOT NULL,
                domain TEXT NOT NULL,
                zoo TEXT NOT NULL,
                constructor_call TEXT NOT NULL,
                input_shape TEXT NOT NULL,
                input_dtype TEXT NOT NULL,
                era TEXT NOT NULL,
                verified INTEGER NOT NULL,
                notes TEXT NOT NULL,
                source TEXT NOT NULL,
                recipe_revision_sha256 TEXT NOT NULL,
                input_is_real INTEGER NOT NULL DEFAULT 1,
                verification_expectation TEXT NOT NULL DEFAULT 'forward_required',
                quarantine INTEGER NOT NULL DEFAULT 0,
                UNIQUE(name, zoo, variant)
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row.model_id,
                    row.display_index,
                    row.stable_id,
                    row.name,
                    row.variant,
                    row.family,
                    row.family_normalized,
                    row.domain,
                    row.zoo,
                    row.constructor_call,
                    row.input_shape,
                    row.input_dtype,
                    row.era,
                    int(row.verified),
                    row.notes,
                    row.source,
                    row.recipe_revision_sha256,
                    int(row.input_is_real),
                    row.verification_expectation,
                    int(row.quarantine),
                )
                for row in rows
            ],
        )
        connection.execute("CREATE UNIQUE INDEX idx_models_stable_id ON models(stable_id)")
        connection.execute(
            "CREATE UNIQUE INDEX idx_models_natural_key ON models(name, zoo, variant)"
        )
        connection.execute("CREATE INDEX idx_models_name ON models(name)")
        connection.execute("CREATE INDEX idx_models_family ON models(family_normalized)")
        connection.execute("CREATE INDEX idx_models_domain ON models(domain)")
        connection.execute("CREATE INDEX idx_models_zoo ON models(zoo)")
        connection.execute("CREATE INDEX idx_models_verified ON models(verified)")


def ensure_catalog(db_path: Path = CATALOG_DB) -> Path:
    """Ensure the SQLite catalog exists.

    Parameters
    ----------
    db_path:
        Catalog database path.

    Returns
    -------
    Path
        Existing or newly built database path.
    """

    if not db_path.exists():
        rows = build_canonical_rows()
        write_catalog(rows, db_path=db_path)
    return db_path


def load_rows(
    family: str | None = None,
    domain: str | None = None,
    zoo: str | None = None,
    verified: bool = False,
    limit: int | None = None,
    db_path: Path = CATALOG_DB,
) -> list[CatalogRow]:
    """Load catalog rows matching filters.

    Parameters
    ----------
    family:
        Case-insensitive family substring.
    domain:
        Case-insensitive domain substring.
    zoo:
        Case-insensitive zoo substring.
    verified:
        Restrict to verified rows.
    limit:
        Optional row limit.
    db_path:
        Catalog database path.

    Returns
    -------
    list[CatalogRow]
        Matching rows.
    """

    ensure_catalog(db_path)
    clauses = []
    params: list[str | int] = []
    if family:
        clauses.append("lower(family_normalized) LIKE ?")
        params.append(f"%{family.lower()}%")
    if domain:
        clauses.append("lower(domain) LIKE ?")
        params.append(f"%{domain.lower()}%")
    if zoo:
        clauses.append("lower(zoo) LIKE ?")
        params.append(f"%{zoo.lower()}%")
    if verified:
        clauses.append("verified = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM models {where} ORDER BY model_id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(sql, params).fetchall()
    return [
        CatalogRow(
            model_id=row[0],
            display_index=row[1],
            stable_id=row[2],
            name=row[3],
            variant=row[4],
            family=row[5],
            family_normalized=row[6],
            domain=row[7],
            zoo=row[8],
            constructor_call=row[9],
            input_shape=row[10],
            input_dtype=row[11],
            era=row[12],
            verified=bool(row[13]),
            notes=row[14],
            source=row[15],
            recipe_revision_sha256=row[16],
            input_is_real=bool(row[17]) if len(row) > 17 else True,
            verification_expectation=str(row[18]) if len(row) > 18 else "forward_required",
            quarantine=bool(row[19]) if len(row) > 19 else False,
        )
        for row in rows
    ]


def find_recipe(name: str, db_path: Path = CATALOG_DB) -> CatalogRow:
    """Find a single model recipe by exact or substring name.

    Parameters
    ----------
    name:
        Exact model name, falling back to case-insensitive substring.
    db_path:
        Catalog database path.

    Returns
    -------
    CatalogRow
        Matching catalog row.
    """

    ensure_catalog(db_path)
    with sqlite3.connect(db_path) as connection:
        exact = connection.execute(
            "SELECT * FROM models WHERE lower(name) = lower(?) ORDER BY verified DESC, model_id LIMIT 1",
            (name,),
        ).fetchone()
        row = (
            exact
            or connection.execute(
                """
            SELECT * FROM models
            WHERE lower(name) LIKE ?
            ORDER BY verified DESC, length(name), model_id
            LIMIT 1
            """,
                (f"%{name.lower()}%",),
            ).fetchone()
        )
    if row is None:
        raise LookupError(f"No catalog model matched {name!r}")
    return CatalogRow(
        model_id=row[0],
        display_index=row[1],
        stable_id=row[2],
        name=row[3],
        variant=row[4],
        family=row[5],
        family_normalized=row[6],
        domain=row[7],
        zoo=row[8],
        constructor_call=row[9],
        input_shape=row[10],
        input_dtype=row[11],
        era=row[12],
        verified=bool(row[13]),
        notes=row[14],
        source=row[15],
        recipe_revision_sha256=row[16],
        input_is_real=bool(row[17]) if len(row) > 17 else True,
        verification_expectation=str(row[18]) if len(row) > 18 else "forward_required",
        quarantine=bool(row[19]) if len(row) > 19 else False,
    )


def stats(db_path: Path = CATALOG_DB) -> dict[str, Counter[str] | int]:
    """Compute catalog statistics.

    Parameters
    ----------
    db_path:
        Catalog database path.

    Returns
    -------
    dict[str, Counter[str] | int]
        Total, verified count, and family/domain/zoo counters.
    """

    rows = load_rows(db_path=db_path)
    return {
        "total": len(rows),
        "verified": sum(row.verified for row in rows),
        "family": Counter(row.family_normalized for row in rows),
        "domain": Counter(row.domain for row in rows),
        "zoo": Counter(row.zoo for row in rows),
    }


def _print_counter(title: str, counter: Counter[str], limit: int = 20) -> None:
    """Print a ranked counter.

    Parameters
    ----------
    title:
        Section title.
    counter:
        Counter to print.
    limit:
        Maximum rows to print.
    """

    print(f"\n{title}")
    for key, count in counter.most_common(limit):
        print(f"{count:6d}  {key}")


def _print_rows(rows: Sequence[CatalogRow]) -> None:
    """Print query rows in a compact TSV-like format.

    Parameters
    ----------
    rows:
        Rows to print.
    """

    print("\t".join(("model_id", "name", "family_normalized", "domain", "zoo", "verified")))
    for row in rows:
        print(
            "\t".join(
                (
                    str(row.model_id),
                    row.name,
                    row.family_normalized,
                    row.domain,
                    row.zoo,
                    str(row.verified),
                )
            )
        )


def _build_command(args: argparse.Namespace) -> int:
    """Run the build subcommand.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    rows = build_canonical_rows(args.source_jsonl, args.deferred_jsonl)
    write_catalog(rows, args.tsv, args.db)
    print(f"wrote {len(rows)} rows")
    print(f"tsv={args.tsv}")
    print(f"db={args.db}")
    return 0


def _stats_command(args: argparse.Namespace) -> int:
    """Run the stats subcommand.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    catalog_stats = stats(args.db)
    print(f"total: {catalog_stats['total']}")
    print(f"verified: {catalog_stats['verified']}")
    _print_counter("families", catalog_stats["family"], args.limit)
    _print_counter("domains", catalog_stats["domain"], args.limit)
    _print_counter("zoos", catalog_stats["zoo"], args.limit)
    return 0


def _query_command(args: argparse.Namespace) -> int:
    """Run the query subcommand.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    rows = load_rows(args.family, args.domain, args.zoo, args.verified, args.limit, args.db)
    _print_rows(rows)
    print(f"\nrows: {len(rows)}")
    return 0


def _recipe_command(args: argparse.Namespace) -> int:
    """Run the recipe subcommand.

    Parameters
    ----------
    args:
        Parsed CLI args.

    Returns
    -------
    int
        Process exit code.
    """

    row = find_recipe(args.name, args.db)
    print(f"name: {row.name}")
    print(f"constructor_call: {row.constructor_call}")
    print(f"input_shape: {row.input_shape}")
    print(f"input_dtype: {row.input_dtype}")
    print(f"notes: {row.notes}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the catalog CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="rebuild canonical TSV and SQLite DB")
    build.add_argument("--source-jsonl", type=Path, default=SOURCE_JSONL)
    build.add_argument("--deferred-jsonl", type=Path, default=DEFERRED_JSONL)
    build.add_argument("--tsv", type=Path, default=CANONICAL_TSV)
    build.add_argument("--db", type=Path, default=CATALOG_DB)
    build.set_defaults(func=_build_command)

    stats_parser = subparsers.add_parser("stats", help="print catalog counts")
    stats_parser.add_argument("--db", type=Path, default=CATALOG_DB)
    stats_parser.add_argument("--limit", type=int, default=20)
    stats_parser.set_defaults(func=_stats_command)

    query = subparsers.add_parser("query", help="query catalog rows")
    query.add_argument("--family")
    query.add_argument("--domain")
    query.add_argument("--zoo")
    query.add_argument("--verified", action="store_true")
    query.add_argument("--limit", type=int, default=50)
    query.add_argument("--db", type=Path, default=CATALOG_DB)
    query.set_defaults(func=_query_command)

    recipe = subparsers.add_parser("recipe", help="print one model recipe")
    recipe.add_argument("name")
    recipe.add_argument("--db", type=Path, default=CATALOG_DB)
    recipe.set_defaults(func=_recipe_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the catalog CLI.

    Parameters
    ----------
    argv:
        Optional argument vector.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LookupError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
