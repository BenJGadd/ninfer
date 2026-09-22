"""Convert the registered Qwen3.5-9B checkpoint into one complete artifact.

Canonical invocation::

    python -m tools.convert.qwen3_5_9b.convert \
      --model /path/to/Qwen3.5-9B \
      --out out/qwen3_5_9b.ninfer

The model directory is the unmodified ``Qwen/Qwen3.5-9B`` snapshot.  That
repository ships no ``generation_config.json``; the engine needs one for its
default stop ids, so the converter takes it from ``--generation-config``
(default: the pinned copy next to this module) and records its hash.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import draft_head, inventory, recipe
from .geometry import GEOMETRY, Geometry


RECIPE_ID = "qwen3_5_9b-v1"
BASE_REPOSITORY = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

_ROOT_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
    "image_token_id": 248056,
    "video_token_id": 248057,
}
_TEXT_CONFIG = {
    **GEOMETRY.text_config_expectations(),
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-6,
    "attn_output_gate": True,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": GEOMETRY.vision_hidden,
    "intermediate_size": 4304,
    "out_hidden_size": GEOMETRY.hidden,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
}

# Frontend resources. Five come from the checkpoint; generation_config.json is the
# pinned file next to this module (the upstream repository has none). Hashes are
# recorded so a re-download or a repository revision change is detected, not absorbed.
GENERATION_CONFIG_PATH = Path(__file__).resolve().parent / "generation_config.json"
OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "frontend/chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "frontend/generation_config.json": (
        "98cc62c0ad60faa4067de78f12b909e65e4dc093d099f406ac5bb163ad95a2f4"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact registered checkpoint dimensions and summarize them."""

    family_conversion.check_members("config", config, _ROOT_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)
    actual_geometry = Geometry.from_hf_config(config)
    if actual_geometry != GEOMETRY:
        raise ValueError(
            f"checkpoint geometry {actual_geometry} does not match registered {GEOMETRY}"
        )
    expected_layer_types = tuple(
        "full_attention" if layer in GEOMETRY.full_attention_layers else "linear_attention"
        for layer in range(GEOMETRY.layers)
    )
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or tuple(layer_types) != expected_layer_types:
        raise ValueError(
            f"text_config.layer_types does not match the registered {GEOMETRY.layers}-layer schedule"
        )
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "layer_types": {
            "layers": len(layer_types),
            "full_attention": len(GEOMETRY.full_attention_layers),
            "linear_attention": len(GEOMETRY.gdn_layers),
            "full_attention_layers": list(GEOMETRY.full_attention_layers),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
        "mtp_num_hidden_layers": text["mtp_num_hidden_layers"],
        "vision_token_ids": {
            name: config[name]
            for name in (
                "vision_start_token_id",
                "vision_end_token_id",
                "image_token_id",
                "video_token_id",
            )
        },
    }


def expected_object_counts() -> tuple[int, ...]:
    """Independent inventory counts derived from the geometry, never typed in."""

    text_core = inventory.expected_text_core_count(GEOMETRY)
    draft = 2
    mtp = 12
    vision = 3 + 12 * len(inventory.VISION_LAYERS) + 6
    tensors = text_core + draft + mtp + vision
    return (6, text_core, draft, mtp, vision, tensors, tensors + 6)


def preflight_inventory() -> None:
    """Establish the one complete target inventory and recipe pairing."""

    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != expected_object_counts():
        raise ValueError("registered inventory is incomplete")
    recipe.validate_recipe_coverage()


def load_resources(
    model_dir: str | Path,
    generation_config: str | Path = GENERATION_CONFIG_PATH,
) -> tuple[ResourcePayload, ...]:
    """Load the six frontend resources and require their pinned hashes."""

    root = Path(model_dir)
    resources: list[ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = (
            Path(generation_config)
            if filename == "generation_config.json"
            else root / filename
        )
        data = path.read_bytes()
        if not data:
            raise ValueError(f"frontend resource {path} is empty")
        actual = hashlib.sha256(data).hexdigest()
        expected = OFFICIAL_RESOURCE_SHA256[spec.name]
        if actual != expected:
            raise ValueError(
                f"frontend resource hash mismatch for {filename}: expected {expected}, got {actual}"
            )
        resources.append(ResourcePayload(spec.name, data))
    return tuple(resources)


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    """Compute every payload-relative object offset for the full inventory."""

    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def preflight_conversion(
    model_dir: str | Path,
    generation_config: str | Path = GENERATION_CONFIG_PATH,
) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    source = recipe.preflight_sources(model)
    resources = load_resources(model, generation_config)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    draft: draft_head.DraftHeadContext,
) -> torch.Tensor:
    derived = None
    if spec.name in (
        draft_head.DRAFT_HEAD_OBJECT,
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT,
    ):
        derived = {
            draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
                draft_head.materialize_draft_head_token_ids(draft)
            )
        }
    tensor = recipe.materialize_recipe(
        recipe.RECIPES_BY_NAME[spec.name],
        reader,
        derived,
    )
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized tensor according to its registered signature."""

    return family_conversion.encode_tensor_payload(tensor, spec, device)


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: recipe.SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    ranking_path: str | Path,
    revision: str | None = None,
    environment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the external descriptive conversion report."""

    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking_path,
        revision=revision,
        environment_summary=environment,
    )
    report["source"]["repository"] = BASE_REPOSITORY
    report["source"]["revision"] = BASE_REVISION
    report["source"]["frontend_sha256"] = dict(OFFICIAL_RESOURCE_SHA256)
    return report


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    generation_config: str | Path = GENERATION_CONFIG_PATH,
) -> Path:
    """Run the complete registered conversion and return the report path."""

    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model, generation_config)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    with ShardReader(model) as reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, reader, preflight.draft)
                    payload = encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                print(
                    f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    arguments = {
        "model": str(model_dir),
        "out": str(out_path),
        "device": requested_device,
        "generation_config": str(generation_config),
    }
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=ranking,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=GENERATION_CONFIG_PATH,
        help="generation_config.json to embed (the upstream repository ships none)",
    )
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, generation_config=args.generation_config)


if __name__ == "__main__":
    main()
