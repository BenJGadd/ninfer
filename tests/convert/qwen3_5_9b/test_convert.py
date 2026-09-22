import json
from pathlib import Path

import pytest

from tools.convert.qwen3_5_9b import convert, inventory
from tools.convert.qwen3_5_9b.geometry import GEOMETRY, Geometry

FIXTURE_CONFIG = Path(__file__).with_name("config.json")


def test_registered_counts_are_derived_not_typed() -> None:
    assert convert.expected_object_counts() == (
        6,
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        2,
        12,
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    )
    convert.preflight_inventory()


def test_published_config_validates() -> None:
    config = json.loads(FIXTURE_CONFIG.read_text())
    summary = convert.validate_config(config)
    assert summary["layer_types"]["full_attention_layers"] == list(GEOMETRY.full_attention_layers)
    assert Geometry.from_hf_config(config) == GEOMETRY


def test_wrong_shape_is_rejected() -> None:
    config = json.loads(FIXTURE_CONFIG.read_text())
    config["text_config"]["hidden_size"] = 5120
    with pytest.raises(ValueError):
        convert.validate_config(config)


def test_generation_config_is_pinned() -> None:
    import hashlib

    data = convert.GENERATION_CONFIG_PATH.read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == convert.OFFICIAL_RESOURCE_SHA256["frontend/generation_config.json"]
    )
    parsed = json.loads(data)
    assert 248046 in parsed["eos_token_id"]  # <|im_end|>
