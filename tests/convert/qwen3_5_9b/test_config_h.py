"""The C++ target's config.h must carry the same primitives as geometry.py."""

import re
from pathlib import Path

from tools.convert.qwen3_5_9b.geometry import GEOMETRY

CONFIG_H = (
    Path(__file__).resolve().parents[3] / "src/targets/qwen3_5_9b/impl/config.h"
)

_PRIMITIVES = {
    "hidden": "hidden",
    "layers": "layers",
    "intermediate": "intermediate",
    "output_rows": "vocab_rows",
    "gdn_conv_kernel": "gdn_conv_kernel",
    "gdn_key_heads": "gdn_key_heads",
    "gdn_key_head_dim": "gdn_head_dim",
    "gdn_value_heads": "gdn_value_heads",
    "gdn_value_head_dim": "gdn_head_dim",
    "query_heads": "query_heads",
    "kv_heads": "kv_heads",
    "head_dim": "head_dim",
}


def _constants() -> dict[str, int]:
    text = CONFIG_H.read_text()
    block = re.search(r"struct TextConfig \{(.*?)\n\};", text, re.S)
    assert block, "TextConfig block not found"
    found: dict[str, int] = {}
    for match in re.finditer(r"static constexpr int (\w+)\s*=\s*(\d+);", block.group(1)):
        found[match.group(1)] = int(match.group(2))
    return found


def test_config_h_primitives_match_geometry() -> None:
    constants = _constants()
    for cpp_name, py_name in _PRIMITIVES.items():
        assert cpp_name in constants, cpp_name
        assert constants[cpp_name] == getattr(GEOMETRY, py_name), cpp_name


def test_config_h_layer_schedule_matches_geometry() -> None:
    text = CONFIG_H.read_text()
    full = re.search(r"full_attention_layers\(\) == (\d+)\)", text)
    gdn = re.search(r"gdn_layers\(\) == (\d+)\)", text)
    assert full and int(full.group(1)) == len(GEOMETRY.full_attention_layers)
    assert gdn and int(gdn.group(1)) == len(GEOMETRY.gdn_layers)
