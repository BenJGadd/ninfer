import json
from pathlib import Path

from tools.convert.qwen3_5_9b import inventory, recipe
from tools.convert.qwen3_5_9b.geometry import GEOMETRY

FIXTURE_INDEX = Path(__file__).with_name("model.safetensors.index.json")


def test_recipe_covers_inventory_in_order() -> None:
    recipe.validate_recipe_coverage()
    assert [r.object_name for r in recipe.RECIPE_SPECS] == [
        s.name for s in inventory.TENSOR_SPECS
    ]


def test_every_checkpoint_tensor_is_consumed_exactly_once() -> None:
    """The recipe's source set equals the published Qwen/Qwen3.5-9B weight map."""

    weight_map = json.loads(FIXTURE_INDEX.read_text())["weight_map"]
    required = recipe.source_requirements()
    assert set(required) == set(weight_map)


def test_gdn_and_attention_slices_match_geometry() -> None:
    g = GEOMETRY
    qk = recipe.RECIPES_BY_NAME["text/layers/0/gdn/query_key"].expression
    assert isinstance(qk, recipe.Slice) and (qk.begin, qk.end) == (0, 2 * g.key_dim)
    vz = recipe.RECIPES_BY_NAME["text/layers/0/gdn/value_z"].expression
    assert isinstance(vz, recipe.Concat)
    first = vz.sources[0]
    assert isinstance(first, recipe.Slice) and (first.begin, first.end) == (
        2 * g.key_dim,
        g.convolution_dim,
    )
    assert recipe.expression_shape(
        recipe.RECIPES_BY_NAME["text/layers/3/attention/query_key"].expression
    ) == (g.query_size + g.kv_size, g.hidden)
