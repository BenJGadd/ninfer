from tools.convert.qwen3_5_9b import inventory
from tools.convert.qwen3_5_9b.geometry import GEOMETRY
from tools.convert.qwen3_6_27b import inventory as inventory_27b


def _tensor_by_name() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_identity_and_derived_counts() -> None:
    assert inventory.MODEL_ID == "qwen3.5-9b"
    assert inventory.WEIGHTS_ID == "groupwise-int"
    assert inventory.TARGET_KEY == "qwen3_5_9b"

    g = GEOMETRY
    assert g.full_attention_layers == (3, 7, 11, 15, 19, 23, 27, 31)
    assert len(g.gdn_layers) == 24
    assert len(inventory.TEXT_CORE_TENSOR_SPECS) == inventory.expected_text_core_count(g)
    assert len(inventory.DRAFT_HEAD_TENSOR_SPECS) == 2
    assert len(inventory.MTP_TENSOR_SPECS) == 12
    assert len(inventory.VISION_TENSOR_SPECS) == len(inventory_27b.VISION_TENSOR_SPECS)
    assert len(inventory.OBJECT_SPECS) == len(inventory.TENSOR_SPECS) + 6

    names = [spec.name for spec in inventory.OBJECT_SPECS]
    assert len(names) == len(set(names))
    assert names[6] == "text/token_embedding"
    assert names[-1] == "vision/merger/norm/bias"


def test_same_roles_and_formats_as_the_27b_template() -> None:
    """Every 27B role that exists on a 9B layer index keeps its numeric format."""

    ours = _tensor_by_name()
    theirs = {spec.name: spec for spec in inventory_27b.TENSOR_SPECS}
    for name, spec in ours.items():
        assert name in theirs, name
        assert spec.format == theirs[name].format, name
        assert spec.layout == theirs[name].layout, name
        assert len(spec.shape) == len(theirs[name].shape), name


def test_key_shapes_follow_geometry() -> None:
    g = GEOMETRY
    t = _tensor_by_name()
    assert t["text/token_embedding"].shape == (g.vocab_rows, g.hidden)
    assert t["text/layers/3/attention/query_key"].shape == (g.query_size + g.kv_size, g.hidden)
    assert t["text/layers/3/attention/output"].shape == (g.hidden, g.query_size)
    assert t["text/layers/0/gdn/convolution"].shape == (g.gdn_conv_kernel, g.convolution_dim)
    assert t["text/layers/0/gdn/query_key"].shape == (2 * g.key_dim, g.hidden)
    assert t["text/layers/0/gdn/value_z"].shape == (2 * g.value_dim, g.hidden)
    assert t["text/layers/0/gdn/a_log"].shape == (g.gdn_value_heads,)
    assert t["text/layers/0/mlp/gate_up"].shape == (2 * g.intermediate, g.hidden)
    assert t["mtp/input_projection"].shape == (g.hidden, 2 * g.hidden)
    assert t["mtp/layer/attention/query_key_gate_value"].shape == (
        2 * g.query_size + 2 * g.kv_size,
        g.hidden,
    )
    assert t["vision/merger/fc2"].shape == (g.hidden, g.vision_merger_input)
    assert t["text/draft_head"].shape == (inventory.DRAFT_ROWS, g.hidden)


def test_row_views_partition_their_parents() -> None:
    t = _tensor_by_name()
    by_parent: dict[str, list[inventory.LogicalRowViewSpec]] = {}
    for view in inventory.LOGICAL_ROW_VIEW_SPECS:
        by_parent.setdefault(view.parent_pattern, []).append(view)
    for parent_pattern, views in by_parent.items():
        layers = views[0].layers
        sample = parent_pattern if layers is None else parent_pattern.format(l=layers[0])
        parent_rows = t[sample].shape[0]
        spans = sorted((v.row_begin, v.row_end) for v in views)
        assert spans[0][0] == 0
        assert spans[-1][1] == parent_rows, parent_pattern
        for (_, end), (begin, _) in zip(spans, spans[1:]):
            assert end == begin, parent_pattern
