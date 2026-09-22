"""Persistent-object contract for the complete Qwen3.5-9B artifact.

Storage roles, numeric formats and writer order mirror the Qwen3.6-27B
groupwise-int artifact exactly; every shape is derived from
:mod:`tools.convert.qwen3_5_9b.geometry`.  Source-checkpoint mapping lives in
the sibling recipe.
"""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    I32,
    LAYOUT_NAMES,
    LogicalAliasSpec,
    LogicalRowViewSpec,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs,
    tensor_spec,
)

from .geometry import GEOMETRY, Geometry


MODEL_ID = "qwen3.5-9b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "qwen3_5_9b"

DRAFT_ROWS = 131072

FULL_ATTENTION_LAYERS = GEOMETRY.full_attention_layers
GDN_LAYERS = GEOMETRY.gdn_layers


_tensor = tensor_spec


def build_text_core_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (g.vocab_rows, g.hidden), Q6),
    ]

    for layer in range(g.layers):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (g.hidden,), BF16))

        if layer in g.full_attention_layers:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key", (g.query_key_rows, g.hidden), Q4),
                    _tensor(prefix + "attention/gate_value", (g.query_key_rows, g.hidden), Q5),
                    _tensor(prefix + "attention/query_norm", (g.head_dim,), BF16),
                    _tensor(prefix + "attention/key_norm", (g.head_dim,), BF16),
                    _tensor(prefix + "attention/output", (g.hidden, g.query_size), Q5),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (g.gdn_value_heads,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (g.gdn_value_heads,), FP32),
                    _tensor(
                        prefix + "gdn/convolution",
                        (g.gdn_conv_kernel, g.convolution_dim),
                        BF16,
                    ),
                    # The 9B engine leaves are plain-linear compositions, so the GDN parents are
                    # the checkpoint's own in_proj_qkv / in_proj_z matrices and one fused a|b
                    # control parent (W8: the BF16 linear catalog has no [32,4096] route).
                    _tensor(prefix + "gdn/a_b_projection", (2 * g.gdn_value_heads, g.hidden), W8),
                    _tensor(prefix + "gdn/query_key_value", (g.convolution_dim, g.hidden), Q5),
                    _tensor(prefix + "gdn/z", (g.value_dim, g.hidden), Q5),
                    _tensor(prefix + "gdn/norm", (g.gdn_head_dim,), BF16),
                    _tensor(prefix + "gdn/output", (g.hidden, g.value_dim), Q5),
                )
            )

        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (g.hidden,), BF16),
                _tensor(prefix + "mlp/gate_up", (g.gate_up_rows, g.hidden), Q4),
                _tensor(prefix + "mlp/down", (g.hidden, g.intermediate), Q5),
            )
        )

    specs.extend(
        (
            _tensor("text/final_norm", (g.hidden,), BF16),
            _tensor("text/output_head", (g.vocab_rows, g.hidden), Q6),
        )
    )
    return tuple(specs)


def build_draft_head_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    return (
        _tensor("text/draft_head", (DRAFT_ROWS, g.hidden), Q4),
        _tensor("text/draft_head_token_ids", (DRAFT_ROWS,), I32),
    )


def build_mtp_specs(g: Geometry) -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (g.hidden, g.mtp_input_rows), W8),
        _tensor("mtp/embedding_norm", (g.hidden,), BF16),
        _tensor("mtp/hidden_norm", (g.hidden,), BF16),
        _tensor("mtp/layer/input_norm", (g.hidden,), BF16),
        _tensor(
            "mtp/layer/attention/query_key_gate_value",
            (g.mtp_attention_input_rows, g.hidden),
            W8,
        ),
        _tensor("mtp/layer/attention/query_norm", (g.head_dim,), BF16),
        _tensor("mtp/layer/attention/key_norm", (g.head_dim,), BF16),
        _tensor("mtp/layer/attention/output", (g.hidden, g.query_size), W8),
        _tensor("mtp/layer/post_attention_norm", (g.hidden,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (g.gate_up_rows, g.hidden), W8),
        _tensor("mtp/layer/mlp/down", (g.hidden, g.intermediate), W8),
        _tensor("mtp/final_norm", (g.hidden,), BF16),
    )


def build_row_view_specs(g: Geometry) -> tuple[LogicalRowViewSpec, ...]:
    full = g.full_attention_layers
    gdn = g.gdn_layers
    every = tuple(range(g.layers))
    q, kv, h = g.query_size, g.kv_size, g.hidden
    kd, vd = g.key_dim, g.value_dim
    mid = g.intermediate
    return (
        LogicalRowViewSpec("text/layers/{l}/attention/query", "text/layers/{l}/attention/query_key", 0, q, (q, h), full),
        LogicalRowViewSpec("text/layers/{l}/attention/key", "text/layers/{l}/attention/query_key", q, q + kv, (kv, h), full),
        LogicalRowViewSpec("text/layers/{l}/attention/output_gate", "text/layers/{l}/attention/gate_value", 0, q, (q, h), full),
        LogicalRowViewSpec("text/layers/{l}/attention/value", "text/layers/{l}/attention/gate_value", q, q + kv, (kv, h), full),
        LogicalRowViewSpec("text/layers/{l}/gdn/query", "text/layers/{l}/gdn/query_key_value", 0, kd, (kd, h), gdn),
        LogicalRowViewSpec("text/layers/{l}/gdn/key", "text/layers/{l}/gdn/query_key_value", kd, 2 * kd, (kd, h), gdn),
        LogicalRowViewSpec("text/layers/{l}/gdn/value", "text/layers/{l}/gdn/query_key_value", 2 * kd, 2 * kd + vd, (vd, h), gdn),
        LogicalRowViewSpec("text/layers/{l}/gdn/a_projection", "text/layers/{l}/gdn/a_b_projection", 0, g.gdn_value_heads, (g.gdn_value_heads, h), gdn),
        LogicalRowViewSpec("text/layers/{l}/gdn/b_projection", "text/layers/{l}/gdn/a_b_projection", g.gdn_value_heads, 2 * g.gdn_value_heads, (g.gdn_value_heads, h), gdn),
        LogicalRowViewSpec("text/layers/{l}/mlp/gate", "text/layers/{l}/mlp/gate_up", 0, mid, (mid, h), every),
        LogicalRowViewSpec("text/layers/{l}/mlp/up", "text/layers/{l}/mlp/gate_up", mid, 2 * mid, (mid, h), every),
        LogicalRowViewSpec("mtp/layer/attention/query", "mtp/layer/attention/query_key_gate_value", 0, q, (q, h), None),
        LogicalRowViewSpec("mtp/layer/attention/key", "mtp/layer/attention/query_key_gate_value", q, q + kv, (kv, h), None),
        LogicalRowViewSpec("mtp/layer/attention/output_gate", "mtp/layer/attention/query_key_gate_value", q + kv, 2 * q + kv, (q, h), None),
        LogicalRowViewSpec("mtp/layer/attention/value", "mtp/layer/attention/query_key_gate_value", 2 * q + kv, 2 * q + 2 * kv, (kv, h), None),
        LogicalRowViewSpec("mtp/layer/mlp/gate", "mtp/layer/mlp/gate_up", 0, mid, (mid, h), None),
        LogicalRowViewSpec("mtp/layer/mlp/up", "mtp/layer/mlp/gate_up", mid, 2 * mid, (mid, h), None),
    )


def build_alias_specs(g: Geometry) -> tuple[LogicalAliasSpec, ...]:
    return (
        LogicalAliasSpec("mtp/token_embedding", ("text/token_embedding",)),
        LogicalAliasSpec("mtp/full_output_head", ("text/output_head",)),
        LogicalAliasSpec(
            "mtp/optimized_proposal_head",
            ("text/draft_head", "text/draft_head_token_ids"),
        ),
        LogicalAliasSpec(
            "text/layers/{l}/gdn/channel_major_convolution",
            ("text/layers/{l}/gdn/convolution",),
            layers=g.gdn_layers,
            axis_order=(1, 0),
        ),
    )


def expected_text_core_count(g: Geometry) -> int:
    """Independent count of text-core objects, used to cross-check the built tuple."""
    full = len(g.full_attention_layers)
    gdn = len(g.gdn_layers)
    # embedding + per layer (input_norm, post_attention_norm, gate_up, down) + mixer objects + final_norm + output_head
    return 1 + g.layers * 4 + full * 5 + gdn * 8 + 2


TEXT_CORE_TENSOR_SPECS = build_text_core_specs(GEOMETRY)
DRAFT_HEAD_TENSOR_SPECS = build_draft_head_specs(GEOMETRY)
MTP_TENSOR_SPECS = build_mtp_specs(GEOMETRY)
VISION_TENSOR_SPECS = build_vision_specs(GEOMETRY.hidden)

TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = build_row_view_specs(GEOMETRY)
ALIAS_SPECS = build_alias_specs(GEOMETRY)

if len(TEXT_CORE_TENSOR_SPECS) != expected_text_core_count(GEOMETRY):
    raise ValueError("Qwen3.5-9B text-core inventory does not match its layer schedule")
