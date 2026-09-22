"""Hugging Face source recipe for the complete Qwen3.5-9B inventory.

Same transform sequence as the Qwen3.6-27B recipe; every shape comes from
:mod:`tools.convert.qwen3_5_9b.geometry`.
"""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    DraftHeadTokenIds,
    Expression,
    GatherRows,
    Reshape,
    ShardReader,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    build_vision_recipes,
    expression_shape,
    expression_sources,
    materialize_expression,
    materialize_recipe,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements as _recipe_source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)

from . import inventory
from .geometry import GEOMETRY, Geometry


DRAFT_ROWS = inventory.DRAFT_ROWS
TOKENIZER_VOCAB_SIZE = 248077
RANKING_PATH = "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64"


_sources = expression_sources
_source = source


def _attention_qproj_part(g: Geometry, source_name: str, gate: bool) -> Expression:
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=g.query_heads,
        hidden_size=g.hidden,
    )


def build_text_recipes(g: Geometry) -> tuple[TensorRecipe, ...]:
    h = g.hidden
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (g.vocab_rows, h)),
        )
    ]

    for layer in range(g.layers):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (h,)),
            )
        )

        if layer in g.full_attention_layers:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            query = _attention_qproj_part(g, q_proj, gate=False)
            gate = _attention_qproj_part(g, q_proj, gate=True)
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key",
                        Concat(
                            (
                                query,
                                _source(source_prefix + "self_attn.k_proj.weight", (g.kv_size, h)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                gate,
                                _source(source_prefix + "self_attn.v_proj.weight", (g.kv_size, h)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight", (g.head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight", (g.head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight", (h, g.query_size)),
                    ),
                )
            )
        else:
            qkv_source = _source(
                source_prefix + "linear_attn.in_proj_qkv.weight",
                (g.convolution_dim, h),
            )
            convolution = _source(
                source_prefix + "linear_attn.conv1d.weight",
                (g.convolution_dim, 1, g.gdn_conv_kernel),
            )
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(
                            _source(source_prefix + "linear_attn.A_log", (g.gdn_value_heads,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(
                            _source(source_prefix + "linear_attn.dt_bias", (g.gdn_value_heads,)),
                            inventory.FP32,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(
                                Slice(convolution, 1, 0, 1),
                                (g.convolution_dim, g.gdn_conv_kernel),
                            ),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight", (g.gdn_value_heads, h)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight", (g.gdn_value_heads, h)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, g.gdn_query_key_rows),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value_z",
                        Concat(
                            (
                                Slice(qkv_source, 0, g.gdn_query_key_rows, g.convolution_dim),
                                _source(
                                    source_prefix + "linear_attn.in_proj_z.weight",
                                    (g.value_dim, h),
                                ),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        _source(source_prefix + "linear_attn.norm.weight", (g.gdn_head_dim,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight", (h, g.value_dim)),
                    ),
                )
            )

        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight", (h,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight", (g.intermediate, h)),
                            _source(source_prefix + "mlp.up_proj.weight", (g.intermediate, h)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight", (h, g.intermediate)),
                ),
            )
        )

    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (h,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("lm_head.weight", (g.vocab_rows, h)),
            ),
        )
    )
    return tuple(recipes)


def build_draft_head_recipes(g: Geometry) -> tuple[TensorRecipe, ...]:
    return (
        TensorRecipe(
            "text/draft_head",
            GatherRows(
                _source("lm_head.weight", (g.vocab_rows, g.hidden)),
                token_ids_object="text/draft_head_token_ids",
                rows=DRAFT_ROWS,
            ),
        ),
        TensorRecipe(
            "text/draft_head_token_ids",
            DraftHeadTokenIds(
                ranking_path=RANKING_PATH,
                tokenizer_resource="frontend/tokenizer_config.json",
                vocab_rows=g.vocab_rows,
                tokenizer_id_count=TOKENIZER_VOCAB_SIZE,
                rows=DRAFT_ROWS,
            ),
        ),
    )


def build_mtp_recipes(g: Geometry) -> tuple[TensorRecipe, ...]:
    h = g.hidden
    source_prefix = "mtp.layers.0."
    q_proj = source_prefix + "self_attn.q_proj.weight"
    return (
        TensorRecipe("mtp/input_projection", _source("mtp.fc.weight", (h, g.mtp_input_rows))),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (h,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (h,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source(source_prefix + "input_layernorm.weight", (h,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(g, q_proj, gate=False),
                    _source(source_prefix + "self_attn.k_proj.weight", (g.kv_size, h)),
                    _attention_qproj_part(g, q_proj, gate=True),
                    _source(source_prefix + "self_attn.v_proj.weight", (g.kv_size, h)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source(source_prefix + "self_attn.q_norm.weight", (g.head_dim,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source(source_prefix + "self_attn.k_norm.weight", (g.head_dim,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source(source_prefix + "self_attn.o_proj.weight", (h, g.query_size)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source(source_prefix + "post_attention_layernorm.weight", (h,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source(source_prefix + "mlp.gate_proj.weight", (g.intermediate, h)),
                    _source(source_prefix + "mlp.up_proj.weight", (g.intermediate, h)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source(source_prefix + "mlp.down_proj.weight", (h, g.intermediate)),
        ),
        TensorRecipe("mtp/final_norm", _source("mtp.norm.weight", (h,))),
    )


RECIPE_SPECS = (
    build_text_recipes(GEOMETRY)
    + build_draft_head_recipes(GEOMETRY)
    + build_mtp_recipes(GEOMETRY)
    + build_vision_recipes(GEOMETRY.hidden)
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(RECIPE_SPECS, inventory.TENSOR_SPECS)


def source_requirements() -> dict[str, SourceTensor]:
    return _recipe_source_requirements(RECIPE_SPECS)


def preflight_sources(model_dir: str | Path) -> SourcePreflight:
    return _preflight_recipe_sources(model_dir, RECIPE_SPECS)


validate_recipe_coverage()
