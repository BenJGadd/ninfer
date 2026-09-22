#include "targets/qwen3_5_9b/impl/load/bindings.h"

#include "artifact/typed_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Every extent below is spelled through TextConfig so the artifact contract and the
// compiled geometry cannot drift apart. A literal here would be a port bug.

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

using artifact::NumericFormat;
using C = TextConfig;

constexpr std::uint64_t u64(int value) { return static_cast<std::uint64_t>(value); }

bool is_full_layer(std::size_t layer) { return C::is_full_attention(static_cast<int>(layer)); }

NumericFormat endpoint_format(WeightsProfile weights_profile) {
    switch (weights_profile) {
    case WeightsProfile::Qwen35GroupwiseInt:
        return NumericFormat::Q6G64_F16S;
    }
    throw std::invalid_argument("qwen3_5_9b: invalid weights profile");
}

WeightPlan bind_weight(artifact::Binder& binder, std::string_view name, NumericFormat format,
                       std::initializer_list<std::uint64_t> shape,
                       artifact::TensorPlacement placement = artifact::TensorPlacement::Device) {
    return WeightPlan{.object = artifact::bind_tensor(binder, name, format, shape, placement),
                      .format = format};
}

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           const WeightPlan& plan, std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(materialized, plan.object, plan.format, rows, columns);
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.layout != QuantLayout::RowSplit) {
        throw std::logic_error("invalid target row view");
    }
    const std::uint64_t groups    = static_cast<std::uint64_t>(block.padded_shape[1] / block.group);
    const std::uint64_t low_group = 32;
    const std::uint64_t high_group = block.qtype == QType::Q5G64_F16S   ? 8
                                     : block.qtype == QType::Q6G64_F16S ? 16
                                                                        : 0;
    const std::uint64_t low_row    = groups * low_group;
    const std::uint64_t high_row   = groups * high_group;
    const std::uint64_t scale_row  = groups * 2;
    Weight out                     = block;
    out.qdata                      = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * low_row;
    out.qhigh  = high_group == 0 ? nullptr
                                 : static_cast<const std::byte*>(block.qhigh) +
                                      static_cast<std::uint64_t>(row_begin) * high_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = materialized_weight(materialized, plan.gate_up, C::mlp_gate_up_rows, C::hidden);
    out.down    = materialized_weight(materialized, plan.down, C::hidden, C::intermediate);
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    FullAttentionProjectionPayload out;
    out.query_key =
        materialized_weight(materialized, plan.query_key, C::attention_query_key_rows, C::hidden);
    out.gate_value =
        materialized_weight(materialized, plan.gate_value, C::attention_query_key_rows, C::hidden);
    out.query       = row_view(out.query_key, 0, C::query_size);
    out.key         = row_view(out.query_key, C::query_size, C::kv_size);
    out.output_gate = row_view(out.gate_value, 0, C::query_size);
    out.value       = row_view(out.gate_value, C::query_size, C::kv_size);
    return out;
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    return GdnInputProjectionPayload{
        .query_key_value =
            materialized_weight(materialized, plan.query_key_value, C::convolution_dim, C::hidden),
        .z = materialized_weight(materialized, plan.z, C::value_dim, C::hidden),
    };
}

GdnControlProjectionPayload
load_gdn_control_projection(const GdnPlan& plan,
                            const artifact::MaterializedArtifact& materialized) {
    GdnControlProjectionPayload out;
    out.a_b_projection = materialized_weight(materialized, plan.a_b_projection,
                                             2 * C::gdn_value_heads, C::hidden);
    out.a = row_view(out.a_b_projection, 0, C::gdn_value_heads);
    out.b = row_view(out.a_b_projection, C::gdn_value_heads, C::gdn_value_heads);
    return out;
}

void bind_groupwise_text_layers(artifact::Binder& binder, BindingPlan& out) {
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = artifact::bind_device_tensor(binder, prefix + "input_norm",
                                                                NumericFormat::BF16, {u64(C::hidden)});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key =
                bind_weight(binder, prefix + "attention/query_key", NumericFormat::Q4G64_F16S,
                            {u64(C::attention_query_key_rows), u64(C::hidden)});
            target.attention.gate_value =
                bind_weight(binder, prefix + "attention/gate_value", NumericFormat::Q5G64_F16S,
                            {u64(C::attention_query_key_rows), u64(C::hidden)});
            target.attention.query_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/query_norm", NumericFormat::BF16, {u64(C::head_dim)});
            target.attention.key_norm = artifact::bind_device_tensor(
                binder, prefix + "attention/key_norm", NumericFormat::BF16, {u64(C::head_dim)});
            target.attention.output =
                bind_weight(binder, prefix + "attention/output", NumericFormat::Q5G64_F16S,
                            {u64(C::hidden), u64(C::query_size)});
        } else {
            target.gdn.a_log       = artifact::bind_device_tensor(binder, prefix + "gdn/a_log",
                                                                  NumericFormat::FP32,
                                                                  {u64(C::gdn_value_heads)});
            target.gdn.dt_bias     = artifact::bind_device_tensor(binder, prefix + "gdn/dt_bias",
                                                                  NumericFormat::FP32,
                                                                  {u64(C::gdn_value_heads)});
            target.gdn.convolution = artifact::bind_device_tensor(
                binder, prefix + "gdn/convolution", NumericFormat::BF16,
                {u64(C::gdn_conv_kernel), u64(C::convolution_dim)});
            target.gdn.a_b_projection =
                bind_weight(binder, prefix + "gdn/a_b_projection", NumericFormat::W8G32_F16S,
                            {u64(2 * C::gdn_value_heads), u64(C::hidden)});
            target.gdn.query_key_value =
                bind_weight(binder, prefix + "gdn/query_key_value", NumericFormat::Q5G64_F16S,
                            {u64(C::convolution_dim), u64(C::hidden)});
            target.gdn.z = bind_weight(binder, prefix + "gdn/z", NumericFormat::Q5G64_F16S,
                                       {u64(C::value_dim), u64(C::hidden)});
            target.gdn.norm = artifact::bind_device_tensor(binder, prefix + "gdn/norm",
                                                           NumericFormat::BF16,
                                                           {u64(C::gdn_value_head_dim)});
            target.gdn.output = bind_weight(binder, prefix + "gdn/output", NumericFormat::Q5G64_F16S,
                                            {u64(C::hidden), u64(C::value_dim)});
        }
        target.post_attention_norm = artifact::bind_device_tensor(
            binder, prefix + "post_attention_norm", NumericFormat::BF16, {u64(C::hidden)});
        target.mlp.gate_up = bind_weight(binder, prefix + "mlp/gate_up", NumericFormat::Q4G64_F16S,
                                         {u64(C::mlp_gate_up_rows), u64(C::hidden)});
        target.mlp.down    = bind_weight(binder, prefix + "mlp/down", NumericFormat::Q5G64_F16S,
                                         {u64(C::hidden), u64(C::intermediate)});
    }
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = static_cast<std::size_t>(C::draft_head_rows);
    constexpr std::size_t kTokenizerVocab = static_cast<std::size_t>(C::token_domain);
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[id]) { throw artifact::ArtifactError("draft-head token ids are not unique"); }
        seen[id] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const NumericFormat vocabulary_format = endpoint_format(weights_profile);
    out.token_embedding = bind_weight(binder, "text/token_embedding", vocabulary_format,
                                      {u64(C::output_rows), u64(C::hidden)});
    switch (weights_profile) {
    case WeightsProfile::Qwen35GroupwiseInt:
        bind_groupwise_text_layers(binder, out);
        break;
    default:
        throw std::invalid_argument("qwen3_5_9b: invalid weights profile");
    }
    out.final_norm = artifact::bind_device_tensor(binder, "text/final_norm", NumericFormat::BF16,
                                                  {u64(C::hidden)});
    out.output_head = bind_weight(binder, "text/output_head", vocabulary_format,
                                  {u64(C::output_rows), u64(C::hidden)});
    const artifact::TensorPlacement proposal_placement =
        features.optimized_proposal() ? artifact::TensorPlacement::Device
                                      : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {u64(C::draft_head_rows), u64(C::hidden)},
                                           proposal_placement);
    out.draft_head_token_ids =
        artifact::bind_tensor(binder, "text/draft_head_token_ids", NumericFormat::I32,
                              {u64(C::draft_head_rows)}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement = features.mtp()
                                                        ? artifact::TensorPlacement::Device
                                                        : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement);
    };
    out.mtp.input_projection = bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S,
                                        {u64(C::hidden), u64(C::mtp_input_rows)});
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {u64(C::hidden)});
    out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {u64(C::hidden)});
    out.mtp.input_norm = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {u64(C::hidden)});
    out.mtp.query_key_gate_value =
        bind_mtp("mtp/layer/attention/query_key_gate_value", NumericFormat::W8G32_F16S,
                 {u64(C::mtp_attention_input_rows), u64(C::hidden)});
    out.mtp.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {u64(C::head_dim)});
    out.mtp.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {u64(C::head_dim)});
    out.mtp.output = bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S,
                              {u64(C::hidden), u64(C::query_size)});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {u64(C::hidden)});
    out.mtp.mlp.gate_up = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/gate_up", NumericFormat::W8G32_F16S,
                           {u64(C::mtp_mlp_gate_up_rows), u64(C::hidden)}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.mlp.down = WeightPlan{
        .object = bind_mtp("mtp/layer/mlp/down", NumericFormat::W8G32_F16S,
                           {u64(C::hidden), u64(C::intermediate)}),
        .format = NumericFormat::W8G32_F16S};
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {u64(C::hidden)});

    const artifact::TensorPlacement vision_placement =
        features.vision ? artifact::TensorPlacement::Device
                        : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2 =
        artifact::bind_tensor(binder, "vision/merger/fc2", NumericFormat::W8G32_F16S,
                              {u64(C::hidden), u64(qwen3_6::VisionBackboneConfig::merger_hidden)},
                              vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {u64(C::hidden)}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    if (features.dflash2() || features.dflash()) {
        throw artifact::ArtifactError("qwen3_5_9b has no masked-draft (DFlash) weight bundle");
    }

    load_plan.materialization = binder.finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding =
        materialized_weight(backing, plan.token_embedding, C::output_rows, C::hidden);
    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm = artifact::materialized_tensor(backing, source.input_norm,
                                                              NumericFormat::BF16, {C::hidden});
            target.projection = load_attention_projection(source.attention, backing);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {C::head_dim});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {C::head_dim});
            target.output =
                materialized_weight(backing, source.attention.output, C::hidden, C::query_size);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {C::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {C::hidden});
            target.projection.a_log = artifact::materialized_tensor(
                backing, source.gdn.a_log, NumericFormat::FP32, {C::gdn_value_heads});
            target.projection.dt_bias = artifact::materialized_tensor(
                backing, source.gdn.dt_bias, NumericFormat::FP32, {C::gdn_value_heads});
            // Stored [taps, channels]; the runtime consumes the channel-major alias.
            target.convolution = artifact::materialized_tensor(
                backing, source.gdn.convolution, NumericFormat::BF16,
                {C::convolution_dim, C::gdn_conv_kernel});
            target.projection.control_projection = load_gdn_control_projection(source.gdn, backing);
            target.projection.input_projection   = load_gdn_input_projection(source.gdn, backing);
            target.norm = artifact::materialized_tensor(backing, source.gdn.norm,
                                                        NumericFormat::BF16, {C::gdn_value_head_dim});
            target.output = materialized_weight(backing, source.gdn.output, C::hidden, C::value_dim);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {C::hidden});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {C::hidden});
    output_head = materialized_weight(backing, plan.output_head, C::output_rows, C::hidden);
    if (plan.features.optimized_proposal()) {
        auto& proposal = runtime.optimized_proposal.emplace();
        proposal.head  = artifact::materialized_weight(backing, plan.draft_head,
                                                       NumericFormat::Q4G64_F16S,
                                                       C::draft_head_rows, C::hidden);
        proposal.token_ids = artifact::materialized_tensor(
            backing, plan.draft_head_token_ids, NumericFormat::I32, {C::draft_head_rows});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(
            backing, plan.mtp.input_projection, NumericFormat::W8G32_F16S, C::hidden,
            C::mtp_input_rows);
        mtp.embedding_norm = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                           NumericFormat::BF16, {C::hidden});
        mtp.hidden_norm    = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                           NumericFormat::BF16, {C::hidden});
        mtp.input_norm     = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                           NumericFormat::BF16, {C::hidden});
        mtp.attention.packed = artifact::materialized_weight(
            backing, plan.mtp.query_key_gate_value, NumericFormat::W8G32_F16S,
            C::mtp_attention_input_rows, C::hidden);
        // Packed row order: query | key | output_gate | value.
        mtp.attention.query = row_view(mtp.attention.packed, 0, C::query_size);
        mtp.attention.key   = row_view(mtp.attention.packed, C::query_size, C::kv_size);
        mtp.attention.output_gate =
            row_view(mtp.attention.packed, C::query_size + C::kv_size, C::query_size);
        mtp.attention.value =
            row_view(mtp.attention.packed, 2 * C::query_size + C::kv_size, C::kv_size);
        mtp.query_norm = artifact::materialized_tensor(backing, plan.mtp.query_norm,
                                                       NumericFormat::BF16, {C::head_dim});
        mtp.key_norm   = artifact::materialized_tensor(backing, plan.mtp.key_norm,
                                                       NumericFormat::BF16, {C::head_dim});
        mtp.output     = artifact::materialized_weight(backing, plan.mtp.output,
                                                       NumericFormat::W8G32_F16S, C::hidden,
                                                       C::query_size);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {C::hidden});
        mtp.post_mixer = load_mlp(plan.mtp.mlp, backing);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {C::hidden});
    }

    if (plan.features.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2 = artifact::materialized_weight(
            backing, plan.vision_merger_fc2, NumericFormat::W8G32_F16S, C::hidden,
            qwen3_6::VisionBackboneConfig::merger_hidden);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {C::hidden});
    }
}

} // namespace ninfer::targets::qwen3_5_9b::detail
