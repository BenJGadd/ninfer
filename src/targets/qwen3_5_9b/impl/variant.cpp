#include "targets/qwen3_5_9b/impl/variant.h"

#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_5_9b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_5_9b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

// Every leaf here is a composition of shape-generic public Ops (`linear`, `rmsnorm`,
// `silu_mul`, `residual_add`, `gdn_gating`) plus the family's projected causal-convolution
// launcher. The fused Ops the 27B/35B leaves call (`attn_input_proj`, `gdn_input_proj`,
// `linear_swiglu`, `linear_add`, `linear_pair`, `gdn_norm_gating_proj`) are closed catalogs
// of exactly those two shapes; composing keeps this target out of the kernel-tuning tables.
// The cost is extra launches and a BF16 intermediate per fused group, which is the untuned
// baseline documented in docs/maintainer/qwen3.5-9b-port.md.

namespace ninfer::targets::qwen3_5_9b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

void require_groupwise(WeightsProfile weights_profile) {
    if (weights_profile != WeightsProfile::Qwen35GroupwiseInt) {
        throw std::logic_error("invalid Qwen3.5-9B weights profile");
    }
}

// Column count of a contiguous activation regardless of how the caller shaped its trailing axes.
std::int32_t columns_of(const Tensor& t) {
    return static_cast<std::int32_t>(static_cast<std::int64_t>(t.ne[1]) * t.ne[2] * t.ne[3]);
}

// `ops::linear` takes strictly two-dimensional operands; leaves receive head views or
// width x batch tensors, so re-view them over the same storage.
Tensor as_matrix(const Tensor& t, std::int32_t rows) {
    if (!t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument("qwen3_5_9b leaf operand must be contiguous and non-null");
    }
    return Tensor(t.data, t.dtype, {rows, columns_of(t)});
}

void linear_into(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    Tensor x_matrix   = as_matrix(x, w.k);
    Tensor out_matrix = as_matrix(out, w.n);
    ops::linear(x_matrix, w, out_matrix, stream);
}

void linear_add_into(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {w.n, columns_of(x)});
    linear_into(x, w, delta, stream);
    Tensor residual_matrix = as_matrix(residual, w.n);
    ops::residual_add(delta, residual_matrix, stream);
}

void swiglu_mlp(const Tensor& hidden, const DensePostMixerPayload& weights, Tensor& residual,
                WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope     = workspace.scope();
    const int cols = columns_of(hidden);
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mlp_gate_up_rows, cols});
    linear_into(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    linear_into(activation, weights.down, delta, stream);
    Tensor residual_matrix = as_matrix(residual, TextConfig::hidden);
    ops::residual_add(delta, residual_matrix, stream);
}

std::size_t swiglu_mlp_workspace_bytes(std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mlp_gate_up_rows, last});
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return layout.peak_bytes(1);
}

std::size_t delta_workspace_bytes(std::int32_t rows, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, last});
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    // These frontier ranges are KV-attention properties shared with the 27B dense target.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile>
Variant::dflash_graph_profiles(std::uint32_t, std::uint32_t, std::uint32_t) {
    throw std::invalid_argument("Qwen3.5-9B has no masked-draft (DFlash) execution route");
}

// ---- full attention -----------------------------------------------------------------------

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    linear_into(hidden, weights.query, query, stream);
    linear_into(hidden, weights.key, key, stream);
    linear_into(hidden, weights.output_gate, gate, stream);
    linear_into(hidden, weights.value, value, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    linear_add_into(attention, weight, residual, workspace, stream);
}

// ---- MTP ------------------------------------------------------------------------------------

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value, WorkspaceArena&,
                                       cudaStream_t stream) {
    linear_into(hidden, weights.query, query, stream);
    linear_into(hidden, weights.key, key, stream);
    linear_into(hidden, weights.output_gate, gate, stream);
    linear_into(hidden, weights.value, value, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena&, cudaStream_t stream) {
    linear_into(hidden, weights.key, key, stream);
    linear_into(hidden, weights.value, value, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena&, cudaStream_t stream) {
    linear_into(hidden, weights.query, query, stream);
    linear_into(hidden, weights.output_gate, gate, stream);
}

// ---- gated DeltaNet ----------------------------------------------------------------------------

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase,
                                   WorkspaceArena&, cudaStream_t stream) {
    linear_into(hidden, weights.input_projection.query_key_value, qkv, stream);
    linear_into(hidden, weights.input_projection.z, output_gate, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    // hidden is [hidden, width, batch]; project the width*batch columns in one pass, then let
    // the family convolution launcher consume the [channels, width, batch] result in place.
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    auto scope               = workspace.scope();
    Tensor projected = workspace.alloc(DType::BF16, {TextConfig::convolution_dim, width * batch});
    linear_into(hidden, weights.input_projection.query_key_value, projected, stream);
    linear_into(hidden, weights.input_projection.z, output_gate, stream);
    Tensor projected_view(projected.data, DType::BF16, {TextConfig::convolution_dim, width, batch});
    ops::detail::gdn_projected_conv_snapshot_launch(projected_view, conv_weight, conv_states,
                                                    valid_columns, initial_slot,
                                                    snapshot_base_slot, query, key, value, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase,
                                          WorkspaceArena&, cudaStream_t stream) {
    // The record row is the projection itself: write it directly, then run the convolution
    // in record mode, which leaves the persistent state untouched.
    linear_into(hidden, weights.input_projection.query_key_value, conv_record, stream);
    linear_into(hidden, weights.input_projection.z, output_gate, stream);
    ops::detail::gdn_projected_conv_record_launch(conv_record, conv_weight, conv_states,
                                                  valid_columns, initial_slots, query, key, value,
                                                  stream);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase, WorkspaceArena& workspace,
                                    cudaStream_t stream) {
    linear_add_into(hidden, weight, residual, workspace, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace,
                                          DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    auto scope                = workspace.scope();
    const int cols            = columns_of(residual);
    // Pre-norm with the family's zero-centred (1+w) gain, then the two control projections and
    // the elementwise gate preparation.
    ops::rmsnorm(residual, norm_weight, eps, true, hidden, stream);
    Tensor a = workspace.alloc(DType::BF16, {TextConfig::gdn_value_heads, cols});
    Tensor b = workspace.alloc(DType::BF16, {TextConfig::gdn_value_heads, cols});
    linear_into(hidden, weights.control_projection.a, a, stream);
    linear_into(hidden, weights.control_projection.b, b, stream);
    ops::gdn_gating(a, b, weights.a_log, weights.dt_bias, g, beta, stream);
}

// ---- dense MLP ------------------------------------------------------------------------------

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase, WorkspaceArena& workspace, cudaStream_t stream) {
    swiglu_mlp(hidden, weights, residual, workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream) {
    swiglu_mlp(hidden, weights, residual, workspace, stream);
}

// ---- workspace capacities (mirror the allocations above exactly) ----------------------------

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return 0;
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return delta_workspace_bytes(TextConfig::hidden, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return delta_workspace_bytes(TextConfig::convolution_dim, batch_size * last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return delta_workspace_bytes(TextConfig::hidden, last);
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::gdn_value_heads, last});
    (void)layout.alloc(DType::BF16, {TextConfig::gdn_value_heads, last});
    return layout.peak_bytes(1);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    require_groupwise(weights_profile);
    return swiglu_mlp_workspace_bytes(last);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    return swiglu_mlp_workspace_bytes(last);
}

} // namespace ninfer::targets::qwen3_5_9b::detail
