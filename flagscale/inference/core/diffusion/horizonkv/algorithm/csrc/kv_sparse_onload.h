// Modified for the HorizonKV inference artifact; see NOTICE.
#pragma once
#include <torch/extension.h>
namespace py = pybind11;

void kv_sparse_frame_materialize_cuda(
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor k_out,
    torch::Tensor v_out,
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor slot_indices,
    torch::Tensor live_k,
    torch::Tensor live_v,
    int64_t frame_seqlen,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    bool has_live);

void kv_sparse_frame_materialize_varlen_cuda(
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor k_out,
    torch::Tensor v_out,
    torch::Tensor head_indices,
    torch::Tensor frame_indices,
    torch::Tensor slot_indices,
    torch::Tensor cu_seqlens_k,
    torch::Tensor live_k,
    torch::Tensor live_v,
    int64_t frame_seqlen,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    bool has_live);

torch::Tensor kv_sparse_select_pooled_qk_topk_cuda(
    torch::Tensor q,
    torch::Tensor frame_k,
    torch::Tensor head_indices,
    torch::Tensor compulsory_indices,
    int64_t frame_seqlen,
    int64_t max_kv_frames);

void kv_sparse_update_frame_slot_map_cuda(
    torch::Tensor frame_to_slot,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor slot_ids,
    torch::Tensor evicted_frame_ids);

int64_t kv_sparse_copy_frame_plan_to_slots_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor frame_indices,
    torch::Tensor slot_plan,
    torch::Tensor copy_plan,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames);



int64_t kv_sparse_copy_frames_to_compact_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor compact_k,
    torch::Tensor compact_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor cu_seqlens,
    int64_t frame_seqlen);

int64_t kv_sparse_copy_kv_span_to_cpu_cuda(
    torch::Tensor gpu_k_span,
    torch::Tensor gpu_v_span,
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    int64_t start_token);

int64_t kv_sparse_copy_compact_to_slots_d2d_cuda(
    torch::Tensor compact_k,
    torch::Tensor compact_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor head_ids,
    torch::Tensor slot_ids,
    torch::Tensor src_rows,
    torch::Tensor src_cols,
    torch::Tensor cu_seqlens,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames);

int64_t kv_sparse_adaptive_count_misses_cpu(
    torch::Tensor frames, torch::Tensor lookup, torch::Tensor ids, torch::Tensor lens,
    int64_t live_first, int64_t live_end, int64_t frame_length);

py::object kv_sparse_plan_slots_tensor_cpp(
    torch::Tensor frame_indices,
    torch::Tensor frame_to_slot,
    torch::Tensor slot_frame_ids,
    torch::Tensor slot_frame_lens,
    torch::Tensor slot_last_used,
    int64_t total_tokens,
    int64_t frame_seqlen,
    int64_t num_heads,
    int64_t gpu_cache_frames,
    int64_t live_start_index,
    int64_t live_len,
    bool has_live,
    int64_t gpu_clock,
    bool build_compact_plans,
    int64_t split_min_miss_items);
