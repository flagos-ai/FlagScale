// Modified for the HorizonKV inference artifact; see NOTICE.
#include "kv_sparse_onload.h"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
constexpr int64_t kLiveSlot = -2;

template <typename scalar_t>
__global__ void kv_sparse_frame_materialize_kernel(
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    int64_t total_elements) {
  for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
       linear < total_elements;
       linear += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t t = linear;
    const int64_t d = t % head_dim;
    t /= head_dim;
    const int64_t token_in_frame = t % frame_seqlen;
    t /= frame_seqlen;
    const int64_t col = t % n_selected;
    t /= n_selected;
    const int64_t row = t % head_count;
    const int64_t b = t / head_count;

    const int64_t pair_idx = row * n_selected + col;
    const int64_t head_id = head_indices[row];
    const int64_t frame_id = frame_indices[pair_idx];
    const int64_t out_token = col * frame_seqlen + token_in_frame;
    const int64_t out_offset =
        ((b * head_count + row) * (n_selected * frame_seqlen) + out_token) * head_dim + d;

    scalar_t k_value = scalar_t(0);
    scalar_t v_value = scalar_t(0);

    if (head_id >= 0 && head_id < num_heads && frame_id >= 0) {
      const int64_t frame_start = frame_id * frame_seqlen;
      if (frame_start < total_tokens) {
        const int64_t remaining = total_tokens - frame_start;
        const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
        if (frame_len > 0) {
          const int64_t src_in_frame =
              token_in_frame < frame_len ? token_in_frame : frame_len - 1;
          const int64_t slot = slot_indices[pair_idx];
          if (slot == kLiveSlot && has_live) {
            const int64_t live_pos = frame_start - live_start_index + src_in_frame;
            if (live_pos >= 0 && live_pos < live_len) {
              const int64_t live_offset =
                  ((b * live_len + live_pos) * num_heads + head_id) * head_dim + d;
              k_value = live_k[live_offset];
              v_value = live_v[live_offset];
            }
          } else if (slot >= 0) {
            const int64_t slot_token =
                (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
            const int64_t gpu_offset =
                ((b * gpu_cache_tokens + slot_token) * head_dim) + d;
            k_value = gpu_k[gpu_offset];
            v_value = gpu_v[gpu_offset];
          }
        }
      }
    }

    k_out[out_offset] = k_value;
    v_out[out_offset] = v_value;
  }
}

template <typename scalar_t>
__global__ void kv_sparse_frame_materialize_varlen_tiled_kernel(
    const scalar_t* __restrict__ gpu_k,
    const scalar_t* __restrict__ gpu_v,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    const int64_t* __restrict__ head_indices,
    const int64_t* __restrict__ frame_indices,
    const int64_t* __restrict__ slot_indices,
    const int32_t* __restrict__ cu_seqlens_k,
    const scalar_t* __restrict__ live_k,
    const scalar_t* __restrict__ live_v,
    int64_t B,
    int64_t total_tokens,
    int64_t num_heads,
    int64_t head_dim,
    int64_t head_count,
    int64_t n_selected,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames,
    int64_t gpu_cache_tokens,
    int64_t live_len,
    int64_t live_start_index,
    bool has_live,
    int64_t elems_per_segment) {
  const int64_t pair_linear = static_cast<int64_t>(blockIdx.x);
  int64_t pair = pair_linear;
  const int64_t col = pair % n_selected;
  pair /= n_selected;
  const int64_t row = pair % head_count;
  const int64_t b = pair / head_count;
  const int64_t elem = static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (elem >= elems_per_segment || b >= B) {
    return;
  }

  const int64_t pair_idx = row * n_selected + col;
  const int64_t head_id = head_indices[row];
  const int64_t frame_id = frame_indices[pair_idx];
  if (head_id < 0 || head_id >= num_heads || frame_id < 0) {
    return;
  }

  const int64_t row_linear = b * head_count + row;
  const int64_t row_start = static_cast<int64_t>(cu_seqlens_k[row_linear]);
  const int64_t row_end = static_cast<int64_t>(cu_seqlens_k[row_linear + 1]);
  const int64_t dst_token = col * frame_seqlen + elem / head_dim;
  if (row_start + dst_token >= row_end) {
    return;
  }

  const int64_t token_in_frame = elem / head_dim;
  const int64_t d = elem - token_in_frame * head_dim;
  const int64_t frame_start = frame_id * frame_seqlen;
  if (frame_start >= total_tokens) {
    return;
  }
  const int64_t remaining = total_tokens - frame_start;
  const int64_t frame_len = frame_seqlen < remaining ? frame_seqlen : remaining;
  if (frame_len <= 0) {
    return;
  }
  const int64_t src_in_frame = token_in_frame < frame_len ? token_in_frame : frame_len - 1;
  const int64_t slot = slot_indices[pair_idx];

  scalar_t k_value = scalar_t(0);
  scalar_t v_value = scalar_t(0);
  if (slot == kLiveSlot && has_live) {
    const int64_t live_pos = frame_start - live_start_index + src_in_frame;
    if (live_pos >= 0 && live_pos < live_len) {
      const int64_t live_offset =
          ((b * live_len + live_pos) * num_heads + head_id) * head_dim + d;
      k_value = live_k[live_offset];
      v_value = live_v[live_offset];
    }
  } else if (slot >= 0) {
    const int64_t slot_token =
        (head_id * gpu_cache_frames + slot) * frame_seqlen + src_in_frame;
    const int64_t gpu_offset =
        ((b * gpu_cache_tokens + slot_token) * head_dim) + d;
    k_value = gpu_k[gpu_offset];
    v_value = gpu_v[gpu_offset];
  }

  const int64_t out_offset = (row_start + dst_token) * head_dim + d;
  k_out[out_offset] = k_value;
  v_out[out_offset] = v_value;
}

__global__ void kv_sparse_update_frame_slot_map_kernel(
    int64_t* __restrict__ frame_to_slot,
    const int64_t* __restrict__ head_ids,
    const int64_t* __restrict__ frame_ids,
    const int64_t* __restrict__ slot_ids,
    const int64_t* __restrict__ evicted_frame_ids,
    int64_t n,
    int64_t num_heads,
    int64_t frame_capacity) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t head_id = head_ids[i];
    if (head_id < 0 || head_id >= num_heads) {
      continue;
    }
    const int64_t evicted = evicted_frame_ids[i];
    if (evicted >= 0 && evicted < frame_capacity) {
      frame_to_slot[head_id * frame_capacity + evicted] = -1;
    }
    const int64_t frame_id = frame_ids[i];
    const int64_t slot = slot_ids[i];
    if (frame_id >= 0 && frame_id < frame_capacity && slot >= 0) {
      frame_to_slot[head_id * frame_capacity + frame_id] = slot;
    }
  }
}


}  // namespace

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
    bool has_live) {
  TORCH_CHECK(k_out.is_cuda() && v_out.is_cuda(), "k_out/v_out must be CUDA tensors");
  TORCH_CHECK(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  TORCH_CHECK(head_indices.is_cuda() && frame_indices.is_cuda() && slot_indices.is_cuda(),
               "indices must be CUDA tensors");
  TORCH_CHECK(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  TORCH_CHECK(k_out.is_contiguous() && v_out.is_contiguous(), "outputs must be contiguous");
  TORCH_CHECK(head_indices.is_contiguous() && frame_indices.is_contiguous() && slot_indices.is_contiguous(),
               "indices must be contiguous");
  TORCH_CHECK(gpu_k.scalar_type() == gpu_v.scalar_type(), "gpu KV dtype mismatch");
  TORCH_CHECK(gpu_k.scalar_type() == k_out.scalar_type(), "gpu/output dtype mismatch");
  TORCH_CHECK(gpu_k.scalar_type() == v_out.scalar_type(), "gpu/output dtype mismatch");
  TORCH_CHECK(frame_indices.dim() == 2, "frame_indices must be [head_count,n_selected]");
  TORCH_CHECK(slot_indices.sizes() == frame_indices.sizes(), "slot_indices shape mismatch");
  TORCH_CHECK(head_indices.dim() == 1 && head_indices.size(0) == frame_indices.size(0),
               "head_indices shape mismatch");
  TORCH_CHECK(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  TORCH_CHECK(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  TORCH_CHECK(slot_indices.scalar_type() == at::kLong, "slot_indices must be int64");

  const int64_t B = gpu_k.size(0);
  const int64_t D = gpu_k.size(3);
  const int64_t head_count = frame_indices.size(0);
  const int64_t n_selected = frame_indices.size(1);
  TORCH_CHECK(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  TORCH_CHECK(k_out.size(0) == B * head_count, "k_out batch/head dimension mismatch");
  TORCH_CHECK(v_out.sizes() == k_out.sizes(), "v_out shape mismatch");
  TORCH_CHECK(k_out.size(1) == n_selected * frame_seqlen, "output token dimension mismatch");
  TORCH_CHECK(k_out.size(2) == 1 && k_out.size(3) == D, "output head/head_dim mismatch");
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(gpu_k.size(2) == 1 && gpu_k.size(3) == D, "gpu KV shape mismatch");
  if (has_live) {
    TORCH_CHECK(live_k.is_cuda() && live_v.is_cuda(), "live KV must be CUDA tensors");
    TORCH_CHECK(live_k.is_contiguous() && live_v.is_contiguous(), "live KV must be contiguous");
    TORCH_CHECK(live_k.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    TORCH_CHECK(live_v.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    TORCH_CHECK(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
    TORCH_CHECK(live_k.size(0) == B && live_k.size(2) == num_heads && live_k.size(3) == D,
                 "live KV shape mismatch");
  }

  const int64_t total_elements = B * head_count * n_selected * frame_seqlen * D;
  if (total_elements <= 0) {
    return;
  }
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t live_len = has_live ? live_k.size(1) : 0;

  const c10::cuda::CUDAGuard device_guard(k_out.device());
  const int threads = 256;
  const int64_t blocks64 = (total_elements + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(blocks64, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gpu_k.scalar_type(),
                                  "kv_sparse_frame_materialize_cuda", [&] {
      kv_sparse_frame_materialize_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          gpu_k.data_ptr<scalar_t>(),
          gpu_v.data_ptr<scalar_t>(),
          k_out.data_ptr<scalar_t>(),
          v_out.data_ptr<scalar_t>(),
          head_indices.data_ptr<int64_t>(),
          frame_indices.data_ptr<int64_t>(),
          slot_indices.data_ptr<int64_t>(),
          has_live ? live_k.data_ptr<scalar_t>() : nullptr,
          has_live ? live_v.data_ptr<scalar_t>() : nullptr,
          B,
          total_tokens,
          num_heads,
          D,
          head_count,
          n_selected,
          frame_seqlen,
          gpu_cache_frames,
          gpu_cache_tokens,
          live_len,
          live_start_index,
          has_live,
          total_elements);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    bool has_live) {
  TORCH_CHECK(k_out.is_cuda() && v_out.is_cuda(), "k_out/v_out must be CUDA tensors");
  TORCH_CHECK(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  TORCH_CHECK(head_indices.is_cuda() && frame_indices.is_cuda() && slot_indices.is_cuda(),
               "indices must be CUDA tensors");
  TORCH_CHECK(cu_seqlens_k.is_cuda(), "cu_seqlens_k must be CUDA tensor");
  TORCH_CHECK(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  TORCH_CHECK(k_out.is_contiguous() && v_out.is_contiguous(), "outputs must be contiguous");
  TORCH_CHECK(head_indices.is_contiguous() && frame_indices.is_contiguous() && slot_indices.is_contiguous(),
               "indices must be contiguous");
  TORCH_CHECK(cu_seqlens_k.is_contiguous(), "cu_seqlens_k must be contiguous");
  TORCH_CHECK(gpu_k.scalar_type() == gpu_v.scalar_type(), "gpu KV dtype mismatch");
  TORCH_CHECK(gpu_k.scalar_type() == k_out.scalar_type(), "gpu/output dtype mismatch");
  TORCH_CHECK(gpu_k.scalar_type() == v_out.scalar_type(), "gpu/output dtype mismatch");
  TORCH_CHECK(frame_indices.dim() == 2, "frame_indices must be [head_count,n_selected]");
  TORCH_CHECK(slot_indices.sizes() == frame_indices.sizes(), "slot_indices shape mismatch");
  TORCH_CHECK(head_indices.dim() == 1 && head_indices.size(0) == frame_indices.size(0),
               "head_indices shape mismatch");
  TORCH_CHECK(cu_seqlens_k.dim() == 1, "cu_seqlens_k must be 1D");
  TORCH_CHECK(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  TORCH_CHECK(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  TORCH_CHECK(slot_indices.scalar_type() == at::kLong, "slot_indices must be int64");
  TORCH_CHECK(cu_seqlens_k.scalar_type() == at::kInt, "cu_seqlens_k must be int32");

  const int64_t B = gpu_k.size(0);
  const int64_t D = gpu_k.size(3);
  const int64_t head_count = frame_indices.size(0);
  const int64_t n_selected = frame_indices.size(1);
  TORCH_CHECK(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  TORCH_CHECK(cu_seqlens_k.size(0) == B * head_count + 1, "cu_seqlens_k shape mismatch");
  TORCH_CHECK(k_out.dim() == 3 && v_out.sizes() == k_out.sizes(), "outputs must be [T,1,D]");
  TORCH_CHECK(k_out.size(1) == 1 && k_out.size(2) == D, "output head/head_dim mismatch");
  TORCH_CHECK(k_out.size(0) > 0, "output token capacity must be positive");
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(gpu_k.size(2) == 1 && gpu_k.size(3) == D, "gpu KV shape mismatch");
  if (has_live) {
    TORCH_CHECK(live_k.is_cuda() && live_v.is_cuda(), "live KV must be CUDA tensors");
    TORCH_CHECK(live_k.is_contiguous() && live_v.is_contiguous(), "live KV must be contiguous");
    TORCH_CHECK(live_k.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    TORCH_CHECK(live_v.scalar_type() == gpu_k.scalar_type(), "live/gpu dtype mismatch");
    TORCH_CHECK(live_k.dim() == 4 && live_v.sizes() == live_k.sizes(), "live KV must be [B,T,H,D]");
    TORCH_CHECK(live_k.size(0) == B && live_k.size(2) == num_heads && live_k.size(3) == D,
                 "live KV shape mismatch");
  }

  const int64_t total_elements = B * head_count * n_selected * frame_seqlen * D;
  if (total_elements <= 0) {
    return;
  }
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const int64_t live_len = has_live ? live_k.size(1) : 0;

  const c10::cuda::CUDAGuard device_guard(k_out.device());
  const int threads = 256;
  const int64_t pair_count = B * head_count * n_selected;
  const int64_t elems_per_segment = frame_seqlen * D;
  const int64_t tiles64 = (elems_per_segment + threads - 1) / threads;
  const dim3 tiled_blocks(
      static_cast<unsigned int>(pair_count),
      static_cast<unsigned int>(tiles64));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, gpu_k.scalar_type(),
                                  "kv_sparse_frame_materialize_varlen_cuda", [&] {
    kv_sparse_frame_materialize_varlen_tiled_kernel<scalar_t><<<tiled_blocks, threads, 0, stream>>>(
        gpu_k.data_ptr<scalar_t>(),
        gpu_v.data_ptr<scalar_t>(),
        k_out.data_ptr<scalar_t>(),
        v_out.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        frame_indices.data_ptr<int64_t>(),
        slot_indices.data_ptr<int64_t>(),
        cu_seqlens_k.data_ptr<int32_t>(),
        has_live ? live_k.data_ptr<scalar_t>() : nullptr,
        has_live ? live_v.data_ptr<scalar_t>() : nullptr,
        B,
        total_tokens,
        num_heads,
        D,
        head_count,
        n_selected,
        frame_seqlen,
        gpu_cache_frames,
        gpu_cache_tokens,
        live_len,
        live_start_index,
        has_live,
        elems_per_segment);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kv_sparse_update_frame_slot_map_cuda(
    torch::Tensor frame_to_slot,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor slot_ids,
    torch::Tensor evicted_frame_ids) {
  TORCH_CHECK(frame_to_slot.is_cuda() && head_ids.is_cuda() && frame_ids.is_cuda() &&
                   slot_ids.is_cuda() && evicted_frame_ids.is_cuda(),
               "slot metadata update tensors must be CUDA tensors");
  TORCH_CHECK(frame_to_slot.is_contiguous() && head_ids.is_contiguous() &&
                   frame_ids.is_contiguous() && slot_ids.is_contiguous() &&
                   evicted_frame_ids.is_contiguous(),
               "slot metadata update tensors must be contiguous");
  TORCH_CHECK(frame_to_slot.scalar_type() == at::kLong, "frame_to_slot must be int64");
  TORCH_CHECK(head_ids.scalar_type() == at::kLong, "head_ids must be int64");
  TORCH_CHECK(frame_ids.scalar_type() == at::kLong, "frame_ids must be int64");
  TORCH_CHECK(slot_ids.scalar_type() == at::kLong, "slot_ids must be int64");
  TORCH_CHECK(evicted_frame_ids.scalar_type() == at::kLong, "evicted_frame_ids must be int64");
  TORCH_CHECK(frame_to_slot.dim() == 2, "frame_to_slot must be 2D");
  TORCH_CHECK(head_ids.dim() == 1 && frame_ids.sizes() == head_ids.sizes() &&
                   slot_ids.sizes() == head_ids.sizes() &&
                   evicted_frame_ids.sizes() == head_ids.sizes(),
               "slot metadata update tensors must have shape [N]");
  TORCH_CHECK(head_ids.device() == frame_to_slot.device() &&
                   frame_ids.device() == frame_to_slot.device() &&
                   slot_ids.device() == frame_to_slot.device() &&
                   evicted_frame_ids.device() == frame_to_slot.device(),
               "slot metadata update device mismatch");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(frame_to_slot.device());
  const int threads = 128;
  const int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kv_sparse_update_frame_slot_map_kernel<<<blocks, threads, 0, stream>>>(
      frame_to_slot.data_ptr<int64_t>(),
      head_ids.data_ptr<int64_t>(),
      frame_ids.data_ptr<int64_t>(),
      slot_ids.data_ptr<int64_t>(),
      evicted_frame_ids.data_ptr<int64_t>(),
      n,
      frame_to_slot.size(0),
      frame_to_slot.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
