// Modified for the HorizonKV inference artifact; see NOTICE.
#include "kv_sparse_onload.h"
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "cuda_batch_copy.h"

namespace {
int64_t infer_gpu_slot_head_count(
    const torch::Tensor& gpu_k,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  TORCH_CHECK(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");
  const int64_t tokens_per_head = frame_seqlen * gpu_cache_frames;
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_k.size(2) == 1, "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(gpu_k.size(1) % tokens_per_head == 0, "gpu KV slot token dimension mismatch");
  return gpu_k.size(1) / tokens_per_head;
}


void check_slot_copy_inputs(
    const torch::Tensor& cpu_k, const torch::Tensor& cpu_v,
    const torch::Tensor& gpu_k, const torch::Tensor& gpu_v,
    const torch::Tensor& frame_indices, const torch::Tensor& slot_plan,
    const torch::Tensor& copy_plan) {
  TORCH_CHECK(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  TORCH_CHECK(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  TORCH_CHECK(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  TORCH_CHECK(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu_k/gpu_v must be contiguous");
  TORCH_CHECK(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  TORCH_CHECK(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  TORCH_CHECK(cpu_k.scalar_type() == gpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  TORCH_CHECK(cpu_k.scalar_type() == gpu_v.scalar_type(), "cpu/gpu dtype mismatch");
  TORCH_CHECK(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(frame_indices.device().is_cpu() && slot_plan.device().is_cpu() && copy_plan.device().is_cpu(),
               "copy plan tensors must be CPU tensors");
  TORCH_CHECK(frame_indices.is_contiguous() && slot_plan.is_contiguous() && copy_plan.is_contiguous(),
               "copy plan tensors must be contiguous");
  TORCH_CHECK(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  TORCH_CHECK(slot_plan.scalar_type() == at::kLong, "slot_plan must be int64");
  TORCH_CHECK(copy_plan.scalar_type() == at::kLong, "copy_plan must be int64");
  TORCH_CHECK(frame_indices.dim() == 2 && slot_plan.sizes() == frame_indices.sizes() &&
                   copy_plan.sizes() == frame_indices.sizes(),
               "copy plan tensors must have shape [H, selected]");

}

}  // namespace

int64_t kv_sparse_copy_frame_plan_to_slots_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor gpu_k,
    torch::Tensor gpu_v,
    torch::Tensor frame_indices,
    torch::Tensor slot_plan,
    torch::Tensor copy_plan,
    int64_t frame_seqlen,
    int64_t gpu_cache_frames) {
  check_slot_copy_inputs(cpu_k, cpu_v, gpu_k, gpu_v, frame_indices, slot_plan, copy_plan);
  const int64_t B = cpu_k.size(0);
  const int64_t H = infer_gpu_slot_head_count(gpu_k, frame_seqlen, gpu_cache_frames);
  const int64_t T = cpu_k.size(2);
  const int64_t D = cpu_k.size(3);
  TORCH_CHECK(gpu_k.size(0) == B && gpu_k.size(2) == 1 && gpu_k.size(3) == D,
               "gpu KV shape mismatch");
  TORCH_CHECK(cpu_k.size(1) == H && frame_indices.size(0) == H, "copy plan head dimension mismatch");
  const int64_t n_selected = frame_indices.size(1);
  const int64_t* frame_ptr = frame_indices.data_ptr<int64_t>();
  const int64_t* slot_ptr = slot_plan.data_ptr<int64_t>();
  const int64_t* copy_ptr = copy_plan.data_ptr<int64_t>();
  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const int64_t src_batch_stride = T * H * D;
  const int64_t gpu_cache_tokens = gpu_k.size(1);
  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KvCopyBatch copies(stream, cudaMemcpyHostToDevice, gpu_k.get_device(),
                     B * frame_indices.numel() * 2);
  int64_t copied = 0;
  for (int64_t head_id = 0; head_id < H; ++head_id) {
    int64_t col = 0;
    while (col < n_selected) {
      const int64_t offset = head_id * n_selected + col;
      if (copy_ptr[offset] == 0) {
        ++col;
        continue;
      }
      const int64_t frame_id = frame_ptr[offset];
      const int64_t slot = slot_ptr[offset];
      TORCH_CHECK(frame_id >= 0, "frame_id must be non-negative");
      TORCH_CHECK(slot >= 0 && slot < gpu_cache_frames, "slot out of range");

      int64_t run_len = 1;
      while (col + run_len < n_selected) {
        const int64_t next_offset = offset + run_len;
        if (copy_ptr[next_offset] == 0 ||
            frame_ptr[next_offset] != frame_id + run_len ||
            slot_ptr[next_offset] != slot + run_len) {
          break;
        }
        if ((frame_id + run_len) * frame_seqlen >= T || slot + run_len >= gpu_cache_frames) {
          break;
        }
        ++run_len;
      }

      const int64_t start = frame_id * frame_seqlen;
      const int64_t copy_tokens = std::min<int64_t>(run_len * frame_seqlen, T - start);
      if (copy_tokens <= 0) {
        col += run_len;
        continue;
      }
      for (int64_t b = 0; b < B; ++b) {
        const int64_t src_elem = b * src_batch_stride + head_id * T * D + start * D;
        const int64_t dst_token = (head_id * gpu_cache_frames + slot) * frame_seqlen;
        const int64_t dst_elem = (b * gpu_cache_tokens + dst_token) * D;
        const size_t copy_width = static_cast<size_t>(copy_tokens * D) * elem_size;
        copies.add(cpu_k_ptr + src_elem * elem_size, gpu_k_ptr + dst_elem * elem_size, copy_width);
        copies.add(cpu_v_ptr + src_elem * elem_size, gpu_v_ptr + dst_elem * elem_size, copy_width);
      }
      copied += run_len;
      col += run_len;
    }
  }
  copies.submit();
  return copied;
}

int64_t kv_sparse_copy_frames_to_compact_cuda(
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    torch::Tensor compact_k,
    torch::Tensor compact_v,
    torch::Tensor head_ids,
    torch::Tensor frame_ids,
    torch::Tensor cu_seqlens,
    int64_t frame_seqlen) {
  TORCH_CHECK(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  TORCH_CHECK(compact_k.is_cuda() && compact_v.is_cuda(), "compact_k/compact_v must be CUDA tensors");
  TORCH_CHECK(compact_v.device() == compact_k.device(), "compact K/V device mismatch");
  TORCH_CHECK(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  TORCH_CHECK(compact_k.is_contiguous() && compact_v.is_contiguous(), "compact K/V must be contiguous");
  TORCH_CHECK(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned for async HtoD copy");
  TORCH_CHECK(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  TORCH_CHECK(cpu_k.scalar_type() == compact_k.scalar_type(), "cpu/compact dtype mismatch");
  TORCH_CHECK(cpu_k.scalar_type() == compact_v.scalar_type(), "cpu/compact dtype mismatch");
  TORCH_CHECK(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");
  TORCH_CHECK(compact_k.dim() == 3 && compact_v.sizes() == compact_k.sizes(),
               "compact KV must be [packed_tokens,1,D]");
  TORCH_CHECK(head_ids.device().is_cpu() && frame_ids.device().is_cpu() && cu_seqlens.device().is_cpu(),
               "compact copy plans must be CPU tensors");
  TORCH_CHECK(head_ids.is_contiguous() && frame_ids.is_contiguous() && cu_seqlens.is_contiguous(),
               "compact copy plans must be contiguous");
  TORCH_CHECK(head_ids.scalar_type() == at::kLong && frame_ids.scalar_type() == at::kLong &&
                   cu_seqlens.scalar_type() == at::kLong,
               "compact copy plans must be int64");
  TORCH_CHECK(head_ids.dim() == 1 && frame_ids.dim() == 2, "head_ids must be 1D and frame_ids 2D");
  TORCH_CHECK(frame_ids.size(0) == head_ids.size(0), "head/frame row count mismatch");
  TORCH_CHECK(frame_seqlen > 0, "frame_seqlen must be positive");

  const int64_t B = cpu_k.size(0);
  const int64_t H = cpu_k.size(1);
  const int64_t T = cpu_k.size(2);
  const int64_t D = cpu_k.size(3);
  const int64_t rows = head_ids.size(0);
  const int64_t max_selected = frame_ids.size(1);
  TORCH_CHECK(H > 0 && T >= 0 && D > 0, "invalid CPU KV shape");
  TORCH_CHECK(compact_k.size(1) == 1 && compact_k.size(2) == D, "compact KV shape mismatch");
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.size(0) == B * rows + 1,
               "cu_seqlens shape mismatch");
  if (B <= 0 || rows <= 0 || max_selected <= 0 || compact_k.size(0) <= 0) {
    return 0;
  }

  const int64_t* head_ptr = head_ids.data_ptr<int64_t>();
  const int64_t* frame_ptr = frame_ids.data_ptr<int64_t>();
  const int64_t* cu_ptr = cu_seqlens.data_ptr<int64_t>();
  TORCH_CHECK(cu_ptr[B * rows] <= compact_k.size(0), "compact KV token capacity too small");

  const size_t elem_size = static_cast<size_t>(cpu_k.element_size());
  const int64_t src_batch_stride = T * H * D;
  const char* cpu_k_ptr = static_cast<const char*>(cpu_k.data_ptr());
  const char* cpu_v_ptr = static_cast<const char*>(cpu_v.data_ptr());
  char* compact_k_ptr = static_cast<char*>(compact_k.data_ptr());
  char* compact_v_ptr = static_cast<char*>(compact_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(compact_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KvCopyBatch copies(stream, cudaMemcpyHostToDevice, compact_k.get_device(),
                     B * rows * max_selected * 2);

  int64_t copied = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t head_id = head_ptr[row];
    TORCH_CHECK(head_id >= 0 && head_id < H, "head_id out of range");
    int64_t col = 0;
    while (col < max_selected) {
      const int64_t frame_id = frame_ptr[row * max_selected + col];
      if (frame_id < 0) {
        ++col;
        continue;
      }
      const int64_t start = frame_id * frame_seqlen;
      if (start >= T) {
        ++col;
        continue;
      }

      int64_t run_len = 1;
      while (col + run_len < max_selected) {
        const int64_t next_frame_id = frame_ptr[row * max_selected + col + run_len];
        if (next_frame_id != frame_id + run_len) {
          break;
        }
        const int64_t next_start = next_frame_id * frame_seqlen;
        if (next_start >= T) {
          break;
        }
        const int64_t prior_start = (next_frame_id - 1) * frame_seqlen;
        if (prior_start + frame_seqlen > T) {
          break;
        }
        ++run_len;
      }

      const int64_t total_copy_tokens = std::min<int64_t>(run_len * frame_seqlen, T - start);
      if (total_copy_tokens <= 0) {
        col += run_len;
        continue;
      }

      for (int64_t b = 0; b < B; ++b) {
        const int64_t packed_row = b * rows + row;
        const int64_t dst_token_start = cu_ptr[packed_row] + col * frame_seqlen;
        TORCH_CHECK(dst_token_start >= cu_ptr[packed_row] &&
                         dst_token_start + total_copy_tokens <= cu_ptr[packed_row + 1],
                     "compact destination span out of range");
        const int64_t src_elem = b * src_batch_stride + head_id * T * D + start * D;
        const int64_t dst_elem = dst_token_start * D;
        const size_t copy_width = static_cast<size_t>(total_copy_tokens * D) * elem_size;
        copies.add(cpu_k_ptr + src_elem * elem_size, compact_k_ptr + dst_elem * elem_size, copy_width);
        copies.add(cpu_v_ptr + src_elem * elem_size, compact_v_ptr + dst_elem * elem_size, copy_width);
      }
      copied += B * run_len;
      col += run_len;
    }
  }
  copies.submit();
  return copied;
}

int64_t kv_sparse_copy_kv_span_to_cpu_cuda(
    torch::Tensor gpu_k_span,
    torch::Tensor gpu_v_span,
    torch::Tensor cpu_k,
    torch::Tensor cpu_v,
    int64_t start_token) {
  TORCH_CHECK(gpu_k_span.is_cuda() && gpu_v_span.is_cuda(), "gpu_k_span/gpu_v_span must be CUDA tensors");
  TORCH_CHECK(cpu_k.device().is_cpu() && cpu_v.device().is_cpu(), "cpu_k/cpu_v must be CPU tensors");
  TORCH_CHECK(gpu_k_span.is_contiguous() && gpu_v_span.is_contiguous(), "gpu_k_span/gpu_v_span must be contiguous");
  TORCH_CHECK(cpu_k.is_contiguous() && cpu_v.is_contiguous(), "cpu_k/cpu_v must be contiguous");
  TORCH_CHECK(cpu_k.is_pinned() && cpu_v.is_pinned(), "cpu_k/cpu_v must be pinned");
  TORCH_CHECK(gpu_k_span.scalar_type() == gpu_v_span.scalar_type(), "gpu span dtype mismatch");
  TORCH_CHECK(cpu_k.scalar_type() == cpu_v.scalar_type(), "cpu KV dtype mismatch");
  TORCH_CHECK(gpu_k_span.scalar_type() == cpu_k.scalar_type(), "cpu/gpu dtype mismatch");
  TORCH_CHECK(gpu_k_span.dim() == 4 && gpu_v_span.sizes() == gpu_k_span.sizes(),
               "gpu KV span must be [B,T,H,D]");
  TORCH_CHECK(cpu_k.dim() == 4 && cpu_v.sizes() == cpu_k.sizes(), "cpu KV must be 4D");

  const int64_t B = gpu_k_span.size(0);
  const int64_t span_tokens = gpu_k_span.size(1);
  const int64_t H = gpu_k_span.size(2);
  const int64_t D = gpu_k_span.size(3);
  const int64_t cpu_tokens = cpu_k.size(2);
  TORCH_CHECK(cpu_k.size(0) == B && (cpu_k.size(1)) == H && cpu_k.size(3) == D,
               "cpu KV shape mismatch");
  TORCH_CHECK(start_token >= 0, "start_token must be non-negative");
  TORCH_CHECK(start_token + span_tokens <= cpu_tokens, "destination token span out of range");
  if (span_tokens <= 0 || B <= 0) {
    return 0;
  }

  const size_t elem_size = static_cast<size_t>(gpu_k_span.element_size());

  const char* gpu_k_ptr = static_cast<const char*>(gpu_k_span.data_ptr());
  const char* gpu_v_ptr = static_cast<const char*>(gpu_v_span.data_ptr());
  char* cpu_k_ptr = static_cast<char*>(cpu_k.data_ptr());
  char* cpu_v_ptr = static_cast<char*>(cpu_v.data_ptr());


  const c10::cuda::CUDAGuard device_guard(gpu_k_span.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const size_t row_bytes = D * elem_size;
  const size_t gpu_token_pitch = H * row_bytes;
  KvCopyBatch copies(stream, cudaMemcpyDeviceToHost, gpu_k_span.get_device(), B * H * 2);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t src_elem = (b * span_tokens * H + h) * D;
      const int64_t dst_elem = (b * H * cpu_tokens + h * cpu_tokens + start_token) * D;
      copies.add(gpu_k_ptr + src_elem * elem_size, cpu_k_ptr + dst_elem * elem_size,
                 row_bytes, span_tokens, gpu_token_pitch, row_bytes);
      copies.add(gpu_v_ptr + src_elem * elem_size, cpu_v_ptr + dst_elem * elem_size,
                 row_bytes, span_tokens, gpu_token_pitch, row_bytes);
    }
  }
  copies.submit();
  return span_tokens;
}

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
    int64_t gpu_cache_frames) {
  TORCH_CHECK(compact_k.is_cuda() && compact_v.is_cuda(), "compact_k/compact_v must be CUDA tensors");
  TORCH_CHECK(gpu_k.is_cuda() && gpu_v.is_cuda(), "gpu_k/gpu_v must be CUDA tensors");
  TORCH_CHECK(compact_k.device() == gpu_k.device() && compact_v.device() == gpu_v.device(),
               "compact/cache device mismatch");
  TORCH_CHECK(compact_k.is_contiguous() && compact_v.is_contiguous(), "compact KV must be contiguous");
  TORCH_CHECK(gpu_k.is_contiguous() && gpu_v.is_contiguous(), "gpu KV must be contiguous");
  TORCH_CHECK(compact_k.scalar_type() == compact_v.scalar_type(), "compact KV dtype mismatch");
  TORCH_CHECK(compact_k.scalar_type() == gpu_k.scalar_type() && compact_k.scalar_type() == gpu_v.scalar_type(),
               "compact/cache dtype mismatch");
  TORCH_CHECK(compact_k.dim() == 3 && compact_v.sizes() == compact_k.sizes(),
               "compact KV must be [packed_tokens,1,D]");
  TORCH_CHECK(gpu_k.dim() == 4 && gpu_v.sizes() == gpu_k.sizes(), "gpu KV must be [B,Tslot,1,D]");
  TORCH_CHECK(compact_k.size(1) == 1 && gpu_k.size(2) == 1 && compact_k.size(2) == gpu_k.size(3),
               "compact/cache shape mismatch");
  TORCH_CHECK(head_ids.device().is_cpu() && slot_ids.device().is_cpu() && src_rows.device().is_cpu() &&
                   src_cols.device().is_cpu() && cu_seqlens.device().is_cpu(),
               "D2D copy plans must be CPU tensors");
  TORCH_CHECK(head_ids.is_contiguous() && slot_ids.is_contiguous() && src_rows.is_contiguous() &&
                   src_cols.is_contiguous() && cu_seqlens.is_contiguous(),
               "D2D copy plans must be contiguous");
  TORCH_CHECK(head_ids.scalar_type() == at::kLong && slot_ids.scalar_type() == at::kLong &&
                   src_rows.scalar_type() == at::kLong && src_cols.scalar_type() == at::kLong &&
                   cu_seqlens.scalar_type() == at::kLong,
               "D2D copy plans must be int64");
  TORCH_CHECK(head_ids.dim() == 1 && slot_ids.sizes() == head_ids.sizes() &&
                   src_rows.sizes() == head_ids.sizes() && src_cols.sizes() == head_ids.sizes(),
               "D2D copy plan tensors must have shape [N]");
  TORCH_CHECK(frame_seqlen > 0 && gpu_cache_frames > 0, "frame/cache sizes must be positive");

  const int64_t n = head_ids.size(0);
  if (n <= 0) {
    return 0;
  }
  const int64_t B = gpu_k.size(0);
  TORCH_CHECK(B > 0, "GPU cache batch must be positive");
  const int64_t D = gpu_k.size(3);
  TORCH_CHECK(cu_seqlens.dim() == 1 && (cu_seqlens.size(0) - 1) % B == 0,
               "cu_seqlens shape mismatch");
  const int64_t rows = (cu_seqlens.size(0) - 1) / B;
  const auto* head_ptr = head_ids.data_ptr<int64_t>();
  const auto* slot_ptr = slot_ids.data_ptr<int64_t>();
  const auto* row_ptr = src_rows.data_ptr<int64_t>();
  const auto* col_ptr = src_cols.data_ptr<int64_t>();
  const auto* cu_ptr = cu_seqlens.data_ptr<int64_t>();
  const size_t copy_bytes = static_cast<size_t>(frame_seqlen * D) * compact_k.element_size();
  const int64_t gpu_tokens = gpu_k.size(1);
  const char* compact_k_ptr = static_cast<const char*>(compact_k.data_ptr());
  const char* compact_v_ptr = static_cast<const char*>(compact_v.data_ptr());
  char* gpu_k_ptr = static_cast<char*>(gpu_k.data_ptr());
  char* gpu_v_ptr = static_cast<char*>(gpu_v.data_ptr());

  const c10::cuda::CUDAGuard device_guard(gpu_k.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  KvCopyBatch copies(stream, cudaMemcpyDeviceToDevice, gpu_k.get_device(), n * B * 2);
  for (int64_t i = 0; i < n; ++i) {
    const int64_t head = head_ptr[i];
    const int64_t slot = slot_ptr[i];
    const int64_t src_row = row_ptr[i];
    const int64_t src_col = col_ptr[i];
    TORCH_CHECK(head >= 0 && slot >= 0 && slot < gpu_cache_frames, "invalid destination slot");
    TORCH_CHECK(src_row >= 0 && src_row < rows && src_col >= 0, "invalid compact source");
    for (int64_t b = 0; b < B; ++b) {
      const int64_t packed_row = b * rows + src_row;
      const int64_t src_token = cu_ptr[packed_row] + src_col * frame_seqlen;
      TORCH_CHECK(src_token >= cu_ptr[packed_row] && src_token + frame_seqlen <= cu_ptr[packed_row + 1],
                   "compact source span out of range");
      const int64_t dst_token = (head * gpu_cache_frames + slot) * frame_seqlen;
      TORCH_CHECK(dst_token >= 0 && dst_token + frame_seqlen <= gpu_tokens,
                   "GPU slot destination out of range");
      const int64_t src_elem = src_token * D;
      const int64_t dst_elem = (b * gpu_tokens + dst_token) * D;
      const size_t elem_size = compact_k.element_size();
      copies.add(compact_k_ptr + src_elem * elem_size, gpu_k_ptr + dst_elem * elem_size, copy_bytes);
      copies.add(compact_v_ptr + src_elem * elem_size, gpu_v_ptr + dst_elem * elem_size, copy_bytes);
    }
  }
  copies.submit();
  return n;
}
