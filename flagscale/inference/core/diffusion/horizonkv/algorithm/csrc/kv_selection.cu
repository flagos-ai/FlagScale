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
template <typename scalar_t>
__global__ void pooled_qk_q_mean_kernel(
    const scalar_t* __restrict__ q,
    const int64_t* __restrict__ head_indices,
    float* __restrict__ q_mean,
    int64_t B,
    int64_t Lq,
    int64_t H,
    int64_t D,
    int64_t Hs,
    int64_t frame_seqlen,
    int64_t qs0,
    int64_t qs1,
    int64_t qs2,
    int64_t qs3) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t total = B * Hs * D;
  if (linear >= total) {
    return;
  }
  int64_t t = linear;
  const int64_t d = t % D;
  t /= D;
  const int64_t row = t % Hs;
  const int64_t b = t / Hs;
  const int64_t head_id = head_indices[row];
  float acc = 0.0f;
  if (head_id >= 0 && head_id < H) {
    const bool simple_mean = frame_seqlen <= 1 || (frame_seqlen > 0 && Lq % frame_seqlen == 0);
    const int64_t full_blocks = simple_mean ? 0 : Lq / frame_seqlen;
    const int64_t rest = simple_mean ? 0 : Lq - full_blocks * frame_seqlen;
    const int64_t pooled_blocks = simple_mean ? 1 : full_blocks + (rest > 0 ? 1 : 0);
    const int64_t full_tokens = full_blocks * frame_seqlen;
    for (int64_t pos = threadIdx.x; pos < Lq; pos += blockDim.x) {
      const int64_t off = b * qs0 + pos * qs1 + head_id * qs2 + d * qs3;
      const float value = static_cast<float>(q[off]);
      if (simple_mean) {
        acc += value / static_cast<float>(Lq);
      } else if (pos < full_tokens) {
        acc += value / static_cast<float>(frame_seqlen * pooled_blocks);
      } else {
        acc += value / static_cast<float>(rest * pooled_blocks);
      }
    }
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    q_mean[(b * Hs + row) * D + d] = smem[0];
  }
}

template <typename scalar_t>
__global__ void pooled_qk_score_kernel(
    const float* __restrict__ q_mean,
    const scalar_t* __restrict__ frame_k,
    const int64_t* __restrict__ head_indices,
    float* __restrict__ scores,
    int64_t B,
    int64_t F,
    int64_t H,
    int64_t D,
    int64_t Hs,
    int64_t fs0,
    int64_t fs1,
    int64_t fs2,
    int64_t fs3) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x);
  const int64_t total = Hs * F;
  if (linear >= total) {
    return;
  }
  const int64_t f = linear % F;
  const int64_t row = linear / F;
  const int64_t head_id = head_indices[row];
  float acc = 0.0f;
  if (head_id >= 0 && head_id < H) {
    const int64_t total_bd = B * D;
    for (int64_t idx = threadIdx.x; idx < total_bd; idx += blockDim.x) {
      const int64_t d = idx % D;
      const int64_t b = idx / D;
      const int64_t off = b * fs0 + f * fs1 + head_id * fs2 + d * fs3;
      acc += q_mean[(b * Hs + row) * D + d] * static_cast<float>(frame_k[off]);
    }
  }
  __shared__ float smem[256];
  smem[threadIdx.x] = acc;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(static_cast<float>(D > 0 ? D : 1));
  const float denom = static_cast<float>(B > 0 ? B : 1);
  if (threadIdx.x == 0) {
    scores[row * F + f] = smem[0] * scale / denom;
  }
}

__device__ bool selected_contains_sorted(const int64_t* values, int count, int64_t value) {
  int64_t lo = 0;
  int64_t hi = count;
  while (lo < hi) {
    const int64_t mid = lo + ((hi - lo) >> 1);
    if (values[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo < count && values[lo] == value;
}

__device__ void selected_insert_sorted(int64_t* values, int count, int64_t value) {
  int pos = count;
  while (pos > 0 && values[pos - 1] > value) {
    values[pos] = values[pos - 1];
    --pos;
  }
  values[pos] = value;
}

__global__ void pooled_qk_topk_kernel(
    const float* __restrict__ scores,
    const int64_t* __restrict__ compulsory_indices,
    int64_t* __restrict__ out,
    int64_t Hs,
    int64_t F,
    int64_t compulsory_count,
    int64_t max_kv_frames,
    int64_t out_cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= Hs) {
    return;
  }

  constexpr int kMaxOut = 256;
  if (F <= max_kv_frames) {
    if (threadIdx.x == 0) {
      for (int64_t i = 0; i < out_cols; ++i) {
        out[row * out_cols + i] = i < F ? i : -1;
      }
    }
    return;
  }

  __shared__ int64_t selected[kMaxOut];
  __shared__ int selected_count;
  __shared__ int64_t best_frame_sh[256];
  __shared__ float best_score_sh[256];

  if (threadIdx.x == 0) {
    selected_count = 0;
    for (int64_t i = 0; i < compulsory_count && selected_count < kMaxOut; ++i) {
      const int64_t frame = compulsory_indices[i];
      if (frame >= 0 && frame < F &&
          !selected_contains_sorted(selected, selected_count, frame)) {
        selected_insert_sorted(selected, selected_count, frame);
        ++selected_count;
      }
    }
  }
  __syncthreads();

  const int target = static_cast<int>(max_kv_frames < kMaxOut ? max_kv_frames : kMaxOut);
  while (selected_count < target) {
    int64_t local_best_frame = -1;
    float local_best_score = -3.4028234663852886e38f;
    for (int64_t f = threadIdx.x; f < F; f += blockDim.x) {
      if (selected_contains_sorted(selected, selected_count, f)) {
        continue;
      }
      const float score = scores[row * F + f];
      if (local_best_frame < 0 || score > local_best_score) {
        local_best_score = score;
        local_best_frame = f;
      }
    }
    best_frame_sh[threadIdx.x] = local_best_frame;
    best_score_sh[threadIdx.x] = local_best_score;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (threadIdx.x < offset) {
        const float other_score = best_score_sh[threadIdx.x + offset];
        const int64_t other_frame = best_frame_sh[threadIdx.x + offset];
        if (other_frame >= 0 &&
            (best_frame_sh[threadIdx.x] < 0 || other_score > best_score_sh[threadIdx.x])) {
          best_score_sh[threadIdx.x] = other_score;
          best_frame_sh[threadIdx.x] = other_frame;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const int64_t best_frame = best_frame_sh[0];
      if (best_frame >= 0) {
        selected_insert_sorted(selected, selected_count, best_frame);
        ++selected_count;
      }
    }
    __syncthreads();
    if (best_frame_sh[0] < 0) {
      break;
    }
  }

  if (threadIdx.x == 0) {
    for (int64_t i = 0; i < out_cols; ++i) {
      out[row * out_cols + i] = i < selected_count ? selected[i] : -1;
    }
  }
}


}  // namespace

torch::Tensor kv_sparse_select_pooled_qk_topk_cuda(
    torch::Tensor q,
    torch::Tensor frame_k,
    torch::Tensor head_indices,
    torch::Tensor compulsory_indices,
    int64_t frame_seqlen,
    int64_t max_kv_frames) {
  TORCH_CHECK(q.is_cuda() && frame_k.is_cuda(), "q/frame_k must be CUDA tensors");
  TORCH_CHECK(head_indices.is_cuda() && compulsory_indices.is_cuda(),
               "head_indices/compulsory_indices must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4, "q must be [B,Lq,H,D]");
  TORCH_CHECK(frame_k.dim() == 4, "frame_k must be [B,F,H,D]");
  TORCH_CHECK(head_indices.dim() == 1, "head_indices must be 1D");
  TORCH_CHECK(compulsory_indices.dim() == 1, "compulsory_indices must be 1D");
  TORCH_CHECK(head_indices.scalar_type() == at::kLong, "head_indices must be int64");
  TORCH_CHECK(compulsory_indices.scalar_type() == at::kLong, "compulsory_indices must be int64");
  TORCH_CHECK(q.scalar_type() == frame_k.scalar_type(), "q/frame_k dtype mismatch");
  TORCH_CHECK(q.size(0) == frame_k.size(0), "q/frame_k batch mismatch");
  TORCH_CHECK(q.size(2) == frame_k.size(2), "q/frame_k head mismatch");
  TORCH_CHECK(q.size(3) == frame_k.size(3), "q/frame_k head_dim mismatch");
  TORCH_CHECK(frame_seqlen >= 0, "frame_seqlen must be non-negative");
  TORCH_CHECK(max_kv_frames >= 0, "max_kv_frames must be non-negative");

  const int64_t B = q.size(0);
  const int64_t Lq = q.size(1);
  const int64_t H = q.size(2);
  const int64_t D = q.size(3);
  const int64_t F = frame_k.size(1);
  const int64_t Hs = head_indices.size(0);
  const int64_t compulsory_count = compulsory_indices.size(0);
  if (B <= 0 || Lq <= 0 || H <= 0 || D <= 0 || F <= 0 || Hs <= 0) {
    return torch::empty({Hs, 0}, torch::TensorOptions().dtype(torch::kLong).device(q.device()));
  }

  int64_t out_cols = 0;
  if (F <= max_kv_frames) {
    out_cols = F;
  } else if (compulsory_count >= max_kv_frames) {
    out_cols = compulsory_count;
  } else {
    out_cols = max_kv_frames;
  }
  TORCH_CHECK(out_cols <= 256, "fused topK output supports at most 256 selected frames");

  const c10::cuda::CUDAGuard device_guard(q.device());
  head_indices = head_indices.contiguous();
  compulsory_indices = compulsory_indices.contiguous();
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
  auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(q.device());
  auto q_mean = torch::empty({B, Hs, D}, float_opts);
  auto scores = torch::empty({Hs, F}, float_opts);
  auto out = torch::empty({Hs, out_cols}, long_opts);

  const int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t q_total = B * Hs * D;
  const int q_blocks = static_cast<int>(q_total);
  const int64_t score_total = Hs * F;
  const int score_blocks = static_cast<int>(score_total);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(),
                                  "kv_sparse_select_pooled_qk_topk_cuda", [&] {
    pooled_qk_q_mean_kernel<scalar_t><<<q_blocks, threads, 0, stream>>>(
        q.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        q_mean.data_ptr<float>(),
        B,
        Lq,
        H,
        D,
        Hs,
        frame_seqlen,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3));
    pooled_qk_score_kernel<scalar_t><<<score_blocks, threads, 0, stream>>>(
        q_mean.data_ptr<float>(),
        frame_k.data_ptr<scalar_t>(),
        head_indices.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        B,
        F,
        H,
        D,
        Hs,
        frame_k.stride(0),
        frame_k.stride(1),
        frame_k.stride(2),
        frame_k.stride(3));
  });
  pooled_qk_topk_kernel<<<static_cast<unsigned int>(Hs), threads, 0, stream>>>(
      scores.data_ptr<float>(),
      compulsory_indices.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(),
      Hs,
      F,
      compulsory_count,
      max_kv_frames,
      out_cols);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
