/**
 * float_split_stride_pin.cu
 *
 * CUDA kernels for bf16/fp32 float split/merge operations.
 * Used by MemRift for compressed weight/activation storage.
 *
 * Split format:
 * - bf16: exp (8-bit) + sign_mantissa (8-bit: 1 sign + 7 mantissa)
 * - fp32: exp (8-bit) + sign_mantissa (24-bit: 1 sign + 23 mantissa)
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <c10/cuda/CUDAGuard.h>
#include <mutex>
#include <unordered_map>
#include <tuple>
#include <optional>

// ════════════════ 1.  SIZE‑BUCKET KEY ════════════════════════════════════════
using Key = std::tuple<int64_t,              // numel (元素个数)
                      c10::ScalarType,      // dtype
                      int,                  // dev (GPU id)  | −1 == pinned host
                      bool>;                // is_pinned
struct KeyHash {
    size_t operator()(const Key &k) const noexcept {
        auto [n,dt,dv,pin] = k;
        return (std::hash<int64_t>{}(n) ^ (static_cast<size_t>(dt)<<6) ^
                ((dv+1)<<12) ^ (pin<<17));
    }
};

// ════════════════ 2.  PER‑BUCKET STORAGE ════════════════════════════════════
constexpr int KEEP_LIMIT = 1;                 // 每 bucket 至多缓存 1 块

struct Bucket {
    std::vector<std::pair<at::Tensor,cudaEvent_t>> free; // event==nullptr 表 CPU/pinned
    std::mutex mtx;
};

class Pool {
    std::unordered_map<Key,Bucket,KeyHash> map_;
    std::mutex global_;                        // 仅保护 map_ 本身的增删
  public:
    static Pool &inst(){ static Pool P; return P; }

    // ───────── acquire ────────────────────────────────────────────────────
    at::Tensor get(int64_t n,c10::ScalarType dt,int dev,bool pinned, cudaStream_t cur) {
        Key k{n,dt,dev,pinned};
        Bucket *bkt;
        {
            std::lock_guard<std::mutex> g(global_);
            bkt = &map_[k];
        }
        std::lock_guard<std::mutex> g(bkt->mtx);
        if(!bkt->free.empty()){
            auto [t,ev] = bkt->free.back();
            bkt->free.pop_back();
            if(ev){                     // GPU tensor – wait event then destroy
                cudaStreamWaitEvent(cur,ev,0);
                cudaEventDestroy(ev);
            }
            return t;
        }
        // miss => real alloc
        if(pinned)
            return at::empty({n}, at::dtype(dt).pinned_memory(true));
        return at::empty({n}, at::TensorOptions().device(at::kCUDA, dev).dtype(dt));
    }
    // ───────── release ────────────────────────────────────────────────────
    void put(at::Tensor t,bool pinned,cudaStream_t last){
        if(!t.defined()||!t.numel()) return;
        int64_t n=t.numel();
        c10::ScalarType dt=t.scalar_type();
        int dev = pinned? -1 : t.device().index();
        Key k{n,dt,dev,pinned};
        Bucket *bkt;
        {
            std::lock_guard<std::mutex> g(global_);
            bkt=&map_[k];
        }
        std::lock_guard<std::mutex> g(bkt->mtx);
        if((int)bkt->free.size()>=KEEP_LIMIT){ return; } // 让引用计数归零 => 真 free

        cudaEvent_t ev=nullptr;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev,last);
        bkt->free.emplace_back(std::move(t),ev);
    }
};

//----------------------------------------------------------------------------
//  utilities: indexer <sizes,strides,offset>  →  flatten / offset
//----------------------------------------------------------------------------
template<int N>
struct StridedIndex {
    int64_t sizes[N];
    int64_t strides[N];
    int64_t base_offset;                        // elements, not bytes
    bool    is_contig = false;

    __host__ __device__ __forceinline__
    int64_t offset(int64_t linear) const {
        if (is_contig) {
            return base_offset + linear;
        }
        int64_t off = base_offset;
        #pragma unroll
        for (int i = N - 1; i >= 0; --i) {
            int64_t cur = linear % sizes[i];
            linear     /= sizes[i];
            off        += cur * strides[i];
        }
        return off;
    }
};

template<int N>
StridedIndex<N> make_indexer(const at::Tensor& t) {
    StridedIndex<N> ix;
    auto sz = t.sizes();
    auto st = t.strides();
    for (int i = 0; i < N; ++i) {
        ix.sizes[i]   = (i < sz.size()) ? sz[i] : 1;
        ix.strides[i] = (i < st.size()) ? st[i] : 1;
    }
    ix.base_offset = 0;
    ix.is_contig = t.is_contiguous();
    return ix;
}

//----------------------------------------------------------------------------
//  kernels (<<<grid,256>>>),  N=4 covers up to 4-D activations
//----------------------------------------------------------------------------
template<int N>
__global__ void pack_bf16_kernel(const uint16_t* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint16_t bits = __ldg(&in[ix.offset(idx)]);
    exp_out[idx]  = (bits >> 7) & 0xFF;
    sm_out[idx]   = ((bits >> 15) & 0x1) << 7 | (bits & 0x7F);
}

template<int N>
__global__ void pack_bf16_kernel_vec2(const uint16_t* __restrict__ in,
                                      StridedIndex<N> ix,
                                      uint8_t* __restrict__ exp_out,
                                      uint8_t* __restrict__ sm_out,
                                      int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= numel) return;

    int64_t off = ix.offset(idx);
    uint16_t bits0 = __ldg(&in[ off ]);

    exp_out[idx]     = (bits0 >> 7) & 0xFF;
    sm_out[idx]      = ((bits0 >> 15) & 0x1) << 7 | (bits0 & 0x7F);

    if (idx + 1 < numel) {
        uint16_t bits1 = __ldg(&in[ off + 1 ]);
        exp_out[idx + 1] = (bits1 >> 7) & 0xFF;
        sm_out[idx + 1]  = ((bits1 >> 15) & 0x1) << 7 | (bits1 & 0x7F);
    }
}

template<int N>
__global__ void pack_fp32_kernel(const float* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint32_t bits = __ldg(& reinterpret_cast<const uint32_t*>(in)[ ix.offset(idx) ]);
    exp_out[idx]  = (bits >> 23) & 0xFF;

    uint32_t sm24 = ((bits & 0x7FFFFF)      )      // mant
                  | ((bits >> 8) & 0x800000);      // sign
    sm_out[idx*3+0] =  sm24        & 0xFF;
    sm_out[idx*3+1] = (sm24 >> 8 ) & 0xFF;
    sm_out[idx*3+2] = (sm24 >> 16) & 0xFF;
}

template<int N>
__global__ void pack_fp32_kernel_vec2(const float* __restrict__ in,
                                 StridedIndex<N> ix,
                                 uint8_t* __restrict__ exp_out,
                                 uint8_t* __restrict__ sm_out,
                                 int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= numel) return;

    int64_t off = ix.offset(idx);
    uint32_t bits = __ldg(& reinterpret_cast<const uint32_t*>(in)[ off ]);
    exp_out[idx]  = (bits >> 23) & 0xFF;

    uint32_t sm24 = ((bits & 0x7FFFFF)      )      // mant
                  | ((bits >> 8) & 0x800000);      // sign

    sm_out[idx*3+0] =  sm24        & 0xFF;
    sm_out[idx*3+1] = (sm24 >> 8 ) & 0xFF;
    sm_out[idx*3+2] = (sm24 >> 16) & 0xFF;

    if (idx + 1 < numel) {
        uint32_t bits_1 = __ldg(& reinterpret_cast<const uint32_t*>(in)[ off + 1 ]);
        exp_out[idx+1]  = (bits_1 >> 23) & 0xFF;

        uint32_t sm24_1 = ((bits_1 & 0x7FFFFF)      )      // mant
                      | ((bits_1 >> 8) & 0x800000);      // sign

        sm_out[(idx+1)*3+0] =  sm24_1        & 0xFF;
        sm_out[(idx+1)*3+1] = (sm24_1 >> 8 ) & 0xFF;
        sm_out[(idx+1)*3+2] = (sm24_1 >> 16) & 0xFF;
    }
}

template<int N>
__global__ void unpack_bf16_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   uint16_t* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint8_t  sm   = __ldg(&sm_in[idx]);
    uint16_t sign = (sm >> 7) & 0x1;
    uint16_t mant =  sm & 0x7F;
    uint16_t bits = (sign << 15) |
                    (static_cast<uint16_t>(__ldg(&exp_in[idx])) << 7) |
                    mant;
    out[ix.offset(idx)] = bits;
}

template<int N>
__global__ void unpack_bf16_kernel_vec2(const uint8_t* __restrict__ exp_in,
                                        const uint8_t* __restrict__ sm_in,
                                        StridedIndex<N> ix,
                                        uint16_t* __restrict__ out,
                                        int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= numel) return;

    uint8_t sm0 = __ldg(&sm_in[idx]);
    uint8_t exp0 = __ldg(&exp_in[idx]);

    uint16_t bits0 = ((sm0 >> 7) << 15) | (static_cast<uint16_t>(exp0) << 7) | (sm0 & 0x7F);

    int64_t off = ix.offset(idx);
    out[ off ]     = bits0;

    if (idx + 1 < numel) {
        uint8_t sm1 = __ldg(&sm_in[idx + 1]);
        uint8_t exp1 = __ldg(&exp_in[idx + 1]);
        uint16_t bits1 = ((sm1 >> 7) << 15) | (static_cast<uint16_t>(exp1) << 7) | (sm1 & 0x7F);
        out[ off + 1 ] = bits1;
    }
}

template<int N>
__global__ void unpack_fp32_kernel(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    uint32_t sm24 =  __ldg(&sm_in[idx*3+0])
                   | (__ldg(&sm_in[idx*3+1]) << 8 )
                   | (__ldg(&sm_in[idx*3+2]) << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;

    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(__ldg(& exp_in[idx])) << 23)
                  |  mant;
    reinterpret_cast<uint32_t*>(out)[ ix.offset(idx) ] = bits;
}

template<int N>
__global__ void unpack_fp32_kernel_vec2(const uint8_t* __restrict__ exp_in,
                                   const uint8_t* __restrict__ sm_in,
                                   StridedIndex<N> ix,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= numel) return;

    uint32_t sm24 =  __ldg(&sm_in[idx*3+0])
                   | (__ldg(&sm_in[idx*3+1]) << 8 )
                   | (__ldg(&sm_in[idx*3+2]) << 16);

    uint32_t sign = (sm24 >> 23) & 0x1;
    uint32_t mant =  sm24 & 0x7FFFFF;

    uint32_t bits = (sign << 31)
                  | (static_cast<uint32_t>(__ldg(& exp_in[idx])) << 23)
                  |  mant;

    int64_t off = ix.offset(idx);
    reinterpret_cast<uint32_t*>(out)[ off ] = bits;

    if (idx + 1 < numel) {
        uint32_t sm24_1 =  __ldg(&sm_in[(idx+1)*3+0])
                       | (__ldg(&sm_in[(idx+1)*3+1]) << 8 )
                       | (__ldg(&sm_in[(idx+1)*3+2]) << 16);

        uint32_t sign_1 = (sm24_1 >> 23) & 0x1;
        uint32_t mant_1 =  sm24_1 & 0x7FFFFF;

        uint32_t bits_1 = (sign_1 << 31)
                      | (static_cast<uint32_t>(__ldg(& exp_in[idx+1])) << 23)
                      |  mant_1;

        reinterpret_cast<uint32_t*>(out)[ off + 1 ] = bits_1;
    }
}

//----------------------------------------------------------------------------
//  host wrappers
//----------------------------------------------------------------------------
static inline dim3 grid_for(int64_t n) { return dim3((n + 255) / 256); }

std::vector<at::Tensor> pack_tensor(const at::Tensor& t, unsigned long long stream_ptr) {
    TORCH_CHECK(t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16,
                "dtype must be fp32 / bf16");
    TORCH_CHECK(t.dim() <= 4, "float_split_stride_pin supports tensors with at most 4 dimensions");

    int dev_idx = t.device().index();
    TORCH_CHECK(dev_idx >= 0, "sm must be on CUDA");
    cudaStream_t  raw = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(raw, dev_idx);
    c10::cuda::CUDAStreamGuard guard{s};

    const int64_t N = t.numel();
    dim3 grid = grid_for(N);

    auto exp_host = Pool::inst().get(N, at::kByte, -1, true, raw);
    int64_t sm_elems=(t.scalar_type()==at::kFloat)?N*3:N;
    auto sm = Pool::inst().get(sm_elems, at::kByte, dev_idx, false, raw);

    uint8_t* host_ptr = exp_host.data_ptr<uint8_t>();
    uint8_t* dev_ptr;
    cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    auto ix  = make_indexer<4>(t);

    if (t.scalar_type() == at::kFloat) {
        if (ix.is_contig)
            pack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
        else
            pack_fp32_kernel<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
    } else {    // bf16
        if (ix.is_contig)
            pack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
        else
            pack_bf16_kernel<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                dev_ptr,
                sm.data_ptr<uint8_t>(), N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {exp_host, sm};
}

std::vector<at::Tensor> pack_tensor_copy(const at::Tensor& t, unsigned long long stream_ptr) {
    TORCH_CHECK(t.is_cuda(), "input must be CUDA");
    TORCH_CHECK(t.scalar_type()==at::kFloat || t.scalar_type()==at::kBFloat16,
                "dtype must be fp32 / bf16");
    TORCH_CHECK(t.dim() <= 4, "float_split_stride_pin supports tensors with at most 4 dimensions");

    int dev_idx = t.device().index();
    TORCH_CHECK(dev_idx >= 0, "input tensor must be on CUDA");
    cudaStream_t raw = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(raw, dev_idx);
    c10::cuda::CUDAStreamGuard guard{s};

    const int64_t N = t.numel();
    dim3 grid = grid_for(N);

    auto exp_host = Pool::inst().get(N, at::kByte, -1, true, raw);
    auto exp_gpu = Pool::inst().get(N, at::kByte, dev_idx, false, raw);
    int64_t sm_elems = (t.scalar_type()==at::kFloat) ? N * 3 : N;
    auto sm = Pool::inst().get(sm_elems, at::kByte, dev_idx, false, raw);

    auto ix = make_indexer<4>(t);

    if (t.scalar_type() == at::kFloat) {
        if (ix.is_contig)
            pack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
        else
            pack_fp32_kernel<4><<<grid,256,0,raw>>>(
                t.data_ptr<float>(), ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
    } else {
        if (ix.is_contig)
            pack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
        else
            pack_bf16_kernel<4><<<grid,256,0,raw>>>(
                reinterpret_cast<const uint16_t*>(t.data_ptr<at::BFloat16>()),
                ix,
                exp_gpu.data_ptr<uint8_t>(),
                sm.data_ptr<uint8_t>(), N);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    cudaError_t err = cudaMemcpyAsync(
        exp_host.data_ptr<uint8_t>(),
        exp_gpu.data_ptr<uint8_t>(),
        static_cast<size_t>(N),
        cudaMemcpyDeviceToHost,
        raw);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync D2H failed: ",
                cudaGetErrorString(err));

    exp_gpu.record_stream(s);
    sm.record_stream(s);
    return {exp_host, sm, exp_gpu};
}

// util: 计算 offset+sizes,strides 需要的底层 storage 大小
int64_t required_storage(int64_t base_off,
            const std::vector<int64_t>& sizes,
            const std::vector<int64_t>& strides) {

    int64_t max_off = base_off;

    for (size_t i = 0; i < sizes.size(); ++i)
        max_off += (sizes[i] - 1) * strides[i];

    return max_off + 1;
}


at::Tensor unpack_tensor(at::Tensor exp,
                         at::Tensor sm,
                         std::vector<int64_t> sizes,
                         std::vector<int64_t> strides,
                         int64_t storage_offset,
                         c10::ScalarType dtype,
                         unsigned long long stream_ptr) {

    TORCH_CHECK(sm.is_cuda(), "sm must be CUDA uint8");
    TORCH_CHECK(dtype == at::kFloat || dtype == at::kBFloat16, "dtype mismatch");
    TORCH_CHECK(sizes.size() <= 4, "float_split_stride_pin supports at most 4 dimensions");
    TORCH_CHECK(sizes.size() == strides.size(), "sizes and strides must have equal length");

    int dev_idx = sm.device().index();
    TORCH_CHECK(dev_idx >= 0, "sm must be on CUDA");
    cudaStream_t  raw = reinterpret_cast<cudaStream_t>(stream_ptr);
    c10::cuda::CUDAStream s = c10::cuda::getStreamFromExternal(raw, dev_idx);
    c10::cuda::CUDAStreamGuard guard{s};

    // 将 exp 指针转成 GPU 可见地址
    uint8_t* exp_dev_ptr = nullptr;
    std::optional<at::Tensor> tmp_gpu_exp;

    if (exp.is_cuda()) {
        exp_dev_ptr = exp.data_ptr<uint8_t>();
    } else {
        TORCH_CHECK(exp.is_pinned(),
                    "exp on CPU must be pinned for zero-copy");

        cudaError_t err = cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&exp_dev_ptr),
                exp.data_ptr(), 0);

        if (err != cudaSuccess) {
            tmp_gpu_exp.emplace(
                exp.to(sm.device(), /*non_blocking=*/true));
            exp_dev_ptr = tmp_gpu_exp->data_ptr<uint8_t>();
        }
    }

    // allocate output tensor (原 stride)
    const int64_t need = required_storage(storage_offset, sizes, strides);
    const int64_t N = exp.numel();
    dim3 grid = grid_for(N);
    auto flat = Pool::inst().get(need, dtype, dev_idx, false, raw);
    auto out  = flat.as_strided(sizes, strides, storage_offset);
    auto ix = make_indexer<4>(out);

    // launch kernel
    if (dtype == at::kFloat) {
        if (ix.is_contig)
            unpack_fp32_kernel_vec2<4><<<grid,256,0,raw>>>(
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                out.data_ptr<float>(),
                N);
        else
            unpack_fp32_kernel<4><<<grid,256,0,raw>>>(
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                out.data_ptr<float>(),
                N);
    } else {
        if (ix.is_contig)
            unpack_bf16_kernel_vec2<4><<<grid,256,0,raw>>>(
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
                N);
        else
            unpack_bf16_kernel<4><<<grid,256,0,raw>>>(
                exp_dev_ptr,
                sm.data_ptr<uint8_t>(),
                ix,
                reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()),
                N);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    out.record_stream(s);
    return out;
}

//----------------------------------------------------------------------------
//  pybind
//----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("split", &pack_tensor, "pack tensor -> (exp, sm)");
    m.def("split_copy", &pack_tensor_copy,
          "pack tensor using GPU exp staging + cudaMemcpyAsync D2H -> (exp_host, sm, exp_gpu)");
    m.def("merge", &unpack_tensor,
          "unpack (exp,sm,sizes,strides,offset,dtype,stream) -> tensor",
          py::arg("exp"), py::arg("sm"),
          py::arg("sizes"), py::arg("strides"), py::arg("storage_offset"),
          py::arg("dtype"), py::arg("stream_ptr"));

    m.def("acquire_pin", [](int64_t numel, c10::ScalarType dtype){
            return Pool::inst().get(numel, dtype, -1, /*pinned=*/true, 0);
        });
    m.def("release_pin", [](at::Tensor t){
            cudaStream_t cur = c10::cuda::getCurrentCUDAStream().stream();
            Pool::inst().put(std::move(t), /*pinned=*/true, cur);
        });
    m.def("release_cuda", [](at::Tensor t){
            cudaStream_t cur = c10::cuda::getCurrentCUDAStream().stream();
            Pool::inst().put(std::move(t), /*pinned=*/false, cur);
        });
}
