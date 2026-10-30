// Modified for the HorizonKV inference artifact; see NOTICE.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>
#include <limits>
#include <vector>

namespace {
// cuMemcpy3DBatchAsync was added after the CUDA 12.4 toolkit used to build this
// extension. Resolve its stable CUDA 12.8 driver ABI at runtime so newer
// drivers can use it without making the extension require a newer toolkit.
constexpr int kMemcpyOperandPointer = 0x1;
constexpr int kMemcpySrcAccessOrderStream = 0x1;
constexpr unsigned int kMemcpyPreferOverlapWithCompute = 0x1;

struct KvMemcpy3DPointerOperand {
  CUdeviceptr ptr;
  size_t row_length;
  size_t layer_height;
  CUmemLocation location_hint;
};

struct KvMemcpy3DArrayOperand {
  CUarray array;
  size_t offset_x;
  size_t offset_y;
  size_t offset_z;
};

struct KvMemcpy3DOperand {
  int type;
  union {
    KvMemcpy3DPointerOperand ptr;
    KvMemcpy3DArrayOperand array;
  } op;
};

struct KvMemcpy3DExtent {
  size_t width;
  size_t height;
  size_t depth;
};

struct KvMemcpy3DBatchOp {
  KvMemcpy3DOperand src;
  KvMemcpy3DOperand dst;
  KvMemcpy3DExtent extent;
  int src_access_order;
  unsigned int flags;
};

using KvMemcpy3DBatchAsyncFn = CUresult(CUDAAPI *)(
    size_t,
    KvMemcpy3DBatchOp*,
    size_t*,
    unsigned long long,
    CUstream);

static_assert(sizeof(KvMemcpy3DOperand) == 40, "Unexpected CUDA batch operand ABI");
static_assert(sizeof(KvMemcpy3DBatchOp) == 112, "Unexpected CUDA batch operation ABI");

KvMemcpy3DBatchAsyncFn resolve_memcpy_3d_batch_async() {
  static std::once_flag once;
  static KvMemcpy3DBatchAsyncFn fn = nullptr;
  std::call_once(once, [] {
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      fn = reinterpret_cast<KvMemcpy3DBatchAsyncFn>(
          dlsym(handle, "cuMemcpy3DBatchAsync"));
    }
  });
  return fn;
}

KvMemcpy3DOperand make_batch_pointer_operand(
    const void* ptr,
    size_t row_length,
    size_t layer_height,
    CUmemLocationType location_type,
    int location_id) {
  KvMemcpy3DOperand operand{};
  operand.type = kMemcpyOperandPointer;
  operand.op.ptr.ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
  operand.op.ptr.row_length = row_length;
  operand.op.ptr.layer_height = layer_height;
  operand.op.ptr.location_hint.type = location_type;
  operand.op.ptr.location_hint.id = location_id;
  return operand;
}

void append_batch_2d_copy(
    std::vector<KvMemcpy3DBatchOp>& ops,
    const void* src,
    size_t src_pitch,
    void* dst,
    size_t dst_pitch,
    size_t width_bytes,
    size_t height,
    CUmemLocationType src_location_type,
    int src_location_id,
    CUmemLocationType dst_location_type,
    int dst_location_id,
    unsigned int flags = kMemcpyPreferOverlapWithCompute) {
  KvMemcpy3DBatchOp op{};
  op.src = make_batch_pointer_operand(
      src, src_pitch, height, src_location_type, src_location_id);
  op.dst = make_batch_pointer_operand(
      dst, dst_pitch, height, dst_location_type, dst_location_id);
  op.extent = {width_bytes, height, 1};
  op.src_access_order = kMemcpySrcAccessOrderStream;
  op.flags = flags;
  ops.push_back(op);
}

void ensure_current_cuda_context_for_driver_batch(int device, const char* label) {
  static thread_local int initialized_device = -1;
  if (initialized_device == device) {
    return;
  }
  cudaError_t runtime_result = cudaSetDevice(device);
  TORCH_CHECK(runtime_result == cudaSuccess, "cudaSetDevice before ", label, " failed: ",
              cudaGetErrorString(runtime_result));
  runtime_result = cudaFree(nullptr);
  TORCH_CHECK(runtime_result == cudaSuccess, "CUDA context initialization before ", label, " failed: ",
              cudaGetErrorString(runtime_result));
  initialized_device = device;
}


// Own descriptor collection and driver/runtime dispatch for one stream.
// Contiguous copies use height=1; D2H transposes token/head layout via pitches.
class KvCopyBatch {
 public:
  KvCopyBatch(cudaStream_t stream, cudaMemcpyKind kind, int device, size_t capacity)
      : stream_(stream), kind_(kind), device_(device),
        batch_fn_(resolve_memcpy_3d_batch_async()) {
    if (batch_fn_ != nullptr && stream_ != nullptr) {
      ops_.reserve(capacity);
    } else {
      batch_fn_ = nullptr;
    }
  }

  void add(const void* src, void* dst, size_t width, size_t height = 1,
           size_t src_pitch = 0, size_t dst_pitch = 0) {
    src_pitch = src_pitch ? src_pitch : width;
    dst_pitch = dst_pitch ? dst_pitch : width;
    if (batch_fn_ != nullptr) {
      const bool src_host = kind_ == cudaMemcpyHostToDevice;
      const bool dst_host = kind_ == cudaMemcpyDeviceToHost;
      append_batch_2d_copy(
          ops_, src, src_pitch, dst, dst_pitch, width, height,
          src_host ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE,
          src_host ? 0 : device_,
          dst_host ? CU_MEM_LOCATION_TYPE_HOST : CU_MEM_LOCATION_TYPE_DEVICE,
          dst_host ? 0 : device_, dst_host ? 0 : kMemcpyPreferOverlapWithCompute);
    } else {
      cudaError_t result = height == 1
          ? cudaMemcpyAsync(dst, src, width, kind_, stream_)
          : cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, height, kind_, stream_);
      TORCH_CHECK(result == cudaSuccess, "KV async copy failed: ", cudaGetErrorString(result));
    }
  }

  void submit() {
    if (ops_.empty()) {
      return;
    }
    ensure_current_cuda_context_for_driver_batch(device_, "KV copy batch");
    size_t fail_idx = std::numeric_limits<size_t>::max();
    CUresult result = batch_fn_(
        ops_.size(), ops_.data(), &fail_idx, 0, reinterpret_cast<CUstream>(stream_));
    TORCH_CHECK(result == CUDA_SUCCESS, "cuMemcpy3DBatchAsync failed with CUresult=",
                static_cast<int>(result), ", failing descriptor=", fail_idx,
                ", descriptor_count=", ops_.size());
  }

 private:
  cudaStream_t stream_;
  cudaMemcpyKind kind_;
  int device_;
  KvMemcpy3DBatchAsyncFn batch_fn_;
  std::vector<KvMemcpy3DBatchOp> ops_;
};

}  // namespace
