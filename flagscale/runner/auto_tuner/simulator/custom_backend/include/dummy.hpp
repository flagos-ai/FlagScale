#pragma once

#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/chrono.h>

#include <variant>

using AnyType = std::variant<int, double, std::string>;


namespace c10d {

class ProcessGroup; // 假设的类
class Store; // 假设的类

class BackendDummy : public Backend {
 public:

  BackendDummy(int rank, int size);

  const std::string getBackendName() const override;
  void startCoalescing() override;
  c10::intrusive_ptr<Work> endCoalescing() override;

c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors/* outputs */,
      std::vector<at::Tensor>& inputTensors/* inputs */,
      const AllgatherOptions& /* opts */ = AllgatherOptions()) override;

c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensors/* outputBuffer */,
      at::Tensor& inputTensors/* inputBuffer */,
      const ReduceScatterOptions& /* opts */ = ReduceScatterOptions()) override;

c10::intrusive_ptr<Work> broadcast(
    std::vector<at::Tensor> &data,
    const BroadcastOptions &opts = BroadcastOptions()) override;

c10::intrusive_ptr<Work> allreduce(
    std::vector<at::Tensor> &tensors,
    const AllreduceOptions &opts = AllreduceOptions()) override;

c10::intrusive_ptr<Work> allreduce_coalesced(
    std::vector<at::Tensor> &tensors,
    const AllreduceCoalescedOptions &opts =
        AllreduceCoalescedOptions()) override;

c10::intrusive_ptr<Work> reduce(
    std::vector<at::Tensor> &tensors,
    const ReduceOptions &opts = ReduceOptions()) override;

c10::intrusive_ptr<Work> all_gather_object(
    std::vector<AnyType> &outputTensors,
    AnyType &inputTensors,
    const AllgatherOptions &opts = AllgatherOptions());

c10::intrusive_ptr<Work> allgather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllgatherOptions &opts = AllgatherOptions()) override;

c10::intrusive_ptr<Work> _allgather_base(
    at::Tensor &outputBuffer,
    at::Tensor &inputBuffer,
    const AllgatherOptions &opts = AllgatherOptions()) override;

c10::intrusive_ptr<Work> barrier(
    const BarrierOptions &opts = BarrierOptions()) override;

c10::intrusive_ptr<Work> gather(
    std::vector<std::vector<at::Tensor>> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const GatherOptions &opts = GatherOptions()) override;

c10::intrusive_ptr<Work> scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ScatterOptions &opts = ScatterOptions()) override;

c10::intrusive_ptr<Work> reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

c10::intrusive_ptr<Work> alltoall_base(
    at::Tensor &outputTensor,
    at::Tensor &inputTensor,
    std::vector<int64_t> &outputSplitSizes,
    std::vector<int64_t> &inputSplitSizes,
    const AllToAllOptions &opts = AllToAllOptions()) override;

c10::intrusive_ptr<Work> alltoall(
    std::vector<at::Tensor> &outputTensors,
    std::vector<at::Tensor> &inputTensors,
    const AllToAllOptions &opts = AllToAllOptions()) override;

c10::intrusive_ptr<Work> send(
    std::vector<at::Tensor> &tensors,
    int dstRank,
    int tag) override;

c10::intrusive_ptr<Work> recv(
    std::vector<at::Tensor> &tensors,
    int srcRank,
    int tag) override;

c10::intrusive_ptr<Work> recvAnysource(
    std::vector<at::Tensor> &tensors,
    int tag) override;

static c10::intrusive_ptr<Backend> createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store> &store,
    int rank,
    int size,
    const std::chrono::duration<float> &timeout);

static void BackendDummyConstructor() __attribute__((constructor))
{
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("dummy", py::cpp_function(createBackendDummy));
  }
};

class WorkDummy : public Work {
    friend class BackendDummy;
public:
    WorkDummy(
        OpType opType,
        c10::intrusive_ptr<c10::ivalue::Future> future) // future of the output
        : Work(
              -1, // rank, only used by recvAnySource, irrelevant in this demo
              opType),
          future_(std::move(future)) {}
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

} // namespace c10d
