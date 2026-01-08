#include "dummy.hpp"
#include <iostream>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <iostream>
// #include <vector>

namespace c10d {


bool WorkDummy::isCompleted() {
  return true;
}

bool WorkDummy::isSuccess() const {
  return true;
}

bool WorkDummy::wait(std::chrono::milliseconds /* unused */) {
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkDummy::getFuture() {
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
BackendDummy::BackendDummy(int rank, int size)
    : Backend(rank, size) {}

const std::string BackendDummy::getBackendName() const{
  return "dummy";
}

void BackendDummy::startCoalescing(){
    return;
  }

c10::intrusive_ptr<Work> BackendDummy::endCoalescing(){
  at::Tensor outputTensors;
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions&) {
        // printf("dummy reduce_scatter_tensor_coalesced\n");
    for (auto& outputTensor : outputTensors) {
      outputTensor.fill_(1);
    }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors/* outputs */,
      std::vector<at::Tensor>& inputTensors/* inputs */,
      const AllgatherOptions& ) {
        // printf("dummy reduce_scatter_tensor_coalesced\n");
    for (auto& outputTensor : outputTensors) {
      outputTensor.fill_(1);
    }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::_reduce_scatter_base(
      at::Tensor& outputTensors/* outputBuffer */,
      at::Tensor& inputTensors/* inputBuffer */,
      const ReduceScatterOptions& ) {
        // printf("dummy _reduce_scatter_base\n");
    // for (auto& outputTensor : outputTensors) {
    //   outputTensor.fill_(1);
    // }
    outputTensors.fill_(1);

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

// This is a dummy allgather that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& /* unused */) {
  // printf("dummy allgather\n");
  for (auto& outputTensorVec : outputTensors) {
      for (auto& outputTensor : outputTensorVec) {
          outputTensor.fill_(1);
      }
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  future->markCompleted(c10::IValue(outputTensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::all_gather_object(
    std::vector<AnyType>& outputTensors,
    AnyType& inputTensors,
    const AllgatherOptions& /* unused */) {
  // printf("dummy all_gather_object Begin\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::_allgather_base(
    at::Tensor& /* unused */,
    at::Tensor& /* unused */,
    const AllgatherOptions& /* unused */) {
  // printf("dummy _allgather_base\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

// This is a dummy allreduce that sets all output tensors to zero
// Modify the implementation to conduct real communication asynchronously
c10::intrusive_ptr<Work> BackendDummy::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  // printf("dummy allreduce\n");
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */) {
  // printf("dummy allreduce_coalesced\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::alltoall(
    std::vector<at::Tensor>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllToAllOptions& /* unused */) {
  // printf("dummy alltoall\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  // printf("dummy alltoall_base\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::barrier(
    const BarrierOptions& /* unused */) {
  // printf("dummy barrier\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  // printf("dummy broadcast\n");
  for (auto& tensor : tensors) {
      tensor.zero_();
  }

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::TensorType::get()));
  future->markCompleted(c10::IValue(tensors));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  // printf("dummy gather\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::reduce(
    std::vector<at::Tensor>& /* unused */,
    const ReduceOptions& /* unused */) {
  // printf("dummy reduce\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  // printf("dummy reduce_scatter\n");
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  // printf("dummy scatter\n");

  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Work> BackendDummy::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  auto future = c10::make_intrusive<c10::ivalue::Future>(
    c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
    const c10::intrusive_ptr<::c10d::Store>& /* unused */,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */) {
  return c10::make_intrusive<BackendDummy>(rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}

} // namespace c10d
