// Modified for the HorizonKV inference artifact; see NOTICE.
#include "kv_sparse_onload.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_sparse_adaptive_count_misses", &kv_sparse_adaptive_count_misses_cpu,
        "Read-only allocation-free live/resident miss count");

  m.def("kv_sparse_plan_slots_tensor", &kv_sparse_plan_slots_tensor_cpp,
        "CPU tensor slot planning for sparse KV onload",
        py::arg("frame_indices"), py::arg("frame_to_slot"), py::arg("slot_frame_ids"),
        py::arg("slot_frame_lens"), py::arg("slot_last_used"), py::arg("total_tokens"),
        py::arg("frame_seqlen"), py::arg("num_heads"), py::arg("gpu_cache_frames"),
        py::arg("live_start_index"), py::arg("live_len"), py::arg("has_live"),
        py::arg("gpu_clock"), py::arg("build_compact_plans") = true,
        py::arg("split_min_miss_items") = 0);
  m.attr("slot_plan_version") = 3;
  m.def("kv_sparse_frame_materialize", &kv_sparse_frame_materialize_cuda,
        "Fused sparse KV frame materialization from GPU slots/live KV (CUDA)");
  m.def("kv_sparse_frame_materialize_varlen", &kv_sparse_frame_materialize_varlen_cuda,
        "Fused sparse KV frame materialization into FlashAttention varlen packed layout (CUDA)");

  m.def("kv_sparse_select_pooled_qk_topk", &kv_sparse_select_pooled_qk_topk_cuda,
        "Fused pooled-QK frame scoring and max-kv-frame selection (CUDA)");

  m.def("kv_sparse_update_frame_slot_map", &kv_sparse_update_frame_slot_map_cuda,
        "Batched GPU update of dense frame-to-slot metadata");

  m.def("kv_sparse_copy_frame_plan_to_slots", &kv_sparse_copy_frame_plan_to_slots_cuda,
        "Batched async HtoD copy of a dense CPU KV frame plan into GPU slots");


  m.def("kv_sparse_copy_kv_span_to_cpu", &kv_sparse_copy_kv_span_to_cpu_cuda,
        "Async DtoH copy of a contiguous KV token span into pinned CPU cache");

  m.def("kv_sparse_copy_frames_to_compact", &kv_sparse_copy_frames_to_compact_cuda,
        "Batched async HtoD copy of CPU-offloaded KV frame/head items into compact packed GPU buffers");

  m.def("kv_sparse_copy_compact_to_slots_d2d", &kv_sparse_copy_compact_to_slots_d2d_cuda,
        "Async DtoD copy from packed compact miss KV into GPU cache slots");
}
