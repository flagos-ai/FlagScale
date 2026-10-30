// Modified for the HorizonKV inference artifact; see NOTICE.
#include "kv_sparse_onload.h"
#include <pybind11/stl.h>
#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

int64_t kv_sparse_adaptive_count_misses_cpu(
    torch::Tensor frames, torch::Tensor lookup, torch::Tensor ids, torch::Tensor lens,
    int64_t live_first, int64_t live_end, int64_t frame_length) {
  for (const auto& tensor : {frames, lookup, ids, lens}) {
    TORCH_CHECK(
        tensor.device().is_cpu() && tensor.scalar_type() == at::kLong &&
            tensor.dim() == 2 && tensor.is_contiguous(),
        "Miss counting requires contiguous CPU int64 matrices");
  }
  const int64_t heads = frames.size(0);
  const int64_t selected = frames.size(1);
  const int64_t frame_count = lookup.size(1);
  const int64_t slots = ids.size(1);
  TORCH_CHECK(
      heads == lookup.size(0) && heads == ids.size(0) && lens.sizes() == ids.sizes() &&
          frame_length > 0,
      "Invalid miss count layout");
  const auto* frame_ptr = frames.data_ptr<int64_t>();
  const auto* lookup_ptr = lookup.data_ptr<int64_t>();
  const auto* id_ptr = ids.data_ptr<int64_t>();
  const auto* length_ptr = lens.data_ptr<int64_t>();
  int64_t misses = 0;
  for (int64_t head = 0; head < heads; ++head) {
    for (int64_t item = 0; item < selected; ++item) {
      const int64_t frame = frame_ptr[head * selected + item];
      if (frame < 0 || frame >= frame_count) {
        return -1;
      }
      if (frame >= live_first && frame < live_end) {
        continue;
      }
      const int64_t slot = lookup_ptr[head * frame_count + frame];
      misses += slot < 0 || slot >= slots || id_ptr[head * slots + slot] != frame ||
                length_ptr[head * slots + slot] != frame_length;
    }
  }
  return misses;
}

namespace {

int64_t kv_frame_span_len(int64_t frame_id, int64_t frame_seqlen, int64_t total_tokens) {
  const int64_t start = frame_id * frame_seqlen;
  if (start >= total_tokens) {
    return 0;
  }
  return std::min<int64_t>(frame_seqlen, total_tokens - start);
}



struct PlannedFillState {
  int64_t slot = -1;
  int64_t evicted = -1;
  int64_t last_used = -1;
};





torch::Tensor make_int64_tensor_1d(const std::vector<int64_t>& values) {
  if (values.empty()) {
    return torch::empty({0}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  }
  return torch::from_blob(
             const_cast<int64_t*>(values.data()),
             {static_cast<int64_t>(values.size())},
             torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
      .clone();
}

py::object build_compact_scatter_plan_cpp(
    const int64_t* frame_ptr,
    const int64_t* slot_ptr,
    const int64_t* copy_ptr,
    int64_t rows,
    int64_t n_selected,
    const py::dict& miss_row_plan,
    int64_t copy_item_count,
    int64_t frame_seqlen,
    int64_t total_tokens) {
  const auto active_obj = miss_row_plan["active_rows"];
  const auto frame_plan_obj = miss_row_plan["frame_plan"];
  if (active_obj.is_none() || frame_plan_obj.is_none()) {
    return py::none();
  }
  torch::Tensor active_rows = py::cast<torch::Tensor>(active_obj);
  torch::Tensor frame_plan = py::cast<torch::Tensor>(frame_plan_obj);
  if (active_rows.device().is_cuda() || frame_plan.device().is_cuda() ||
      active_rows.scalar_type() != at::kLong || frame_plan.scalar_type() != at::kLong ||
      active_rows.dim() != 1 || frame_plan.dim() != 2 ||
      frame_plan.size(0) != active_rows.size(0)) {
    return py::none();
  }
  active_rows = active_rows.contiguous();
  frame_plan = frame_plan.contiguous();
  const int64_t active_count = active_rows.size(0);
  const int64_t max_selected = active_count > 0 ? frame_plan.size(1) : 0;
  const auto* active_ptr = active_rows.data_ptr<int64_t>();
  const auto* frame_plan_ptr = frame_plan.data_ptr<int64_t>();

  std::unordered_map<uint64_t, std::pair<int64_t, int64_t>> source_by_head_frame;
  source_by_head_frame.reserve(static_cast<size_t>(std::max<int64_t>(1, active_count * max_selected)));
  std::unordered_set<int64_t> wait_frame_set;
  std::vector<int64_t> frame_ids_for_wait;
  int64_t valid_item_count = 0;
  bool all_frames_full = true;
  wait_frame_set.reserve(static_cast<size_t>(std::max<int64_t>(1, active_count * max_selected)));
  frame_ids_for_wait.reserve(static_cast<size_t>(std::max<int64_t>(1, active_count * max_selected)));
  auto make_key = [](int64_t head_id, int64_t frame_id) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(head_id)) << 32) |
           static_cast<uint32_t>(frame_id);
  };

  for (int64_t dst_row = 0; dst_row < active_count; ++dst_row) {
    const int64_t head_id = active_ptr[dst_row];
    if (head_id < 0 || head_id >= rows) {
      return py::none();
    }
    for (int64_t dst_col = 0; dst_col < max_selected; ++dst_col) {
      const int64_t frame_id = frame_plan_ptr[dst_row * max_selected + dst_col];
      if (frame_id >= 0) {
        ++valid_item_count;
        if ((frame_id + 1) * frame_seqlen > total_tokens) {
          all_frames_full = false;
        }
        source_by_head_frame.emplace(make_key(head_id, frame_id), std::make_pair(dst_row, dst_col));
        if (wait_frame_set.insert(frame_id).second) {
          frame_ids_for_wait.push_back(frame_id);
        }
      }
    }
  }

  std::vector<int64_t> scatter_heads;
  std::vector<int64_t> scatter_slots;
  std::vector<int64_t> scatter_src_rows;
  std::vector<int64_t> scatter_src_cols;
  std::vector<int64_t> scatter_frame_ids;
  scatter_heads.reserve(static_cast<size_t>(std::max<int64_t>(0, copy_item_count)));
  scatter_slots.reserve(scatter_heads.capacity());
  scatter_src_rows.reserve(scatter_heads.capacity());
  scatter_src_cols.reserve(scatter_heads.capacity());
  scatter_frame_ids.reserve(scatter_heads.capacity());

  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t idx = row * n_selected + col;
      if (copy_ptr[idx] == 0) {
        continue;
      }
      const int64_t frame_id = frame_ptr[idx];
      const int64_t slot = slot_ptr[idx];
      if (frame_id < 0 || slot < 0) {
        return py::none();
      }
      const auto source_it = source_by_head_frame.find(make_key(row, frame_id));
      if (source_it == source_by_head_frame.end()) {
        return py::none();
      }
      scatter_heads.push_back(row);
      scatter_slots.push_back(slot);
      scatter_src_rows.push_back(source_it->second.first);
      scatter_src_cols.push_back(source_it->second.second);
      scatter_frame_ids.push_back(frame_id);
    }
  }
  if (static_cast<int64_t>(scatter_heads.size()) != copy_item_count) {
    return py::none();
  }
  const int64_t expected_item_count = py::cast<int64_t>(miss_row_plan["item_count"]);
  if (valid_item_count != expected_item_count) {
    return py::none();
  }

  torch::Tensor cu_seqlens_long;
  const auto cu_obj = miss_row_plan["cu_seqlens"];
  if (!cu_obj.is_none()) {
    auto cu_seqlens = py::cast<torch::Tensor>(cu_obj);
    if (!cu_seqlens.device().is_cuda() && cu_seqlens.scalar_type() == at::kInt &&
        cu_seqlens.dim() == 1 && cu_seqlens.size(0) == active_count + 1) {
      cu_seqlens_long = cu_seqlens.to(torch::kLong).contiguous();
    }
  }

  py::dict out;
  out["scatter_head_ids"] = make_int64_tensor_1d(scatter_heads);
  out["scatter_slot_ids"] = make_int64_tensor_1d(scatter_slots);
  out["scatter_src_rows"] = make_int64_tensor_1d(scatter_src_rows);
  out["scatter_src_cols"] = make_int64_tensor_1d(scatter_src_cols);
  out["scatter_frame_ids"] = make_int64_tensor_1d(scatter_frame_ids);
  out["frame_ids_for_wait"] = make_int64_tensor_1d(frame_ids_for_wait);
  if (cu_seqlens_long.defined()) {
    out["cu_seqlens_long"] = cu_seqlens_long;
  }
  out["all_frames_full"] = py::bool_(all_frames_full);
  return std::move(out);
}

py::dict build_compact_row_plan_cpp(
    const int64_t* frame_ptr,
    const int64_t* slot_ptr,
    const int64_t* fill_ptr,
    int64_t rows,
    int64_t n_selected,
    int64_t frame_seqlen,
    bool miss_plan) {
  std::vector<int64_t> row_counts(static_cast<size_t>(rows), 0);
  int64_t active_count = 0;
  int64_t max_selected = 0;
  int64_t item_count = 0;

  auto include_item = [=](int64_t frame_id, int64_t slot, int64_t fill) {
    const bool readable = frame_id >= 0 && (slot == -2 || slot >= 0);
    if (!readable) {
      return false;
    }
    const bool miss = fill != 0 && slot >= 0;
    return miss_plan ? miss : !miss;
  };

  for (int64_t row = 0; row < rows; ++row) {
    int64_t count = 0;
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t idx = row * n_selected + col;
      if (include_item(frame_ptr[idx], slot_ptr[idx], fill_ptr[idx])) {
        ++count;
      }
    }
    row_counts[static_cast<size_t>(row)] = count;
    if (count > 0) {
      ++active_count;
      max_selected = std::max(max_selected, count);
      item_count += count;
    }
  }

  const auto long_options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
  const auto int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  py::dict out;
  if (active_count <= 0 || max_selected <= 0) {
    out["active_rows"] = torch::empty({0}, long_options);
    out["frame_plan"] = torch::empty({0, 0}, long_options);
    out["slot_plan"] = torch::empty({0, 0}, long_options);
    out["frame_positions"] = torch::empty({0, 0}, long_options);
    out["k_lens"] = torch::empty({0}, int_options);
    out["cu_seqlens"] = torch::empty({0}, int_options);
    out["control_long"] = torch::empty({0}, long_options);
    out["control_int"] = torch::empty({0}, int_options);
    out["item_count"] = py::int_(0);
    out["all_full_length"] = py::bool_(true);
    out["covers_all_heads"] = py::bool_(false);
    return out;
  }

  // One allocation per dtype; the Python control uploader pins these buffers
  // once and retains views instead of pinning every field independently.
  const int64_t plan_items = active_count * max_selected;
  auto control_long = torch::full({active_count + 3 * plan_items}, static_cast<int64_t>(-1), long_options);
  auto active_rows = control_long.narrow(0, 0, active_count);
  auto frame_plan = control_long.narrow(0, active_count, plan_items).view({active_count, max_selected});
  auto slot_plan = control_long.narrow(0, active_count + plan_items, plan_items).view({active_count, max_selected});
  auto frame_positions = control_long.narrow(0, active_count + 2 * plan_items, plan_items).view({active_count, max_selected});
  auto control_int = torch::empty({2 * active_count + 1}, int_options);
  auto k_lens = control_int.narrow(0, 0, active_count);
  auto cu_seqlens = control_int.narrow(0, active_count, active_count + 1);

  auto* active_ptr = active_rows.data_ptr<int64_t>();
  auto* frame_out_ptr = frame_plan.data_ptr<int64_t>();
  auto* slot_out_ptr = slot_plan.data_ptr<int64_t>();
  auto* position_ptr = frame_positions.data_ptr<int64_t>();
  auto* k_lens_ptr = k_lens.data_ptr<int32_t>();
  auto* cu_ptr = cu_seqlens.data_ptr<int32_t>();

  int64_t dst_row = 0;
  int64_t prefix = 0;
  bool all_full_length = true;
  cu_ptr[0] = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t count = row_counts[static_cast<size_t>(row)];
    if (count <= 0) {
      continue;
    }
    active_ptr[dst_row] = row;
    if (count != max_selected) {
      all_full_length = false;
    }
    int64_t dst_col = 0;
    int64_t padding_position = -1;
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t idx = row * n_selected + col;
      if (frame_ptr[idx] == -1 && padding_position < 0) {
        padding_position = col;
      }
      if (!include_item(frame_ptr[idx], slot_ptr[idx], fill_ptr[idx])) {
        continue;
      }
      frame_out_ptr[dst_row * max_selected + dst_col] = frame_ptr[idx];
      slot_out_ptr[dst_row * max_selected + dst_col] = slot_ptr[idx];
      // Match the first occurrence, including compatibility with duplicate
      // selections. Ordinary top-K selections have unique frame IDs.
      int64_t position = 0;
      while (position < col && frame_ptr[row * n_selected + position] != frame_ptr[idx]) {
        ++position;
      }
      position_ptr[dst_row * max_selected + dst_col] = position;
      ++dst_col;
    }
    for (; dst_col < max_selected; ++dst_col) {
      position_ptr[dst_row * max_selected + dst_col] = padding_position;
    }
    const int64_t row_tokens = count * frame_seqlen;
    k_lens_ptr[dst_row] = static_cast<int32_t>(row_tokens);
    prefix += row_tokens;
    cu_ptr[dst_row + 1] = static_cast<int32_t>(prefix);
    ++dst_row;
  }

  out["active_rows"] = active_rows;
  out["frame_plan"] = frame_plan;
  out["slot_plan"] = slot_plan;
  out["frame_positions"] = frame_positions;
  out["k_lens"] = k_lens;
  out["cu_seqlens"] = cu_seqlens;
  out["control_long"] = control_long;
  out["control_int"] = control_int;
  out["item_count"] = py::int_(item_count);
  out["all_full_length"] = py::bool_(all_full_length);
  out["covers_all_heads"] = py::bool_(active_count == rows);
  return out;
}


}  // namespace

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
    int64_t split_min_miss_items) {
  TORCH_CHECK(!frame_indices.is_cuda(), "frame_indices must be a CPU tensor");
  TORCH_CHECK(!frame_to_slot.is_cuda(), "frame_to_slot must be a CPU tensor");
  TORCH_CHECK(!slot_frame_ids.is_cuda(), "slot_frame_ids must be a CPU tensor");
  TORCH_CHECK(!slot_frame_lens.is_cuda(), "slot_frame_lens must be a CPU tensor");
  TORCH_CHECK(!slot_last_used.is_cuda(), "slot_last_used must be a CPU tensor");
  TORCH_CHECK(frame_indices.scalar_type() == at::kLong, "frame_indices must be int64");
  TORCH_CHECK(frame_to_slot.scalar_type() == at::kLong, "frame_to_slot must be int64");
  TORCH_CHECK(slot_frame_ids.scalar_type() == at::kLong, "slot_frame_ids must be int64");
  TORCH_CHECK(slot_frame_lens.scalar_type() == at::kLong, "slot_frame_lens must be int64");
  TORCH_CHECK(slot_last_used.scalar_type() == at::kLong, "slot_last_used must be int64");
  TORCH_CHECK(frame_indices.dim() == 2, "frame_indices must be 2D");
  TORCH_CHECK(frame_to_slot.dim() == 2, "frame_to_slot must be 2D");
  TORCH_CHECK(slot_frame_ids.dim() == 2, "slot_frame_ids must be 2D");
  TORCH_CHECK(slot_frame_lens.sizes() == slot_frame_ids.sizes(),
              "slot_frame_lens shape mismatch");
  TORCH_CHECK(slot_last_used.sizes() == slot_frame_ids.sizes(),
              "slot_last_used shape mismatch");
  TORCH_CHECK(frame_seqlen > 0, "frame_seqlen must be positive");
  TORCH_CHECK(num_heads > 0, "num_heads must be positive");
  TORCH_CHECK(gpu_cache_frames > 0, "gpu_cache_frames must be positive");
  TORCH_CHECK(frame_indices.size(0) == num_heads, "frame_indices head dimension mismatch");
  TORCH_CHECK(frame_to_slot.size(0) == num_heads, "frame_to_slot head dimension mismatch");
  TORCH_CHECK(slot_frame_ids.size(0) == num_heads, "slot_frame_ids head dimension mismatch");
  TORCH_CHECK(slot_frame_ids.size(1) == gpu_cache_frames, "slot_frame_ids slot dimension mismatch");

  frame_indices = frame_indices.contiguous();
  frame_to_slot = frame_to_slot.contiguous();
  slot_frame_ids = slot_frame_ids.contiguous();
  slot_last_used = slot_last_used.contiguous();

  const auto* frame_ptr = frame_indices.data_ptr<int64_t>();
  const auto* frame_to_slot_ptr = frame_to_slot.data_ptr<int64_t>();
  const auto* slot_frame_ids_ptr = slot_frame_ids.data_ptr<int64_t>();
  const auto* slot_last_used_ptr = slot_last_used.data_ptr<int64_t>();
  const int64_t head_count = num_heads;
  const int64_t n_selected = frame_indices.size(1);
  const int64_t frame_capacity = frame_to_slot.size(1);
  if (head_count <= 0 || n_selected <= 0 || total_tokens <= 0 || frame_capacity <= 0) {
    return py::none();
  }

  std::vector<std::vector<int64_t>> local_slot_frame_ids(
      static_cast<size_t>(num_heads),
      std::vector<int64_t>(static_cast<size_t>(gpu_cache_frames), -1));
  std::vector<std::vector<int64_t>> local_slot_last_used(
      static_cast<size_t>(num_heads),
      std::vector<int64_t>(static_cast<size_t>(gpu_cache_frames), -1));
  for (int64_t head_id = 0; head_id < num_heads; ++head_id) {
    const int64_t slot_base = head_id * gpu_cache_frames;
    auto& local_ids = local_slot_frame_ids[static_cast<size_t>(head_id)];
    auto& local_last = local_slot_last_used[static_cast<size_t>(head_id)];
    for (int64_t slot = 0; slot < gpu_cache_frames; ++slot) {
      local_ids[static_cast<size_t>(slot)] = slot_frame_ids_ptr[slot_base + slot];
      local_last[static_cast<size_t>(slot)] = slot_last_used_ptr[slot_base + slot];
    }
  }

  auto slot_plan = torch::full(
      {head_count, n_selected},
      static_cast<int64_t>(-3),
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  auto fill_plan = torch::zeros(
      {head_count, n_selected},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  auto copy_plan = torch::zeros(
      {head_count, n_selected},
      torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
  auto* slot_plan_ptr = slot_plan.data_ptr<int64_t>();
  auto* fill_plan_ptr = fill_plan.data_ptr<int64_t>();
  auto* copy_plan_ptr = copy_plan.data_ptr<int64_t>();

  std::vector<std::unordered_set<int64_t>> history_by_head(static_cast<size_t>(num_heads));
  std::vector<std::vector<std::tuple<int64_t, int64_t, int64_t>>> valid_history_entries_by_row(
      static_cast<size_t>(head_count));

  const int64_t live_end_index = live_start_index + live_len;
  for (int64_t row = 0; row < head_count; ++row) {
    const int64_t head_id = row;
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t frame_id = frame_ptr[row * n_selected + col];
      if (frame_id < 0) {
        continue;
      }
      const int64_t frame_start = frame_id * frame_seqlen;
      const int64_t frame_len = kv_frame_span_len(frame_id, frame_seqlen, total_tokens);
      if (frame_len <= 0) {
        continue;
      }
      if (frame_id >= frame_capacity) {
        return py::none();
      }
      const int64_t frame_end = frame_start + frame_len;
      const bool is_live = has_live && frame_start >= live_start_index && frame_end <= live_end_index;
      if (is_live) {
        slot_plan_ptr[row * n_selected + col] = -2;
        continue;
      }
      auto& history = history_by_head[static_cast<size_t>(head_id)];
      history.insert(frame_id);
      valid_history_entries_by_row[static_cast<size_t>(row)].emplace_back(col, frame_id, frame_len);
    }
  }

  for (int64_t head_id = 0; head_id < num_heads; ++head_id) {
    if (static_cast<int64_t>(history_by_head[static_cast<size_t>(head_id)].size()) > gpu_cache_frames) {
      return py::none();
    }
  }

  int64_t clock = gpu_clock;
  for (int64_t row = 0; row < head_count; ++row) {
    const int64_t head_id = row;
    const int64_t map_base = head_id * frame_capacity;
    auto& local_ids = local_slot_frame_ids[static_cast<size_t>(head_id)];
    auto& local_last = local_slot_last_used[static_cast<size_t>(head_id)];
    for (const auto& entry : valid_history_entries_by_row[static_cast<size_t>(row)]) {
      const int64_t col = std::get<0>(entry);
      const int64_t frame_id = std::get<1>(entry);
      const int64_t slot = frame_to_slot_ptr[map_base + frame_id];
      if (slot < 0 || slot >= gpu_cache_frames) {
        continue;
      }
      if (local_ids[static_cast<size_t>(slot)] != frame_id) {
        continue;
      }
      slot_plan_ptr[row * n_selected + col] = slot;
      const int64_t last_used = clock++;
      local_last[static_cast<size_t>(slot)] = last_used;

    }
  }

  std::vector<std::vector<unsigned char>> protected_slots_by_head(
      static_cast<size_t>(num_heads),
      std::vector<unsigned char>(static_cast<size_t>(gpu_cache_frames), 0));
  for (int64_t row = 0; row < head_count; ++row) {
    const int64_t head_id = row;
    auto& protected_slots = protected_slots_by_head[static_cast<size_t>(head_id)];
    for (const auto& entry : valid_history_entries_by_row[static_cast<size_t>(row)]) {
      const int64_t col = std::get<0>(entry);
      const int64_t slot = slot_plan_ptr[row * n_selected + col];
      if (slot >= 0 && slot < gpu_cache_frames) {
        protected_slots[static_cast<size_t>(slot)] = 1;
      }
    }
  }

  std::unordered_map<uint64_t, PlannedFillState> planned_fills;
  planned_fills.reserve(static_cast<size_t>(head_count * n_selected));
  auto make_key = [](int64_t head_id, int64_t frame_id) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(head_id)) << 32) |
           static_cast<uint32_t>(frame_id);
  };

  for (int64_t row = 0; row < head_count; ++row) {
    const int64_t head_id = row;
    auto& local_ids = local_slot_frame_ids[static_cast<size_t>(head_id)];
    auto& local_last = local_slot_last_used[static_cast<size_t>(head_id)];

    std::vector<int64_t> free_slots;
    free_slots.reserve(static_cast<size_t>(gpu_cache_frames));
    for (int64_t slot = 0; slot < gpu_cache_frames; ++slot) {
      if (local_ids[static_cast<size_t>(slot)] == -1) {
        free_slots.push_back(slot);
      }
    }

    for (const auto& entry : valid_history_entries_by_row[static_cast<size_t>(row)]) {
      const int64_t col = std::get<0>(entry);
      const int64_t frame_id = std::get<1>(entry);
      if (slot_plan_ptr[row * n_selected + col] != -3) {
        continue;
      }
      const uint64_t key = make_key(head_id, frame_id);
      const auto planned_it = planned_fills.find(key);
      if (planned_it != planned_fills.end()) {
        slot_plan_ptr[row * n_selected + col] = planned_it->second.slot;
        fill_plan_ptr[row * n_selected + col] = 1;
        continue;
      }

      int64_t slot = -1;
      int64_t evicted = -1;
      if (!free_slots.empty()) {
        slot = free_slots.back();
        free_slots.pop_back();
      } else {
        int64_t best_last_used = std::numeric_limits<int64_t>::max();
        const auto& protected_slots = protected_slots_by_head[static_cast<size_t>(head_id)];
        for (int64_t candidate = 0; candidate < gpu_cache_frames; ++candidate) {
          if (local_ids[static_cast<size_t>(candidate)] == -2) {
            continue;
          }
          if (protected_slots[static_cast<size_t>(candidate)] != 0) {
            continue;
          }
          const int64_t candidate_last_used = local_last[static_cast<size_t>(candidate)];
          if (slot < 0 || candidate_last_used < best_last_used) {
            slot = candidate;
            best_last_used = candidate_last_used;
          }
        }
        if (slot < 0) {
          return py::none();
        }
        evicted = local_ids[static_cast<size_t>(slot)];
      }

      const int64_t last_used = clock++;
      local_ids[static_cast<size_t>(slot)] = frame_id;
      local_last[static_cast<size_t>(slot)] = last_used;
      PlannedFillState fill_state;
      fill_state.slot = slot;
      fill_state.evicted = evicted;
      fill_state.last_used = last_used;
      planned_fills.emplace(key, fill_state);
      slot_plan_ptr[row * n_selected + col] = slot;
      fill_plan_ptr[row * n_selected + col] = 1;
      copy_plan_ptr[row * n_selected + col] = 1;
    }
  }

  int64_t readable_item_count = 0;
  int64_t miss_item_count = 0;
  int64_t copy_item_count = 0;
  for (int64_t row = 0; row < head_count; ++row) {
    for (int64_t col = 0; col < n_selected; ++col) {
      const int64_t idx = row * n_selected + col;
      const int64_t frame_id = frame_ptr[idx];
      const int64_t slot = slot_plan_ptr[idx];
      const int64_t fill = fill_plan_ptr[idx];
      const int64_t copy = copy_plan_ptr[idx];
      const bool readable = frame_id >= 0 && (slot == -2 || slot >= 0);
      const bool miss = readable && fill != 0 && slot >= 0;
      if (readable) {
        ++readable_item_count;
      }
      if (miss) {
        ++miss_item_count;
      }
      if (copy != 0) {
        ++copy_item_count;

      }
    }
  }

  // Compact row/scatter plans are only consumed by the split hit/miss path.
  // Avoid their allocations and repeated scans for all-hit/all-miss plans.
  py::object hit_row_plan = py::none();
  py::object miss_row_plan = py::none();
  py::object compact_scatter_plan = py::none();
  if (build_compact_plans && readable_item_count > miss_item_count &&
      miss_item_count > 0 && copy_item_count > 0 &&
      miss_item_count >= split_min_miss_items) {
    hit_row_plan = build_compact_row_plan_cpp(
        frame_ptr,
        slot_plan_ptr,
        fill_plan_ptr,
        head_count,
        n_selected,
        frame_seqlen,
        false);
    miss_row_plan = build_compact_row_plan_cpp(
        frame_ptr,
        slot_plan_ptr,
        fill_plan_ptr,
        head_count,
        n_selected,
        frame_seqlen,
        true);
    if (copy_item_count > 0) {
      compact_scatter_plan = build_compact_scatter_plan_cpp(
          frame_ptr,
          slot_plan_ptr,
          copy_plan_ptr,
          head_count,
          n_selected,
          py::cast<py::dict>(miss_row_plan),
          copy_item_count,
          frame_seqlen,
          total_tokens);
    }
  }

  std::vector<int64_t> fill_head_ids;
  std::vector<int64_t> fill_frame_ids;
  std::vector<int64_t> fill_slot_ids;
  std::vector<int64_t> fill_last_used;
  fill_head_ids.reserve(planned_fills.size());
  fill_frame_ids.reserve(planned_fills.size());
  fill_slot_ids.reserve(planned_fills.size());
  fill_last_used.reserve(planned_fills.size());
  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> fill_items;
  fill_items.reserve(planned_fills.size());
  for (const auto& item : planned_fills) {
    const int64_t head_id = static_cast<int64_t>(item.first >> 32);
    const int64_t frame_id = static_cast<int64_t>(static_cast<uint32_t>(item.first & 0xffffffffULL));
    fill_items.emplace_back(
        head_id,
        frame_id,
        item.second.slot,
        item.second.evicted,
        item.second.last_used);
  }
  std::sort(fill_items.begin(), fill_items.end());
  for (const auto& item : fill_items) {
    fill_head_ids.push_back(std::get<0>(item));
    fill_frame_ids.push_back(std::get<1>(item));
    fill_slot_ids.push_back(std::get<2>(item));
    fill_last_used.push_back(std::get<4>(item));
  }

  py::dict out;
  out["slot_plan"] = slot_plan;
  out["fill_plan"] = fill_plan;
  out["copy_plan"] = copy_plan;
  out["readable_item_count"] = py::int_(readable_item_count);
  out["miss_item_count"] = py::int_(miss_item_count);
  out["copy_item_count"] = py::int_(copy_item_count);
  out["hit_row_plan"] = hit_row_plan;
  out["miss_row_plan"] = miss_row_plan;
  if (!compact_scatter_plan.is_none()) {
    out["compact_scatter_plan"] = compact_scatter_plan;
  }
  out["fill_head_ids"] = make_int64_tensor_1d(fill_head_ids);
  out["fill_frame_ids"] = make_int64_tensor_1d(fill_frame_ids);
  out["fill_slot_ids"] = make_int64_tensor_1d(fill_slot_ids);
  out["fill_last_used"] = make_int64_tensor_1d(fill_last_used);
  out["gpu_clock"] = py::int_(clock);
  out["tensor_slot_state_incremental"] = py::bool_(true);
  return std::move(out);
}
