// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <nccl.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <time.h>
#include <unistd.h>

namespace flagscale::tracing {

constexpr std::size_t kQueueCapacity = 16384;
constexpr std::size_t kMaxBatchSize = 512;

enum class EventKind : uint8_t {
  kProcessStart,
  kCommInit,
  kCommDestroy,
  kNcclCall,
};

struct Event {
  EventKind kind = EventKind::kNcclCall;
  uint64_t timestamp_unix_ns = 0;
  uint64_t timestamp_mono_ns = 0;
  uint64_t comm_seq = 0;
  uint64_t call_seq = 0;
  uint64_t count = 0;
  uint64_t stream = 0;
  uint64_t group_id = 0;
  uint64_t group_op_index = 0;
  uint64_t p2p_op_index = 0;
  int comm_rank = -1;
  int comm_nranks = 0;
  int datatype = -1;
  int op = -1;
  int root = -1;
  int peer = -1;
  int result = 0;
  char api[32] = {};
  char phase[8] = {};
  char comm_uid_hash[17] = {};
  char result_name[32] = {};
};

uint64_t clock_ns(clockid_t clock_id) {
  timespec value{};
  if (::clock_gettime(clock_id, &value) != 0) return 0;
  return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL +
         static_cast<uint64_t>(value.tv_nsec);
}

int env_int(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') return fallback;
  return static_cast<int>(parsed);
}

bool env_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) return false;
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "yes") == 0;
}

void copy_text(char* destination, std::size_t size, const char* source) {
  if (size == 0) return;
  if (source == nullptr) source = "";
  std::snprintf(destination, size, "%s", source);
}

std::string json_string(const char* value) {
  std::string output;
  output.push_back('"');
  if (value != nullptr) {
    for (const unsigned char ch : std::string(value)) {
      switch (ch) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
          if (ch < 0x20) {
            char escaped[7] = {};
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
            output += escaped;
          } else {
            output.push_back(static_cast<char>(ch));
          }
      }
    }
  }
  output.push_back('"');
  return output;
}

const char* result_name(ncclResult_t result) {
  // NCCL keeps these public result values stable across the versions supported
  // by FlagScale. Avoid a direct ncclGetErrorString reference here because
  // PyTorch may load its bundled NCCL with RTLD_LOCAL.
  switch (static_cast<int>(result)) {
    case 0: return "ncclSuccess";
    case 1: return "ncclUnhandledCudaError";
    case 2: return "ncclSystemError";
    case 3: return "ncclInternalError";
    case 4: return "ncclInvalidArgument";
    case 5: return "ncclInvalidUsage";
    case 6: return "ncclRemoteError";
    case 7: return "ncclInProgress";
    default: return "ncclUnknownError";
  }
}

uint64_t hash_unique_id(const ncclUniqueId& unique_id) {
  constexpr uint64_t kOffset = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t hash = kOffset;
  const auto* bytes = reinterpret_cast<const unsigned char*>(&unique_id);
  for (std::size_t index = 0; index < sizeof(unique_id); ++index) {
    hash ^= bytes[index];
    hash *= kPrime;
  }
  return hash;
}

void format_hash(uint64_t hash, char output[17]) {
  std::snprintf(output, 17, "%016llx", static_cast<unsigned long long>(hash));
}

struct CommState {
  uintptr_t handle = 0;
  int rank = -1;
  int nranks = 0;
  char uid_hash[17] = {};
  // P2P calls do not involve every communicator rank, so they must not shift the
  // collective sequence used for cross-rank collective alignment.
  std::atomic<uint64_t> next_collective_sequence{0};
  std::atomic<uint64_t> next_p2p_sequence{0};
  std::unique_ptr<std::atomic<uint64_t>[]> next_send_sequences;
  std::unique_ptr<std::atomic<uint64_t>[]> next_recv_sequences;
  std::atomic<bool> active{true};
};

class CommRegistry {
 public:
  CommState* Register(ncclComm_t comm, int rank, int nranks, const ncclUniqueId& unique_id) {
    if (comm == nullptr) return nullptr;
    auto state = std::make_unique<CommState>();
    state->handle = reinterpret_cast<uintptr_t>(comm);
    state->rank = rank;
    state->nranks = nranks;
    if (nranks > 0) {
      state->next_send_sequences = std::make_unique<std::atomic<uint64_t>[]>(nranks);
      state->next_recv_sequences = std::make_unique<std::atomic<uint64_t>[]>(nranks);
      for (int peer = 0; peer < nranks; ++peer) {
        state->next_send_sequences[peer].store(0);
        state->next_recv_sequences[peer].store(0);
      }
    }
    format_hash(hash_unique_id(unique_id), state->uid_hash);
    CommState* pointer = state.get();
    std::lock_guard<std::mutex> lock(mutex_);
    const auto existing = active_.find(state->handle);
    if (existing != active_.end()) existing->second->active.store(false);
    storage_.push_back(std::move(state));
    active_[pointer->handle] = pointer;
    return pointer;
  }

  CommState* Find(ncclComm_t comm) {
    if (comm == nullptr) return nullptr;
    const uintptr_t handle = reinterpret_cast<uintptr_t>(comm);
    thread_local uintptr_t cached_handle = 0;
    thread_local CommState* cached_state = nullptr;
    if (cached_handle == handle && cached_state != nullptr && cached_state->active.load()) {
      return cached_state;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = active_.find(handle);
    if (found == active_.end()) return nullptr;
    cached_handle = handle;
    cached_state = found->second;
    return cached_state;
  }

  CommState* Remove(ncclComm_t comm) {
    if (comm == nullptr) return nullptr;
    const uintptr_t handle = reinterpret_cast<uintptr_t>(comm);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = active_.find(handle);
    if (found == active_.end()) return nullptr;
    CommState* state = found->second;
    state->active.store(false);
    active_.erase(found);
    return state;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<uintptr_t, CommState*> active_;
  std::vector<std::unique_ptr<CommState>> storage_;
};

class Runtime {
 public:
  static Runtime& Instance() {
    // Deliberately keep the runtime alive until the process is reclaimed by the OS.
    // A normal static destructor would have to join the writer thread; if the trace
    // filesystem is unhealthy, that diagnostic join could delay training shutdown.
    static Runtime* runtime = new Runtime();
    return *runtime;
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  bool enabled() const { return enabled_; }
  int rank() const { return rank_; }
  CommRegistry& comms() { return comms_; }
  uint64_t NextCallSequence() { return next_call_sequence_.fetch_add(1); }
  uint64_t NextGroupId() { return next_group_id_.fetch_add(1) + 1; }

  void Enqueue(const Event& event) {
    if (!enabled_) return;
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (queue_size_ == kQueueCapacity) {
        dropped_events_.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      queue_[queue_tail_] = event;
      queue_tail_ = (queue_tail_ + 1) % kQueueCapacity;
      ++queue_size_;
    }
    queue_cv_.notify_one();
  }

 private:
  Runtime() noexcept {
    try {
      if (!env_enabled("FLAGSCALE_TRACE_ENABLE")) return;
      const char* rank_value = std::getenv("RANK");
      const char* trace_dir = std::getenv("FLAGSCALE_TRACE_DIR");
      const char* run_id = std::getenv("FLAGSCALE_TRACE_RUN_ID");
      // The torchrun parent also inherits LD_PRELOAD. Only worker processes have RANK.
      if (rank_value == nullptr || trace_dir == nullptr || run_id == nullptr) return;

      rank_ = env_int("RANK", -1);
      if (rank_ < 0) return;
      local_rank_ = env_int("LOCAL_RANK", -1);
      world_size_ = env_int("WORLD_SIZE", 0);
      run_id_ = run_id;
      trace_dir_ = trace_dir;
      char hostname[256] = {};
      if (::gethostname(hostname, sizeof(hostname) - 1) == 0) hostname_ = hostname;

      // The launch script creates the directory before LD_PRELOAD reaches workers.
      // Keep directory and file I/O off the loader thread.
      file_path_ = trace_dir_ + "/rank_" + std::to_string(rank_) + "_pid_" +
                   std::to_string(static_cast<long long>(::getpid())) + ".jsonl";
      enabled_ = true;
      worker_ = std::thread(&Runtime::WorkerLoop, this);
    } catch (...) {
      // Tracing is diagnostic-only. Failure to allocate or start its worker must not
      // prevent the training process from loading.
      enabled_ = false;
    }
  }

  ~Runtime() {
    if (!enabled_) return;
    stopping_.store(true);
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

  Event MakeProcessStart() const {
    Event event;
    event.kind = EventKind::kProcessStart;
    event.timestamp_unix_ns = clock_ns(CLOCK_REALTIME);
    event.timestamp_mono_ns = clock_ns(CLOCK_MONOTONIC);
    return event;
  }

  void WorkerLoop() noexcept {
    try {
      WorkerLoopImpl();
    } catch (...) {
      // Never let a logging/allocation exception terminate the training process.
    }
  }

  void WorkerLoopImpl() {
    const int fd = ::open(file_path_.c_str(), O_CREAT | O_WRONLY | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) return;
    WriteEvent(fd, MakeProcessStart());
    WriteProbeStatus(fd, 0);

    std::vector<Event> batch;
    batch.reserve(kMaxBatchSize);
    uint64_t reported_drops = 0;

    while (!stopping_.load()) {
      batch.clear();
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return queue_size_ > 0 || stopping_.load(); });
        while (queue_size_ > 0 && batch.size() < kMaxBatchSize) {
          batch.push_back(queue_[queue_head_]);
          queue_head_ = (queue_head_ + 1) % kQueueCapacity;
          --queue_size_;
        }
      }
      for (const Event& event : batch) WriteEvent(fd, event);
      const uint64_t drops = dropped_events_.load(std::memory_order_relaxed);
      if (drops != reported_drops) {
        WriteProbeStatus(fd, drops);
        reported_drops = drops;
      }
    }

    while (true) {
      batch.clear();
      {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (queue_size_ > 0 && batch.size() < kMaxBatchSize) {
          batch.push_back(queue_[queue_head_]);
          queue_head_ = (queue_head_ + 1) % kQueueCapacity;
          --queue_size_;
        }
      }
      if (batch.empty()) break;
      for (const Event& event : batch) WriteEvent(fd, event);
    }
    ::fsync(fd);
    ::close(fd);
  }

  void WriteAll(int fd, const std::string& line) const {
    const char* cursor = line.data();
    std::size_t remaining = line.size();
    while (remaining > 0) {
      const ssize_t written = ::write(fd, cursor, remaining);
      if (written < 0 && errno == EINTR) continue;
      if (written <= 0) return;
      cursor += written;
      remaining -= static_cast<std::size_t>(written);
    }
  }

  std::string CommonJsonPrefix(const char* event_name, uint64_t unix_ns,
                               uint64_t mono_ns) const {
    std::string line = "{\"schema_version\":1,\"event\":" + json_string(event_name);
    line += ",\"run_id\":" + json_string(run_id_.c_str());
    line += ",\"timestamp_unix_ns\":" + std::to_string(unix_ns);
    line += ",\"timestamp_mono_ns\":" + std::to_string(mono_ns);
    line += ",\"hostname\":" + json_string(hostname_.c_str());
    line += ",\"rank\":" + std::to_string(rank_);
    line += ",\"local_rank\":" + std::to_string(local_rank_);
    line += ",\"world_size\":" + std::to_string(world_size_);
    line += ",\"pid\":" + std::to_string(static_cast<long long>(::getpid()));
    return line;
  }

  void WriteProbeStatus(int fd, uint64_t dropped_events) const {
    const uint64_t unix_ns = clock_ns(CLOCK_REALTIME);
    const uint64_t mono_ns = clock_ns(CLOCK_MONOTONIC);
    std::string line = CommonJsonPrefix("probe_status", unix_ns, mono_ns);
    line += ",\"dropped_events\":" + std::to_string(dropped_events);
    line += "}\n";
    WriteAll(fd, line);
  }

  void WriteEvent(int fd, const Event& event) const {
    const char* event_name = "nccl_call";
    if (event.kind == EventKind::kProcessStart) event_name = "process_start";
    if (event.kind == EventKind::kCommInit) event_name = "comm_init";
    if (event.kind == EventKind::kCommDestroy) event_name = "comm_destroy";

    std::string line =
        CommonJsonPrefix(event_name, event.timestamp_unix_ns, event.timestamp_mono_ns);
    if (event.kind != EventKind::kProcessStart) {
      line += ",\"api\":" + json_string(event.api);
      line += ",\"phase\":" + json_string(event.phase);
      line += ",\"comm_uid_hash\":" + json_string(event.comm_uid_hash);
      line += ",\"comm_rank\":" + std::to_string(event.comm_rank);
      line += ",\"comm_nranks\":" + std::to_string(event.comm_nranks);
      line += ",\"comm_seq\":" + std::to_string(event.comm_seq);
      line += ",\"call_seq\":" + std::to_string(event.call_seq);
      line += ",\"count\":" + std::to_string(event.count);
      line += ",\"datatype\":" + std::to_string(event.datatype);
      line += ",\"op\":" + std::to_string(event.op);
      line += ",\"root\":" + std::to_string(event.root);
      line += ",\"peer\":" + std::to_string(event.peer);
      line += ",\"stream\":" + std::to_string(event.stream);
      line += ",\"group_id\":" + std::to_string(event.group_id);
      line += ",\"group_op_index\":" + std::to_string(event.group_op_index);
      line += ",\"p2p_op_index\":" + std::to_string(event.p2p_op_index);
      if (std::strcmp(event.api, "ncclSend") == 0 ||
          std::strcmp(event.api, "ncclRecv") == 0) {
        line += ",\"p2p_op_index_scope\":\"peer_direction\"";
      }
      line += ",\"result\":" + std::to_string(event.result);
      line += ",\"result_name\":" + json_string(event.result_name);
    }
    line += "}\n";
    WriteAll(fd, line);
  }

  bool enabled_ = false;
  int rank_ = -1;
  int local_rank_ = -1;
  int world_size_ = 0;
  std::string run_id_;
  std::string trace_dir_;
  std::string file_path_;
  std::string hostname_;

  CommRegistry comms_;
  std::atomic<bool> stopping_{false};
  std::thread worker_;
  std::atomic<uint64_t> next_call_sequence_{0};
  std::atomic<uint64_t> next_group_id_{0};
  std::atomic<uint64_t> dropped_events_{0};

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  Event queue_[kQueueCapacity]{};
  std::size_t queue_head_ = 0;
  std::size_t queue_tail_ = 0;
  std::size_t queue_size_ = 0;
};

thread_local uint64_t current_group_id = 0;
thread_local uint64_t current_group_op_index = 0;
thread_local unsigned int group_depth = 0;

Event MakeBaseEvent(EventKind kind, const char* api, const char* phase) {
  Event event;
  event.kind = kind;
  event.timestamp_unix_ns = clock_ns(CLOCK_REALTIME);
  event.timestamp_mono_ns = clock_ns(CLOCK_MONOTONIC);
  event.call_seq = Runtime::Instance().NextCallSequence();
  event.group_id = current_group_id;
  copy_text(event.api, sizeof(event.api), api);
  copy_text(event.phase, sizeof(event.phase), phase);
  return event;
}

struct CallContext {
  CommState* comm = nullptr;
  uint64_t comm_seq = 0;
  uint64_t call_seq = 0;
  uint64_t group_id = 0;
  uint64_t group_op_index = 0;
  uint64_t p2p_op_index = 0;
};

CallContext BeginCall(const char* api, ncclComm_t comm, std::size_t count,
                      ncclDataType_t datatype, int op, int root, int peer,
                      cudaStream_t stream, bool is_p2p = false) noexcept {
  try {
    Runtime& runtime = Runtime::Instance();
    Event event = MakeBaseEvent(EventKind::kNcclCall, api, "enter");
    CommState* state = runtime.comms().Find(comm);
    if (state != nullptr) {
      event.comm_seq = is_p2p ? state->next_p2p_sequence.fetch_add(1)
                              : state->next_collective_sequence.fetch_add(1);
      event.comm_rank = state->rank;
      event.comm_nranks = state->nranks;
      copy_text(event.comm_uid_hash, sizeof(event.comm_uid_hash), state->uid_hash);
    }
    if (is_p2p) {
      event.p2p_op_index = event.comm_seq;
      if (state != nullptr && peer >= 0 && peer < state->nranks) {
        auto& peer_sequence = std::strcmp(api, "ncclSend") == 0
                                  ? state->next_send_sequences[peer]
                                  : state->next_recv_sequences[peer];
        event.p2p_op_index = peer_sequence.fetch_add(1);
      }
      if (current_group_id != 0) event.group_op_index = current_group_op_index++;
    }
    event.count = static_cast<uint64_t>(count);
    event.datatype = static_cast<int>(datatype);
    event.op = op;
    event.root = root;
    event.peer = peer;
    event.stream = reinterpret_cast<uintptr_t>(stream);
    runtime.Enqueue(event);
    return CallContext{state, event.comm_seq, event.call_seq, event.group_id,
                       event.group_op_index, event.p2p_op_index};
  } catch (...) {
    return {};
  }
}

void EndCall(const char* api, const CallContext& context, ncclResult_t result,
             std::size_t count, ncclDataType_t datatype, int op, int root,
             int peer, cudaStream_t stream) noexcept {
  try {
    Runtime& runtime = Runtime::Instance();
    Event event = MakeBaseEvent(EventKind::kNcclCall, api, "exit");
    event.call_seq = context.call_seq;
    event.comm_seq = context.comm_seq;
    event.group_id = context.group_id;
    event.group_op_index = context.group_op_index;
    event.p2p_op_index = context.p2p_op_index;
    if (context.comm != nullptr) {
      event.comm_rank = context.comm->rank;
      event.comm_nranks = context.comm->nranks;
      copy_text(event.comm_uid_hash, sizeof(event.comm_uid_hash), context.comm->uid_hash);
    }
    event.count = static_cast<uint64_t>(count);
    event.datatype = static_cast<int>(datatype);
    event.op = op;
    event.root = root;
    event.peer = peer;
    event.stream = reinterpret_cast<uintptr_t>(stream);
    event.result = static_cast<int>(result);
    copy_text(event.result_name, sizeof(event.result_name), result_name(result));
    runtime.Enqueue(event);
  } catch (...) {
  }
}

void RecordCommInit(const char* api, ncclComm_t comm, int nranks,
                    const ncclUniqueId& unique_id, int comm_rank,
                    ncclResult_t result) noexcept {
  try {
    Runtime& runtime = Runtime::Instance();
    Event event = MakeBaseEvent(EventKind::kCommInit, api, "exit");
    event.comm_rank = comm_rank;
    event.comm_nranks = nranks;
    event.result = static_cast<int>(result);
    copy_text(event.result_name, sizeof(event.result_name), result_name(result));
    format_hash(hash_unique_id(unique_id), event.comm_uid_hash);
    if (result == ncclSuccess && comm != nullptr) {
      runtime.comms().Register(comm, comm_rank, nranks, unique_id);
    }
    runtime.Enqueue(event);
  } catch (...) {
  }
}

template <typename Function>
Function LoadNext(const char* name) {
  void* symbol = ::dlsym(RTLD_NEXT, name);
  if (symbol != nullptr) return reinterpret_cast<Function>(symbol);

  // PyTorch commonly opens its bundled NCCL with RTLD_LOCAL. Such symbols are not
  // visible through RTLD_NEXT even though libtorch can call our preloaded wrapper.
  // Resolve directly from the NCCL handle in that case.
  static void* nccl_handle = []() noexcept -> void* {
    void* handle = ::dlopen("libnccl.so.2", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) handle = ::dlopen("libnccl.so", RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr) handle = ::dlopen("libnccl.so.2", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) handle = ::dlopen("libnccl.so", RTLD_LAZY | RTLD_LOCAL);
    return handle;
  }();
  if (nccl_handle == nullptr) return nullptr;
  return reinterpret_cast<Function>(::dlsym(nccl_handle, name));
}

}  // namespace flagscale::tracing

__attribute__((constructor)) static void flagscale_nccl_probe_initialize() {
  // Initialize only NCCL event tracing. Rank liveness is provided by the independent
  // libflagscale_rank_heartbeat.so component when requested.
  try {
    (void)flagscale::tracing::Runtime::Instance();
  } catch (...) {
  }
}

#define FLAGSCALE_EXPORT extern "C" __attribute__((visibility("default")))

using flagscale::tracing::BeginCall;
using flagscale::tracing::CallContext;
using flagscale::tracing::EndCall;
using flagscale::tracing::Event;
using flagscale::tracing::EventKind;
using flagscale::tracing::LoadNext;
using flagscale::tracing::MakeBaseEvent;
using flagscale::tracing::RecordCommInit;
using flagscale::tracing::Runtime;

FLAGSCALE_EXPORT ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks,
                                               ncclUniqueId comm_id, int rank) {
  using Function = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int);
  static Function real_function = LoadNext<Function>("ncclCommInitRank");
  if (real_function == nullptr) return ncclInternalError;
  const ncclResult_t result = real_function(comm, nranks, comm_id, rank);
  RecordCommInit("ncclCommInitRank", comm != nullptr ? *comm : nullptr, nranks,
                 comm_id, rank, result);
  return result;
}

#if defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 21400
FLAGSCALE_EXPORT ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks,
                                                     ncclUniqueId comm_id, int rank,
                                                     ncclConfig_t* config) {
  using Function = ncclResult_t (*)(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*);
  static Function real_function = LoadNext<Function>("ncclCommInitRankConfig");
  if (real_function == nullptr) return ncclInternalError;
  const ncclResult_t result = real_function(comm, nranks, comm_id, rank, config);
  RecordCommInit("ncclCommInitRankConfig", comm != nullptr ? *comm : nullptr,
                 nranks, comm_id, rank, result);
  return result;
}
#endif

FLAGSCALE_EXPORT ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  using Function = ncclResult_t (*)(ncclComm_t);
  static Function real_function = LoadNext<Function>("ncclCommDestroy");
  if (real_function == nullptr) return ncclInternalError;
  const ncclResult_t result = real_function(comm);
  try {
    auto* state = Runtime::Instance().comms().Find(comm);
    Event event = MakeBaseEvent(EventKind::kCommDestroy, "ncclCommDestroy", "exit");
    event.result = static_cast<int>(result);
    flagscale::tracing::copy_text(
        event.result_name, sizeof(event.result_name),
        flagscale::tracing::result_name(result));
    if (state != nullptr) {
      event.comm_rank = state->rank;
      event.comm_nranks = state->nranks;
      std::snprintf(event.comm_uid_hash, sizeof(event.comm_uid_hash), "%s", state->uid_hash);
    }
    Runtime::Instance().Enqueue(event);
    if (result == ncclSuccess) Runtime::Instance().comms().Remove(comm);
  } catch (...) {
  }
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclAllReduce(const void* send_buffer, void* receive_buffer,
                                            size_t count, ncclDataType_t datatype,
                                            ncclRedOp_t op, ncclComm_t comm,
                                            cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t,
                                    ncclRedOp_t, ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclAllReduce");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context = BeginCall("ncclAllReduce", comm, count, datatype,
                                        static_cast<int>(op), -1, -1, stream);
  const ncclResult_t result =
      real_function(send_buffer, receive_buffer, count, datatype, op, comm, stream);
  EndCall("ncclAllReduce", context, result, count, datatype, static_cast<int>(op),
          -1, -1, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclAllGather(const void* send_buffer, void* receive_buffer,
                                            size_t send_count, ncclDataType_t datatype,
                                            ncclComm_t comm, cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t,
                                    ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclAllGather");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context =
      BeginCall("ncclAllGather", comm, send_count, datatype, -1, -1, -1, stream);
  const ncclResult_t result =
      real_function(send_buffer, receive_buffer, send_count, datatype, comm, stream);
  EndCall("ncclAllGather", context, result, send_count, datatype, -1, -1, -1,
          stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclReduceScatter(const void* send_buffer,
                                                void* receive_buffer, size_t receive_count,
                                                ncclDataType_t datatype, ncclRedOp_t op,
                                                ncclComm_t comm, cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t,
                                    ncclRedOp_t, ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclReduceScatter");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context = BeginCall("ncclReduceScatter", comm, receive_count,
                                        datatype, static_cast<int>(op), -1, -1, stream);
  const ncclResult_t result = real_function(send_buffer, receive_buffer, receive_count,
                                            datatype, op, comm, stream);
  EndCall("ncclReduceScatter", context, result, receive_count, datatype,
          static_cast<int>(op), -1, -1, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclBroadcast(const void* send_buffer, void* receive_buffer,
                                            size_t count, ncclDataType_t datatype, int root,
                                            ncclComm_t comm, cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t, int,
                                    ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclBroadcast");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context =
      BeginCall("ncclBroadcast", comm, count, datatype, -1, root, -1, stream);
  const ncclResult_t result =
      real_function(send_buffer, receive_buffer, count, datatype, root, comm, stream);
  EndCall("ncclBroadcast", context, result, count, datatype, -1, root, -1, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclReduce(const void* send_buffer, void* receive_buffer,
                                         size_t count, ncclDataType_t datatype,
                                         ncclRedOp_t op, int root, ncclComm_t comm,
                                         cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, void*, size_t, ncclDataType_t,
                                    ncclRedOp_t, int, ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclReduce");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context = BeginCall("ncclReduce", comm, count, datatype,
                                        static_cast<int>(op), root, -1, stream);
  const ncclResult_t result =
      real_function(send_buffer, receive_buffer, count, datatype, op, root, comm, stream);
  EndCall("ncclReduce", context, result, count, datatype, static_cast<int>(op), root,
          -1, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclSend(const void* send_buffer, size_t count,
                                       ncclDataType_t datatype, int peer, ncclComm_t comm,
                                       cudaStream_t stream) {
  using Function = ncclResult_t (*)(const void*, size_t, ncclDataType_t, int,
                                    ncclComm_t, cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclSend");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context =
      BeginCall("ncclSend", comm, count, datatype, -1, -1, peer, stream, true);
  const ncclResult_t result = real_function(send_buffer, count, datatype, peer, comm, stream);
  EndCall("ncclSend", context, result, count, datatype, -1, -1, peer, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclRecv(void* receive_buffer, size_t count,
                                       ncclDataType_t datatype, int peer, ncclComm_t comm,
                                       cudaStream_t stream) {
  using Function = ncclResult_t (*)(void*, size_t, ncclDataType_t, int, ncclComm_t,
                                    cudaStream_t);
  static Function real_function = LoadNext<Function>("ncclRecv");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context =
      BeginCall("ncclRecv", comm, count, datatype, -1, -1, peer, stream, true);
  const ncclResult_t result = real_function(receive_buffer, count, datatype, peer, comm, stream);
  EndCall("ncclRecv", context, result, count, datatype, -1, -1, peer, stream);
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclGroupStart() {
  using Function = ncclResult_t (*)();
  static Function real_function = LoadNext<Function>("ncclGroupStart");
  if (real_function == nullptr) return ncclInternalError;
  bool tracing_group_started = false;
  try {
    if (flagscale::tracing::group_depth == 0) {
      flagscale::tracing::current_group_id = Runtime::Instance().NextGroupId();
      flagscale::tracing::current_group_op_index = 0;
    }
    ++flagscale::tracing::group_depth;
    tracing_group_started = true;
  } catch (...) {
  }
  const CallContext context = BeginCall("ncclGroupStart", nullptr, 0, ncclChar, -1,
                                        -1, -1, nullptr);
  const ncclResult_t result = real_function();
  EndCall("ncclGroupStart", context, result, 0, ncclChar, -1, -1, -1, nullptr);
  if (result != ncclSuccess && tracing_group_started &&
      flagscale::tracing::group_depth > 0 && --flagscale::tracing::group_depth == 0) {
    flagscale::tracing::current_group_id = 0;
    flagscale::tracing::current_group_op_index = 0;
  }
  return result;
}

FLAGSCALE_EXPORT ncclResult_t ncclGroupEnd() {
  using Function = ncclResult_t (*)();
  static Function real_function = LoadNext<Function>("ncclGroupEnd");
  if (real_function == nullptr) return ncclInternalError;
  const CallContext context = BeginCall("ncclGroupEnd", nullptr, 0, ncclChar, -1,
                                        -1, -1, nullptr);
  const ncclResult_t result = real_function();
  EndCall("ncclGroupEnd", context, result, 0, ncclChar, -1, -1, -1, nullptr);
  if (flagscale::tracing::group_depth > 0 && --flagscale::tracing::group_depth == 0) {
    flagscale::tracing::current_group_id = 0;
    flagscale::tracing::current_group_op_index = 0;
  }
  return result;
}
