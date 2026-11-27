// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// Built against the official NCCL Profiler v5 SDK. No vendor ABI is copied here.
#include <sys/types.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "profiler.h"

namespace {
using Getter = uint64_t (*)(const char*, uint64_t, uint64_t*);
struct Context {
  void* delegate = nullptr;
  uint64_t comm = 0;
  int rank = -1, ranks = 0;
};
struct Event {
  uint64_t type = 0, id = 0, call = 0, token = 0, count = 0;
  uint64_t revision = 0;
  Context* context = nullptr;
  void* delegate = nullptr;
  std::string api, datatype;
  int peer = -1, expected = 0;
  bool captured = false, stopped = false, invalid = false;
  struct Channel { uint64_t begin = 0, end = 0; };
  std::unordered_map<unsigned, Channel> channels;
  std::shared_ptr<Event> task;
  unsigned channel = 0;
};
using Clock = std::chrono::steady_clock;
constexpr auto kSnapshotInterval = std::chrono::milliseconds(50);
constexpr auto kFinishTimeout = std::chrono::milliseconds(250);
constexpr size_t kMaxPendingBytes = 8 * 1024 * 1024;
constexpr size_t kMaxPendingRows = 4096;
constexpr size_t kSnapshotsPerInterval = 128;
struct BridgeState {
  std::mutex mutex;
  std::unordered_map<uintptr_t, std::shared_ptr<Event>> events;
  std::unordered_set<uintptr_t> dirty;
  std::condition_variable changed;
  std::deque<std::string> pending;
  size_t pending_bytes = 0, pending_rows = 0, contexts = 0;
  uint64_t submitted = 0, published = 0;
  int fd = -1;
  const char* failure = nullptr;
};
// A publisher blocked in write() can outlive normal NCCL finalization.
BridgeState& bridge = *new BridgeState;
auto& mutex = bridge.mutex;
auto& events = bridge.events;
ncclProfiler_v5_t* delegate = nullptr;
Getter getter = nullptr;
uintptr_t serial = 0;
uint64_t process_token = 0;
size_t peak_events = 0;
static_assert(sizeof(uintptr_t) == sizeof(uint64_t), "Bridge requires 64-bit opaque handles");
constexpr uintptr_t kApiTag = uintptr_t{1} << 63;
constexpr uintptr_t kCapturedTag = uintptr_t{1} << 62;
constexpr uintptr_t kCallMask = kCapturedTag - 1;
bool broken = false;
char hostname[256] = {};
// Bound only live task/channel/delegate events. API identities are immutable
// integer tokens, never dereferenced, and do not allocate retained objects.
constexpr size_t kMaxEvents = 262144;

std::string quoted(const std::string& text) {
  std::string out = "\"";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c < 32) out += '?';
    else out += c;
  }
  return out + '"';
}
void invalidate(const char* reason = "bridge_invalid") {
  if (broken) return;
  broken = true;
  bridge.failure = reason;
  bridge.changed.notify_all();
}
void write_line(std::string line) {
  if (bridge.fd < 0 || broken) return;
  line += '\n';
  if (line.size() > kMaxPendingBytes - bridge.pending_bytes ||
      bridge.pending_rows == kMaxPendingRows) {
    invalidate("publisher_queue_full");
    return;
  }
  const size_t size = line.size();
  bridge.pending.push_back(std::move(line));
  bridge.pending_bytes += size;
  ++bridge.pending_rows;
  ++bridge.submitted;
  bridge.changed.notify_all();
}
std::shared_ptr<Event> lookup(void* handle) {
  if (reinterpret_cast<uintptr_t>(handle) & kApiTag) return nullptr;
  auto it = events.find(reinterpret_cast<uintptr_t>(handle));
  return it == events.end() ? nullptr : it->second;
}
void* original(void* handle) {
  auto item = lookup(handle);
  return item ? item->delegate : nullptr;
}
void emit(const std::shared_ptr<Event>& e, const char* observation = nullptr) {
  if (!e || !e->call || broken) return;
  std::ostringstream out;
  out << "{\"record_type\":\"flagscale_profiler_v5_v1\",\"hostname\":" << quoted(hostname)
      << ",\"pid\":" << getpid() << ",\"process_token\":" << e->token
      << ",\"call_seq\":" << e->call << ",\"task_id\":" << e->id
      << ",\"revision\":" << ++e->revision << ",\"comm_id\":\"" << std::hex
      << e->context->comm << std::dec << "\",\"rank\":" << e->context->rank
      << ",\"nranks\":" << e->context->ranks << ",\"api\":" << quoted(e->api)
      << ",\"count\":" << e->count << ",\"datatype\":" << quoted(e->datatype)
      << ",\"peer\":" << e->peer << ",\"expected_channels\":" << e->expected
      << ",\"graph_captured\":" << (e->captured ? "true" : "false")
      << ",\"invalid\":" << (e->invalid ? "true" : "false") << ",\"channels\":[";
  bool first = true;
  for (const auto& pair : e->channels) {
    if (!first) out << ',';
    first = false;
    out << "{\"channel_id\":" << pair.first << ",\"begin\":" << pair.second.begin
        << ",\"end\":" << pair.second.end << '}';
  }
  out << "]";
  if (observation) out << ",\"observation\":" << quoted(observation);
  out << "}";
  write_line(out.str());
}
bool finished(const std::shared_ptr<Event>& e) {
  if (e->expected <= 0 || e->channels.size() != size_t(e->expected)) return false;
  for (const auto& p : e->channels)
    if (!p.second.begin || p.second.end < p.second.begin) return false;
  return true;
}

bool write_all(int fd, const std::string& line) {
  const char* cursor = line.data();
  size_t remaining = line.size();
  while (remaining) {
    const ssize_t written = ::write(fd, cursor, remaining);
    if (written < 0 && errno == EINTR) continue;
    if (written <= 0) return false;
    cursor += written;
    remaining -= static_cast<size_t>(written);
  }
  return true;
}

void publish() noexcept {
  try {
    std::deque<std::string> batch;
    std::unique_lock<std::mutex> lock(mutex);
    auto next_snapshot = Clock::now() + kSnapshotInterval;
    while (!broken) {
      if (Clock::now() >= next_snapshot) {
        for (size_t n = 0; n < kSnapshotsPerInterval && !bridge.dirty.empty() && !broken; ++n) {
          auto item = bridge.dirty.begin();
          auto task = events.find(*item);
          bridge.dirty.erase(item);
          if (task != events.end()) emit(task->second, "periodic_partial_snapshot");
        }
        next_snapshot = Clock::now() + kSnapshotInterval;
      }
      if (broken) break;
      if (bridge.pending.empty()) {
        if (bridge.dirty.empty()) bridge.changed.wait(lock);
        else bridge.changed.wait_until(lock, next_snapshot);
        continue;
      }
      batch.swap(bridge.pending);
      const int fd = bridge.fd;
      lock.unlock();
      size_t published_bytes = 0, published_rows = 0;
      for (const auto& line : batch) {
        if (!write_all(fd, line)) break;
        published_bytes += line.size();
        ++published_rows;
      }
      lock.lock();
      bridge.pending_bytes -= published_bytes;
      bridge.pending_rows -= published_rows;
      bridge.published += published_rows;
      if (published_rows != batch.size()) invalidate("publisher_write_failed");
      batch.clear();
      bridge.changed.notify_all();
    }
    const char* failure = bridge.failure;
    bridge.pending.clear();
    lock.unlock();
    if (failure) {
      std::fprintf(stderr, "FlagScale bridge publication stopped: %s\n", failure);
      write_all(bridge.fd, "{\"record_type\":\"flagscale_bridge_error_v1\",\"hostname\":" +
                quoted(hostname) + ",\"pid\":" + std::to_string(getpid()) +
                ",\"reason\":" + quoted(failure) + "}\n");
    }
  } catch (...) {
    std::lock_guard<std::mutex> lock(mutex);
    invalidate("publisher_exception");
    std::fputs("FlagScale bridge publisher failed\n", stderr);
  }
}

ncclResult_t init(void** context, uint64_t comm, int* mask, const char* name,
                  int nodes, int ranks, int rank, ncclDebugLogger_t logger) {
  std::lock_guard<std::mutex> guard(mutex);
  try {
    if (!delegate) {
      const char* path = std::getenv("FLAGSCALE_INSPECTOR_DELEGATE");
      void* lib = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
      delegate = lib ? static_cast<ncclProfiler_v5_t*>(dlsym(lib, "ncclProfiler_v5")) : nullptr;
      getter = reinterpret_cast<Getter>(dlsym(RTLD_DEFAULT, "flagscale_trace_profiler_call_v1"));
      if (!delegate || delegate->init == init || !getter) {
        delegate = nullptr;
        *mask = 0; *context = nullptr; return ncclSuccess;
      }
      gethostname(hostname, sizeof(hostname) - 1);
      const char* dir = std::getenv("NCCL_INSPECTOR_DUMP_DIR");
      if (!dir) { *mask = 0; *context = nullptr; return ncclSuccess; }
      std::string file = std::string(dir) + "/bridge-" + hostname + "-" + std::to_string(getpid()) + ".jsonl";
      bridge.fd = ::open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
      if (bridge.fd < 0) { delegate = nullptr; *mask = 0; *context = nullptr; return ncclSuccess; }
      Dl_info library{};
      if (!dladdr(reinterpret_cast<void*>(publish), &library) ||
          !dlopen(library.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE)) {
        ::close(bridge.fd);
        bridge.fd = -1;
        delegate = nullptr;
        *mask = 0; *context = nullptr; return ncclSuccess;
      }
      try {
        std::thread(publish).detach();
      } catch (...) {
        ::close(bridge.fd);
        bridge.fd = -1;
        delegate = nullptr;
        throw;
      }
    }
    auto c = std::make_unique<Context>();
    c->comm = comm; c->rank = rank; c->ranks = ranks;
    auto result = delegate->init(&c->delegate, comm, mask, name, nodes, ranks, rank, logger);
    if (result != ncclSuccess) return result;
    *mask |= ncclProfileCollApi | ncclProfileP2pApi | ncclProfileColl |
             ncclProfileP2p | ncclProfileKernelCh;
    *context = c.release();
    ++bridge.contexts;
    return ncclSuccess;
  } catch (...) { invalidate(); *mask = 0; *context = nullptr; return ncclSuccess; }
}
ncclResult_t start(void* context, void** handle, ncclProfilerEventDescr_v5_t* descr) {
  std::lock_guard<std::mutex> guard(mutex);
  *handle = nullptr;
  if (!context || broken) return ncclSuccess;
  try {
    if (descr->type == ncclProfileCollApi || descr->type == ncclProfileP2pApi) {
      const bool coll = descr->type == ncclProfileCollApi;
      const char* api = coll ? descr->collApi.func : descr->p2pApi.func;
      const uint64_t count = coll ? descr->collApi.count : descr->p2pApi.count;
      const bool captured = coll ? descr->collApi.graphCaptured : descr->p2pApi.graphCaptured;
      uint64_t token = 0;
      const uint64_t call = getter(api, count, &token);
      if (!call || !token) return ncclSuccess;
      if (call > kCallMask || (process_token && process_token != token)) {
        invalidate(); return ncclSuccess;
      }
      process_token = token;
      // Profiler ABI handles are opaque tokens. Never forward these to the
      // delegate or reuse them; an async task can resolve its API even after
      // the API/group stop callback, without retaining per-API memory.
      *handle = reinterpret_cast<void*>(kApiTag | (captured ? kCapturedTag : 0) | call);
      return ncclSuccess;
    }
    if (events.size() >= kMaxEvents || serial >= kCallMask) {
      invalidate(); return ncclSuccess;
    }
    auto e = std::make_shared<Event>();
    e->context = static_cast<Context*>(context); e->type = descr->type; e->id = ++serial;
    bool task = e->type == ncclProfileColl || e->type == ncclProfileP2p;
    {
      if (task) {
        const auto parent = reinterpret_cast<uintptr_t>(descr->parentObj);
        if (parent & kApiTag) {
          e->call = parent & kCallMask;
          e->token = process_token;
          e->captured = (parent & kCapturedTag) != 0;
        }
        const bool coll = e->type == ncclProfileColl;
        e->api = coll ? descr->coll.func : descr->p2p.func;
        e->count = coll ? descr->coll.count : descr->p2p.count;
        e->datatype = coll ? descr->coll.datatype : descr->p2p.datatype;
        e->expected = coll ? descr->coll.nChannels : descr->p2p.nChannels;
        e->peer = coll ? -1 : descr->p2p.peer;
        // The reader checks these device parameters against the original
        // Probe API, including NCCL's byte encoding for copy collectives.
      }
      if (e->type == ncclProfileKernelCh) {
        e->task = lookup(descr->parentObj); e->channel = descr->kernelCh.channelId;
        if (e->task) {
          auto& channels = e->task->channels;
          if (channels.count(e->channel)) e->task->invalid = true;
          channels[e->channel].begin = descr->kernelCh.pTimer;
          if (e->task->call && bridge.dirty.insert(e->task->id).second) bridge.changed.notify_all();
        }
      }
      auto forwarded = *descr;
      forwarded.parentObj = original(descr->parentObj);
      if (e->type == ncclProfileColl) forwarded.coll.parentGroup = original(descr->coll.parentGroup);
      if (e->type == ncclProfileP2p) forwarded.p2p.parentGroup = original(descr->p2p.parentGroup);
      // The delegate may not implement P2P. Never pass a null P2P parent to
      // its channel callback; the bridge still collects that channel itself.
      if (e->type != ncclProfileKernelCh || forwarded.parentObj) {
        auto result = delegate->startEvent(e->context->delegate, &e->delegate, &forwarded);
        if (result != ncclSuccess) return result;
      }
    }
    *handle = reinterpret_cast<void*>(e->id);
    events.emplace(e->id, e);
    if (events.size() > peak_events) peak_events = events.size();
    if (task) emit(e);
    return ncclSuccess;
  } catch (...) { invalidate(); return ncclSuccess; }
}
ncclResult_t stop(void* handle) {
  std::lock_guard<std::mutex> guard(mutex);
  auto e = lookup(handle);
  if (!e) return ncclSuccess;
  auto result = e->delegate ? delegate->stopEvent(e->delegate) : ncclSuccess;
  e->stopped = true;
  if (e->type != ncclProfileColl && e->type != ncclProfileP2p) events.erase(e->id);
  else if (finished(e)) events.erase(e->id);
  return result;
}
ncclResult_t state(void* handle, ncclProfilerEventState_v5_t state,
                   ncclProfilerEventStateArgs_v5_t* args) {
  std::lock_guard<std::mutex> guard(mutex);
  auto e = lookup(handle);
  if (!e) return ncclSuccess;
  auto result = e->delegate ? delegate->recordEventState(e->delegate, state, args) : ncclSuccess;
  try {
    if (e->task && args && state == ncclProfilerKernelChStop) {
      e->task->channels[e->channel].end = args->kernelCh.pTimer;
      const bool complete = finished(e->task);
      if (complete || e->task->invalid) {
        bridge.dirty.erase(e->task->id);
        emit(e->task);
      } else if (e->task->call && !broken && bridge.dirty.insert(e->task->id).second) {
        bridge.changed.notify_all();
      }
      if (e->task->stopped && complete) events.erase(e->task->id);
    }
  } catch (...) { invalidate(); }
  return result;
}
ncclResult_t finalize(void* context) {
  if (!context) return ncclSuccess;
  auto c = static_cast<Context*>(context);
  // Delegate finalization may wait for callbacks; do not hold our mutex here.
  auto result = delegate->finalize(c->delegate);
  std::unique_lock<std::mutex> guard(mutex);
  try {
    size_t incomplete = 0;
    for (const auto& item : events) {
      const auto& e = item.second;
      if (e->context == c && (e->type == ncclProfileColl || e->type == ncclProfileP2p) &&
          !finished(e)) {
        emit(e, "context_finalize");
        ++incomplete;
      }
    }
    write_line("{\"record_type\":\"flagscale_bridge_lifecycle_v1\",\"event\":\"context_finalize\","
               "\"hostname\":" + quoted(hostname) + ",\"pid\":" + std::to_string(getpid()) +
               ",\"incomplete_tasks\":" + std::to_string(incomplete) + "}");
  } catch (...) {
    invalidate("context_finalize_failed");
  }
  for (auto i = events.begin(); i != events.end();) {
    if (i->second->context == c) {
      bridge.dirty.erase(i->first);
      i = events.erase(i);
    } else {
      ++i;
    }
  }
  delete c;
  if (--bridge.contexts == 0 && !broken) {
    const auto target = bridge.submitted;
    if (!bridge.changed.wait_for(guard, kFinishTimeout, [target] {
          return broken || bridge.published >= target;
        })) {
      const auto pending = target - bridge.published;
      invalidate("publisher_flush_timeout");
      guard.unlock();
      std::fprintf(stderr, "FlagScale bridge final publication timed out: %llu rows unconfirmed\n",
                   static_cast<unsigned long long>(pending));
    }
  }
  return result;
}
}  // namespace
extern "C" {
// Read-only diagnostics for bounded-state acceptance and operational checks.
__attribute__((visibility("default"))) void flagscale_profiler_bridge_stats_v1(
    uint64_t* created, size_t* live, size_t* peak) {
  std::lock_guard<std::mutex> guard(mutex);
  if (created) *created = serial;
  if (live) *live = events.size();
  if (peak) *peak = peak_events;
}
__attribute__((visibility("default"))) ncclProfiler_v5_t ncclProfiler_v5 = {
    "FlagScale API-to-task bridge (v5)", init, start, stop, state, finalize};
}
