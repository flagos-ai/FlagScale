# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real v5 bridge ABI without CUDA or live collectives."""

import json
import os
import shutil
import signal
import subprocess

import pytest

_HARNESS = r"""
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <csignal>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include "profiler.h"
#ifdef FAKE_DELEGATE
static ncclResult_t init(void** c,uint64_t,int* mask,const char*,int,int,int,ncclDebugLogger_t) {
  *c=nullptr;*mask=0;return ncclSuccess;
}
static ncclResult_t start(void*,void** h,ncclProfilerEventDescr_v5_t*) {*h=nullptr;return ncclSuccess;}
static ncclResult_t stop(void*) {return ncclSuccess;}
static ncclResult_t state(void*,ncclProfilerEventState_v5_t,ncclProfilerEventStateArgs_v5_t*) {return ncclSuccess;}
extern "C" {ncclProfiler_v5_t ncclProfiler_v5={"test delegate",init,start,stop,state,stop};}
#else
static uint64_t current_call=0;
extern "C" uint64_t flagscale_trace_profiler_call_v1(const char*,uint64_t,uint64_t* token) {
  *token=999;return current_call;
}
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char** argv) {
  CHECK(argc==4);
  setenv("FLAGSCALE_INSPECTOR_DELEGATE",argv[2],1);
  setenv("NCCL_INSPECTOR_DUMP_DIR",argv[3],1);
  void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);CHECK(lib);
  auto* plugin=(ncclProfiler_v5_t*)dlsym(lib,"ncclProfiler_v5");CHECK(plugin);
  auto stats=(void(*)(uint64_t*,size_t*,size_t*))dlsym(lib,"flagscale_profiler_bridge_stats_v1");CHECK(stats);
  void* context=nullptr;int mask=0;
  CHECK(plugin->init(&context,0x1234,&mask,"test",1,2,0,nullptr)==ncclSuccess);
  CHECK(context && (mask&ncclProfileCollApi));
  void* first=nullptr;
  for(uint64_t i=1;i<=300000;++i) {
    current_call=i;ncclProfilerEventDescr_v5_t d={};
    d.type=ncclProfileCollApi;d.collApi.func="AllReduce";d.collApi.count=16;d.collApi.datatype="ncclFloat32";
    void* h=nullptr;CHECK(plugin->startEvent(context,&h,&d)==ncclSuccess && h);
    if(i==1)first=h;
    CHECK(plugin->stopEvent(h)==ncclSuccess);
  }
  uint64_t created=0;size_t live=0,peak=0;stats(&created,&live,&peak);
  CHECK(live==0 && created==0);
  // A delayed device task can still resolve API #1 after 300000 API stops.
  ncclProfilerEventDescr_v5_t task={};task.type=ncclProfileColl;task.parentObj=first;
  task.coll.func="AllReduce";task.coll.count=16;task.coll.datatype="ncclFloat32";task.coll.nChannels=2;
  void* htask=nullptr;CHECK(plugin->startEvent(context,&htask,&task)==ncclSuccess && htask);
  CHECK(plugin->stopEvent(htask)==ncclSuccess); // host task may stop before GPU evidence
  for(int ch=0;ch<2;++ch) {
    ncclProfilerEventDescr_v5_t channel={};channel.type=ncclProfileKernelCh;
    channel.parentObj=htask;channel.kernelCh.channelId=ch;channel.kernelCh.pTimer=100;
    void* h=nullptr;CHECK(plugin->startEvent(context,&h,&channel)==ncclSuccess && h);
    ncclProfilerEventStateArgs_v5_t args={};args.kernelCh.pTimer=200;
    CHECK(plugin->recordEventState(h,ncclProfilerKernelChStop,&args)==ncclSuccess);
    CHECK(plugin->stopEvent(h)==ncclSuccess);
  }
  stats(&created,&live,&peak);CHECK(live==0 && peak<=3);
  // Missing Probe identity produces no candidate and consumes no state.
  current_call=0;ncclProfilerEventDescr_v5_t no_identity={};no_identity.type=ncclProfileCollApi;
  no_identity.collApi.func="AllReduce";no_identity.collApi.count=16;
  void* missing=nullptr;CHECK(plugin->startEvent(context,&missing,&no_identity)==ncclSuccess && !missing);
  // Preserve an unfinished task with only one observed channel at finalization.
  void* unfinished=nullptr;
  CHECK(plugin->startEvent(context,&unfinished,&task)==ncclSuccess && unfinished);
  ncclProfilerEventDescr_v5_t partial={};partial.type=ncclProfileKernelCh;
  partial.parentObj=unfinished;partial.kernelCh.channelId=0;partial.kernelCh.pTimer=300;
  void* partial_channel=nullptr;
  CHECK(plugin->startEvent(context,&partial_channel,&partial)==ncclSuccess);
  if (std::getenv("BRIDGE_TEST_TERMINATE_PARTIAL")) {
    char hostname[256] = {};
    CHECK(gethostname(hostname, sizeof(hostname)-1)==0);
    const std::string path = std::string(argv[3]) + "/bridge-" + hostname + "-" +
                             std::to_string(getpid()) + ".jsonl";
    bool published = false;
    for (int i=0; i<200 && !published; ++i) {
      std::ifstream file(path);
      std::string line;
      while (std::getline(file, line)) {
        if (line.find("periodic_partial_snapshot") != std::string::npos) published = true;
      }
      if (!published) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    CHECK(published);
    std::raise(SIGTERM);
  }
  CHECK(plugin->finalize(context)==ncclSuccess);
  stats(&created,&live,&peak);CHECK(live==0);
  std::printf("{\"api_calls\":300000,\"live\":%zu,\"peak\":%zu}\n",live,peak);
  return 0;
}
#endif
"""


@pytest.fixture
def native_bridge_harness(tmp_path):
    include = os.environ.get("FLAGSCALE_PROFILER_SDK_INCLUDE")
    library = os.environ.get("FLAGSCALE_PROFILER_BRIDGE_TEST_LIBRARY")
    compiler = shutil.which("g++")
    if not include or not library or not compiler:
        pytest.skip("requires official v5 SDK, built bridge, and Linux C++ compiler")
    source = tmp_path / "harness.cpp"
    source.write_text(_HARNESS)
    delegate = tmp_path / "delegate.so"
    harness = tmp_path / "harness"
    base = [compiler, "-std=c++17", "-pthread", "-I", include, str(source)]
    subprocess.run([*base, "-DFAKE_DELEGATE", "-fPIC", "-shared", "-o", str(delegate)], check=True)
    subprocess.run([*base, "-rdynamic", "-ldl", "-o", str(harness)], check=True)
    return [str(harness), library, str(delegate), str(tmp_path)]


def _read_bridge_rows(tmp_path):
    return [
        json.loads(line)
        for path in tmp_path.glob("bridge*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_native_bridge_reclaims_state_and_keeps_delayed_identity(tmp_path, native_bridge_harness):
    completed = subprocess.run(
        native_bridge_harness,
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["live"] == 0
    rows = _read_bridge_rows(tmp_path)
    tasks = [row for row in rows if row["record_type"] == "flagscale_profiler_v5_v1"]
    assert all(row["call_seq"] == 1 and row["process_token"] == 999 for row in tasks)
    (completed,) = (
        row
        for row in tasks
        if len(row["channels"]) == 2 and all(channel["end"] == 200 for channel in row["channels"])
    )
    assert completed["expected_channels"] == 2
    assert all(channel["end"] == 200 for channel in completed["channels"])
    assert tasks[-1]["observation"] == "context_finalize"
    assert tasks[-1]["channels"] == [{"channel_id": 0, "begin": 300, "end": 0}]
    assert rows[-1]["event"] == "context_finalize"
    assert rows[-1]["incomplete_tasks"] == 1


def test_native_bridge_publishes_partial_callbacks_before_abrupt_stop(tmp_path, native_bridge_harness):
    result = subprocess.run(
        native_bridge_harness,
        env=dict(os.environ, BRIDGE_TEST_TERMINATE_PARTIAL="1"),
        timeout=30,
    )
    assert result.returncode == -signal.SIGTERM
    rows = _read_bridge_rows(tmp_path)
    tasks = [row for row in rows if row["record_type"] == "flagscale_profiler_v5_v1"]
    (partial,) = (
        row
        for row in tasks
        if row.get("observation") == "periodic_partial_snapshot" and row["channels"][0]["begin"] == 300
    )
    assert partial["channels"] == [{"channel_id": 0, "begin": 300, "end": 0}]
    assert any(len(row["channels"]) == 2 for row in tasks)
    assert not any(row.get("event") == "context_finalize" for row in rows)


def test_native_bridge_finalization_does_not_wait_for_blocked_file(tmp_path, native_bridge_harness):
    source = tmp_path / "block_write.c"
    source.write_text(
        r"""
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

ssize_t write(int fd, const void* buf, size_t count) {
    char target[4096], name[64];
    snprintf(name, sizeof(name), "/proc/self/fd/%d", fd);
    ssize_t n = readlink(name, target, sizeof(target) - 1);
    if (n > 0) {
        target[n] = 0;
        if (strstr(target, "/bridge-")) sleep(4);
    }
    return syscall(SYS_write, fd, buf, count);
}
"""
    )
    shim = tmp_path / "block_write.so"
    subprocess.run(["g++", "-x", "c", "-fPIC", "-shared", str(source), "-o", str(shim)], check=True)
    result = subprocess.run(
        native_bridge_harness,
        env=dict(os.environ, LD_PRELOAD=str(shim)),
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert "final publication timed out" in result.stderr
