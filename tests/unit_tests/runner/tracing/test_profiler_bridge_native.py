# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real v5 bridge ABI without CUDA or live collectives."""

import json
import os
import shutil
import subprocess

import pytest

_HARNESS = r"""
#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
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
  CHECK(plugin->finalize(context)==ncclSuccess);
  std::printf("{\"api_calls\":300000,\"live\":%zu,\"peak\":%zu}\n",live,peak);
  return 0;
}
#endif
"""


def test_native_bridge_reclaims_state_and_keeps_delayed_identity(tmp_path):
    include = os.environ.get("FLAGSCALE_PROFILER_SDK_INCLUDE")
    library = os.environ.get("FLAGSCALE_PROFILER_BRIDGE_TEST_LIBRARY")
    compiler = shutil.which("g++")
    if not include or not library or not compiler:
        pytest.skip("requires official v5 SDK, built bridge, and Linux C++ compiler")
    source = tmp_path / "harness.cpp"
    source.write_text(_HARNESS)
    delegate = tmp_path / "delegate.so"
    harness = tmp_path / "harness"
    base = [compiler, "-std=c++17", "-I", include, str(source)]
    subprocess.run([*base, "-DFAKE_DELEGATE", "-fPIC", "-shared", "-o", str(delegate)], check=True)
    subprocess.run([*base, "-rdynamic", "-ldl", "-o", str(harness)], check=True)
    completed = subprocess.run(
        [str(harness), library, str(delegate), str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(completed.stdout)["live"] == 0
    rows = [
        json.loads(line)
        for path in tmp_path.glob("bridge*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all(row["call_seq"] == 1 and row["process_token"] == 999 for row in rows)
    assert rows[-1]["expected_channels"] == len(rows[-1]["channels"]) == 2
    assert all(channel["end"] == 200 for channel in rows[-1]["channels"])
