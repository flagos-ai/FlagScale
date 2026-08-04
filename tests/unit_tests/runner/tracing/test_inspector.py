# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from flagscale.runner.tracing.inspector import InspectorIndex, InspectorReader


def _payload(start_us=1_000_100):
    return {
        "header": {
            "id": "0x1234",
            "rank": 0,
            "n_ranks": 2,
            "nnodes": 1,
        },
        "metadata": {
            "inspector_output_format_version": "v4.0",
            "dump_timestamp_us": start_us + 100,
            "hostname": "host-a",
            "pid": 10,
        },
        "coll_perf": {
            "coll": "AllReduce",
            "coll_sn": 7,
            "coll_msg_size_bytes": 2048,
            "coll_exec_time_us": 100,
            "coll_timing_source": "kernel_gpu",
            "event_trace_ts": {
                "coll_start_ts": start_us,
                "coll_stop_ts": start_us + 100,
                "kernel_events": [
                    {
                        "channel_id": 0,
                        "kernel_start_ts": start_us + 5,
                        "kernel_stop_ts": start_us + 95,
                    }
                ],
            },
        },
    }


def _event():
    return {
        "hostname": "host-a",
        "pid": 10,
        "comm_rank": 0,
        "api": "ncclAllReduce",
        "count": 1024,
        "datatype": 6,
        "timestamp_unix_ns": 1_000_000_000,
    }


def test_unique_completed_record_is_positive_kernel_evidence(tmp_path):
    output = tmp_path / "inspector.json"
    output.write_text(json.dumps(_payload()) + "\n", encoding="utf-8")

    evidence = InspectorReader(tmp_path, enabled=True).poll().for_call(_event())

    assert evidence["inspector_correlation"] == "unique"
    assert evidence["inspector_kernel_status"] == "completed"
    assert evidence["kernel_execution_status"] == "completed"
    assert evidence["inspector_completion"]["timing_source"] == "kernel_gpu"
    assert len(evidence["inspector_completion"]["kernel_events"]) == 1


def test_absent_or_ambiguous_inspector_record_stays_unknown(tmp_path):
    reader = InspectorReader(tmp_path, enabled=True)
    assert reader.poll().for_call(_event())["inspector_kernel_status"] == "unknown"

    output = tmp_path / "inspector.jsonl"
    output.write_text(
        json.dumps(_payload()) + "\n" + json.dumps(_payload(start_us=1_000_200)) + "\n",
        encoding="utf-8",
    )
    evidence = reader.poll().for_call(_event())

    assert evidence["inspector_correlation"] == "ambiguous"
    assert evidence["inspector_candidate_count"] == 2
    assert evidence["inspector_kernel_status"] == "unknown"
    assert evidence["kernel_execution_status"] == "unknown"


def test_disabled_inspector_is_reported_as_not_collected():
    evidence = InspectorIndex(False).for_call(_event())

    assert evidence["inspector_correlation"] == "not_collected"
    assert evidence["inspector_kernel_status"] == "not_collected"
    assert evidence["kernel_execution_status"] == "unknown"
