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

from flagscale.runner.tracing.monitor import JsonlTailer, _completion_exit_code


def test_jsonl_tailer_waits_for_a_complete_line(tmp_path):
    path = tmp_path / "rank_0_pid_10.jsonl"
    path.write_text('{"event":"heartbeat"', encoding="utf-8")
    tailer = JsonlTailer(tmp_path)

    assert tailer.poll() == []
    assert tailer.source_issues() == {(0, 10): ("partial_json_record",)}

    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(',"run_id":"r"}\n')
    assert tailer.poll() == [{"event": "heartbeat", "run_id": "r"}]
    assert tailer.source_issues() == {(0, 10): ()}


def test_jsonl_tailer_records_malformed_source_data(tmp_path):
    path = tmp_path / "rank_1_pid_11.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    tailer = JsonlTailer(tmp_path)

    assert tailer.poll() == []
    assert tailer.source_issues() == {(1, 11): ("malformed_json_record",)}


def test_jsonl_tailer_ignores_reports_and_reads_only_new_events(tmp_path):
    trace = tmp_path / "rank_0_pid_10.jsonl"
    trace.write_text(json.dumps({"event": "process_start"}) + "\n", encoding="utf-8")
    (tmp_path / "findings.jsonl").write_text("{}\n", encoding="utf-8")
    tailer = JsonlTailer(tmp_path)

    assert tailer.poll() == [{"event": "process_start"}]
    assert tailer.poll() == []


def test_completion_exit_code_accepts_success_and_failure(tmp_path):
    completion = tmp_path / "training.exit_code"
    assert _completion_exit_code(completion) is None
    completion.write_text("0\n", encoding="utf-8")
    assert _completion_exit_code(completion) == 0
    completion.write_text("17\n", encoding="utf-8")
    assert _completion_exit_code(completion) == 17
