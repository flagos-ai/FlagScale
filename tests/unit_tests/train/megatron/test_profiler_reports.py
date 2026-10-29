import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).parents[4] / "flagscale/train/megatron/training/profiler_reports.py"
SPEC = importlib.util.spec_from_file_location("profiler_reports", MODULE_PATH)
profiler_reports = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiler_reports)


def _event(shapes=None, dtypes=None, event_id=None, kernel_name="test_kernel"):
    kernel = SimpleNamespace(name=kernel_name, duration=2.5)
    return SimpleNamespace(
        id=event_id,
        is_user_annotation=False,
        key="aten::mm",
        structured_input_shapes=shapes,
        input_shapes=None,
        input_dtypes=dtypes,
        kernels=[kernel],
    )


def test_mapping_status_reflects_available_input_metadata():
    events = [
        _event([[2, 4], [4, 8]], ["float", "float"]),
        _event([[2, 4], [4, 8]], None),
        _event(None, ["float", "float"]),
        _event(None, None),
    ]

    details, _summary = profiler_reports.build_kernel_report_rows(events)

    assert {row["mapping_status"] for row in details} == {
        "operator_shape_matched",
        "operator_matched_dtype_missing",
        "operator_matched_shape_missing",
        "operator_matched_metadata_missing",
    }
    missing = next(
        row for row in details if row["mapping_status"] == "operator_matched_metadata_missing"
    )
    assert missing["input_shapes"] == "null"
    assert missing["input_dtypes"] == "null"


def test_external_id_metadata_precedes_sequence_order(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": 1,
                        "dur": 1,
                        "args": {
                            "External id": 11,
                            "Input Dims": [[2, 2]],
                            "Input type": ["float"],
                        },
                    },
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": 3,
                        "dur": 1,
                        "args": {
                            "External id": 22,
                            "Input Dims": [[2, 2]],
                            "Input type": ["half"],
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    events = [
        _event([[2, 2]], event_id=22, kernel_name="half_kernel"),
        _event([[2, 2]], event_id=11, kernel_name="float_kernel"),
    ]

    details, _summary = profiler_reports.build_kernel_report_rows(events, trace_path)

    by_kernel = {row["kernel_name"]: row for row in details}
    assert by_kernel["half_kernel"]["input_dtypes"] == '["half"]'
    assert by_kernel["float_kernel"]["input_dtypes"] == '["float"]'
    assert {row["mapping_source"] for row in details} == {"external_id"}


def test_external_id_ambiguity_retains_candidates(tmp_path):
    trace_path = tmp_path / "ambiguous.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": index * 2,
                        "dur": 1,
                        "args": {
                            "External id": 7,
                            "Input Dims": [[2, 2]],
                            "Input type": [dtype],
                        },
                    }
                    for index, dtype in enumerate(("float", "half"), start=1)
                ]
            }
        ),
        encoding="utf-8",
    )

    details, _summary = profiler_reports.build_kernel_report_rows(
        [_event([[2, 2]], event_id=7)], trace_path
    )

    assert details[0]["mapping_status"] == "shape_ambiguous"
    assert details[0]["mapping_source"] == "external_id_ambiguous"
    assert len(json.loads(details[0]["candidate_operators"])) == 2


def test_trace_parser_streams_across_chunk_boundaries(tmp_path, monkeypatch):
    trace_path = tmp_path / "chunked.json"
    expected = [{"cat": "cpu_op", "name": f"aten::op_{index}", "args": {}} for index in range(5)]
    trace_path.write_text(json.dumps({"traceEvents": expected}), encoding="utf-8")
    monkeypatch.setattr(profiler_reports, "TRACE_READ_CHUNK_SIZE", 17)

    assert list(profiler_reports._iter_trace_events(trace_path)) == expected


def test_kernel_callable_normalization_is_lossless_for_templates():
    kernel = "void custom::kernel_rank_3<half, nested::Type<int, 4>>(half*, int)"

    assert profiler_reports._kernel_callable_identity_name(kernel) == (
        "custom::kernel<half, nested::Type<int, 4>>"
    )


def test_unattributed_kernels_have_distinct_semantic_identities():
    rows = profiler_reports.build_operator_rows(
        [
            {
                "custom_operator": "null",
                "execution_operator": "null",
                "kernel_name": kernel,
            }
            for kernel in ("void first(int)", "void second(int)")
        ]
    )

    assert {row["operator_id"] for row in rows} == {1, 2}
    assert {row["kernel_callable"] for row in rows} == {"first", "second"}


def test_export_reports_accounts_for_filtered_kernels(tmp_path):
    compute_event = _event([[2, 4], [4, 8]], ["float", "float"])
    communication_event = _event([[2]], ["float"])
    communication_event.kernels[0].name = "ncclKernel_AllReduce"
    non_compute_event = _event([[2]], ["float"])
    non_compute_event.kernels[0].name = "cudaMemcpyAsync"
    fused_event = _event([[2]], ["float"])
    fused_event.kernels[0].name = "all_gather_gemm_kernel"

    paths = profiler_reports.export_kernel_reports(
        [compute_event, communication_event, non_compute_event, fused_event],
        tmp_path,
        rank=3,
    )

    assert [path.name for path in paths] == [
        "rank-3_kernel_shape_dtype.csv",
        "rank-3_kernel_time.csv",
        "rank-3_operator_list.csv",
        "rank-3_summary.json",
    ]
    diagnostics = json.loads(paths[3].read_text(encoding="utf-8"))
    assert diagnostics["scope"]["rank"] == 3
    assert list(profiler_reports._read_csv(paths[0])[0])[:9] == [
        "operator_name",
        "kernel_name",
        "variant_index",
        "mapping_status",
        "input_shapes",
        "input_dtypes",
        "candidate_operators",
        "kernel_event_count",
        "kernel_time_us",
    ]
    assert list(profiler_reports._read_csv(paths[1])[0])[:5] == [
        "operator_name",
        "kernel_name",
        "kernel_call_count",
        "kernel_time_us",
        "percent",
    ]
    assert list(profiler_reports._read_csv(paths[2])[0])[:5] == [
        "operator_id",
        "operator_name",
        "operator_kind",
        "kernel_name",
        "kernel_time_percent(%)",
    ]
    assert diagnostics["kernel_event_count_by_classification"] == {
        "communication": 1,
        "compute": 1,
        "filtered_non_compute": 1,
        "fused_communication_compute": 1,
        "included_kernel": 3,
        "observed": 4,
    }
    assert diagnostics["kernel_event_count"] == 3
    assert all(diagnostics["conservation"].values())
    assert not list(tmp_path.glob("*.tmp"))
    with paths[2].open(encoding="utf-8-sig", newline="") as source:
        operator_rows = list(csv.DictReader(source))
    assert {row["operator_kind"] for row in operator_rows} >= {
        "communication",
        "fused_communication_compute",
    }
    assert operator_rows[-1]["operator_kind"] == "communication"


def test_cross_rank_report_aggregates_operator_time(tmp_path):
    profiler_reports.export_kernel_reports([_event()], tmp_path, rank=0)
    profiler_reports.export_kernel_reports([_event()], tmp_path, rank=2)

    csv_path, json_path = profiler_reports.export_cross_rank_reports(tmp_path)

    rows = profiler_reports._read_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["ranks"] == "[0,2]"
    assert rows[0]["rank_count"] == "2"
    assert rows[0]["total_kernel_call_count"] == "2"
    assert rows[0]["total_kernel_time_us"] == "5"
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["profiled_ranks"] == [0, 2]


def test_offline_trace_export_preserves_custom_parent_and_unattributed_kernel(
    tmp_path,
):
    trace_path = tmp_path / "rank-4.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "FusedAttnFunc",
                        "pid": 1,
                        "tid": 1,
                        "ts": 0,
                        "dur": 10,
                        "args": {"External id": 1},
                    },
                    {
                        "cat": "cpu_op",
                        "ph": "X",
                        "name": "aten::mm",
                        "pid": 1,
                        "tid": 1,
                        "ts": 1,
                        "dur": 5,
                        "args": {
                            "External id": 7,
                            "Input Dims": [[2, 2]],
                            "Input type": ["float"],
                        },
                    },
                    {
                        "cat": "kernel",
                        "ph": "X",
                        "name": "cublas_kernel",
                        "dur": 2.5,
                        "args": {"External id": 7},
                    },
                    {
                        "cat": "kernel",
                        "ph": "X",
                        "name": "unattributed_kernel",
                        "dur": 1.0,
                        "args": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = profiler_reports.export_kernel_reports_from_trace(trace_path)

    details = profiler_reports._read_csv(paths[0])
    by_kernel = {row["kernel_name"]: row for row in details}
    assert by_kernel["cublas_kernel"]["custom_operator"] == "FusedAttnFunc"
    assert by_kernel["cublas_kernel"]["execution_operator"] == "aten::mm"
    assert by_kernel["unattributed_kernel"]["custom_operator"] == "null"
    assert by_kernel["unattributed_kernel"]["mapping_status"] == "missing_external_id"
    diagnostics = json.loads(paths[3].read_text(encoding="utf-8"))
    assert diagnostics["scope"]["rank"] == 4
    assert diagnostics["kernel_event_count"] == 2
    assert diagnostics["kernel_event_count_by_classification"]["observed"] == 2
    assert all(diagnostics["conservation"].values())


def test_trace_directory_export_processes_all_ranks(tmp_path):
    for rank in (0, 3):
        (tmp_path / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "cat": "kernel",
                            "ph": "X",
                            "name": "unattributed_kernel",
                            "dur": rank + 1,
                            "args": {},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    per_rank, cross_rank = profiler_reports.export_kernel_reports_from_trace_dir(tmp_path)
    repeated_per_rank, _ = profiler_reports.export_kernel_reports_from_trace_dir(tmp_path)

    assert len(per_rank) == 2
    assert len(repeated_per_rank) == 2
    summary = json.loads(cross_rank[1].read_text(encoding="utf-8"))
    assert summary["profiled_ranks"] == [0, 3]
