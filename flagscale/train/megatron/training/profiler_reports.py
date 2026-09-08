"""Generate operator-to-kernel reports from PyTorch profiler events."""

import csv
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path

DETAIL_FIELDS = [
    "custom_operator", "execution_operator", "kernel_name", "variant_index", "mapping_status",
    "input_shapes", "input_dtypes", "candidate_operators",
    "kernel_event_count", "kernel_time_us",
]
SUMMARY_FIELDS = [
    "custom_operator", "execution_operator", "kernel_name", "kernel_call_count",
    "kernel_time_us", "percent",
]
OPERATOR_FIELDS = [
    "operator_id", "custom_operator", "execution_operator", "operator_kind", "kernel_name",
]

# These are backend/library identities, not generic tensor operation names. Keep
# them centralized so support for another communication backend is a data-only change.
COMMUNICATION_OPERATOR_MARKERS = (
    "record_param_comms", "torch.distributed", "distributed::", "_c10d", "c10d::",
    "symm_mem::", "custom_ar::", "processgroup", "allreduce", "all_reduce",
    "allgather", "all_gather", "reducescatter", "reduce_scatter", "alltoall",
    "all_to_all", "sendrecv", "sequenceparallelregion",
)
COMMUNICATION_KERNEL_MARKERS = (
    "nccl", "rccl", "hccl", "oneccl", "custom_all_reduce", "cross_device_reduce",
    "allreduce", "all_reduce", "allgather", "all_gather", "reducescatter",
    "reduce_scatter", "alltoall", "all_to_all", "sendrecv",
)
NON_COMPUTE_KERNEL_MARKERS = ("memcpy", "memset")
NON_COMPUTE_EVENT_NAMES = {
    "Lazy Function Loading",
    "Runtime Triggered Module Loading",
}
TORCH_COMPILE_PREFIXES = ("triton_", "CompiledFunction")
EXECUTION_BACKENDS = {
    "cuBLASLt", "cuBLAS", "CUTLASS", "TransformerEngine", "Triton",
    "FlashAttention", "cuDNN", "CUDA",
}
FRAMEWORK_OPERATOR_PREFIXES = (
    "aten::", "autograd::", "torch::", "prims::", "ProfilerStep#",
    "TorchDynamo ", "Torch-Compiled Region", "CompiledFunction", "triton_",
)

# Profiler-visible autograd classes scanned from TE-FL 21b4b4b1. Keep the full
# inventory auditable, then exclude infrastructure and metadata-only helpers.
TE_AUTOGRAD_FUNCTIONS = frozenset({
    "AttnFuncFL", "FP8EmulationFunc", "_PrepareQKVForFA", "FusedAttnFunc",
    "AttnFuncWithCPAndKVP2P", "AttnFuncWithCPAndKVAllGather",
    "AttnFuncWithCPAndQKVOA2A", "FusedAttentionWithScoreModFunc",
    "ScaledUpperTriangMaskedSoftmax", "ScaledAlignedCausalMaskedSoftmax",
    "ScaledMaskedSoftmax", "ScaledSoftmax", "PackTensors", "UnpackTensor",
    "ConvertTHDtoBSHD", "ConvertBSHDtoTHD", "FusedRoPEFunc", "FusedQKVRoPEFunc",
    "CrossEntropyFunction", "_Fp8Padding", "_Fp8Unpadding", "_GroupedLinear",
    "_LayerNormLinear", "_LayerNormMLP", "_Linear",
    "_OperationFuserAutogradFunction", "FusedTopkScoreFunction",
    "FusedComputeScoresForMoEAuxLoss", "FusedAuxLoss", "_QuantizeFunc",
    "_FromFloat8Func", "_FromMXFP8Func", "_FromNVFP4Func", "mHCProjectionOp",
    "mHCScaleFusedOp", "mHCSinkhornOp", "mHCAggregateOp",
    "mHCExpandCombineOp", "SplitAlongDim", "AllGatherFunc",
    "GroupCommitFunction", "_CheckpointFunction", "_EpDispatch", "_EpCombine",
    "Graphed", "_NoopCatFunc", "_IdentityFunc", "_ViewFunc", "_ReshapeFunc",
    "_GroupedIdentityFunc",
})
TE_NON_COMPUTE_AUTOGRAD_FUNCTIONS = frozenset({
    "AllGatherFunc", "GroupCommitFunction", "_CheckpointFunction", "_EpDispatch",
    "_EpCombine", "Graphed", "_NoopCatFunc", "_IdentityFunc", "_ViewFunc",
    "_ReshapeFunc", "_GroupedIdentityFunc",
})
TE_CUSTOM_AUTOGRAD_FUNCTIONS = (
    TE_AUTOGRAD_FUNCTIONS - TE_NON_COMPUTE_AUTOGRAD_FUNCTIONS
)

# Profiler-visible compute boundaries from the Megatron main revision used by FlagScale.
MEGATRON_CUSTOM_AUTOGRAD_FUNCTIONS = frozenset({
    "BiasGeGLUFunction", "GeGLUFunction", "WeightedQuickGeGLUFunction",
    "WeightedBiasQuickGeGLUFunction", "GeLUFunction", "BiasSwiGLUFunction",
    "SwiGLUFunction", "WeightedSwiGLUFunction", "_VocabParallelCrossEntropy",
    "_VocabParallelCrossEntropyChunked", "IndicesToMultihot", "TritonFusedSinkhorn",
    "CutileSinkhornKnopp", "CutileHAggregate", "CutileProjRms",
    "CutileProjRmsComputeH", "FusedHAggregate", "FusedHPostBDA",
    "_FusedMLARoPEInplace", "_FusedMLARoPEKVSplit",
    "WeightedSquaredReLUFunction", "LinearWithFrozenWeight",
    "LinearWithGradAccumulationAndAsyncCommunication",
    "LinearWithGradAccumulationAndAsyncCommunicationKunlunxin",
    "BatchInvariantTEGemmFn", "BatchInvariantRMSNormFn", "FusedDSAIndexerLoss",
    "SparseAttnFunc", "FusedIndexerSparseAttnFunc", "_DSASparseAttnFunc",
    "SinkhornKnopp", "BroadcastTensorFused", "RandomSTE", "RandomSTEShared",
    "RouterGatingLinearFunction", "RotaryPositionalEmbeddingWithFreqFunction",
})

CUSTOM_AUTOGRAD_FUNCTIONS = (
    TE_CUSTOM_AUTOGRAD_FUNCTIONS | MEGATRON_CUSTOM_AUTOGRAD_FUNCTIONS
)
CUSTOM_OPERATOR_NAMES = CUSTOM_AUTOGRAD_FUNCTIONS | frozenset(
    f"{name}Backward" for name in CUSTOM_AUTOGRAD_FUNCTIONS
)
CUSTOM_OPERATOR_PREFIXES = ("te_moe::", "tex::")


def _json_field(value):
    value = [] if value in (None, "") else value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _is_communication(operator_name, kernel_name):
    operator = operator_name.lower()
    kernel = kernel_name.lower()
    return any(marker in operator for marker in COMMUNICATION_OPERATOR_MARKERS) or any(
        marker in kernel for marker in COMMUNICATION_KERNEL_MARKERS
    )


def _is_custom_operator(name):
    """Return whether a CPU op is a known custom compute boundary."""
    if not name or name.startswith(FRAMEWORK_OPERATOR_PREFIXES):
        return False
    if _is_communication(name, ""):
        return False
    return name in CUSTOM_OPERATOR_NAMES or name.startswith(CUSTOM_OPERATOR_PREFIXES)


def _load_trace_metadata(trace_path):
    """Load CPU-op metadata and nearest custom parents from a Kineto trace."""
    if trace_path is None:
        return None, {}, {}
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)

    cpu_events = []
    cpu_operator_names = set()
    input_dtypes = defaultdict(deque)
    for event in trace.get("traceEvents", ()):
        if event.get("cat") != "cpu_op" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        cpu_operator_names.add(name)
        cpu_events.append(event)
        args = event.get("args", {})
        if "Input type" in args:
            key = (name, _json_field(args.get("Input Dims")))
            input_dtypes[key].append(args["Input type"])

    custom_parents = defaultdict(deque)
    events_by_thread = defaultdict(list)
    for event in cpu_events:
        events_by_thread[(event.get("pid"), event.get("tid"))].append(event)
    for events in events_by_thread.values():
        events.sort(key=lambda event: (event["ts"], -event.get("dur", 0.0)))
        stack = []
        for event in events:
            start = event["ts"]
            end = start + event.get("dur", 0.0)
            while stack and (start >= stack[-1][1] or end > stack[-1][1]):
                stack.pop()
            name = str(event.get("name", ""))
            shapes = _json_field(event.get("args", {}).get("Input Dims"))
            custom_parent = name if _is_custom_operator(name) else "null"
            if custom_parent == "null":
                for parent, _ in reversed(stack):
                    parent_name = str(parent.get("name", ""))
                    if _is_custom_operator(parent_name):
                        custom_parent = parent_name
                        break
            custom_parents[(name, shapes)].append(custom_parent)
            stack.append((event, end))
    return cpu_operator_names, input_dtypes, custom_parents


def _is_non_compute(operator_name, kernel_name, cpu_operator_names):
    if operator_name in NON_COMPUTE_EVENT_NAMES:
        return True
    # Category is more stable than CUDA/HIP API naming. If a trace is available,
    # reject events that Kineto did not classify as CPU operators.
    if cpu_operator_names is not None and operator_name not in cpu_operator_names:
        return True
    kernel = kernel_name.lower()
    return any(marker in kernel for marker in NON_COMPUTE_KERNEL_MARKERS)


def _kernel_backend(kernel_name):
    """Infer a stable implementation backend from an observed kernel name."""
    name = kernel_name.lower()
    if "nvjet" in name or "cublaslt" in name:
        return "cuBLASLt"
    if "cutlass" in name:
        return "CUTLASS"
    if "transformer_engine" in name:
        return "TransformerEngine"
    if "triton" in name:
        return "Triton"
    if any(marker in name for marker in ("flash_attn", "flashattention", "fmha")):
        return "FlashAttention"
    if "cudnn" in name:
        return "cuDNN"
    if "cublas" in name:
        return "cuBLAS"
    return "CUDA"


def _execution_operator(custom_operator, operator_name, kernel_name):
    """Use an observed inner ATen op, otherwise the inferred kernel backend."""
    if custom_operator == "null":
        return operator_name
    if custom_operator != operator_name and operator_name.startswith("aten::"):
        return operator_name
    return _kernel_backend(kernel_name)


def _operator_kind(name):
    if name == "null":
        return "unattributed"
    if name in EXECUTION_BACKENDS:
        return "backend"
    if name.startswith("aten::"):
        return "aten"
    if name.startswith(TORCH_COMPILE_PREFIXES):
        return "torch_compile"
    if _is_custom_operator(name):
        return "custom"
    if "::" in name:
        return "custom"
    return "runtime_operator"


def _duration_us(kernel):
    duration = getattr(kernel, "duration", 0.0)
    return float(duration() if callable(duration) else duration)


def _time(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _percent(value):
    return "<0.001%" if 0 < value < 0.001 else f"{value:.3f}%"


def build_kernel_report_rows(events, trace_path=None):
    """Build detailed and summary rows from ``torch.profiler`` events."""
    variants = defaultdict(lambda: [0, 0.0])
    summaries = defaultdict(lambda: [0, 0.0])
    cpu_operator_names, trace_dtypes, trace_custom_parents = _load_trace_metadata(trace_path)

    for event in events:
        if getattr(event, "is_user_annotation", False):
            continue
        operator = str(getattr(event, "key", None) or getattr(event, "name", "null"))
        shapes = _json_field(
            getattr(event, "structured_input_shapes", None)
            or getattr(event, "input_shapes", None)
        )
        dtype_queue = trace_dtypes.get((operator, shapes))
        trace_event_dtypes = dtype_queue.popleft() if dtype_queue else None
        custom_parent_queue = trace_custom_parents.get((operator, shapes))
        custom_operator = custom_parent_queue.popleft() if custom_parent_queue else "null"
        event_dtypes = getattr(event, "input_dtypes", None)
        if not event_dtypes and trace_event_dtypes:
            event_dtypes = trace_event_dtypes
        kernels = getattr(event, "kernels", None) or ()
        if not kernels:
            continue
        dtypes = _json_field(event_dtypes)

        for kernel in kernels:
            name = str(getattr(kernel, "name", "null"))
            if _is_communication(operator, name) or _is_non_compute(
                operator, name, cpu_operator_names
            ):
                continue
            duration = _duration_us(kernel)
            execution_operator = _execution_operator(custom_operator, operator, name)
            variants[(custom_operator, execution_operator, name, shapes, dtypes)][0] += 1
            variants[(custom_operator, execution_operator, name, shapes, dtypes)][1] += duration
            summaries[(custom_operator, execution_operator, name)][0] += 1
            summaries[(custom_operator, execution_operator, name)][1] += duration

    grouped = defaultdict(list)
    for (custom, execution, kernel, shapes, dtypes), (count, duration) in variants.items():
        grouped[(custom, execution, kernel)].append((shapes, dtypes, count, duration))

    details = []
    for (custom, execution, kernel), items in sorted(grouped.items()):
        items.sort(key=lambda item: (-item[3], item[0], item[1]))
        for index, (shapes, dtypes, count, duration) in enumerate(items, 1):
            details.append({
                "custom_operator": custom,
                "execution_operator": execution,
                "kernel_name": kernel,
                "variant_index": index,
                "mapping_status": "operator_shape_matched",
                "input_shapes": shapes,
                "input_dtypes": dtypes,
                "candidate_operators": "null",
                "kernel_event_count": count,
                "kernel_time_us": _time(duration),
            })

    total = sum(item[1] for item in summaries.values())
    summary = []
    for (custom, execution, kernel), (count, duration) in sorted(
        summaries.items(), key=lambda item: (-item[1][1], item[0])
    ):
        summary.append({
            "custom_operator": custom,
            "execution_operator": execution,
            "kernel_name": kernel,
            "kernel_call_count": count,
            "kernel_time_us": _time(duration),
            "percent": _percent(duration / total * 100.0 if total else 0.0),
        })
    return details, summary


def build_operator_rows(summary_rows):
    """List unique operator-to-kernel mappings using stable grouped IDs."""
    pairs = {
        (row["custom_operator"], row["execution_operator"], row["kernel_name"])
        for row in summary_rows
    }
    kind_order = {
        "aten": 0, "backend": 1, "custom": 2, "runtime_operator": 3,
        "torch_compile": 4, "unattributed": 5,
    }
    pairs = sorted(
        pairs,
        key=lambda pair: (kind_order[_operator_kind(pair[1])], pair[0], pair[1], pair[2]),
    )
    operator_ids = {}
    rows = []
    for custom, operator, kernel in pairs:
        operator_key = (custom, operator)
        operator_ids.setdefault(operator_key, len(operator_ids) + 1)
        rows.append({
            "operator_id": operator_ids[operator_key],
            "custom_operator": custom,
            "execution_operator": operator,
            "operator_kind": _operator_kind(operator),
            "kernel_name": kernel,
        })
    return rows


def _write(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_kernel_reports(events, profile_dir, rank, trace_path=None):
    """Write detailed, summary, and operator-to-kernel reports for one rank."""
    details, summary = build_kernel_report_rows(events, trace_path)
    operators = build_operator_rows(summary)
    profile_dir = Path(profile_dir)
    details_path = profile_dir / f"rank-{rank}_kernel_details_report.csv"
    summary_path = profile_dir / f"rank-{rank}_kernel_summary.csv"
    operator_path = profile_dir / f"rank-{rank}_operator_list.csv"
    _write(details_path, DETAIL_FIELDS, details)
    _write(summary_path, SUMMARY_FIELDS, summary)
    _write(operator_path, OPERATOR_FIELDS, operators)
    return details_path, summary_path, operator_path
