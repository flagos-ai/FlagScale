"""Generate operator-to-kernel reports from PyTorch profiler events."""

import csv
import gzip
import json
import re
from collections import Counter, defaultdict, deque
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from types import SimpleNamespace

DETAIL_FIELDS = [
    "operator_name", "kernel_name", "variant_index", "mapping_status",
    "input_shapes", "input_dtypes", "candidate_operators", "kernel_event_count",
    "kernel_time_us", "custom_operator", "execution_operator", "mapping_source",
]
SUMMARY_FIELDS = [
    "operator_name", "kernel_name", "kernel_call_count", "kernel_time_us", "percent",
    "custom_operator", "execution_operator",
]
OPERATOR_FIELDS = [
    "operator_id", "operator_name", "operator_kind", "kernel_name",
    "kernel_time_percent(%)", "operator_identity", "custom_operator",
    "execution_operator", "kernel_callable",
]
CROSS_RANK_FIELDS = [
    "operator_identity", "custom_operator", "execution_operator", "operator_kind",
    "ranks", "rank_count", "kernel_names", "total_kernel_call_count",
    "total_kernel_time_us", "min_rank_kernel_time_us", "max_rank_kernel_time_us",
    "mean_rank_kernel_time_us",
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
COMMUNICATION_COMPUTE_MARKERS = ("fused", "fusion", "gemm", "matmul", "norm")
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
TRACE_READ_CHUNK_SIZE = 1024 * 1024
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def _iter_trace_events(trace_path):
    """Yield Chrome trace events without loading the complete trace into memory."""
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as source:
        decoder = json.JSONDecoder()
        key = '"traceEvents"'
        buffer = ""
        position = 0
        array_started = False
        eof = False
        need_more = True
        while True:
            if need_more and not eof:
                chunk = source.read(TRACE_READ_CHUNK_SIZE)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            need_more = False
            if not array_started:
                key_index = buffer.find(key, position)
                if key_index < 0:
                    if eof:
                        raise ValueError(f"traceEvents not found: {trace_path}")
                    buffer = buffer[-(len(key) - 1):]
                    position = 0
                    need_more = True
                    continue
                colon_index = buffer.find(":", key_index + len(key))
                array_index = buffer.find("[", colon_index + 1) if colon_index >= 0 else -1
                if array_index < 0:
                    if eof:
                        raise ValueError(f"invalid traceEvents field: {trace_path}")
                    buffer = buffer[key_index:]
                    position = 0
                    need_more = True
                    continue
                position = array_index + 1
                array_started = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                continue
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"unterminated traceEvents array: {trace_path}")
                buffer = ""
                position = 0
                need_more = True
                continue
            try:
                event, end_index = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError(f"invalid traceEvents entry: {trace_path}") from error
                buffer = buffer[position:]
                position = 0
                need_more = True
                continue
            if not isinstance(event, dict):
                raise ValueError(f"traceEvents entry is not an object: {trace_path}")
            yield event
            position = end_index
            if position >= TRACE_READ_CHUNK_SIZE:
                buffer = buffer[position:]
                position = 0


def _json_field(value):
    if value in (None, "", []):
        value = None
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _mapping_status(shapes, dtypes):
    """Describe which profiler input metadata was available for an operator."""
    has_shapes = shapes != "null"
    has_dtypes = dtypes != "null"
    if has_shapes and has_dtypes:
        return "operator_shape_matched"
    if has_shapes:
        return "operator_matched_dtype_missing"
    if has_dtypes:
        return "operator_matched_shape_missing"
    return "operator_matched_metadata_missing"


def _is_communication(operator_name, kernel_name):
    operator = operator_name.lower()
    kernel = kernel_name.lower()
    return any(marker in operator for marker in COMMUNICATION_OPERATOR_MARKERS) or any(
        marker in kernel for marker in COMMUNICATION_KERNEL_MARKERS
    )


def _communication_kind(operator_name, kernel_name):
    """Classify pure communication separately from communication-compute fusion."""
    if not _is_communication(operator_name, kernel_name):
        return None
    combined = f"{operator_name} {kernel_name}".lower()
    if any(marker in combined for marker in COMMUNICATION_COMPUTE_MARKERS):
        return "fused_communication_compute"
    return "communication"


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
        return None, {}, {}, {}, None
    cpu_events = []
    cpu_operator_names = set()
    input_dtypes = defaultdict(deque)
    cpu_metadata_counts = Counter()
    gpu_category_counts = Counter()
    gpu_category_times = Counter()
    for event in _iter_trace_events(trace_path):
        category = str(event.get("cat", ""))
        if category in GPU_CATEGORIES and event.get("ph") == "X":
            gpu_category_counts[category] += 1
            gpu_category_times[category] += round(
                float(event.get("dur", 0.0)) * 1000
            )
        if event.get("cat") != "cpu_op" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        cpu_operator_names.add(name)
        cpu_events.append(event)
        args = event.get("args", {})
        has_shapes = "Input Dims" in args
        has_dtypes = "Input type" in args
        metadata_status = (
            "shape_and_dtype"
            if has_shapes and has_dtypes
            else "shape_only"
            if has_shapes
            else "dtype_only"
            if has_dtypes
            else "no_input_metadata"
        )
        cpu_metadata_counts[metadata_status] += 1
        key = (name, _json_field(args.get("Input Dims")))
        input_dtypes[key].append(args.get("Input type"))

    custom_parents = defaultdict(deque)
    metadata_by_external_id = defaultdict(list)
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
            external_id = event.get("args", {}).get("External id")
            if isinstance(external_id, (int, str)):
                metadata_by_external_id[external_id].append({
                    "operator": name,
                    "input_shapes": shapes,
                    "input_dtypes": event.get("args", {}).get("Input type"),
                    "custom_operator": custom_parent,
                })
            stack.append((event, end))
    trace_statistics = {
        "cpu_operator_event_count": len(cpu_events),
        "unique_cpu_operator_names": len(cpu_operator_names),
        "cpu_operator_event_count_by_metadata_status": dict(
            sorted(cpu_metadata_counts.items())
        ),
        "gpu_event_count": sum(gpu_category_counts.values()),
        "gpu_event_count_by_category": dict(sorted(gpu_category_counts.items())),
        "gpu_activity_ns_by_category": dict(sorted(gpu_category_times.items())),
    }
    return (
        cpu_operator_names,
        input_dtypes,
        custom_parents,
        metadata_by_external_id,
        trace_statistics,
    )


def _candidate_rows(candidates):
    return [
        {
            "operator": candidate["operator"],
            "input_shapes": json.loads(candidate["input_shapes"]),
            "input_dtypes": candidate["input_dtypes"],
            "custom_operator": candidate["custom_operator"],
        }
        for candidate in candidates
    ]


def _resolve_trace_metadata(
    event,
    operator,
    shapes,
    fallback_dtype,
    fallback_custom_parent,
    metadata_by_external_id,
):
    """Prefer correlation ID metadata and retain ambiguity instead of guessing."""
    external_id = getattr(event, "id", None)
    is_trace_kernel = getattr(event, "is_trace_kernel", False)
    candidates = metadata_by_external_id.get(external_id, ())
    if candidates:
        named = [candidate for candidate in candidates if candidate["operator"] == operator]
        candidates = named or list(candidates)
        shaped = [
            candidate
            for candidate in candidates
            if shapes == "null" or candidate["input_shapes"] == shapes
        ]
        candidates = shaped or candidates
        unique = {
            (
                candidate["operator"],
                candidate["input_shapes"],
                _json_field(candidate["input_dtypes"]),
                candidate["custom_operator"],
            ): candidate
            for candidate in candidates
        }
        candidates = list(unique.values())
        if len(candidates) == 1:
            candidate = candidates[0]
            return {
                "shapes": shapes if shapes != "null" else candidate["input_shapes"],
                "dtypes": candidate["input_dtypes"],
                "custom_operator": candidate["custom_operator"],
                "mapping_source": "external_id",
                "mapping_status": None,
                "candidate_operators": None,
            }
        names = {candidate["operator"] for candidate in candidates}
        return {
            "shapes": shapes,
            "dtypes": fallback_dtype,
            "custom_operator": fallback_custom_parent,
            "mapping_source": "external_id_ambiguous",
            "mapping_status": (
                "shape_ambiguous" if len(names) == 1 else "operator_ambiguous"
            ),
            "candidate_operators": _candidate_rows(candidates),
        }
    if is_trace_kernel:
        return {
            "shapes": shapes,
            "dtypes": fallback_dtype,
            "custom_operator": fallback_custom_parent,
            "mapping_source": "trace_kernel",
            "mapping_status": (
                "missing_external_id"
                if external_id is None
                else "no_cpu_op_match"
            ),
            "candidate_operators": None,
        }
    return {
        "shapes": shapes,
        "dtypes": fallback_dtype,
        "custom_operator": fallback_custom_parent,
        "mapping_source": (
            "operator_shape_sequence"
            if fallback_dtype is not None or fallback_custom_parent != "null"
            else "function_event"
        ),
        "mapping_status": None,
        "candidate_operators": None,
    }


def _is_non_compute(
    operator_name, kernel_name, cpu_operator_names, require_cpu_operator=True
):
    if operator_name in NON_COMPUTE_EVENT_NAMES:
        return True
    # Category is more stable than CUDA/HIP API naming. If a trace is available,
    # reject events that Kineto did not classify as CPU operators.
    if (
        require_cpu_operator
        and cpu_operator_names is not None
        and operator_name not in cpu_operator_names
    ):
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


def _operator_kind(name, kernel_name=""):
    communication_kind = _communication_kind(name, kernel_name)
    if communication_kind is not None:
        return communication_kind
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


def _kernel_callable_identity_name(kernel_name):
    """Remove launch arguments while preserving semantic C++ template values."""
    name = kernel_name.strip()
    if name.startswith("void "):
        name = name[5:].lstrip()
    if name.endswith(")"):
        depth = 0
        for index in range(len(name) - 1, -1, -1):
            character = name[index]
            if character == ")":
                depth += 1
            elif character == "(":
                depth -= 1
                if depth == 0:
                    name = name[:index].rstrip()
                    break
    template_start = name.find("<")
    if template_start >= 0:
        prefix = re.sub(r"_rank_\d+$", "", name[:template_start])
        return prefix + name[template_start:]
    return re.sub(r"_rank_\d+$", "", name)


def _operator_identity(custom_operator, execution_operator, kernel_name):
    kind = _operator_kind(execution_operator, kernel_name)
    if kind == "unattributed":
        return ("kernel_callable", _kernel_callable_identity_name(kernel_name))
    if kind == "torch_compile":
        return ("compile_kernel", kernel_name)
    if custom_operator != "null":
        return ("custom_execution", custom_operator, kind, execution_operator)
    return ("execution_operator", kind, execution_operator)


def _duration_ns(kernel):
    duration = getattr(kernel, "duration", 0.0)
    duration_us = float(duration() if callable(duration) else duration)
    return round(duration_us * 1000)


def _time_from_ns(value):
    return f"{value / 1000:.3f}".rstrip("0").rstrip(".")


def _us_from_ns(value):
    return value / 1000


def _percent(value):
    return "<0.001%" if 0 < value < 0.001 else f"{value:.3f}%"


def _operator_percent(value_ns, total_ns):
    percent = Decimal(value_ns) * 100 / Decimal(total_ns) if total_ns else Decimal(0)
    if percent < Decimal("0.01"):
        return "<0.01"
    return format(
        percent.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        "f",
    )


def _custom_operator_summary(custom_operators):
    """Keep the legacy scalar for one parent and encode multiple parents losslessly."""
    values = sorted(custom_operators)
    return values[0] if len(values) == 1 else _json_field(values)


def _collect_kernel_report_rows(
    events,
    trace_path=None,
    events_are_trace_kernels=False,
    trace_metadata=None,
):
    """Build report rows and accounting data from ``torch.profiler`` events."""
    variants = defaultdict(lambda: [0, 0])
    summaries = defaultdict(lambda: [0, 0, set()])
    event_counts = Counter()
    event_times = Counter()
    mapping_counts = Counter()
    mapping_times = Counter()
    if trace_metadata is None:
        trace_metadata = _load_trace_metadata(trace_path)
    (
        cpu_operator_names,
        trace_dtypes,
        trace_custom_parents,
        metadata_by_external_id,
        trace_statistics,
    ) = trace_metadata

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
        fallback_custom_parent = (
            custom_parent_queue.popleft() if custom_parent_queue else "null"
        )
        event_dtypes = getattr(event, "input_dtypes", None)
        if not event_dtypes and trace_event_dtypes:
            event_dtypes = trace_event_dtypes
        resolved = _resolve_trace_metadata(
            event,
            operator,
            shapes,
            event_dtypes,
            fallback_custom_parent,
            metadata_by_external_id,
        )
        shapes = resolved["shapes"]
        event_dtypes = resolved["dtypes"]
        custom_operator = resolved["custom_operator"]
        kernels = getattr(event, "kernels", None) or ()
        if not kernels:
            continue
        dtypes = _json_field(event_dtypes)
        mapping_status = resolved["mapping_status"] or _mapping_status(shapes, dtypes)
        mapping_source = resolved["mapping_source"]
        candidate_operators = _json_field(resolved["candidate_operators"])

        for kernel in kernels:
            name = str(getattr(kernel, "name", "null"))
            duration = _duration_ns(kernel)
            event_counts["observed"] += 1
            event_times["observed"] += duration
            communication_kind = _communication_kind(operator, name)
            if not events_are_trace_kernels and _is_non_compute(
                operator,
                name,
                cpu_operator_names,
                require_cpu_operator=True,
            ):
                event_counts["filtered_non_compute"] += 1
                event_times["filtered_non_compute"] += duration
                continue
            event_counts["included_kernel"] += 1
            event_times["included_kernel"] += duration
            classification = communication_kind or "compute"
            event_counts[classification] += 1
            event_times[classification] += duration
            mapping_counts[mapping_status] += 1
            mapping_times[mapping_status] += duration
            execution_operator = _execution_operator(custom_operator, operator, name)
            variant_key = (
                custom_operator,
                execution_operator,
                name,
                mapping_status,
                mapping_source,
                shapes,
                dtypes,
                candidate_operators,
            )
            variants[variant_key][0] += 1
            variants[variant_key][1] += duration
            summaries[(execution_operator, name)][0] += 1
            summaries[(execution_operator, name)][1] += duration
            summaries[(execution_operator, name)][2].add(custom_operator)

    grouped = defaultdict(list)
    for (
        custom,
        execution,
        kernel,
        mapping_status,
        mapping_source,
        shapes,
        dtypes,
        candidate_operators,
    ), (count, duration) in variants.items():
        grouped[(custom, execution, kernel)].append(
            (
                mapping_status,
                mapping_source,
                shapes,
                dtypes,
                candidate_operators,
                count,
                duration,
            )
        )

    details = []
    for (custom, execution, kernel), items in sorted(grouped.items()):
        items.sort(key=lambda item: (-item[6], item[0], item[1], item[2], item[3]))
        for index, (
            mapping_status,
            mapping_source,
            shapes,
            dtypes,
            candidate_operators,
            count,
            duration,
        ) in enumerate(items, 1):
            details.append({
                "operator_name": execution,
                "custom_operator": custom,
                "execution_operator": execution,
                "kernel_name": kernel,
                "variant_index": index,
                "mapping_status": mapping_status,
                "mapping_source": mapping_source,
                "input_shapes": shapes,
                "input_dtypes": dtypes,
                "candidate_operators": candidate_operators,
                "kernel_event_count": count,
                "kernel_time_us": _time_from_ns(duration),
            })

    total = sum(item[1] for item in summaries.values())
    summary = []
    for (execution, kernel), (count, duration, customs) in sorted(
        summaries.items(), key=lambda item: (-item[1][1], item[0])
    ):
        summary.append({
            "operator_name": execution,
            "custom_operator": _custom_operator_summary(customs),
            "execution_operator": execution,
            "kernel_name": kernel,
            "kernel_call_count": count,
            "kernel_time_us": _time_from_ns(duration),
            "percent": _percent(duration / total * 100.0 if total else 0.0),
        })
    diagnostics = {
        "kernel_event_count_by_classification": dict(sorted(event_counts.items())),
        "kernel_time_us_by_classification": {
            key: _us_from_ns(value) for key, value in sorted(event_times.items())
        },
        "mapping_event_count_by_status": dict(sorted(mapping_counts.items())),
        "mapping_time_us_by_status": {
            key: _us_from_ns(value) for key, value in sorted(mapping_times.items())
        },
        "trace_statistics": trace_statistics,
    }
    return details, summary, diagnostics


def build_kernel_report_rows(events, trace_path=None):
    """Build detailed and summary rows from ``torch.profiler`` events."""
    details, summary, _diagnostics = _collect_kernel_report_rows(
        events, trace_path
    )
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
        "fused_communication_compute": 6, "communication": 7,
    }
    pairs = sorted(
        pairs,
        key=lambda pair: (
            kind_order[_operator_kind(pair[1], pair[2])],
            pair[0],
            pair[1],
            pair[2],
        ),
    )
    relation_times = {
        (row["custom_operator"], row["execution_operator"], row["kernel_name"]):
        _csv_us_to_ns(row.get("kernel_time_us", 0))
        for row in summary_rows
    }
    total_time = sum(relation_times.values())
    operator_ids = {}
    rows = []
    for custom, operator, kernel in pairs:
        identity = _operator_identity(custom, operator, kernel)
        operator_key = identity
        operator_ids.setdefault(operator_key, len(operator_ids) + 1)
        rows.append({
            "operator_id": operator_ids[operator_key],
            "operator_name": operator,
            "operator_identity": _json_field(identity),
            "custom_operator": custom,
            "execution_operator": operator,
            "operator_kind": _operator_kind(operator, kernel),
            "kernel_name": kernel,
            "kernel_time_percent(%)": _operator_percent(
                relation_times[(custom, operator, kernel)], total_time
            ),
            "kernel_callable": _kernel_callable_identity_name(kernel),
        })
    return rows


def _write(path, fields, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _csv_us_to_ns(value):
    return int(Decimal(str(value)) * 1000)


def _validate_report_rows(details, summary, diagnostics):
    detail_count = sum(row["kernel_event_count"] for row in details)
    summary_count = sum(row["kernel_call_count"] for row in summary)
    detail_time_ns = sum(_csv_us_to_ns(row["kernel_time_us"]) for row in details)
    summary_time_ns = sum(_csv_us_to_ns(row["kernel_time_us"]) for row in summary)
    classification_counts = diagnostics["kernel_event_count_by_classification"]
    classification_times = diagnostics["kernel_time_us_by_classification"]
    included_count = classification_counts.get("included_kernel", 0)
    included_time_ns = _csv_us_to_ns(
        classification_times.get("included_kernel", 0.0)
    )
    detail_relations = {
        (row["operator_name"], row["kernel_name"]) for row in details
    }
    summary_relations = {
        (row["operator_name"], row["kernel_name"]) for row in summary
    }
    observed_time_ns = _csv_us_to_ns(
        classification_times.get("observed", 0.0)
    )
    accounted_time_ns = sum(
        _csv_us_to_ns(classification_times.get(key, 0.0))
        for key in ("included_kernel", "filtered_non_compute")
    )
    detail_counts_by_kernel = Counter()
    detail_times_by_kernel = Counter()
    for row in details:
        detail_counts_by_kernel[row["kernel_name"]] += row["kernel_event_count"]
        detail_times_by_kernel[row["kernel_name"]] += _csv_us_to_ns(
            row["kernel_time_us"]
        )
    summary_counts_by_kernel = Counter()
    summary_times_by_kernel = Counter()
    for row in summary:
        summary_counts_by_kernel[row["kernel_name"]] += row["kernel_call_count"]
        summary_times_by_kernel[row["kernel_name"]] += _csv_us_to_ns(
            row["kernel_time_us"]
        )
    return {
        "observed_kernel_event_count_accounted": (
            classification_counts.get("observed", 0)
            == included_count
            + classification_counts.get("filtered_non_compute", 0)
        ),
        "observed_kernel_time_accounted": observed_time_ns == accounted_time_ns,
        "kernel_key_sets_match": (
            set(detail_counts_by_kernel) == set(summary_counts_by_kernel)
        ),
        "kernel_event_count_in_trace": included_count,
        "kernel_event_count_in_summary": summary_count,
        "kernel_event_count_in_report": detail_count,
        "detail_summary_relations_match": detail_relations == summary_relations,
        "kernel_event_count_matches": detail_count == summary_count == included_count,
        "kernel_time_us_in_trace": _us_from_ns(included_time_ns),
        "kernel_time_us_in_summary": _us_from_ns(summary_time_ns),
        "kernel_time_us_in_report": _us_from_ns(detail_time_ns),
        "kernel_time_matches": (
            detail_time_ns == summary_time_ns == included_time_ns
        ),
        "detail_kernel_time_matches": detail_time_ns == included_time_ns,
        "summary_kernel_time_matches": summary_time_ns == included_time_ns,
        "per_kernel_call_count_matches": (
            detail_counts_by_kernel == summary_counts_by_kernel
        ),
        "per_kernel_time_matches": (
            detail_times_by_kernel == summary_times_by_kernel
        ),
    }


def _validate_csv_outputs(details_path, summary_path, operator_path, operators):
    """Validate serialized report relationships, IDs, order, and totals."""
    details = _read_csv(details_path)
    summary = _read_csv(summary_path)
    operator_rows = _read_csv(operator_path)
    summary_relations = [
        (row["operator_name"], row["kernel_name"]) for row in summary
    ]
    detail_relations = {
        (row["operator_name"], row["kernel_name"]) for row in details
    }
    operator_relations = [
        (row["operator_name"], row["operator_kind"], row["kernel_name"])
        for row in operator_rows
    ]
    detail_counts = Counter()
    detail_times = Counter()
    for row in details:
        detail_counts[row["kernel_name"]] += int(row["kernel_event_count"])
        detail_times[row["kernel_name"]] += _csv_us_to_ns(row["kernel_time_us"])
    summary_counts = Counter()
    summary_times = Counter()
    for row in summary:
        summary_counts[row["kernel_name"]] += int(row["kernel_call_count"])
        summary_times[row["kernel_name"]] += _csv_us_to_ns(row["kernel_time_us"])

    identity_to_id = {}
    id_to_identity = {}
    ids_stable = True
    for row in operator_rows:
        identity = row["operator_identity"]
        operator_id = row["operator_id"]
        ids_stable &= identity_to_id.setdefault(identity, operator_id) == operator_id
        ids_stable &= id_to_identity.setdefault(operator_id, identity) == identity

    seen_non_aten = False
    aten_rows_first = True
    seen_communication = False
    communication_rows_last = True
    communication_kinds = {"communication", "fused_communication_compute"}
    for row in operator_rows:
        is_aten = row["operator_kind"] == "aten"
        is_communication = row["operator_kind"] in communication_kinds
        if seen_non_aten and is_aten:
            aten_rows_first = False
        seen_non_aten |= not is_aten
        if seen_communication and not is_communication:
            communication_rows_last = False
        seen_communication |= is_communication

    expected_serialized = [
        {field: str(row[field]) for field in OPERATOR_FIELDS} for row in operators
    ]
    return {
        "csv_kernel_key_sets_match": set(detail_counts) == set(summary_counts),
        "csv_kernel_event_count_matches": detail_counts == summary_counts,
        "csv_kernel_time_matches": detail_times == summary_times,
        "csv_summary_relations_unique": (
            len(summary_relations) == len(set(summary_relations))
        ),
        "csv_summary_detail_relations_match": (
            set(summary_relations) == detail_relations
        ),
        "csv_operator_list_relations_unique": (
            len(operator_relations) == len(set(operator_relations))
        ),
        "csv_operator_list_rows_match_expected": (
            operator_rows == expected_serialized
        ),
        "csv_operator_list_kernel_sets_match": (
            {row["kernel_name"] for row in operator_rows} == set(summary_counts)
        ),
        "csv_operator_ids_present": all(
            re.fullmatch(r"[1-9]\d*", row["operator_id"]) is not None
            for row in operator_rows
        ),
        "csv_operator_ids_stable": ids_stable,
        "csv_operator_classification_matches": all(
            row["operator_kind"]
            == _operator_kind(row["execution_operator"], row["kernel_name"])
            for row in operator_rows
        ),
        "csv_aten_rows_first": aten_rows_first,
        "csv_communication_rows_last": communication_rows_last,
    }


def export_kernel_reports(
    events,
    profile_dir,
    rank,
    trace_path=None,
    events_are_trace_kernels=False,
    trace_metadata=None,
):
    """Write kernel reports and an accounting summary for one rank."""
    details, summary, diagnostics = _collect_kernel_report_rows(
        events,
        trace_path,
        events_are_trace_kernels=events_are_trace_kernels,
        trace_metadata=trace_metadata,
    )
    operators = build_operator_rows(summary)
    diagnostics["scope"] = {
        "rank": rank,
        "phase": "training_profile",
        "graph_capture_included": False,
        "timing_denominator": "sum_of_rank_kernel_durations",
    }
    if trace_path is not None:
        diagnostics["runtime_trace_files"] = [str(trace_path)]
    trace_statistics = diagnostics.pop("trace_statistics", None)
    if trace_statistics is not None:
        gpu_activity_ns = trace_statistics.pop("gpu_activity_ns_by_category")
        diagnostics.update(trace_statistics)
        diagnostics["gpu_activity_us_by_category"] = {
            key: _us_from_ns(value)
            for key, value in gpu_activity_ns.items()
        }
    included_time_ns = sum(
        _csv_us_to_ns(row["kernel_time_us"]) for row in summary
    )
    classification_counts = diagnostics["kernel_event_count_by_classification"]
    diagnostics["kernel_event_count"] = classification_counts.get(
        "included_kernel", 0
    )
    if trace_statistics is None:
        diagnostics["gpu_event_count"] = diagnostics["kernel_event_count"]
        diagnostics["gpu_event_count_by_category"] = {
            "kernel": diagnostics["kernel_event_count"]
        }
        diagnostics["gpu_activity_us_by_category"] = {
            "kernel": _us_from_ns(included_time_ns)
        }
    diagnostics["unique_kernel_names"] = len(
        {row["kernel_name"] for row in summary}
    )
    diagnostics["kernel_time_total_us"] = _us_from_ns(included_time_ns)
    diagnostics["kernel_mapping_event_count_by_status"] = diagnostics.pop(
        "mapping_event_count_by_status"
    )
    diagnostics["kernel_mapping_time_us_by_status"] = diagnostics.pop(
        "mapping_time_us_by_status"
    )
    diagnostics["conservation"] = _validate_report_rows(
        details, summary, diagnostics
    )
    failed_checks = [
        name
        for name, passed in diagnostics["conservation"].items()
        if isinstance(passed, bool) and not passed
    ]
    if failed_checks:
        raise RuntimeError(f"kernel report conservation failed: {failed_checks}")
    profile_dir = Path(profile_dir)
    details_path = profile_dir / f"rank-{rank}_kernel_shape_dtype.csv"
    summary_path = profile_dir / f"rank-{rank}_kernel_time.csv"
    operator_path = profile_dir / f"rank-{rank}_operator_list.csv"
    diagnostics_path = profile_dir / f"rank-{rank}_summary.json"
    _write(details_path, DETAIL_FIELDS, details)
    _write(summary_path, SUMMARY_FIELDS, summary)
    _write(operator_path, OPERATOR_FIELDS, operators)
    diagnostics["conservation"].update(
        _validate_csv_outputs(
            details_path, summary_path, operator_path, operators
        )
    )
    failed_checks = [
        name
        for name, passed in diagnostics["conservation"].items()
        if isinstance(passed, bool) and not passed
    ]
    if failed_checks:
        raise RuntimeError(f"kernel report CSV conservation failed: {failed_checks}")
    diagnostics_temporary = diagnostics_path.with_suffix(
        diagnostics_path.suffix + ".tmp"
    )
    diagnostics_temporary.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    diagnostics_temporary.replace(diagnostics_path)
    return (
        details_path,
        summary_path,
        operator_path,
        diagnostics_path,
    )


def _events_from_trace(trace_path, metadata_by_external_id=None):
    """Convert raw Kineto kernel events to the event interface used by reports."""
    if metadata_by_external_id is None:
        _, _, _, metadata_by_external_id, _ = _load_trace_metadata(trace_path)
    for trace_event in _iter_trace_events(trace_path):
        if trace_event.get("cat") != "kernel" or trace_event.get("ph") != "X":
            continue
        args = trace_event.get("args", {})
        external_id = args.get("External id")
        if not isinstance(external_id, (int, str)):
            external_id = None
        candidates = metadata_by_external_id.get(external_id, ())
        operator_names = {candidate["operator"] for candidate in candidates}
        operator = next(iter(operator_names)) if len(operator_names) == 1 else "null"
        candidate_shapes = {candidate["input_shapes"] for candidate in candidates}
        shapes = (
            json.loads(next(iter(candidate_shapes)))
            if len(candidate_shapes) == 1
            else None
        )
        kernel = SimpleNamespace(
            name=str(trace_event.get("name", "null")),
            duration=float(trace_event.get("dur", 0.0)),
        )
        yield SimpleNamespace(
            id=external_id,
            is_trace_kernel=True,
            is_user_annotation=False,
            key=operator,
            structured_input_shapes=shapes,
            input_shapes=None,
            input_dtypes=None,
            kernels=[kernel],
        )


def _rank_from_trace_path(trace_path):
    match = re.search(r"(?:^|[_-])rank[-_]?(\d+)(?:[._-]|$)", Path(trace_path).name)
    if match is None:
        match = re.search(r"^rank-(\d+)(?:[._-]|$)", Path(trace_path).name)
    return int(match.group(1)) if match else None


def export_kernel_reports_from_trace(trace_path, output_dir=None, rank=None):
    """Generate all per-rank reports offline from one Chrome trace."""
    trace_path = Path(trace_path)
    output_dir = Path(output_dir) if output_dir is not None else trace_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank is None:
        rank = _rank_from_trace_path(trace_path)
    if rank is None:
        raise ValueError(
            f"cannot infer rank from {trace_path.name}; pass rank explicitly"
        )
    trace_metadata = _load_trace_metadata(trace_path)
    return export_kernel_reports(
        _events_from_trace(trace_path, trace_metadata[3]),
        output_dir,
        rank,
        trace_path,
        events_are_trace_kernels=True,
        trace_metadata=trace_metadata,
    )


def _trace_files(trace_dir):
    trace_dir = Path(trace_dir)
    files = list(trace_dir.glob("rank-*.json"))
    files.extend(trace_dir.glob("rank-*.json.gz"))
    return sorted(
        file
        for file in set(files)
        if re.fullmatch(r"rank-\d+\.json(?:\.gz)?", file.name)
    )


def export_kernel_reports_from_trace_dir(trace_dir, output_dir=None):
    """Extract every rank trace in a directory, then aggregate across ranks."""
    trace_dir = Path(trace_dir)
    output_dir = Path(output_dir) if output_dir is not None else trace_dir
    traces = _trace_files(trace_dir)
    if not traces:
        raise FileNotFoundError(f"no rank trace files under {trace_dir}")
    per_rank_paths = [
        export_kernel_reports_from_trace(trace, output_dir) for trace in traces
    ]
    cross_rank_paths = export_cross_rank_reports(output_dir)
    return per_rank_paths, cross_rank_paths


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def export_cross_rank_reports(profile_dir):
    """Aggregate completed per-rank reports without distributed synchronization."""
    profile_dir = Path(profile_dir)
    operator_paths = sorted(profile_dir.glob("rank-*_operator_list.csv"))
    if not operator_paths:
        raise FileNotFoundError(f"no per-rank operator reports under {profile_dir}")

    grouped = {}
    discovered_ranks = []
    for operator_path in operator_paths:
        match = re.fullmatch(r"rank-(\d+)_operator_list\.csv", operator_path.name)
        if match is None:
            continue
        rank = int(match.group(1))
        discovered_ranks.append(rank)
        summary_path = profile_dir / f"rank-{rank}_kernel_time.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing rank-{rank} kernel summary: {summary_path}")
        operators_by_relation = {
            (
                row["custom_operator"],
                row["execution_operator"],
                row["kernel_name"],
            ): row
            for row in _read_csv(operator_path)
        }
        for summary_row in _read_csv(summary_path):
            relation = (
                summary_row["custom_operator"],
                summary_row["execution_operator"],
                summary_row["kernel_name"],
            )
            operator_row = operators_by_relation[relation]
            identity = operator_row["operator_identity"]
            item = grouped.setdefault(identity, {
                "custom_operators": set(),
                "execution_operators": set(),
                "operator_kinds": set(),
                "ranks": set(),
                "kernel_names": set(),
                "total_call_count": 0,
                "rank_time_ns": Counter(),
            })
            item["custom_operators"].add(operator_row["custom_operator"])
            item["execution_operators"].add(operator_row["execution_operator"])
            item["operator_kinds"].add(operator_row["operator_kind"])
            item["ranks"].add(rank)
            item["kernel_names"].add(operator_row["kernel_name"])
            item["total_call_count"] += int(summary_row["kernel_call_count"])
            item["rank_time_ns"][rank] += _csv_us_to_ns(
                summary_row["kernel_time_us"]
            )

    rows = []
    for identity, item in sorted(grouped.items()):
        rank_times = list(item["rank_time_ns"].values())
        total_time = sum(rank_times)
        rows.append({
            "operator_identity": identity,
            "custom_operator": _json_field(sorted(item["custom_operators"])),
            "execution_operator": _json_field(sorted(item["execution_operators"])),
            "operator_kind": _json_field(sorted(item["operator_kinds"])),
            "ranks": _json_field(sorted(item["ranks"])),
            "rank_count": len(item["ranks"]),
            "kernel_names": _json_field(sorted(item["kernel_names"])),
            "total_kernel_call_count": item["total_call_count"],
            "total_kernel_time_us": _time_from_ns(total_time),
            "min_rank_kernel_time_us": _time_from_ns(min(rank_times)),
            "max_rank_kernel_time_us": _time_from_ns(max(rank_times)),
            "mean_rank_kernel_time_us": _time_from_ns(
                round(total_time / len(rank_times))
            ),
        })

    csv_path = profile_dir / "cross_rank_operator_summary.csv"
    json_path = profile_dir / "cross_rank_kernel_report_summary.json"
    _write(csv_path, CROSS_RANK_FIELDS, rows)
    json_temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    json_temporary.write_text(
        json.dumps(
            {
                "profiled_ranks": sorted(discovered_ranks),
                "profiled_rank_count": len(discovered_ranks),
                "operator_identity_count": len(rows),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    json_temporary.replace(json_path)
    return csv_path, json_path


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract training profiler reports")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace", type=Path, help="one rank-N.json[.gz] trace")
    mode.add_argument("--trace-dir", type=Path, help="directory of per-rank traces")
    mode.add_argument(
        "--aggregate-profile-dir",
        type=Path,
        help="aggregate already extracted per-rank CSV reports",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank", type=int, help="rank override for --trace")
    args = parser.parse_args()
    if args.aggregate_profile_dir is not None:
        if args.output_dir is not None or args.rank is not None:
            parser.error("--output-dir/--rank cannot be used with --aggregate-profile-dir")
        paths = export_cross_rank_reports(args.aggregate_profile_dir)
    elif args.trace is not None:
        paths = export_kernel_reports_from_trace(
            args.trace, args.output_dir, rank=args.rank
        )
    else:
        if args.rank is not None:
            parser.error("--rank can only be used with --trace")
        per_rank_paths, cross_rank_paths = export_kernel_reports_from_trace_dir(
            args.trace_dir, args.output_dir
        )
        paths = [path for rank_paths in per_rank_paths for path in rank_paths]
        paths.extend(cross_rank_paths)
    for path in paths:
        print(f"profiler report is saved to: {path}")


if __name__ == "__main__":
    _main()
