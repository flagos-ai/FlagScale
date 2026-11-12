# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Tensor descriptors, precision policies and FP8/MXFP4 storage conversion."""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .codecs import decode_quantized_matrix

if TYPE_CHECKING:
    from .hf import SafetensorReader

DENSE_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
HEADER_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN"}
E8M0_DTYPES = {"F8_E8M0", "F8_E8M0FNU"}
SCALE_SUFFIXES = (".weight_scale_inv", ".scale")


class PrecisionError(ValueError):
    pass


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    kind: str = "dense"
    scale: str | None = None
    scale_dtype: str | None = None


@dataclass(frozen=True)
class PrecisionPolicy:
    # dtype is a config default / explicit override, not every tensor's dtype.
    dtype: str
    device: str
    reason: str
    preserve_source_dtypes: bool = True
    warnings: tuple[str, ...] = ()
    source_dtype_counts: tuple[tuple[str, int], ...] = ()
    source_quantized_kinds: tuple[str, ...] = ()

    def report(self) -> dict:
        return dict(
            dtype=self.dtype,
            device=self.device,
            dtype_scope="config default / quantized-input decode dtype; auto preserves individual dense dtypes",
            preserve_source_dtypes=self.preserve_source_dtypes,
            reason=self.reason,
            warnings=list(self.warnings),
            source_dtype_counts=dict(self.source_dtype_counts),
            source_quantized_kinds=list(self.source_quantized_kinds),
            storage_mode="dense",
            scale_handling=(
                "decode_to_dense_dcp" if self.source_quantized_kinds else "no_quantization"
            ),
        )


def validate_quantization_config(config: Mapping[str, Any]) -> None:
    quant = config.get("quantization_config")
    if quant is not None:
        if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
            raise PrecisionError("Only the fp8 quantization_config is supported")
        if quant.get("fmt", "e4m3") != "e4m3":
            raise PrecisionError("FP8 weights must use E4M3")
        if list(quant.get("weight_block_size", (128, 128))) != [128, 128]:
            raise PrecisionError("FP8 weights require weight_block_size=[128,128]")
        if quant.get("scale_fmt") not in (None, "ue8m0"):
            raise PrecisionError("Unsupported scale_fmt")
        excluded = quant.get("modules_to_not_convert")
        if excluded is not None and (
            not isinstance(excluded, list)
            or any(not isinstance(name, str) or not name for name in excluded)
        ):
            raise PrecisionError("modules_to_not_convert must be a list of nonempty module names")
    if config.get("expert_dtype") not in (
        None,
        "fp4",
        "fp8",
        "bf16",
        "bfloat16",
        "fp16",
        "float16",
        "fp32",
        "float32",
    ):
        raise PrecisionError("Unsupported expert_dtype (MXFP4 is distinct from NVFP4/int8)")


class PrecisionSource:
    """Compile physical tensors and scales into logical, dense mapping inputs."""

    def __init__(
        self,
        reader: SafetensorReader,
        config: Mapping[str, Any],
        *,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        normalize=None,
        skip=None,
        mxfp4=None,
    ):
        normalize = normalize or (lambda name: name)
        mxfp4 = mxfp4 or (lambda name: False)
        validate_quantization_config(config)
        self.reader = reader
        self.dtype = dtype
        self.device = device
        self.specs: dict[str, TensorSpec] = {}
        self.consumed: set[str] = set()
        dtypes, shapes = reader.tensor_dtypes(), reader.tensor_shapes()
        self.physical_keys = {name for name in reader.keys if not (skip and skip(name))}
        candidates = {name for name in self.physical_keys if name.endswith(SCALE_SUFFIXES)}
        # A normal dense parameter may legitimately be named *.scale. Only
        # quantized payloads claim companion scales in the generic reader.
        scales = {
            name.removesuffix(".weight") + suffix
            for name in self.physical_keys
            if dtypes[name] in FP8_DTYPES or (dtypes[name] in {"I8", "U8"} and mxfp4(name))
            for suffix in SCALE_SUFFIXES
            if name.removesuffix(".weight") + suffix in candidates
        }
        paired_scales = set()
        quant = config.get("quantization_config") or {}
        for name in sorted(self.physical_keys - scales):
            dtype_code, shape = dtypes[name], shapes[name]
            canonical = normalize(name)
            if canonical in self.specs:
                raise PrecisionError(f"Duplicate HF semantic key: {canonical}")
            kind, scale_name = "dense", None
            if dtype_code in FP8_DTYPES:
                kind = "fp8"
                if not quant:
                    raise PrecisionError(f"FP8 weight requires quantization_config: {name}")
            elif dtype_code in {"I8", "U8"}:
                if not mxfp4(name):
                    raise PrecisionError(
                        f"Unsupported HF tensor {name} ({dtype_code}); not declared MXFP4"
                    )
                kind = "mxfp4"
                if len(shape) != 2:
                    raise PrecisionError(f"Packed MXFP4 weight must be a matrix: {name}")
                shape = (*shape[:-1], shape[-1] * 2)
            elif dtype_code not in {*HEADER_DTYPES, "I32", "I64"}:
                raise PrecisionError(f"Unsupported HF tensor {name} ({dtype_code})")
            if kind != "dense":
                if len(shape) != 2 or not all(shape) or not name.endswith(".weight"):
                    raise PrecisionError(
                        f"Quantized linear weight must be a nonempty matrix: {name}"
                    )
                candidates = [name.removesuffix(".weight") + suffix for suffix in SCALE_SUFFIXES]
                matches = [candidate for candidate in candidates if candidate in scales]
                if len(matches) != 1:
                    raise PrecisionError(
                        f"Expected exactly one scale tensor for {name}; found {matches}"
                    )
                scale_name = matches[0]
                scale_dtype = dtypes[scale_name]
                if scale_dtype not in {*HEADER_DTYPES, *E8M0_DTYPES, "U8"}:
                    raise PrecisionError(f"Unsupported scale dtype for {scale_name}: {scale_dtype}")
                if scale_dtype == "U8" and quant.get("scale_fmt") != "ue8m0":
                    raise PrecisionError(f"Byte scale requires scale_fmt=ue8m0: {scale_name}")
                rows, cols = shape
                expected = (math.ceil(rows / 128), math.ceil(cols / 128))
                if kind == "mxfp4":
                    if cols % 32:
                        raise PrecisionError(
                            f"MXFP4 reduction dimension must be divisible by 32: {name}"
                        )
                    expected = (rows, cols // 32)
                if shapes[scale_name] != expected:
                    raise PrecisionError(
                        f"Scale shape mismatch: {scale_name} {shapes[scale_name]} != {expected}"
                    )
                paired_scales.add(scale_name)
            self.specs[canonical] = TensorSpec(
                name,
                shape,
                dtype_code,
                kind,
                scale_name,
                dtypes[scale_name] if scale_name else None,
            )
        if scales != paired_scales:
            raise PrecisionError(f"Orphan scale tensors: {sorted(scales - paired_scales)[:20]}")

    @property
    def keys(self) -> set[str]:
        return set(self.specs)

    @property
    def low_precision_kinds(self) -> set[str]:
        return {spec.kind for spec in self.specs.values() if spec.kind != "dense"}

    def _scale(self, spec: TensorSpec) -> torch.Tensor:
        if spec.scale_dtype in E8M0_DTYPES or spec.scale_dtype == "U8":
            encoded = self.reader.get_bytes(spec.scale).to(self.device)
            if bool((encoded == 255).any()):
                raise PrecisionError(f"Invalid E8M0 NaN scale: {spec.scale}")
            result = torch.exp2(encoded.to(torch.float32) - 127)
        else:
            result = self.reader.get_tensor(spec.scale).to(self.device, dtype=torch.float32)
        if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
            raise PrecisionError(f"Scale must be finite and positive: {spec.scale}")
        return result

    def get_tensor(self, name: str) -> torch.Tensor:
        spec = self.specs[name]
        if spec.kind == "dense":
            result = self.reader.get_tensor(spec.name)
        else:
            packed = self.reader.get_bytes(spec.name)
            scale = self._scale(spec)
            try:
                result = decode_quantized_matrix(
                    packed, scale, spec.kind, spec.shape, self.dtype, device=self.device
                )
            except ValueError as error:
                raise PrecisionError(
                    f"Non-finite/overflowing dequantized weight: {spec.name}"
                ) from error
            self.consumed.add(spec.scale)
        self.consumed.add(spec.name)
        return result

    def validate_consumed(self) -> None:
        remaining = self.physical_keys - self.consumed
        if remaining:
            raise PrecisionError(f"Unconsumed physical HF tensors: {sorted(remaining)[:20]}")


def resolve_precision(
    source,
    config: Mapping[str, Any],
    requested: str,
    device: str,
    *,
    to_mcore: bool,
    storage_mode: str = "dense",
) -> PrecisionPolicy:
    """Decode quantized sources in FP32 unless a lower precision is explicitly requested."""
    if storage_mode != "dense":
        raise PrecisionError(
            "Only ordinary tensor DCP is supported; preserved wrappers require training adapters"
        )
    if requested not in {"auto", *DENSE_DTYPES}:
        raise PrecisionError(f"Unsupported dtype: {requested}")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise PrecisionError("Conversion arithmetic supports --device cpu or cuda[:N]")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            device = "cpu"
        elif selected.index is not None and selected.index >= torch.cuda.device_count():
            raise PrecisionError(f"CUDA device does not exist: {device}")
    kinds = source.low_precision_kinds
    if kinds - {"fp8", "mxfp4"}:
        raise PrecisionError(f"Unsupported source quantization kinds: {sorted(kinds)}")
    if kinds and not to_mcore:
        raise PrecisionError("MCore export requires ordinary tensor storage")
    # Only the config default follows the largest floating storage population.
    # Auto retains each dense tensor's dtype and decodes quantized imports to FP32.
    populations = Counter()
    dtype_names = {"BF16": "bf16", "F16": "fp16", "F32": "fp32"}
    for spec in source.specs.values():
        code = "fp32" if spec.kind != "dense" else dtype_names.get(spec.dtype)
        if code:
            populations[code] += math.prod(spec.shape)
    dtype = (
        requested if requested != "auto" else max(populations, key=populations.get, default="bf16")
    )
    if to_mcore and kinds and requested == "auto":
        dtype = "fp32"
    notes = []
    declared = bool(config.get("quantization_config")) or config.get("expert_dtype") in {
        "fp8",
        "fp4",
    }
    if declared and not kinds:
        label = "HF source config" if to_mcore else "HF reference config"
        notes.append(
            f"{label} declares quantization, but actual source weights are dense; "
            "ignoring that recipe. No quantization or scale tensors will be generated."
        )
    reason = (
        "Preserve actual source tensor dtypes with auto; decode quantized inputs to dense tensors"
    )
    if requested != "auto":
        reason = "Explicit dense dtype override; FP32 and integer tensors remain unchanged"
    return PrecisionPolicy(
        dtype=dtype,
        device=device,
        reason=reason,
        preserve_source_dtypes=requested == "auto",
        warnings=tuple(notes),
        source_dtype_counts=tuple(
            sorted(Counter(spec.dtype for spec in source.specs.values()).items())
        ),
        source_quantized_kinds=tuple(sorted(kinds)),
    )


def output_config(config: Mapping[str, Any], policy: PrecisionPolicy) -> dict:
    result = copy.deepcopy(dict(config))
    dtype = str(DENSE_DTYPES[policy.dtype]).removeprefix("torch.")
    result["dtype" if "dtype" in result else "torch_dtype"] = dtype
    if "dtype" in result and "torch_dtype" in result:
        result["torch_dtype"] = dtype
    # Reference-only precision declarations must never survive a dense export.
    for field in ("quantization_config", "scale_fmt", "scale_dtype"):
        result.pop(field, None)
    if "expert_dtype" in result:
        result["expert_dtype"] = policy.dtype
    return result


def encode_hf_tensors(tensors, source, policy: PrecisionPolicy):
    seen = set()
    for name, tensor in tensors:
        if name in seen:
            raise PrecisionError(f"Duplicate logical HF output: {name}")
        seen.add(name)
        spec = source.specs.get(name)
        if spec is None:
            raise PrecisionError(f"HF output absent from mapped schema: {name}")
        if tuple(tensor.shape) != spec.shape:
            raise PrecisionError(
                f"HF output shape mismatch: {name} {tuple(tensor.shape)} != {spec.shape}"
            )
        if tensor.is_floating_point():
            if tensor.dtype not in HEADER_DTYPES.values():
                raise PrecisionError(f"Unsupported MCore storage dtype: {name}: {tensor.dtype}")
            if not policy.preserve_source_dtypes and tensor.dtype != torch.float32:
                tensor = tensor.to(DENSE_DTYPES[policy.dtype])
            if not bool(torch.isfinite(tensor).all()):
                raise PrecisionError(f"Non-finite/overflowing HF output: {name}")
        elif tensor.dtype not in {torch.int32, torch.int64}:
            raise PrecisionError(f"Unsupported integer HF output: {name}: {tensor.dtype}")
        yield spec.name, tensor
    if seen != source.keys:
        raise PrecisionError(f"Unwritten HF targets: {sorted(source.keys - seen)[:20]}")
