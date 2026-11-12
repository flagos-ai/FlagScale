# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""FP32 software decoding of HF E4M3 and MXFP4 weights, independent of training."""

import torch

FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def e4m3_values():
    values = []
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        magnitude = (
            mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        )
        if exponent == 15 and mantissa == 7:
            magnitude = float("nan")
        values.append(-magnitude if byte & 128 else magnitude)
    return values


def decode_quantized_matrix(payload, scale, kind, shape, dtype, *, device="cpu"):
    """Decode into CPU model storage with at most 128 rows of FP32 work space.

    The caller validates the physical layout and supplies positive FP32 scales:
    FP8 uses 128x128 blocks; MXFP4 packs the low nibble first in groups of 32.
    Only the converter decodes these encodings; output checkpoints contain ordinary tensors.
    """
    if kind not in {"fp8", "mxfp4"}:
        raise ValueError(f"Unsupported quantized matrix encoding: {kind}")
    rows, cols = shape[-2:]
    result = torch.empty((rows, cols), dtype=dtype, device="cpu")
    lut = torch.tensor(
        FP4_VALUES if kind == "mxfp4" else e4m3_values(), dtype=torch.float32, device=device
    )
    for row in range(0, rows, 128):
        data = payload[row : row + 128].to(device)
        if kind == "mxfp4":
            decoded = torch.stack((lut[(data & 15).long()], lut[(data >> 4).long()]), -1)
            values = decoded.reshape(data.shape[0], -1, 32) * scale[row : row + 128, :, None]
            values = values.reshape(data.shape[0], cols)
        else:
            multiplier = scale[row // 128].repeat_interleave(128)[:cols]
            values = lut[data.long()] * multiplier
        converted = values.to(dtype)
        if not bool(torch.isfinite(converted).all()):
            raise ValueError("Non-finite/overflowing decoded checkpoint weight")
        result[row : row + data.shape[0]].copy_(converted.cpu())
    return result.reshape(shape)
