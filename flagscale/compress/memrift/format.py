"""
MemRift compressed weight format utilities.

Format: split_zstd
- index.json: List of weight entries with metadata
- *.bin files: Binary data with format:
    - <Q numel> (8 bytes little-endian uint64)
    - sm_bytes (sign+mantissa, 1 byte/elem for bf16, 3 bytes/elem for fp32)
    - exp_zstd_bytes (zstd-compressed exponent bytes)

Supported dtypes: bfloat16, float32
"""

import json
import os
import struct
from typing import Any

import numpy as np
import torch

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


def load_index(comp_dir: str) -> list[dict[str, Any]]:
    """
    Load the index.json file from a compressed weight directory.

    Args:
        comp_dir: Path to compressed weight directory

    Returns:
        List of weight entries, each containing:
        - name: Original parameter name
        - file: Binary file name
        - shape: Original tensor shape
        - dtype: Data type string
        - scheme: Compression scheme (split_zstd or raw_torch)
    """
    index_path = os.path.join(comp_dir, "index.json")
    with open(index_path, "r") as f:
        return json.load(f)


def read_compressed_weight(
    comp_dir: str,
    entry: dict[str, Any],
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, bytes, torch.dtype]:
    """
    Read a single compressed weight from disk.

    Args:
        comp_dir: Path to compressed weight directory
        entry: Weight entry from index.json
        device: Device to place sm_gpu tensor (only for split_zstd scheme)

    Returns:
        For split_zstd scheme:
            (sm_gpu, exp_bytes, dtype): sm tensor on device, compressed exp bytes, dtype
        For raw_torch scheme:
            (tensor, None, dtype): Full tensor, None, dtype
    """
    file_path = os.path.join(comp_dir, entry["file"])
    dtype_str = entry["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

    if entry["scheme"] == "split_zstd":
        with open(file_path, "rb") as f:
            # Read numel as 8-byte uint64 (new format)
            numel_bytes = f.read(8)
            if len(numel_bytes) != 8:
                raise ValueError(f"Truncated numel header in {file_path}")
            numel = struct.unpack("<Q", numel_bytes)[0]
            expected_numel = int(np.prod(entry["shape"]))
            if numel != expected_numel:
                raise ValueError(
                    f"numel mismatch for {entry['name']}: file={numel}, index={expected_numel}"
                )

            # Calculate sm_bytes size
            sm_size = numel * (1 if dtype_str == "bfloat16" else 3)

            # Read sm bytes
            sm_payload = f.read(sm_size)
            if len(sm_payload) != sm_size:
                raise ValueError(f"Truncated sign/mantissa payload in {file_path}")
            sm_bytes = np.frombuffer(sm_payload, dtype=np.uint8)
            sm_gpu = torch.as_tensor(sm_bytes, dtype=torch.uint8, device=device)

            # Read remaining as compressed exponent
            exp_bytes = f.read()

        return sm_gpu, exp_bytes, dtype

    elif entry["scheme"] == "raw_torch":
        tensor = torch.load(file_path, map_location=device)
        return tensor, None, dtype
    else:
        raise ValueError(f"Unknown scheme: {entry['scheme']}")


def write_compressed_weight(
    out_path: str,
    sm_cpu: np.ndarray,
    exp_compressed: bytes,
    numel: int,
) -> None:
    """
    Write a single compressed weight to disk.

    Args:
        out_path: Output file path
        sm_cpu: Sign+mantissa bytes as numpy array
        exp_compressed: Zstd-compressed exponent bytes
        numel: Number of elements in original tensor
    """
    with open(out_path, "wb") as f:
        # Write numel as 8-byte uint64
        f.write(struct.pack("<Q", numel))
        # Write sm bytes
        f.write(sm_cpu.tobytes())
        # Write compressed exponent
        f.write(exp_compressed)


def decompress_exponent(
    exp_bytes: bytes,
    numel: int,
) -> np.ndarray:
    """
    Decompress exponent bytes using zstd.

    Args:
        exp_bytes: Zstd-compressed exponent bytes
        numel: Expected number of elements

    Returns:
        Decompressed exponent as numpy uint8 array
    """
    if not ZSTD_AVAILABLE:
        raise RuntimeError("zstandard not installed. Install with: pip install zstandard")

    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(exp_bytes, max_output_size=numel)
    return np.frombuffer(decompressed, dtype=np.uint8)


def compress_exponent(
    exp_cpu: np.ndarray,
    level: int = 18,
) -> bytes:
    """
    Compress exponent bytes using zstd.

    Args:
        exp_cpu: Exponent bytes as numpy uint8 array
        level: Zstd compression level (1-22, default 18)

    Returns:
        Compressed bytes
    """
    if not ZSTD_AVAILABLE:
        raise RuntimeError("zstandard not installed. Install with: pip install zstandard")

    cctx = zstd.ZstdCompressor(level=level, write_checksum=False)
    return cctx.compress(exp_cpu.tobytes())


def reconstruct_bf16_cpu(
    sm_bytes: np.ndarray,
    exp_bytes: np.ndarray,
    shape: list[int],
) -> torch.Tensor:
    """
    Reconstruct bfloat16 tensor from split components (CPU-only, for verification).

    Args:
        sm_bytes: Sign+mantissa uint8 array
        exp_bytes: Exponent uint8 array
        shape: Target shape

    Returns:
        Reconstructed bfloat16 tensor
    """
    sm_u16 = sm_bytes.astype(np.uint16)
    sign_u16 = (sm_u16 & 0x80) << 8  # bit7 → bit15
    mant_u16 = sm_u16 & 0x7F
    exp_u16 = exp_bytes.astype(np.uint16) << 7

    recon_u16 = sign_u16 | exp_u16 | mant_u16
    recon_bf16 = torch.from_numpy(recon_u16).view(torch.bfloat16)
    return recon_bf16.view(shape)


def get_layer_name_from_param(param_name: str) -> str | None:
    """
    Extract layer name (e.g., 'model.layers.0') from parameter name.

    Args:
        param_name: Full parameter name like 'model.layers.0.self_attn.q_proj.weight'

    Returns:
        Layer prefix like 'model.layers.0', or None if not a layer parameter
    """
    parts = param_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                int(parts[i + 1])  # Check if next part is layer index
                return ".".join(parts[: i + 2])
            except ValueError:
                pass
    return None
