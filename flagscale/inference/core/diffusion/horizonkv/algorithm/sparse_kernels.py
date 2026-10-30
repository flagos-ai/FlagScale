# Modified for the HorizonKV inference artifact; see NOTICE.
"""Compacted RoPE kernels adapted from LightForcing."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

_LF_GRID_DEVICE_CACHE: dict[tuple[str, int, tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
_LF_GRID_CPU_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}
_LF_GRID_DEVICE_READY: dict[
    tuple[str, int, tuple[int, ...], tuple[int, ...]], tuple[torch.cuda.Event, int]
] = {}


def _lf_grid_sizes_on_device(grid_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not isinstance(grid_sizes, torch.Tensor):
        raise TypeError("grid_sizes must be a tensor")
    device = torch.device(device)
    if (
        grid_sizes.device == device
        and grid_sizes.dtype == torch.int64
        and grid_sizes.is_contiguous()
    ):
        return grid_sizes
    if grid_sizes.device.type != "cpu":
        return grid_sizes.to(device=device, dtype=torch.int64).contiguous()
    values = tuple((int(v) for v in grid_sizes.detach().reshape(-1).tolist()))
    shape = tuple((int(v) for v in grid_sizes.shape))
    device_index = int(device.index) if device.index is not None else -1
    key = (str(device.type), device_index, shape, values)
    cached = _LF_GRID_DEVICE_CACHE.get(key)
    if (
        isinstance(cached, torch.Tensor)
        and cached.device == device
        and (cached.dtype == torch.int64)
        and (tuple(cached.shape) == tuple(grid_sizes.shape))
    ):
        ready = _LF_GRID_DEVICE_READY.get(key)
        if ready is not None and torch.cuda.is_available():
            current_stream = torch.cuda.current_stream(device)
            producer_stream = int(ready[1])
            if int(current_stream.cuda_stream) != producer_stream:
                current_stream.wait_event(ready[0])
        return cached
    source = grid_sizes
    if torch.cuda.is_available():
        cpu_key = (shape, values)
        cached_cpu = _LF_GRID_CPU_CACHE.get(cpu_key)
        if (
            isinstance(cached_cpu, torch.Tensor)
            and cached_cpu.device.type == "cpu"
            and (cached_cpu.dtype == torch.int64)
            and (tuple(cached_cpu.shape) == shape)
        ):
            source = cached_cpu
        else:
            try:
                if not source.is_pinned():
                    source = source.pin_memory()
            except Exception:
                source = grid_sizes
            if source.device.type == "cpu" and source.dtype == torch.int64 and source.is_pinned():
                _LF_GRID_CPU_CACHE[cpu_key] = source
    uploaded = source.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    _LF_GRID_DEVICE_CACHE[key] = uploaded
    if torch.cuda.is_available() and device.type == "cuda":
        producer_stream = torch.cuda.current_stream(device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)
        _LF_GRID_DEVICE_READY[key] = (ready_event, int(producer_stream.cuda_stream))
    return uploaded


def _rope_reference(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0
):
    n, c = (x.size(2), x.size(3) // 2)
    split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = int(f * h * w)
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                split_freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                split_freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                split_freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]], dim=0))
    return torch.stack(output).type_as(x)


if triton is not None:

    @triton.jit
    def _causal_rope_apply_kernel(
        OUT,
        X,
        FREQS_REAL,
        GRID,
        NUM_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SEQ_STRIDE: tl.constexpr,
        T_DIM: tl.constexpr,
        H_DIM: tl.constexpr,
        START_FRAME: tl.constexpr,
        CONJUGATE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_MAJOR: tl.constexpr = False,
        SELECTED_FRAMES: tl.constexpr = 0,
    ):
        row_idx = tl.program_id(0)
        batch_idx = row_idx // (SEQ_STRIDE * NUM_HEADS)
        seq_head_idx = row_idx - batch_idx * SEQ_STRIDE * NUM_HEADS
        token_idx = seq_head_idx % SEQ_STRIDE if HEAD_MAJOR else seq_head_idx // NUM_HEADS
        if SELECTED_FRAMES > 0:
            f = SELECTED_FRAMES
        else:
            f = tl.load(GRID + batch_idx * 3)
        h = tl.load(GRID + batch_idx * 3 + 1)
        w = tl.load(GRID + batch_idx * 3 + 2)
        valid_seq_len = f * h * w
        half_head_dim: tl.constexpr = HEAD_DIM // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < half_head_dim
        x_row = X + row_idx * HEAD_DIM
        out_row = OUT + row_idx * HEAD_DIM
        x_real = tl.load(x_row + col_offsets * 2, mask=mask, other=0.0)
        x_imag = tl.load(x_row + col_offsets * 2 + 1, mask=mask, other=0.0)
        is_valid = token_idx < valid_seq_len
        hw = h * w
        t = token_idx // hw
        rem = token_idx - t * hw
        y = rem // w
        z = rem - y * w
        freq_row = tl.where(
            col_offsets < T_DIM, START_FRAME + t, tl.where(col_offsets < T_DIM + H_DIM, y, z)
        )
        freq_offset = (freq_row * half_head_dim + col_offsets) * 2
        load_mask = mask & is_valid
        cos_vals = tl.load(FREQS_REAL + freq_offset, mask=load_mask, other=1.0)
        sin_vals = tl.load(FREQS_REAL + freq_offset + 1, mask=load_mask, other=0.0)
        if CONJUGATE:
            sin_vals = -sin_vals
        xr = x_real.to(tl.float32)
        xi = x_imag.to(tl.float32)
        rr = xr * cos_vals.to(tl.float32) - xi * sin_vals.to(tl.float32)
        ri = xr * sin_vals.to(tl.float32) + xi * cos_vals.to(tl.float32)
        tl.store(out_row + col_offsets * 2, tl.where(is_valid, rr, xr), mask=mask)
        tl.store(out_row + col_offsets * 2 + 1, tl.where(is_valid, ri, xi), mask=mask)


def compacted_rope_apply(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0
) -> torch.Tensor:
    if triton is None or not x.is_cuda or x.shape[-1] % 2:
        return _rope_reference(x, grid_sizes, freqs, start_frame=start_frame)
    b, seq_len, heads, head_dim = x.shape
    c = head_dim // 2
    t_dim = c - 2 * (c // 3)
    h_dim = c // 3
    x_contig = x.contiguous()
    out = torch.empty_like(x_contig)
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device))
    block_size = triton.next_power_of_2(c)
    _causal_rope_apply_kernel[b * seq_len * heads,](
        out,
        x_contig,
        freqs_real,
        grid,
        heads,
        head_dim,
        seq_len,
        t_dim,
        h_dim,
        int(start_frame),
        False,
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=2,
    )
    return out.type_as(x)


def _rope_apply_packed_rows_reference(
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    row_lens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int,
    inplace: bool,
) -> torch.Tensor:
    if x.ndim == 4:
        rows, max_tokens, heads, head_dim = (int(v) for v in x.shape)
        packed = False
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens is only valid for packed row tensors")
    elif x.ndim == 3:
        total_tokens, heads, head_dim = (int(v) for v in x.shape)
        rows = int(frame_positions.shape[0])
        packed = True
        if cu_seqlens is None:
            raise ValueError("packed row RoPE requires cu_seqlens")
    else:
        raise ValueError("packed row RoPE expects a 3D or 4D tensor")
    if heads <= 0 or head_dim <= 0 or head_dim % 2:
        return x
    if frame_positions.ndim != 2 or int(frame_positions.shape[0]) != rows:
        raise ValueError("frame_positions shape does not match compact rows")
    if row_lens.ndim != 1 or int(row_lens.numel()) != rows:
        raise ValueError("row_lens shape does not match compact rows")
    if int(grid_sizes.ndim) != 2 or int(grid_sizes.shape[1]) < 3:
        raise ValueError("grid_sizes must have shape [B,3]")
    height = int(grid_sizes[0, 1].item())
    width = int(grid_sizes[0, 2].item())
    if height <= 0 or width <= 0:
        return x
    frame_seqlen = int(frame_seqlen)
    if frame_seqlen <= 0:
        return x
    c = head_dim // 2
    t_dim = c - 2 * (c // 3)
    h_dim = c // 3
    split_freqs = freqs.split([t_dim, h_dim, h_dim], dim=1)
    out = x if inplace else x.clone()

    def apply_one(row: int, src: torch.Tensor, dst: torch.Tensor) -> None:
        valid_tokens = int(row_lens[row].item())
        valid_tokens = max(0, min(valid_tokens, int(src.shape[0])))
        if valid_tokens <= 0:
            return
        frame_count = (valid_tokens + frame_seqlen - 1) // frame_seqlen
        positions = frame_positions[row, :frame_count].to(device=src.device, dtype=torch.long)
        token_idx = torch.arange(valid_tokens, device=src.device, dtype=torch.long)
        frame_idx = token_idx // frame_seqlen
        temporal = positions.index_select(0, frame_idx) + int(start_frame)
        in_frame = token_idx - frame_idx * frame_seqlen
        y = in_frame // width
        z = in_frame - y * width
        freq_rows = torch.cat(
            [
                split_freqs[0].index_select(0, temporal),
                split_freqs[1].index_select(0, y),
                split_freqs[2].index_select(0, z),
            ],
            dim=1,
        )
        values = torch.view_as_complex(
            src[:valid_tokens].to(torch.float64).reshape(valid_tokens, heads, -1, 2)
        )
        values = torch.view_as_real(values * freq_rows.view(valid_tokens, 1, -1)).flatten(2)
        dst[:valid_tokens].copy_(values.type_as(dst))

    if packed:
        cu_cpu = cu_seqlens.detach().to(device="cpu", dtype=torch.long)
        for row in range(rows):
            start = int(cu_cpu[row].item())
            end = int(cu_cpu[row + 1].item())
            apply_one(row, x[start:end], out[start:end])
    else:
        for row in range(rows):
            apply_one(row, x[row], out[row])
    return out.type_as(x)


if triton is not None:

    @triton.jit
    def _packed_rows_rope_apply_kernel(
        OUT,
        X,
        FRAME_POSITIONS,
        ROW_LENS,
        CU_SEQLENS,
        GRID,
        FREQS_REAL,
        MAX_TOKENS: tl.constexpr,
        MAX_FRAMES: tl.constexpr,
        FRAME_SEQLEN: tl.constexpr,
        START_FRAME: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        T_DIM: tl.constexpr,
        H_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        PACKED: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        row = pid // MAX_TOKENS
        token = pid - row * MAX_TOKENS
        row_len = tl.load(ROW_LENS + row).to(tl.int32)
        valid = token < row_len
        if PACKED:
            token_offset = tl.load(CU_SEQLENS + row).to(tl.int64) + token
        else:
            token_offset = row * MAX_TOKENS + token
        half_dim: tl.constexpr = HEAD_DIM // 2
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_mask = col_offsets < half_dim
        x_ptr = X + token_offset * HEAD_DIM
        out_ptr = OUT + token_offset * HEAD_DIM
        if PACKED:
            data_mask = col_mask & valid
        else:
            data_mask = col_mask
        x_real = tl.load(x_ptr + col_offsets * 2, mask=data_mask, other=0.0)
        x_imag = tl.load(x_ptr + col_offsets * 2 + 1, mask=data_mask, other=0.0)
        frame_idx = token // FRAME_SEQLEN
        frame_pos = (
            tl.load(
                FRAME_POSITIONS + row * MAX_FRAMES + frame_idx,
                mask=valid & (frame_idx < MAX_FRAMES),
                other=0,
            ).to(tl.int64)
            + START_FRAME
        )
        in_frame = token - frame_idx * FRAME_SEQLEN
        width = tl.load(GRID + 2).to(tl.int32)
        y = in_frame // width
        z = in_frame - y * width
        freq_row = tl.where(
            col_offsets < T_DIM, frame_pos, tl.where(col_offsets < T_DIM + H_DIM, y, z)
        )
        freq_offset = (freq_row * half_dim + col_offsets) * 2
        load_mask = col_mask & valid
        cos_vals = tl.load(FREQS_REAL + freq_offset, mask=load_mask, other=1.0)
        sin_vals = tl.load(FREQS_REAL + freq_offset + 1, mask=load_mask, other=0.0)
        xr = x_real.to(tl.float32)
        xi = x_imag.to(tl.float32)
        rr = xr * cos_vals.to(tl.float32) - xi * sin_vals.to(tl.float32)
        ri = xr * sin_vals.to(tl.float32) + xi * cos_vals.to(tl.float32)
        tl.store(out_ptr + col_offsets * 2, tl.where(valid, rr, xr), mask=data_mask)
        tl.store(out_ptr + col_offsets * 2 + 1, tl.where(valid, ri, xi), mask=data_mask)


def compacted_rope_apply_packed_rows(
    x: torch.Tensor,
    frame_positions: torch.Tensor,
    row_lens: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    *,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
    inplace: bool = False,
) -> torch.Tensor | None:
    if not isinstance(x, torch.Tensor) or x.ndim not in (3, 4):
        return None
    if not isinstance(frame_positions, torch.Tensor) or frame_positions.ndim != 2:
        return None
    if not isinstance(row_lens, torch.Tensor) or row_lens.ndim != 1:
        return None
    if not isinstance(grid_sizes, torch.Tensor) or not isinstance(freqs, torch.Tensor):
        return None
    if x.shape[-1] % 2 or int(x.shape[-2]) != 1:
        return None
    rows = int(x.shape[0]) if x.ndim == 4 else int(row_lens.numel())
    if int(frame_positions.shape[0]) != rows or int(row_lens.numel()) != rows:
        return None
    if x.ndim == 3 and (
        not isinstance(cu_seqlens, torch.Tensor)
        or cu_seqlens.ndim != 1
        or int(cu_seqlens.numel()) != rows + 1
    ):
        return None
    if int(frame_seqlen) <= 0:
        return None
    if frame_positions.device.type == "cpu" and torch.cuda.is_available():
        try:
            if not frame_positions.is_pinned():
                frame_positions = frame_positions.pin_memory()
        except Exception:
            pass
    frame_positions = frame_positions.to(
        device=x.device, dtype=torch.long, non_blocking=True
    ).contiguous()
    row_lens = row_lens.to(device=x.device, dtype=torch.int32).contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(device=x.device, dtype=torch.int32).contiguous()
    max_frames = int(frame_positions.shape[1])
    max_tokens = int(x.shape[1]) if x.ndim == 4 else max(1, max_frames * int(frame_seqlen))
    if max_frames <= 0 or max_tokens <= 0:
        return None
    inplace = bool(inplace and x.is_contiguous())
    if triton is None or not x.is_cuda:
        return _rope_apply_packed_rows_reference(
            x,
            frame_positions,
            row_lens,
            cu_seqlens,
            int(frame_seqlen),
            grid_sizes,
            freqs,
            int(start_frame),
            inplace,
        )
    head_dim = int(x.shape[-1])
    half_dim = head_dim // 2
    t_dim = half_dim - 2 * (half_dim // 3)
    h_dim = half_dim // 3
    block_size = triton.next_power_of_2(half_dim)
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device)).contiguous()
    out = x if inplace else torch.empty_like(x)
    _packed_rows_rope_apply_kernel[rows * max_tokens,](
        out,
        x.contiguous(),
        frame_positions,
        row_lens,
        cu_seqlens if cu_seqlens is not None else frame_positions,
        grid,
        freqs_real,
        MAX_TOKENS=max_tokens,
        MAX_FRAMES=max_frames,
        FRAME_SEQLEN=int(frame_seqlen),
        START_FRAME=int(start_frame),
        HEAD_DIM=head_dim,
        T_DIM=t_dim,
        H_DIM=h_dim,
        BLOCK_SIZE=block_size,
        PACKED=x.ndim == 3,
        num_warps=1,
        num_stages=2,
    )
    return out.type_as(x)


def compacted_rope_apply_compact(
    x: torch.Tensor,
    *,
    num_heads: int,
    selected_frames: int,
    frame_seqlen: int,
    grid_sizes: torch.Tensor,
    freqs: torch.Tensor,
    start_frame: int = 0,
) -> torch.Tensor | None:
    if (
        triton is None
        or not x.is_cuda
        or (not x.is_contiguous())
        or (x.ndim != 4)
        or (int(x.shape[2]) != 1)
        or int(x.shape[-1]) % 2
        or (int(num_heads) <= 0)
        or int(x.shape[0]) % int(num_heads)
        or (int(selected_frames) <= 0)
        or (int(frame_seqlen) <= 0)
        or (int(x.shape[1]) != int(selected_frames) * int(frame_seqlen))
    ):
        return None
    rows, tokens, _, head_dim = x.shape
    half_dim = head_dim // 2
    grid = _lf_grid_sizes_on_device(grid_sizes, x.device)
    freqs_real = torch.view_as_real(freqs.to(device=x.device)).contiguous()
    out = torch.empty_like(x)
    _causal_rope_apply_kernel[rows * tokens,](
        out,
        x,
        freqs_real,
        grid,
        int(num_heads),
        head_dim,
        tokens,
        half_dim - 2 * (half_dim // 3),
        half_dim // 3,
        int(start_frame),
        False,
        BLOCK_SIZE=triton.next_power_of_2(half_dim),
        HEAD_MAJOR=True,
        SELECTED_FRAMES=int(selected_frames),
        num_warps=1,
        num_stages=2,
    )
    return out
