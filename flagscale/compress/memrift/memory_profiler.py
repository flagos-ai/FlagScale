"""
MemRift 训练显存分解 Profiler

在 FlagScale 训练中逐层追踪显存变化，区分:
- 权重物化/释放 (MemRift weight hooks)
- 激活累积/压缩 (activation compression)
- 梯度 + 临时缓冲

使用方式:
  在 train_hooks.py 的 inject_memrift_if_configured 中调用 install_memory_profiler()
  当 args.memrift_profile_memory == True 时自动启用
"""

from typing import Any

import torch

MB = 1024.0 * 1024.0


class MemRiftMemoryProfiler:
    """
    逐层显存追踪器。

    在 Transformer 每层的 forward_pre/post, backward_pre/post 记录
    torch.cuda.memory_allocated() 和 peak，生成训练过程的显存时间线。
    """

    def __init__(self, num_layers: int, profile_iters: int = 3, rank: int = 0):
        self.num_layers = num_layers
        self.profile_iters = profile_iters
        self.rank = rank
        self._handles = []
        self._current_iter = 0
        self._active = True
        self._model_ref = None  # 用于 _print_report 时统计 sm_gpu / LoRA 等

        # 每个 iteration 的记录: list of dicts
        # Each record: {event, layer_idx, allocated_mb, peak_mb}
        self._iter_records = []
        self._all_iters = []  # list of iter_records

    def _log(self, event: str, layer_idx: int):
        if not self._active:
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / MB
        peak = torch.cuda.max_memory_allocated() / MB
        self._iter_records.append(
            {
                "event": event,
                "layer": layer_idx,
                "allocated_mb": allocated,
                "peak_mb": peak,
            }
        )

    def on_iter_start(self):
        """在每轮 iteration 开始时调用（forward 之前）"""
        if not self._active:
            return
        self._iter_records = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / MB
        self._iter_records.append(
            {
                "event": "iter_start",
                "layer": -1,
                "allocated_mb": allocated,
                "peak_mb": allocated,
            }
        )

    def on_iter_end(self):
        """在每轮 iteration 结束时调用（backward + optimizer step 之后）"""
        if not self._active:
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / MB
        peak = torch.cuda.max_memory_allocated() / MB
        self._iter_records.append(
            {
                "event": "iter_end",
                "layer": -1,
                "allocated_mb": allocated,
                "peak_mb": peak,
            }
        )
        self._all_iters.append(self._iter_records)
        self._current_iter += 1
        if self._current_iter >= self.profile_iters:
            self._active = False
            self._print_report()

    def install(self, model):
        """在 decoder layers 上安装 forward/backward hooks"""
        if self._model_ref is None:
            self._model_ref = model
        decoder_layers = None
        unwrapped = model.module if hasattr(model, "module") else model

        if hasattr(unwrapped, "decoder") and hasattr(unwrapped.decoder, "layers"):
            decoder_layers = unwrapped.decoder.layers
        elif hasattr(unwrapped, "language_model") and hasattr(unwrapped.language_model, "decoder"):
            decoder_layers = unwrapped.language_model.decoder.layers

        if decoder_layers is None:
            if self.rank == 0:
                print("[MemProfiler] WARNING: 未找到 decoder.layers，跳过")
            return

        for i, layer in enumerate(decoder_layers):
            idx = i

            def make_fwd_pre(layer_idx):
                def hook(mod, inp):
                    self._log("fwd_pre", layer_idx)

                return hook

            def make_fwd_post(layer_idx):
                def hook(mod, inp, out):
                    self._log("fwd_post", layer_idx)

                return hook

            def make_bwd_pre(layer_idx):
                def hook(mod, grad_out):
                    self._log("bwd_pre", layer_idx)

                return hook

            def make_bwd_post(layer_idx):
                def hook(mod, grad_in, grad_out):
                    self._log("bwd_post", layer_idx)

                return hook

            # 使用 prepend=False 确保在 MemRift hooks 之后执行
            self._handles.append(layer.register_forward_pre_hook(make_fwd_pre(idx)))
            self._handles.append(layer.register_forward_hook(make_fwd_post(idx)))
            self._handles.append(layer.register_full_backward_pre_hook(make_bwd_pre(idx)))
            self._handles.append(layer.register_full_backward_hook(make_bwd_post(idx)))

        if self.rank == 0:
            print(
                f"[MemProfiler] 已安装 {len(decoder_layers)} 层 × 4 显存追踪 hooks，"
                f"将在前 {self.profile_iters} 轮 iteration 采集数据"
            )

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _print_report(self):
        """打印显存分解报告"""
        if self.rank != 0 or not self._all_iters:
            return

        # 使用最后一个 iteration 的数据（稳态）
        records = self._all_iters[-1]
        if not records:
            return

        iter_idx = len(self._all_iters) - 1
        print(f"\n{'=' * 78}")
        print(f"  MemRift + LoRA 显存分解报告（Iteration {iter_idx}）")
        print(f"{'=' * 78}")

        # 提取关键节点
        iter_start = next((r for r in records if r["event"] == "iter_start"), None)
        iter_end = next((r for r in records if r["event"] == "iter_end"), None)

        fwd_pres = [r for r in records if r["event"] == "fwd_pre"]
        fwd_posts = [r for r in records if r["event"] == "fwd_post"]
        bwd_pres = [r for r in records if r["event"] == "bwd_pre"]
        bwd_posts = [r for r in records if r["event"] == "bwd_post"]

        baseline = iter_start["allocated_mb"] if iter_start else 0
        overall_peak = iter_end["peak_mb"] if iter_end else 0

        # ── 逐层前向 ──
        print(f"\n{'─' * 78}")
        print("  逐层前向传播 (MemRift: 每层 物化权重 → 计算 → 释放权重 + 压缩激活)")
        print(f"{'─' * 78}")
        print(f"  迭代前稳态基线: {baseline:.1f} MB (LoRA + optimizer + sm_gpu 等)")
        print(
            f"  {'Layer':<8} {'fwd_pre(MB)':>12} {'fwd_post(MB)':>13} {'Δ(MB)':>10} {'Δ累积(MB)':>12}"
        )
        print(f"  {'─' * 8} {'─' * 12} {'─' * 13} {'─' * 10} {'─' * 12}")

        for pre, post in zip(fwd_pres, fwd_posts):
            delta = post["allocated_mb"] - pre["allocated_mb"]
            cumul = post["allocated_mb"] - baseline
            print(
                f"  L{pre['layer']:02d}    "
                f"{pre['allocated_mb']:>12.1f} "
                f"{post['allocated_mb']:>13.1f} "
                f"{delta:>+10.1f} "
                f"{cumul:>12.1f}"
            )

        fwd_end = fwd_posts[-1]["allocated_mb"] if fwd_posts else baseline
        fwd_activation_total = fwd_end - baseline
        print(f"\n  前向结束时: {fwd_end:.1f} MB")
        print(f"  前向激活总增量 = {fwd_activation_total:.1f} MB")

        # ── 逐层反向 ──
        print(f"\n{'─' * 78}")
        print("  逐层反向传播 (MemRift: 每层 物化权重 + 解压激活 → 计算梯度 → 释放)")
        print(f"{'─' * 78}")
        print(f"  {'Layer':<8} {'bwd_pre(MB)':>12} {'bwd_post(MB)':>13} {'Δ(MB)':>10}")
        print(f"  {'─' * 8} {'─' * 12} {'─' * 13} {'─' * 10}")

        for pre, post in zip(bwd_pres, bwd_posts):
            delta = post["allocated_mb"] - pre["allocated_mb"]
            print(
                f"  L{pre['layer']:02d}    "
                f"{pre['allocated_mb']:>12.1f} "
                f"{post['allocated_mb']:>13.1f} "
                f"{delta:>+10.1f}"
            )

        # ── 汇总 ──
        fwd_peak = max(r["allocated_mb"] for r in records if r["event"].startswith("fwd"))
        bwd_peak = max(r["allocated_mb"] for r in records if r["event"].startswith("bwd"))
        grad_and_misc = overall_peak - fwd_end  # backward 过程中超出 forward 结束的部分

        print(f"\n{'─' * 78}")
        print("  显存分解汇总")
        print(f"{'─' * 78}")
        print("  ┌───────────────────────────────────────────────────────────┐")
        print(f"  │ 稳态基线 (LoRA + optimizer + sm_gpu)    {baseline:>10.1f} MB │")
        print(f"  │ 前向激活增量 (压缩后残留 GPU)           {fwd_activation_total:>10.1f} MB │")
        print(f"  │ 反向临时 + 梯度 (peak - fwd_end)        {grad_and_misc:>10.1f} MB │")
        print("  ├───────────────────────────────────────────────────────────┤")
        print(f"  │ 前向过程 peak allocated                  {fwd_peak:>10.1f} MB │")
        print(f"  │ 反向过程 peak allocated                  {bwd_peak:>10.1f} MB │")
        print(f"  │ 整轮 peak (max_memory_allocated)         {overall_peak:>10.1f} MB │")
        print("  ├───────────────────────────────────────────────────────────┤")
        if overall_peak > 0:
            print(
                f"  │ 稳态基线 / 峰值                         {baseline / overall_peak * 100:>9.1f}% │"
            )
            print(
                f"  │ 前向激活 / 峰值                         {fwd_activation_total / overall_peak * 100:>9.1f}% │"
            )
            print(
                f"  │ 反向临时+梯度 / 峰值                   {grad_and_misc / overall_peak * 100:>9.1f}% │"
            )
        print("  └───────────────────────────────────────────────────────────┘")

        # ── 峰值组成（稳态分解：sm_gpu / LoRA / 优化器 / 梯度缓冲 等）──
        comp = self._estimate_peak_components(
            baseline, fwd_activation_total, grad_and_misc, overall_peak
        )
        print(f"\n{'─' * 78}")
        print("  峰值组成 (Peak composition)")
        print(f"{'─' * 78}")
        print("  ┌───────────────────────────────────────────────────────────┐")
        print(f"  │ 稳态基线 (总)                         {comp['baseline_mb']:>10.1f} MB │")
        if comp.get("sm_gpu_mb") is not None:
            print(f"  │   ├─ sm_gpu (常驻, sign+mantissa)        {comp['sm_gpu_mb']:>10.1f} MB │")
        if comp.get("lora_mb") is not None:
            print(f"  │   ├─ LoRA 参数                         {comp['lora_mb']:>10.1f} MB │")
        if comp.get("other_steady_mb") is not None:
            print(
                f"  │   └─ 其它 (优化器/梯度缓冲/embed 等)     {comp['other_steady_mb']:>10.1f} MB │"
            )
        print(f"  │ 前向激活增量                           {comp['fwd_activation_mb']:>10.1f} MB │")
        print(f"  │ 反向梯度 + 临时                        {comp['grad_misc_mb']:>10.1f} MB │")
        print("  ├───────────────────────────────────────────────────────────┤")
        print(f"  │ 整轮峰值                              {comp['peak_mb']:>10.1f} MB │")
        if comp["peak_mb"] > 0:
            print(
                f"  │ 占比: 稳态 {comp['baseline_mb'] / comp['peak_mb'] * 100:.0f}%  激活 {comp['fwd_activation_mb'] / comp['peak_mb'] * 100:.0f}%  反向 {comp['grad_misc_mb'] / comp['peak_mb'] * 100:.0f}% │"
            )
        print("  └───────────────────────────────────────────────────────────┘")
        if comp.get("exp_cpu_mb") is not None:
            print(f"  (exp 压缩在 CPU: {comp['exp_cpu_mb']:.1f} MB，不占 GPU 峰值)")

    def _estimate_peak_components(
        self,
        baseline: float,
        fwd_activation_total: float,
        grad_and_misc: float,
        overall_peak: float,
    ) -> dict[str, Any]:
        """从当前 model 估计稳态基线组成：sm_gpu、LoRA、其它；以及 exp CPU 占用。"""
        comp = {
            "baseline_mb": baseline,
            "fwd_activation_mb": fwd_activation_total,
            "grad_misc_mb": grad_and_misc,
            "peak_mb": overall_peak,
        }
        sm_gpu_mb = None
        lora_mb = None
        exp_cpu_mb = None

        if self._model_ref is not None:
            try:
                model = (
                    self._model_ref.module
                    if hasattr(self._model_ref, "module")
                    else self._model_ref
                )
                sm_gpu_bytes = 0
                lora_numel = 0
                exp_bytes = 0
                for name, param in model.named_parameters():
                    # CompressedParam: _sm_gpu on GPU, exp on CPU
                    if type(param).__name__ == "CompressedParam":
                        if getattr(param, "_sm_gpu", None) is not None and isinstance(
                            param._sm_gpu, torch.Tensor
                        ):
                            sm_gpu_bytes += param._sm_gpu.numel() * param._sm_gpu.element_size()
                        if getattr(param, "exp_mv", None) is not None:
                            exp_bytes += len(param.exp_mv)
                    # LoRA / adapter 参数 (可训练)
                    elif "adapter" in name.lower() or "lora" in name.lower():
                        lora_numel += param.numel()

                if sm_gpu_bytes > 0:
                    sm_gpu_mb = sm_gpu_bytes / MB
                    comp["sm_gpu_mb"] = sm_gpu_mb
                if lora_numel > 0:
                    # bf16
                    lora_mb = lora_numel * 2 / MB
                    comp["lora_mb"] = lora_mb
                if exp_bytes > 0:
                    exp_cpu_mb = exp_bytes / MB
                    comp["exp_cpu_mb"] = exp_cpu_mb
                if sm_gpu_mb is not None or lora_mb is not None:
                    other = baseline
                    if sm_gpu_mb is not None:
                        other -= sm_gpu_mb
                    if lora_mb is not None:
                        other -= lora_mb
                    comp["other_steady_mb"] = max(0.0, other)
            except Exception:
                pass
        return comp


# ── 全局实例管理 ──

_profiler_instance: MemRiftMemoryProfiler | None = None


def get_memory_profiler() -> MemRiftMemoryProfiler | None:
    return _profiler_instance


def install_memory_profiler(
    model_chunks: list,
    num_layers: int = 22,
    profile_iters: int = 3,
    rank: int = 0,
) -> MemRiftMemoryProfiler:
    """安装 MemRift 显存 profiler，返回实例。"""
    global _profiler_instance
    profiler = MemRiftMemoryProfiler(
        num_layers=num_layers,
        profile_iters=profile_iters,
        rank=rank,
    )
    for chunk in model_chunks:
        profiler.install(chunk)
    _profiler_instance = profiler
    return profiler
