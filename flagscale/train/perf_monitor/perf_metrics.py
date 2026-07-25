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

"""Performance metrics collection for training."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from flagscale.train.perf_monitor.flops_calculator import FLOPSFormulas
from flagscale.train.perf_monitor.perf_logger import PerfMonitorLogger

try:
    from megatron.core.num_microbatches_calculator import get_num_microbatches
except ImportError:
    get_num_microbatches = None


@dataclass
class TFLOPSMetrics:
    tflops_per_gpu: float = 0.0
    tflops_total: float = 0.0
    model_flops: float = 0.0
    avg_step_time: float = 0.0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    forward_flops: float = 0.0
    backward_flops: float = 0.0
    optimizer_flops: float = 0.0
    min_step_time: float = float("inf")
    max_step_time: float = 0.0
    std_step_time: float = 0.0


class ModelFLOPSCalculator:
    """Estimate per-step FLOPS from model hyperparameters."""

    def __init__(self, args):
        self.args = args
        self.formulas = FLOPSFormulas()
        self.model_type = self._determine_model_type()

    def _determine_model_type(self):
        model_type = getattr(self.args, "perf_model_type", "auto")
        if model_type != "auto":
            return model_type

        model_name = getattr(self.args, "model_name", "") or getattr(
            self.args, "wandb_exp_name", ""
        )
        model_name = str(model_name).lower()
        if "qwen" in model_name:
            return "qwen"
        if "llama" in model_name:
            return "llama"
        if "aquila" in model_name:
            return "aquila"
        if "mixtral" in model_name or getattr(self.args, "num_experts", None):
            return "moe"
        return "gpt"

    def _get_batch_size(self):
        num_micro_batches = getattr(self.args, "num_micro_batches", 1)
        if get_num_microbatches is not None:
            try:
                num_micro_batches = get_num_microbatches()
            except AttributeError:
                pass
        micro_batch_size = getattr(self.args, "micro_batch_size", 1)
        return max(1, micro_batch_size * num_micro_batches)

    def calculate_total_flops(self, batch_size=None):
        if batch_size is None:
            batch_size = self._get_batch_size()

        seq_length = getattr(self.args, "seq_length", 512)
        hidden_size = getattr(self.args, "hidden_size", 768)
        num_layers = getattr(self.args, "num_layers", 12)
        vocab_size = getattr(
            self.args, "vocab_size", getattr(self.args, "padded_vocab_size", 50257)
        )
        num_attention_heads = getattr(self.args, "num_attention_heads", 12)
        ffn_hidden_size = getattr(self.args, "ffn_hidden_size", 4 * hidden_size)
        use_swiglu = getattr(self.args, "swiglu", False)

        if self.model_type in ("llama", "qwen"):
            attention_flops = self.formulas.gqa_attention_flops(
                batch_size=batch_size,
                seq_length=seq_length,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_query_groups=getattr(self.args, "num_query_groups", num_attention_heads),
            )
            use_swiglu = True
        else:
            attention_flops = self.formulas.attention_flops(
                batch_size=batch_size,
                seq_length=seq_length,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
            )

        if self.model_type == "moe":
            ffn_flops = self.formulas.moe_flops(
                batch_size=batch_size,
                seq_length=seq_length,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                num_experts=getattr(self.args, "num_experts", 8),
                top_k=getattr(self.args, "moe_router_topk", 2),
                use_swiglu=use_swiglu,
            )
        else:
            ffn_flops = self.formulas.ffn_flops(
                batch_size=batch_size,
                seq_length=seq_length,
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                use_swiglu=use_swiglu,
            )

        embedding_flops = 2 * batch_size * seq_length * hidden_size * vocab_size
        layer_flops = (attention_flops + ffn_flops) * num_layers
        return 3 * (layer_flops + embedding_flops)

    def get_flops_breakdown(self):
        batch_size = self._get_batch_size()
        total = self.calculate_total_flops(batch_size=batch_size)
        forward = total / 3
        backward = 2 * forward
        return {
            "forward": forward,
            "backward": backward,
            "optimizer": 0.0,
            "total": total,
        }


class PerformanceMonitor:
    """Track step time, estimated FLOPS and throughput."""

    def __init__(self, args, enable_memory_tracking=True):
        self.args = args
        self.enable_memory_tracking = enable_memory_tracking
        self.iteration_start_time = None
        self.step_times = []
        self.current_memory_gb = 0.0
        self.peak_memory_gb = 0.0
        self.metrics = TFLOPSMetrics()
        self.flops_calculator = ModelFLOPSCalculator(args)
        self.file_logger = PerfMonitorLogger(
            log_dir=getattr(args, "perf_log_dir", "logs/perf_monitor"),
            enable_console=getattr(args, "perf_console_output", False),
            max_log_files=getattr(args, "perf_max_log_files", 10),
            log_format=getattr(args, "perf_log_format", "both"),
        )

    def start_iteration(self):
        self.iteration_start_time = time.time()

    def end_iteration(self):
        if self.iteration_start_time is None:
            return
        step_time = time.time() - self.iteration_start_time
        self.step_times.append(step_time)
        self.iteration_start_time = None
        self.metrics.min_step_time = min(self.metrics.min_step_time, step_time)
        self.metrics.max_step_time = max(self.metrics.max_step_time, step_time)

    def update_memory_stats(self):
        if not self.enable_memory_tracking or not torch.cuda.is_available():
            return
        self.current_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        self.peak_memory_gb = max(
            self.peak_memory_gb, torch.cuda.max_memory_allocated() / (1024**3)
        )

    def calculate_metrics(self):
        if not self.step_times:
            return self.metrics

        half_idx = len(self.step_times) // 2
        recent_times = self.step_times[half_idx:] if half_idx > 0 else self.step_times
        avg_step_time = statistics.median(recent_times)
        self.metrics.avg_step_time = avg_step_time
        self.metrics.std_step_time = (
            statistics.pstdev(recent_times) if len(recent_times) > 1 else 0.0
        )

        batch_size = self.flops_calculator._get_batch_size()
        model_flops = self.flops_calculator.calculate_total_flops(batch_size=batch_size)
        self.metrics.model_flops = model_flops

        if avg_step_time > 0:
            world_size = max(1, getattr(self.args, "world_size", 1))
            self.metrics.tflops_total = model_flops / (1e12 * avg_step_time)
            self.metrics.tflops_per_gpu = self.metrics.tflops_total / world_size
            self.metrics.samples_per_second = batch_size / avg_step_time
            self.metrics.tokens_per_second = self.metrics.samples_per_second * getattr(
                self.args, "seq_length", 0
            )

        breakdown = self.flops_calculator.get_flops_breakdown()
        self.metrics.forward_flops = breakdown.get("forward", 0.0)
        self.metrics.backward_flops = breakdown.get("backward", 0.0)
        self.metrics.optimizer_flops = breakdown.get("optimizer", 0.0)
        return self.metrics

    def log_metrics(self, iteration, writer=None, wandb_writer=None):
        metrics = self.calculate_metrics()
        metrics_dict = {
            "TFLOPS_per_GPU": metrics.tflops_per_gpu,
            "TFLOPS_total": metrics.tflops_total,
            "samples_per_sec": metrics.samples_per_second,
            "tokens_per_sec": metrics.tokens_per_second,
            "step_time_ms": metrics.avg_step_time * 1000,
        }
        if self.enable_memory_tracking:
            metrics_dict["memory_GB"] = self.current_memory_gb
            metrics_dict["peak_memory_GB"] = self.peak_memory_gb
        self.file_logger.log_metrics(iteration, metrics_dict)

        if getattr(self.args, "perf_breakdown", False):
            self.file_logger.log_breakdown(
                iteration,
                {
                    "forward": metrics.forward_flops,
                    "backward": metrics.backward_flops,
                    "optimizer": metrics.optimizer_flops,
                    "total": metrics.model_flops,
                },
            )

        if writer is not None:
            writer.add_scalar("performance/tflops_per_gpu", metrics.tflops_per_gpu, iteration)
            writer.add_scalar("performance/tflops_total", metrics.tflops_total, iteration)
            writer.add_scalar(
                "performance/avg_step_time_ms", metrics.avg_step_time * 1000, iteration
            )
            writer.add_scalar(
                "performance/samples_per_second", metrics.samples_per_second, iteration
            )
            writer.add_scalar("performance/tokens_per_second", metrics.tokens_per_second, iteration)
            if self.enable_memory_tracking:
                writer.add_scalar("memory/current_gb", self.current_memory_gb, iteration)
                writer.add_scalar("memory/peak_gb", self.peak_memory_gb, iteration)

        if wandb_writer is not None:
            wandb_writer.log(
                {
                    "performance/tflops_per_gpu": metrics.tflops_per_gpu,
                    "performance/tflops_total": metrics.tflops_total,
                    "performance/avg_step_time_ms": metrics.avg_step_time * 1000,
                    "performance/samples_per_second": metrics.samples_per_second,
                    "performance/tokens_per_second": metrics.tokens_per_second,
                    "memory/current_gb": self.current_memory_gb,
                    "memory/peak_gb": self.peak_memory_gb,
                },
                iteration,
            )


class FLOPSMeasurementCallback:
    """Train-loop callback wrapper around :class:`PerformanceMonitor`."""

    def __init__(self, args, log_interval=100):
        self.args = args
        self.log_interval = max(1, log_interval)
        self.monitor = PerformanceMonitor(
            args, enable_memory_tracking=getattr(args, "perf_memory_tracking", True)
        )

    def on_train_batch_start(self, iteration):
        self.monitor.start_iteration()

    def on_train_batch_end(self, iteration, writer=None, wandb_writer=None):
        self.monitor.end_iteration()
        self.monitor.update_memory_stats()
        if iteration > 0 and (iteration == 1 or iteration % self.log_interval == 0):
            self.monitor.log_metrics(iteration, writer, wandb_writer)

    def on_train_end(self, writer=None, wandb_writer=None):
        if self.monitor.step_times and not self.monitor.file_logger.json_data:
            self.monitor.log_metrics(len(self.monitor.step_times), writer, wandb_writer)

        metrics = self.monitor.calculate_metrics()
        self.monitor.file_logger.save_summary(
            {
                "avg_tflops_per_gpu": metrics.tflops_per_gpu,
                "avg_tflops_total": metrics.tflops_total,
                "avg_step_time_ms": metrics.avg_step_time * 1000,
                "min_step_time_ms": metrics.min_step_time * 1000
                if metrics.min_step_time != float("inf")
                else 0.0,
                "max_step_time_ms": metrics.max_step_time * 1000,
                "peak_memory_gb": self.monitor.peak_memory_gb,
            },
            total_iterations=len(self.monitor.step_times),
        )
