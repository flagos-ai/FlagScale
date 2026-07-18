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

# ruff: noqa: RUF013
## built-in
from typing import Optional

import torch
from torch import Tensor

from megatron.plugin.platform import get_platform

from megatron.core.inference.contexts import BaseInferenceContext

## megatron-core
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params
from megatron.core.transformer.engram import get_or_create_hash_mapping

## engram
from .engram_transformer_layer import EngramTransformerBlock

cur_platform = get_platform()


class LazyHashInputIds:
    """
    Lazy wrapper for hash input IDs that computes asynchronously and
    synchronizes only when accessed. This allows hash computation to overlap
    with preprocessing and early decoder layers.
    """

    def __init__(self, hash_mapping, input_ids, hash_stream=None):
        self.hash_mapping = hash_mapping
        self.input_ids = input_ids
        self.hash_stream = hash_stream
        self._result = None
        self._is_async_pending = False
        # Async
        if self.hash_stream is not None:
            # self.hash_stream.wait_stream(cur_platform.current_stream())
            with cur_platform.stream(self.hash_stream):
                self._result = self.hash_mapping.hash(self.input_ids)
            self._is_async_pending = True
            # record result to use across stream
            self._record_current_stream()

    def _record_current_stream(self):
        """Helper to record current stream on all result tensors"""
        if self._result is None:
            return
        current_stream = cur_platform.current_stream()
        if isinstance(self._result, dict):
            for t in self._result.values():
                if isinstance(t, torch.Tensor):
                    t.record_stream(current_stream)
        elif isinstance(self._result, torch.Tensor):
            self._result.record_stream(current_stream)

    def __getitem__(self, key):
        # Case 1: Async compute -> wait
        if self._is_async_pending:
            cur_platform.current_stream().wait_stream(self.hash_stream)
            self._is_async_pending = False  # Async finish
            self._record_current_stream()

        # Case 2: Sync but no compute -> start compute
        elif self._result is None:
            self._result = self.hash_mapping.hash(self.input_ids)

        # Case 3: Async or sync compute is finished.
        # print(f"[rank{torch.distributed.get_rank()}]: LazyHashInputIds result = {self._result}")
        return self._result[key]

    def get(self, key, default=None):
        """Get hash result with default value."""
        try:
            return self[key]
        except KeyError:
            return default


class EngramModel(GPTModel):
    def __init__(self, *args, **kwargs):
        # NOTE: We temporarily replace TransformerBlock with EngramTransformerBlock
        # during super().__init__() to avoid creating decoder twice.
        # This is necessary because GPTModel.__init__ hardcodes TransformerBlock.
        # The replacement is scoped to this initialization only.
        import megatron.core.models.gpt.gpt_model as gpt_module

        original_block = gpt_module.TransformerBlock
        gpt_module.TransformerBlock = EngramTransformerBlock

        try:
            super().__init__(*args, **kwargs)
            # self.decoder is now EngramTransformerBlock, no need to recreate
        finally:
            gpt_module.TransformerBlock = original_block

        self.engram_hash = get_or_create_hash_mapping(
            engram_vocab_size=self.config.engram_vocab_size,
            max_ngram_size=self.config.max_ngram_size,
            n_embed_per_ngram=self.config.n_embed_per_ngram,
            n_head_per_ngram=self.config.n_head_per_ngram,
            layer_ids=self.config.engram_layer_ids,
            tokenizer_name_or_path=self.config.engram_tokenizer_name_or_path,
            pad_id=self.config.engram_pad_id,
            seed=self.config.engram_seed,
        )

        # Optional: Create a separate CUDA stream for hash computation
        # This allows overlapping hash computation with preprocessing
        self._hash_stream = None
        if cur_platform.is_available():
            self._hash_stream = cur_platform.Stream()

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: bool | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        assert input_ids is not None, "Input ids can not be None for EngramModel"
        inference_context = deprecate_inference_params(inference_context, inference_params)

        # Create lazy hash input IDs that computes asynchronously
        # The computation will start immediately in a separate stream (if available)
        # but will only synchronize when actually accessed in the decoder
        engram_hash_input_ids = LazyHashInputIds(
            hash_mapping=self.engram_hash,
            input_ids=input_ids,
            hash_stream=self._hash_stream,
        )

        # Preprocessing can run in parallel with hash computation
        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = (
            preproc_output[:5]
        )

        rotary_pos_cos_sin = preproc_output[5] if len(preproc_output) == 6 else None

        # torch.cuda.nvtx.range_push("EngramModel decoder")
        # Run decoder with engram
        hidden_states = self.decoder(
            input_ids=input_ids,
            engram_hash_input_ids=engram_hash_input_ids,
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )
        # torch.cuda.nvtx.range_pop()

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def build_schedule_plan(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ):
        """
        Adaptation of overlap_moe_expert_parallel_comm.
        """
        # Precompute the engram_hash_iput_ids, it will be used to create a TransformerChunkSchedulePlan.
        engram_hash_input_ids = LazyHashInputIds(
            hash_mapping=self.engram_hash,
            input_ids=input_ids,
            hash_stream=self._hash_stream,
        )
        if extra_block_kwargs is None:
            extra_block_kwargs = {
                "engram_hash_input_ids": engram_hash_input_ids,
            }
        return super().build_schedule_plan(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            extra_block_kwargs=extra_block_kwargs,
        )
