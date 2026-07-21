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

import torch
import torch.nn.functional as F

# Default chunk_tokens=128 is sized so that chunk_tokens × vocab_size × 4 bytes
# stays close to H100 L2 cache capacity (~50 MB), keeping the float32 upcast
# cache-resident. For vocab_size=248320: 128 × 248320 × 4 ≈ 127 MB.


def chunked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int = 128,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Memory-efficient cross-entropy that processes logits in fixed-size token chunks.

    Avoids materializing the full [batch*seq_len, vocab_size] float32 tensor at once
    by splitting along the sequence dimension and accumulating the loss per chunk.

    Args:
        logits: (batch_size, seq_len, vocab_size) in bf16/fp16
        labels: (batch_size, seq_len) with ignore_index for masked positions
        chunk_tokens: number of tokens per chunk (0 = no chunking, use standard CE)
        ignore_index: label value to ignore

    Returns:
        Scalar loss (mean over non-masked tokens).
    """
    if chunk_tokens <= 0:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    total_elements = (labels != ignore_index).sum()
    if total_elements == 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    logit_chunks = logits.split(chunk_tokens, dim=1)
    label_chunks = labels.split(chunk_tokens, dim=1)

    total_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    for logit_chunk, label_chunk in zip(logit_chunks, label_chunks):
        total_loss += F.cross_entropy(
            logit_chunk.reshape(-1, logit_chunk.size(-1)).float(),
            label_chunk.reshape(-1),
            ignore_index=ignore_index,
            reduction="sum",
        )

    return total_loss / total_elements
