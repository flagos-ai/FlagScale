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

from megatron.plugin.platform import get_platform


class LazyHashInputIds:
    """Compute Engram hash IDs lazily, optionally on a separate device stream."""

    def __init__(self, hash_mapping, input_ids, hash_stream=None):
        self.hash_mapping = hash_mapping
        self.input_ids = input_ids
        self.hash_stream = hash_stream
        self._platform = get_platform()
        self._result = None
        self._is_async_pending = False

        if self.hash_stream is not None:
            producer_stream = self._platform.current_stream()
            self.hash_stream.wait_stream(producer_stream)
            with self._platform.stream(self.hash_stream):
                self._result = self.hash_mapping.hash(self.input_ids)
            self._is_async_pending = True
            self._record_current_stream()

    def _record_current_stream(self):
        """Keep computed tensors alive while they are consumed on the current stream."""
        if self._result is None:
            return

        current_stream = self._platform.current_stream()
        tensors = self._result.values() if isinstance(self._result, dict) else (self._result,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                tensor.record_stream(current_stream)

    def __getitem__(self, key):
        if self._is_async_pending:
            self._platform.current_stream().wait_stream(self.hash_stream)
            self._is_async_pending = False
            self._record_current_stream()
        elif self._result is None:
            self._result = self.hash_mapping.hash(self.input_ids)

        return self._result[key]

    def get(self, key, default=None):
        """Return the hash IDs for key, or default when the key is absent."""
        try:
            return self[key]
        except KeyError:
            return default


def pre_compute_first_local_engram_layer(decoder, engram_hash_input_ids):
    """Pre-compute Engram embeddings when the first local layer has no predecessor."""
    if decoder.layers and getattr(decoder.layers[0], "is_engram_layer", False):
        decoder.layers[0].pre_compute_embedding(engram_hash_input_ids)
