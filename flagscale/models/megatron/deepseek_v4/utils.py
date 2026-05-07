import torch


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
            # self.hash_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.hash_stream):
                self._result = self.hash_mapping.hash(self.input_ids)
            self._is_async_pending = True
            # record result to use across stream
            self._record_current_stream()

    def _record_current_stream(self):
        """Helper to record current stream on all result tensors"""
        if self._result is None:
            return
        current_stream = torch.cuda.current_stream()
        if isinstance(self._result, dict):
            for t in self._result.values():
                if isinstance(t, torch.Tensor):
                    t.record_stream(current_stream)
        elif isinstance(self._result, torch.Tensor):
            self._result.record_stream(current_stream)

    def __getitem__(self, key):
        # Case 1: Async compute -> wait
        if self._is_async_pending:
            torch.cuda.current_stream().wait_stream(self.hash_stream)
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
