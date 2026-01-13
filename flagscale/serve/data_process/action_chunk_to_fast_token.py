# Adopted from FlagOpen/RoboBrain-X0 (https://github.com/FlagOpen/RoboBrain-X0/blob/main/data_process/action_token/action_chunk_to_fast_token.py)


import numpy as np
from scipy.fft import idct
from transformers import AutoProcessor

from flagscale.runner.utils import logger


class ActionChunkProcessor:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self.max_len = max_len
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True
        )

    def extract_actions_from_tokens(
        self, action_tokens: list[list[int]], action_horizon: int, action_dim: int
    ) -> np.ndarray:
        assert action_horizon is not None and action_dim is not None, (
            "Tokenizer not initialized, call encode() once or pass in time_horizon and action_dim."
        )

        decoded_actions = []
        output_dims = []
        for token in action_tokens:
            try:
                decoded_tokens = self.fast_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = (
                    np.array(list(map(ord, decoded_tokens))) + self.fast_tokenizer.min_token
                )
                output_dim = len(decoded_dct_coeff)
                output_dims.append(output_dim)
                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (
                    action_horizon,
                    action_dim,
                ), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, expected ({action_horizon}, {action_dim})"
                )
            except Exception as e:
                logger.info(f"Error decoding tokens: {e}")
                logger.info(f"Tokens: {token}")
                if len(decoded_dct_coeff.shape) == 2:
                    results = []
                    output_horizon = decoded_dct_coeff.shape[0]
                    needed_rows = 30
                    zero_rows_needed = max(0, needed_rows - output_horizon)
                    results.extend(decoded_dct_coeff[:needed_rows])
                    zero_vector = [0] * action_dim
                    results.extend([zero_vector for _ in range(zero_rows_needed)])
                    decoded_dct_coeff = np.array(results)
                else:
                    decoded_dct_coeff = np.zeros((action_horizon, action_dim))
            decoded_actions.append(
                idct(decoded_dct_coeff / self.fast_tokenizer.scale, axis=0, norm="ortho")
            )
        return np.stack(decoded_actions), output_dims
