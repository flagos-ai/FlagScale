class BaseTaskHandler:
    def __init__(self, tokenizer, special_tokens, data_config):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.data_config = data_config

    def encode(self, sample: dict, subflavors: dict, **kwargs):
        raise NotImplementedError
