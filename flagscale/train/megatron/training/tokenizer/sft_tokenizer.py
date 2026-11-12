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

"""Assistant-only supervision for the default SFT chat template."""

import json
import re

import numpy as np

from megatron.core.tokenizers.text.libraries.sft_tokenizer import (
    IGNORE_INDEX,
    SFTTokenizer as CoreSFTTokenizer,
)
from megatron.core.tokenizers.text.models import DefaultTokenizerText


class CustomSFTTokenizer(CoreSFTTokenizer):
    """Extend core SFT tokenization with chat-template generation masks.

    Only default-format training targets change. Other formats and inference
    continue to use the core implementation.
    """

    def __init__(
        self,
        tokenizer_path: str,
        prompt_format: str,
        assistant_only_loss: bool = True,
        custom_chat_template: str | None = None,
    ) -> None:
        # Keep compatibility with core versions without assistant_only_loss.
        super().__init__(tokenizer_path, prompt_format)
        self._assistant_only_loss = assistant_only_loss
        if prompt_format == "default":
            if custom_chat_template:
                self._prompt_config.custom_chat_template = self._tokenizer.get_chat_template(
                    custom_chat_template
                )
            elif self._tokenizer.chat_template:
                self._prompt_config.custom_chat_template = self._tokenizer.get_chat_template()
            else:
                raise ValueError(
                    "CustomSFTTokenizer with prompt_format='default' requires a chat template. "
                    "Set custom_chat_template (--custom-chat-template), or use an HF tokenizer "
                    "with its own chat_template. Disabling assistant-only loss does not "
                    "supply a template."
                )
        self.chat_template = self._prompt_config.custom_chat_template

    def tokenize_conversation(
        self, conversation: list[dict], return_target: bool, add_generation_prompt: bool
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Mask targets outside generation spans without changing input tokens."""
        if not return_target or self._prompt_format != "default":
            return super().tokenize_conversation(
                conversation, return_target, add_generation_prompt
            )

        template = self._prompt_config.custom_chat_template
        if self._assistant_only_loss and not re.search(r"\{%-?\s*generation\b", template or ""):
            raise ValueError(
                "Default SFT assistant-only loss requires {% generation %} spans in the "
                "chat template. Mark assistant outputs (including EOS) with generation spans, "
                "or disable masking with --no-sft-assistant-only-loss."
            )

        encoded = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_tokens_mask=self._assistant_only_loss,
            return_dict=self._assistant_only_loss,
            return_tensors="np",
            chat_template=template,
        )
        tokens = self._extract_token_ids(encoded)
        targets = tokens.copy()
        if not self._assistant_only_loss:
            return tokens, targets

        if "assistant_masks" not in encoded:
            raise ValueError("The tokenizer did not return the requested assistant mask.")
        mask = self._extract_token_ids({"input_ids": encoded["assistant_masks"]})
        if mask.shape != tokens.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError("The assistant mask must be binary and match the token shape.")
        if not mask.any() and any(turn["role"] == "assistant" for turn in conversation):
            raise ValueError(
                "The chat template produced no assistant tokens for an assistant turn. "
                "Check its generation spans and tokenizer character alignment."
            )
        targets[mask == 0] = IGNORE_INDEX
        return tokens, targets


class CustomSFTTokenizerText(DefaultTokenizerText):
    """Keep core tokenizer properties and cache identifiers around the extension."""

    def __init__(self, path: str, config: dict, **kwargs) -> None:
        super().__init__(path, config, **kwargs)
        self.chat_template = self._tokenizer.chat_template

    def _restore_model(self, **kwargs) -> CustomSFTTokenizer:
        return CustomSFTTokenizer(self.path, **kwargs)


def build_sft_tokenizer(args, **kwargs) -> CustomSFTTokenizerText:
    """Build the FlagScale SFT extension through the existing tokenizer registry."""
    metadata = getattr(args, "metadata_path", None)
    if isinstance(metadata, str):
        with open(metadata, encoding="utf-8") as stream:
            metadata = json.load(stream)
    config = dict(metadata) if metadata else {"library": "sft"}
    if config.get("library") != "sft":
        raise ValueError("SFTTokenizer metadata must use the 'sft' library.")
    kwargs["prompt_format"] = args.sft_tokenizer_prompt_format
    kwargs["assistant_only_loss"] = getattr(args, "sft_assistant_only_loss", True)
    kwargs["custom_chat_template"] = (
        kwargs.get("custom_chat_template")
        or getattr(args, "custom_chat_template", None)
    )
    return CustomSFTTokenizerText(path=args.tokenizer_model, config=config, **kwargs)
