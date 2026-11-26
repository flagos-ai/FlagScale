"""Message-level SFT targets for the Qwen ChatML checkpoint template."""

import re
from uuid import uuid4

import numpy as np

from .tokenizer import _HFTokenizerFS


_START = "<|im_start|>"
_END = "<|im_end|>"
_ASSISTANT = _START + "assistant\n"
_EMPTY_THINK = re.compile(r"<think>\s*</think>(?:\r?\n){0,2}")


class QwenMessageLossTokenizer(_HFTokenizerFS):
    """Keep the checkpoint's input IDs and select supervised spans in the rendered chat."""

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        template = self.tokenizer.get_chat_template()
        if not self.tokenizer.is_fast or _START not in template or _END not in template:
            raise ValueError("Message loss masking requires a fast tokenizer with a Qwen ChatML template")

        # Mark delimiters emitted by Jinja. Literal ChatML examples inside user or
        # assistant content must never start/end an actual supervised turn.
        self._boundary = f"__chat_boundary_{uuid4().hex}__"
        self._marked_template = template.replace(_START, self._boundary + _START).replace(
            _END, self._boundary + _END
        )

    @property
    def vocab_size(self):
        return len(self.tokenizer)  # Include added ChatML tokens when calculating TP padding.

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        if not return_target:
            encoded = self.tokenizer.apply_chat_template(
                conversation, tokenize=True, add_generation_prompt=add_generation_prompt,
                return_dict=True,
            )
            return np.asarray(encoded["input_ids"])
        if add_generation_prompt:
            raise ValueError("SFT targets require a completed conversation")

        rendered = self.tokenizer.apply_chat_template(
            conversation, chat_template=self._marked_template, tokenize=False,
            add_generation_prompt=False,
        )
        parts = rendered.split(self._boundary)
        assistant_turns = [m for m in conversation if m["role"] == "assistant"]
        spans = []
        offset = 0
        turn = 0
        for i, part in enumerate(parts):
            if i and part.startswith(_ASSISTANT):
                if (turn >= len(assistant_turns) or i + 1 == len(parts)
                        or not parts[i + 1].startswith(_END)):
                    raise ValueError("Cannot align assistant messages with Qwen ChatML boundaries")
                # Arrow fills absent optional fields with None. Only explicit false opts out.
                if assistant_turns[turn].get("loss") is not False:
                    start = len(_ASSISTANT)
                    empty_think = _EMPTY_THINK.match(part, start)
                    if empty_think:
                        start = empty_think.end()
                    spans.append((offset + start, offset + len(part) + len(_END)))
                turn += 1
            offset += len(part)
        if turn != len(assistant_turns):
            raise ValueError("Qwen ChatML template did not render every assistant message")

        # Tokenize the complete unmarked text once, preserving BPE at turn boundaries.
        encoding = self.tokenizer(
            "".join(parts), add_special_tokens=False, return_offsets_mapping=True,
            return_attention_mask=False, return_tensors="np",
        )
        tokens = encoding["input_ids"][0]
        offsets = encoding["offset_mapping"][0]
        targets = np.full_like(tokens, -100)
        for start, end in spans:
            first = offsets[:, 0].searchsorted(start)
            stop = offsets[:, 1].searchsorted(end, side="right")
            targets[first:stop] = tokens[first:stop]
        return tokens, targets
