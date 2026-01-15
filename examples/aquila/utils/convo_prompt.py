import dataclasses
from enum import Enum, auto
from typing import Any

from llava import conversation as conversation_lib


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    instruction: str
    roles: list[str]
    messages: list[list[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            if self.instruction is not None and len(self.instruction) > 0:
                ret += self.roles[2] + ": " + self.instruction + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            if self.instruction is not None and len(self.instruction) > 0:
                ret += self.roles[2] + ": " + self.instruction + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            instruction=self.instruction,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
        )

    def dict(self):
        return {
            "system": self.system,
            "instruction": self.instruction,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


conv_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    instruction="",
    roles=("Human", "Assistant", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    instruction="",
    roles=("Human", "Assistant", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    instruction="",
    roles=("USER", "GPT", "System"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_v1_2
conv_templates = {"v1": conv_v1_2, "bair_v1": conv_bair_v1}

# utils
"""Add speaker and start/end signal on each round."""
BEGIN_SIGNAL = "### "
END_SIGNAL = "\n"
header = f"{default_conversation.system}\n\n"
unknown_role = "unknown"  # use default unknown role
roles = {
    "human": default_conversation.roles[0],  # human role
    "gpt": default_conversation.roles[1],  # gpt role
}


def _add_speaker_and_signal(source, get_conversation=True):
    conversation = header
    if (
        "instruction" in source
        and source["instruction"] is not None
        and len(source["instruction"]) > 0
    ):
        source["instruction"] = (
            BEGIN_SIGNAL
            + conversation_lib.default_conversation.roles[2]
            + ": "
            + source["instruction"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += source["instruction"]
    for sentence in source["conversations"]:
        sentence_from = sentence["from"].lower()
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation


if __name__ == "__main__":
    print(default_conversation.get_prompt())
