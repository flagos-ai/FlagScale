from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTaskHandler
from .t2i import T2IHandler
from .vlm import VLMHandler

TASK_REGISTRY: dict[str, "BaseTaskHandler"] = {}


def register_task(name, cls):
    TASK_REGISTRY[name] = cls


register_task("vlm_sft", VLMHandler)
register_task("t2i", T2IHandler)
