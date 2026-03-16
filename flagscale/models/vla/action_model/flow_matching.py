import torch
import torch.nn as nn

from flagscale.models.utils.constants import ACTION
from flagscale.models.vla.action_model.gr00t_action_header import (
    FlowmatchingActionHead as _FlowmatchingActionHead,
)
from flagscale.models.vla.registry import register_action_model
from flagscale.models.vla.utils import get_vlm_config
from flagscale.train.train_config import TrainConfig


@register_action_model("flow_matching")
class FlowMatchingHead(nn.Module):
    """
    Flow matching action head wrapper for VLA framework.

    Args:
        vlm_config: HF config object from VLM (used to get hidden_size).
        action_config: dict with action model settings.
        full_config: TrainConfig for initializing the underlying FlowmatchingActionHead.
    """

    def __init__(self, vlm_config, action_config: dict, full_config: TrainConfig = None):
        super().__init__()
        vlm_info = get_vlm_config(vlm_config)
        self.hidden_size = vlm_info["hidden_size"]

        # TODO: pass cross_attention_dim directly to action head instead of mutating full_config
        full_config.model.action_model.diffusion_model_cfg.cross_attention_dim = self.hidden_size

        self._head = _FlowmatchingActionHead(full_config=full_config)

    def forward(
        self, vlm_output: dict[str, torch.Tensor], action_input: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            vlm_output: From VLM, contains 'hidden_states'.
            action_input: Raw batch with 'actions', 'state', etc.
        Returns:
            dict with 'loss'.
        """
        vl_embs = vlm_output["hidden_states"]
        actions = action_input["actions"]
        state = action_input.get("state")
        encoder_attention_mask = action_input.get("attention_mask")
        mask = action_input.get("mask")

        loss = self._head.forward(
            vl_embs=vl_embs,
            actions=actions,
            state=state,
            encoder_attention_mask=encoder_attention_mask,
            mask=mask,
        )
        return {"loss": loss}

    def predict_action(
        self, vlm_output: dict[str, torch.Tensor], action_input: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            vlm_output: From VLM, contains 'hidden_states'.
            action_input: Raw batch with 'state', etc.
        Returns:
            dict with 'actions': Tensor [B, horizon, action_dim].
        """
        vl_embs = vlm_output["hidden_states"]
        state = action_input.get("state")

        actions = self._head.predict_action(vl_embs=vl_embs, state=state)
        return {ACTION: actions}

    def fsdp_units(self) -> list[nn.Module]:
        return list(self._head.model.transformer_blocks)
