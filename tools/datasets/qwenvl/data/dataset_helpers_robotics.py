# Adopted from https://github.com/alibaba/Pai-Megatron-Patch/blob/8949a6647cbf6b39837ad3dd911fa4aa0726895b/megatron_patch/data/energon/chatml.py.
import logging

import numpy as np
import PIL
import robotics.models.model as robotics_model
import robotics.training.config as _config
import robotics.transforms as _transforms
import torch
from tools.datasets.qwenvl.data.energon.chatml_robotics import ChatMLSample
from torchvision import transforms

from megatron.energon import DefaultTaskEncoder

dataset_logger = logging.getLogger(__name__)


class TaskEncoder(
    DefaultTaskEncoder[
        ChatMLSample,
        ChatMLSample,
        ChatMLSample,
        ChatMLSample,
        # tuple[robotics_model.Observation, robotics_model.Actions],
    ]
):
    def __init__(self, config: _config.TrainConfig):
        super().__init__()
        self.vision_root = ""
        self.config = config
        return

    def encode_sample(
        self, sample: ChatMLSample
    ) -> tuple[robotics_model.Observation, robotics_model.Actions]:
        # action_qpos = torch.from_numpy(np.load(sample.action_qpos))
        action_eepose = torch.from_numpy(np.load(sample.action_eepose))
        # state_qpos = torch.from_numpy(np.load(sample.state_qpos))
        state_eepose = torch.from_numpy(np.load(sample.state_eepose))
        imgs = []
        for i in sample.imgs:
            image = PIL.Image.open(i)
            image_tensor = transforms.ToTensor()(image)
            imgs.append(image_tensor)
        # image = torch.concat(imgs, dim=0)
        lerobot_data = {
            "image": imgs[0].flip(dims=[0]),  # [3,256,256] [3,240,320]
            "wrist_image": imgs[2].flip(dims=[0]),  # [3,256,256] [3,240,320]
            "wrist_image2": imgs[1].flip(dims=[0]),  # [3,256,256] [3,240,320]
            # "state": state_eepose[0,:8], # [8] [30,14]
            "state": state_eepose[0],
            # "actions": action_eepose[:30,:7], # [50,7] [30, 14]
            "actions": action_eepose,
            "timestamp": torch.tensor(0.0),
            "frame_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "index": torch.tensor(0),
            "task_index": torch.tensor(0),
            "actions_is_pad": torch.zeros(action_eepose.shape[0]).bool(),  # [50] [30]
            "task": sample.conversation["conversations"][0]["value"],
            "prompt": sample.conversation["conversations"][0]["value"],
        }
        lerobot_data["action"] = lerobot_data["actions"]
        lerobot_data["wrist_image_right"] = lerobot_data["wrist_image"]
        lerobot_data["wrist_image_left"] = lerobot_data["wrist_image2"]

        data_config = self.config.data.create(self.config.assets_dirs, self.config.model)
        data_transform_fn = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                # _transforms.Normalize(None, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        data_transformed = data_transform_fn(lerobot_data)
        # {k: (v.shape, v.dtype) if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in data_transformed.items()}
        # {'state': ((32,), dtype('float64')),
        # 'actions': ((30, 32), dtype('float64')),
        # 'tokenized_prompt': ((200,), dtype('int64')),
        # 'tokenized_prompt_mask': ((200,), dtype('bool')),
        # 'pixel_values': ((512, 1176), dtype('float32')),
        # 'image_grid_thw': ((2, 3), dtype('int64'))}

        # [b, action_steps, action_dim]
        actions = torch.from_numpy(data_transformed["actions"])[None,]
        obs = robotics_model.Observation()
        obs.state = torch.from_numpy(data_transformed["state"])[None,]
        obs.prompt = None
        obs.images = None
        obs.image_masks = None
        obs.tokenized_prompt = torch.from_numpy(data_transformed["tokenized_prompt"])[None,]
        obs.tokenized_prompt_mask = torch.from_numpy(data_transformed["tokenized_prompt_mask"])[
            None,
        ]
        obs.pixel_values = torch.from_numpy(data_transformed["pixel_values"])[None,]
        obs.image_grid_thw = torch.from_numpy(data_transformed["image_grid_thw"])[None,]

        return obs, actions

    def batch(
        self, samples: list[tuple[robotics_model.Observation, robotics_model.Actions]]
    ) -> tuple[robotics_model.Observation, robotics_model.Actions]:
        rsp_obs, rsp_action = samples[0]
        for s in samples[1:]:
            obs, action = s
            rsp_action = torch.cat([rsp_action, action], dim=0)
            rsp_obs.state = torch.cat([rsp_obs.state, obs.state], dim=0)
            rsp_obs.tokenized_prompt = torch.cat(
                [rsp_obs.tokenized_prompt, obs.tokenized_prompt], dim=0
            )
            rsp_obs.tokenized_prompt_mask = torch.cat(
                [rsp_obs.tokenized_prompt_mask, obs.tokenized_prompt_mask], dim=0
            )
            rsp_obs.pixel_values = torch.cat([rsp_obs.pixel_values, obs.pixel_values], dim=0)
            rsp_obs.image_grid_thw = torch.cat([rsp_obs.image_grid_thw, obs.image_grid_thw], dim=0)

        return rsp_obs, rsp_action

    def encode_batch(
        self, samples: tuple[robotics_model.Observation, robotics_model.Actions]
    ) -> tuple[robotics_model.Observation, robotics_model.Actions]:
        return samples
