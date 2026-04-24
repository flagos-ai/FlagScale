# Copyright (c) 2025, BAAI. All rights reserved.
#
# Mainly adapted from: https://github.com/NVIDIA-NeMo/Megatron-Bridge

import itertools
import logging

from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch
from transformers.modeling_utils import PreTrainedModel

from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import unwrap_model

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from megatron.bridge.models.conversion.utils import (
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.bridge.models.decorators.dispatch import dispatch

logger = logging.getLogger(__name__)

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge as OriginalMegatronModelBridge
from megatron.bridge.models.conversion.model_bridge import (
    _megatron_local_name_to_global,
    HFWeightTuple, 
    WeightConversionTask,
)
MappingT = TypeVar("MappingT", bound=MegatronParamMapping)
HFPreTrained = TypeVar("HFPreTrained")
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")

def padding_embedd_size(mcore_weight: torch.Tensor, hf_vocab_size: int):
    hf_size = hf_vocab_size
    mcore_size = mcore_weight.shape[0]
    full_word = {}
    is_rank0 = torch.distributed.get_rank() == 0
    # Cut out extra padding we don't need
    if mcore_size > hf_size:
        full_word = mcore_weight[0:hf_size, :]
        if is_rank0:
            print(f"> padding embedding size mcore {mcore_size} to hf {hf_size}")

    # Expanding embedding to larger size by replicating final entry
    elif mcore_size < hf_size:
        padding_size = hf_size - mcore_size

        full_word = torch.cat(
            (mcore_weight, mcore_weight[-1].unsqueeze(0).expand(padding_size, -1))
        )
        if is_rank0:
            print(f"> padding embedding size mcore {mcore_size} to hf {hf_size}")
    # Same size!
    else:
        full_word = mcore_weight
    return full_word

class MegatronModelBridge(OriginalMegatronModelBridge):

    def _broadcast_shared_embeddings(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> None:
        """Broadcast shared embeddings and output weights across embedding group.

        When embeddings and output weights are shared and pipeline parallelism is enabled,
        this method ensures all ranks in the embedding group have the same weights by
        broadcasting from rank 0.

        Args:
            megatron_model: Megatron model instance or list of model instances.
        """
        unwrapped_model = unwrap_model(megatron_model)[0]
        # hack for vlm to work properly
        if (
            hasattr(unwrapped_model, "language_model")
            and unwrapped_model.language_model is not None
        ):
            unwrapped_model = unwrapped_model.language_model
        model_config = unwrapped_model.config
        if (
            not model_config.untie_embeddings_and_output_weights
            and model_config.pipeline_model_parallel_size > 1
        ):
            # Broadcast embeddings and output weights from rank 0 to embedding group
            embd_group = parallel_state.get_embedding_group()
            embd_group_ranks = torch.distributed.get_process_group_ranks(embd_group)
            if embd_group is not None and torch.distributed.get_rank() in embd_group_ranks:
                # Get embeddings and output weights from rank 0
                if hasattr(unwrapped_model, "embedding") and hasattr(
                    unwrapped_model.embedding, "word_embeddings"
                ):
                    embd_weights = unwrapped_model.embedding.word_embeddings.weight.data
                else:
                    assert hasattr(unwrapped_model, "output_layer"), "Output layer not found"
                    embd_weights = torch.empty_like(unwrapped_model.output_layer.weight.data)
                torch.distributed.broadcast(embd_weights, src=embd_group_ranks[0], group=embd_group)
                if hasattr(unwrapped_model, "output_layer"):
                    unwrapped_model.output_layer.weight.data.copy_(embd_weights)

    @classmethod
    def register_bridge(
        cls, *, source: Type[PreTrainedModel] | str, target: Type[MegatronModel]
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        return create_bridge_decorator(source=source, target=target)

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        """Export Megatron weights to HuggingFace format.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]
        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        megatron_to_hf_tasks = conversion_tasks
        model_config = unwrap_model(megatron_model)[0].config
        # embeddings_are_tied = model_config.share_embeddings_and_output_weights
        embeddings_are_tied = not model_config.untie_embeddings_and_output_weights
        for task in self._with_progress_tracking(
            megatron_to_hf_tasks, "Converting to HuggingFace", show_progress
        ):
            converted_weights_dict = task.mapping.megatron_to_hf(
                task.param_weight, task.megatron_module
            )

            # All ranks get the full tensor
            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu()

                if hf_name == "model.embed_tokens.weight" or hf_name == "lm_head.weight":
                    final_tensor = padding_embedd_size(
                        final_tensor, hf_pretrained.config.vocab_size
                    )

                # Handle tied embeddings case
                # TODO(yuya): fix this hard coded naming
                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    # Yield the embedding weight
                    yield HFWeightTuple(hf_name, final_tensor)

                    # Also yield as lm_head.weight if it's expected
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            final_tensor = final_tensor.detach().clone()
                            yield HFWeightTuple("lm_head.weight", final_tensor)
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    # This should not happen when embeddings are tied - assert error
                    raise ValueError(
                        "Encountered lm_head.weight when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    # Regular case - yield the tensor normally
                    yield HFWeightTuple(hf_name, final_tensor)

    def build_conversion_tasks(
        self, hf_pretrained: HFPreTrained, megatron_model: List[MegatronModel]
    ) -> List[None | WeightConversionTask]:
        """Construct the conversion tasks between HF and megatron.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronMappingRegistry` whether it has a mapping for that
        parameter, and – if the corresponding HF weights actually exist – yields
        an :class:`_HFLoadTask` describing exactly how that parameter will be
        populated.
        """

        # Ensure hf_pretrained has the required state structure
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()
        mapping_registry = self.mapping_registry()
        model_config = unwrap_model(megatron_model)[0].config
        # embeddings_are_tied = model_config.share_embeddings_and_output_weights
        embeddings_are_tied = not model_config.untie_embeddings_and_output_weights
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(
            megatron_model
        )

        # Filter out output_layer related parameters if embeddings are tied
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name
                for name in sorted_global_param_names_all_pp_ranks
                if "output_layer" not in name
            ]

        global_names_index_dict = {
            name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)
        }

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)
        for vp_stage, model in enumerate(megatron_model):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_name, _ in itertools.chain(
                model.named_parameters(), persistent_buffers(model)
            ):
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(
                    megatron_model, model_config, local_name, vp_stage
                )
                # if name removed due to some reason, continue. e.g. embeddings_are_tied
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)
                if not mapping:
                    logger.warning(f"WARNING: No mapping found for megatron_param: {global_name}")
                    continue
                # ensure hf weights exist
                if isinstance(mapping.hf_param, str):
                    if mapping.hf_param not in hf_keys:
                        prefix = '.'.join(mapping.hf_param.split('.')[:-2])
                        if not (('q_proj.weight' in mapping.hf_param) and (
                            f'{prefix}.q_a_layernorm.weight' in hf_keys
                            and f'{prefix}.q_a_proj.weight' in hf_keys
                            and f'{prefix}.q_b_proj.weight' in hf_keys
                        )):
                            logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_keys")
                            continue
                else:
                    missing_params = [
                        hf_param
                        for hf_param in mapping.hf_param.values()
                        if hf_param not in hf_keys
                    ]
                    if missing_params:
                        logger.warning(
                            f"WARNING: Can't find the following HF parameters in hf_keys: {missing_params}"
                        )
                        continue

                local_module, local_weights = get_module_and_param_from_name(
                    megatron_model, local_name, vp_stage
                )
                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        # Fill the remaining ones for pp communications
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if tasks[idx] is None:
                # This is an exception here we pass in global name
                # we are not using global_name to extract module and weights
                # only use it for param mapping auto dispatch checks
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        return tasks

@dispatch
def get_model_bridge(hf_architecture) -> "MegatronModelBridge":
    """Get the appropriate model bridge for a given HuggingFace architecture."""
    ...

@dispatch
def stream_weights_megatron_to_hf(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    hf_pretrained: HFPreTrained,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...



def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string.
            Using a string allows registering bridges for architectures that are
            available only via auto_map.
        target: Megatron model class (e.g., GPTModel)
        bridge_class: MegatronModelBridge implementation class
    """
    bridge_class_name = bridge_class.__name__

    @get_model_bridge.impl(source)
    def _get_model_bridge_impl(_) -> "MegatronModelBridge":
        bridge = bridge_class()
        return bridge

    @stream_weights_megatron_to_hf.impl((source, target))
    def _megatron_to_hf_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()

        # allow bridge to access model config
        bridge.hf_config = hf_pretrained.config

        return bridge.stream_weights_megatron_to_hf(
            megatron_model, hf_pretrained, cpu=cpu, show_progress=show_progress, conversion_tasks=conversion_tasks
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _megatron_to_hf_registered_impl.__name__ = f"_megatron_to_hf_with_{bridge_class_name}"


def create_bridge_decorator(
    *, source: Type["PreTrainedModel"] | str, target: Type["MegatronModule"]
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string
            (useful for auto_map architectures)
        target: Megatron model class

    Returns:
        Decorator function that registers the bridge implementation
    """

    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class

    return decorator
