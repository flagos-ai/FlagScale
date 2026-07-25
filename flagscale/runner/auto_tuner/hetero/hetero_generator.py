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

from flagscale.runner.auto_tuner.generate import Generator


class HeteroGenerator(Generator):
    """
    Generator for heterogeneous strategies. It translates an abstract strategy
    dictionary into a concrete, runnable OmegaConf object.
    """

    def __init__(self, config):
        super().__init__(config)

    def _set_value(self, strategy: dict, config: dict):
        """
        Overrides the base method to write all tunable parameters from the
        strategy into the configuration object.
        """
        # First, call the parent's _set_value to handle all common parameters
        # like MBS, recompute, etc., that are not hetero-specific.
        super()._set_value(strategy, config)

        # Then, set all hetero-specific parameters.
        hetero_config = config.train.system.setdefault("hetero", {})
        hetero_config["enable_hetero"] = True
        hetero_config["hetero_pipeline_layer_split"] = strategy["hetero_pipeline_layer_split"]
        hetero_config["hetero_process_meshes"] = [
            item for sublist in strategy["hetero_process_meshes"] for item in sublist
        ]
        hetero_config["hetero_device_types"] = strategy["hetero_device_types"]

        config.train.system["pipeline_model_parallel_size"] = strategy[
            "pipeline_model_parallel_size"
        ]

        # Override sequence parallel based on the strategy.
        config.train.system["sequence_parallel"] = strategy["sequence_parallel"]

        if strategy.get("use_recompute"):
            if strategy.get("recompute_granularity_per_stage_micro_batch"):
                config.train.system.recompute_granularity_per_stage_micro_batch = strategy[
                    "recompute_granularity_per_stage_micro_batch"
                ]

            if strategy.get("recompute_method_per_stage_micro_batch"):
                config.train.system.recompute_method_per_stage_micro_batch = strategy[
                    "recompute_method_per_stage_micro_batch"
                ]

            if strategy.get("recompute_num_layers_per_stage_micro_batch"):
                config.train.system.recompute_num_layers_per_stage_micro_batch = strategy[
                    "recompute_num_layers_per_stage_micro_batch"
                ]
