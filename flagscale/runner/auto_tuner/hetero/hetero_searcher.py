import copy
import itertools
import logging
import time

from omegaconf import DictConfig, ListConfig, OmegaConf

from flagscale.runner.auto_tuner.hetero.hetero_theoretical_memory import (
    hetero_report_theoretical_memory,
)

# from flagscale.runner.auto_tuner.memory_model import calculate_hetero_memory
from flagscale.runner.auto_tuner.search.algorithm import GridAlgo
from flagscale.runner.auto_tuner.utils import convert_config_to_megatron_args, divisible


def calculate_hetero_memory(strategy, config):
    """Calculates theoretical memory for a heterogeneous strategy."""
    # Get base args using compatibility keys
    base_args = convert_config_to_megatron_args(config, strategy)

    # Add global batch size to base_args if not present
    if not hasattr(base_args, "global_batch_size"):
        base_args.global_batch_size = config.train.model.global_batch_size

    # Call the dedicated hetero memory calculation function
    # This will now return a LIST [stage0_mem, stage1_mem, ...] or float('inf')
    total_memory_list_or_inf = hetero_report_theoretical_memory(
        strategy=strategy,
        config=config,
        base_args=base_args,
        # verbose=True # For debugging
    )
    return total_memory_list_or_inf


def _generate_all_partitions_with_max_diff(n, k, max_diff):
    """
    Generates all integer partitions of n into k parts with max difference constraint.
    """
    if k == 0:
        if n == 0:
            yield []
        return
    if k == 1:
        if n > 0:
            yield [n]
        return

    # Optimized bounds
    start = (n + k - 1) // k
    end = n - (k - 1) + 1

    for i in range(start, end):
        if n - i < (k - 1):
            continue
        for rest in _generate_all_partitions_with_max_diff(n - i, k - 1, max_diff):
            if i < rest[0]:
                continue
            if i - rest[-1] > max_diff:
                continue
            yield [i, *rest]


class HeteroSearcher:
    """
    The Optimized HeteroSearcher.
    Combines:
    1. Staged Generation Pipeline (High Performance)
    2. Early Pruning (Prevents Explosion)
    3. Specific Business Logic for SP/TP/DP (From Original Implementation)
    4. Hetero-Specific Recompute Logic (Restored)
    """

    def __init__(self, config, resources):
        self.logger = logging.getLogger("FlagScale-AutoTuner")
        self.config = config
        self.resources = resources
        self.recompute_search_space = {}

        # Build search space
        start_time = time.time()
        self.space = self.build_space(self.config)
        end_time = time.time()
        self.logger.info(
            f"HeteroSearcher: build search space in {end_time - start_time:.2f} seconds."
        )

        # Build strategies
        start_time = time.time()
        self.strategies = self.build_strategies(self.space, self.config)
        end_time = time.time()
        self.logger.info(
            f"HeteroSearcher: build {len(self.strategies)} candidate strategies in {end_time - start_time:.2f} seconds."
        )

        # Calculate Memory Model
        if "hetero_memory_model" in self.config.experiment.auto_tuner:
            self.logger.info("HeteroSearcher: calculating memory estimates...")
            for idx, strategy in enumerate(self.strategies):
                try:
                    strategy["hetero_memory_model"] = calculate_hetero_memory(strategy, self.config)
                except Exception:
                    # self.logger.error(f"Failed to calculate memory for strategy {idx}: {e}")
                    strategy["hetero_memory_model"] = float("inf")

        # Build search algorithm
        self.algo = self.build_algo(self.strategies, self.config)

    def _sort(self, key, dim, priority=None):
        if priority is not None:
            if key in ["micro_batch_size"]:
                if priority == "memory":
                    dim.sort()
                elif priority == "performance":
                    dim.sort(reverse=True)
            elif key in ["use_recompute"]:
                if priority == "memory":
                    dim.sort(reverse=True)
                elif priority == "performance":
                    dim.sort()
            elif key in ["use_distributed_optimizer"]:
                if priority == "memory":
                    dim.sort(reverse=True)

    def build_space(self, config):
        space = {}
        auto_tuner_config = config.experiment.auto_tuner
        hetero_space = auto_tuner_config.space

        if "algo" not in auto_tuner_config:
            auto_tuner_config.algo = {"name": "grid", "priority": None}
        priority = auto_tuner_config.algo.get("priority", None)

        def safe_to_container(value):
            if isinstance(value, (DictConfig, ListConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        # 1. Hardware & Templates
        raw_mesh_templates = safe_to_container(hetero_space.get("hetero_process_meshes", []))
        if not raw_mesh_templates or len(raw_mesh_templates) % 5 != 0:
            raise ValueError("'hetero_process_meshes' must be a non-empty list divisible by 5.")

        space["mesh_templates"] = [
            raw_mesh_templates[i : i + 5] for i in range(0, len(raw_mesh_templates), 5)
        ]
        space["device_types"] = safe_to_container(
            config.train.system.hetero.get("hetero_device_types", [])
        )

        # 2. Constraints
        space["hetero_pipeline_layer_split"] = safe_to_container(
            hetero_space.get("hetero_pipeline_layer_split", "auto")
        )
        space["inter_mesh_max_diff"] = safe_to_container(
            hetero_space.get("hetero_inter_mesh_max_layer_diff", "auto")
        )
        space["intra_mesh_max_diff"] = safe_to_container(
            hetero_space.get("hetero_intra_mesh_max_layer_diff", "auto")
        )

        # 3. Training Params

        space["micro_batch_size"] = safe_to_container(hetero_space.get("micro_batch_size", [1]))
        self._sort("micro_batch_size", space["micro_batch_size"], priority)

        space["use_distributed_optimizer"] = safe_to_container(
            hetero_space.get("use_distributed_optimizer", [True, False])
        )
        self._sort("use_distributed_optimizer", space["use_distributed_optimizer"], priority)

        space["use_recompute"] = safe_to_container(hetero_space.get("use_recompute", [True, False]))
        self._sort("use_recompute", space["use_recompute"], priority)

        # [Hetero Recompute Config Parsing]
        # We store these in self.recompute_search_space for complex template parsing later
        self.recompute_search_space["use_recompute"] = space["use_recompute"]
        self.recompute_search_space["granularity"] = safe_to_container(
            hetero_space.get("recompute_granularity_per_stage_micro_batch", "auto")
        )
        self.recompute_search_space["method"] = safe_to_container(
            hetero_space.get("recompute_method_per_stage_micro_batch", "auto")
        )
        self.recompute_search_space["num_layers"] = safe_to_container(
            hetero_space.get("recompute_num_layers_per_stage_micro_batch", "auto")
        )

        space["sequence_parallel"] = safe_to_container(
            hetero_space.get("sequence_parallel", [True, False])
        )

        return space

    def build_strategies(self, space, config):
        # Stage 1: Hardware Assignment (With Early Pruning)
        assignment_part = self._product_assignment_dims(space, config)
        self.logger.info(
            f"HeteroSearcher: Generated {len(assignment_part)} valid hardware assignments."
        )

        # Stage 2: Layer Split Generation
        layer_split_part = self._product_layer_split_dims(assignment_part, space, config)
        self.logger.info(
            f"HeteroSearcher: Generated {len(layer_split_part)} strategies after layer splitting."
        )

        # Stage 3: Training Params (Merged Logic)
        final_strategies = self._product_training_params_dims(layer_split_part, space, config)

        return final_strategies

    def build_algo(self, strategies, config):
        name = self.config.experiment.auto_tuner.algo.name
        if name == "grid":
            return GridAlgo(strategies, self.config)
        else:
            raise NotImplementedError("Currently only grid search is supported.")

    def search(self):
        return self.algo.search()

    def has_done(self):
        return self.algo.has_done()

    # Hetero Recompute Config Generator
    def _generate_recompute_configs(self, pp_size: int, num_micro_batches: int) -> list[dict]:
        """Generates dynamic recompute configurations based on hetero templates."""
        if pp_size == 0:
            return [{}]

        def get_options_for(key: str) -> list[list]:
            user_config = self.recompute_search_space.get(key)
            if user_config == "auto":
                # 'auto' provides simple templates: [all off] and [all on (1 layer)]
                auto_templates = [[[pp_size, "ALL", 0]], [[pp_size, "ALL", 1]]]
                if key == "num_layers":
                    return [[[pp_size, "ALL", 1]]]  # 'auto' for num_layers is just 1
                return auto_templates
            elif isinstance(user_config, list):
                # Validate user provided templates
                valid_options = []
                for template_list in user_config:
                    if isinstance(template_list, list):
                        # Check if stages sum up to PP size
                        total_stages_in_template = sum(item[0] for item in template_list)
                        if total_stages_in_template == pp_size:
                            valid_options.append(template_list)
                return valid_options
            return []

        granularity_options = get_options_for("granularity")
        method_options = get_options_for("method")
        num_layers_options = get_options_for("num_layers")

        # If any option list is empty, we can't form a valid combo
        if not granularity_options or not method_options or not num_layers_options:
            return [{}]

        all_recompute_combinations = []
        for gran_list, meth_list, num_list in itertools.product(
            granularity_options, method_options, num_layers_options
        ):

            def render_template(template_list):
                """Replaces 'ALL' placeholder with actual MBS count."""
                rendered_list = []
                for item in template_list:
                    # item format: [num_stages, 'ALL'|int, val]
                    rendered_item = [val if val != "ALL" else num_micro_batches for val in item]
                    rendered_list.append(rendered_item)
                return rendered_list

            all_recompute_combinations.append(
                {
                    "recompute_granularity_per_stage_micro_batch": render_template(gran_list),
                    "recompute_method_per_stage_micro_batch": render_template(meth_list),
                    "recompute_num_layers_per_stage_micro_batch": render_template(num_list),
                }
            )

        return all_recompute_combinations if all_recompute_combinations else [{}]

    # Stage 1: Assignment Generation Logic
    def _product_assignment_dims(self, space, config):
        result = []
        unique_result = set()

        available_nodes = []
        if not self.resources:
            raise ValueError("HeteroSearcher requires resources (hostfile info).")

        for hostname, info in self.resources.items():
            available_nodes.append(
                {
                    "name": hostname,
                    "type": info.get("type", "default_gpu"),
                    "slots": info.get("slots", 8),
                }
            )

        # Check if fixed PP size is implied by fixed splits
        target_global_pp = None
        split_config = space.get("hetero_pipeline_layer_split")
        if isinstance(split_config, list):
            if len(split_config) > 0:
                first = split_config[0]
                target_global_pp = len(split_config) if isinstance(first, int) else len(first)

        assignments = self._find_valid_assignments_recursive(
            0,
            available_nodes,
            [],
            space["mesh_templates"],
            space["device_types"],
            config,
            target_global_pp,
            space["micro_batch_size"],
        )

        for assignment in assignments:
            dims = {}
            dims["hetero_process_meshes"] = [item["mesh"] for item in assignment]
            dims["hetero_device_types"] = [item["device_type"] for item in assignment]
            self._append(result, unique_result, dims)

        return result

    def _find_valid_assignments_recursive(
        self,
        mesh_idx,
        available_nodes,
        current_assignment,
        mesh_templates,
        device_types,
        config,
        target_pp,
        mbs_candidates,
    ):
        # Base case
        if mesh_idx == len(mesh_templates):
            if target_pp is not None:
                if sum(item["mesh"][4] for item in current_assignment) != target_pp:
                    return []
            return [current_assignment]

        results = []
        current_template = mesh_templates[mesh_idx]
        current_device_type = device_types[mesh_idx]

        candidate_nodes = [n for n in available_nodes if n["type"] == current_device_type]
        if not candidate_nodes:
            return []

        # Pruning: Check if PP already exceeded
        current_pp = sum(item["mesh"][4] for item in current_assignment)
        if target_pp is not None and current_pp >= target_pp:
            return []

        for k in range(1, len(candidate_nodes) + 1):
            for nodes_to_assign in itertools.combinations(candidate_nodes, k):
                # Early Pruning: Check DP/GBS compatibility inside here
                valid_configs = self._get_valid_mesh_configs(
                    current_template, nodes_to_assign, config, mbs_candidates
                )
                if not valid_configs:
                    continue

                remaining = [n for n in available_nodes if n not in nodes_to_assign]

                for mesh_config in valid_configs:
                    # Constraint: Tied embeddings
                    if not config.train.model.get("untie_embeddings_and_output_weights", False):
                        if mesh_idx == len(mesh_templates) - 1 and current_assignment:
                            first_tp = current_assignment[0]["mesh"][0]
                            last_tp = mesh_config[0]
                            if first_tp != last_tp:
                                continue

                    new_step = {
                        "mesh": mesh_config,
                        "device_type": current_device_type,
                        "nodes": nodes_to_assign,
                    }

                    results.extend(
                        self._find_valid_assignments_recursive(
                            mesh_idx + 1,
                            remaining,
                            [*current_assignment, new_step],
                            mesh_templates,
                            device_types,
                            config,
                            target_pp,
                            mbs_candidates,
                        )
                    )
        return results

    def _get_valid_mesh_configs(self, template, nodes, config, mbs_candidates):
        total_gpus = sum(n["slots"] for n in nodes)
        if total_gpus == 0:
            return []

        def get_candidates(val, limit):
            if val == "auto":
                return [i for i in range(1, limit + 1) if limit % i == 0]
            if isinstance(val, (list, tuple)):
                return sorted([v for v in val if v <= limit])
            try:
                v = int(val)
                return [v] if v <= limit else []
            except:
                return []

        tp_opts = get_candidates(template[0], total_gpus)
        cp_opts = get_candidates(template[1], total_gpus)
        ep_opts = get_candidates(template[2], total_gpus)
        pp_opts = get_candidates(template[4], total_gpus)
        dp_tmpl = template[3]
        gbs = config.train.model.global_batch_size

        valid_configs = []
        for tp in tp_opts:
            if config.train.model.hidden_size % tp != 0:
                continue
            for cp in cp_opts:
                for ep in ep_opts:
                    for pp in pp_opts:
                        prod = tp * cp * ep * pp
                        if prod == 0 or prod > total_gpus or total_gpus % prod != 0:
                            continue

                        dp = total_gpus // prod
                        dp_opts = get_candidates(dp_tmpl, total_gpus)

                        if dp in dp_opts:
                            # Early Pruning: GBS check
                            is_viable = False
                            for mbs in mbs_candidates:
                                if divisible(gbs, dp * mbs):
                                    is_viable = True
                                    break
                            if is_viable:
                                valid_configs.append([tp, cp, ep, dp, pp])
        return valid_configs

    # Stage 2: Layer Split Generation Logic
    def _product_layer_split_dims(self, assignment_part, space, config):
        result = []
        unique_result = set()
        total_layers = config.train.model.num_layers

        # Inter-Mesh: Full Permutations (Find optimal load balance between A800/MLU)
        # Intra-Mesh: Top-5 Permutations (Find optimal memory balance inside a Mesh, e.g., avoid heavy Stage 0)

        for strategy in assignment_part:
            pp_sizes = [mesh[4] for mesh in strategy["hetero_process_meshes"]]
            global_pp_size = sum(pp_sizes)
            valid_splits = []
            split_config = space["hetero_pipeline_layer_split"]

            if split_config == "auto":
                # 1. Constraints
                inter_diff = (
                    space["inter_mesh_max_diff"]
                    if isinstance(space["inter_mesh_max_diff"], int)
                    else total_layers
                )
                intra_diff = (
                    space["intra_mesh_max_diff"]
                    if isinstance(space["intra_mesh_max_diff"], int)
                    else total_layers
                )

                # 2. Generate Inter-Mesh Partitions
                inter_parts = _generate_all_partitions_with_max_diff(
                    total_layers, len(pp_sizes), inter_diff
                )

                for part in inter_parts:
                    # Permute Mesh-to-Mesh assignments
                    for dist in set(itertools.permutations(part)):
                        possible_splits_per_mesh = []
                        is_distribution_possible = True

                        # 3. Generate Intra-Mesh Splits
                        for i, layers_for_this_mesh in enumerate(dist):
                            local_pp = pp_sizes[i]
                            if layers_for_this_mesh < local_pp:
                                is_distribution_possible = False
                                break

                            # Generate partitions (e.g. 18 -> [5, 5, 4, 4])
                            mesh_i_partitions = _generate_all_partitions_with_max_diff(
                                layers_for_this_mesh, local_pp, intra_diff
                            )

                            # [Restored Optimization: Top 5 Permutations]
                            # We need permutations to test if [4,4,5,5] is better than [5,5,4,4] for Stage 0 memory.
                            # Limiting to 5 prevents explosion while keeping diversity.
                            mesh_i_candidates = []
                            for p in mesh_i_partitions:
                                perms = list(set(itertools.permutations(p)))
                                mesh_i_candidates.extend(perms[:5])  # Keep top 5 variations

                            if not mesh_i_candidates:
                                is_distribution_possible = False
                                break
                            possible_splits_per_mesh.append(mesh_i_candidates)

                        # 4. Combine
                        if is_distribution_possible:
                            for combined_splits_tuple in itertools.product(
                                *possible_splits_per_mesh
                            ):
                                final_split = []
                                for single_mesh_split in combined_splits_tuple:
                                    final_split.extend(single_mesh_split)
                                valid_splits.append(final_split)

            elif isinstance(split_config, (list, tuple)):
                if len(split_config) > 0 and isinstance(split_config[0], (list, tuple)):
                    for c in split_config:
                        if len(c) == global_pp_size and sum(c) == total_layers:
                            valid_splits.append(list(c))
                elif len(split_config) == global_pp_size and sum(split_config) == total_layers:
                    valid_splits.append(list(split_config))

            for split in valid_splits:
                new_s = copy.deepcopy(strategy)
                new_s["hetero_pipeline_layer_split"] = split
                new_s["pipeline_model_parallel_size"] = global_pp_size
                self._append(result, unique_result, new_s)

        return result

    # Stage 3: Training Parameters Logic (With Specific SP & Recompute Logic)
    def _product_training_params_dims(self, layer_split_part, space, config):
        result = []
        unique_result = set()
        gbs = config.train.model.global_batch_size

        for strategy in layer_split_part:
            dp_list = [mesh[3] for mesh in strategy["hetero_process_meshes"]]
            pp_size = strategy["pipeline_model_parallel_size"]

            for mbs in space["micro_batch_size"]:
                # 1. Check GBS compatibility
                if not all(dp > 0 and divisible(gbs, dp * mbs) for dp in dp_list):
                    continue

                # 2. Check Hetero DP Alignment (Base Batch Unit)
                first_dp = dp_list[0]
                base_unit = first_dp * mbs
                if not all(dp > 0 and divisible(base_unit, dp) for dp in dp_list):
                    continue

                for use_do in space["use_distributed_optimizer"]:
                    if all(dp == 1 for dp in dp_list) and use_do:
                        continue

                    # Specific SP Logic
                    tp_list = [mesh[0] for mesh in strategy["hetero_process_meshes"]]
                    all_tp_are_one = all(tp == 1 for tp in tp_list)
                    tps_are_mixed = len(set(tp_list)) > 1
                    added_effective_sps = set()

                    for sp_option in space["sequence_parallel"]:
                        effective_sp = sp_option

                        # Rule 1: Mixed TP -> Must enable SP
                        if tps_are_mixed:
                            if not sp_option:
                                continue
                            effective_sp = True
                        # Rule 2: All TP=1 -> SP is effectively False (Prune True)
                        elif all_tp_are_one:
                            effective_sp = False

                        if effective_sp in added_effective_sps:
                            continue
                        added_effective_sps.add(effective_sp)

                        # Determine Micro Batch Count for this configuration
                        dp_0 = strategy["hetero_process_meshes"][0][3]
                        if (mbs * dp_0) == 0:
                            num_micro_batches = gbs
                        else:
                            num_micro_batches = gbs // (mbs * dp_0)
                        if num_micro_batches == 0:
                            num_micro_batches = 1

                        # Expand Recompute Options based on 'use_recompute' flag
                        use_recompute_options = self.recompute_search_space["use_recompute"]

                        for use_recompute in use_recompute_options:
                            base_dim = copy.deepcopy(strategy)
                            base_dim["micro_batch_size"] = mbs
                            base_dim["use_distributed_optimizer"] = use_do
                            base_dim["sequence_parallel"] = effective_sp
                            base_dim["use_recompute"] = use_recompute

                            # Compatibility Fields
                            m0 = strategy["hetero_process_meshes"][0]
                            base_dim["tensor_model_parallel_size"] = m0[0]
                            base_dim["context_parallel_size"] = m0[1]
                            base_dim["expert_model_parallel_size"] = m0[2]
                            base_dim["data_parallel_size"] = m0[3]
                            base_dim["num_layers_per_virtual_pipeline_stage"] = None
                            base_dim["decoder_first_pipeline_num_layers"] = None
                            base_dim["decoder_last_pipeline_num_layers"] = None

                            if not use_recompute:
                                base_dim["recompute_method"] = None
                                base_dim["recompute_granularity"] = None
                                base_dim["recompute_num_layers"] = None
                                self._append(result, unique_result, base_dim)
                            else:
                                # Use the sophisticated template generator for valid hetero configs
                                recompute_configs = self._generate_recompute_configs(
                                    pp_size, num_micro_batches
                                )
                                for r_cfg in recompute_configs:
                                    if r_cfg:
                                        # Flatten the config into the strategy dict
                                        # Map to standard keys for generator compatibility if uniform,
                                        # or keep as complex keys if generator supports it.
                                        # Assuming standard keys for simple 'auto' case to match previous logic:
                                        # If template logic returns list, we use it.

                                        # NOTE: To maintain compatibility with Homogeneous Runner which expects
                                        # simple 'recompute_method' strings if possible:
                                        # We set the complex keys, and also set simple keys as proxies for the first stage/MBS.

                                        final_dim = copy.deepcopy(base_dim)
                                        final_dim.update(r_cfg)

                                        # Set proxies for compatibility (e.g. for memory model / runner validation)
                                        # Taking the first element of the template as representative
                                        try:
                                            final_dim["recompute_granularity"] = (
                                                "full"  # Approximation
                                            )
                                            final_dim["recompute_method"] = (
                                                "uniform"  # Approximation
                                            )
                                            final_dim["recompute_num_layers"] = 1  # Approximation
                                        except:
                                            pass

                                        self._append(result, unique_result, final_dim)

        return result

    def _append(self, result, unique_result, product_dim):
        """Helper to append unique strategies, handling list types in hashing."""

        def make_hashable(d):
            new_d = {}
            for k, v in d.items():
                if isinstance(v, list):
                    if len(v) > 0 and isinstance(v[0], list):
                        new_d[k] = tuple(tuple(x) for x in v)
                    else:
                        new_d[k] = tuple(v)
                else:
                    new_d[k] = v
            return tuple(sorted(new_d.items()))

        sorted_items = make_hashable(product_dim)
        if sorted_items not in unique_result:
            unique_result.add(sorted_items)
            copied_dim = copy.deepcopy(product_dim)
            result.append(copied_dim)
