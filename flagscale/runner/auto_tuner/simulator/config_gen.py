import ast
import json
import os

from functools import reduce
from itertools import combinations, product

import analylize_pipeline_time

# from itertools import product
import flagscale.train.theoretical_memory_usage as mem_usg

BYTES_OF_GB = 10**9


device_type_list = ["A800", "A800"]
device_num_list = [4, 4]
memory_capacity_of_devices = [80, 80]  # GB

global_batch_size = 512
num_micro_batches = 8
num_layers = 32

num_gpus = sum(device_num_list)


class DevicesInfo:
    def __init__(self, device_type_list: list, device_num_list: list):
        assert len(device_type_list) == len(
            device_num_list
        ), "\flength of list {device_type_list} should match {device_num_list}"
        self.device_type_list = device_type_list
        self.device_num_list = device_num_list
        self.device_types_count = len(device_type_list)
        self.possible_parallelisms = []


class HeteroConfig:
    def __init__(
        self,
        mesh,
        device_types,
        pp_layer_split,
        recompute_granularity=None,
        recompute_method="uniform",
        recompute_num_layers=1,
        theory_peak_memory=0.0,
        oom_error=False,
    ):
        self.mesh = mesh
        self.device_types = device_types
        self.pp_layer_split = pp_layer_split
        # self.micro_batch_size = 1
        self.recompute_granularity = recompute_granularity
        self.recompute_method = recompute_method
        self.recompute_num_layers = recompute_num_layers

        self.simulated_time = 0.0
        self.theory_peak_memory = theory_peak_memory
        self.oom_error = oom_error


def generate_hetero_meshes(
    devices_info: DevicesInfo,
    global_batch_size: int = None,
    num_layers: int = None,
    output_file: str = "results.json",
):
    def enumerate_parallelism(device_num: int = None):
        possible_parallelisms = []
        for tp in range(1, device_num + 1):
            for dp in range(1, device_num // tp + 1):
                if device_num % (dp * tp) == 0:
                    pp = device_num // (dp * tp)
                    # mesh: [tp, cp, ep, dp, pp]
                    possible_parallelisms.append([tp, 1, 1, dp, pp])
        return possible_parallelisms

    def is_legal_combination(comb: list):
        pp = sum(comb[4::5])
        # check dp is legal
        max_dp = global_batch_size // pp
        for dp in comb[3::5]:
            if max_dp % dp != 0:
                return False
        for i in range(len(comb) // 5):
            tp, _, _, dp, pp = comb[i * 5 : i * 5 + 5]
            device_num = devices_info.device_num_list[i]
            if tp * dp * pp != device_num:
                return False
        return True

    def is_extreme_strategy(comb: list):
        for mesh_index in range(len(comb) // 5):
            num_devices_in_mesh = reduce(
                lambda x, y: x * y, comb[mesh_index * 5 : mesh_index * 5 + 5]
            )
            dp_size_in_mesh = comb[mesh_index * 5 + 3]
            tp_size_in_mesh = comb[mesh_index * 5 + 0]
            pp_size_in_mesh = comb[mesh_index * 5 + 4]
            print(
                mesh_index,
                comb[mesh_index * 5 : mesh_index * 5 + 5],
                num_devices_in_mesh,
                dp_size_in_mesh,
                tp_size_in_mesh,
                pp_size_in_mesh,
            )
            if (
                pp_size_in_mesh > num_devices_in_mesh // 2
                or tp_size_in_mesh > 8
                or dp_size_in_mesh > num_devices_in_mesh / 4
            ):
                return True
            else:
                return False

    def combine_possible_parallelisms(possible_parallelisms, output_file):
        '''Combine and filter results, writing them to a file to avoid OOM.'''
        all_combinations = product(*possible_parallelisms)
        with open(output_file, "w") as f:
            for comb in all_combinations:
                result = sum(comb, [])
                if is_legal_combination(result):
                    if not is_extreme_strategy(result):
                        f.write(",".join(map(str, result)) + "\n")

    # Ensure output file does not exist initially
    if os.path.exists(output_file):
        os.remove(output_file)

    # Enumerate all possible meshes for each kind of device
    for i in range(devices_info.device_types_count):
        device_num = devices_info.device_num_list[i]
        devices_info.possible_parallelisms.append(enumerate_parallelism(device_num))

    # Combine possibilities and write results to file
    combine_possible_parallelisms(devices_info.possible_parallelisms, output_file)
    print(f"Results written to {output_file}")


def extract_mesh_stage_structure(mesh):
    stage_counts = []
    for i in range(0, len(mesh), 5):
        stage_counts.append(mesh[i + 4])
    return stage_counts


def split_layers(num_layers, pp_stages, mesh):
    results = []
    mesh_stage_counts = extract_mesh_stage_structure(mesh)
    # print(pp_stages)
    for split_points in combinations(range(1, num_layers), pp_stages - 1):
        # print(split_points)
        if len(split_points) == 0:
            continue
        splits = (
            [split_points[0]]
            + [split_points[i] - split_points[i - 1] for i in range(1, len(split_points))]
            + [num_layers - split_points[-1]]
        )
        # to prune some extreme splits
        stage_index = 0
        mesh_total_layers = []
        violate = False

        for m in mesh_stage_counts:
            sub_splits = splits[stage_index : stage_index + m]
            stage_index += m

            if not sub_splits:
                continue

            if max(sub_splits) - min(sub_splits) > 4:
                violate = True
                break

            mesh_total_layers.append(sum(sub_splits))

        if violate:
            continue

        if max(mesh_total_layers) - min(mesh_total_layers) > 4:
            continue

        results.append(splits)

    return results


class MeshArguments:
    def __init__(self, mesh_config: HeteroConfig):
        # [tp, cp, ep, dp, pp]
        self.data_parallel_size = mesh_config.mesh[3]
        # TODO: pp size not correct when computing memory, because former method divides the layers evenly
        # no embed and dropout for stages except the 1st, and make the layers changable

        # if args.pipeline_model_parallel_size > 1:
        #     activation_memory = (
        #         perlayer_activation
        #         * args.num_layers
        #         / args.pipeline_model_parallel_size
        #         * in_flight_microbatches
        #         + embedding_activation_memory
        #         + dropout_activation_memory
        #     )
        # else:
        #     activation_memory = (
        #         perlayer_activation * args.num_layers
        #         + embedding_activation_memory
        #         + dropout_activation_memory
        #         + output_layer_and_loss_activation_memory
        #     )
        self.pipeline_model_parallel_size = sum(mesh_config.mesh[4::5])
        self.tensor_model_parallel_size = mesh_config.mesh[0]
        self.virtual_pipeline_model_parallel_size = None
        self.num_experts = None

        self.context_parallel_size = 1
        self.swiglu = True
        self.micro_batch_size = global_batch_size / num_micro_batches / self.data_parallel_size
        self.num_layers = num_layers
        self.num_attention_heads = 32
        self.group_query_attention = None  # TODO
        self.num_query_groups = 1  # TODO
        # self.moe_layer_freq = 2
        # self.moe_router_topk = 1
        self.multi_latent_attention = False
        self.seq_length = 2048
        self.padded_vocab_size = 4096  # TODO
        self.hidden_size = 4096
        self.qk_layernorm = False
        self.mtp_num_layers = None
        self.expert_model_parallel_size = 1
        self.world_size = 8
        self.moe_shared_expert_intermediate_size = None
        self.moe_ffn_hidden_size = None
        ## self.ffn_hidden_size
        self.multiple_of = 256
        hidden_dim = int(4 * self.hidden_size * 2 / 3)
        self.ffn_hidden_size = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )
        # self.kv_channels
        self.kv_channels = self.hidden_size // self.num_attention_heads

        self.recompute_granularity = mesh_config.recompute_granularity
        self.recompute_method = mesh_config.recompute_method
        self.recompute_num_layers = mesh_config.recompute_num_layers

        self.expert_tensor_parallel_size = 1

        self.use_flash_attn = True
        self.sequence_parallel = True
        self.use_distributed_optimizer = True
        self.untie_embeddings_and_output_weights = False  # TODO

        self.enable_hetero = True


def report_oom_error(
    memory_capacity_of_devices: list, meshes_config: list, peak_memory_usage_per_stage: list
):
    stage_index = 0
    for mesh_index, num_stages_in_current_mesh in enumerate(meshes_config[4::5]):
        for i in range(num_stages_in_current_mesh):
            if (
                peak_memory_usage_per_stage[stage_index + i]
                >= memory_capacity_of_devices[mesh_index]
            ):
                return True
        stage_index = stage_index + num_stages_in_current_mesh
    return False


def calculate_peak_memory_per_stage(mesh_config):
    peak_memory_usage_per_stage = []
    model_parallel_training_args = MeshArguments(mesh_config)
    stage_index = 0
    mesh_index = 0
    for pp_stage_num_per_mesh in mesh_config.mesh[4::5]:
        model_parallel_training_args.data_parallel_size = mesh_config.mesh[3 + 5 * mesh_index]
        model_parallel_training_args.tensor_model_parallel_size = mesh_config.mesh[
            0 + 5 * mesh_index
        ]
        for stage in range(pp_stage_num_per_mesh):
            model_parallel_training_args.num_layers = mesh_config.pp_layer_split[stage_index]

            peak_activation_memory_usage = mem_usg.compute_activation_memory(
                args=model_parallel_training_args, num_microbatches=num_micro_batches
            )
            peak_weight_optimizer_usage = mem_usg.compute_weight_and_optimizer_memory(
                args=model_parallel_training_args
            )
            peak_memory_usage = peak_activation_memory_usage + peak_weight_optimizer_usage

            peak_memory_usage_per_stage.append(peak_memory_usage / BYTES_OF_GB)
            stage_index = stage_index + 1

        mesh_index = mesh_index + 1

    return peak_memory_usage_per_stage


def gen_hetero_configs(
    device_type_list,
    device_num_list,
    global_batch_size,
    num_layers,
    # num_micro_batches,
    # hetero_configs: list,
    output_config_file: str = "hetero_configs.json",
):
    devices_info = DevicesInfo(device_type_list=device_type_list, device_num_list=device_num_list)

    generate_hetero_meshes(
        devices_info=devices_info,
        global_batch_size=global_batch_size,
        num_layers=num_layers,
        output_file="results.json",
    )

    hetero_meshes = []
    with open("results.json", "r") as f:
        for line in f:
            hetero_meshes.append(list(map(int, line.strip().split(","))))
    # print(hetero_meshes)
    # assert False
    seen = set()
    with open(output_config_file, "w") as config_file:  # 打开输出文件
        for mesh in hetero_meshes:
            pp_stages = sum(mesh[4::5])
            # in order to prune the num of layers in each stage to even number
            pp_layer_splits = split_layers(num_layers=num_layers, pp_stages=pp_stages, mesh=mesh)
            for split in pp_layer_splits:
                split = [x for x in split]
                hetero_config = HeteroConfig(
                    mesh=mesh, pp_layer_split=split, device_types=device_type_list
                )
                # hetero_configs.append(hetero_config)
                theory_peak_memory_per_stage = calculate_peak_memory_per_stage(hetero_config)
                oom_error = report_oom_error(
                    memory_capacity_of_devices=memory_capacity_of_devices,
                    meshes_config=mesh,
                    peak_memory_usage_per_stage=theory_peak_memory_per_stage,
                )
                # if oom_error:
                #     continue

                key = (tuple(mesh), tuple(split), tuple(device_type_list))
                if key in seen:
                    continue  # 跳过重复项
                seen.add(key)

                config_data = {
                    "mesh": hetero_config.mesh,
                    "device_types": hetero_config.device_types,
                    "pp_layer_split": hetero_config.pp_layer_split,
                    "recompute_granularity": hetero_config.recompute_granularity,
                    "recompute_method": hetero_config.recompute_method,
                    "recompute_num_layers": hetero_config.recompute_num_layers,
                    "simulated_time": hetero_config.simulated_time,
                    "theory_peak_memory": theory_peak_memory_per_stage,
                    "oom_error": oom_error,
                }
                config_file.write(f"{config_data}\n")

    print(f"Hetero configurations saved to {output_config_file}")


import ast
import json


def read_configs_from_json(file_path: str):
    configs_list = []
    with open(file_path, "r") as file:
        for line in file:
            # config_data = json.loads(line.strip())
            config_data = ast.literal_eval(line.strip())
            configs_list.append(config_data)
    return configs_list


def get_min_simulated_time_config(hetero_configs):
    if not hetero_configs:
        return None
    return min(hetero_configs, key=lambda x: x.get("simulated_time", float("inf")))


def append_config_to_file(file_path: str, config: dict):
    with open(file_path, "a") as f:
        f.write(str(config) + "\n")


# for test and usage
if __name__ == "__main__":
    # hetero_configs = []

    # generate all possible and legal mesh configs, each element of hetero_configs is a mesh list
    gen_hetero_configs(
        device_type_list=device_type_list,
        device_num_list=device_num_list,
        global_batch_size=global_batch_size,
        num_layers=num_layers,
        output_config_file="hetero_configs.json",
        # num_micro_batches=num_micro_batches,
        # hetero_configs=hetero_configs
    )

    # assert False
    # simulation
    file_path = "hetero_configs.json"
    result_path = "simulate_time.json"
    hetero_configs = read_configs_from_json(file_path)
    for hetero_config in hetero_configs:
        print(hetero_config)
        pp_cost = hetero_config['simulated_time'] = analylize_pipeline_time.analyze_pp_time(
            # pp_cost = hetero_config.simulated_time = analylize_pipeline_time.analyze_pp_time(
            scheme="vpp",
            # scheme="1F1B",
            num_micro_batches=num_micro_batches,
            process_mesh=hetero_config['mesh'],
            pp_layers_split=hetero_config['pp_layer_split'],
        )
        print(f"pipeline cost: {pp_cost}")
        append_config_to_file(result_path, hetero_config)

    best_config = get_min_simulated_time_config(hetero_configs)
    print(best_config)
