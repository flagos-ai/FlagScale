import os
import re
import subprocess
import time

# from megatron.training import get_args


def kill_other_python_processes():
    current_pid = os.getpid()
    clear_cmd = f"pkill -f python -o --signal TERM --ignore \"${current_pid}\""
    subprocess.run(clear_cmd, text=True, shell=True)


def compute_pipeline_parallelism_cost(
    scheme: str = '1F1B',
    # num_stages: int=1,
    num_micro_batches: int = 1,
    process_mesh: list = None,
    pp_layers_split: list = None,
    fwd_time_per_stage_chunk: list = None,
    bwd_time_per_stage_chunk: list = None,
    comm_time_between_stages: list = None,
    vpp_partition: list = None,
    # TODO: add fine-greaied recomputation
):
    print(f"--- Compute Pipeline Cost ---")

    # process_mesh: [tp0,cp0,ep0,dp0,pp0,(tp1,cp1,...)]
    # comm_time_between_stages[i] means the comm time between stage i-1 and stage i
    num_pp_stages = sum(process_mesh[4::5])

    assert (
        len(pp_layers_split) == num_pp_stages
    ), "\flength of list {num_layers_per_stage} should match {num_stages}"
    if scheme == 'vpp':
        num_pp_stages = sum(vpp_partition)

    assert (
        len(fwd_time_per_stage_chunk) == num_pp_stages
    ), "\flength of list {fwd_time_per_stage_chunk} should match {num_stages}"
    assert (
        len(bwd_time_per_stage_chunk) == num_pp_stages
    ), "\flength of list {bwd_time_per_stage_chunk} should match {num_stages}"
    assert (
        len(comm_time_between_stages) == num_pp_stages
    ), "\flength of list {comm_time_between_stages} should match {num_stages}"

    pp_last_stage_time = num_micro_batches * (
        fwd_time_per_stage_chunk[num_pp_stages - 1] + bwd_time_per_stage_chunk[num_pp_stages - 1]
    )
    if num_pp_stages == 1:
        return num_micro_batches * (
            fwd_time_per_stage_chunk[num_pp_stages - 1]
            + bwd_time_per_stage_chunk[num_pp_stages - 1]
        )

    pipeline_cost = 0
    # TODO: consider when comm time > comp time
    # each stage onlt depends on its next stage
    if scheme == '1F1B' or scheme == 'AFAB':
        pipeline_cost = pp_last_stage_time
        for stage_from_last in range(2, num_pp_stages):
            pp_this_stage_overlapped_time = (num_micro_batches - 1) * (
                fwd_time_per_stage_chunk[num_pp_stages - 1]
                + bwd_time_per_stage_chunk[num_pp_stages - 1]
            )
            pp_this_stage_compute_time = (
                fwd_time_per_stage_chunk[num_pp_stages - stage_from_last]
                + bwd_time_per_stage_chunk[num_pp_stages - stage_from_last]
            )
            pp_last_stage_overall_time = (
                pipeline_cost + 2 * comm_time_between_stages[num_pp_stages - stage_from_last + 1]
            )
            # not consider the situation that comm stucks the comp
            # which means the comm time should no more than the comp time(fwd time)
            pipeline_cost = pp_this_stage_compute_time + max(
                pp_last_stage_overall_time, pp_this_stage_overlapped_time
            )
    # else:
    #    raise (ValueError("Scheme must be '1F1B' or 'AFAB'."))
    elif scheme == 'vpp':
        num_vp_stages = len(fwd_time_per_stage_chunk)
        num_pp_stages = len(comm_time_between_stages)  # error
        vstage_to_pp = []
        for i, count in enumerate(vpp_partition):
            vstage_to_pp += [i] * count

        comm_per_vstage = [0.0] * num_vp_stages
        for i in range(num_vp_stages - 1):
            cur_pp, next_pp = vstage_to_pp[i], vstage_to_pp[i + 1]
            if next_pp != cur_pp:
                comm_per_vstage[i] = comm_time_between_stages[cur_pp + 1]

        vp_last_stage_time = num_micro_batches * (
            fwd_time_per_stage_chunk[-1] + bwd_time_per_stage_chunk[-1]
        )
        pipeline_cost = vp_last_stage_time
        for vp_from_last in range(2, num_vp_stages + 1):
            this_vp_idx = num_vp_stages - vp_from_last
            this_stage_fwd = fwd_time_per_stage_chunk[this_vp_idx]
            this_stage_bwd = bwd_time_per_stage_chunk[this_vp_idx]
            this_stage_compute_time = this_stage_fwd + this_stage_bwd

            pp_idx = this_vp_idx % num_pp_stages
            comm_time = comm_time_between_stages[min(pp_idx + 1, num_pp_stages - 1)]

            this_vp_overlapped_time = (num_micro_batches - 1) * this_stage_compute_time

            last_vp_total_time = pipeline_cost + 2 * comm_time

            pipeline_cost = this_stage_compute_time + max(
                this_vp_overlapped_time, last_vp_total_time
            )

    return pipeline_cost


import random


def simulator(
    process_mesh: list = None,
    stage: int = 0,
    num_layers: int = None,
    simulated_rank: int = None,
    pp_layers_split: list = None,
):

    # os.environ["PYTHONPATH"] = "/share/project/heyongzhe/FlagScale/megatron:/share/project/heyongzhe/FlagScale"
    os.environ["PYTHONPATH"] = (
        "/workspace/single_process_simulator_nd/FlagScale:"
        "/workspace/single_process_simulator_nd/FlagScale/third_party/Megatron-LM"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["RANK"] = str(simulated_rank)
    os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = args.world_size
    os.environ["WORLD_SIZE"] = "8"
    # os.environ["WORLD_SIZE"] = "32"
    rdav_endpoint = random.randint(0, 40000)
    os.environ["RDZV_ENDPOINT"] = "localhost:" + str(rdav_endpoint)
    # os.environ["RZDV_ENDPOINT"]="localhost:37832"
    os.environ["RDZV_BACKEND"] = "c10d"
    os.environ["MASTER_ADDR"] = "localhost"

    program_entry = " ./flagscale/train/train_aquila_sft.py "
    simulation_arguments = " --enable-hetero --enable-simulator --distributed-backend dummy "
    # fine_grained_recomputation_args = "--recompute-granularity-per-stage-micro-batch '[1, 1, 1]' --recompute-method-per-stage-micro-batch '[1, 1, 1]' --recompute-num-layers-per-stage-micro-batch '[1, 1, 1]'"
    fine_grained_recomputation_args = ""
    # print(stage)

    pp_layer_split_args = " --hetero-pipeline-layer-split "
    for layers in pp_layers_split:
        pp_layer_split_args = pp_layer_split_args + str(layers) + " "

    process_mesh_str = " --hetero-process-meshes  "
    for dim in process_mesh:
        process_mesh_str = process_mesh_str + str(dim) + " "

    num_pp_stages = sum(process_mesh[4::5])
    pp_size_args = " --pipeline-model-parallel-size " + str(num_pp_stages) + " "

    # TODO: too ugly to show this command in the code, re-organize these parameters in another way later
    train_command = (
        "python "
        + program_entry
        + "--tensor-model-parallel-size 1 --timing-log-level 2  --disable-bias-linear --use-flash-attn --sequence-parallel --use-distributed-optimizer --use-mcore-models --transformer-impl transformer_engine --hetero-device-types A800 A800 --hetero-current-device-type A800   --bf16 --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 --log-interval 1 --log-throughput --tensorboard-log-interval 1 --wandb-project aquila2 --wandb-exp-name test --tensorboard-dir /share/project/heyongzhe/FlagScale/outputs/tensorboard --wandb-save-dir /share/project/heyongzhe/FlagScale/outputs/wandb --num-layers 32 --hidden-size 4096 --num-attention-heads 32 --seq-length 2048 --max-position-embeddings 2048 --norm-epsilon 1e-05 --use-rotary-position-embeddings --no-position-embedding --swiglu --multiple-of 256 --normalization RMSNorm  --untie-embeddings-and-output-weights --init-method-std 0.0165 --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 --clip-grad 1.0 --train-samples 128 --global-batch-size 64 --micro-batch-size 1 --seed 42 --lr 0.0002 --weight-decay 0.01 --adam-beta1 0.9 --adam-beta2 0.95 --lr 0.00015 --min-lr 1.5e-05 --lr-warmup-samples 0 --lr-decay-style cosine --data-path /workspace/FlagScale/datapath/pile_wikipedia_demo  --split 1 --tokenizer-type AquilaTokenizerFS --vocab-file ./examples/aquila/tokenizer/vocab.json --merge-file ./examples/aquila/tokenizer/merges.txt --special-tokens-file ./examples/aquila/tokenizer/special_tokens.txt --vocab-size 100008 "
        + process_mesh_str
        + simulation_arguments
        + pp_layer_split_args
        + fine_grained_recomputation_args
        + pp_size_args
    )

    # enough sleeping time is needed to really kill the survival megatron process
    # as least 5 sec before & after killing can not succeed every time
    print("sleeping...")
    # print(train_command)
    # time.sleep(10)
    kill_other_python_processes()
    # time.sleep(10)
    print("start...")
    result = subprocess.run(train_command, capture_output=True, text=True, shell=True)
    print(result)
    output = result.stdout.strip()
    print(train_command)
    print(output)
    # example output: "[simulatior output] forward: 12.34, backward: 56.78, communication: 90.12"
    match = re.search(r"forward:\s*([\d.]+),\s*backward:\s*([\d.]+)", output)
    if match:
        fwd_time = float(match.group(1))
        bwd_time = float(match.group(2))
        # comm_time = float(match.group(3))
        comm_time = estimate_comm_time_between_stages(1, 2048, 4096)
        print("forward:", fwd_time)
        print("backward:", bwd_time)
        print("communication:", comm_time)
    else:
        raise (
            ValueError(
                "Results not found. Example output: \"[simulatior output] forward: 12.34, backward: 56.78, communication: 90.12\""
            )
        )
    return fwd_time, bwd_time, comm_time


def compute_vpp_from_layers(
    pp_layers_split, target_layers_per_vstage=2, device_speed=None, min_layers_per_virtual_stage=2
):
    """
    Args:
        pp_layers_split: list[int]
        target_layers_per_vstage: int
        device_speed: list[float]
        min_layers_per_virtual_stage:
    Returns:
        vpp_list: list[int],
    """
    vpp_list = []
    max_speed = max(device_speed) if device_speed else 1.0

    for i, num_layers in enumerate(pp_layers_split):
        base_vpp = max(1, round(num_layers / target_layers_per_vstage))

        if device_speed:
            scale = device_speed[i] / max_speed
            base_vpp = max(1, round(base_vpp * scale))

        base_vpp = min(base_vpp, num_layers // min_layers_per_virtual_stage)
        if base_vpp == 0:
            base_vpp = 1

        while num_layers % base_vpp != 0 and base_vpp > 1:
            base_vpp -= 1

        vpp_list.append(base_vpp)

    return vpp_list


def estimate_comm_time_between_stages(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype_bytes: int = 2,  # bf16
    bandwidth_GBps: float = 300.0,
    latency_ms: float = 0.01,
    tensor_model_parallel_size: int = 1,
    virtual_pipeline_size: int = 1,
    activation_fraction: float = 1.0,
    use_allgather_for_activation: bool = False,
):
    bytes_one_way = batch_size * seq_len * hidden_size * dtype_bytes * activation_fraction
    if tensor_model_parallel_size > 1:
        bytes_one_way /= tensor_model_parallel_size

    K = max(1, virtual_pipeline_size)
    per_transfer_bytes = bytes_one_way / K
    bw_Bps = bandwidth_GBps * 1e9
    one_way_time = per_transfer_bytes / bw_Bps + latency_ms / 1000.0
    comm_time = 2 * K * one_way_time  # fwd+bwd

    if use_allgather_for_activation and tensor_model_parallel_size > 1:
        extra_bytes = (
            (tensor_model_parallel_size - 1) / tensor_model_parallel_size
        ) * bytes_one_way
        comm_time += extra_bytes / bw_Bps + (latency_ms / 1000.0)
    return comm_time


# call simulator to obtain the execution of each stage
def simulate_pipeline_parallelism_per_stage_time(
    process_mesh: list = None,
    pp_layers_split: list = None,
    scheme: str = '1F1B',
    fwd_time_per_stage_chunk: list = None,
    bwd_time_per_stage_chunk: list = None,
    comm_time_between_stages: list = None,
):
    print(f"--- Simulation Begin ---")
    print(f"Process Mesh: {process_mesh}")
    print(f"PP Layer Split: {pp_layers_split}")
    if scheme == '1F1B':
        for stage, num_layers in enumerate(pp_layers_split):
            # TODO: confirm simulated_rank for different stage
            print(f"Stage: {stage}; Num Layers: {num_layers}")
            simulated_rank = stage
            try:
                fwd_time, bwd_time, comm_time = simulator(
                    process_mesh, stage, num_layers, simulated_rank, pp_layers_split
                )
                fwd_time_per_stage_chunk.append(fwd_time)
                bwd_time_per_stage_chunk.append(bwd_time)
                comm_time_between_stages.append(comm_time)
            except Exception as e:
                print(f"[Error] Simulator failed at stage {stage}, skip. Reason: {e}")
                continue

    elif scheme == 'vpp':
        vpp_list = compute_vpp_from_layers(pp_layers_split)
        print(vpp_list)
        for stage_idx, (num_layers, vpp) in enumerate(zip(pp_layers_split, vpp_list)):
            layers_per_chunk = num_layers // vpp
            for vstage_idx in range(vpp):
                vstage_name = f"{stage_idx}-{vstage_idx}"
                print(f"  ->Stage {vstage_name} : ( {layers_per_chunk})")
                try:
                    fwd_time, bwd_time, comm_time = simulator(
                        process_mesh=process_mesh,
                        stage=vstage_name,
                        num_layers=layers_per_chunk,
                        simulated_rank=stage_idx,
                        pp_layers_split=pp_layers_split,
                    )
                    fwd_time_per_stage_chunk.append(fwd_time)
                    bwd_time_per_stage_chunk.append(bwd_time)
                    comm_time_between_stages.append(comm_time)
                except Exception as e:
                    print(f"[Error] Simulator failed at V-stage {vstage_name}, skip. Reason: {e}")
                    continue

    print(f"--- Simulation End ---")


def analyze_pp_time(
    scheme: str = '1F1B',
    num_micro_batches: int = 1,
    process_mesh: list = None,
    pp_layers_split: list = None,
):
    fwd_time_per_stage_chunk = []
    bwd_time_per_stage_chunk = []
    comm_time_between_stages = []
    vpp_partition = compute_vpp_from_layers(pp_layers_split)

    simulate_pipeline_parallelism_per_stage_time(
        process_mesh=process_mesh,
        pp_layers_split=pp_layers_split,
        scheme=scheme,
        fwd_time_per_stage_chunk=fwd_time_per_stage_chunk,
        bwd_time_per_stage_chunk=bwd_time_per_stage_chunk,
        comm_time_between_stages=comm_time_between_stages,
    )

    pipeline_cost = compute_pipeline_parallelism_cost(
        scheme=scheme,
        num_micro_batches=num_micro_batches,
        process_mesh=process_mesh,
        pp_layers_split=pp_layers_split,
        fwd_time_per_stage_chunk=fwd_time_per_stage_chunk,
        bwd_time_per_stage_chunk=bwd_time_per_stage_chunk,
        comm_time_between_stages=comm_time_between_stages,
        vpp_partition=vpp_partition,
    )

    return pipeline_cost
