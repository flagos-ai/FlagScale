# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import argparse
import os

import torch


def combine(vision_input, llm_input, output, mlp_input=""):
    vision_dirs = os.listdir(vision_input)
    llm_dirs = os.listdir(llm_input)
    # Pipeline not supported yet.
    assert len(vision_dirs) == len(llm_dirs)
    for vision_dir in vision_dirs:
        assert vision_dir in llm_dirs
        llm_dir = vision_dir
        mlp_dir = vision_dir
        vision_params = torch.load(
            os.path.join(vision_input, vision_dir, "model_optim_rng.pt"), weights_only=False
        )
        llm_params = torch.load(
            os.path.join(llm_input, llm_dir, "model_optim_rng.pt"), weights_only=False
        )
        combined_state_dict = {}
        combined_state_dict["model"] = {}

        if mlp_input != "":
            mlp_params = torch.load(
                os.path.join(mlp_input, mlp_dir, "model_optim_rng.pt"), weights_only=False
            )
            for name, param in mlp_params["model"].items():
                new_name = f"vision_projection.{name}"
                combined_state_dict["model"][new_name] = param

        for name, param in vision_params["model"].items():
            new_name = f"vision_model.{name}"
            combined_state_dict["model"][new_name] = param

        for name, param in llm_params["model"].items():
            new_name = f"language_model.{name}"
            combined_state_dict["model"][new_name] = param

        output_dir = os.path.join(output, "iter_0000001", vision_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "model_optim_rng.pt")
        torch.save(combined_state_dict, output_file)
        print("saved:", output_file)
    latest_checkpointed_iteration = os.path.join(output, "latest_checkpointed_iteration.txt")

    with open(latest_checkpointed_iteration, "w") as f:
        f.write("1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Combine llm weights and vision weights to megatron format.


Example usage:
python combine_llm_vision.py --vision-input /some/vision_folder --llm-input /some/llm_folder --output /some/output/folder
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--vision-input", type=str, required=True, help="vision folder")
    parser.add_argument("--llm-input", type=str, required=True, help="llm folder")
    parser.add_argument("--mlp-input", type=str, default="", help="mlp folder")
    parser.add_argument(
        "--output", type=str, required=True, help="output directory for megatron state dict file(s)"
    )

    args = parser.parse_args()

    combine(args.vision_input, args.llm_input, args.output, args.mlp_input)

    print("done.")
