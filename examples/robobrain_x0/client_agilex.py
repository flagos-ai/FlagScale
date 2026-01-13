# Adopted from FlagOpen/RoboBrain-X0 (https://github.com/FlagOpen/RoboBrain-X0/blob/main/agilex/client_agilex.py)
import argparse
import base64
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

IMG_WIDTH = 640
IMG_HEIGHT = 480


def encode_image(path: str) -> str:
    """Read image as base64 string."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def check_health(base_url: str) -> None:
    """Ping /health; raise RuntimeError if unhealthy."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Health-check request failed: {e}") from e

    data = r.json()
    if not (data.get("status") == "healthy" and data.get("model_loaded")):
        raise RuntimeError(f"Server not ready: {json.dumps(data, indent=2)}")
    print(f"[√] Server healthy - GPU: {data['gpu_info']['device_name']}")


def build_payload(args) -> dict[str, Any]:
    """Construct JSON payload for /infer."""
    # 1. Dummy robot state (batch=1, dim=args.state_dim)
    state = np.random.uniform(-1, 1, size=(1, args.state_dim)).tolist()
    # 2. Encode images
    img_sample = {
        "cam_high": encode_image(args.base_img),
        "cam_left_wrist": encode_image(args.left_wrist_img),
        "cam_right_wrist": encode_image(args.right_wrist_img),
    }
    # 3. Image masks (True: image is valid)
    image_masks = {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": True}
    return {
        "instruction": "Grab the orange and put it into the basket.",
        "qpos": [[random.random() for _ in range(args.state_dim)]],
        "eef_pose": [[random.random() for _ in range(args.state_dim)]],
        "state": state,
        "high_level_instruction": args.high_level_instruction,
        "fine_grained_instruction": args.fine_grained_instruction,
        "images": img_sample,
        "image_masks": [image_masks],
        "num_steps": args.num_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }


def pretty_print_resp(resp: requests.Response) -> None:
    """Nicely print JSON or raw content."""
    try:
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(resp.text)


def main():
    parser = argparse.ArgumentParser(description="Client for RoboBrain-Robotics inference API")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host of local SSH tunnel (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=5001, help="Port of local SSH tunnel (default: 5001)"
    )
    parser.add_argument("--base-img", required=True, help="Path to base camera RGB image")
    parser.add_argument(
        "--left-wrist-img", required=True, help="Path to left wrist camera RGB image"
    )
    parser.add_argument(
        "--right-wrist-img", required=True, help="Path to right wrist camera RGB image"
    )
    parser.add_argument(
        "--state-dim", type=int, default=14, help="Dim of robot low-dim state vector (default: 14)"
    )
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--high-level-instruction", default="Grab the orange and put it into the basket."
    )
    parser.add_argument("--fine-grained-instruction", default=None)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"-> Using endpoint: {base_url}")

    payload = build_payload(args)
    try:
        t0 = time.time()
        resp = requests.post(
            f"{base_url}/infer",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=300,
        )
        elapsed = (time.time() - t0) * 1000
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[Error] HTTP request failed: {e}")
        sys.exit(1)
    print(f"[√] Response OK ({resp.status_code})  -  {elapsed:.1f}ms")
    pretty_print_resp(resp)


if __name__ == "__main__":
    main()
