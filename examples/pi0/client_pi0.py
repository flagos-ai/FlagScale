import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch


def encode_image(path: str) -> str:
    """Read image as base64 string."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path.resolve()}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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


def load_state_from_file(state_path: str) -> np.ndarray:
    """Load state tensor from file and convert to numpy array.

    Args:
        state_path: Path to state file (.pt file)

    Returns:
        State array with shape (1, state_dim)
    """
    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, torch.Tensor):
        state = state.numpy()
    # Ensure shape is (1, state_dim)
    if state.ndim == 1:
        state = state[np.newaxis, :]
    return state


def build_payload(args) -> dict[str, Any]:
    """Construct JSON payload for /infer.

    The client must send images with keys matching the config's images_keys.
    Default keys are:
    - observation.images.base_0_rgb
    - observation.images.left_wrist_0_rgb
    - observation.images.right_wrist_0_rgb
    """
    # Encode images with keys matching config images_keys
    img_sample = {
        "observation.images.base_0_rgb": encode_image(args.img1),
        "observation.images.left_wrist_0_rgb": encode_image(args.img2),
        "observation.images.right_wrist_0_rgb": encode_image(args.img3),
    }
    # Load state from file
    state = load_state_from_file(args.state_path)
    state = state.tolist()

    return {"instruction": args.instruction, "state": state, "images": [img_sample]}


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
        "--port", type=int, default=5000, help="Port of local SSH tunnel (default: 15000)"
    )
    parser.add_argument("--img1", required=True, help="Path to first camera RGB image")
    parser.add_argument("--img2", required=True, help="Path to second camera RGB image")
    parser.add_argument("--img3", required=True, help="Path to third camera RGB image")
    parser.add_argument(
        "--state-path",
        required=True,
        help="Path to state tensor file (.pt file) with shape (1, state_dim)",
    )
    parser.add_argument(
        "--instruction",
        default="Grab the orange and put it into the basket.",
        help="Task instruction for the robot",
    )
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
