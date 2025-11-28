#!/usr/bin/env python3

import base64
import io
import json
import time

from pathlib import Path
from typing import List

import numpy as np
import requests

from PIL import Image


class PI05Client:
    """Client for interacting with Pi0.5 inference server"""

    def __init__(self, host="127.0.0.1", port=5000):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to expected size (480x640)
            img = img.resize((640, 480))

            # Encode to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"

    def predict(
        self,
        image_paths: List[str],
        instruction: str,
        states: List[float] = None,
        action_horizon: int = 16,
    ) -> dict:
        """Send prediction request to Pi0.5 server"""

        if states is None:
            # Default to zeros for Pi0.5 (32-dim actions)
            states = [0.0] * 32

        # Prepare request data
        data = {"instruction": instruction, "states": states}

        # Add images
        image_keys = [
            "observation.images.camera0",
            "observation.images.camera1",
            "observation.images.camera2",
        ]
        for i, image_path in enumerate(image_paths):
            if i < len(image_keys):
                try:
                    data[image_keys[i]] = self.encode_image(image_path)
                except Exception as e:
                    print(f"Warning: Could not encode image {image_path}: {e}")
                    data[image_keys[i]] = None

        # Send request
        try:
            response = self.session.post(f"{self.base_url}/infer", json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Request failed: {e}"}

    def check_health(self) -> dict:
        """Check server health status"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            result = response.json()
            result["success"] = True  # Ensure success flag is set
            return result
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Health check failed: {e}"}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Invalid JSON response: {e}"}


def main():
    """Example usage of Pi0.5 client"""

    # Initialize client
    client = PI05Client(host="127.0.0.1", port=5000)

    # Check server health
    print("🔍 Checking server health...")
    health = client.check_health()
    if health.get("success", False):
        print("✅ Server is healthy")
        print(f"   Model: {health.get('model', 'Unknown')}")
        print(f"   Pi0.5 Mode: {health.get('pi05_mode', False)}")
        print(f"   Device: {health.get('device', 'Unknown')}")
        print(f"   Max Token Length: {health.get('max_token_len', 'Unknown')}")
        print(f"   Action Dimension: {health.get('action_dim', 'Unknown')}")
        print(f"   Action Horizon: {health.get('action_horizon', 'Unknown')}")
    else:
        print(f"❌ Server health check failed: {health.get('error', 'Unknown error')}")
        return

    # Example test data (you'll need actual image files)
    test_images = [
        "/path/to/camera0.jpg",  # Replace with actual image paths
        "/path/to/camera1.jpg",
        "/path/to/camera2.jpg",
    ]

    # Check if images exist
    missing_images = [img for img in test_images if not Path(img).exists()]
    if missing_images:
        print(f"⚠️  Warning: Some test images not found: {missing_images}")
        print("   Please provide actual image paths to test the client.")
        print("   Skipping prediction test...")
        return

    print("\n🚀 Testing prediction...")

    # Test with Pi0.5 specific instruction
    instruction = "Pick up the red block and place it on the blue box"

    # Initial state (32-dim for Pi0.5)
    initial_states = [0.0] * 32

    start_time = time.time()
    result = client.predict(
        image_paths=test_images,
        instruction=instruction,
        states=initial_states,
        action_horizon=16,  # Pi0.5 typically uses 16
    )

    end_time = time.time()

    if result.get("success", False):
        predictions = result.get("predictions", [])
        print("✅ Prediction successful!")
        print(f"   Inference time: {end_time - start_time:.2f}s")
        print(f"   Predictions shape: {len(predictions) if predictions else 0}")
        print(f"   Pi0.5 mode: {result.get('pi05_mode', False)}")
        print(f"   Action horizon: {result.get('action_horizon', 'Unknown')}")

        if predictions:
            print(
                f"   Sample prediction (first 5 values): {predictions[0][:5] if isinstance(predictions[0], list) else predictions[0]}"
            )
    else:
        print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    print("\n📊 Test completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi0.5 Inference Client")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--images", nargs="+", help="Image paths for testing")
    parser.add_argument(
        "--instruction", default="Pick up the red object", help="Instruction for the model"
    )
    args = parser.parse_args()

    # Override host/port if provided
    client = PI05Client(host=args.host, port=args.port)

    # Health check
    print("🔍 Checking server health...")
    health = client.check_health()
    if health.get("success", False):
        print("✅ Server is healthy")
        print(f"   Model: {health.get('model', 'Unknown')}")
        print(f"   Pi0.5 Mode: {health.get('pi05_mode', False)}")
    else:
        print(f"❌ Server health check failed: {health.get('error', 'Unknown error')}")
        exit(1)

    # Test prediction if images provided
    if args.images:
        print("\n🚀 Testing prediction...")
        result = client.predict(
            image_paths=args.images,
            instruction=args.instruction,
            states=[0.0] * 32,  # Pi0.5 uses 32-dim states
            action_horizon=16,
        )

        if result.get("success", False):
            print("✅ Prediction successful!")
            print(f"   Pi0.5 mode: {result.get('pi05_mode', False)}")
            predictions = result.get("predictions", [])
            if predictions:
                print(f"   Predictions: {predictions[:5]}...")  # Show first 5 values
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
    else:
        print("ℹ️  No images provided. Use --images to test prediction functionality.")
