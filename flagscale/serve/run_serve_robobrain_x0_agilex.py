"""
Adopted from FlagOpen/RoboBrain-X0 (https://github.com/FlagOpen/RoboBrain-X0/blob/main/agilex/server_agilex.py)

RoboBrain Robotics API Service - Agilex

This service provides an HTTP interface to receive robot state and images, and uses a pre-trained vision-language model for inference,
returning predicted dual-arm action sequences.

Supports two operation modes:
1. Standard Mode (SUBTASK_MODE = False): The model directly outputs control actions.
2. Subtask Mode (SUBTASK_MODE = True): The model first generates a text description of a extremetask, then outputs corresponding control actions.

Switch modes and models by modifying the `SUBTASK_MODE` and `MODEL_PATH` global variables below.

POST /infer Input Example:
{
  "eef_pose": [[...], [...]],
  "instruction": "Please classify the items",
  "images": {
      "cam_high": "<base64 string>",
      "cam_left_wrist": "<base64 string>",
      "cam_right_wrist": "<base64 string>"
  }
}
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

root = Path(__file__).parent.parent
sys.path.append(str(root))

from flagscale import serve
from flagscale.runner.utils import logger
from flagscale.serve.data_process.action_chunk_to_fast_token import ActionChunkProcessor
from flagscale.serve.managers.pynvml_manager import PYNVML_MANAGER

serve.load_args()

TASK_CONFIG = serve.task_config
ENGINE_CONFIG = TASK_CONFIG.serve[0].engine_args
SUBTASK_MODE = ENGINE_CONFIG["subtask_mode"]
if SUBTASK_MODE:
    MODEL_PATH = ENGINE_CONFIG["model_sub_task"]
else:
    MODEL_PATH = ENGINE_CONFIG["model"]
SERVICE_CONFIG = {
    "host": ENGINE_CONFIG["host"],
    "port": ENGINE_CONFIG["port"],
    "debug": ENGINE_CONFIG["debug"],
    "threaded": ENGINE_CONFIG["threaded"],
    "max_content_length": 16 * 1024 * 1024,
}
_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}

app = Flask(__name__)
CORS(app)
model = None
processor = None
action_tokenizer = None


def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Cache and return an ActionChunkProcessor instance for each process"""
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(
            max_len=max_len, fast_tokenizer_path=ENGINE_CONFIG["tokenizer_path"]
        )
        _TOKENIZER_CACHE[max_len] = tok
    return tok


def load_model():
    """Load and initialize the model and processor"""
    global model, processor, action_tokenizer
    try:
        logger.info(f"Loading model: {MODEL_PATH} (Subtask Mode: {SUBTASK_MODE})")
        device_id = os.environ.get("EGL_DEVICE_ID", "0")
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, padding_side="left")
        model.eval()

        action_tokenizer = get_tokenizer(max_len=256)

        if torch.cuda.is_available():
            logger.info(f"Model successfully loaded to GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("Model successfully loaded to CPU")
        return True
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return False


def inverse_transform(x_norm, scale, offset):
    """Denormalize actions based on mean and standard deviation"""
    x_norm = np.asarray(x_norm)
    return (x_norm - offset) / scale


def decode_image_base64_to_pil(image_base64: str) -> Image:
    try:
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise ValueError("Invalid Base64 image string")


def process_images(images_dict: dict) -> list:
    try:
        image_keys = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
        processed_list = [
            decode_image_base64_to_pil(images_dict[k]).resize((320, 240)) for k in image_keys
        ]
        return processed_list
    except KeyError as e:
        raise ValueError(f"Missing required image: {e}")
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise ValueError("Image processing failed")


# --- Flask API Endpoints ---


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint, returns service and model status"""
    if model is None or processor is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": True,
            "subtask_mode": SUBTASK_MODE,
            "model_path": MODEL_PATH,
            "gpu_info": PYNVML_MANAGER.get_gpu_info(),
        }
    )


@app.route("/info", methods=["GET"])
def service_info():
    """Provide service metadata"""
    return jsonify(
        {
            "service_name": "RoboBrain Robotics API for Agilex",
            "version": "2.0.0",
            "subtask_mode": SUBTASK_MODE,
            "model_path": MODEL_PATH,
            "endpoints": {"/health": "GET", "/info": "GET", "/infer": "POST"},
        }
    )


@app.route("/infer", methods=["POST"])
def infer_api():
    """Core inference API endpoint"""
    start_time = time.time()

    if model is None:
        return (
            jsonify({"success": False, "error": "Model not loaded, please check service status"}),
            503,
        )

    data = request.get_json()
    if not data or "eef_pose" not in data or "instruction" not in data or "images" not in data:
        return (
            jsonify(
                {"success": False, "error": "Request data is incomplete or in incorrect format"}
            ),
            400,
        )

    instruction = data["instruction"]
    images = data["images"]
    images_pil = process_images(images)

    if SUBTASK_MODE:
        prompt_template = (
            "You are controlling an Agilex dual-arm robot. Your task is to adjust the end effector poses (EEPose) at 30Hz to complete a specified task. "
            "Your output must include two components: 1. Immediate sub-task: The specific action you will execute first to progress toward the overall task; 2. Control tokens: These will be decoded into a 30×14 action sequence to implement the sub-task. "
            "The action sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEPose and the last 7 dimensions control the left arm EEPose. "
            "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n"
            "Your current visual inputs are: robot front image"
        )
        content = [
            {"type": "text", "text": prompt_template},
            {"type": "image", "image": f"data:image;base64,{images['cam_high']}"},
            {"type": "text", "text": ", right wrist image"},
            {"type": "image", "image": f"data:image;base64,{images['cam_right_wrist']}"},
            {"type": "text", "text": " and left wrist image"},
            {"type": "image", "image": f"data:image;base64,{images['cam_left_wrist']}"},
            {"type": "text", "text": f"\nYour overall task is: {instruction.lower()}."},
        ]
    else:
        prompt_template = (
            "You are controlling an Agilex dual-arm robot. Your task is to adjust the end effector poses (EEPose) at 30Hz to complete a specified task. "
            "You need to output control tokens that can be decoded into a 30×14 action sequence. The sequence has 30 consecutive actions, each with 14 dimensions. "
            "The first 7 dimensions control the right arm EEPose and the last 7 dimensions control the left arm EEPose. "
            "Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range)\n\n"
            "Your current visual inputs are: robot front image"
        )
        content = [
            {"type": "text", "text": prompt_template},
            {"type": "image", "image": f"data:image;base64,{images['cam_high']}"},
            {"type": "text", "text": ", right wrist image"},
            {"type": "image", "image": f"data:image;base64,{images['cam_right_wrist']}"},
            {"type": "text", "text": " and left wrist image"},
            {"type": "image", "image": f"data:image;base64,{images['cam_left_wrist']}"},
            {
                "type": "text",
                "text": f"\nYour overall task is: {instruction.lower()}. Currently, focus on completing the subtask: {instruction.lower()}",
            },
        ]

    messages = [{"role": "user", "content": content}]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text_prompt], images=images_pil, padding=True, return_tensors="pt").to(
        model.device
    )

    gen_kwargs = {
        "max_new_tokens": 768,
        "do_sample": False,
        "temperature": 0.0,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "repetition_penalty": 1.0,
        "use_cache": True,
    }
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)[0]

    input_length = inputs.input_ids.shape[1]
    output_tokens = output_ids[input_length:].detach().cpu().tolist()

    subtask_result = "N/A"
    if SUBTASK_MODE:
        try:
            boa_token = 151665
            split_index = output_tokens.index(boa_token)
            subtask_tokens = output_tokens[:split_index]
            action_tokens_raw = output_tokens[split_index + 1 :]
            subtask_result = processor.tokenizer.decode(
                subtask_tokens, skip_special_tokens=True
            ).strip()
            logger.info(f"Parsed subtask: {subtask_result}")
        except ValueError:
            logger.warning(
                "<boa> token not found, unable to parse subtask. Treating entire output as action."
            )
            action_tokens_raw = output_tokens
            subtask_result = "Parsing failed: <boa> token not found"
    else:
        action_tokens_raw = output_tokens

    try:
        eoa_token = 151667
        end_index = action_tokens_raw.index(eoa_token)
        action_tokens_raw = action_tokens_raw[:end_index]
    except ValueError:
        logger.warning("<eoa> token not found, using complete output sequence.")

    action_ids = [t - 149595 for t in action_tokens_raw if 149595 <= t < 151643]
    actions_norm, _ = action_tokenizer.extract_actions_from_tokens(
        [action_ids], action_horizon=30, action_dim=14
    )
    delta_actions = actions_norm[0]

    processing_time = time.time() - start_time
    response = {
        "success": True,
        "delta_actions": delta_actions.tolist(),
        "processing_time": processing_time,
    }
    if SUBTASK_MODE:
        response["subtask"] = subtask_result

    return jsonify(response)


# --- Main Program Entry ---
if __name__ == "__main__":
    if not load_model():
        sys.exit(1)

    logger.info("RoboBrain Agilex API service starting...")
    logger.info(f"Service address: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"Current mode: {'Subtask' if SUBTASK_MODE else 'Standard'}")

    app.run(
        host=SERVICE_CONFIG["host"],
        port=SERVICE_CONFIG["port"],
        debug=SERVICE_CONFIG["debug"],
        threaded=SERVICE_CONFIG["threaded"],
    )
