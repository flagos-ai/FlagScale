import json

from megatron.energon import stateless
from megatron.energon.task_encoder.cooking import Cooker

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv")


@stateless
def cook_bagel_images_sample(sample: dict) -> dict:
    """Cook a CrudeSample into the dict format expected by encode_sample.

    CrudeSample from tar with files like:
      000000000.000.jpg  -> sample['000.jpg'] (PIL Image or bytes)
      000000000.json     -> sample['json'] (decoded dict)
    """
    json_data = sample.get("json", {})
    if isinstance(json_data, (bytes, str)):
        json_data = json.loads(json_data)

    # Collect all image keys (e.g., '000.jpg', '000.png', etc.)
    images = []
    for key, value in sample.items():
        if key.startswith("__"):
            continue
        if any(key.endswith(ext) for ext in IMAGE_EXTENSIONS):
            images.append(value)

    result = {
        **{k: v for k, v in sample.items() if k.startswith("__")},  # preserve meta keys
        "json_data": json_data,
        "images": images,
    }
    return result


@stateless
def cook_bagel_video_sample(sample: dict) -> dict:
    """Cook a CrudeSample containing video into the dict format expected by encode_sample.

    CrudeSample from tar with files like:
      000000000.000.mp4  -> sample['000.mp4'] (AVDecoder object, Energon default decode)
      000000000.json     -> sample['json'] (decoded dict)

    Output:
      {
        '__key__': ...,
        '__subflavors__': ...,
        'json_data': dict,
        'images': [],           # no standalone images
        'video_bytes': bytes,   # raw video bytes for FrameSampler
      }
    """
    json_data = sample.get("json", {})
    if isinstance(json_data, (bytes, str)):
        json_data = json.loads(json_data)

    # Collect video: Energon's AVWebdatasetDecoder returns an AVDecoder object
    # which holds a BytesIO stream of the raw video bytes.
    video_bytes = None
    images = []
    for key, value in sample.items():
        if key.startswith("__"):
            continue
        if any(key.endswith(ext) for ext in VIDEO_EXTENSIONS):
            # AVDecoder object → extract raw bytes for decord
            if hasattr(value, "stream"):
                value.stream.seek(0)
                video_bytes = value.stream.read()
                value.stream.seek(0)
            elif isinstance(value, bytes):
                # If decoder is disabled, value is raw bytes directly
                video_bytes = value
            break  # only one video per sample expected
        elif any(key.endswith(ext) for ext in IMAGE_EXTENSIONS):
            # There might also be a thumbnail or cover image alongside the video
            images.append(value)

    result = {
        **{k: v for k, v in sample.items() if k.startswith("__")},  # preserve meta keys
        "json_data": json_data,
        "images": images,
        "video_bytes": video_bytes,
    }
    return result


# Cooker instance for registration in TaskEncoder.cookers list
video_cooker = Cooker(cook_bagel_video_sample, has_subflavors={"task": "vlm_video"})
image_cooker = Cooker(cook_bagel_images_sample)
