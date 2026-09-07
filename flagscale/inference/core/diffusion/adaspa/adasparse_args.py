"""Runtime arguments for AdaSpa sparse attention."""

MODEL_ID = None
PROMPT = None
HEIGHT = None
WIDTH = None
FRAMES = None
NUM_STEPS = None
SEED = None
FPS = None

ENABLE_LOG = None
NUM_LAYERS = None
SPARSITY = None
SEARCH_STEPS = None
MIN_RECALL = None
BLOCK_SIZE = None
SPARSITY_MODES = None


def init_sparse_attn_paras(config):
    basic_config = config.get("basic", {})
    set_model_id(basic_config.get("model_id", "Wan2.1"))
    set_prompt(basic_config.get("prompt", "A cat walks on the grass"))
    set_height(basic_config.get("height", 768))
    set_width(basic_config.get("width", 1344))
    set_frames(basic_config.get("frames", 81))
    set_num_steps(basic_config.get("num_steps", 50))
    set_seed(basic_config.get("seed", 0))
    set_fps(basic_config.get("fps", 16))

    adaspa_config = config.get("adaspa", {})
    set_enable_log(adaspa_config.get("enable_log", True))
    set_num_layers(adaspa_config.get("num_layers", 30))
    set_sparsity(adaspa_config.get("sparsity", 0.8))
    set_search_steps(adaspa_config.get("search_steps", [10, 30]))
    set_min_recall(adaspa_config.get("min_recall", 0.9))
    set_block_size(adaspa_config.get("block_size", 64))
    set_sparsity_modes(adaspa_config.get("sparsity_modes", []))


def set_model_id(model_id):
    global MODEL_ID
    MODEL_ID = model_id


def set_prompt(prompt):
    global PROMPT
    PROMPT = prompt


def set_height(height):
    global HEIGHT
    HEIGHT = height


def set_width(width):
    global WIDTH
    WIDTH = width


def set_frames(frames):
    global FRAMES
    FRAMES = frames


def set_num_steps(steps):
    global NUM_STEPS
    NUM_STEPS = steps


def set_seed(seed):
    global SEED
    SEED = seed


def set_fps(fps):
    global FPS
    FPS = fps


def set_enable_log(enable_log):
    global ENABLE_LOG
    ENABLE_LOG = enable_log


def set_num_layers(num_layers):
    global NUM_LAYERS
    NUM_LAYERS = num_layers


def set_sparsity(sparsity):
    global SPARSITY
    SPARSITY = sparsity


def set_search_steps(steps):
    global SEARCH_STEPS
    SEARCH_STEPS = steps


def set_min_recall(min_recall):
    global MIN_RECALL
    MIN_RECALL = min_recall


def set_block_size(block_size):
    global BLOCK_SIZE
    BLOCK_SIZE = block_size


def set_sparsity_modes(sparsity_modes):
    global SPARSITY_MODES
    SPARSITY_MODES = sparsity_modes


def get_model_id():
    return MODEL_ID


def get_prompt():
    return PROMPT


def get_height():
    return HEIGHT


def get_width():
    return WIDTH


def get_frames():
    return FRAMES


def get_enable_log():
    return ENABLE_LOG


def get_num_steps():
    return NUM_STEPS


def get_seed():
    return SEED


def get_fps():
    return FPS


def get_num_layers():
    return NUM_LAYERS


def get_sparsity():
    return SPARSITY


def get_search_steps():
    return SEARCH_STEPS


def get_min_recall():
    return MIN_RECALL


def get_block_size():
    return BLOCK_SIZE


def get_sparsity_modes():
    return SPARSITY_MODES
