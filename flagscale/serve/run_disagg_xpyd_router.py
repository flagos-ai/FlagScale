# Copyright (c) 2025, BAAI. All rights reserved.
#
# Adopted from https://github.com/vllm-project/vllm/blob/1ad957950ffc1552af5abda78c03d88ddb67945b/examples/online_serving/disagg_xpyd/disagg_prefill_proxy_xpyd.py. Below is the original copyright:
#
# SPDX-License-Identifier: Apache-2.0
#


import os
import random
import socket
import threading
import uuid
from functools import lru_cache
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request
from transformers import AutoTokenizer

from flagscale import serve
from flagscale.logger import logger

serve.load_args()
TASK_CONFIG = serve.task_config
MODEL_PATH = TASK_CONFIG.serve[0].get("engine_args", {}).get("model", None)

# Scheduling strategy: 'random', 'robin', 'slo'
SCHEDULING_STRATEGY = (
    TASK_CONFIG.experiment.get("runner", {}).get("deploy", {}).get("prefill_decode_strategy", "slo")
)


@lru_cache(maxsize=32)
def load_hf_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def count_chat_tokens(messages: list[dict[str, Any]]) -> int:
    tokenizer = load_hf_tokenizer(MODEL_PATH)
    normalized_message = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            content = "".join(part["text"] for part in content if part.get("type") == "text")
        normalized_message.append({"role": msg["role"], "content": content})
    text = tokenizer.apply_chat_template(
        normalized_message, add_generation_prompt=False, tokenize=False
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_text_tokens(prompt: str) -> int:
    tokenizer = load_hf_tokenizer(MODEL_PATH)
    return len(tokenizer.encode(prompt, add_special_tokens=False))


# -----------------------------------------------------------------------------
# LoadManager: unified management of P/D instances and their load
# -----------------------------------------------------------------------------
class LoadManager:
    def __init__(self):
        self._lock = threading.Lock()
        # Each resource type 'P' or 'D' maps to {http_addr: {'zmq': zmq_addr, 'load_num': int, 'load_len': int, 'compute_ratio': float}}
        # load_num: num of req, load_len: num of tokens
        self._instances: dict[str, dict[str, dict[str, object]]] = {"P": {}, "D": {}}

    def register(self, rtype: str, http_addr: str, zmq_addr: str):
        with self._lock:
            if http_addr not in self._instances[rtype]:
                self._instances[rtype][http_addr] = {
                    "zmq": zmq_addr,
                    "load_num": 0,
                    "load_len": 0,
                    "compute_ratio": 1.0,
                }
                logger.info(f"Registered new {rtype}-instance {http_addr} (zmq={zmq_addr})")
            else:
                # If zmq address changed, synchronize it
                self._instances[rtype][http_addr]["zmq"] = zmq_addr

    def increment_load(self, rtype: str, http_addr: str, tokens=0):
        with self._lock:
            self._instances[rtype][http_addr]["load_num"] += 1
            self._instances[rtype][http_addr]["load_len"] += tokens
            logger.debug(
                f"[{rtype}] +1 load on {http_addr}, now={self._instances[rtype][http_addr]['load_num']}"
            )

    def decrement_load(self, rtype: str, http_addr: str, tokens=0):
        with self._lock:
            self._instances[rtype][http_addr]["load_num"] -= 1
            self._instances[rtype][http_addr]["load_len"] -= tokens
            logger.debug(
                f"[{rtype}] -1 load on {http_addr}, now={self._instances[rtype][http_addr]['load_num']}"
            )

    def get_random(self, rtype: str) -> tuple[str, str]:
        with self._lock:
            items = list(self._instances[rtype].items())
            logger.info(f"========== whole instance status {self._instances}==========")
        http_addr, info = random.choice(items)
        return http_addr, info["zmq"]

    def get_robin_loaded(self, rtype: str) -> tuple[str, str]:
        with self._lock:
            http_addr, info = min(self._instances[rtype].items(), key=lambda kv: kv[1]["load_num"])
            logger.info(f"========== whole instance status {self._instances}==========")
        return http_addr, info["zmq"]

    def get_slo_loaded(self, rtype: str, token_num: int = -1) -> tuple[str, str]:
        with self._lock:
            http_addr, info = min(
                self._instances[rtype].items(),
                key=lambda kv: (kv[1]["load_len"] + token_num) / kv[1]["compute_ratio"],
            )
            logger.info(f"========== whole instance status {self._instances}==========")
        return http_addr, info["zmq"]

    def get_loaded(
        self, rtype: str, load_type: str = "robin", token_num: int = 0
    ) -> tuple[str, str]:
        if load_type == "random":
            return self.get_random(rtype)
        elif load_type == "robin":
            return self.get_robin_loaded(rtype)
        elif load_type == "slo":
            return self.get_slo_loaded(rtype, token_num)
        else:
            raise ValueError(f"Unknown load type: {load_type}")


# -----------------------------------------------------------------------------
# Globals & configuration
# -----------------------------------------------------------------------------
lm = LoadManager()

# Legacy registration dicts & Conditions retained for external waiting
prefill_instances: dict[str, str] = {}
decode_instances: dict[str, str] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()


# -----------------------------------------------------------------------------
# Service discovery: receive instance registrations
# -----------------------------------------------------------------------------
def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_addr, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            typ = data.get("type")
            http_addr = data.get("http_address")
            zmq_addr = data.get("zmq_address")
            if typ == "P":
                with prefill_cv:
                    prefill_instances[http_addr] = zmq_addr
                lm.register("P", http_addr, zmq_addr)
            elif typ == "D":
                with decode_cv:
                    decode_instances[http_addr] = zmq_addr
                lm.register("D", http_addr, zmq_addr)
            else:
                logger.warning(f"Unexpected registration message: {data}")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    listener = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    listener.start()
    return listener


# -----------------------------------------------------------------------------
# HTTP proxy & request forwarding
# -----------------------------------------------------------------------------
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


def random_uuid() -> str:
    return uuid.uuid4().hex


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as resp:
            if resp.status == 200:
                async for chunk in resp.content.iter_chunked(1024):
                    yield chunk
            else:
                content = await resp.read()
                yield content


# support both /v1/completions and /v1/chat/completions
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_data = await request.get_json()
        endpoint = request.path  # this will be '/v1/completions' or '/v1/chat/completions'

        # calculate tokens num
        prompt_tokens_num = 0
        if SCHEDULING_STRATEGY == "slo":
            if request.path.endswith("/chat/completions"):
                prompt_tokens_num = count_chat_tokens(original_data["messages"])
            else:
                prompt_tokens_num = count_text_tokens(original_data["prompt"])
        logger.info(f"---------------- prompt_tokens_num {prompt_tokens_num} -------------- ")

        # Prefill request: max_tokens=1
        prefill_request = original_data.copy()
        prefill_request["max_tokens"] = 1

        # Select Prefill instance
        prefill_addr, prefill_zmq = lm.get_loaded("P", SCHEDULING_STRATEGY, prompt_tokens_num)
        logger.info(f"Selected P-instance {prefill_addr} via '{SCHEDULING_STRATEGY}'")

        # Select Decode instance
        decode_addr, decode_zmq = lm.get_loaded("D", SCHEDULING_STRATEGY, prompt_tokens_num)
        logger.info(f"Selected D-instance {decode_addr} via '{SCHEDULING_STRATEGY}'")

        # Keep original request_id composition format
        request_id = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{random_uuid()}"

        # Execute Prefill and update load
        lm.increment_load("P", prefill_addr, prompt_tokens_num)
        try:
            async for _ in forward_request(
                f"http://{prefill_addr}{endpoint}", prefill_request, request_id
            ):
                pass
        finally:
            lm.decrement_load("P", prefill_addr, prompt_tokens_num)

        # Execute Decode and update load
        async def tracked_decode():
            lm.increment_load("D", decode_addr, prompt_tokens_num)
            try:
                async for chunk in forward_request(
                    f"http://{decode_addr}{endpoint}", original_data, request_id
                ):
                    yield chunk
            finally:
                lm.decrement_load("D", decode_addr, prompt_tokens_num)

        resp = await make_response(tracked_decode())
        resp.timeout = None
        return resp

    except Exception as e:
        logger.error("Error in proxy server", exc_info=e)
        return {"error": str(e)}, 500


def main():
    deploy_config = TASK_CONFIG.experiment.get("runner", {}).get("deploy", {})
    serve_port = deploy_config.get("port", None)
    # Used to register with the pd service discovery
    pd_proxy_port = deploy_config.get("pd_proxy_port", None)
    if not serve_port:
        raise ValueError("No port specified in deploy config")
    if not pd_proxy_port:
        raise ValueError("No pd_proxy_port specified in deploy config")
    logger.info(
        f"Starting Proxy Server...with pd_proxy_port {pd_proxy_port} and serve_port {serve_port}"
    )
    listener = start_service_discovery("0.0.0.0", pd_proxy_port)
    app.run(host="0.0.0.0", port=serve_port)
    listener.join()


if __name__ == "__main__":
    main()
