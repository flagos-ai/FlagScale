import argparse
import hashlib
import hmac
import json
import os
import subprocess
import sys
import time

import requests

from flagscale.logger import logger

SUPPORTED_MODELS = ["pi0_5", "qwen_gr00t"]
DEFAULT_BASE_URL = "https://flageval.baai.ac.cn/api/hf"
MODEL_CONFIG_DIR_FMT = "examples/{model_name}/conf"


def parse_args():
    parser = argparse.ArgumentParser(description="Run online eval via FlagEval API")
    parser.add_argument("--model-name", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--model-id", default=None, help="Model ID for FlagEval UI (defaults to model-name)"
    )
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset keys to evaluate")
    parser.add_argument("--server-host", required=True, help="IP/hostname FlagEval calls back to")
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--attach", action="store_true", help="Server already running, skip startup"
    )
    parser.add_argument("--detach", action="store_true", help="Leave server running after eval")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--server-timeout", type=int, default=300)
    return parser.parse_args()


def start_server(model_name: str):
    config_dir = MODEL_CONFIG_DIR_FMT.format(model_name=model_name)
    proc = subprocess.Popen(
        [
            sys.executable,
            "flagscale/run.py",
            "--config-path",
            config_dir,
            "--config-name",
            "serve",
            "action=run",
        ],
    )
    return proc


def wait_for_server(host: str, port: int, timeout: int) -> None:
    url = f"http://{host}:{port}/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError(f"Server at {url} did not become ready within {timeout}s")


def stop_server(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def make_headers(url: str, body: str, secret: str, include_content_type: bool = False) -> dict:
    timestamp = str(int(time.time()))
    to_sign = f"{timestamp}{url}{body}"
    sign = hmac.new(secret.encode(), to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {
        "X-Flageval-Sign": sign,
        "X-Flageval-Timestamp": timestamp,
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def get_dataset_ids(base_url: str, secret: str, dataset_keys: list[str]) -> list[int]:
    url = f"{base_url}/robo-datasets"
    headers = make_headers(url, "", secret)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    logger.info(f"robo-datasets response: {json.dumps(data, ensure_ascii=False)[:500]}")
    name_to_id = {item["name"]: item["id"] for item in data}
    missing = [k for k in dataset_keys if k not in name_to_id]
    if missing:
        logger.info(f"Available dataset keys: {sorted(name_to_id)}")
        raise ValueError(f"Dataset keys not found: {missing}")
    return [name_to_id[k] for k in dataset_keys]


def submit_eval(
    base_url: str,
    secret: str,
    model_name: str,
    dataset_ids: list[int],
    ws_url: str,
    description: str,
) -> int:
    url = f"{base_url}/robo-batches"
    body = json.dumps(
        {
            "modelId": model_name,
            "datasets": dataset_ids,
            "url": ws_url,
            "description": description,
        }
    )
    headers = make_headers(url, body, secret, include_content_type=True)
    resp = requests.post(url, headers=headers, data=body)
    if resp.status_code == 401:
        sys.exit("Authentication failed — check FLAGEVAL_SECRET")
    if resp.status_code == 400:
        sys.exit(f"Submission error: {resp.json().get('detail', resp.text)}")
    resp.raise_for_status()
    return resp.json()["id"]


def poll_eval(base_url: str, secret: str, batch_id: int, poll_interval: int) -> None:
    url = f"{base_url}/robo-batches/{batch_id}"
    while True:
        headers = make_headers(url, "", secret)
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        logger.info(f"result: {data}")
        for detail in data.get("details", []):
            name = detail.get("dataset", "?")
            logger.info(f"  [{name}] status={detail.get('status', '?')}")
        if status in ("S", "F"):
            logger.info(f"Eval batch {batch_id} finished with status: {status}")
            logger.info(json.dumps(data, indent=2, ensure_ascii=False))
            return
        logger.info(f"Batch status: {status}, next poll in {poll_interval}s...")
        time.sleep(poll_interval)


def get_archive_url(
    base_url: str, secret: str, batch_id: int, dataset_ids: list[int], poll_interval: int
) -> None:
    for dataset_id in dataset_ids:
        url = f"{base_url}/robo-batches/{batch_id}/datasets/{dataset_id}/archive"
        while True:
            headers = make_headers(url, "", secret)
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("done"):
                logger.info(f"Download link (dataset {dataset_id}): {data['url']}")
                break
            logger.info(f"Archive not ready yet for dataset {dataset_id}, polling...")
            time.sleep(poll_interval)


def main():
    args = parse_args()

    secret = os.environ.get("FLAGEVAL_SECRET")
    if not secret:
        sys.exit("Error: FLAGEVAL_SECRET environment variable not set")

    ws_url = (
        f"ws://{args.server_host}:{args.server_port}"
        if args.server_port
        else f"ws://{args.server_host}"
    )

    proc = None
    try:
        if not args.attach:
            proc = start_server(args.model_name)
            logger.info(f"Server process started (pid={proc.pid}), waiting for ready...")
            wait_for_server(args.server_host, args.server_port or 80, args.server_timeout)
            logger.info(f"Server ready at {ws_url}")

        dataset_ids = get_dataset_ids(args.base_url, secret, args.datasets)
        logger.info(f"Dataset IDs: {dataset_ids}")

        batch_id = submit_eval(
            args.base_url,
            secret,
            args.model_id or args.model_name,
            dataset_ids,
            ws_url,
            args.description,
        )
        logger.info(f"Submitted eval batch {batch_id}, polling every {args.poll_interval}s...")

        poll_eval(args.base_url, secret, batch_id, args.poll_interval)
        get_archive_url(args.base_url, secret, batch_id, dataset_ids, args.poll_interval)

    finally:
        if proc and not args.detach:
            logger.info("Stopping server...")
            stop_server(proc)


if __name__ == "__main__":
    main()
