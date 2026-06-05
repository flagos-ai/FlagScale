# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/deployment/model_server/tools/websocket_policy_server.py

import asyncio
import http
import time
import traceback
from typing import Protocol, runtime_checkable

import websockets.asyncio.server as _server
import websockets.frames
from websockets.http11 import Request, Response

from . import msgpack_numpy
from flagscale.logger import logger


@runtime_checkable
class ServablePolicy(Protocol):
    def inference(self, obs: dict) -> dict: ...


class WebsocketPolicyServer:
    """Serves a policy over websocket for evaluation inference.

    Protocol:
      1. On connect, server sends metadata dict to client.
      2. Client sends msgpack-encoded obs dict, server returns msgpack-encoded action dict.
      3. Each response includes a "server_timing" key with latency info.
    """

    def __init__(
        self,
        policy: ServablePolicy,
        host: str = "0.0.0.0",
        port: int = 10093,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection) -> None:
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs: dict = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action: dict = self._policy.inference(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: Request) -> Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
