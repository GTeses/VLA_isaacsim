from __future__ import annotations

import argparse
import asyncio
import math
import traceback

import websockets
import websockets.server as websocket_server
import websockets.frames

from zhishu_dualarm_lab.utils import msgpack_numpy


def _make_action_chunk(step_index: int, chunk_length: int, action_dim: int) -> dict:
    import numpy as np

    actions = np.zeros((chunk_length, action_dim), dtype=np.float32)
    for t in range(chunk_length):
        phase = 0.15 * (step_index + t)
        actions[t, 0] = 0.7 * math.sin(phase)
        actions[t, 1] = 0.5 * math.sin(phase + 0.3)
        actions[t, 3] = 0.4 * math.sin(phase + 0.8)
        actions[t, 5] = 0.35 * math.sin(phase + 1.2)
        actions[t, 7] = -0.7 * math.sin(phase)
        actions[t, 8] = -0.5 * math.sin(phase + 0.3)
        actions[t, 10] = -0.4 * math.sin(phase + 0.8)
        actions[t, 12] = -0.35 * math.sin(phase + 1.2)
    return {
        "actions": actions,
        "server_timing": {"infer_ms": 0.1},
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a fake openpi-compatible websocket policy for local testing.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chunk_length", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=14)
    args = parser.parse_args()

    packer = msgpack_numpy.Packer()

    async def handler(websocket):
        metadata = {"fake_policy": True, "action_dim": args.action_dim, "chunk_length": args.chunk_length}
        await websocket.send(packer.pack(metadata))
        step_index = 0
        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                print(f"[FAKE SERVER] received obs keys: {sorted(obs.keys())}")
                action_chunk = _make_action_chunk(step_index, args.chunk_length, args.action_dim)
                await websocket.send(packer.pack(action_chunk))
                print(f"[FAKE SERVER] sent action chunk shape: {action_chunk['actions'].shape}")
                step_index += args.chunk_length
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal fake policy server error",
                )
                raise

    print(f"[FAKE SERVER] listening on ws://{args.host}:{args.port}")
    async with websocket_server.serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
