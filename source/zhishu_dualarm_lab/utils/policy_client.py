from __future__ import annotations

import abc
import socket
import time
from dataclasses import dataclass

import numpy as np
import websockets.sync.client
from PIL import Image

from zhishu_dualarm_lab.utils.action_adapter import PolicyActionChunkAdapter
from zhishu_dualarm_lab.utils import msgpack_numpy


class PolicyClientError(RuntimeError):
    """Base class for policy client failures."""


class PolicyConnectionError(PolicyClientError):
    """Raised when the websocket connection cannot be established or maintained."""


class PolicyTimeoutError(PolicyClientError):
    """Raised when policy inference times out."""


class PolicyResponseError(PolicyClientError):
    """Raised when the policy server returns an invalid or incomplete payload."""


@dataclass
class PolicyClientConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    timeout_s: float = 10.0
    action_dim: int = 14
    input_schema: str = "zhishu14"
    output_contract: str = "zhishu14"


class BasePolicyClient(abc.ABC):
    """Abstract client interface used by the demo loop."""

    def __init__(self, cfg: PolicyClientConfig):
        self.cfg = cfg
        self.action_adapter = PolicyActionChunkAdapter(action_dim=cfg.action_dim)

    @abc.abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, policy_input_dict: dict) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def _adapt_policy_input(self, policy_input_dict: dict) -> dict:
        if self.cfg.input_schema in {"native", "zhishu14"}:
            return policy_input_dict
        if self.cfg.input_schema == "libero":
            state = np.asarray(policy_input_dict["observation/state"], dtype=np.float32)
            # This is a smoke-test-only bridge for pi05_libero. Keep the long-term Zhishu
            # schema out of this path so the temporary LIBERO contract does not leak into
            # the normal runtime interface.
            state8 = state[:8].astype(np.float32, copy=True)
            return {
                "prompt": policy_input_dict["prompt"],
                "observation/image": _resize_uint8_hwc(policy_input_dict["observation/external_image"], 224),
                "observation/wrist_image": _resize_uint8_hwc(policy_input_dict["observation/left_wrist_image"], 224),
                "observation/state": state8,
            }
        raise PolicyResponseError(f"Unsupported input schema: {self.cfg.input_schema}")

    def _adapt_policy_output(self, response_dict: dict) -> np.ndarray:
        if "actions" in response_dict:
            action_payload = response_dict["actions"]
        elif "action" in response_dict:
            action_payload = response_dict["action"]
        else:
            raise PolicyResponseError(
                f"Policy response missing 'actions' or 'action' field. Keys: {sorted(response_dict.keys())}"
            )

        if self.cfg.output_contract in {"native14", "zhishu14"}:
            return self.action_adapter.normalize_action_chunk(action_payload)
        if self.cfg.output_contract == "libero7":
            return self.action_adapter.normalize_libero_action_chunk(action_payload)
        raise PolicyResponseError(f"Unsupported output contract: {self.cfg.output_contract}")


class FakePolicyClient(BasePolicyClient):
    """Local non-network client used to validate the action-buffer / replan loop."""

    def __init__(self, cfg: PolicyClientConfig, chunk_length: int = 8):
        super().__init__(cfg)
        self.chunk_length = chunk_length
        self._connected = False
        self._step_index = 0

    def connect(self) -> None:
        self._connected = True

    def infer(self, policy_input_dict: dict) -> np.ndarray:
        if not self._connected:
            raise PolicyConnectionError("FakePolicyClient.connect() must be called before infer().")
        del policy_input_dict
        chunk = np.zeros((self.chunk_length, self.cfg.action_dim), dtype=np.float32)
        for t in range(self.chunk_length):
            phase = 0.15 * (self._step_index + t)
            chunk[t, 0] = 0.7 * np.sin(phase)
            chunk[t, 1] = 0.5 * np.sin(phase + 0.3)
            chunk[t, 3] = 0.4 * np.sin(phase + 0.8)
            chunk[t, 5] = 0.35 * np.sin(phase + 1.2)
            chunk[t, 7] = -0.7 * np.sin(phase)
            chunk[t, 8] = -0.5 * np.sin(phase + 0.3)
            chunk[t, 10] = -0.4 * np.sin(phase + 0.8)
            chunk[t, 12] = -0.35 * np.sin(phase + 1.2)
        self._step_index += self.chunk_length
        return self.action_adapter.normalize_action_chunk(chunk)

    def close(self) -> None:
        self._connected = False


class OpenPiWebsocketClient(BasePolicyClient):
    """Websocket client compatible with the local openpi policy server protocol."""

    def __init__(self, cfg: PolicyClientConfig):
        super().__init__(cfg)
        self._uri = f"ws://{cfg.host}:{cfg.port}"
        self._packer = msgpack_numpy.Packer()
        self._ws = None
        self._server_metadata: dict | None = None

    @property
    def server_metadata(self) -> dict:
        return self._server_metadata or {}

    def connect(self) -> None:
        try:
            self._ws = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                open_timeout=self.cfg.timeout_s,
                close_timeout=self.cfg.timeout_s,
            )
        except (ConnectionRefusedError, OSError, TimeoutError, socket.timeout) as exc:
            raise PolicyConnectionError(f"Failed to connect to policy server at {self._uri}: {exc}") from exc

        try:
            metadata_frame = self._ws.recv(timeout=self.cfg.timeout_s)
        except TimeoutError as exc:
            raise PolicyTimeoutError(
                f"Timed out while waiting for policy server metadata from {self._uri}"
            ) from exc

        if isinstance(metadata_frame, str):
            raise PolicyResponseError(f"Expected binary metadata frame from {self._uri}, got text: {metadata_frame}")
        self._server_metadata = msgpack_numpy.unpackb(metadata_frame)

    def infer(self, policy_input_dict: dict) -> np.ndarray:
        if self._ws is None:
            raise PolicyConnectionError("Policy client is not connected. Call connect() first.")

        try:
            adapted_input = self._adapt_policy_input(policy_input_dict)
            payload = self._packer.pack(adapted_input)
            start_time = time.monotonic()
            self._ws.send(payload)
            response = self._ws.recv(timeout=self.cfg.timeout_s)
            _ = time.monotonic() - start_time
        except TimeoutError as exc:
            raise PolicyTimeoutError(f"Timed out while waiting for policy response from {self._uri}") from exc
        except (OSError, websockets.exceptions.WebSocketException) as exc:  # type: ignore[attr-defined]
            raise PolicyConnectionError(f"Websocket communication error with {self._uri}: {exc}") from exc

        if isinstance(response, str):
            raise PolicyResponseError(f"Policy server returned text error:\n{response}")

        response_dict = msgpack_numpy.unpackb(response)
        if not isinstance(response_dict, dict):
            raise PolicyResponseError(f"Expected dict response from policy server, got {type(response_dict)}")
        return self._adapt_policy_output(response_dict)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


def _resize_uint8_hwc(image: np.ndarray, size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise PolicyResponseError(f"Expected uint8 image for policy input adaptation, got {image.dtype}")
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((size, size), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)
