"""Policy client and action buffer directly reusing inference-origin's StreamActionBuffer."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np


def ensure_inference_origin_paths(inference_origin_root: str | Path) -> None:
    origin = Path(inference_origin_root)
    # Put client/inference at index 0 so robot_io's `from config import camera_serial`
    # resolves inference-origin's config.py rather than kai0-sdk-dagger/config/.
    for rel in ("packages/openpi-client/src", "client/inference"):
        candidate = str(origin / rel)
        if candidate in sys.path:
            sys.path.remove(candidate)
        sys.path.insert(0, candidate)


class TemporalSmoothingScheduler:
    """Thread-safe wrapper over inference-origin's exact StreamActionBuffer(smooth_method='temporal')."""

    def __init__(self, *, chunk_size: int = 50, latency_k: int = 8, min_smooth_steps: int = 8,
                 inference_origin_root: str | Path = "/home/ubuntu/dev/inference-origin"):
        ensure_inference_origin_paths(inference_origin_root)
        from action_buffers import StreamActionBuffer

        self._StreamActionBuffer = StreamActionBuffer
        self.chunk_size = int(chunk_size)
        self.latency_k = int(latency_k)
        self.min_smooth_steps = int(min_smooth_steps)
        self._lock = threading.Lock()
        self._generation = 0
        self._buf = self._new_buffer()

    def _new_buffer(self):
        return self._StreamActionBuffer(max_chunks=10, state_dim=14, smooth_method="temporal")

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def reset(self) -> int:
        with self._lock:
            self._generation += 1
            self._buf = self._new_buffer()
            return self._generation

    def update(self, actions: np.ndarray, *, generation: int, chunk_id: int | None = None) -> bool:
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 14 or not np.isfinite(arr).all():
            raise ValueError(f"策略输出动作非法：shape={arr.shape}")
        if arr.shape[0] != self.chunk_size:
            raise ValueError(f"策略输出步数不符：期望 {self.chunk_size}，实际 {arr.shape[0]}")
        with self._lock:
            if generation != self._generation:
                return False
            self._buf.integrate_new_chunk(
                arr,
                max_k=self.latency_k,
                min_m=self.min_smooth_steps,
                chunk_id=chunk_id,
            )
            return True

    def pop_step(self) -> dict[str, Any] | None:
        with self._lock:
            return self._buf.pop_next_action()

    def pop(self) -> np.ndarray | None:
        step = self.pop_step()
        if step is None:
            return None
        return np.asarray(step["action"], dtype=np.float64)

    def remaining(self) -> int:
        with self._lock:
            return int(self._buf.pending_count())


class SDKPolicyClient:
    """Thin wrapper over inference-origin's WebsocketClientPolicy / mock policy."""

    def __init__(self, cfg: dict[str, Any], *, mock: bool = False):
        self.cfg = cfg
        self.mock = bool(mock)
        self.server_cfg = cfg["server"]
        self.infer_cfg = cfg["inference"]
        self.chunk_size = int(self.infer_cfg.get("chunk_size", 50))
        self.image_size = int(self.infer_cfg.get("image_size", 224))
        self.image_preprocessing = str(self.infer_cfg.get("image_preprocessing", "openpi")).strip().lower()
        self.prompt = str(self.infer_cfg.get("prompt", "Flatten and fold the cloth."))
        self.num_steps = int(self.infer_cfg.get("num_denoising_steps", 10))
        self._client = None

        if not self.mock:
            ensure_inference_origin_paths(cfg["runtime"]["inference_origin_root"])
            host = str(self.server_cfg.get("host", "127.0.0.1"))
            exclusions = [x for x in os.environ.get("NO_PROXY", "").split(",") if x]
            for item in ("127.0.0.1", "localhost", host):
                if item not in exclusions:
                    exclusions.append(item)
            os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(exclusions)
            from policy_clients import create_policy_client
            self._client = create_policy_client(self.server_cfg)

    def build_payload(self, images_bgr: dict[str, np.ndarray], qpos: np.ndarray) -> dict[str, Any]:
        top_rgb = cv2.cvtColor(images_bgr["cam_high"], cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(images_bgr["cam_right_wrist"], cv2.COLOR_BGR2RGB)
        left_rgb = cv2.cvtColor(images_bgr["cam_left_wrist"], cv2.COLOR_BGR2RGB)

        if self.image_preprocessing == "model":
            resized = [top_rgb, right_rgb, left_rgb]
        elif self.mock:
            resized = [cv2.resize(img, (self.image_size, self.image_size)) for img in (top_rgb, right_rgb, left_rgb)]
        else:
            from openpi_client import image_tools
            resized = image_tools.resize_with_pad(
                np.asarray([top_rgb, right_rgb, left_rgb]),
                self.image_size,
                self.image_size,
            )

        return {
            "images": {
                "top_head": np.asarray(resized[0], dtype=np.uint8).transpose(2, 0, 1),
                "hand_right": np.asarray(resized[1], dtype=np.uint8).transpose(2, 0, 1),
                "hand_left": np.asarray(resized[2], dtype=np.uint8).transpose(2, 0, 1),
            },
            # Match inference-origin's BaseInferenceMode (float64 proprio)
            "state": np.asarray(qpos, dtype=float),
            "prompt": self.prompt,
            "num_steps": self.num_steps,
        }

    def infer(self, images_bgr: dict[str, np.ndarray], qpos: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.monotonic()
        if self.mock:
            base = np.asarray(qpos, dtype=np.float64).reshape(14)
            step = np.linspace(0.0, 0.02, self.chunk_size)[:, None]
            actions = np.repeat(base[None, :], self.chunk_size, axis=0)
            actions[:, 1] += step[:, 0]
            actions[:, 8] += step[:, 0]
            return actions, (time.monotonic() - t0) * 1000.0

        payload = self.build_payload(images_bgr, qpos)
        out = self._client.infer(payload)
        if not isinstance(out, dict) or "actions" not in out:
            raise RuntimeError(f"策略服务返回异常：{type(out)}")
        actions = np.asarray(out["actions"], dtype=np.float64)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.shape != (self.chunk_size, 14) or not np.isfinite(actions).all():
            raise RuntimeError(f"策略输出动作维度或数值非法：{actions.shape}")
        return actions, (time.monotonic() - t0) * 1000.0

    def close(self) -> None:
        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()
        else:
            ws = getattr(self._client, "_ws", None)
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
