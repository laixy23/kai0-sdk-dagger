"""Zero-ROS LeRobot v2.1 (and legacy HDF5) episode writer with fast parallel async MP4 export."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
LEROBOT_CAMERA_KEYS = {
    "cam_high": "observation.images.top_head",
    "cam_left_wrist": "observation.images.hand_left",
    "cam_right_wrist": "observation.images.hand_right",
}
DEFAULT_CHUNK_SIZE = 1000


@dataclass
class FrameSample:
    timestamp: float
    qpos: np.ndarray        # (14,) float32
    qvel: np.ndarray        # (14,) float32
    effort: np.ndarray      # (14,) float32
    images_bgr: dict[str, np.ndarray]  # cam_high / cam_left_wrist / cam_right_wrist -> (H, W, 3) uint8 BGR


@dataclass
class VideoExportJob:
    tasks: list[tuple[str, list[np.ndarray], Path]]  # (label, frames_bgr, output_mp4_path)
    episode_idx: int
    fps: int = 30
    codec: str = "libx264"
    preset: str = "ultrafast"
    crf: int = 18
    video_paths: list[Path] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None
    on_complete: Callable[[], None] | None = None

    def cancel(self) -> None:
        self.cancel_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self.done_event.wait(timeout=timeout)


def write_mp4_from_bgr(frames_bgr: list[np.ndarray], output_path: Path, *,
                       fps: int = 30, codec: str = "libx264", preset: str = "ultrafast",
                       crf: int = 18, cancel_event: threading.Event | None = None) -> bool:
    if not frames_bgr:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_bgr[0].shape[:2]
    container = av.open(str(output_path), mode="w")
    cancelled = False
    try:
        stream = container.add_stream(codec, rate=int(fps))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        opts = {"crf": str(int(crf)), "threads": "1"}
        if preset:
            opts["preset"] = str(preset)
        stream.options = opts
        for bgr in frames_bgr:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        if not cancelled:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    if cancelled and output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass
        return False
    return True


def export_video_tasks_parallel(tasks: list[tuple[str, list[np.ndarray], Path]], *,
                                fps: int = 30, codec: str = "libx264", preset: str = "ultrafast",
                                crf: int = 18, cancel_event: threading.Event | None = None) -> list[Path]:
    if not tasks:
        return []

    def _encode(item: tuple[str, list[np.ndarray], Path]) -> Path | None:
        _label, frames, vp = item
        ok = write_mp4_from_bgr(frames, vp, fps=fps, codec=codec, preset=preset,
                                crf=crf, cancel_event=cancel_event)
        return vp if ok else None

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        results = list(pool.map(_encode, tasks))
    return [vp for vp in results if vp is not None]


class AsyncVideoExporter:
    """Background worker encoding 3-camera MP4s in parallel without blocking robot control."""

    def __init__(self) -> None:
        self._queue: queue.Queue[VideoExportJob | None] = queue.Queue()
        self._pending_lock = threading.Lock()
        self._pending = 0
        self._jobs: list[VideoExportJob] = []
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return self._pending

    def submit(self, job: VideoExportJob) -> VideoExportJob:
        with self._pending_lock:
            self._pending += 1
            self._jobs.append(job)
        self._queue.put(job)
        return job

    def cancel_all(self) -> None:
        with self._pending_lock:
            jobs = list(self._jobs)
        for job in jobs:
            job.cancel()
        self._queue.join()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                if not job.cancel_event.is_set():
                    t0 = time.monotonic()
                    paths = export_video_tasks_parallel(
                        job.tasks,
                        fps=job.fps, codec=job.codec, preset=job.preset, crf=job.crf,
                        cancel_event=job.cancel_event,
                    )
                    job.video_paths = paths
                    if not job.cancel_event.is_set() and paths:
                        print(
                            f"\033[32m[后台视频] episode_{job.episode_idx:06d} 三路视频导出完成（耗时 {time.monotonic() - t0:.2f} 秒）\033[0m",
                            flush=True,
                        )
                        if job.on_complete is not None:
                            try:
                                job.on_complete()
                            except Exception as vexc:
                                print(
                                    f"\033[31m[校验失败] episode_{job.episode_idx:06d} 读回校验异常：{vexc}\033[0m",
                                    flush=True,
                                )
            except Exception as exc:
                job.error = exc
                print(f"\033[31m[后台视频] episode_{job.episode_idx:06d} 导出失败：{exc}\033[0m", flush=True)
            finally:
                job.tasks.clear()
                job.done_event.set()
                with self._pending_lock:
                    self._pending = max(0, self._pending - 1)
                    try:
                        self._jobs.remove(job)
                    except ValueError:
                        pass
                self._queue.task_done()

    def wait_all(self, *, announce: bool = True) -> None:
        if announce and self.pending_count > 0:
            print(f"[保存] 正在等待后台视频导出完成（剩余 {self.pending_count} 段）……", flush=True)
        self._queue.join()

    def close(self, *, wait: bool = True) -> None:
        if wait:
            self.wait_all(announce=True)
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def build_training_arrays(samples: list[FrameSample]) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    """Pair observation[t] with action[t] = qpos[t+1], matching official Kai0 DAgger."""
    if len(samples) < 2:
        return {}, {}
    obs_samples = samples[:-1]
    actions = np.stack([s.qpos for s in samples[1:]], axis=0).astype(np.float32)
    qpos = np.stack([s.qpos for s in obs_samples], axis=0).astype(np.float32)
    qvel = np.stack([s.qvel for s in obs_samples], axis=0).astype(np.float32)
    effort = np.stack([s.effort for s in obs_samples], axis=0).astype(np.float32)
    base_action = np.zeros((len(obs_samples), 2), dtype=np.float32)
    arrays = {
        "/observations/qpos": qpos,
        "/observations/qvel": qvel,
        "/observations/effort": effort,
        "/action": actions,
        "/base_action": base_action,
    }
    videos: dict[str, list[np.ndarray]] = {cam: [] for cam in CAMERA_NAMES}
    for s in obs_samples:
        for cam in CAMERA_NAMES:
            if cam in s.images_bgr:
                videos[cam].append(s.images_bgr[cam])
    return arrays, videos


def _vector_stats(arr: np.ndarray) -> dict[str, Any]:
    arr64 = np.asarray(arr, dtype=np.float64)
    if arr64.ndim == 1:
        arr64 = arr64[:, None]
    return {
        "min": arr64.min(axis=0).tolist(),
        "max": arr64.max(axis=0).tolist(),
        "mean": arr64.mean(axis=0).tolist(),
        "std": arr64.std(axis=0).tolist(),
        "count": [int(arr64.shape[0])],
    }


def _video_rgb_stats(frames_bgr: list[np.ndarray]) -> dict[str, Any]:
    """Fast sampled per-channel RGB stats in [0, 1] with shape (3, 1, 1) matching LeRobot v2.1."""
    n = len(frames_bgr)
    if n == 0:
        zeros = [[[0.0]], [[0.0]], [[0.0]]]
        return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
    stride = max(1, n // 16)
    sampled = frames_bgr[::stride]
    # BGR -> RGB in [0, 1], shape (K, H, W, 3) -> channel stats over (K, H, W)
    stack = np.stack([f[:, :, ::-1] for f in sampled], axis=0).astype(np.float32) / 255.0
    c_min = stack.min(axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    c_max = stack.max(axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    c_mean = stack.mean(axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    c_std = stack.std(axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
    return {"min": c_min, "max": c_max, "mean": c_mean, "std": c_std, "count": [int(len(sampled))]}


class LeRobotV21DatasetWriter:
    """Direct Kai0-compatible LeRobot v2.1 dataset writer (Parquet + 3x MP4 + meta/*)."""

    def __init__(self, root_dir: str | Path, *, prompt: str, fps: int = 30,
                 robot_type: str = "agilex", video_codec: str = "libx264",
                 video_preset: str = "ultrafast", video_crf: int = 18,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 video_exporter: AsyncVideoExporter | None = None):
        self.root_dir = Path(root_dir)
        self.meta_dir = self.root_dir / "meta"
        self.prompt = str(prompt)
        self.fps = int(fps)
        self.robot_type = str(robot_type)
        self.video_codec = str(video_codec)
        self.video_preset = str(video_preset)
        self.video_crf = int(video_crf)
        self.chunk_size = int(chunk_size)
        self.video_exporter = video_exporter

        self._lock = threading.Lock()
        self._episodes: list[dict[str, Any]] = []
        self._episodes_stats: list[dict[str, Any]] = []
        self._next_attempt_index: int = 0
        self._pending_deleted_attempts: list[int] = []
        self._video_shape: tuple[int, int] = (480, 640)
        self._load_existing()

    def _load_existing(self) -> None:
        ep_path = self.meta_dir / "episodes.jsonl"
        if ep_path.is_file():
            for line in ep_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._episodes.append(json.loads(line))
        stats_path = self.meta_dir / "episodes_stats.jsonl"
        if stats_path.is_file():
            for line in stats_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self._episodes_stats.append(json.loads(line))
        info_path = self.meta_dir / "info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                top_feat = info.get("features", {}).get("observation.images.top_head", {})
                sh = top_feat.get("shape")
                if isinstance(sh, list) and len(sh) >= 2:
                    self._video_shape = (int(sh[0]), int(sh[1]))
            except Exception:
                pass
        timeline_path = self.meta_dir / "session_timeline.jsonl"
        if timeline_path.is_file():
            max_attempt = -1
            pending_deleted: list[int] = []
            for line in timeline_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                att = ev.get("attempt_index")
                if att is not None:
                    max_attempt = max(max_attempt, int(att))
                if ev.get("event") == "saved":
                    pending_deleted = []
                elif ev.get("event") == "deleted" and att is not None:
                    prev_del = ev.get("deleted_attempts_before") or []
                    pending_deleted = [int(x) for x in prev_del] + [int(att)]
            self._next_attempt_index = max_attempt + 1
            self._pending_deleted_attempts = pending_deleted
        elif self._episodes:
            self._next_attempt_index = max(
                int(ep.get("attempt_index", ep["episode_index"])) for ep in self._episodes
            ) + 1

    def _append_timeline_locked(self, event: dict[str, Any]) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        with open(self.meta_dir / "session_timeline.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_collision_avoidance(
        self, info: dict[str, Any], *, after_inference_episode: int | None = None
    ) -> None:
        """Record a handle collision-avoidance follower nudge into meta/session_timeline.jsonl."""
        ev: dict[str, Any] = {
            "event": "collision_avoidance",
            "timestamp": info.get("timestamp") or datetime.datetime.now().isoformat(timespec="milliseconds"),
            "safety_margin_mm": info.get("safety_margin_mm"),
            "sides": info.get("sides") or {},
        }
        if after_inference_episode is not None:
            ev["after_inference_episode"] = int(after_inference_episode)
        with self._lock:
            self._append_timeline_locked(ev)

    @property
    def next_episode_index(self) -> int:
        with self._lock:
            if not self._episodes:
                return 0
            return max(int(ep["episode_index"]) for ep in self._episodes) + 1

    @property
    def episode_count(self) -> int:
        with self._lock:
            return len(self._episodes)

    @property
    def total_frames(self) -> int:
        with self._lock:
            return sum(int(ep["length"]) for ep in self._episodes)

    @property
    def deleted_attempts_count(self) -> int:
        with self._lock:
            return max(0, self._next_attempt_index - len(self._episodes))

    def _codec_name_for_info(self) -> str:
        c = self.video_codec.lower()
        if "264" in c:
            return "h264"
        if "265" in c or "hevc" in c:
            return "hevc"
        if "av1" in c:
            return "av1"
        return c

    def _write_meta_locked(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        # 1. meta/tasks.jsonl
        (self.meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": self.prompt}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        # 2. meta/episodes.jsonl
        (self.meta_dir / "episodes.jsonl").write_text(
            "".join(json.dumps(ep, ensure_ascii=False) + "\n" for ep in self._episodes),
            encoding="utf-8",
        )
        # 3. meta/episodes_stats.jsonl
        (self.meta_dir / "episodes_stats.jsonl").write_text(
            "".join(json.dumps(st, ensure_ascii=False) + "\n" for st in self._episodes_stats),
            encoding="utf-8",
        )
        # 4. meta/info.json
        total_ep = len(self._episodes)
        total_fr = sum(int(ep["length"]) for ep in self._episodes)
        total_chunks = max(1, (total_ep + self.chunk_size - 1) // self.chunk_size) if total_ep > 0 else 0
        h, w = self._video_shape
        codec_str = self._codec_name_for_info()
        video_feature = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": codec_str,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": self.fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.robot_type,
            "total_episodes": total_ep,
            "total_frames": total_fr,
            "total_tasks": 1,
            "total_videos": total_ep * len(CAMERA_NAMES),
            "total_chunks": total_chunks,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_ep}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.top_head": dict(video_feature),
                "observation.images.hand_left": dict(video_feature),
                "observation.images.hand_right": dict(video_feature),
                "observation.state": {"dtype": "float32", "shape": [14], "names": None},
                "action": {"dtype": "float32", "shape": [14], "names": None},
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        tmp = self.meta_dir / "info.json.tmp"
        tmp.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.meta_dir / "info.json")

    def save_episode(self, samples: list[FrameSample], *, export_video: bool = True,
                     after_inference_episode: int | None = None,
                     collector_name: str | None = None,
                     collision_avoidance_before: dict[str, Any] | None = None) -> dict[str, Any]:
        arrays, videos = build_training_arrays(samples)
        if not arrays:
            return {"saved": False, "frames": 0, "data_path": None, "video_paths": [], "video_job": None}

        t0 = time.monotonic()
        qpos = arrays["/observations/qpos"]   # (N, 14) float32
        action = arrays["/action"]            # (N, 14) float32
        num_frames = int(action.shape[0])

        for cam in CAMERA_NAMES:
            if videos.get(cam):
                h, w = videos[cam][0].shape[:2]
                self._video_shape = (int(h), int(w))
                break

        with self._lock:
            ep_idx = (max(int(ep["episode_index"]) for ep in self._episodes) + 1) if self._episodes else 0
            start_global_idx = sum(int(ep["length"]) for ep in self._episodes)
            chunk_idx = ep_idx // self.chunk_size

            parquet_path = self.root_dir / f"data/chunk-{chunk_idx:03d}/episode_{ep_idx:06d}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)

            timestamps = (np.arange(num_frames, dtype=np.float32) / float(self.fps))
            frame_indices = np.arange(num_frames, dtype=np.int64)
            episode_indices = np.full(num_frames, ep_idx, dtype=np.int64)
            global_indices = np.arange(start_global_idx, start_global_idx + num_frames, dtype=np.int64)
            task_indices = np.zeros(num_frames, dtype=np.int64)

            table = pa.table({
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(qpos.reshape(-1), type=pa.float32()), 14
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(action.reshape(-1), type=pa.float32()), 14
                ),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(episode_indices, type=pa.int64()),
                "index": pa.array(global_indices, type=pa.int64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
            })
            pq.write_table(table, parquet_path)

            ep_stats = {
                "observation.state": _vector_stats(qpos),
                "action": _vector_stats(action),
                "timestamp": _vector_stats(timestamps),
                "frame_index": _vector_stats(frame_indices),
                "episode_index": _vector_stats(episode_indices),
                "index": _vector_stats(global_indices),
                "task_index": _vector_stats(task_indices),
            }
            for cam in CAMERA_NAMES:
                lr_key = LEROBOT_CAMERA_KEYS[cam]
                ep_stats[lr_key] = _video_rgb_stats(videos.get(cam) or [])

            attempt_idx = self._next_attempt_index
            self._next_attempt_index += 1
            deleted_before = list(self._pending_deleted_attempts)
            self._pending_deleted_attempts.clear()

            ep_entry: dict[str, Any] = {
                "episode_index": ep_idx,
                "tasks": [self.prompt],
                "length": num_frames,
                "attempt_index": attempt_idx,
                "deleted_attempts_before": deleted_before,
            }
            if after_inference_episode is not None:
                ep_entry["after_inference_episode"] = int(after_inference_episode)
            if collector_name:
                ep_entry["collector_name"] = str(collector_name)
            if collision_avoidance_before is not None:
                ep_entry["collision_avoidance_before"] = collision_avoidance_before

            self._episodes.append(ep_entry)
            self._episodes_stats.append({
                "episode_index": ep_idx,
                "stats": ep_stats,
            })
            self._write_meta_locked()
            timeline_ev: dict[str, Any] = {
                "event": "saved",
                "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "attempt_index": attempt_idx,
                "episode_index": ep_idx,
                "frames": num_frames,
                "deleted_attempts_before": deleted_before,
            }
            if after_inference_episode is not None:
                timeline_ev["after_inference_episode"] = int(after_inference_episode)
            if collector_name:
                timeline_ev["collector_name"] = str(collector_name)
            if collision_avoidance_before is not None:
                timeline_ev["collision_avoidance_before"] = collision_avoidance_before
            self._append_timeline_locked(timeline_ev)

        print(f"\033[32mLeRobot Parquet 保存完成（耗时 {time.monotonic() - t0:.2f} 秒）：{parquet_path}\033[0m", flush=True)

        def _verify_and_write_receipt(*, check_vid: bool) -> None:
            from sdk_dagger.verify import verify_episode_v21, write_save_receipt
            rep = verify_episode_v21(
                self.root_dir, ep_idx, expected_frames=num_frames, check_video=check_vid
            )
            write_save_receipt(self.root_dir, ep_idx, rep)
            if rep.get("warnings"):
                print(
                    f"\033[33m[校验通过·含提示] {self.root_dir.name}/episode_{ep_idx:06d}（{num_frames} 帧）：{'; '.join(rep['warnings'])}\033[0m",
                    flush=True,
                )

        video_paths: list[Path] = []
        video_job: VideoExportJob | None = None
        if export_video:
            tasks: list[tuple[str, list[np.ndarray], Path]] = []
            for cam in CAMERA_NAMES:
                frames = videos.get(cam) or []
                if frames:
                    lr_key = LEROBOT_CAMERA_KEYS[cam]
                    vp = self.root_dir / f"videos/chunk-{chunk_idx:03d}/{lr_key}/episode_{ep_idx:06d}.mp4"
                    tasks.append((lr_key, frames, vp))
            expected_paths = [vp for _, _, vp in tasks]
            if self.video_exporter is not None:
                video_job = VideoExportJob(
                    tasks=tasks,
                    episode_idx=ep_idx,
                    fps=self.fps,
                    codec=self.video_codec,
                    preset=self.video_preset,
                    crf=self.video_crf,
                    video_paths=expected_paths,
                    on_complete=lambda: _verify_and_write_receipt(check_vid=True),
                )
                self.video_exporter.submit(video_job)
                video_paths = expected_paths
                print(f"\033[33m已提交后台异步并行导出 LeRobot 三路视频（episode_{ep_idx:06d}，不阻塞操作）……\033[0m", flush=True)
            else:
                print("\033[33m正在并行导出 LeRobot 三路视频……\033[0m", flush=True)
                video_paths = export_video_tasks_parallel(
                    tasks, fps=self.fps, codec=self.video_codec, preset=self.video_preset, crf=self.video_crf
                )
                print(f"\033[32m全部视频导出完成（总耗时 {time.monotonic() - t0:.2f} 秒）\033[0m", flush=True)
                try:
                    _verify_and_write_receipt(check_vid=True)
                except Exception as vexc:
                    print(f"\033[31m[校验失败] episode_{ep_idx:06d} 读回校验异常：{vexc}\033[0m", flush=True)
        else:
            try:
                _verify_and_write_receipt(check_vid=False)
            except Exception as vexc:
                print(f"\033[31m[校验失败] episode_{ep_idx:06d} 读回校验异常：{vexc}\033[0m", flush=True)

        return {
            "saved": True,
            "format": "lerobot",
            "frames": num_frames,
            "data_path": parquet_path,
            "hdf5_path": parquet_path,  # alias for uniform callers
            "video_paths": video_paths,
            "video_job": video_job,
            "episode_idx": ep_idx,
            "attempt_idx": attempt_idx,
            "deleted_attempts_before": deleted_before,
            "after_inference_episode": after_inference_episode,
            "collector_name": collector_name,
        }

    def delete_last_episode(self, res: dict[str, Any], *, reason: str = "operator_w_delete") -> None:
        video_job = res.get("video_job")
        if video_job is not None:
            video_job.cancel()
            video_job.wait(timeout=5.0)
        ep_idx = res.get("episode_idx")
        attempt_idx = res.get("attempt_idx")
        prev_deleted = list(res.get("deleted_attempts_before") or [])
        after_inf = res.get("after_inference_episode")
        data_path = res.get("data_path")
        if data_path and Path(data_path).exists():
            Path(data_path).unlink()
            print(f"\033[33m已删除 Parquet 文件：{data_path}\033[0m", flush=True)
        for vp in res.get("video_paths") or []:
            if Path(vp).exists():
                Path(vp).unlink()
                print(f"\033[33m已删除视频文件：{vp}\033[0m", flush=True)
        if ep_idx is not None:
            from sdk_dagger.verify import delete_save_receipt
            delete_save_receipt(self.root_dir, int(ep_idx))
        with self._lock:
            self._episodes = [ep for ep in self._episodes if int(ep["episode_index"]) != ep_idx]
            self._episodes_stats = [st for st in self._episodes_stats if int(st["episode_index"]) != ep_idx]
            if attempt_idx is not None:
                self._pending_deleted_attempts = [int(x) for x in prev_deleted] + [int(attempt_idx)]
            self._write_meta_locked()
            del_ev: dict[str, Any] = {
                "event": "deleted",
                "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "attempt_index": attempt_idx,
                "episode_index": ep_idx,
                "frames": int(res.get("frames", 0)),
                "deleted_attempts_before": prev_deleted,
                "reason": reason,
            }
            if after_inf is not None:
                del_ev["after_inference_episode"] = int(after_inf)
            if res.get("collector_name"):
                del_ev["collector_name"] = str(res["collector_name"])
            self._append_timeline_locked(del_ev)


# ---- Legacy HDF5 helpers (kept when recording.format == 'hdf5') ----

def next_episode_idx(dataset_dir: str | Path) -> int:
    directory = Path(dataset_dir)
    if not directory.is_dir():
        return 0
    indices: list[int] = []
    for name in os.listdir(directory):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            try:
                indices.append(int(name.split("_", 1)[1].rsplit(".", 1)[0]))
            except ValueError:
                continue
    return (max(indices) + 1) if indices else 0


def save_episode(samples: list[FrameSample], dataset_dir: str | Path, episode_idx: int, *,
                 export_video: bool = True, video_fps: int = 30,
                 video_codec: str = "libx264", video_preset: str = "ultrafast",
                 video_crf: int = 18,
                 video_exporter: AsyncVideoExporter | None = None) -> dict[str, Any]:
    arrays, videos = build_training_arrays(samples)
    if not arrays:
        return {"saved": False, "frames": 0, "data_path": None, "hdf5_path": None, "video_paths": [], "video_job": None}

    t0 = time.monotonic()
    out_dir = Path(dataset_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_dir / f"episode_{episode_idx}.hdf5"
    num_frames = int(arrays["/action"].shape[0])

    with h5py.File(hdf5_path, "w", rdcc_nbytes=1024 * 1024 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        obs = root.create_group("observations")
        for key in ("qpos", "qvel", "effort"):
            obs.create_dataset(key, data=arrays[f"/observations/{key}"], dtype="float32")
        root.create_dataset("action", data=arrays["/action"], dtype="float32")
        root.create_dataset("base_action", data=arrays["/base_action"], dtype="float32")

    print(f"\033[32mHDF5 保存完成（耗时 {time.monotonic() - t0:.2f} 秒）：{hdf5_path}\033[0m", flush=True)

    video_paths: list[Path] = []
    video_job: VideoExportJob | None = None
    if export_video:
        tasks = [
            (cam, videos[cam], out_dir / "video" / cam / f"episode_{episode_idx}.mp4")
            for cam in CAMERA_NAMES
            if videos.get(cam)
        ]
        expected_paths = [vp for _, _, vp in tasks]
        if video_exporter is not None:
            video_job = VideoExportJob(
                tasks=tasks,
                episode_idx=episode_idx,
                fps=video_fps,
                codec=video_codec,
                preset=video_preset,
                crf=video_crf,
                video_paths=expected_paths,
            )
            video_exporter.submit(video_job)
            video_paths = expected_paths
            print(f"\033[33m已提交后台异步并行导出三路视频（episode_{episode_idx}，不阻塞操作）……\033[0m", flush=True)
        else:
            print("\033[33m正在并行导出三路相机视频……\033[0m", flush=True)
            video_paths = export_video_tasks_parallel(
                tasks, fps=video_fps, codec=video_codec, preset=video_preset, crf=video_crf
            )
            print(f"\033[32m全部视频导出完成（总耗时 {time.monotonic() - t0:.2f} 秒）\033[0m", flush=True)

    return {
        "saved": True,
        "format": "hdf5",
        "frames": num_frames,
        "data_path": hdf5_path,
        "hdf5_path": hdf5_path,
        "video_paths": video_paths,
        "video_job": video_job,
        "episode_idx": episode_idx,
    }


def delete_episode(hdf5_path: Path, video_paths: list[Path], *,
                   video_job: VideoExportJob | None = None) -> None:
    if video_job is not None:
        video_job.cancel()
        video_job.wait(timeout=5.0)
    if hdf5_path and hdf5_path.exists():
        hdf5_path.unlink()
        print(f"\033[33m已删除数据文件：{hdf5_path}\033[0m", flush=True)
    for vp in video_paths:
        if vp.exists():
            vp.unlink()
            print(f"\033[33m已删除视频文件：{vp}\033[0m", flush=True)


# ---- Round-end continuous LeRobot v2.1 stitching (splits on collision_avoidance or dagger_deleted) ----

def _concat_mp4_files(src_paths: list[Path], dst_path: Path, *, fps: int = 30) -> bool:
    valid = [p for p in src_paths if p.is_file()]
    if not valid:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if len(valid) == 1:
        shutil.copy2(valid[0], dst_path)
        return True
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            list_file = Path(f.name)
            for p in valid:
                escaped = str(p.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        try:
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                 "-i", str(list_file), "-c", "copy", str(dst_path)],
                check=True,
                timeout=60.0,
            )
            return True
        finally:
            try:
                list_file.unlink()
            except OSError:
                pass
    # Pure-PyAV fallback if ffmpeg binary is unavailable
    frames_bgr: list[np.ndarray] = []
    for p in valid:
        with av.open(str(p)) as container:
            for frame in container.decode(video=0):
                frames_bgr.append(frame.to_ndarray(format="bgr24"))
    return write_mp4_from_bgr(frames_bgr, dst_path, fps=fps)


def _pool_image_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-episode (3,1,1) RGB stats across stitched segments without decoding videos."""
    valid = [st for st in stats_list if st and (st.get("count") or [0])[0] > 0]
    if not valid:
        zeros = [[[0.0]], [[0.0]], [[0.0]]]
        return {"min": zeros, "max": zeros, "mean": zeros, "std": zeros, "count": [0]}
    counts = np.array([float(st["count"][0]) for st in valid], dtype=np.float64)
    total = float(counts.sum())
    mins = np.stack([np.asarray(st["min"], dtype=np.float64) for st in valid], axis=0)
    maxs = np.stack([np.asarray(st["max"], dtype=np.float64) for st in valid], axis=0)
    means = np.stack([np.asarray(st["mean"], dtype=np.float64) for st in valid], axis=0)
    stds = np.stack([np.asarray(st["std"], dtype=np.float64) for st in valid], axis=0)
    w = counts[:, None, None, None]
    pooled_mean = (w * means).sum(axis=0) / total
    pooled_var = (w * (stds ** 2 + (means - pooled_mean[None, ...]) ** 2)).sum(axis=0) / total
    return {
        "min": mins.min(axis=0).tolist(),
        "max": maxs.max(axis=0).tolist(),
        "mean": pooled_mean.tolist(),
        "std": np.sqrt(np.maximum(pooled_var, 0.0)).tolist(),
        "count": [int(round(total))],
    }


def _load_source_stats_map(dataset_dir: Path) -> dict[int, dict[str, Any]]:
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    out: dict[int, dict[str, Any]] = {}
    if stats_path.is_file():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                out[int(row["episode_index"])] = row.get("stats") or {}
    return out


def split_stitch_events_into_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split chronological round events into contiguous segment runs separated by breaks."""
    runs: list[dict[str, Any]] = []
    cur_segments: list[dict[str, Any]] = []
    pending_breaks: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("kind") == "break":
            if cur_segments:
                runs.append({"segments": cur_segments, "breaks_before": pending_breaks})
                cur_segments = []
                pending_breaks = [ev]
            else:
                pending_breaks.append(ev)
        elif ev.get("kind") == "segment":
            cur_segments.append(ev)
    if cur_segments:
        runs.append({"segments": cur_segments, "breaks_before": pending_breaks})
    return runs


def merge_round_lerobot_v21(
    *,
    inference_dir: str | Path,
    dagger_dir: str | Path,
    merged_dir: str | Path,
    stitch_events: list[dict[str, Any]],
    prompt: str,
    fps: int = 30,
    robot_type: str = "agilex",
    video_codec: str = "libx264",
    export_video: bool = True,
    collector_name: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Stitch continuous inference+DAgger runs into a standalone LeRobot v2.1 dataset.

    Splits automatically wherever a collision-avoidance nudge or a deleted DAgger attempt occurred.
    Original inference/dagger datasets remain untouched.
    """
    t0 = time.monotonic()
    runs = split_stitch_events_into_runs(stitch_events)
    if not runs:
        return {"merged": False, "episodes": 0, "frames": 0, "merged_dir": str(merged_dir)}

    inf_root = Path(inference_dir)
    dag_root = Path(dagger_dir)
    out_root = Path(merged_dir)
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    src_roots = {"inference": inf_root, "dagger": dag_root}
    src_stats = {
        "inference": _load_source_stats_map(inf_root),
        "dagger": _load_source_stats_map(dag_root),
    }

    # Detect video resolution from first available source meta/info.json
    video_shape = (480, 640)
    for r in (inf_root, dag_root):
        ip = r / "meta" / "info.json"
        if ip.is_file():
            try:
                info = json.loads(ip.read_text(encoding="utf-8"))
                sh = info.get("features", {}).get("observation.images.top_head", {}).get("shape")
                if isinstance(sh, list) and len(sh) >= 2:
                    video_shape = (int(sh[0]), int(sh[1]))
                    break
            except Exception:
                pass

    episodes_rows: list[dict[str, Any]] = []
    episodes_stats_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    global_idx = 0
    merged_ep_idx = 0

    for run in runs:
        seg_states: list[np.ndarray] = []
        seg_actions: list[np.ndarray] = []
        seg_is_dagger: list[np.ndarray] = []
        seg_records: list[dict[str, Any]] = []
        frame_cursor = 0
        per_cam_mp4s: dict[str, list[Path]] = {lr_key: [] for lr_key in LEROBOT_CAMERA_KEYS.values()}
        per_cam_stats: dict[str, list[dict[str, Any]]] = {lr_key: [] for lr_key in LEROBOT_CAMERA_KEYS.values()}

        for seg in run["segments"]:
            src = str(seg["source"])
            src_ep = int(seg["episode_index"])
            src_chunk = src_ep // chunk_size
            root = src_roots[src]
            pq_path = root / f"data/chunk-{src_chunk:03d}/episode_{src_ep:06d}.parquet"
            if not pq_path.is_file():
                continue
            tbl = pq.read_table(pq_path)
            st_arr = np.asarray(tbl["observation.state"].to_pylist(), dtype=np.float32)
            ac_arr = np.asarray(tbl["action"].to_pylist(), dtype=np.float32)
            if st_arr.shape[0] == 0:
                continue
            n_seg = int(st_arr.shape[0])
            seg_states.append(st_arr)
            seg_actions.append(ac_arr)
            seg_is_dagger.append(np.full(n_seg, 1 if src == "dagger" else 0, dtype=np.int64))
            seg_records.append({
                "source": src,
                "episode_index": src_ep,
                "frames": n_seg,
                "start_frame": frame_cursor,
                "end_frame": frame_cursor + n_seg,
            })
            frame_cursor += n_seg
            ep_st = src_stats[src].get(src_ep, {})
            for lr_key in LEROBOT_CAMERA_KEYS.values():
                vp = root / f"videos/chunk-{src_chunk:03d}/{lr_key}/episode_{src_ep:06d}.mp4"
                if vp.is_file():
                    per_cam_mp4s[lr_key].append(vp)
                if lr_key in ep_st:
                    per_cam_stats[lr_key].append(ep_st[lr_key])

        if not seg_states:
            continue

        # Ensure exact action[seam] = observation.state[seam + 1] continuity across stitched segments
        for i in range(len(seg_actions) - 1):
            seg_actions[i] = seg_actions[i].copy()
            seg_actions[i][-1] = seg_states[i + 1][0]

        qpos = np.concatenate(seg_states, axis=0).astype(np.float32)
        action = np.concatenate(seg_actions, axis=0).astype(np.float32)
        is_dagger = np.concatenate(seg_is_dagger, axis=0).astype(np.int64)
        num_frames = int(qpos.shape[0])
        out_chunk = merged_ep_idx // chunk_size

        parquet_path = out_root / f"data/chunk-{out_chunk:03d}/episode_{merged_ep_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        timestamps = np.arange(num_frames, dtype=np.float32) / float(fps)
        frame_indices = np.arange(num_frames, dtype=np.int64)
        episode_indices = np.full(num_frames, merged_ep_idx, dtype=np.int64)
        global_indices = np.arange(global_idx, global_idx + num_frames, dtype=np.int64)
        task_indices = np.zeros(num_frames, dtype=np.int64)

        table = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(qpos.reshape(-1), type=pa.float32()), 14
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), 14
            ),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "is_dagger": pa.array(is_dagger, type=pa.int64()),
        })
        pq.write_table(table, parquet_path)

        # Stitch 3 camera MP4s in parallel (lossless stream-copy via ffmpeg)
        if export_video:
            cam_tasks = [
                (lr_key, per_cam_mp4s[lr_key],
                 out_root / f"videos/chunk-{out_chunk:03d}/{lr_key}/episode_{merged_ep_idx:06d}.mp4")
                for lr_key in LEROBOT_CAMERA_KEYS.values()
                if per_cam_mp4s[lr_key]
            ]
            if cam_tasks:
                with ThreadPoolExecutor(max_workers=len(cam_tasks)) as pool:
                    list(pool.map(lambda it: _concat_mp4_files(it[1], it[2], fps=fps), cam_tasks))

        ep_stats: dict[str, Any] = {
            "observation.state": _vector_stats(qpos),
            "action": _vector_stats(action),
            "timestamp": _vector_stats(timestamps),
            "frame_index": _vector_stats(frame_indices),
            "episode_index": _vector_stats(episode_indices),
            "index": _vector_stats(global_indices),
            "task_index": _vector_stats(task_indices),
            "is_dagger": _vector_stats(is_dagger),
        }
        for lr_key in LEROBOT_CAMERA_KEYS.values():
            ep_stats[lr_key] = _pool_image_stats(per_cam_stats[lr_key])

        breaks_before = [
            {k: v for k, v in brk.items() if k != "kind"}
            for brk in run["breaks_before"]
        ]
        ep_entry: dict[str, Any] = {
            "episode_index": merged_ep_idx,
            "tasks": [prompt],
            "length": num_frames,
            "source_segments": seg_records,
            "breaks_before": breaks_before,
        }
        if collector_name:
            ep_entry["collector_name"] = collector_name
        episodes_rows.append(ep_entry)
        episodes_stats_rows.append({"episode_index": merged_ep_idx, "stats": ep_stats})

        for brk in breaks_before:
            timeline_rows.append({"event": brk.get("reason", "break"), **brk})
        timeline_rows.append({
            "event": "merged_episode",
            "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "episode_index": merged_ep_idx,
            "frames": num_frames,
            "source_segments": seg_records,
        })

        global_idx += num_frames
        merged_ep_idx += 1

    if not episodes_rows:
        return {"merged": False, "episodes": 0, "frames": 0, "merged_dir": str(out_root)}

    # Write LeRobot v2.1 meta/*
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": prompt}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in episodes_rows), encoding="utf-8"
    )
    (meta_dir / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in episodes_stats_rows), encoding="utf-8"
    )
    (meta_dir / "session_timeline.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in timeline_rows), encoding="utf-8"
    )

    h, w = video_shape
    codec_l = video_codec.lower()
    codec_str = "h264" if "264" in codec_l else ("hevc" if ("265" in codec_l or "hevc" in codec_l) else codec_l)
    video_feature = {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": h,
            "video.width": w,
            "video.codec": codec_str,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }
    total_ep = len(episodes_rows)
    total_chunks = max(1, (total_ep + chunk_size - 1) // chunk_size)
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_ep,
        "total_frames": global_idx,
        "total_tasks": 1,
        "total_videos": total_ep * len(CAMERA_NAMES),
        "total_chunks": total_chunks,
        "chunks_size": chunk_size,
        "fps": int(fps),
        "splits": {"train": f"0:{total_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.top_head": dict(video_feature),
            "observation.images.hand_left": dict(video_feature),
            "observation.images.hand_right": dict(video_feature),
            "observation.state": {"dtype": "float32", "shape": [14], "names": None},
            "action": {"dtype": "float32", "shape": [14], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "is_dagger": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    tmp_info = meta_dir / "info.json.tmp"
    tmp_info.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_info, meta_dir / "info.json")

    elapsed = time.monotonic() - t0
    print(
        f"\033[32m[自动拼接] LeRobot v2.1 连续片段已生成（共 {total_ep} 段 / {global_idx} 帧，耗时 {elapsed:.2f} 秒）：{out_root}\033[0m",
        flush=True,
    )
    verification: dict[str, Any] | None = None
    verified = False
    try:
        from sdk_dagger.verify import verify_dataset_v21
        verification = verify_dataset_v21(out_root, write_receipts=True, check_video=export_video)
        verified = bool(verification.get("passed"))
        if verified:
            wtxt = f"（提示：{'; '.join(verification['warnings'])}）" if verification.get("warnings") else ""
            print(
                f"\033[32m[校验通过] 拼接数据集读回校验完成（{total_ep} 段 / {global_idx} 帧，已写 save_receipts 回执）{wtxt}\033[0m",
                flush=True,
            )
        else:
            print(
                f"\033[31m[校验失败] 拼接数据集存在异常：{'; '.join(verification.get('errors') or [])}\033[0m",
                flush=True,
            )
    except Exception as vexc:
        print(f"\033[31m[校验失败] 拼接数据集读回校验异常：{vexc}\033[0m", flush=True)
    return {
        "merged": True,
        "verified": verified,
        "verification": verification,
        "episodes": total_ep,
        "frames": global_idx,
        "merged_dir": str(out_root),
        "runs": episodes_rows,
    }

