#!/usr/bin/env python3
"""Zero-ROS pure-SDK DAgger collector aligned with inference-origin's async architecture."""
from __future__ import annotations

import argparse
from collections import deque
import datetime
import os
from pathlib import Path
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sdk_dagger.hardware import MockDaggerRig, SDKDaggerRig
from sdk_dagger.policy import SDKPolicyClient, TemporalSmoothingScheduler
from sdk_dagger.prepare import load_config, prepare_four_arms_current_pose
from sdk_dagger.storage import (
    AsyncVideoExporter, FrameSample, LeRobotV21DatasetWriter,
    delete_episode, merge_round_lerobot_v21, next_episode_idx, save_episode,
)
from sdk_dagger.voice import VoiceAnnouncer


def _sleep_rate(last_t: float, rate_hz: float) -> float:
    period = 1.0 / max(float(rate_hz), 1e-6)
    now = time.monotonic()
    sleep_s = period - (now - last_t)
    if sleep_s > 0:
        time.sleep(sleep_s)
    return time.monotonic()


def _env_fd(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        fd = int(raw)
    except ValueError:
        return None
    return fd if fd >= 0 else None


class SDKDaggerApp:
    """Complete pure-SDK DAgger state machine with decoupled 30Hz obs / 3Hz infer / 30Hz control."""

    def __init__(self, cfg: dict[str, Any], *, dataset_name: str,
                 dataset_dir: str | Path | None = None, mock: bool = False,
                 enable_terminal_input: bool = True,
                 collector_name: str | None = None,
                 on_event=None,
                 voice: VoiceAnnouncer | None = None):
        self.cfg = cfg
        self.mock = bool(mock)
        self.dataset_name = str(dataset_name)
        self.enable_terminal_input = bool(enable_terminal_input)
        self.collector_name: str | None = str(collector_name) if collector_name else None
        self._on_event = on_event
        self._owns_voice = voice is None
        self.voice = voice if voice is not None else VoiceAnnouncer.from_config(cfg, mock=self.mock)
        rec = cfg["recording"]
        base_dir = Path(dataset_dir or rec.get("dataset_dir", str(ROOT / "data")))
        task_name = str(rec.get("task_name", "aloha_mobile_dummy"))
        self.recording_format = str(rec.get("format", "lerobot")).strip().lower()
        self.robot_type = str(rec.get("robot_type", "agilex"))

        self.frame_rate = float(rec.get("frame_rate", 30))
        self.export_video = bool(rec.get("export_video", True))
        self.video_fps = int(rec.get("video_fps", 30))
        self.video_codec = str(rec.get("video_codec", "libx264"))
        self.video_preset = str(rec.get("video_preset", "ultrafast"))
        self.video_crf = int(rec.get("video_crf", 18))
        self.async_video_export = bool(rec.get("async_video_export", True))
        self.video_exporter = AsyncVideoExporter() if (self.export_video and self.async_video_export) else None

        prompt = str(cfg["inference"].get("prompt", "Flatten and fold the cloth."))
        self.auto_merge_round = bool(rec.get("auto_merge_round", True))
        if self.recording_format == "lerobot":
            self.inference_dir = base_dir / f"{self.dataset_name}_inference"
            self.dagger_dir = base_dir / f"{self.dataset_name}_dagger"
            self.merged_dir = base_dir / f"{self.dataset_name}_merged"
            self.inference_writer = LeRobotV21DatasetWriter(
                self.inference_dir, prompt=prompt, fps=self.video_fps, robot_type=self.robot_type,
                video_codec=self.video_codec, video_preset=self.video_preset, video_crf=self.video_crf,
                video_exporter=self.video_exporter,
            )
            self.dagger_writer = LeRobotV21DatasetWriter(
                self.dagger_dir, prompt=prompt, fps=self.video_fps, robot_type=self.robot_type,
                video_codec=self.video_codec, video_preset=self.video_preset, video_crf=self.video_crf,
                video_exporter=self.video_exporter,
            )
        else:
            self.inference_dir = base_dir / f"{self.dataset_name}_inference_hdf5" / task_name
            self.dagger_dir = base_dir / f"{self.dataset_name}_dagger_hdf5" / task_name
            self.inference_dir.mkdir(parents=True, exist_ok=True)
            self.dagger_dir.mkdir(parents=True, exist_ok=True)
            self.merged_dir = None
            self.inference_writer = None
            self.dagger_writer = None
        self.inference_tail_s = float(rec.get("inference_tail_s", 1.0))
        self.dagger_preroll_s = float(rec.get("dagger_preroll_s", 1.0))
        self.save_confirm_timeout_s = float(rec.get("save_confirm_timeout_s", 10.0))
        self.confirm_inference_save = bool(rec.get("confirm_inference_save", True))
        self.confirm_round_save = bool(rec.get("confirm_round_save", True))

        inf = cfg["inference"]
        self.publish_rate = float(inf.get("publish_rate", 30))
        self.observation_rate = float(inf.get("observation_rate", self.publish_rate))
        self.inference_rate = float(inf.get("inference_rate", 3))
        self.scheduler = TemporalSmoothingScheduler(
            chunk_size=int(inf.get("chunk_size", 50)),
            latency_k=int(inf.get("latency_k", 8)),
            min_smooth_steps=int(inf.get("min_smooth_steps", 8)),
            inference_origin_root=cfg["runtime"]["inference_origin_root"],
        )
        self.policy = SDKPolicyClient(cfg, mock=self.mock)
        self.rig = MockDaggerRig(cfg) if self.mock else SDKDaggerRig(cfg)

        # Runtime state
        self._lock = threading.RLock()
        self._samples_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._input_stop_event = threading.Event()
        self._signal_count = 0
        self._received_signal: int | None = None

        # Modes: INFERENCE | INFERENCE_TAIL | TRANSITION_TO_DAGGER | DAGGER_RECORDING | DAGGER_PAUSED | CONFIRM_SAVE
        self.mode = "INFERENCE"
        self._pending_enter_dagger = False
        self._pending_save_dagger = False
        self._pending_resume_inference = False
        self._pending_new_segment = False
        self._confirm_event = threading.Event()
        self._confirm_delete = False
        self._confirm_target: str | None = None  # "inference" | "dagger" | "round" | None
        self._round_discarded: bool = False
        self._skip_shutdown_confirm: bool = False

        self._latest_sample: FrameSample | None = None
        self._latest_sample_lock = threading.Lock()
        self._obs_error: str | None = None
        self._obs_error_since: float | None = None
        self._obs_error_last_emit: float = 0.0
        self._obs_error_announced: bool = False
        self._inference_samples: list[FrameSample] = []
        self._dagger_samples: list[FrameSample] = []
        self._preroll_ring: deque[FrameSample] = deque(maxlen=120)
        self._dagger_episode_idx = self.dagger_writer.next_episode_index if self.dagger_writer is not None else next_episode_idx(self.dagger_dir)
        self._last_saved_inference_episode: int | None = None
        self._pending_collision_avoidance: dict[str, Any] | None = None
        self._round_stitch_events: list[dict[str, Any]] = []
        self._inference_step = 0
        self._started_at: float | None = None
        self._ready: bool = False
        self._in_exit_handoff: bool = False
        self._last_infer_ms: float | None = None
        self._confirm_deadline: float | None = None
        self._status_message: str = "正在准备本轮采集……"
        self._event_log: deque[tuple[str, str, str]] = deque(maxlen=200)
        self._obs_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None
        self._input_thread: threading.Thread | None = None

    @property
    def round_discarded(self) -> bool:
        with self._lock:
            return self._round_discarded

    def set_collector_name(self, name: str | None) -> None:
        with self._lock:
            self.collector_name = str(name) if name else None

    def _emit_event(self, text: str, level: str = "info") -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with self._lock:
            self._status_message = text
            self._event_log.append((ts, level, text))
        if self._on_event is not None:
            try:
                self._on_event(level, text)
            except Exception:
                pass

    def get_ui_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._latest_sample_lock:
            sample = self._latest_sample
        with self._samples_lock:
            cur_inf = len(self._inference_samples)
            cur_dag = len(self._dagger_samples)
        with self._lock:
            mode = self.mode
            ready = self._ready
            in_exit_handoff = self._in_exit_handoff
            status_message = self._status_message
            started_at = self._started_at
            last_infer_ms = self._last_infer_ms
            confirm_deadline = self._confirm_deadline
            confirm_target = self._confirm_target
            collector_name = self.collector_name
            inf_step = self._inference_step
            recent_events = list(self._event_log)[-50:]

        confirm_rem = max(0.0, confirm_deadline - now) if (mode == "CONFIRM_SAVE" and confirm_deadline is not None) else 0.0
        inf_eps = self.inference_writer.episode_count if self.inference_writer is not None else next_episode_idx(self.inference_dir)
        inf_frames = self.inference_writer.total_frames if self.inference_writer is not None else 0
        inf_deleted = self.inference_writer.deleted_attempts_count if self.inference_writer is not None else 0
        dag_eps = self.dagger_writer.episode_count if self.dagger_writer is not None else self._dagger_episode_idx
        dag_frames = self.dagger_writer.total_frames if self.dagger_writer is not None else 0
        dag_deleted = self.dagger_writer.deleted_attempts_count if self.dagger_writer is not None else 0
        pending_vid = self.video_exporter.pending_count if self.video_exporter is not None else 0

        return {
            "mode": mode,
            "ready": ready,
            "stopping": self._stop_event.is_set(),
            "in_exit_handoff": in_exit_handoff,
            "status_message": status_message,
            "dataset_name": self.dataset_name,
            "collector_name": collector_name,
            "prompt": str(self.cfg["inference"].get("prompt", "")),
            "model_service": str(self.cfg.get("startup", {}).get("model_service", "")),
            "server_addr": f"{self.cfg['server']['host']}:{self.cfg['server']['port']}",
            "elapsed_s": (now - started_at) if started_at is not None else 0.0,
            "inference_step": inf_step,
            "last_infer_ms": last_infer_ms,
            "confirm_remaining_s": confirm_rem,
            "confirm_target": confirm_target,
            "round_discarded": self._round_discarded,
            "current_inference_frames": cur_inf,
            "current_dagger_frames": cur_dag,
            "inference_episodes": inf_eps,
            "inference_saved_frames": inf_frames,
            "inference_deleted_attempts": inf_deleted,
            "dagger_episodes": dag_eps,
            "dagger_saved_frames": dag_frames,
            "dagger_deleted_attempts": dag_deleted,
            "pending_video_jobs": pending_vid,
            "sample": sample,
            "sample_age_s": (sample_age := (max(0.0, time.time() - sample.timestamp) if sample is not None else None)),
            "camera_health": (cam_health := (self.rig.get_camera_health() if hasattr(self.rig, "get_camera_health") else {})),
            "faulty_cameras": [k for k, v in cam_health.items() if v.get("stale")],
            "obs_error": self._obs_error,
            "obs_error_s": (max(0.0, now - self._obs_error_since) if self._obs_error_since is not None else 0.0),
            "obs_stale": bool(
                any(v.get("stale") for v in cam_health.values())
                or (sample_age is not None and sample_age > 2.0)
                or (self._obs_error is not None)
            ),
            "inference_dir": str(self.inference_dir),
            "dagger_dir": str(self.dagger_dir),
            "recent_events": recent_events,
        }

    def request_stop(self, signum: int | None = None, _frame=None, *, skip_confirm: bool = False) -> None:
        self._signal_count += 1
        if skip_confirm:
            self._skip_shutdown_confirm = True
        if self._received_signal is None and signum is not None:
            self._received_signal = signum
        self._stop_event.set()
        self._confirm_event.set()
        if self._signal_count > 1:
            self._input_stop_event.set()
            print("\n[强制退出] 收到第二次中断信号，立即退出（未写完的文件可能不完整）。", flush=True)
            raise KeyboardInterrupt

    def _signal_pedal_ready(self) -> None:
        ready_fd = _env_fd("DAGGER_PEDAL_READY_FD")
        if ready_fd is None:
            return
        try:
            os.write(ready_fd, b"ready\n")
        except OSError:
            pass
        finally:
            try:
                os.close(ready_fd)
            except OSError:
                pass

    def handle_operator_action(self, action: str, *, source: str = "keyboard") -> None:
        with self._lock:
            if self.mode == "CONFIRM_SAVE":
                target_zh = "整轮数据" if self._confirm_target == "round" else "本段数据"
                if action in ("w", "delete"):
                    self._confirm_delete = True
                    print(f"\n已请求删除，正在删除{target_zh}……\n", flush=True)
                else:
                    self._confirm_delete = False
                    print(f"\n已确认保留{target_zh}。\n", flush=True)
                self._confirm_event.set()
                return

            if action == "delete":
                return

            if action in ("d", "left"):
                if self.mode == "INFERENCE":
                    self._pending_enter_dagger = True
                    print("\n" + "🎯" * 20, flush=True)
                    print("🎯 已触发人工接管（DAgger）！正在暂停模型推理……", flush=True)
                    print("🎯" * 20 + "\n", flush=True)
                elif self.mode == "DAGGER_PAUSED":
                    self._pending_new_segment = True
                    print("\n已请求开始录制新一段人工纠正数据……\n", flush=True)
                elif self.mode == "DAGGER_RECORDING":
                    print("\n当前已在人工纠正录制中，无需重复触发。\n", flush=True)
                else:
                    print("\n正在模式切换中，请稍候……\n", flush=True)
                return

            if action == " ":
                if self.mode == "DAGGER_PAUSED":
                    self._pending_new_segment = True
                    print("\n已请求开始录制新一段人工纠正数据……\n", flush=True)
                elif self.mode == "DAGGER_RECORDING":
                    print("\n当前已在人工纠正录制中，已有数据保持不变。\n", flush=True)
                else:
                    print("\n请先踩「左单踩」或按 d 进入人工示教模式。\n", flush=True)
                return

            if action in ("s", "save"):
                if self.mode == "DAGGER_RECORDING":
                    self._pending_save_dagger = True
                    print("\n" + "💾" * 20, flush=True)
                    print("💾 已收到保存请求，正在暂停跟随并保存本段数据……", flush=True)
                    print("💾" * 20 + "\n", flush=True)
                elif self.mode == "DAGGER_PAUSED":
                    print("\n当前没有未保存的纠正数据。\n", flush=True)
                else:
                    print("\n⚠️ 当前不在人工示教模式，无法保存纠正段\n", flush=True)
                return

            if action in ("r", "resume"):
                if self.mode in ("DAGGER_RECORDING", "DAGGER_PAUSED"):
                    self._pending_resume_inference = True
                    print("\n" + "▶️" * 20, flush=True)
                    print("▶️ 已请求恢复模型推理，正在等待双主臂交接确认……", flush=True)
                    print("▶️" * 20 + "\n", flush=True)
                elif self.mode == "INFERENCE":
                    print("\n⚠️ 当前已处于模型自动运行模式\n", flush=True)
                else:
                    print("\n正在模式切换中，请稍候……\n", flush=True)

    def _input_monitor_loop(self) -> None:
        cmd_fd = _env_fd("DAGGER_PEDAL_COMMAND_FD")
        stdin_fd = sys.stdin.fileno() if sys.stdin and sys.stdin.isatty() else None
        old_term = None
        if stdin_fd is not None:
            try:
                old_term = termios.tcgetattr(stdin_fd)
                tty.setcbreak(stdin_fd)
            except termios.error:
                old_term = None
        pedal_buf = b""
        token_map = {b"left": "left", b"save": "save", b"resume": "resume", b"delete": "delete"}
        try:
            while not self._input_stop_event.is_set():
                watch = [fd for fd in (stdin_fd, cmd_fd) if fd is not None]
                if not watch:
                    time.sleep(0.05)
                    continue
                try:
                    readable, _, _ = select.select(watch, [], [], 0.05)
                except (OSError, ValueError):
                    break
                for fd in readable:
                    if fd == stdin_fd:
                        try:
                            ch = os.read(stdin_fd, 1)
                        except OSError:
                            stdin_fd = None
                            continue
                        if not ch:
                            stdin_fd = None
                            continue
                        if ch == b"\x03":
                            self.request_stop(signal.SIGINT)
                            break
                        text = ch.decode("utf-8", errors="ignore").lower()
                        if text in ("d", "s", "r", "w", " "):
                            self.handle_operator_action(text, source="keyboard")
                    elif fd == cmd_fd:
                        try:
                            chunk = os.read(cmd_fd, 4096)
                        except OSError:
                            cmd_fd = None
                            continue
                        if not chunk:
                            cmd_fd = None
                            continue
                        pedal_buf += chunk
                        while b"\n" in pedal_buf:
                            line, pedal_buf = pedal_buf.split(b"\n", 1)
                            tok = token_map.get(line.strip())
                            if tok:
                                self.handle_operator_action(tok, source="pedal")
        finally:
            if old_term is not None and stdin_fd is not None:
                try:
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_term)
                except termios.error:
                    pass
            if cmd_fd is not None:
                try:
                    os.close(cmd_fd)
                except OSError:
                    pass

    def _observation_worker_loop(self) -> None:
        """Single owner of rig.read_sample() at observation_rate (decoupled from 30Hz control loop)."""
        last_t = time.monotonic()
        while not self._stop_event.is_set():
            try:
                sample = self.rig.read_sample()
            except Exception as exc:
                if not self._stop_event.is_set():
                    now_m = time.monotonic()
                    err_s = str(exc)
                    print(f"⚠️ 观测采集暂时异常：{err_s}", flush=True)
                    with self._lock:
                        if self._obs_error_since is None:
                            self._obs_error_since = now_m
                        self._obs_error = err_s
                        should_emit = (now_m - self._obs_error_last_emit) >= 8.0
                        if should_emit:
                            self._obs_error_last_emit = now_m
                        should_announce = not self._obs_error_announced
                        if should_announce:
                            self._obs_error_announced = True
                    if should_emit:
                        self._emit_event(
                            f"❌ 相机/观测采帧异常（当前已停止录帧，请检查相机 USB 线）：{err_s}",
                            level="error",
                        )
                    if should_announce:
                        self.voice.announce("camera_error")
                time.sleep(0.05)
                continue

            recovered = False
            with self._lock:
                if self._obs_error is not None or self._obs_error_since is not None:
                    self._obs_error = None
                    self._obs_error_since = None
                    self._obs_error_announced = False
                    recovered = True
            if recovered:
                self._emit_event("✅ 相机/观测采帧已恢复正常")

            with self._latest_sample_lock:
                self._latest_sample = sample

            with self._lock:
                cur_mode = self.mode

            with self._samples_lock:
                if cur_mode in ("INFERENCE", "INFERENCE_TAIL"):
                    self._inference_samples.append(sample)
                elif cur_mode == "TRANSITION_TO_DAGGER":
                    self._preroll_ring.append(sample)
                elif cur_mode == "DAGGER_RECORDING":
                    self._dagger_samples.append(sample)
                    if len(self._dagger_samples) % 30 == 0:
                        print(f"人工纠正录制中（已录 {len(self._dagger_samples)} 帧）| 右单踩 / s 保存本段", flush=True)

            last_t = _sleep_rate(last_t, self.observation_rate)

    def _inference_worker_loop(self) -> None:
        last_t = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                in_inference = (self.mode == "INFERENCE" and not self._pending_enter_dagger)
            if not in_inference:
                time.sleep(0.01)
                last_t = time.monotonic()
                continue
            with self._latest_sample_lock:
                sample = self._latest_sample
            if sample is None:
                time.sleep(0.005)
                continue
            gen = self.scheduler.generation
            try:
                actions, elapsed_ms = self.policy.infer(sample.images_bgr, sample.qpos)
                with self._lock:
                    self._last_infer_ms = float(elapsed_ms)
                self.scheduler.update(actions, generation=gen)
            except Exception as exc:
                print(f"⚠️ 模型推理暂不可用：{exc}", flush=True)
                time.sleep(0.1)
                continue
            last_t = _sleep_rate(last_t, self.inference_rate)

    def _save_inference_segment(self, *, capture_tail: bool, wait_for_confirm: bool | None = None) -> dict[str, Any]:
        if capture_tail and self.inference_tail_s > 0 and not self._stop_event.is_set():
            with self._lock:
                self.mode = "INFERENCE_TAIL"
            self._stop_event.wait(self.inference_tail_s)

        with self._lock:
            prev_mode = self.mode
            if self.mode == "INFERENCE_TAIL":
                self.mode = "TRANSITION_TO_DAGGER"
                prev_mode = "TRANSITION_TO_DAGGER"

        with self._samples_lock:
            samples = list(self._inference_samples)
            self._inference_samples.clear()

        if len(samples) < 2:
            print("没有未保存的模型推理帧。", flush=True)
            return {"saved": False, "frames": 0}

        if self.inference_writer is not None:
            ep_idx = self.inference_writer.next_episode_index
            print(f"正在保存模型推理段（LeRobot v2.1，episode_{ep_idx:06d}，共 {len(samples) - 1} 帧）……", flush=True)
            res = self.inference_writer.save_episode(
                samples, export_video=self.export_video, collector_name=self.collector_name
            )
        else:
            ep_idx = next_episode_idx(self.inference_dir)
            print(f"正在保存模型推理段（共 {len(samples) - 1} 帧）-> {self.inference_dir / f'episode_{ep_idx}.hdf5'}", flush=True)
            res = save_episode(
                samples, self.inference_dir, ep_idx,
                export_video=self.export_video, video_fps=self.video_fps,
                video_codec=self.video_codec, video_preset=self.video_preset, video_crf=self.video_crf,
                video_exporter=self.video_exporter,
            )
        if not res["saved"]:
            return res

        print(f"\033[32m>>> 模型推理段已保存：{res['data_path']}（共 {res['frames']} 帧）\033[0m", flush=True)

        do_confirm = self.confirm_inference_save if wait_for_confirm is None else bool(wait_for_confirm)
        should_delete = False
        if do_confirm and self.save_confirm_timeout_s > 0 and self._signal_count <= 1:
            print("\n" + "=" * 70, flush=True)
            print("请确认刚刚保存的模型推理段是否正确：", flush=True)
            print(f"  - 如果推理跑坏了，请在 {self.save_confirm_timeout_s:.0f} 秒内按键盘 w 丢弃本段；", flush=True)
            print("  - 如果正确，等待超时或踩任意踏板/按其他键立即确认保留。", flush=True)
            print("=" * 70, flush=True)
            print(f"等待确认（{self.save_confirm_timeout_s:.0f} 秒后自动保留）……", flush=True)

            with self._lock:
                self._confirm_delete = False
                self._confirm_event.clear()
                self._confirm_deadline = time.monotonic() + self.save_confirm_timeout_s
                self._confirm_target = "inference"
                self.mode = "CONFIRM_SAVE"
            self._emit_event(
                f"模型推理段 episode_{res['episode_idx']:06d} 已保存（{res['frames']} 帧）：{self.save_confirm_timeout_s:.0f} 秒内按 w 或点「撤销删除」可丢弃",
                level="warn",
            )
            self.voice.announce("saved_confirm")

            self._confirm_event.wait(timeout=self.save_confirm_timeout_s)

            with self._lock:
                should_delete = self._confirm_delete
                self._confirm_deadline = None
                self._confirm_target = None
                self.mode = prev_mode

        if should_delete:
            self._round_stitch_events.append({
                "kind": "break",
                "reason": "inference_deleted",
                "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "attempt_index": res.get("attempt_idx"),
            })
            if self.inference_writer is not None:
                self.inference_writer.delete_last_episode(res, reason="operator_w_delete_inference")
                print(
                    f"\n\033[32m本段模型推理已丢弃（编号保持连续；缺失记录 attempt={res.get('attempt_idx')} 已写入 meta/session_timeline.jsonl）。\033[0m",
                    flush=True,
                )
                self._emit_event(
                    f"已丢弃模型推理段 attempt={res.get('attempt_idx')}（编号保持连续，缺口已记入 session_timeline.jsonl）",
                    level="warn",
                )
                self.voice.announce("deleted")
            else:
                delete_episode(res["hdf5_path"], res["video_paths"], video_job=res.get("video_job"))
                print("\n\033[32m本段模型推理已丢弃。\033[0m", flush=True)
            res["deleted"] = True
            return res

        res["deleted"] = False
        self._last_saved_inference_episode = res.get("episode_idx")
        self._round_stitch_events.append({
            "kind": "segment",
            "source": "inference",
            "episode_index": int(res["episode_idx"]),
            "frames": int(res["frames"]),
        })
        self._emit_event(f"模型推理段 episode_{res['episode_idx']:06d} 已确认保留（{res['frames']} 帧）")
        if do_confirm and self.save_confirm_timeout_s > 0:
            self.voice.announce("confirmed")
        return res

    def _save_dagger_segment(self, *, wait_for_confirm: bool = True) -> dict[str, Any]:
        self.rig.set_following(False)
        with self._lock:
            self.mode = "DAGGER_PAUSED"
        with self._samples_lock:
            samples = list(self._dagger_samples)
            self._dagger_samples.clear()
        if len(samples) < 2:
            with self._latest_sample_lock:
                ls = self._latest_sample
            ls_age = (time.time() - ls.timestamp) if ls is not None else None
            obs_err = self._obs_error
            if obs_err or (ls_age is not None and ls_age > 2.0):
                reason = f"相机已停帧 {ls_age:.1f}s" if (ls_age is not None and ls_age > 2.0) else f"相机异常：{obs_err}"
                msg = f"❌ 本段仅录到 {len(samples)} 帧，未保存（{reason}）！不会进入 10 秒删除确认，请检查相机 USB 线并结束本轮重开。"
                self.voice.announce("camera_error")
            else:
                msg = f"⚠️ 本段仅录到 {len(samples)} 帧（不足 2 帧），未保存，未触发 10 秒删除确认。"
            print(f"\033[31m{msg}\033[0m", flush=True)
            self._emit_event(msg, level="error")
            return {"saved": False, "frames": 0}

        if self.dagger_writer is not None:
            ep_idx = self.dagger_writer.next_episode_index
            self._dagger_episode_idx = ep_idx
            print(f"\n正在保存纠正段（LeRobot v2.1）episode_{ep_idx:06d}……（帧数：{len(samples) - 1}）", flush=True)
            res = self.dagger_writer.save_episode(
                samples,
                export_video=self.export_video,
                after_inference_episode=self._last_saved_inference_episode,
                collector_name=self.collector_name,
                collision_avoidance_before=self._pending_collision_avoidance,
            )
        else:
            ep_idx = self._dagger_episode_idx
            print(f"\n正在保存纠正段 episode_{ep_idx}……（帧数：{len(samples) - 1}）", flush=True)
            res = save_episode(
                samples, self.dagger_dir, ep_idx,
                export_video=self.export_video, video_fps=self.video_fps,
                video_codec=self.video_codec, video_preset=self.video_preset, video_crf=self.video_crf,
                video_exporter=self.video_exporter,
            )
        if not res["saved"]:
            return res

        print(f"\n\033[32m纠正段 episode_{ep_idx} 已保存成功。\033[0m", flush=True)
        should_delete = False
        if wait_for_confirm:
            print("\n" + "=" * 70, flush=True)
            print("请确认刚刚保存的这一段是否正确：", flush=True)
            print("  - 如果录坏了，请在 10 秒内按键盘 w 删除本段；", flush=True)
            print("  - 如果正确，等待超时或踩任意踏板/按其他键立即确认保留。", flush=True)
            print("=" * 70, flush=True)
            print(f"等待确认（{self.save_confirm_timeout_s:.0f} 秒后自动保留）……", flush=True)

            with self._lock:
                self._confirm_delete = False
                self._confirm_event.clear()
                self._confirm_deadline = time.monotonic() + self.save_confirm_timeout_s
                self._confirm_target = "dagger"
                self.mode = "CONFIRM_SAVE"
            self._emit_event(
                f"纠正段 episode_{ep_idx:06d} 已保存（{res['frames']} 帧）：{self.save_confirm_timeout_s:.0f} 秒内按 w 或点「撤销删除」可删除",
                level="warn",
            )
            self.voice.announce("saved_confirm")

            self._confirm_event.wait(timeout=self.save_confirm_timeout_s)

            with self._lock:
                should_delete = self._confirm_delete
                self._confirm_deadline = None
                self._confirm_target = None

        if should_delete:
            self._round_stitch_events.append({
                "kind": "break",
                "reason": "dagger_deleted",
                "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                "attempt_index": res.get("attempt_idx"),
                "after_inference_episode": self._last_saved_inference_episode,
            })
            if self.dagger_writer is not None:
                self.dagger_writer.delete_last_episode(res)
                self._dagger_episode_idx = self.dagger_writer.next_episode_index
                print(
                    f"\n\033[32m本段数据已删除（编号保持连续；缺失记录 attempt={res.get('attempt_idx')} 已写入 meta/session_timeline.jsonl）。\033[0m",
                    flush=True,
                )
                self._emit_event(
                    f"已撤销删除纠正段 attempt={res.get('attempt_idx')}（编号保持连续，缺口已记入 session_timeline.jsonl）",
                    level="warn",
                )
                self.voice.announce("deleted")
            else:
                delete_episode(res["hdf5_path"], res["video_paths"], video_job=res.get("video_job"))
                print("\n\033[32m本段数据已删除。\033[0m", flush=True)
            res["deleted"] = True
        else:
            self._dagger_episode_idx += 1
            self._pending_collision_avoidance = None
            res["deleted"] = False
            self._round_stitch_events.append({
                "kind": "segment",
                "source": "dagger",
                "episode_index": int(res["episode_idx"]),
                "frames": int(res["frames"]),
            })
            self._emit_event(f"已确认保留纠正段 episode_{ep_idx:06d}（{res['frames']} 帧）")
            if wait_for_confirm:
                self.voice.announce("confirmed")

        with self._lock:
            self.mode = "DAGGER_PAUSED"

        if wait_for_confirm:
            print("\n" + "=" * 70, flush=True)
            print("保存完成，本段录制已结束。", flush=True)
            print("从臂跟随保持暂停状态（移动主臂不会带动从臂）。", flush=True)
            print("踩「左双踩」或按 r：退出示教，恢复模型自动运行。", flush=True)
            print("踩「左单踩」或按空格：恢复主臂跟随，开始录制下一段人工纠正。", flush=True)
            print("=" * 70 + "\n", flush=True)
        else:
            print("\n自动保存完成，即将恢复模型自动运行。\n", flush=True)
        return res

    def _do_enter_dagger(self) -> None:
        with self._lock:
            self._pending_enter_dagger = False
        self.scheduler.reset()
        self.rig.hold_followers()

        print("\n" + "=" * 70, flush=True)
        print("⏸️  模型已暂停 — 正在保存本段推理并进入人工示教模式", flush=True)
        print("=" * 70 + "\n", flush=True)

        self._emit_event("已暂停模型推理，正在保存本段推理并进入双主臂对齐……", level="warn")
        self.voice.announce("enter_dagger")
        self._save_inference_segment(capture_tail=True)
        if self._stop_event.is_set():
            return

        with self._samples_lock:
            self._preroll_ring.clear()
        with self._lock:
            self.mode = "TRANSITION_TO_DAGGER"
        self._emit_event("正在执行双主臂对齐，请勿触碰机械臂，等待示教就绪……", level="warn")

        ok = self.rig.enter_teaching_mode(shutdown_fn=self._stop_event.is_set)
        if not ok or self._stop_event.is_set():
            with self._samples_lock:
                self._preroll_ring.clear()
            if not self._stop_event.is_set():
                print("\n❌ 进入人工示教失败，保持暂停状态。可重新尝试或结束本轮。\n", flush=True)
                with self._lock:
                    self.mode = "INFERENCE"
            return

        with self._samples_lock:
            ring = list(self._preroll_ring)
            self._preroll_ring.clear()
            if self.dagger_preroll_s > 0 and ring:
                cutoff = ring[-1].timestamp - self.dagger_preroll_s
                self._dagger_samples = [s for s in ring if s.timestamp >= cutoff]
            else:
                self._dagger_samples = []
        ca = getattr(self.rig, "last_collision_avoidance", None)
        if ca is not None:
            self._pending_collision_avoidance = ca
            self._round_stitch_events.append({
                "kind": "break",
                "reason": "collision_avoidance",
                "timestamp": ca.get("timestamp"),
                "after_inference_episode": self._last_saved_inference_episode,
                "sides": ca.get("sides"),
            })
            if self.dagger_writer is not None:
                self.dagger_writer.record_collision_avoidance(
                    ca, after_inference_episode=self._last_saved_inference_episode
                )
            side_desc = "、".join(
                f"{'左' if k == 'left' else '右'}臂微调{v.get('max_delta_deg', 0):.1f}°"
                for k, v in (ca.get("sides") or {}).items()
            )
            self._emit_event(f"🛡️ 手柄防碰撞已介入（{side_desc}），已记录至 session_timeline.jsonl", level="warn")

        with self._lock:
            self.mode = "DAGGER_RECORDING"
        self._emit_event(f"人工示教已就绪并自动录制中（episode_{self._dagger_episode_idx:06d}），完成后踩右单踩或点保存")
        self.voice.announce("teaching_ready")

        print("\n" + "=" * 70, flush=True)
        print("✅ 已进入人工示教模式（纯 SDK 直通转发已就绪）", flush=True)
        print("🎮 现在可以拖动主臂控制从臂进行纠正", flush=True)
        print(f"本轮数据集：{self.dataset_name}（当前纠正编号：episode_{self._dagger_episode_idx}）", flush=True)
        print("人工纠正已自动开始录制（无需再按空格或踩踏板）", flush=True)
        print("纠正完成后：踩「右单踩」或按 s 保存本段", flush=True)
        print("保存并确认后：踩「左双踩」或按 r 恢复模型自动运行", flush=True)
        print("=" * 70 + "\n", flush=True)

    def _do_resume_inference(self) -> None:
        with self._lock:
            self._pending_resume_inference = False
        with self._lock:
            self._in_exit_handoff = True
        self._emit_event("正在退出示教：请将双主臂移回安全姿态并保持静止 3 秒……", level="warn")
        self.voice.announce("exit_handoff")
        try:
            self.rig.exit_teaching_mode(shutdown_fn=self._stop_event.is_set)
        except Exception as exc:
            with self._lock:
                self._in_exit_handoff = False
            print(f"\n双主臂交接未完成，已保持人工示教与模型暂停状态：{exc}", flush=True)
            print("已有纠正数据均已保留。请按 r 或左双踩重试，或按 Ctrl+C / 右双踩结束本轮。\n", flush=True)
            self._emit_event(f"双主臂交接未完成，已保持示教暂停：{exc}", level="error")
            self.voice.announce("handoff_failed")
            return
        with self._lock:
            self._in_exit_handoff = False

        with self._samples_lock:
            has_unsaved = len(self._dagger_samples) >= 2
        if has_unsaved:
            print("\n💾 检测到未保存的人工纠正数据，正在自动保存后再恢复模型……", flush=True)
            self._save_dagger_segment(wait_for_confirm=False)

        self._pending_collision_avoidance = None
        self.scheduler.reset()
        with self._samples_lock:
            self._inference_samples.clear()
        with self._lock:
            self.mode = "INFERENCE"
        self._emit_event("已退出人工示教，恢复模型自动运行")
        self.voice.announce("resumed")
        print("\n" + "=" * 70, flush=True)
        print("▶️  已退出人工示教 — 恢复模型自动运行", flush=True)
        print("=" * 70 + "\n", flush=True)

    def _do_start_new_segment(self) -> None:
        with self._samples_lock:
            self._dagger_samples.clear()
        with self._lock:
            self._pending_new_segment = False
            self.mode = "DAGGER_RECORDING"
        self.rig.set_following(True)
        self._emit_event(f"已开始录制新一段人工纠正（episode_{self._dagger_episode_idx:06d}）")
        self.voice.announce("new_segment")
        print("\n" + "=" * 70, flush=True)
        print(f"已开始录制新一段人工纠正：episode_{self._dagger_episode_idx}", flush=True)
        print("录制完成后请踩「右单踩」或按 s 保存本段数据", flush=True)
        print("=" * 70 + "\n", flush=True)

    def _confirm_and_maybe_discard_round(self) -> bool:
        """Ask operator whether to keep or discard the entire round before final merge/exit."""
        if (
            not self.confirm_round_save
            or self._skip_shutdown_confirm
            or self.save_confirm_timeout_s <= 0
            or self._signal_count > 1
        ):
            return False

        if self.inference_writer is not None and self.dagger_writer is not None:
            inf_eps = self.inference_writer.episode_count
            dag_eps = self.dagger_writer.episode_count
        else:
            inf_eps = next_episode_idx(self.inference_dir)
            dag_eps = next_episode_idx(self.dagger_dir)
        if inf_eps + dag_eps <= 0:
            return False

        print("\n" + "=" * 70, flush=True)
        print(f"请确认是否保留本轮全部数据（模型推理 {inf_eps} 段 / 人工纠正 {dag_eps} 段）：", flush=True)
        print(f"  - 如果整轮数据不要，请在 {self.save_confirm_timeout_s:.0f} 秒内按键盘 w 丢弃整轮；", flush=True)
        print("  - 如果保留本轮，等待超时或踩任意踏板/按其他键立即确认保留。", flush=True)
        print("=" * 70, flush=True)
        print(f"等待整轮确认（{self.save_confirm_timeout_s:.0f} 秒后自动保留）……", flush=True)

        with self._lock:
            prev_mode = self.mode
            self._confirm_delete = False
            self._confirm_event.clear()
            self._confirm_deadline = time.monotonic() + self.save_confirm_timeout_s
            self._confirm_target = "round"
            self.mode = "CONFIRM_SAVE"
        self._emit_event(
            f"整轮数据待确认（推理 {inf_eps} 段 / 纠正 {dag_eps} 段）：{self.save_confirm_timeout_s:.0f} 秒内按 w 或点「丢弃整轮数据」删除整轮，其他键/踏板保留",
            level="warn",
        )
        self.voice.announce("saved_confirm")

        self._confirm_event.wait(timeout=self.save_confirm_timeout_s)

        with self._lock:
            should_delete = self._confirm_delete
            self._confirm_deadline = None
            self._confirm_target = None
            self.mode = prev_mode

        if not should_delete:
            self._emit_event(f"已确认保留本轮全部数据（推理 {inf_eps} 段 / 纠正 {dag_eps} 段），正在完成收尾落盘……")
            self.voice.announce("confirmed")
            return False

        if self.video_exporter is not None:
            self.video_exporter.cancel_all()

        dirs_to_remove: list[Path] = []
        if self.recording_format == "lerobot":
            dirs_to_remove = [d for d in (self.inference_dir, self.dagger_dir, self.merged_dir) if d is not None]
        else:
            dirs_to_remove = [self.inference_dir.parent, self.dagger_dir.parent]
        removed_names: list[str] = []
        for d in dirs_to_remove:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                removed_names.append(d.name)

        self._round_stitch_events.clear()
        with self._lock:
            self._round_discarded = True
        print(
            f"\n\033[33m[整轮丢弃] 已删除本轮全部数据目录：{', '.join(removed_names) or '(无目录)'}\033[0m",
            flush=True,
        )
        self._emit_event(f"已丢弃本轮全部数据（已清理：{', '.join(removed_names) or '无'}）", level="warn")
        self.voice.announce("deleted")
        return True

    def run(self) -> int:
        in_main = threading.current_thread() is threading.main_thread()
        old_sigint = signal.signal(signal.SIGINT, self.request_stop) if in_main else None
        old_sigterm = signal.signal(signal.SIGTERM, self.request_stop) if in_main else None
        try:
            self._started_at = time.monotonic()
            self._emit_event(f"正在初始化本轮数据集：{self.dataset_name}（四臂准备与归位中）……")
            self.voice.announce("round_starting")
            print(f"[新一轮][纯 SDK] 数据集：{self.dataset_name}", flush=True)
            if not self.mock and self.cfg.get("startup", {}).get("prepare_arms_current_pose", True):
                prepare_four_arms_current_pose(self.cfg)
            self.rig.start()
            self.rig.move_followers_to_init()

            # Take initial timestamp-synced observation & warmup one policy step
            first = self.rig.read_sample()
            with self._latest_sample_lock:
                self._latest_sample = first
            actions, ms = self.policy.infer(first.images_bgr, first.qpos)
            with self._lock:
                self._last_infer_ms = float(ms)
                self._ready = True
            self.scheduler.update(actions, generation=self.scheduler.generation)
            print(f"[就绪] 模型首次推理成功（耗时 {ms:.1f} ms），开始执行模型动作。", flush=True)
            self._emit_event(f"模型首次推理成功（{ms:.1f} ms），自动运行并录制中")
            self.voice.announce("inference_running")

            self._obs_thread = threading.Thread(target=self._observation_worker_loop, daemon=True)
            self._obs_thread.start()
            if self.enable_terminal_input:
                self._input_thread = threading.Thread(target=self._input_monitor_loop, daemon=True)
                self._input_thread.start()
            self._infer_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
            self._infer_thread.start()

            self._signal_pedal_ready()
            print("\033[36m>>> 模型推理已开始录制。需要人工纠正时踩「左单踩」或按 d。\033[0m", flush=True)

            # Dedicated jitter-free 30Hz control loop (never blocked by camera reads)
            last_t = time.monotonic()
            while not self._stop_event.is_set():
                with self._lock:
                    do_enter = self._pending_enter_dagger
                    do_save = self._pending_save_dagger
                    do_resume = self._pending_resume_inference
                    do_new = self._pending_new_segment
                    cur_mode = self.mode

                if do_enter:
                    self._do_enter_dagger()
                    last_t = time.monotonic()
                    continue
                if do_save:
                    with self._lock:
                        self._pending_save_dagger = False
                    self._save_dagger_segment()
                    last_t = time.monotonic()
                    continue
                if do_new:
                    self._do_start_new_segment()
                    last_t = time.monotonic()
                    continue
                if do_resume:
                    self._do_resume_inference()
                    last_t = time.monotonic()
                    continue

                if cur_mode == "INFERENCE":
                    step = self.scheduler.pop_step()
                    if step is None:
                        time.sleep(0.001)
                        continue
                    self.rig.apply_policy_action(step["action"], action_step=step)
                    self._inference_step += 1
                    if self._inference_step % 30 == 0:
                        print(
                            f"模型运行中（第 {self._inference_step} 步）| 左单踩 / d 人工接管，右双踩结束本轮，Ctrl+C 退出",
                            flush=True,
                        )
                    last_t = _sleep_rate(last_t, self.publish_rate)
                else:
                    self._stop_event.wait(0.01)
                    last_t = time.monotonic()
        except KeyboardInterrupt:
            if self._received_signal is None:
                self._received_signal = signal.SIGINT
        finally:
            self._stop_event.set()
            try:
                self.rig.hold_followers()
            except Exception:
                pass
            print("\n[停止] 正在停止机械臂控制与转发（不失能、不归位）……", flush=True)
            self.voice.announce("stopping")
            if self._obs_thread is not None and self._obs_thread.is_alive():
                self._obs_thread.join(timeout=1.5)
            self.rig.close()
            with self._samples_lock:
                unsaved_inf = len(self._inference_samples)
                unsaved_dag = len(self._dagger_samples)
            if self._signal_count <= 1:
                if unsaved_inf >= 2:
                    print("[保存] 正在自动保存尚未保存的模型推理段……", flush=True)
                    self._save_inference_segment(
                        capture_tail=False,
                        wait_for_confirm=(False if self._skip_shutdown_confirm else None),
                    )
                if unsaved_dag >= 2:
                    print("[保存] 正在自动保存尚未保存的人工纠正段……", flush=True)
                    self._save_dagger_segment(wait_for_confirm=(not self._skip_shutdown_confirm))
                self._confirm_and_maybe_discard_round()
            self._input_stop_event.set()
            if self._input_thread is not None and self._input_thread.is_alive():
                self._input_thread.join(timeout=0.5)
            if self.video_exporter is not None:
                self.video_exporter.close(wait=(self._signal_count <= 1 and not self._round_discarded))
            if (
                self._signal_count <= 1
                and not self._round_discarded
                and self.recording_format == "lerobot"
                and self.auto_merge_round
                and self.merged_dir is not None
                and self._round_stitch_events
            ):
                try:
                    mres = merge_round_lerobot_v21(
                        inference_dir=self.inference_dir,
                        dagger_dir=self.dagger_dir,
                        merged_dir=self.merged_dir,
                        stitch_events=self._round_stitch_events,
                        prompt=str(self.cfg["inference"].get("prompt", "")),
                        fps=self.video_fps,
                        robot_type=self.robot_type,
                        video_codec=self.video_codec,
                        export_video=self.export_video,
                        collector_name=self.collector_name,
                    )
                    if mres.get("merged"):
                        vtag = "（✅ 读回校验通过）" if mres.get("verified") else "（⚠️ 校验有异常，请查看历史页）"
                        self._emit_event(
                            f"已自动拼接本轮连续 LeRobot 片段：{mres['episodes']} 段 / {mres['frames']} 帧{vtag} -> {self.merged_dir.name}"
                        )
                except Exception as exc:
                    print(f"\033[31m[自动拼接] 拼接连续片段失败（原始 inference/dagger 数据完整不受影响）：{exc}\033[0m", flush=True)
            self.policy.close()
            if self._owns_voice:
                self.voice.close()
            if in_main:
                signal.signal(signal.SIGINT, old_sigint)
                signal.signal(signal.SIGTERM, old_sigterm)
            self._emit_event("本轮已安全停止，全部数据与后台视频已落盘")
            print("[本轮结束] 纯 SDK 采集器已安全停止。", flush=True)

        if self._received_signal is not None:
            return 128 + self._received_signal
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/dagger_precision_5090.yaml"))
    parser.add_argument("--dataset-name", dest="dataset_name", default=None)
    parser.add_argument("--dataset-dir", dest="dataset_dir", default=None)
    parser.add_argument("--mock", action="store_true", help="无硬件全链路模拟运行")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = str((cfg.get("recording") or {}).get("dataset_prefix", "cloth_sdk"))
    dataset_name = args.dataset_name or os.environ.get("DATASET_NAME") or f"{prefix}_{ts}_{os.getpid()}"
    dataset_dir = args.dataset_dir or os.environ.get("DATASET_DIR") or None
    app = SDKDaggerApp(cfg, dataset_name=dataset_name, dataset_dir=dataset_dir, mock=args.mock)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
