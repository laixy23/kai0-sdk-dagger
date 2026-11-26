"""Non-blocking Mandarin Chinese voice announcer for Kai0 pure-SDK DAgger (CLI & GUI)."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICE_DIR = ROOT / "assets" / "voice"

VOICE_PROMPTS: dict[str, str] = {
    "round_starting": "新一轮开始，正在归位",
    "inference_running": "模型开始运行",
    "enter_dagger": "人工接管，请勿触碰，主臂对齐中",
    "teaching_ready": "示教已就绪，开始录制",
    "new_segment": "开始录制下一段纠正",
    "saved_confirm": "本段已保存，十秒内按 W 可删除",
    "deleted": "本段已删除",
    "confirmed": "已确认保留",
    "exit_handoff": "请将双主臂移回安全位置，保持三秒",
    "handoff_failed": "双主臂交接未完成，请重试",
    "resumed": "已恢复模型运行",
    "stopping": "正在停止并保存本轮数据",
    "standby": "本轮已保存完成，进入待机",
    "camera_error": "警告，相机画面中断，已停止录帧，请检查接线",
    "voice_test": "中文语音播报正常",
}


def _pipewire_env() -> dict[str, str]:
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR"):
        runtime_dir = f"/run/user/{os.getuid()}"
        if os.path.isdir(runtime_dir):
            env["XDG_RUNTIME_DIR"] = runtime_dir
    return env


def find_pipewire_sink_id(match_substring: str, *, env: dict[str, str] | None = None) -> str | None:
    """Resolve WirePlumber/PipeWire sink ID matching a substring (e.g., 'SPEAKER-Z')."""
    if not match_substring or not shutil.which("wpctl"):
        return None
    try:
        out = subprocess.run(
            ["wpctl", "status"],
            capture_output=True, text=True, timeout=2.0,
            env=env or _pipewire_env(),
        ).stdout
    except Exception:
        return None
    in_sinks = False
    needle = match_substring.lower()
    for line in out.splitlines():
        if "Sinks:" in line:
            in_sinks = True
            continue
        if in_sinks:
            if "Sink endpoints:" in line or "Sources:" in line or "Streams:" in line or line.strip().startswith("Video"):
                break
            if needle in line.lower():
                m = re.search(r"(\d+)\.", line)
                if m:
                    return m.group(1)
    return None


class VoiceAnnouncer:
    """Background latest-wins Mandarin WAV player (never blocks robot control or Qt UI)."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        voice_dir: str | Path = DEFAULT_VOICE_DIR,
        prefer_sink_match: str = "SPEAKER-Z",
        volume: float = 0.85,
        runner=None,
    ):
        self.voice_dir = Path(voice_dir)
        self.prefer_sink_match = str(prefer_sink_match or "")
        self.volume = max(0.0, min(1.5, float(volume)))
        self._runner = runner  # optional test hook: callable(clip_key, wav_path, fallback_text)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._enabled = bool(enabled)
        self._closed = False
        self._pending: tuple[str, str] | None = None
        self._active_proc: subprocess.Popen | None = None
        self._env = _pipewire_env()
        self._pw_play = shutil.which("pw-play")
        self._aplay = shutil.which("aplay")
        self._spd_say = shutil.which("spd-say")
        self._sink_id: str | None = None
        self._volume_applied = False

        self._thread = threading.Thread(target=self._worker, name="sdk-voice", daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], *, mock: bool = False) -> "VoiceAnnouncer":
        vc = cfg.get("voice") or {}
        enabled = bool(vc.get("enabled", True)) and not mock
        return cls(
            enabled=enabled,
            voice_dir=vc.get("voice_dir", DEFAULT_VOICE_DIR),
            prefer_sink_match=str(vc.get("prefer_sink_match", "SPEAKER-Z")),
            volume=float(vc.get("volume", 0.85)),
        )

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        with self._cond:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._pending = None
                self._terminate_active_locked()
            self._cond.notify_all()

    def announce(self, clip_key: str, *, fallback_text: str | None = None) -> None:
        """Queue a clip immediately; preempts any older queued/playing clip."""
        text = fallback_text if fallback_text is not None else VOICE_PROMPTS.get(clip_key, clip_key)
        with self._cond:
            if not self._enabled or self._closed:
                return
            self._pending = (clip_key, text)
            self._terminate_active_locked()
            self._cond.notify_all()

    def test_voice(self) -> None:
        self.announce("voice_test")

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._pending = None
            self._terminate_active_locked()
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)

    def _terminate_active_locked(self) -> None:
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def _ensure_sink(self) -> str | None:
        if self._sink_id is None and self.prefer_sink_match:
            self._sink_id = find_pipewire_sink_id(self.prefer_sink_match, env=self._env)
        if self._sink_id and not self._volume_applied and shutil.which("wpctl"):
            try:
                subprocess.run(
                    ["wpctl", "set-default", self._sink_id],
                    capture_output=True, timeout=1.5, env=self._env,
                )
                subprocess.run(
                    ["wpctl", "set-mute", self._sink_id, "0"],
                    capture_output=True, timeout=1.5, env=self._env,
                )
                subprocess.run(
                    ["wpctl", "set-volume", self._sink_id, f"{self.volume:.2f}"],
                    capture_output=True, timeout=1.5, env=self._env,
                )
                self._volume_applied = True
            except Exception:
                pass
        return self._sink_id

    def _build_command(self, wav_path: Path, text: str) -> list[str] | None:
        if wav_path.is_file():
            if self._pw_play:
                sink_id = self._ensure_sink()
                if sink_id:
                    return [self._pw_play, "--target", sink_id, "--volume", f"{self.volume:.2f}", str(wav_path)]
                return [self._pw_play, "--volume", f"{self.volume:.2f}", str(wav_path)]
            if self._aplay:
                return [self._aplay, "-q", str(wav_path)]
        if self._spd_say and text:
            return [self._spd_say, "-w", "-l", "cmn", text]
        return None

    def _worker(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait(timeout=0.5)
                if self._closed:
                    return
                item = self._pending
                self._pending = None
                if item is None or not self._enabled:
                    continue

            clip_key, text = item
            wav_path = self.voice_dir / f"{clip_key}.wav"
            if self._runner is not None:
                try:
                    self._runner(clip_key, wav_path, text)
                except Exception:
                    pass
                continue

            cmd = self._build_command(wav_path, text)
            if not cmd:
                continue
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env,
                )
            except OSError:
                continue
            with self._lock:
                self._active_proc = proc
                should_kill = self._closed or (not self._enabled) or (self._pending is not None)
                if should_kill:
                    self._terminate_active_locked()
            try:
                proc.wait(timeout=12.0)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
            finally:
                with self._lock:
                    if self._active_proc is proc:
                        self._active_proc = None
