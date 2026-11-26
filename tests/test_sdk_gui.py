"""Headless PyQt5 end-to-end tests for SDKDaggerMainWindow (live workbench + pedal standby + v2.1 history)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "operator"))

from sdk_dagger.gui import SDKDaggerMainWindow, scan_lerobot_v21_datasets
from sdk_dagger.prepare import load_config


def _fast_gui_cfg(tmp_dir: Path) -> dict:
    cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
    cfg["recording"]["dataset_dir"] = str(tmp_dir)
    cfg["recording"]["inference_tail_s"] = 0.08
    cfg["recording"]["dagger_preroll_s"] = 0.08
    cfg["recording"]["save_confirm_timeout_s"] = 2.0
    cfg["recording"]["confirm_inference_save"] = False
    cfg["recording"]["confirm_round_save"] = False
    cfg["inference"]["publish_rate"] = 50
    cfg["inference"]["inference_rate"] = 10
    return cfg


def _pump_until(qapp: QtWidgets.QApplication, predicate, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    qapp.processEvents()
    return bool(predicate())


class _PipePedalDevice:
    def __init__(self, fifo_path: Path):
        self.path = fifo_path
        self.fd: int | None = None
        self.buf = b""

    @property
    def filenos(self):
        return () if self.fd is None else (self.fd,)

    def open(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)

    def read_events(self, fd: int):
        try:
            data = os.read(fd, 65536)
        except BlockingIOError:
            return []
        self.buf += data
        rows = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            rows.append(tuple(json.loads(line)))
        return rows

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


class SDKDaggerGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test_sdk_gui"])

    def test_gui_full_cycle_confirm_delete_gap_and_history_playback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            win = SDKDaggerMainWindow(
                cfg,
                dataset_dir=tmp_path,
                mock=True,
                enable_pedal=False,
                auto_start=False,
                collector_name="徐鑫鑫",
            )
            try:
                # Initially in safe STANDBY
                self.assertIsNone(win.active_app)
                self.assertTrue(win.btn_start_round.isEnabled())
                self.assertFalse(win.btn_takeover.isEnabled())

                # Click "Start Round"
                win.btn_start_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_inference_frames"] >= 5))
                win._refresh_ui()
                self.assertTrue(win.btn_takeover.isEnabled())
                self.assertIsNotNone(win.cam_cards["cam_high"]._pixmap)

                # Enter DAgger via GUI button
                win.btn_takeover.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "DAGGER_RECORDING"))
                time.sleep(0.15)
                win._refresh_ui()
                self.assertTrue(win.btn_save.isEnabled())

                # Save -> CONFIRM_SAVE (delete button active with countdown!) -> click Delete
                win.btn_save.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "CONFIRM_SAVE"))
                win._refresh_ui()
                self.assertTrue(win.btn_delete.isEnabled())
                self.assertIn("撤销删除", win.btn_delete.text())
                win.btn_delete.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "DAGGER_PAUSED"))
                win._refresh_ui()
                self.assertIn("删1", win.metrics["dagger"].text())

                # Re-record -> Save -> Confirm keep -> Resume inference
                win.btn_takeover.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "DAGGER_RECORDING"))
                time.sleep(0.15)
                win._refresh_ui()
                win.btn_save.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "CONFIRM_SAVE"))
                win._refresh_ui()
                win.btn_confirm.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "DAGGER_PAUSED"))
                win._refresh_ui()
                win.btn_resume.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "INFERENCE"))
                time.sleep(0.12)

                # Stop round -> returns to STANDBY and flushes all videos
                win.btn_stop_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is None, timeout=10.0))
                self.assertTrue(win.btn_start_round.isEnabled())

                # Switch to History tab (LeRobot v2.1) and verify datasets, deleted gap, and video frame rendering!
                datasets = scan_lerobot_v21_datasets(tmp_path)
                self.assertEqual(len(datasets), 3)
                dagger_ds = [d for d in datasets if d["kind"] == "DAgger"][0]
                merged_ds = [d for d in datasets if d["kind"] == "拼接"][0]
                self.assertEqual(dagger_ds["episodes"], 1)
                self.assertEqual(dagger_ds["deleted"], 1)
                # Split into 2 continuous episodes because attempt 0 was deleted!
                self.assertEqual(merged_ds["episodes"], 2)

                win.tabs.setCurrentIndex(1)
                self.qapp.processEvents()
                self.assertEqual(win.history_tab.ds_list.count(), 3)
                # Select the DAgger dataset row and verify gap column + video frame rendered
                dag_row = next(i for i, d in enumerate(win.history_tab._datasets) if d["kind"] == "DAgger")
                win.history_tab.ds_list.setCurrentRow(dag_row)
                self.qapp.processEvents()
                self.assertEqual(win.history_tab.ep_table.rowCount(), 1)
                self.assertIn("attempt#0", win.history_tab.ep_table.item(0, 5).text())
                self.assertEqual(win.history_tab.ep_table.item(0, 7).text(), "徐鑫鑫")
                self.assertIn("已删除缺口", win.history_tab.timeline_box.toPlainText())
                self.assertIsNotNone(win.history_tab.cam_cards["cam_high"]._pixmap)
            finally:
                win.close()
                self.qapp.processEvents()

    def test_gui_integrated_dual_pedal_standby_loop_without_pty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            pedal_cfg = {
                "version": 1,
                "device": {"vendor": "16c0", "product": "05e2", "serial": "030032217272"},
                "bindings": {
                    "left": {"interface": "kbd", "code": 64, "modifiers": [29]},
                    "right": {"interface": "kbd", "code": 66, "modifiers": [29]},
                },
            }
            pedal_cfg_path = tmp_path / "pedal.json"
            pedal_cfg_path.write_text(json.dumps(pedal_cfg), encoding="utf-8")
            fifo = tmp_path / "pedal.fifo"
            os.mkfifo(fifo, 0o600)
            writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)

            def send_click(code: int):
                for ev in (["kbd", 1, 29, 1], ["kbd", 1, code, 1]):
                    os.write(writer, (json.dumps(ev) + "\n").encode())
                time.sleep(0.01)
                for ev in (["kbd", 1, 29, 0], ["kbd", 1, code, 0]):
                    os.write(writer, (json.dumps(ev) + "\n").encode())

            def send_double(code: int):
                send_click(code)
                time.sleep(0.08)
                send_click(code)

            win = SDKDaggerMainWindow(
                cfg,
                dataset_dir=tmp_path,
                mock=True,
                enable_pedal=True,
                pedal_config=pedal_cfg_path,
                pedal_device=_PipePedalDevice(fifo),
                auto_start=False,
            )
            try:
                self.assertTrue(_pump_until(self.qapp, lambda: "双踏板已连接" in win.pedal_badge.text()))

                # Standby: single left click & right double must be ignored!
                send_click(64)
                send_double(66)
                time.sleep(0.5)
                self.qapp.processEvents()
                self.assertIsNone(win.active_app)

                # Standby: left double starts Round 1!
                send_double(64)
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertEqual(win.round_count, 1)
                time.sleep(0.15)

                # Active round: right double ends Round 1 and enters STANDBY!
                send_double(66)
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is None, timeout=10.0))

                # Standby again: left double starts Round 2!
                send_double(64)
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertEqual(win.round_count, 2)
                time.sleep(0.12)

                # Window close cleanly saves Round 2 and stops pedal worker
                win.close()
                self.qapp.processEvents()
                self.assertFalse(win.pedal_worker.isRunning())

                inf_datasets = [d for d in scan_lerobot_v21_datasets(tmp_path) if d["kind"] == "推理"]
                self.assertGreaterEqual(len(inf_datasets), 2)
            finally:
                if not win._closing:
                    win.close()
                    self.qapp.processEvents()
                os.close(writer)


    def test_voice_announcer_clips_and_gui_toggle(self):
        from sdk_dagger.voice import DEFAULT_VOICE_DIR, VOICE_PROMPTS, VoiceAnnouncer
        for key in VOICE_PROMPTS:
            self.assertTrue((DEFAULT_VOICE_DIR / f"{key}.wav").is_file(), f"Missing WAV: {key}")

        played: list[str] = []
        voice = VoiceAnnouncer(enabled=True, runner=lambda k, _p, _t: played.append(k))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            win = SDKDaggerMainWindow(
                cfg, dataset_dir=tmp_path, mock=True, enable_pedal=False, auto_start=False, voice=voice,
            )
            try:
                self.assertTrue(win.voice_toggle.isChecked())
                win.btn_voice_test.click()
                self.assertTrue(_pump_until(self.qapp, lambda: "voice_test" in played, timeout=3.0))

                # Start round -> verify inference_running announced -> stop -> verify standby announced
                win.btn_start_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: "inference_running" in played, timeout=6.0))
                win.btn_stop_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: "standby" in played, timeout=8.0))

                # Mute toggle silences subsequent announcements
                win.voice_toggle.setChecked(False)
                self.qapp.processEvents()
                before = len(played)
                win.btn_voice_test.click()
                time.sleep(0.15)
                self.assertEqual(len(played), before)
            finally:
                win.close()
                self.qapp.processEvents()


    def test_stale_round_error_banner_clears_after_clean_round(self):
        """Stale error banner from an earlier failed round must clear when a later round succeeds."""
        import yaml
        from pathlib import Path
        from PyQt5 import QtWidgets
        from sdk_dagger.gui import SDKDaggerMainWindow
        from sdk_dagger.voice import VoiceAnnouncer

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cfg = yaml.safe_load(Path("config/dagger_precision_5090.yaml").read_text(encoding="utf-8"))
        win = SDKDaggerMainWindow(cfg, mock=True, voice=VoiceAnnouncer(enabled=False))
        try:
            win._on_round_finished(1, "can_right_mas 准备失败: TimeoutError")
            win._refresh_ui()
            self.assertIn("上一轮启动/运行异常", win.guide_banner.text())

            win._on_round_finished(0, "")
            win._refresh_ui()
            self.assertEqual(win._last_round_error, "")
            self.assertNotIn("异常", win.guide_banner.text())
            self.assertIn("待机中", win.guide_banner.text())
        finally:
            win.close()



    def test_pedal_worker_auto_retries_after_transient_ebusy(self):
        """_PedalWorker retries dev.open() after transient EBUSY instead of permanently giving up."""
        import errno, time
        from pathlib import Path
        from PyQt5 import QtWidgets
        from sdk_dagger.gui import _PedalWorker

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        class FlakyDevice:
            def __init__(self):
                self.attempts = 0
                self.filenos = {}

            def open(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError(errno.EBUSY, "Device or resource busy")
                self.filenos = {}  # triggers clean exit on injected device after emitting connected

            def close(self):
                pass

        flaky = FlakyDevice()
        statuses = []
        w = _PedalWorker(Path("_runtime/foot-pedal.json"), device=flaky)
        w.status_changed.connect(lambda ok, txt: statuses.append((ok, txt)))
        w.start()
        deadline = time.monotonic() + 4.0
        while w.isRunning() and time.monotonic() < deadline:
            _app.processEvents()
            time.sleep(0.05)
        w.stop()
        w.wait(2000)
        _app.processEvents()
        self.assertGreaterEqual(flaky.attempts, 2)
        self.assertTrue(any((not ok and "自动重试中" in txt) for ok, txt in statuses))
        self.assertTrue(any((ok and "双踏板已连接" in txt) for ok, txt in statuses))

    def test_gui_inference_segment_confirm_delete_on_takeover_and_stop(self):
        """Inference segments enter CONFIRM_SAVE countdown (both before takeover and on round stop) and can be discarded."""
        from sdk_dagger.voice import VoiceAnnouncer

        announced: list[str] = []
        voice = VoiceAnnouncer(enabled=True, runner=lambda _k, _p, _t: None)
        _orig_announce = voice.announce

        def _spy_announce(clip_key: str, *, fallback_text: str | None = None) -> None:
            announced.append(clip_key)
            _orig_announce(clip_key, fallback_text=fallback_text)

        voice.announce = _spy_announce  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            cfg["recording"]["confirm_inference_save"] = True
            cfg["recording"]["save_confirm_timeout_s"] = 3.0
            win = SDKDaggerMainWindow(
                cfg, dataset_dir=tmp_path, mock=True, enable_pedal=False, auto_start=False, voice=voice,
            )
            try:
                win.btn_start_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_inference_frames"] >= 5))
                win._refresh_ui()
                self.assertTrue(win.btn_takeover.isEnabled())

                # 1. Takeover triggers inference save -> CONFIRM_SAVE (target="inference") -> discard it!
                win.btn_takeover.click()
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "CONFIRM_SAVE",
                ))
                win._refresh_ui()
                snap = win.active_app.get_ui_snapshot()
                self.assertEqual(snap["confirm_target"], "inference")
                self.assertTrue(win.btn_delete.isEnabled())
                self.assertIn("撤销删除推理段", win.btn_delete.text())
                win.btn_delete.click()

                # After deleting inference segment, flow proceeds smoothly into DAGGER_RECORDING
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "DAGGER_RECORDING",
                ))
                time.sleep(0.16)
                win._refresh_ui()
                self.assertTrue(win.btn_save.isEnabled())

                # Save a DAgger segment and confirm keeping it
                win.btn_save.click()
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "CONFIRM_SAVE",
                ))
                win._refresh_ui()
                self.assertEqual(win.active_app.get_ui_snapshot()["confirm_target"], "dagger")
                win.btn_confirm.click()
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "DAGGER_PAUSED",
                ))
                win._refresh_ui()
                self.assertTrue(win.btn_resume.isEnabled())

                # Resume inference -> collect a second inference segment -> stop round -> CONFIRM_SAVE even while stopping -> discard!
                win.btn_resume.click()
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "INFERENCE",
                ))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_inference_frames"] >= 5))
                win._refresh_ui()
                win.btn_stop_round.click()
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None and win.active_app.mode == "CONFIRM_SAVE",
                ))
                win._refresh_ui()
                self.assertTrue(win.btn_delete.isEnabled())
                self.assertIn("撤销删除推理段", win.btn_delete.text())
                win.btn_delete.click()

                # Round finishes cleanly back to standby
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is None, timeout=8.0))
                self.assertEqual(win._last_round_error, "")

                # Verify both inference segments were deleted & recorded in timeline, while DAgger segment was kept
                inf_dirs = sorted(tmp_path.glob("*_inference"))
                dag_dirs = sorted(tmp_path.glob("*_dagger"))
                merged_dirs = sorted(tmp_path.glob("*_merged"))
                self.assertEqual(len(inf_dirs), 1)
                self.assertEqual(len(dag_dirs), 1)
                self.assertEqual(len(merged_dirs), 1)
                self.assertEqual(list((inf_dirs[0] / "data/chunk-000").glob("*.parquet")), [])
                inf_timeline = [
                    json.loads(line)
                    for line in (inf_dirs[0] / "meta/session_timeline.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                deleted_inf_events = [ev for ev in inf_timeline if ev.get("event") == "deleted"]
                self.assertEqual(len(deleted_inf_events), 2)
                self.assertEqual(len(list((dag_dirs[0] / "data/chunk-000").glob("*.parquet"))), 1)
                # Merged dataset only contains the 1 kept DAgger segment (no discarded inference segments)
                merged_eps = [
                    json.loads(line)
                    for line in (merged_dirs[0] / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(merged_eps), 1)
                self.assertEqual([s["source"] for s in merged_eps[0]["source_segments"]], ["dagger"])
                self.assertEqual(announced.count("deleted"), 2)
                self.assertGreaterEqual(announced.count("saved_confirm"), 3)
            finally:
                win.close()
                self.qapp.processEvents()


    def test_daily_merged_rounds_counter_and_history_summary(self):
        """Live workbench KPI card and history tab summary must count valid on-disk *_merged rounds by date and collector."""
        import datetime, json, tempfile
        from pathlib import Path
        import yaml
        from PyQt5 import QtWidgets
        from sdk_dagger.gui import (
            SDKDaggerMainWindow,
            format_daily_summary_banner,
            summarize_daily_merged_rounds,
        )
        from sdk_dagger.voice import VoiceAnnouncer

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        today = datetime.date.today()
        today_compact = today.strftime("%Y%m%d")
        yest_compact = (today - datetime.timedelta(days=1)).strftime("%Y%m%d")

        def _make_merged(root_dir: Path, name: str, *, episodes: int, frames: int, collector: str | None = None):
            meta = root_dir / name / "meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "info.json").write_text(
                json.dumps({"total_episodes": episodes, "total_frames": frames, "fps": 30}),
                encoding="utf-8",
            )
            ep = {"episode_index": 0, "length": frames}
            if collector:
                ep["collector_name"] = collector
            (meta / "episodes.jsonl").write_text(json.dumps(ep, ensure_ascii=False) + "\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            # Valid today rounds (2 total: 1 郑进宝, 1 赖星羽)
            _make_merged(td, f"cloth_sdk_{today_compact}_100001_100_r1_merged", episodes=1, frames=120, collector="郑进宝")
            _make_merged(td, f"cloth_sdk_{today_compact}_100501_100_r2_merged", episodes=1, frames=150, collector="赖星羽")
            # Invalid empty round today (0 frames) + non-merged dirs must NOT be counted
            _make_merged(td, f"cloth_sdk_{today_compact}_100601_100_r3_merged", episodes=0, frames=0, collector="郑进宝")
            _make_merged(td, f"cloth_sdk_{today_compact}_100001_100_r1_inference", episodes=1, frames=120, collector="郑进宝")
            # Valid yesterday rounds (3 total)
            for i in range(1, 4):
                _make_merged(td, f"cloth_sdk_{yest_compact}_18000{i}_200_r{i}_merged", episodes=1, frames=200)

            summary = summarize_daily_merged_rounds(td, today=today)
            self.assertEqual(summary["today_rounds"], 2)
            self.assertEqual(summary["yesterday_rounds"], 3)
            self.assertEqual(summary["today_by_collector"], {"郑进宝": 1, "赖星羽": 1})
            banner = format_daily_summary_banner(summary)
            self.assertIn(f"今日 {today.isoformat()}：2 轮", banner)
            self.assertIn(f"昨日 {(today - datetime.timedelta(days=1)).isoformat()}：3 轮", banner)

            cfg = yaml.safe_load(Path("config/dagger_precision_5090.yaml").read_text(encoding="utf-8"))
            cfg.setdefault("recording", {})["dataset_dir"] = str(td)
            win = SDKDaggerMainWindow(cfg, mock=True, voice=VoiceAnnouncer(enabled=False))
            try:
                self.assertEqual(win.metrics["daily"].text(), "2 轮")
                # Switch collector to 郑进宝 -> updates live card immediately
                idx = win.collector_combo.findData("郑进宝")
                self.assertGreaterEqual(idx, 0)
                win.collector_combo.setCurrentIndex(idx)
                self.assertEqual(win.metrics["daily"].text(), "2 轮（郑进宝 1）")
                # History tab summary banner matches
                win.history_tab.refresh()
                self.assertIn(f"今日 {today.isoformat()}：2 轮", win.history_tab.daily_summary_label.text())
                self.assertIn(f"昨日 {(today - datetime.timedelta(days=1)).isoformat()}：3 轮", win.history_tab.daily_summary_label.text())
            finally:
                win.close()


    def test_verify_receipts_and_episode_quality_tags_in_gui(self):
        """Auto save receipts, manual read-back verification, and piper_ms_teleop-style quality tagging update both tabs."""
        import datetime, tempfile
        from pathlib import Path
        import numpy as np
        import yaml
        from PyQt5 import QtWidgets
        from sdk_dagger.episode_tags import read_tags, save_tags
        from sdk_dagger.gui import SDKDaggerMainWindow
        from sdk_dagger.storage import FrameSample, LeRobotV21DatasetWriter, merge_round_lerobot_v21
        from sdk_dagger.verify import read_save_receipts, verify_dataset_v21
        from sdk_dagger.voice import VoiceAnnouncer

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        today_compact = datetime.date.today().strftime("%Y%m%d")

        def _make_samples(n: int) -> list[FrameSample]:
            out = []
            for i in range(n):
                img = np.full((64, 80, 3), 100 + i * 2, dtype=np.uint8)
                q = np.linspace(0.0, 0.2, 14, dtype=np.float32) + i * 0.005
                out.append(FrameSample(
                    timestamp=1000.0 + i / 30.0,
                    qpos=q,
                    qvel=np.zeros(14, dtype=np.float32),
                    effort=np.zeros(14, dtype=np.float32),
                    images_bgr={"cam_high": img, "cam_left_wrist": img, "cam_right_wrist": img},
                ))
            return out

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            prefix = f"cloth_sdk_{today_compact}_120000_999_r1"
            inf_dir = td / f"{prefix}_inference"
            dag_dir = td / f"{prefix}_dagger"
            mrg_dir = td / f"{prefix}_merged"

            w_inf = LeRobotV21DatasetWriter(inf_dir, prompt="Organize", fps=30)
            r_inf = w_inf.save_episode(_make_samples(12), export_video=True, collector_name="郑进宝")
            self.assertTrue(r_inf["saved"])
            # Auto read-back receipt created on save
            inf_receipts = read_save_receipts(inf_dir)
            self.assertIn(0, inf_receipts)
            self.assertTrue(inf_receipts[0]["passed"])

            mres = merge_round_lerobot_v21(
                inference_dir=inf_dir,
                dagger_dir=dag_dir,
                merged_dir=mrg_dir,
                stitch_events=[{"kind": "segment", "source": "inference", "episode_index": 0}],
                prompt="Organize",
                fps=30,
                export_video=True,
                collector_name="郑进宝",
            )
            self.assertTrue(mres["merged"])
            self.assertTrue(mres["verified"])
            self.assertTrue(read_save_receipts(mrg_dir)[0]["passed"])

            # Offline CLI/API verifier passes
            vrep = verify_dataset_v21(mrg_dir, write_receipts=False)
            self.assertTrue(vrep["passed"])

            # Optimistic revision lock on episode_tags
            row1 = save_tags(mrg_dir, 0, quality="usable", tags=["操作失误"], note="ok", actor="郑进宝", expected_revision=0)
            self.assertEqual(row1["revision"], 1)
            with self.assertRaises(ValueError):
                save_tags(mrg_dir, 0, quality="unusable", actor="郑进宝", expected_revision=0)

            # GUI History tab shows verified badge and tagging updates live daily counter
            cfg = yaml.safe_load(Path("config/dagger_precision_5090.yaml").read_text(encoding="utf-8"))
            cfg.setdefault("recording", {})["dataset_dir"] = str(td)
            win = SDKDaggerMainWindow(cfg, mock=True, voice=VoiceAnnouncer(enabled=False))
            try:
                self.assertEqual(win.metrics["daily"].text(), "1 轮")
                win.history_tab.refresh()
                # Find merged dataset in list
                mrg_row = next(i for i, d in enumerate(win.history_tab._datasets) if d["name"] == mrg_dir.name)
                win.history_tab.ds_list.setCurrentRow(mrg_row)
                self.assertIn("✅", win.history_tab.ds_list.item(mrg_row).text())
                self.assertIn("✅", win.history_tab.ep_table.item(0, 2).text())

                # Mark episode 0 as unusable with preset tag via UI panel
                win.history_tab.tags_panel.quality.setCurrentIndex(
                    win.history_tab.tags_panel.quality.findData("unusable")
                )
                win.history_tab.tags_panel.add_preset(1)  # 画面异常
                win.history_tab.tags_panel.note.setText("左腕短暂遮挡")
                win.history_tab.tags_panel.store()

                doc = read_tags(mrg_dir)
                self.assertEqual(doc["episodes"]["0"]["quality"], "unusable")
                self.assertIn("画面异常", doc["episodes"]["0"]["tags"])
                # Live KPI card and history banner immediately reflect 可用 0
                self.assertIn("可用 0", win.metrics["daily"].text())
                self.assertIn("不可用 1", win.history_tab.daily_summary_label.text())
                self.assertIn("🚫不可用:1", win.history_tab.ds_list.item(mrg_row).text())
            finally:
                win.close()


    def test_camera_stall_and_zero_frame_save_show_prominent_red_gui_alert(self):
        """Stale/disconnected camera must trigger prominent red status badge, red guide banner, red camera card overlay, and explicit 0-frame save alert."""
        import tempfile, time
        from pathlib import Path
        import numpy as np
        import yaml
        from PyQt5 import QtWidgets
        from sdk_dagger.app import SDKDaggerApp
        from sdk_dagger.gui import SDKDaggerMainWindow
        from sdk_dagger.storage import FrameSample
        from sdk_dagger.voice import VoiceAnnouncer

        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            cfg = yaml.safe_load(Path("config/dagger_precision_5090.yaml").read_text(encoding="utf-8"))
            cfg.setdefault("recording", {})["dataset_dir"] = tmp
            spoken = []
            voice = VoiceAnnouncer(enabled=True, runner=lambda key, wav, txt: spoken.append(key))
            win = SDKDaggerMainWindow(cfg, mock=True, voice=voice)
            try:
                app = SDKDaggerApp(cfg, dataset_name="cam_alert_test", dataset_dir=tmp, mock=True, enable_terminal_input=False, voice=voice)
                img = np.full((64, 80, 3), 120, dtype=np.uint8)
                stale_sample = FrameSample(
                    timestamp=time.time() - 12.5,
                    qpos=np.zeros(14, dtype=np.float32),
                    qvel=np.zeros(14, dtype=np.float32),
                    effort=np.zeros(14, dtype=np.float32),
                    images_bgr={"cam_high": img, "cam_left_wrist": img, "cam_right_wrist": img},
                )
                app._ready = True
                app.mode = "DAGGER_RECORDING"
                app._latest_sample = stale_sample
                app._obs_error = "Frame didn'''t arrive within 5000"
                # Simulate ONLY left wrist camera stalled while front & right wrist are healthy!
                app.rig.get_camera_health = lambda: {
                    "cam_high": {"role": "front", "label": "正面相机", "serial": "263022072158", "age_s": 0.03, "stale": False},
                    "cam_left_wrist": {"role": "left", "label": "左腕相机", "serial": "263022071527", "age_s": 12.5, "stale": True},
                    "cam_right_wrist": {"role": "right", "label": "右腕相机", "serial": "263022070473", "age_s": 0.03, "stale": False},
                }
                win.active_app = app
                win._refresh_ui()

                self.assertIn("左腕相机中断", win.status_badge.text())
                self.assertIn("左腕相机", win.guide_banner.text())
                self.assertIn("263022071527", win.guide_banner.text())
                self.assertIn("无法保存", win.guide_banner.text())
                # Only the faulty left wrist camera card turns red; healthy front stays normal green!
                self.assertIn("画面中断", win.cam_cards["cam_left_wrist"].health.text())
                self.assertIn("#ff4d5a", win.cam_cards["cam_left_wrist"].styleSheet())
                self.assertIn("实时", win.cam_cards["cam_high"].health.text())
                self.assertEqual(win.cam_cards["cam_high"].styleSheet(), "")

                res = app._save_dagger_segment()
                self.assertFalse(res["saved"])
                deadline = time.monotonic() + 1.0
                while "camera_error" not in spoken and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertIn("camera_error", spoken)
                self.assertIn("不会进入 10 秒删除确认", app.get_ui_snapshot()["status_message"])
            finally:
                win.active_app = None
                win.close()

    def test_gui_stop_round_unsaved_dagger_and_whole_round_discard_or_keep(self):
        """Right-double stop prompts for both unsaved last segment (inference/dagger) and whole-round keep/discard."""
        from sdk_dagger.voice import VoiceAnnouncer

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            cfg["recording"]["confirm_inference_save"] = True
            cfg["recording"]["confirm_round_save"] = True
            cfg["recording"]["save_confirm_timeout_s"] = 3.0
            win = SDKDaggerMainWindow(
                cfg,
                dataset_dir=tmp_path,
                mock=True,
                enable_pedal=False,
                auto_start=False,
                voice=VoiceAnnouncer(enabled=False),
            )
            try:
                # --- Round 1: Keep inference -> enter DAgger -> stop directly while DAGGER_RECORDING ---
                # Step 1: unsaved DAgger segment prompts confirm -> delete it
                # Step 2: whole-round prompts confirm -> delete ENTIRE round (removes all dirs!)
                win.btn_start_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_inference_frames"] >= 5))
                win._refresh_ui()
                win.btn_takeover.click()

                # Keep inference segment before DAgger
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "CONFIRM_SAVE"))
                win._refresh_ui()
                self.assertEqual(win.active_app.get_ui_snapshot()["confirm_target"], "inference")
                win.btn_confirm.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.mode == "DAGGER_RECORDING"))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_dagger_frames"] >= 5))
                win._refresh_ui()

                # Trigger right-double stop while actively recording DAgger (no prior 's')
                win._on_pedal_gesture("stop")
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None
                    and win.active_app.mode == "CONFIRM_SAVE"
                    and win.active_app.get_ui_snapshot()["confirm_target"] == "dagger",
                ))
                win._refresh_ui()
                self.assertIn("撤销删除纠正段", win.btn_delete.text())
                win.btn_delete.click()

                # Next: whole-round confirmation (because earlier inference segment is still kept)
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None
                    and win.active_app.mode == "CONFIRM_SAVE"
                    and win.active_app.get_ui_snapshot()["confirm_target"] == "round",
                ))
                win._refresh_ui()
                self.assertIn("丢弃整轮数据", win.btn_delete.text())
                self.assertIn("确认保留整轮", win.btn_confirm.text())
                win.btn_delete.click()

                # Round finishes cleanly and ALL directories from Round 1 are deleted
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is None, timeout=8.0))
                self.assertEqual(win._last_round_error, "")
                self.assertEqual(list(tmp_path.iterdir()), [])

                # --- Round 2: Stop during INFERENCE -> keep final inference -> keep whole round ---
                win.btn_start_round.click()
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is not None and win.active_app.get_ui_snapshot()["ready"]))
                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app.get_ui_snapshot()["current_inference_frames"] >= 5))
                win._refresh_ui()

                win._on_pedal_gesture("stop")
                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None
                    and win.active_app.mode == "CONFIRM_SAVE"
                    and win.active_app.get_ui_snapshot()["confirm_target"] == "inference",
                ))
                win._refresh_ui()
                # Right-double again during CONFIRM_SAVE confirms keeping (never force-kills!)
                win._on_pedal_gesture("stop")

                self.assertTrue(_pump_until(
                    self.qapp,
                    lambda: win.active_app is not None
                    and win.active_app.mode == "CONFIRM_SAVE"
                    and win.active_app.get_ui_snapshot()["confirm_target"] == "round",
                ))
                win._refresh_ui()
                win.btn_confirm.click()

                self.assertTrue(_pump_until(self.qapp, lambda: win.active_app is None, timeout=8.0))
                self.assertEqual(win._last_round_error, "")
                self.assertEqual(len(list(tmp_path.glob("*_inference"))), 1)
                self.assertEqual(len(list(tmp_path.glob("*_merged"))), 1)
            finally:
                win.close()
                self.qapp.processEvents()


    def test_ime_immune_delete_shortcuts_and_tab_ime_isolation(self):
        from PyQt5 import QtCore, QtGui, QtWidgets

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_gui_cfg(tmp_path)
            win = SDKDaggerMainWindow(
                cfg,
                dataset_dir=tmp_path,
                mock=True,
                enable_pedal=False,
                auto_start=False,
            )
            try:
                # Layer 1: Tab 0 disables IME on window/tabs/live_page/children; Tab 1 restores IME
                self.assertFalse(win.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                self.assertFalse(win.tabs.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                self.assertFalse(win.live_page.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                self.assertFalse(win.log_box.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                self.assertFalse(win.btn_delete.testAttribute(QtCore.Qt.WA_InputMethodEnabled))

                win.tabs.setCurrentIndex(1)
                self.qapp.processEvents()
                self.assertTrue(win.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                self.assertTrue(win.tabs.testAttribute(QtCore.Qt.WA_InputMethodEnabled))
                if hasattr(win.history_tab, "tags_panel"):
                    edits = win.history_tab.tags_panel.findChildren(QtWidgets.QLineEdit)
                    self.assertTrue(len(edits) >= 1)
                    for ed in edits:
                        self.assertTrue(ed.testAttribute(QtCore.Qt.WA_InputMethodEnabled))

                win.tabs.setCurrentIndex(0)
                self.qapp.processEvents()
                self.assertFalse(win.testAttribute(QtCore.Qt.WA_InputMethodEnabled))

                # Stub active_app to test Layer 2 shortcut fallbacks deterministically
                actions: list[str] = []
                mode_box = {"mode": "INFERENCE"}

                class _DummyApp:
                    def get_ui_snapshot(self):
                        return {"mode": mode_box["mode"]}

                    def handle_operator_action(self, action: str, source: str = "gui"):
                        actions.append(action)

                win.active_app = _DummyApp()
                win._refresh_ui = lambda: None

                # Backspace/Delete do NOT trigger 'w' outside CONFIRM_SAVE
                bs_ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Backspace, QtCore.Qt.NoModifier)
                self.assertFalse(win.eventFilter(win.live_page, bs_ev))
                self.assertEqual(actions, [])

                # Hardware X11 scancode fallback (25 == W) works even when key is Key_unknown
                scan_w = QtGui.QKeyEvent(
                    QtCore.QEvent.KeyPress, QtCore.Qt.Key_unknown, QtCore.Qt.NoModifier, 25, 0, 0, ""
                )
                self.assertTrue(win.eventFilter(win.live_page, scan_w))
                self.assertEqual(actions, ["w"])
                actions.clear()

                # During CONFIRM_SAVE: Backspace, Delete, and InputMethod preedit 'w' all trigger 'w'
                mode_box["mode"] = "CONFIRM_SAVE"
                self.assertTrue(win.eventFilter(win.live_page, bs_ev))
                del_ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Delete, QtCore.Qt.NoModifier)
                self.assertTrue(win.eventFilter(win.live_page, del_ev))
                im_ev = QtGui.QInputMethodEvent("w", [])
                self.assertTrue(win.eventFilter(win.live_page, im_ev))
                self.assertEqual(actions, ["w", "w", "w"])
            finally:
                win.active_app = None
                win.close()
                self.qapp.processEvents()

if __name__ == "__main__":
    unittest.main()
