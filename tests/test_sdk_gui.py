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
                collector_name="collector_1",
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
                self.assertIn("撤销删除本段", win.btn_delete.text())
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
                self.assertIn("attempt#0", win.history_tab.ep_table.item(0, 3).text())
                self.assertEqual(win.history_tab.ep_table.item(0, 5).text(), "collector_1")
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


if __name__ == "__main__":
    unittest.main()
