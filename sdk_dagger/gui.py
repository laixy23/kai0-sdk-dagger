#!/usr/bin/env python3
"""PyQt5 desktop workbench for pure-SDK (zero-ROS) Kai0 DAgger + LeRobot v2.1 history viewer."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time
from typing import Any

import PyQt5 as _pyqt5
_PYQT5_PLUGINS = Path(_pyqt5.__file__).resolve().parent / "Qt5" / "plugins"
if _PYQT5_PLUGINS.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_PYQT5_PLUGINS)

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

# Re-assert PyQt5 plugin path because opencv-python overwrites QT_QPA_PLATFORM_PLUGIN_PATH on import
if _PYQT5_PLUGINS.is_dir():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_PYQT5_PLUGINS)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "operator") not in sys.path:
    sys.path.insert(0, str(ROOT / "operator"))

from foot_pedal_device import FootPedalDevice, PedalGestures, load_config as load_pedal_config
from sdk_dagger.app import SDKDaggerApp
from sdk_dagger.prepare import load_config
from sdk_dagger.storage import CAMERA_NAMES, LEROBOT_CAMERA_KEYS
from sdk_dagger.voice import VoiceAnnouncer

DEFAULT_COLLECTORS = (
    "collector_1",
    "collector_2",
    "collector_3",
)

STYLE = """
QWidget { background: #0c121b; color: #dfe7f1; font-size: 13px; }
QMainWindow { background: #0c121b; }
QFrame#panel { background: #131d2b; border: 1px solid #28374b; border-radius: 10px; }
QLabel { background: transparent; }
QLabel#brand { color: #72e5c3; font-size: 15px; font-weight: 700; letter-spacing: 1px; }
QLabel#heading { font-size: 18px; font-weight: 700; color: #eef5fc; }
QLabel#section { font-size: 14px; font-weight: 600; color: #dbe7f5; }
QLabel#muted { color: #8b9db4; }
QLabel#metric { font-size: 20px; font-weight: 700; color: #f2f7fd; }
QLabel#key { background: #1c2a3b; border: 1px solid #354a64; border-radius: 4px;
             padding: 2px 7px; color: #a9c7e4; font-weight: 600; }
QPushButton { background: #202f43; border: 1px solid #364b66; border-radius: 7px;
              padding: 10px 16px; color: #e6eff9; font-weight: 600; }
QPushButton:hover { background: #2a3e58; border-color: #6b91b8; }
QPushButton:disabled { background: #151f2c; border-color: #253345; color: #56687e; }
QPushButton#start_round { background: #6ddcba; border-color: #6ddcba; color: #0a241d; font-weight: 700; }
QPushButton#start_round:hover { background: #93edd2; }
QPushButton#takeover { background: #266db3; border-color: #4694df; color: #ffffff; }
QPushButton#takeover:hover { background: #3180cc; }
QPushButton#save { background: #183a30; border-color: #3f997d; color: #9eead3; }
QPushButton#save:hover { background: #214f41; }
QPushButton#delete_warn { background: #5a2329; border-color: #d65a66; color: #ffd4d8; font-weight: 700; }
QPushButton#delete_warn:hover { background: #742c34; }
QPushButton#resume { background: #21415c; border-color: #4b8ec2; color: #c7e6ff; }
QPushButton#resume:hover { background: #2b5476; }
QPushButton#stop_round { background: #362a21; border-color: #8f6b47; color: #f7d6a8; }
QPushButton#stop_round:hover { background: #4a392c; }
QPushButton#start_round:disabled, QPushButton#takeover:disabled, QPushButton#save:disabled,
QPushButton#delete_warn:disabled, QPushButton#resume:disabled, QPushButton#stop_round:disabled {
    background: #151f2c; border-color: #253345; color: #56687e;
}
QComboBox { background: #1d2b3c; border: 1px solid #344760; border-radius: 6px; padding: 6px 12px; }
QComboBox QAbstractItemView { background: #1d2b3c; color: #dfe7f1; selection-background-color: #2e4d6c; }
QTabWidget::pane { border: none; background: #0c121b; }
QTabBar::tab { background: #111c2a; color: #879ab1; padding: 10px 24px; font-weight: 600; }
QTabBar::tab:selected { color: #72e5c3; background: #1a2a3b; border-bottom: 2px solid #72e5c3; }
QTableWidget { background: #131d2b; border: 1px solid #28374b; border-radius: 6px;
               gridline-color: #233447; selection-background-color: #264561; }
QHeaderView::section { background: #19283a; color: #9ab1c9; padding: 7px; border: 0; font-weight: 600; }
QPlainTextEdit { background: #0e1621; color: #b7cbe0; border: 1px solid #243447; border-radius: 6px; padding: 8px; }
QListWidget { background: #131d2b; border: 1px solid #28374b; border-radius: 6px; }
QListWidget::item { padding: 8px 10px; border-bottom: 1px solid #1e2d3f; }
QListWidget::item:selected { background: #24415c; color: #ffffff; }
"""


def _make_label(text: str = "", obj: str | None = None, *, mono: bool = False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    if obj:
        w.setObjectName(obj)
    if mono:
        w.setFont(QtGui.QFont("DejaVu Sans Mono", 11))
    return w


def _make_panel(compact: bool = False) -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
    frame = QtWidgets.QFrame()
    frame.setObjectName("panel")
    box = QtWidgets.QVBoxLayout(frame)
    box.setContentsMargins(12, 8 if compact else 10, 12, 8 if compact else 10)
    box.setSpacing(6 if compact else 8)
    return frame, box


def _bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()


class CameraPreviewCard(QtWidgets.QFrame):
    def __init__(self, index_label: str, title: str, code: str):
        super().__init__()
        self.setObjectName("panel")
        self._title = title
        self._pixmap: QtGui.QPixmap | None = None
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(10, 8, 10, 8)
        box.setSpacing(6)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(_make_label(index_label, "muted", mono=True))
        top.addWidget(_make_label(title, "section"))
        top.addStretch()
        top.addWidget(_make_label(code, "muted", mono=True))
        box.addLayout(top)

        self.view = QtWidgets.QLabel("等待相机画面")
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setMinimumSize(180, 160)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.view.setStyleSheet("background: #080d14; color: #647890; border-radius: 5px;")
        self.view.mouseDoubleClickEvent = self._zoom
        box.addWidget(self.view, 1)

        foot = QtWidgets.QHBoxLayout()
        self.health = _make_label("● 等待连接", "muted")
        self.detail = _make_label("—  ·  双击放大", "muted", mono=True)
        foot.addWidget(self.health)
        foot.addStretch()
        foot.addWidget(self.detail)
        box.addLayout(foot)

    def set_bgr(self, bgr: np.ndarray, *, age_s: float | None = None) -> None:
        qimg = _bgr_to_qimage(bgr)
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self._rescale()
        h, w = bgr.shape[:2]
        if age_s is None:
            self.health.setText("● 已加载")
            self.health.setStyleSheet("color: #72e5c3;")
        else:
            stale = age_s > 1.0
            self.health.setText(f'● {"画面停更" if stale else "实时"}  {age_s * 1000:.0f} ms')
            self.health.setStyleSheet(f'color: {"#efae79" if stale else "#72e5c3"};')
        self.detail.setText(f"{w}×{h}  ·  双击放大")

    def clear_frame(self, text: str = "待机中（启动新一轮后显示实时画面）") -> None:
        self._pixmap = None
        self.view.clear()
        self.view.setText(text)
        self.health.setText("● 待机")
        self.health.setStyleSheet("color: #8b9db4;")
        self.detail.setText("—  ·  双击放大")

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.view.setPixmap(
                self._pixmap.scaled(self.view.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _zoom(self, _event) -> None:
        if self._pixmap is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{self._title} · 当前帧")
        dlg.resize(960, 720)
        lay = QtWidgets.QVBoxLayout(dlg)
        lbl = QtWidgets.QLabel()
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setPixmap(self._pixmap.scaled(920, 660, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        lay.addWidget(lbl, 1)
        dlg.exec_()


class DualArmReadoutBar(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        self._sides: dict[str, dict[str, Any]] = {}
        for side, title in (("left", "左从臂 (can_left_slave)"), ("right", "右从臂 (can_right_slave)")):
            card, box = _make_panel(compact=True)
            top = QtWidgets.QHBoxLayout()
            top.addWidget(_make_label(title, "section"))
            top.addStretch()
            state = _make_label("● 待机", "muted")
            top.addWidget(state)
            box.addLayout(top)
            grid = QtWidgets.QGridLayout()
            cells = []
            for j in range(6):
                grid.addWidget(_make_label(f"J{j + 1}", "muted", mono=True), 0, j)
                val = _make_label("—", mono=True)
                val.setStyleSheet("font-size: 15px; font-weight: 600; color: #eef5fc;")
                grid.addWidget(val, 1, j)
                cells.append(val)
            box.addLayout(grid)
            grip = _make_label("夹爪：—  ·  力矩：—", "muted", mono=True)
            box.addWidget(grip)
            row.addWidget(card, 1)
            self._sides[side] = {"state": state, "cells": cells, "grip": grip}

    def update_from_sample(self, sample, *, mode_label: str, age_s: float | None) -> None:
        if sample is None:
            for w in self._sides.values():
                w["state"].setText(f"● {mode_label}")
                w["state"].setStyleSheet("color: #8b9db4;")
                for c in w["cells"]:
                    c.setText("—")
                w["grip"].setText("夹爪：—  ·  力矩：—")
            return
        qpos = np.asarray(sample.qpos, dtype=np.float64)
        effort = np.asarray(sample.effort, dtype=np.float64)
        stale = age_s is not None and age_s > 1.0
        color = "#efae79" if stale else "#72e5c3"
        age_txt = f"{age_s * 1000:.0f} ms" if age_s is not None else "—"
        for side, off in (("left", 0), ("right", 7)):
            w = self._sides[side]
            w["state"].setText(f"● {mode_label} · {age_txt}")
            w["state"].setStyleSheet(f"color: {color};")
            for j in range(6):
                w["cells"][j].setText(f"{math.degrees(qpos[off + j]):+6.1f}°")
            grip_mm = qpos[off + 6] * 1000.0
            grip_eff = effort[off + 6] if len(effort) > off + 6 else 0.0
            w["grip"].setText(f"夹爪：{grip_mm:5.1f} mm  ·  力矩：{grip_eff:+.2f}")


class _PedalWorker(QtCore.QThread):
    """Background USB dual-pedal reader using exact FootPedalDevice + PedalGestures (no PTY needed)."""

    gesture = QtCore.pyqtSignal(str)
    status_changed = QtCore.pyqtSignal(bool, str)

    def __init__(self, pedal_config_path: str | Path, *, device=None, double_click_window_s: float = 0.8):
        super().__init__()
        self._cfg_path = Path(pedal_config_path)
        self._injected_device = device
        self._double_click_window_s = float(double_click_window_s)
        self._stop = threading.Event()
        self._allow_short = True
        self._reset_requested = False
        self._state_lock = threading.Lock()

    def set_allow_short(self, allow: bool) -> None:
        with self._state_lock:
            self._allow_short = bool(allow)

    def reset_gestures(self) -> None:
        with self._state_lock:
            self._reset_requested = True

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        try:
            cfg = load_pedal_config(self._cfg_path)
        except Exception as exc:
            self.status_changed.emit(False, f"踏板配置不可用：{exc}")
            return

        last_err_text = ""
        while not self._stop.is_set():
            dev = FootPedalDevice(cfg) if self._injected_device is None else self._injected_device
            try:
                dev.open()
            except Exception as exc:
                err_text = f"踏板暂不可用（自动重试中）：{exc}"
                if err_text != last_err_text:
                    self.status_changed.emit(False, err_text)
                    last_err_text = err_text
                self._stop.wait(1.0)
                continue

            last_err_text = ""
            gestures = PedalGestures(cfg["bindings"], double_click_window_s=self._double_click_window_s)
            self.status_changed.emit(True, "双踏板已连接（左/右单双踩均可用）")
            try:
                while not self._stop.is_set():
                    fds = list(dev.filenos)
                    if not fds:
                        self.status_changed.emit(False, "双踏板已断开（自动重连中）")
                        break
                    try:
                        readable = select.select(fds, [], [], 0.05)[0]
                    except (OSError, ValueError):
                        break
                    events = []
                    read_err = None
                    for fd in readable:
                        try:
                            events.extend(dev.read_events(fd))
                        except (OSError, RuntimeError) as exc:
                            read_err = exc
                            break
                    if read_err is not None:
                        self.status_changed.emit(False, f"踏板读取断开（自动重连中）：{read_err}")
                        break
                    now = time.monotonic()
                    events.sort(key=lambda ev: ev[4] if len(ev) == 5 else now)
                    with self._state_lock:
                        allow_short = self._allow_short
                        if self._reset_requested:
                            gestures = PedalGestures(cfg["bindings"], double_click_window_s=self._double_click_window_s)
                            self._reset_requested = False
                    for ev in events:
                        for tok in gestures.feed(ev, now, allow_short=allow_short):
                            self.gesture.emit(tok)
                    for tok in gestures.tick(time.monotonic()):
                        self.gesture.emit(tok)
            finally:
                try:
                    dev.close()
                except Exception:
                    pass
            if self._injected_device is not None:
                return
            self._stop.wait(1.0)


class _RoundWorker(QtCore.QThread):
    finished_round = QtCore.pyqtSignal(int, str)

    def __init__(self, app: SDKDaggerApp):
        super().__init__()
        self.app = app

    def run(self) -> None:
        err = ""
        rc = 0
        try:
            rc = int(self.app.run())
        except Exception as exc:
            import traceback
            rc = 1
            err = str(exc)
            print(f"\n❌ 本轮运行异常：{err}", flush=True)
            traceback.print_exc()
        self.finished_round.emit(rc, err)


def scan_lerobot_v21_datasets(root_dir: str | Path) -> list[dict[str, Any]]:
    """Scan local directory for Kai0 LeRobot v2.1 datasets with timeline gap counts."""
    base = Path(root_dir).expanduser()
    if not base.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for info_path in sorted(base.rglob("meta/info.json")):
        if any(p.startswith(".") for p in info_path.relative_to(base).parts):
            continue
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if info.get("codebase_version") != "v2.1":
            continue
        ds_dir = info_path.parent.parent
        timeline_path = ds_dir / "meta/session_timeline.jsonl"
        deleted_cnt = 0
        collision_cnt = 0
        if timeline_path.is_file():
            for line in timeline_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        ev_name = json.loads(line).get("event")
                        if ev_name == "deleted":
                            deleted_cnt += 1
                        elif ev_name == "collision_avoidance":
                            collision_cnt += 1
                    except Exception:
                        pass
        found.append({
            "path": str(ds_dir),
            "name": ds_dir.name,
            "kind": "拼接" if ds_dir.name.endswith("_merged") else ("DAgger" if ds_dir.name.endswith("_dagger") else ("推理" if ds_dir.name.endswith("_inference") else "其他")),
            "episodes": int(info.get("total_episodes", 0)),
            "frames": int(info.get("total_frames", 0)),
            "fps": int(info.get("fps", 30)),
            "deleted": deleted_cnt,
            "collision_avoidances": collision_cnt,
        })
    found.sort(key=lambda x: x["name"], reverse=True)
    return found


def load_lerobot_v21_details(ds_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(ds_dir)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    episodes: list[dict[str, Any]] = []
    ep_path = root / "meta/episodes.jsonl"
    if ep_path.is_file():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(json.loads(line))
    timeline: list[dict[str, Any]] = []
    tl_path = root / "meta/session_timeline.jsonl"
    if tl_path.is_file():
        for line in tl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                timeline.append(json.loads(line))
    return episodes, timeline, info


class LeRobotV21HistoryTab(QtWidgets.QWidget):
    """Read-only LeRobot v2.1 dataset browser, deleted-gap inspector, and 3-cam MP4 player."""

    def __init__(self, data_root: str | Path):
        super().__init__()
        self.data_root = Path(data_root)
        self._datasets: list[dict[str, Any]] = []
        self._current_ds: Path | None = None
        self._current_episodes: list[dict[str, Any]] = []
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._total_frames = 0
        self._cur_frame = 0

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(33)
        self._play_timer.timeout.connect(self._step_play)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(_make_label("LeRobot v2.1 历史数据与缺口回看", "heading"))
        top.addWidget(_make_label(f"目录：{self.data_root}", "muted"))
        top.addStretch()
        self.btn_refresh = QtWidgets.QPushButton("刷新列表")
        self.btn_refresh.clicked.connect(self.refresh)
        top.addWidget(self.btn_refresh)
        root.addLayout(top)

        split = QtWidgets.QHBoxLayout()
        split.setSpacing(12)

        # Left column: dataset list + timeline log
        left = QtWidgets.QVBoxLayout()
        left.addWidget(_make_label("已保存数据集（按时间倒序）", "section"))
        self.ds_list = QtWidgets.QListWidget()
        self.ds_list.setMinimumWidth(340)
        self.ds_list.currentRowChanged.connect(self._select_dataset)
        left.addWidget(self.ds_list, 2)
        left.addWidget(_make_label("时间线与删除缺口（session_timeline.jsonl）", "section"))
        self.timeline_box = QtWidgets.QPlainTextEdit()
        self.timeline_box.setReadOnly(True)
        self.timeline_box.setMaximumHeight(180)
        left.addWidget(self.timeline_box, 1)
        split.addLayout(left, 1)

        # Right column: episode table + 3-cam video player
        right = QtWidgets.QVBoxLayout()
        self.ep_table = QtWidgets.QTableWidget(0, 7)
        self.ep_table.setHorizontalHeaderLabels([
            "episode_index", "帧数", "attempt_index", "前面被删缺口", "紧跟推理段", "采集员", "防碰撞微调",
        ])
        hdr = self.ep_table.horizontalHeader()
        hdr.setMinimumSectionSize(76)
        for col_idx in range(6):
            hdr.setSectionResizeMode(col_idx, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        self.ep_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.ep_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ep_table.setMaximumHeight(200)
        self.ep_table.currentCellChanged.connect(lambda r, *_: self._select_episode(r))
        right.addWidget(self.ep_table)

        cam_row = QtWidgets.QHBoxLayout()
        cam_row.setSpacing(10)
        self.cam_cards: dict[str, CameraPreviewCard] = {}
        for idx, (cam_key, title, lr_key) in enumerate((
            ("cam_high", "正面相机", "top_head"),
            ("cam_left_wrist", "左腕相机", "hand_left"),
            ("cam_right_wrist", "右腕相机", "hand_right"),
        )):
            card = CameraPreviewCard(f"0{idx + 1}", title, lr_key.upper())
            self.cam_cards[cam_key] = card
            cam_row.addWidget(card, 1)
        right.addLayout(cam_row, 1)

        ctrl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("播放 / 暂停")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self._seek)
        self.frame_label = _make_label("0 / 0", "muted", mono=True)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.slider, 1)
        ctrl.addWidget(self.frame_label)
        right.addLayout(ctrl)

        split.addLayout(right, 2)
        root.addLayout(split, 1)

    def refresh(self) -> None:
        self._close_caps()
        self._datasets = scan_lerobot_v21_datasets(self.data_root)
        self.ds_list.clear()
        for item in self._datasets:
            gap = f"  ⚠ 缺口:{item['deleted']}" if item["deleted"] else ""
            col = f"  🛡 防碰:{item['collision_avoidances']}" if item.get("collision_avoidances") else ""
            self.ds_list.addItem(f"[{item['kind']}] {item['name']}  ({item['episodes']}段/{item['frames']}帧{gap}{col})")
        if self._datasets:
            self.ds_list.setCurrentRow(0)
        else:
            self.ep_table.setRowCount(0)
            self.timeline_box.setPlainText("暂无 LeRobot v2.1 数据集。")

    def _select_dataset(self, row: int) -> None:
        self._close_caps()
        if row < 0 or row >= len(self._datasets):
            return
        ds_path = Path(self._datasets[row]["path"])
        self._current_ds = ds_path
        episodes, timeline, _info = load_lerobot_v21_details(ds_path)
        self._current_episodes = episodes
        self.ep_table.setRowCount(len(episodes))
        for r, ep in enumerate(episodes):
            deleted_before = ep.get("deleted_attempts_before") or []
            gap_str = ", ".join(f"attempt#{x}" for x in deleted_before) if deleted_before else "无"
            after_inf = ep.get("after_inference_episode")
            ca = ep.get("collision_avoidance_before")
            if isinstance(ca, dict) and ca.get("sides"):
                ca_str = " / ".join(
                    f"{'左' if sk == 'left' else '右'} {sv.get('max_delta_deg', 0):.1f}° ({sv.get('reason', 'base')})"
                    for sk, sv in ca["sides"].items()
                )
            else:
                ca_str = "无"
            vals = [
                f"episode_{int(ep['episode_index']):06d}",
                str(ep.get("length", 0)),
                str(ep.get("attempt_index", ep["episode_index"])),
                gap_str,
                f"inf#{after_inf:06d}" if after_inf is not None else "—",
                str(ep.get("collector_name") or "—"),
                ca_str,
            ]
            for c, txt in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(txt)
                if c == 3 and deleted_before:
                    item.setForeground(QtGui.QColor("#f5b864"))
                elif c == 6 and ca_str != "无":
                    item.setForeground(QtGui.QColor("#7ad8ff"))
                self.ep_table.setItem(r, c, item)

        lines = []
        for ev in timeline:
            ts = str(ev.get("timestamp", ""))[-12:]
            ev_type = ev.get("event")
            if ev_type == "collision_avoidance":
                sides = ev.get("sides") or {}
                desc = " / ".join(
                    f"{'左' if sk == 'left' else '右'} {sv.get('max_delta_deg', 0):.1f}° ({sv.get('reason')}, 余量 {sv.get('initial_clearance_mm')}→{sv.get('final_clearance_mm')}mm)"
                    for sk, sv in sides.items()
                )
                after_s = f" (紧跟 inf#{int(ev['after_inference_episode']):06d})" if ev.get("after_inference_episode") is not None else ""
                lines.append(f"{ts}  【防碰撞微调】 {desc}{after_s}")
            elif ev_type == "merged_episode":
                segs = " + ".join(f"{'推理' if sg['source']=='inference' else '纠正'}#{sg['episode_index']}({sg['frames']}帧)" for sg in (ev.get("source_segments") or []))
                lines.append(f"{ts}  【连续拼接段】 episode_{int(ev.get('episode_index', 0)):06d} ({ev.get('frames', 0)}帧) = {segs}")
            elif ev_type == "dagger_deleted":
                lines.append(f"{ts}  【因删段切分】 attempt={ev.get('attempt_index')} 被删除，前后已断开")
            else:
                tag = "【已删除缺口】" if ev_type == "deleted" else "【已保存】"
                lines.append(
                    f"{ts}  {tag} attempt={ev.get('attempt_index')} -> episode_{int(ev.get('episode_index', 0)):06d} "
                    f"({ev.get('frames', 0)}帧)"
                )
        self.timeline_box.setPlainText("\n".join(lines) if lines else "无时间线事件记录。")
        if episodes:
            self.ep_table.selectRow(0)
            self._select_episode(0)

    def _close_caps(self) -> None:
        self._play_timer.stop()
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        self._total_frames = 0
        self._cur_frame = 0
        self.btn_play.setEnabled(False)
        self.slider.setEnabled(False)
        self.frame_label.setText("0 / 0")

    def _select_episode(self, row: int) -> None:
        self._close_caps()
        if self._current_ds is None or row < 0 or row >= len(self._current_episodes):
            return
        ep = self._current_episodes[row]
        ep_idx = int(ep["episode_index"])
        chunk_idx = ep_idx // 1000
        counts = []
        for cam_key, lr_key in LEROBOT_CAMERA_KEYS.items():
            vp = self._current_ds / f"videos/chunk-{chunk_idx:03d}/{lr_key}/episode_{ep_idx:06d}.mp4"
            if vp.is_file():
                cap = cv2.VideoCapture(str(vp))
                if cap.isOpened():
                    self._caps[cam_key] = cap
                    counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not self._caps:
            for c in self.cam_cards.values():
                c.clear_frame("视频尚未导出或不存在")
            return
        self._total_frames = max(1, min(counts)) if counts else 1
        self.slider.setRange(0, max(0, self._total_frames - 1))
        self.slider.setValue(0)
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self._render_frame(0)

    def _render_frame(self, idx: int) -> None:
        self._cur_frame = max(0, min(idx, max(0, self._total_frames - 1)))
        for cam_key, card in self.cam_cards.items():
            cap = self._caps.get(cam_key)
            if cap is None:
                card.clear_frame("无此路视频")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._cur_frame)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                card.set_bgr(bgr, age_s=None)
        self.slider.blockSignals(True)
        self.slider.setValue(self._cur_frame)
        self.slider.blockSignals(False)
        self.frame_label.setText(f"{self._cur_frame + 1} / {self._total_frames}")

    def _seek(self, value: int) -> None:
        self._play_timer.stop()
        self._render_frame(value)

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
        else:
            if self._cur_frame >= self._total_frames - 1:
                self._cur_frame = 0
            self._play_timer.start()

    def _step_play(self) -> None:
        if self._cur_frame + 1 >= self._total_frames:
            self._play_timer.stop()
            return
        self._render_frame(self._cur_frame + 1)

    def pause(self) -> None:
        self._play_timer.stop()

    def close_resources(self) -> None:
        self._close_caps()


class SDKDaggerMainWindow(QtWidgets.QMainWindow):
    """Pure-SDK DAgger desktop workbench with integrated zero-PTY USB dual-pedal standby loop."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        dataset_dir: str | Path | None = None,
        mock: bool = False,
        enable_pedal: bool = True,
        pedal_config: str | Path | None = None,
        pedal_device=None,
        auto_start: bool = False,
        collector_name: str | None = None,
        voice: VoiceAnnouncer | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.mock = bool(mock)
        self.dataset_dir = Path(dataset_dir or cfg["recording"].get("dataset_dir", str(ROOT / "data")))
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.enable_pedal = bool(enable_pedal)
        self.pedal_config = Path(
            pedal_config or cfg.get("pedal", {}).get("config_path", str(ROOT / "_runtime/foot-pedal.json"))
        )
        self._injected_pedal_device = pedal_device

        self.voice = voice if voice is not None else VoiceAnnouncer.from_config(cfg, mock=self.mock)
        self.active_app: SDKDaggerApp | None = None
        self.round_worker: _RoundWorker | None = None
        self.pedal_worker: _PedalWorker | None = None
        self.round_count = 0
        self._closing = False
        self._last_log_count = 0
        self._last_round_error: str = ""
        self._standby_events: list[tuple[str, str, str]] = []

        self.setWindowTitle("Kai0 纯 SDK DAgger 采集工作台（零 ROS · LeRobot v2.1）")
        self.resize(1460, 940)
        self.setMinimumSize(1120, 760)
        self.setStyleSheet(STYLE)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_page = QtWidgets.QWidget()
        self._build_live_tab(collector_name)
        self.tabs.addTab(self.live_page, "实时 DAgger 工作台")

        self.history_tab = LeRobotV21HistoryTab(self.dataset_dir)
        self.tabs.addTab(self.history_tab, "历史数据与缺口 (v2.1)")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        QtWidgets.QApplication.instance().installEventFilter(self)

        # 10 Hz UI refresh timer (strictly decoupled from 30Hz/200Hz control threads)
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(100)
        self._ui_timer.timeout.connect(self._refresh_ui)
        self._ui_timer.start()

        self._append_standby_log("工作台已启动（待机状态）。铺好衣物后点击「开始新一轮」或踩「左双踩」。")
        if self.enable_pedal:
            self._start_pedal_worker()
        else:
            self.pedal_badge.setText("● 未启用踏板（键盘/按钮模式）")

        self._refresh_ui()
        if auto_start:
            QtCore.QTimer.singleShot(50, self.start_new_round)

    def _build_live_tab(self, collector_name: str | None) -> None:
        layout = QtWidgets.QVBoxLayout(self.live_page)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(10)

        # Header
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(14)
        header.addWidget(_make_label("KAI0 SDK", "brand"))
        header.addWidget(_make_label("DAgger 实时采集工作台", "heading"))
        svc = self.cfg.get("startup", {}).get("model_service", "")
        srv = f"{self.cfg['server']['host']}:{self.cfg['server']['port']}"
        mode_tag = "MOCK 模拟" if self.mock else f"{svc or srv} · LeRobot v2.1"
        header.addWidget(_make_label(mode_tag, "muted"))
        header.addStretch()

        header.addWidget(_make_label("采集员", "muted"))
        self.collector_combo = QtWidgets.QComboBox()
        self.collector_combo.setMinimumWidth(130)
        self.collector_combo.addItem("未指定", None)
        for name in DEFAULT_COLLECTORS:
            self.collector_combo.addItem(name, name)
        if collector_name:
            idx = self.collector_combo.findData(collector_name)
            if idx < 0:
                self.collector_combo.addItem(collector_name, collector_name)
                idx = self.collector_combo.findData(collector_name)
            self.collector_combo.setCurrentIndex(idx)
        self.collector_combo.currentIndexChanged.connect(self._on_collector_changed)
        header.addWidget(self.collector_combo)

        self.voice_toggle = QtWidgets.QCheckBox("语音播报")
        self.voice_toggle.setChecked(self.voice.enabled)
        self.voice_toggle.toggled.connect(self.voice.set_enabled)
        header.addWidget(self.voice_toggle)

        self.btn_voice_test = QtWidgets.QPushButton("试音")
        self.btn_voice_test.setStyleSheet("padding: 5px 10px;")
        self.btn_voice_test.clicked.connect(self.voice.test_voice)
        header.addWidget(self.btn_voice_test)

        self.pedal_badge = _make_label("● 正在连接双踏板…", "muted")
        header.addWidget(self.pedal_badge)

        self.status_badge = QtWidgets.QLabel("● 待机中")
        self.status_badge.setMinimumWidth(210)
        self.status_badge.setAlignment(QtCore.Qt.AlignCenter)
        self.status_badge.setStyleSheet(
            "background: #1f2d3e; color: #a7bdd4; border-radius: 7px; padding: 8px 14px; font-weight: 700;"
        )
        header.addWidget(self.status_badge)
        layout.addLayout(header)

        # Guidance Banner
        self.guide_banner = QtWidgets.QLabel("就绪：点击「开始新一轮」或踩「左双踩」开始本轮采集。")
        self.guide_banner.setWordWrap(True)
        self.guide_banner.setStyleSheet(
            "background: #152334; border: 1px solid #2a435e; border-radius: 8px; "
            "color: #cfe4f7; padding: 9px 14px; font-size: 14px; font-weight: 600;"
        )
        layout.addWidget(self.guide_banner)

        # Metric Cards
        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(10)
        self.metrics: dict[str, QtWidgets.QLabel] = {}
        for key, title, default in (
            ("inference", "推理已存 (段/帧)", "0 段 / 0 帧"),
            ("dagger", "DAgger 已存 (段/帧/缺口)", "0 段 / 0 帧"),
            ("current", "当前段帧数 / 步数", "0 帧"),
            ("perf", "模型耗时 / 后台视频", "— · 队列 0"),
            ("elapsed", "本轮时长", "00:00:00"),
        ):
            card, box = _make_panel(compact=True)
            box.addWidget(_make_label(title, "muted"))
            val = _make_label(default, "metric", mono=True)
            self.metrics[key] = val
            box.addWidget(val)
            metrics_row.addWidget(card, 1)
        layout.addLayout(metrics_row)

        # 3-Camera Preview Row
        cam_row = QtWidgets.QHBoxLayout()
        cam_row.setSpacing(12)
        self.cam_cards: dict[str, CameraPreviewCard] = {}
        for idx, (cam_key, title, code) in enumerate((
            ("cam_high", "正面相机", "TOP_HEAD"),
            ("cam_left_wrist", "左腕相机", "HAND_LEFT"),
            ("cam_right_wrist", "右腕相机", "HAND_RIGHT"),
        )):
            card = CameraPreviewCard(f"0{idx + 1}", title, code)
            self.cam_cards[cam_key] = card
            cam_row.addWidget(card, 1)
        layout.addLayout(cam_row, 3)

        # Dual-Arm Readouts
        self.arm_bar = DualArmReadoutBar()
        layout.addWidget(self.arm_bar)

        # Action Controls + Legend Panel
        footer, fbox = _make_panel(compact=True)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)
        self.btn_start_round = QtWidgets.QPushButton("开始新一轮 (左双踩)")
        self.btn_start_round.setObjectName("start_round")
        self.btn_start_round.clicked.connect(self.start_new_round)

        self.btn_takeover = QtWidgets.QPushButton("人工接管 / 下段 (d / 左单)")
        self.btn_takeover.setObjectName("takeover")
        self.btn_takeover.clicked.connect(lambda: self.trigger_action("left"))

        self.btn_save = QtWidgets.QPushButton("保存纠正段 (s / 右单)")
        self.btn_save.setObjectName("save")
        self.btn_save.clicked.connect(lambda: self.trigger_action("save"))

        self.btn_delete = QtWidgets.QPushButton("撤销删除刚存段 (w)")
        self.btn_delete.setObjectName("delete_warn")
        self.btn_delete.clicked.connect(lambda: self.trigger_action("w"))

        self.btn_confirm = QtWidgets.QPushButton("确认保留")
        self.btn_confirm.clicked.connect(lambda: self.trigger_action("resume"))

        self.btn_resume = QtWidgets.QPushButton("恢复模型推理 (r / 左双)")
        self.btn_resume.setObjectName("resume")
        self.btn_resume.clicked.connect(lambda: self.trigger_action("resume"))

        self.btn_stop_round = QtWidgets.QPushButton("结束本轮进待机 (右双)")
        self.btn_stop_round.setObjectName("stop_round")
        self.btn_stop_round.clicked.connect(self.stop_current_round)

        for b in (
            self.btn_start_round, self.btn_takeover, self.btn_save,
            self.btn_delete, self.btn_confirm, self.btn_resume, self.btn_stop_round,
        ):
            btn_row.addWidget(b)
        fbox.addLayout(btn_row)

        legend = QtWidgets.QHBoxLayout()
        legend.setSpacing(8)
        for k, desc in (
            ("D / 左单", "人工接管 / 开始下段"),
            ("S / 右单", "保存纠正段"),
            ("W", "10秒内撤销删除"),
            ("R / 左双", "恢复模型 / 待机开新一轮"),
            ("右双", "结束本轮进待机"),
        ):
            legend.addWidget(_make_label(k, "key", mono=True))
            legend.addWidget(_make_label(desc, "muted"))
        legend.addStretch()
        self.dataset_label = _make_label("当前数据集：—", "muted", mono=True)
        legend.addWidget(self.dataset_label)
        fbox.addLayout(legend)
        layout.addWidget(footer)

        # Compact live event log
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(105)
        layout.addWidget(self.log_box)

    def _start_pedal_worker(self) -> None:
        dc_win = float((self.cfg.get("pedal") or {}).get("double_click_window_s", 0.8))
        self.pedal_worker = _PedalWorker(
            self.pedal_config, device=self._injected_pedal_device, double_click_window_s=dc_win
        )
        self.pedal_worker.gesture.connect(self._on_pedal_gesture)
        self.pedal_worker.status_changed.connect(self._on_pedal_status)
        self.pedal_worker.start()

    def _on_pedal_status(self, ok: bool, text: str) -> None:
        self.pedal_badge.setText(f"● {text}")
        self.pedal_badge.setStyleSheet(f'color: {"#72e5c3" if ok else "#efae79"}; font-weight: 600;')

    def _on_pedal_gesture(self, token: str) -> None:
        if self._closing:
            return
        # Standby state: ONLY left double ('resume') starts a new round; all others ignored!
        if self.active_app is None:
            if token == "resume":
                self._append_standby_log("[踏板] 待机状态下收到「左双踩」，正在开启新一轮采集……")
                self.start_new_round()
            return

        snap = self.active_app.get_ui_snapshot()
        if token == "stop":
            self._append_standby_log("[踏板] 收到「右双踩」，正在结束本轮并进入待机……")
            self.stop_current_round()
            return
        if snap["ready"] and token in ("left", "save", "resume", "delete"):
            self.active_app.handle_operator_action(token, source="pedal")

    def _on_collector_changed(self) -> None:
        name = self.collector_combo.currentData()
        if self.active_app is not None:
            self.active_app.set_collector_name(name)

    def _on_tab_changed(self, idx: int) -> None:
        if idx == 1:
            self.history_tab.refresh()
        else:
            self.history_tab.pause()

    def _append_standby_log(self, msg: str, level: str = "info") -> None:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._standby_events.append((ts, level, msg))
        self._standby_events = self._standby_events[-80:]
        self.log_box.appendPlainText(f"[{ts}] {msg}")

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
            if not isinstance(watched, QtWidgets.QWidget) or watched.window() is not self:
                return False
            if self.tabs.currentIndex() != 0:
                return False
            if isinstance(watched, QtWidgets.QLineEdit):
                return False
            if isinstance(watched, QtWidgets.QPlainTextEdit) and not watched.isReadOnly():
                return False
            if event.modifiers() & (QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier | QtCore.Qt.MetaModifier):
                return False
            key_map = {
                QtCore.Qt.Key_D: "d",
                QtCore.Qt.Key_S: "s",
                QtCore.Qt.Key_W: "w",
                QtCore.Qt.Key_R: "r",
                QtCore.Qt.Key_Space: " ",
            }
            act = key_map.get(event.key())
            if act is not None:
                self.trigger_action(act)
                return True
        return super().eventFilter(watched, event)

    def trigger_action(self, action: str) -> None:
        if self.active_app is None:
            return
        self.active_app.handle_operator_action(action, source="gui")
        self._refresh_ui()

    def start_new_round(self) -> None:
        if self.active_app is not None or (self.round_worker is not None and self.round_worker.isRunning()):
            return
        self._last_round_error = ""
        self.round_count += 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"cloth_sdk_{ts}_{os.getpid()}_r{self.round_count}"
        collector = self.collector_combo.currentData()
        app = SDKDaggerApp(
            self.cfg,
            dataset_name=dataset_name,
            dataset_dir=self.dataset_dir,
            mock=self.mock,
            enable_terminal_input=False,
            collector_name=collector,
            voice=self.voice,
        )
        self.active_app = app
        self._last_log_count = 0
        self._append_standby_log(f"=== 第 {self.round_count} 轮启动：{dataset_name} ===")
        self.round_worker = _RoundWorker(app)
        self.round_worker.finished_round.connect(self._on_round_finished)
        self.round_worker.start()
        self._refresh_ui()

    def stop_current_round(self) -> None:
        if self.active_app is None:
            return
        self.active_app.request_stop(signal.SIGINT)
        self._refresh_ui()

    def _on_round_finished(self, rc: int, err: str) -> None:
        ds_name = self.active_app.dataset_name if self.active_app is not None else ""
        self.active_app = None
        if self.pedal_worker is not None:
            self.pedal_worker.reset_gestures()
            self.pedal_worker.set_allow_short(True)
        if err:
            self._last_round_error = err
            self._append_standby_log(f"本轮异常结束（rc={rc}）：{err}", level="error")
        else:
            self._last_round_error = ""
            self._append_standby_log(f"本轮已安全保存并进入待机（{ds_name}）。铺好衣物后踩「左双踩」或点「开始新一轮」。")
            if not self._closing:
                self.voice.announce("standby")
        for c in self.cam_cards.values():
            c.clear_frame()
        self.arm_bar.update_from_sample(None, mode_label="待机", age_s=None)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        app = self.active_app
        if app is None:
            self.status_badge.setText("● 待机中（等待下一轮）")
            self.status_badge.setStyleSheet(
                "background: #1d2c3d; color: #9fc4e6; border-radius: 7px; padding: 8px 14px; font-weight: 700;"
            )
            if self._last_round_error:
                self.guide_banner.setText(f"⚠️ 上一轮启动/运行异常：{self._last_round_error}")
                self.guide_banner.setStyleSheet(
                    "background: #361b20; border: 1px solid #d65a66; border-radius: 8px; "
                    "color: #ffd5da; padding: 9px 14px; font-size: 14px; font-weight: 600;"
                )
            else:
                self.guide_banner.setText("待机中：重新铺好衣物并确认安全后，点击「开始新一轮」或踩「左双踩」。")
                self.guide_banner.setStyleSheet(
                    "background: #152334; border: 1px solid #2a435e; border-radius: 8px; "
                    "color: #cfe4f7; padding: 9px 14px; font-size: 14px; font-weight: 600;"
                )
            self.btn_start_round.setEnabled(True)
            self.btn_takeover.setEnabled(False)
            self.btn_save.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_delete.setText("撤销删除刚存段 (w)")
            self.btn_confirm.setEnabled(False)
            self.btn_resume.setEnabled(False)
            self.btn_stop_round.setEnabled(False)
            self.tabs.setTabEnabled(1, True)
            if self.pedal_worker is not None:
                self.pedal_worker.set_allow_short(True)
            return

        snap = app.get_ui_snapshot()
        mode = snap["mode"]
        ready = snap["ready"]
        stopping = snap["stopping"]
        in_handoff = snap["in_exit_handoff"]

        if self.pedal_worker is not None:
            self.pedal_worker.set_allow_short(ready and not stopping)

        # Status badge & guidance banner
        if stopping:
            badge_txt, fg, bg = "● 正在安全停止并落盘…", "#f7d6a8", "#3d2f1e"
        elif not ready:
            badge_txt, fg, bg = "● 正在初始化与归位…", "#9ec7f2", "#1c2f45"
        elif in_handoff:
            badge_txt, fg, bg = "● 双主臂 3 秒交接中…", "#ffe08a", "#42341a"
        elif mode == "INFERENCE":
            badge_txt, fg, bg = "● 模型自动运行中", "#72e5c3", "#15382e"
        elif mode in ("INFERENCE_TAIL", "TRANSITION_TO_DAGGER"):
            badge_txt, fg, bg = "● 主臂自动对齐中…", "#ffe08a", "#42341a"
        elif mode == "DAGGER_RECORDING":
            badge_txt, fg, bg = "● 人工纠正录制中", "#7ad8ff", "#15364a"
        elif mode == "CONFIRM_SAVE":
            rem = snap["confirm_remaining_s"]
            badge_txt, fg, bg = f"● 删除确认 {rem:.1f}s", "#ffb4bc", "#4d1f26"
        elif mode == "DAGGER_PAUSED":
            badge_txt, fg, bg = "● 示教已暂停（跟随暂停）", "#d2dceb", "#273548"
        else:
            badge_txt, fg, bg = f"● {mode}", "#cfe0f2", "#1f2d3e"

        self.status_badge.setText(badge_txt)
        self.status_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 7px; padding: 8px 14px; font-weight: 700;"
        )
        self.guide_banner.setText(snap["status_message"])
        border_col = "#d65a66" if mode == "CONFIRM_SAVE" else ("#c79a44" if in_handoff or mode == "TRANSITION_TO_DAGGER" else "#2a435e")
        self.guide_banner.setStyleSheet(
            f"background: #152334; border: 1px solid {border_col}; border-radius: 8px; "
            "color: #eef6ff; padding: 9px 14px; font-size: 14px; font-weight: 600;"
        )

        # Metrics
        self.metrics["inference"].setText(f"{snap['inference_episodes']} 段 / {snap['inference_saved_frames']} 帧")
        del_cnt = snap["dagger_deleted_attempts"]
        del_suffix = f" · 删{del_cnt}" if del_cnt else ""
        self.metrics["dagger"].setText(f"{snap['dagger_episodes']} 段 / {snap['dagger_saved_frames']} 帧{del_suffix}")
        if mode == "DAGGER_RECORDING":
            self.metrics["current"].setText(f"纠正 {snap['current_dagger_frames']} 帧")
        else:
            self.metrics["current"].setText(f"推理 {snap['current_inference_frames']} 帧 · #{snap['inference_step']}")
        ms = f"{snap['last_infer_ms']:.0f} ms" if snap["last_infer_ms"] is not None else "—"
        self.metrics["perf"].setText(f"{ms} · 视频队列 {snap['pending_video_jobs']}")
        el = int(max(0.0, snap["elapsed_s"]))
        self.metrics["elapsed"].setText(f"{el // 3600:02d}:{el // 60 % 60:02d}:{el % 60:02d}")
        self.dataset_label.setText(f"当前数据集：{snap['dataset_name']}")

        # Buttons
        self.btn_start_round.setEnabled(False)
        self.btn_takeover.setEnabled(ready and not stopping and mode in ("INFERENCE", "DAGGER_PAUSED"))
        self.btn_takeover.setText("开始下段纠正 (d / 左单)" if mode == "DAGGER_PAUSED" else "人工接管 (d / 左单)")
        self.btn_save.setEnabled(ready and not stopping and mode == "DAGGER_RECORDING")
        in_confirm = (mode == "CONFIRM_SAVE" and not stopping)
        self.btn_delete.setEnabled(in_confirm)
        self.btn_delete.setText(
            f"撤销删除本段 ({snap['confirm_remaining_s']:.1f}s · w)" if in_confirm else "撤销删除刚存段 (w)"
        )
        self.btn_confirm.setEnabled(in_confirm)
        self.btn_resume.setEnabled(ready and not stopping and not in_handoff and mode in ("DAGGER_RECORDING", "DAGGER_PAUSED"))
        self.btn_stop_round.setEnabled(not stopping)
        self.tabs.setTabEnabled(1, False)

        # Cameras + Arm readouts
        sample = snap["sample"]
        age_s = snap["sample_age_s"]
        if sample is not None:
            for cam_key in CAMERA_NAMES:
                bgr = sample.images_bgr.get(cam_key)
                if bgr is not None:
                    self.cam_cards[cam_key].set_bgr(bgr, age_s=age_s)
        arm_mode_lbl = (
            "主臂直通转发" if mode == "DAGGER_RECORDING"
            else ("跟随暂停" if mode in ("DAGGER_PAUSED", "CONFIRM_SAVE")
                  else ("主臂对齐/交接" if mode == "TRANSITION_TO_DAGGER" or in_handoff else "200Hz high_follow"))
        )
        self.arm_bar.update_from_sample(sample, mode_label=arm_mode_lbl, age_s=age_s)

        # Append new events from app
        events = snap["recent_events"]
        if len(events) > self._last_log_count:
            for ts, _lvl, msg in events[self._last_log_count:]:
                self.log_box.appendPlainText(f"[{ts}] {msg}")
            self._last_log_count = len(events)

    def closeEvent(self, event) -> None:
        self._closing = True
        self._ui_timer.stop()
        qapp = QtWidgets.QApplication.instance()
        if qapp is not None:
            qapp.removeEventFilter(self)
        self.history_tab.close_resources()
        if self.active_app is not None:
            self.active_app.request_stop(signal.SIGINT)
        if self.round_worker is not None and self.round_worker.isRunning():
            self.round_worker.wait(15000)
        if self.pedal_worker is not None and self.pedal_worker.isRunning():
            self.pedal_worker.stop()
            self.pedal_worker.wait(2000)
        self.voice.close()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/dagger_precision_5090.yaml"))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--collector-name", default=None)
    parser.add_argument("--pedal-config", default=None)
    parser.add_argument("--no-pedal", action="store_true", help="禁用 USB 双踏板（仅使用鼠标与键盘）")
    parser.add_argument("--auto-start", action="store_true", help="打开窗口后立即自动开始第一轮采集")
    parser.add_argument("--mock", action="store_true", help="无硬件全链路模拟模式")
    args = parser.parse_args(argv)

    if (
        sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        and os.environ.get("QT_QPA_PLATFORM") not in ("offscreen", "minimal")
    ):
        print("[错误] 未检测到桌面显示环境（DISPLAY/WAYLAND_DISPLAY）。请在本地桌面终端启动，或设置 DISPLAY=:1。", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    qapp.setApplicationName("Kai0 SDK DAgger Workbench")

    win = SDKDaggerMainWindow(
        cfg,
        dataset_dir=args.dataset_dir,
        mock=args.mock,
        enable_pedal=not args.no_pedal,
        pedal_config=args.pedal_config,
        auto_start=args.auto_start,
        collector_name=args.collector_name,
    )
    win.show()

    # Allow terminal Ctrl+C to cleanly close the Qt window & save active round
    def _handle_sigint(_sig, _frame):
        QtCore.QTimer.singleShot(0, win.close)

    signal.signal(signal.SIGINT, _handle_sigint)
    sig_timer = QtCore.QTimer()
    sig_timer.start(200)
    sig_timer.timeout.connect(lambda: None)

    return int(qapp.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
