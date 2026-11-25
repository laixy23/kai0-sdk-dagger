#!/usr/bin/env python3
"""Passive read-only CAN watcher: identify each arm's USB-CAN serial via teach button or slight movement."""
from __future__ import annotations

from pathlib import Path
import select
import socket
import struct
import sys
import time


def list_up_can_interfaces() -> list[tuple[str, str]]:
    sys_net = Path("/sys/class/net")
    out: list[tuple[str, str]] = []
    for p in sorted(sys_net.iterdir()):
        try:
            if (p / "type").read_text().strip() != "280":
                continue
            oper = (p / "operstate").read_text().strip()
            flags = int((p / "flags").read_text().strip(), 16)
            if not (flags & 0x1):
                print(f"[跳过] {p.name} 当前为 DOWN（{oper}），请先用 sudo ip link set {p.name} up type can bitrate 1000000")
                continue
            dev = (p / "device").resolve()
            serial = "unknown"
            for anc in (dev, *dev.parents):
                if (anc / "serial").is_file():
                    serial = (anc / "serial").read_text().strip()
                    break
            out.append((p.name, serial))
        except OSError:
            continue
    return out


def main() -> int:
    ifaces = list_up_can_interfaces()
    if not ifaces:
        print("没有处于 UP 状态的 CAN 接口。")
        return 1

    socks: dict[int, tuple[socket.socket, str, str]] = {}
    state: dict[str, dict] = {}
    for name, serial in ifaces:
        s = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        s.bind((name,))
        s.setblocking(False)
        socks[s.fileno()] = (s, name, serial)
        state[name] = {"serial": serial, "ids": set(), "status": None, "joints": {}, "last_print": 0.0}

    print("=" * 72)
    print("只读 CAN 监听器已启动（绝不发送任何控制帧）。正在采样基线（1 秒）……")
    for name, serial in ifaces:
        print(f"  - {name:16s}  serial={serial}")
    print("=" * 72, flush=True)

    # 1. Baseline 1.0s
    baseline_end = time.monotonic() + 1.0
    while time.monotonic() < baseline_end:
        ready, _, _ = select.select(list(socks.keys()), [], [], 0.05)
        for fd in ready:
            s, name, _ = socks[fd]
            while True:
                try:
                    raw = s.recv(16)
                except BlockingIOError:
                    break
                cid, dlc, data = struct.unpack("=IB3x8s", raw)
                cid &= 0x1FFFFFFF
                st = state[name]
                st["ids"].add(cid)
                if cid == 0x2A1 and dlc >= 8:
                    st["status"] = data[:8]
                elif cid in (0x2A5, 0x2A6, 0x2A7, 0x155, 0x156, 0x157) and dlc >= 8:
                    st["joints"][cid] = struct.unpack(">ii", data[:8])

    print("\n基线帧类型识别：")
    for name, serial in ifaces:
        ids = state[name]["ids"]
        has_master = any(x in ids for x in (0x155, 0x156, 0x157, 0x159))
        has_slave = any(x in ids for x in (0x2A5, 0x2A6, 0x2A7, 0x2A8))
        role_hint = "主臂指令流(0x15x)" if has_master and not has_slave else (
            "从臂反馈流(0x2Ax)" if has_slave and not has_master else (
                "双流(0x15x+0x2Ax)" if has_master and has_slave else "静默/未发帧"
            )
        )
        st_hex = state[name]["status"].hex() if state[name]["status"] else "无"
        print(f"  {name:16s}  serial={serial}  [{role_hint}]  0x2A1={st_hex}")

    print("\n现在请逐个按一下每条臂的【示教键】（或轻轻晃动一下关节），终端会立刻显示触发的是哪个 CAN 与序列号。")
    print("按 Ctrl+C 结束。\n", flush=True)

    try:
        while True:
            ready, _, _ = select.select(list(socks.keys()), [], [], 0.1)
            now = time.monotonic()
            for fd in ready:
                s, name, serial = socks[fd]
                while True:
                    try:
                        raw = s.recv(16)
                    except BlockingIOError:
                        break
                    cid, dlc, data = struct.unpack("=IB3x8s", raw)
                    cid &= 0x1FFFFFFF
                    st = state[name]
                    if cid == 0x2A1 and dlc >= 8:
                        prev = st["status"]
                        cur = data[:8]
                        if prev is not None and cur != prev:
                            # Byte 0: ctrl_mode, Byte 1: arm_status, Byte 2: mode_feed, Byte 3: teach_status
                            print(
                                f">>> [状态/示教键变化] {name} (serial={serial})  "
                                f"0x2A1: {prev.hex()} -> {cur.hex()}  "
                                f"(ctrl={cur[0]}, teach={cur[3]})",
                                flush=True,
                            )
                            st["status"] = cur
                        elif prev is None:
                            st["status"] = cur
                    elif cid in (0x2A5, 0x2A6, 0x2A7, 0x155, 0x156, 0x157) and dlc >= 8:
                        pair = struct.unpack(">ii", data[:8])
                        prev_pair = st["joints"].get(cid)
                        if prev_pair is not None:
                            delta = max(abs(pair[0] - prev_pair[0]), abs(pair[1] - prev_pair[1]))
                            if delta >= 1500 and now - st["last_print"] > 0.6:  # >= 1.5 deg
                                kind = "主臂" if cid < 0x200 else "从臂"
                                print(
                                    f">>> [关节移动 {delta/1000:.1f}°] {name} (serial={serial})  [{kind} {hex(cid)}]",
                                    flush=True,
                                )
                                st["joints"][cid] = pair
                                st["last_print"] = now
                        else:
                            st["joints"][cid] = pair
    except KeyboardInterrupt:
        print("\n已结束监听。")
    finally:
        for s, _, _ in socks.values():
            s.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
