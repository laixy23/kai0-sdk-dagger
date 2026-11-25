#!/usr/bin/env python3
"""Zero-ROS SDK DAgger startup checker and environment preparer.

Checks/prepares:
  1. 4-channel CAN naming & UP state by USB serial (can_left/right_slave, can_left/right_mas)
  2. Optional USB two-pedal calibration & evdev read access
  3. Local model systemd service & WebSocket /healthz + warmup (referencing inference-origin)
  4. Releases legacy ROS camera service (kai0-official-cameras.service) if active, and verifies 3 RealSense serials via pyrealsense2
  5. Safe current-pose CAN1 arm preparation (reusing verified setup/prepare_follower_enabled & recover_master_enabled)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "setup"))
sys.path.insert(0, str(ROOT / "operator"))

from sdk_dagger.policy import ensure_inference_origin_paths


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _run(argv: list[str], *, timeout: float = 20.0, capture: bool = True) -> str:
    result = subprocess.run(argv, text=True, capture_output=capture, timeout=timeout)
    if result.returncode:
        detail = ((result.stderr or result.stdout or "").strip() if capture else "")
        raise RuntimeError(detail or f"命令执行失败，退出码 {result.returncode}: {' '.join(argv)}")
    return result.stdout or ""


def _sudo(argv: list[str]) -> None:
    if os.geteuid() != 0:
        if not sys.stdin.isatty():
            raise RuntimeError("设备配置需要 sudo，请用交互终端运行（远程请使用 ssh -t）。")
        argv = ["sudo", *argv]
    result = subprocess.run(argv, stdout=subprocess.PIPE, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"管理员设备配置未完成，退出码 {result.returncode}")


def service_state(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", name, "-p", "ActiveState", "--value"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def check_can_status() -> dict:
    script = ROOT / "setup/activate_can_by_serial.py"
    spec = importlib.util.spec_from_file_location("sdk_dagger_can_setup", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    links = json.loads(_run(["/usr/sbin/ip", "-j", "-details", "link", "show"]))
    identities = {
        row["ifname"]: module.usb_identity(row["ifname"])
        for row in links
        if row.get("linkinfo", {}).get("info_kind") == "can"
    }
    plan = module.build_plan(links, identities, allow_up_rename=True)
    return {"ready": all(row["skip"] for row in plan), "plan": plan}


def check_pedal_status(pedal_config: Path) -> dict:
    from foot_pedal_device import discover_interfaces, load_config as load_pedal_cfg
    if not pedal_config.is_file():
        raise RuntimeError(f"缺少踏板校准文件：{pedal_config}")
    calibrated = load_pedal_cfg(pedal_config)
    bindings = calibrated["bindings"]
    interfaces = discover_interfaces(**calibrated["device"])
    if not {row["interface"] for row in bindings.values()}.issubset({row["interface"] for row in interfaces}):
        raise RuntimeError("踏板或接口标识与校准记录不一致，请检查 USB 接线。")
    return {
        "bindings": list(bindings.keys()),
        "readable": all(os.access(row["path"], os.R_OK) for row in interfaces),
    }


def check_model_health(cfg: dict, *, start_if_needed: bool, warmup: bool) -> dict:
    startup = cfg.get("startup") or {}
    service = startup.get("model_service")
    host = str(cfg["server"]["host"])
    port = int(cfg["server"]["port"])
    exclusions = [x for x in os.environ.get("NO_PROXY", "").split(",") if x]
    for item in ("127.0.0.1", "localhost", host):
        if item not in exclusions:
            exclusions.append(item)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(exclusions)

    state = service_state(service) if service else "external"
    if start_if_needed and service and state != "active":
        print(f"[准备] 启动模型服务：{service}……", flush=True)
        subprocess.run(["systemctl", "--user", "start", service], check=True)

    url = f"http://{host}:{port}/healthz"
    deadline = time.monotonic() + (float(startup.get("model_timeout_s", 180)) if start_if_needed else 5.0)
    healthy = False
    while True:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            break
        if start_if_needed and service and service_state(service) in ("failed", "inactive"):
            raise RuntimeError(f"模型服务已退出，请排查：journalctl --user -u {service}")
        time.sleep(1.0)

    if not healthy:
        raise RuntimeError(f"模型服务未就绪：{url}（服务状态：{service_state(service) if service else 'external'}）")

    if warmup:
        ensure_inference_origin_paths(cfg["runtime"]["inference_origin_root"])
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client = WebsocketClientPolicy(host, port)
        try:
            img_size = int(cfg["inference"].get("image_size", 224))
            result = client.infer({
                "images": {
                    key: np.zeros((3, img_size, img_size), dtype=np.uint8)
                    for key in ("top_head", "hand_left", "hand_right")
                },
                "state": np.zeros(14, dtype=np.float32),
                "prompt": str(cfg["inference"]["prompt"]),
                "num_steps": int(cfg["inference"].get("num_denoising_steps", 10)),
            })
            actions = np.asarray(result["actions"])
            expected_horizon = int(cfg["inference"]["chunk_size"])
            if actions.shape != (expected_horizon, 14) or not np.isfinite(actions).all():
                raise RuntimeError(f"模型预热返回非法动作：{actions.shape}")
        finally:
            ws = getattr(client, "_ws", None)
            if ws is not None:
                ws.close()

    return {"service": service, "state": service_state(service) if service else "external",
            "endpoint": f"ws://{host}:{port}", "healthy": True, "warmed_up": warmup}


def check_realsense_cameras(cfg: dict, *, release_ros_service: bool) -> dict:
    import pyrealsense2 as rs
    startup = cfg.get("startup") or {}
    ros_cam_service = startup.get("release_ros_camera_service")
    ros_cam_state = service_state(ros_cam_service) if ros_cam_service else "none"
    released = False

    if release_ros_service and ros_cam_service and ros_cam_state in ("active", "activating"):
        print(f"[准备] 释放旧 ROS 相机服务占用：{ros_cam_service}……", flush=True)
        subprocess.run(["systemctl", "--user", "stop", ros_cam_service], check=True)
        time.sleep(0.5)
        ros_cam_state = service_state(ros_cam_service)
        released = True

    ctx = rs.context()
    actual = sorted({dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()})
    expected = {str(cfg["camera"][side]["serial"]) for side in ("front", "left", "right")}
    missing = sorted(expected - set(actual))
    if missing:
        raise RuntimeError(f"未检测到全部 RealSense 相机，缺失序列号：{missing}（当前可见：{actual}）")

    return {
        "expected": sorted(expected),
        "actual": actual,
        "ros_camera_service": ros_cam_service,
        "ros_camera_state": ros_cam_state,
        "ros_camera_released": released,
    }


def _enable_clean_standby_arm_in_place(iface: str, initial_row: dict, *, timeout_s: float = 4.0) -> dict:
    """Enable a cleanly booted standby arm (ctrl_mode=0, zero faults) anchored at its current pose."""
    from prepare_follower_mode import make_sdk, sample_feedback

    joints = [int(v) for v in initial_row["joint_mdeg"][:6]]
    sdk = make_sdk(iface)
    try:
        sdk.ConnectPort()
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            sdk.MotionCtrl_2(0x01, 0x01, 30, 0x00)
            sdk.JointCtrl(*joints)
            sdk.EnableArm(7)
            time.sleep(0.15)
            try:
                row = sample_feedback(iface, since_timestamp=time.time() - 0.3)
            except Exception:
                continue
            if (
                row.get("ctrl_mode") == 1
                and row.get("arm_status") == 0
                and all(row.get("driver_enabled", []))
            ):
                sdk.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                sdk.JointCtrl(*joints)
                return {"interface": iface, "action": "enabled_clean_standby_in_place", "passed": True}
        return {"interface": iface, "passed": False, "error": "等待六轴使能与 CAN1 确认超时"}
    finally:
        try:
            sdk.DisconnectPort()
        except Exception:
            pass


def prepare_four_arms_current_pose(cfg: dict) -> dict:
    """Safely ensure all 4 arms are in CAN1 and enabled at their current measured pose."""
    from prepare_follower_mode import sample_feedback
    from prepare_follower_enabled import prepare_enabled as prepare_follower_arm
    from recover_master_enabled import cached_firmware_version, recover_enabled as recover_master_arm

    can_cfg = cfg["can"]
    arms = [
        ("follower_left", can_cfg["follower_left"]["name"], "follower"),
        ("follower_right", can_cfg["follower_right"]["name"], "follower"),
        ("master_left", can_cfg["master_left"]["name"], "master"),
        ("master_right", can_cfg["master_right"]["name"], "master"),
    ]
    reports = {}
    print("[准备] 正在按当前实测姿态核对/使能四条机械臂（不先失能或归零）……", flush=True)
    for role, iface, kind in arms:
        row = None
        try:
            row = sample_feedback(iface, since_timestamp=time.time() - 0.5)
        except Exception:
            row = None
        if row is not None and row.get("ctrl_mode") == 1 and all(row.get("driver_enabled", [])) and row.get("arm_status") == 0:
            reports[role] = {"interface": iface, "action": "already_enabled_can1"}
            continue
        # Freshly power-cycled arm in clean standby (ctrl_mode==0, no faults, motors disabled):
        # safely anchor target to current measured pose and enable in-place (no jump, no homing).
        if (
            row is not None
            and row.get("ctrl_mode") in (0, 1)
            and row.get("arm_status") == 0
            and not any(row.get("driver_enabled", []))
            and all(int(st) == 0 for st in (row.get("driver_status") or [0]))
        ):
            print(f"[上电使能] {iface}：检测到刚上电待机态（无故障），正在原地锚定当前姿态并使能六轴……", flush=True)
            rep = _enable_clean_standby_arm_in_place(iface, row)
            if not rep.get("passed"):
                raise RuntimeError(f"{iface} 原地使能失败：{rep.get('error') or rep}")
            reports[role] = rep
            continue
        print(f"[恢复] {iface}：按当前实测姿态恢复 CAN1 并使能……", flush=True)
        if kind == "follower":
            rep = prepare_follower_arm(iface, apply=True)
        else:
            fw = cached_firmware_version(ROOT, iface)
            rep = recover_master_arm(iface, apply=True, firmware_version=fw)
        if not rep.get("passed"):
            raise RuntimeError(f"{iface} 准备失败：{rep.get('error') or rep}")
        reports[role] = {"interface": iface, "action": "recovered_at_current_pose"}
    print("[通过] 四条机械臂 CAN1 与六轴使能状态已就绪。", flush=True)
    return reports


def prepare_runtime(config_path: str | Path, *, check_only: bool = False,
                    with_pedal: bool = False, pedal_config: str | Path | None = None,
                    warmup_model: bool = False) -> dict:
    cfg = load_config(config_path)
    can = check_can_status()
    pedal = None
    if with_pedal:
        p_cfg = Path(pedal_config or cfg["pedal"]["config_path"])
        pedal = check_pedal_status(p_cfg)

    if check_only:
        model = check_model_health(cfg, start_if_needed=False, warmup=False)
        cams = check_realsense_cameras(cfg, release_ros_service=False)
        if not can["ready"]:
            raise RuntimeError("四路 CAN 尚未全部按序列号就绪；正式启动时会自动配置并按需提示 sudo 密码。")
        if pedal is not None and not pedal["readable"]:
            raise RuntimeError("当前用户缺少踏板读取权限；正式启动时会自动配置并按需提示 sudo 密码。")
        return {"check_only": True, "can_ready": can["ready"], "pedal": pedal,
                "model": model, "cameras": cams}

    if not can["ready"]:
        print("[准备] 按已确认的 USB 序列号配置四路 CAN……", flush=True)
        with socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW):
            pass
        _sudo(["/usr/bin/python3", str(ROOT / "setup/activate_can_by_serial.py"),
               "--apply", "--allow-up-rename"])
        can = check_can_status()
        if not can["ready"]:
            raise RuntimeError("四路 CAN 配置后验证未通过。")

    if pedal is not None and not pedal["readable"]:
        print("[准备] 配置双踏板读取权限……", flush=True)
        _sudo(["/usr/bin/python3", str(ROOT / "operator/setup_foot_pedal_access.py"), "--apply"])
        pedal = check_pedal_status(Path(pedal_config or cfg["pedal"]["config_path"]))
        if not pedal["readable"]:
            raise RuntimeError("配置后仍无法读取双踏板。")

    model = check_model_health(cfg, start_if_needed=True, warmup=warmup_model)
    cams = check_realsense_cameras(cfg, release_ros_service=True)
    return {"check_only": False, "can_ready": can["ready"], "pedal": pedal,
            "model": model, "cameras": cams}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--with-pedal", action="store_true")
    parser.add_argument("--pedal-config")
    parser.add_argument("--warmup-model", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_runtime(
            args.config,
            check_only=args.check_only,
            with_pedal=args.with_pedal,
            pedal_config=args.pedal_config,
            warmup_model=args.warmup_model,
        )
        if args.check_only:
            ros_cam = report["cameras"].get("ros_camera_state")
            note = ""
            if ros_cam in ("active", "activating"):
                note = f"（检测到旧 ROS 相机服务 {report['cameras']['ros_camera_service']} 运行中，正式启动 SDK 版时会自动停止以释放相机）"
            print(f"[通过] 纯 SDK 启动依赖检查完成（CAN/模型/三路 RealSense 相机均正常）{note}。", flush=True)
        else:
            print("[通过] CAN、模型服务与三路 RealSense SDK 相机已全部就绪（零 ROS）。", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[阻塞] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
