#!/usr/bin/env python3
"""Prepare one supported follower for official ROS control; default is passive.

Applying this disables motor torque. Both --apply and physical support are
required. --allow-reset explicitly selects one reset, never an automatic retry.
No homing, gripper command, trajectory, ROS publisher, or official node edit.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.metadata
import json
from pathlib import Path
import struct
import time


CHANNELS = ("can_left_slave", "can_right_slave")
REQUIRED_IDS = (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8) + tuple(range(0x261, 0x267))
MAX_AGE_S = 0.2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def decode_feedback(frames, now, since_timestamp=0.0):
    """Decode independent raw frames; SDK aggregate timestamps are insufficient."""
    require(all(key in frames for key in REQUIRED_IDS), "Missing required CAN feedback frames")
    ages = {}
    for key in REQUIRED_IDS:
        data, stamp = frames[key]
        require(len(data) == 8, f"Invalid feedback length: 0x{key:03X}")
        age = now - stamp
        require(-0.05 <= age <= MAX_AGE_S and stamp >= since_timestamp,
                f"Stale/pre-command feedback: 0x{key:03X}, age={age:.4f}")
        ages[f"0x{key:03X}"] = age
    gripper_mode = frames[0x2A8][0][7]
    require(gripper_mode == 0, f"V3 gripper mode must be WIDTH(0), got {gripper_mode}; no automatic mode change")
    status = frames[0x2A1][0]
    errors = int.from_bytes(status[6:8], "big")
    foc = [frames[key][0][5] for key in range(0x261, 0x267)]
    require(status[1] == 0 and errors == 0, f"Arm fault: status={status[1]} err_code={errors}")
    require(not any(value & 0xBF for value in foc), f"Motor driver fault: {foc}")
    joints = [value for key in (0x2A5, 0x2A6, 0x2A7)
              for value in struct.unpack(">ii", frames[key][0])]
    grip_data, grip_stamp = frames[0x2A8]
    return {"ctrl_mode": status[0], "arm_status": status[1], "err_code": errors, "gripper_mode": gripper_mode,
            "joint_mdeg": joints, "gripper_um": struct.unpack(">i", grip_data[:4])[0],
            "gripper_effort_mnm": struct.unpack(">h", grip_data[4:6])[0],
            "gripper_foc_byte": grip_data[6], "gripper_enabled": bool(grip_data[6] & 0x40),
            "gripper_hex": grip_data.hex(), "gripper_source_time": grip_stamp,
            "driver_enabled": [bool(value & 0x40) for value in foc],
            "driver_foc_bytes": foc, "component_age_s": ages, "timestamp_sec": now}


def sample_feedback(channel, since_timestamp=0.0):
    import can  # No SDK import or instance is needed for passive inspection.
    frames = {}
    deadline = time.monotonic() + 0.12
    with can.Bus(channel=channel, interface="socketcan", receive_own_messages=False) as bus:
        while time.monotonic() < deadline:
            message = bus.recv(timeout=max(0, deadline - time.monotonic()))
            if message is None:
                continue
            require(not message.is_error_frame, "SocketCAN error frame observed")
            if message.is_remote_frame or message.arbitration_id not in REQUIRED_IDS:
                continue
            data = bytes(message.data)
            require(len(data) == 8, "Short Piper feedback frame")
            if message.arbitration_id == 0x2A1:
                require(data[1] == 0 and int.from_bytes(data[6:8], "big") == 0,
                        "Arm fault observed during sampling")
            elif 0x261 <= message.arbitration_id <= 0x266:
                require(not data[5] & 0xBF, "Driver fault observed during sampling")
            frames[message.arbitration_id] = (data, message.timestamp)
    return decode_feedback(frames, time.time(), since_timestamp)


def make_sdk(channel):
    require(importlib.metadata.version("piper_sdk") == "0.7.0a1", "Expected piper_sdk 0.7.0a1")
    from piper_sdk import C_PiperInterface_V3
    return C_PiperInterface_V3(can_name=channel)


def prepare(channel, apply=False, allow_reset=False, *, reader=sample_feedback,
            sdk_factory=make_sdk, clock=time.monotonic, wall=time.time,
            step_timeout=3.0, settle_span_mdeg=500, enable_drift_mdeg=2000):
    require(channel in CHANNELS, "Only explicitly mapped left/right follower channels are allowed")
    require(not allow_reset or apply, "--allow-reset requires --apply")
    report = {"channel": channel, "apply": apply, "reset_selected": allow_reset,
              "passed": False, "sdk_command_log": [], "sdk_instances_created": 0,
              "gripper_commands_requested": 0, "automatic_disable_after_enable": False,
              "plan": ["read fresh fault-free raw feedback", "disable and confirm all six drivers disabled",
                       "exit teach (optionally one explicitly selected reset)", "wait for supported pose to settle",
                       "preload measured target while disabled", "select normal CAN joint mode at speed 10",
                       "confirm fresh CAN mode while disabled", "reseed measured holding target",
                       "enable with bounded fresh feedback and pose checks"]}
    sdk = None
    enable_started = False

    def read(since=0.0):
        return reader(channel, since_timestamp=since)

    def command(name, *args, **kwargs):
        report["sdk_command_log"].append({"method": name, "args": list(args),
                                           "kwargs": kwargs, "timestamp_sec": wall()})
        return getattr(sdk, name)(*args, **kwargs)

    def disabled(snapshot):
        require(not any(snapshot["driver_enabled"]), "Expected all six drivers to remain disabled")

    def settled(since, mode=None):
        poses = deque(maxlen=3)
        deadline = clock() + step_timeout
        while clock() < deadline:
            snapshot = read(since)
            disabled(snapshot)
            require(mode is None or snapshot["ctrl_mode"] == mode,
                    "Follower changed mode before enable")
            poses.append(snapshot["joint_mdeg"])
            if len(poses) == 3 and all(max(row[j] for row in poses) - min(row[j] for row in poses)
                                       <= settle_span_mdeg for j in range(6)):
                return snapshot
        raise TimeoutError("Supported follower pose did not settle while disabled")

    try:
        report["initial_feedback"] = read()
        if not apply:
            report["passed"] = True
            report["hardware_control_writes"] = 0
            return report
        sdk = sdk_factory(channel)
        report["sdk_instances_created"] = 1
        # Default piper_init=True sends query writes; suppress that initialization.
        sdk.ConnectPort(piper_init=False)
        phase = wall()
        deadline = clock() + step_timeout
        while clock() < deadline:
            command("DisableArm", 7)
            snapshot = read(phase)
            if not any(snapshot["driver_enabled"]):
                break
        else:
            raise TimeoutError("All six drivers did not acknowledge disabled state")
        report["disabled_feedback"] = snapshot
        phase = wall()
        command("MotionCtrl_1", grag_teach_ctrl=0x02)
        if allow_reset:
            # This separate invocation deliberately selects one reset, not fallback.
            command("ResetPiper")
            phase = wall()
        snapshot = settled(phase)
        report["settled_feedback"] = snapshot
        target = list(snapshot["joint_mdeg"])
        command("JointCtrl", *target)
        phase = wall()
        command("MotionCtrl_2", ctrl_mode=0x01, move_mode=0x01,
                move_spd_rate_ctrl=10, is_mit_mode=0x00)
        deadline = clock() + step_timeout
        while clock() < deadline:
            snapshot = read(phase)
            disabled(snapshot)
            if snapshot["ctrl_mode"] == 0x01:
                break
        else:
            raise TimeoutError("No fresh CAN mode acknowledgment; left disabled, no automatic reset")
        # JointCtrl before joint mode may be ignored. Reread settled actual pose
        # and seed again after mode acknowledgment, before any EnableArm call.
        snapshot = settled(phase, mode=0x01)
        target = list(snapshot["joint_mdeg"])
        report["holding_target_mdeg"] = target
        command("JointCtrl", *target)
        phase = wall()
        enable_started = True
        deadline = clock() + step_timeout
        checks = []
        while clock() < deadline:
            command("JointCtrl", *target)
            command("EnableArm", 7)
            # Some firmware may ignore targets received while disabled. Reseed
            # immediately after enable, before the blocking feedback sample.
            command("JointCtrl", *target)
            snapshot = read(phase)
            require(snapshot["ctrl_mode"] == 0x01, "Follower lost CAN mode during enabling")
            delta = max(abs(a - b) for a, b in zip(snapshot["joint_mdeg"], target))
            checks.append(delta)
            require(delta <= enable_drift_mdeg, "Follower pose changed beyond the enable drift limit")
            if all(snapshot["driver_enabled"]):
                break
        else:
            raise TimeoutError("All six drivers did not acknowledge enabled state")
        for _ in range(3):
            snapshot = read(phase)
            require(snapshot["ctrl_mode"] == 0x01 and all(snapshot["driver_enabled"]),
                    "Follower did not retain enabled CAN mode")
            delta = max(abs(a - b) for a, b in zip(snapshot["joint_mdeg"], target))
            require(delta <= enable_drift_mdeg, "Follower drifted after enabling")
            checks.append(delta)
        report.update(final_feedback=snapshot, enable_pose_max_drift_mdeg=max(checks), passed=True)
    except (Exception, KeyboardInterrupt) as exc:
        report["error"] = repr(exc)
        report["enable_started"] = enable_started
        # Never remove torque automatically once enabling was requested. If a
        # fresh fault-free CAN snapshot is available, hold its measured pose.
        try:
            snapshot = read()
            report["failure_feedback"] = snapshot
            if sdk is not None and enable_started and snapshot["ctrl_mode"] == 0x01:
                command("JointCtrl", *snapshot["joint_mdeg"])
                report["failure_measured_hold_requested"] = True
        except Exception as hold_exc:
            report["failure_feedback_or_hold_error"] = repr(hold_exc)
    finally:
        if sdk is not None:
            try:
                sdk.DisconnectPort()
            except Exception as exc:
                report["disconnect_error"] = repr(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=CHANNELS)
    parser.add_argument("--apply", action="store_true", help="Disable/prepare/re-enable this supported follower")
    parser.add_argument("--allow-reset", action="store_true", help="Explicitly select one reset while disabled")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.allow_reset and not args.apply:
        parser.error("--allow-reset requires --apply")
    result = prepare(args.channel, args.apply, args.allow_reset)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
