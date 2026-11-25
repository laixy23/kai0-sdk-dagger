#!/usr/bin/env python3
"""Restore an already enabled follower without requesting motor torque removal.

Normal CAN control is passive. A former teaching state is reseeded at its fresh
measured pose before and immediately after the mode transition. This helper
never enables a disabled motor, resets the arm, or falls back to torque-off
recovery. It does not move to the session's initial pose; the collector owns that.
"""
from __future__ import annotations

import math
import time

from prepare_follower_mode import MAX_AGE_S, REQUIRED_IDS, make_sdk, sample_feedback


CHANNELS = ("can_left_slave", "can_right_slave")
ALLOWED_COMMANDS = frozenset(("JointCtrl", "MotionCtrl_1", "MotionCtrl_2"))


def checked_enabled_feedback(row, *, now, since_timestamp=0.0, previous=None,
                             require_enabled=True):
    """Check raw components; require all six motors enabled unless explicitly relaxed."""
    if (row.get("arm_status") != 0 or row.get("err_code") != 0
            or row.get("gripper_mode") != 0):
        raise RuntimeError("Follower fault or non-WIDTH gripper feedback")
    enabled = row.get("driver_enabled", [])
    if (len(enabled) != 6
            or (require_enabled and not all(value is True for value in enabled))):
        raise RuntimeError("Follower must already have all six motors enabled; automatic torque recovery is disabled")
    if not require_enabled and not all(type(value) is bool for value in enabled):
        raise RuntimeError("Invalid motor enable feedback")
    foc = row.get("driver_foc_bytes", [])
    if (len(foc) != 6 or not all(type(value) is int and 0 <= value <= 255 for value in foc)
            or any(value & 0xBF for value in foc)
            or (require_enabled and not all(value & 0x40 for value in foc))):
        raise RuntimeError("Incomplete follower driver feedback or motor fault")
    joints = row.get("joint_mdeg", [])
    if (len(joints) != 6
            or not all(type(value) in (int, float) and math.isfinite(value) for value in joints)):
        raise RuntimeError("Invalid measured follower pose")
    stamp = float(row.get("timestamp_sec", float("nan")))
    if not math.isfinite(stamp) or not -0.05 <= now - stamp <= MAX_AGE_S:
        raise RuntimeError("Stale or future follower feedback")
    ages = row.get("component_age_s", {})
    stamps = {}
    for key in REQUIRED_IDS:
        name = f"0x{key:03X}"
        age = float(ages.get(name, float("nan")))
        component_stamp = stamp - age
        if (not math.isfinite(age) or not -0.05 <= age <= MAX_AGE_S
                or not -0.05 <= now - component_stamp <= MAX_AGE_S
                or component_stamp < since_timestamp):
            raise RuntimeError("Missing/stale/pre-command follower component " + name)
        if previous is not None and component_stamp <= previous[name]:
            raise RuntimeError("Follower CAN component did not advance: " + name)
        stamps[name] = component_stamp
    if row.get("ctrl_mode") not in (0, 1, 2):
        raise RuntimeError("Unsupported follower control mode")
    return list(joints), stamps


def prepare_enabled(channel, apply=False, *, reader=sample_feedback, sdk_factory=make_sdk,
                    clock=time.monotonic, wall=time.time, step_timeout=3.0,
                    stable_seconds=0.3):
    """Prepare an enabled follower; failures never fall back to disable/enable."""
    report = {"channel": channel, "apply": apply, "passed": False,
              "sdk_instances_created": 0, "sdk_command_log": [],
              "torque_removal_requested": False, "enable_requested": False,
              "reset_requested": False, "gripper_commands_requested": 0,
              "initial_pose_requested": False, "drift_is_diagnostic_only": True,
              "plan": ["require fresh fault-free already-enabled follower feedback",
                       "CAN1: no SDK or hardware command",
                       "CAN2: preload measured pose before exit-teach and reseed immediately",
                       "select CAN joint mode and immediately reseed the same measured pose",
                       "require enabled CAN1 feedback; record drift without an angular rejection threshold"]}
    sdk = None
    previous = None
    anchor = None
    drifts = []

    def read(since=0.0):
        nonlocal previous
        row = reader(channel, since_timestamp=since)
        joints, stamps = checked_enabled_feedback(
            row, now=wall(), since_timestamp=since, previous=previous)
        previous = stamps
        if anchor is not None:
            drift = max(abs(actual - initial) for actual, initial in zip(joints, anchor))
            drifts.append(drift)
        return row, joints

    def command(name, *args, **kwargs):
        if name not in ALLOWED_COMMANDS:
            raise RuntimeError("Command is not allowed in enabled follower preparation")
        report["sdk_command_log"].append({"method": name, "args": list(args),
                                           "kwargs": kwargs, "timestamp_sec": wall()})
        return getattr(sdk, name)(*args, **kwargs)

    try:
        if channel not in CHANNELS:
            raise RuntimeError("Only mapped physical follower channels may use this helper")
        if (not math.isfinite(step_timeout) or step_timeout <= 0
                or not math.isfinite(stable_seconds) or stable_seconds < 0):
            raise ValueError("Invalid enabled follower preparation bounds")
        initial, anchor = read()
        report.update(initial_feedback=initial, holding_target_mdeg=anchor)
        if initial["ctrl_mode"] not in (1, 2):
            raise RuntimeError("Follower must be in CAN1 or teaching2; no automatic standby recovery")
        if initial["ctrl_mode"] == 1:
            report.update(passed=True, already_ready=True, final_feedback=initial,
                          hardware_control_writes=0, observed_max_drift_mdeg=0)
            return report
        if not apply:
            report.update(passed=True, plan_only=True, hardware_control_writes=0)
            return report
        sdk = sdk_factory(channel)
        report["sdk_instances_created"] = 1
        sdk.ConnectPort(piper_init=False)
        # The connection itself can take time. Choose the fresh actual pose
        # once before the first write, then keep that same holding target.
        current, current_joints = read()
        if current["ctrl_mode"] != 2:
            raise RuntimeError("Follower mode changed before enabled preparation")
        anchor = list(current_joints)
        report.update(pre_write_feedback=current, holding_target_mdeg=anchor,
                      holding_target_refreshed_before_first_write=True)
        command("JointCtrl", *anchor)
        command("MotionCtrl_1", grag_teach_ctrl=0x02)
        command("JointCtrl", *anchor)
        phase = wall()
        command("MotionCtrl_2", ctrl_mode=0x01, move_mode=0x01,
                move_spd_rate_ctrl=10, is_mit_mode=0x00)
        command("JointCtrl", *anchor)
        deadline = clock() + step_timeout
        stable_since = None
        while clock() < deadline:
            snapshot, _ = read(phase)
            if clock() >= deadline:
                raise TimeoutError("Enabled follower transition exceeded its time limit")
            if snapshot["ctrl_mode"] == 1:
                now = clock()
                if stable_since is None:
                    stable_since = now
                report["stable_elapsed_s"] = now - stable_since
                if now - stable_since >= stable_seconds:
                    report.update(passed=True, final_feedback=snapshot,
                                  observed_max_drift_mdeg=max(drifts, default=0))
                    break
                command("JointCtrl", *anchor)
            else:
                stable_since = None
        else:
            raise TimeoutError("Enabled follower did not retain stable CAN1 feedback; no disable/reset fallback")
    except (Exception, KeyboardInterrupt) as exc:
        report.update(error=repr(exc), interrupted=isinstance(exc, KeyboardInterrupt),
                      observed_max_drift_mdeg=max(drifts, default=0))
    finally:
        report["hardware_control_writes"] = len(report["sdk_command_log"])
        if sdk is not None:
            try:
                sdk.DisconnectPort()
            except Exception as exc:
                report.update(passed=False, disconnect_error=repr(exc))
    return report
