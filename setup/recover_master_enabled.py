#!/usr/bin/env python3
"""Recover an already enabled master without requesting Disable/Enable/Reset.

For an old silent FA role on supported firmware, one FC role command may restore
native feedback. FC changes the role/control mode; this code does not claim that
firmware torque is physically uninterrupted. No position or mode command is sent
until fresh native pose and all six actual enable bits have been validated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import time

from prepare_follower_enabled import checked_enabled_feedback
from prepare_follower_mode import make_sdk, sample_feedback
from recover_master_mode import active_controllers


CHANNELS = ("can_left_mas", "can_right_mas")
MISSING_PREFIX = "Missing required CAN feedback frames"
ALLOWED_COMMANDS = frozenset(("MasterSlaveConfig", "JointCtrl", "MotionCtrl_1", "MotionCtrl_2"))
FIRMWARE_PATTERN = re.compile(
    r"S-V(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)"
    r"(?:(?:\.(?P<patch>0|[1-9]\d*))(?:-(?P<build>\d+))?|-(?P<suffix>\d+))?")


def firmware_supports_enabled_role_change(version):
    """Recognize >=1.8.9, distinguishing 1.8-9 from a date/build suffix."""
    if type(version) is not str:
        return False
    match = FIRMWARE_PATTERN.fullmatch(version.strip())
    if match is None:
        return False
    major, minor = int(match["major"]), int(match["minor"])
    if (major, minor) > (1, 8):
        return True  # S-V1.9-0260716 has a build suffix, not patch 260716.
    if (major, minor) < (1, 8):
        return False
    patch = match["patch"]
    if patch is None:
        suffix = match["suffix"]
        if suffix is None or len(suffix) > 2 or (len(suffix) > 1 and suffix.startswith("0")):
            return False  # An ambiguous build/date must not impersonate patch >=9.
        patch = suffix
    return int(patch) >= 9


def cached_firmware_version(root, channel):
    """Read a cached receipt only; never query the arm or construct an SDK."""
    if channel not in CHANNELS:
        return None
    try:
        receipt = json.loads((Path(root) / "firmware-compat/provenance.json").read_text())
        value = receipt.get("firmwares", {}).get(channel)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return value.strip() if type(value) is str and value.strip() else None


def recover_enabled(channel, apply=False, *, firmware_version=None,
                    reader=sample_feedback, sdk_factory=make_sdk,
                    controller_reader=active_controllers, clock=time.monotonic,
                    wall=time.time, pause=time.sleep, step_timeout=3.0,
                    stable_seconds=0.3):
    """Return a bounded no-torque-toggle preparation report, with no fallback."""
    report = {
        "channel": channel, "apply": apply, "plan_only": not apply, "passed": False,
        "sdk_instances_created": 0, "sdk_command_log": [], "ownership_checks": 0,
        "torque_removal_requested": False, "enable_requested": False,
        "reset_requested": False, "gripper_commands_requested": 0,
        "initial_pose_requested": False, "fa_command_frames_used_as_pose": False,
        "cached_firmware_version": firmware_version,
        "firmware_version_source": "caller_supplied_cache" if firmware_version is not None else "unavailable",
        "firmware_version_live_verified": False,
        "bootstrap_required": False, "bootstrap_attempted": False,
        "drift_is_diagnostic_only": True,
        "bootstrap_risk": "FC exits the old teaching role into standby; no software torque-toggle is requested, but firmware transient torque continuity is not guaranteed",
        "plan": ["verify no active controller owns the hardware",
                 "CAN1 with fresh native pose and six enabled motors: read-only pass",
                 "silent FA only on recognized >=1.8.9 firmware: send one FC with zero offsets",
                 "before any target/mode write, require fresh post-FC native pose, six enabled motors and no faults",
                 "select a fresh native pose before the first target write; preload, exit teach, reseed, select CAN1 and reseed",
                 "passively verify enabled CAN1; record pose drift without an angular rejection threshold"],
    }
    sdk = None
    anchor = None
    previous = None
    drifts = []

    def ownership():
        owners = controller_reader()
        report["ownership_checks"] += 1
        report["active_controllers"] = owners
        if owners:
            raise RuntimeError("All arm/collector controllers must be stopped before enabled master recovery")

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
            raise RuntimeError("Command is forbidden in enabled master recovery")
        if name == "MasterSlaveConfig":
            if report["bootstrap_attempted"] or kwargs != dict(
                    linkage_config=0xFC, feedback_offset=0, ctrl_offset=0, linkage_offset=0) or args:
                raise RuntimeError("Only one zero-offset FC bootstrap is allowed")
        else:
            if anchor is None or not report.get("native_pose_verified"):
                raise RuntimeError("No native actual pose; target/mode command blocked")
            if name == "JointCtrl" and tuple(args) != tuple(anchor):
                raise RuntimeError("Only the fixed native holding target may be sent")
            if name == "MotionCtrl_1" and (args or kwargs != {"grag_teach_ctrl": 0x02}):
                raise RuntimeError("Only exit-teach is allowed")
            if name == "MotionCtrl_2" and (args or kwargs != dict(
                    ctrl_mode=1, move_mode=1, move_spd_rate_ctrl=10, is_mit_mode=0)):
                raise RuntimeError("Only normal CAN1 position control at speed 10 is allowed")
        ownership()
        if name == "MasterSlaveConfig":
            report["bootstrap_attempted"] = True
        report["sdk_command_log"].append({"method": name, "args": list(args),
                                           "kwargs": kwargs, "timestamp_sec": wall()})
        return getattr(sdk, name)(*args, **kwargs)

    def await_native(since, deadline):
        """Only missing frames are retried, without any further hardware write."""
        while clock() < deadline:
            try:
                result = read(since)
            except RuntimeError as exc:
                if not str(exc).startswith(MISSING_PREFIX):
                    raise
                report["feedback_wait_error"] = str(exc)
                pause(0.03)
                continue
            if clock() >= deadline:
                break
            return result
        raise TimeoutError("No complete fresh enabled native master feedback before timeout")

    try:
        if channel not in CHANNELS:
            raise RuntimeError("Only explicitly mapped physical master channels are allowed")
        if (not math.isfinite(step_timeout) or step_timeout <= 0
                or not math.isfinite(stable_seconds) or stable_seconds < 0):
            raise ValueError("Invalid enabled master preparation bounds")
        ownership()
        initial = None
        try:
            initial, anchor = read()
            report.update(initial_feedback=initial, holding_target_mdeg=anchor,
                          native_pose_verified=True, anchor_source="fresh native 0x2A5/0x2A6/0x2A7")
        except RuntimeError as exc:
            if not str(exc).startswith(MISSING_PREFIX):
                raise
            report.update(initial_feedback_missing=str(exc), bootstrap_required=True)
            if not firmware_supports_enabled_role_change(firmware_version):
                raise RuntimeError("Missing native master feedback requires a recognized cached firmware >=1.8.9; no role change requested")
        if initial is not None and initial["ctrl_mode"] == 1:
            report.update(passed=True, already_ready=True, final_feedback=initial,
                          observed_max_drift_mdeg=0)
            return report
        if not apply:
            report.update(passed=True, hardware_ready_not_verified=True)
            return report
        ownership()
        sdk = sdk_factory(channel)
        report["sdk_instances_created"] = 1
        sdk.ConnectPort(piper_init=False)
        if report["bootstrap_required"]:
            phase = wall()
            command("MasterSlaveConfig", linkage_config=0xFC, feedback_offset=0,
                    ctrl_offset=0, linkage_offset=0)
            initial, anchor = await_native(phase, clock() + step_timeout)
            report.update(post_fc_feedback=initial, holding_target_mdeg=anchor,
                          native_pose_verified=True, anchor_source="fresh post-FC native 0x2A5/0x2A6/0x2A7")
        else:
            current, current_joints = read()
            if current["ctrl_mode"] != initial["ctrl_mode"]:
                raise RuntimeError("Master mode changed while connecting; refusing stale preparation")
            # Select the freshly observed pose once, immediately before the
            # first write. The subsequent sequence keeps this same target.
            anchor = list(current_joints)
            report.update(pre_write_feedback=current, holding_target_mdeg=anchor,
                          anchor_source="fresh pre-write native 0x2A5/0x2A6/0x2A7",
                          holding_target_refreshed_before_first_write=True)
        # A role bootstrap can expose CAN1 while retaining an old target. Only
        # the initial fully observed CAN1 fast path may skip explicit reseeding.
        if report["bootstrap_required"] or initial["ctrl_mode"] != 1:
            command("JointCtrl", *anchor)
            command("MotionCtrl_1", grag_teach_ctrl=0x02)
            command("JointCtrl", *anchor)
            phase = wall()
            command("MotionCtrl_2", ctrl_mode=1, move_mode=1,
                    move_spd_rate_ctrl=10, is_mit_mode=0)
            command("JointCtrl", *anchor)
        else:
            phase = wall()
        deadline = clock() + step_timeout
        stable_since = None
        while clock() < deadline:
            try:
                snapshot, _ = read(phase)
            except RuntimeError as exc:
                if not str(exc).startswith(MISSING_PREFIX):
                    raise
                stable_since = None
                report["feedback_wait_error"] = str(exc)
                pause(0.03)
                continue
            if clock() >= deadline:
                break
            if snapshot["ctrl_mode"] == 1:
                now = clock()
                if stable_since is None:
                    stable_since = now
                report["stable_elapsed_s"] = now - stable_since
                if now - stable_since >= stable_seconds:
                    report.update(passed=True, final_feedback=snapshot,
                                  observed_max_drift_mdeg=max(drifts, default=0))
                    return report
            else:
                stable_since = None
        raise TimeoutError("Enabled master did not retain stable CAN1 feedback; no recovery fallback")
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
