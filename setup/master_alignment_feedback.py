#!/usr/bin/env python3
"""Verify both masters' six actual joints using fresh, passive raw CAN feedback.

This module never creates an SDK instance or writes CAN commands. The optional
publisher only repeats the caller's existing final ROS target while waiting.
Gripper width is diagnostic: its error never establishes or rejects alignment.
"""
from __future__ import annotations

import math
from numbers import Real
import time

from prepare_follower_mode import MAX_AGE_S, REQUIRED_IDS, sample_feedback


CHANNELS = {"left": "can_left_mas", "right": "can_right_mas"}
# Match the deployed official ROS driver's conversion, including its legacy pi.
ROS_TO_MDEG = 57324.840764
COMPONENTS = tuple(f"0x{identifier:03X}" for identifier in REQUIRED_IDS)


def finite_vector(values, length, label):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(label + f" must contain exactly {length} finite numbers")
    try:
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError
        output = [float(value) for value in values]
        if not all(math.isfinite(value) for value in output):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(label + f" must contain exactly {length} finite numbers") from exc
    return output


def alignment_targets(targets):
    """Validate before motion and return the exact six-joint wire targets."""
    if set(targets) != set(CHANNELS):
        raise ValueError("Both left and right targets are required")
    output = {}
    for side in CHANNELS:
        vector = finite_vector(targets[side], 7, side + " target")
        try:
            raw = [round(value * ROS_TO_MDEG) for value in vector[:6]]
        except (ValueError, OverflowError) as exc:
            raise ValueError(side + " joint target cannot be encoded") from exc
        if any(value < -(2**31) or value >= 2**31 for value in raw):
            raise ValueError(side + " joint target exceeds CAN integer range")
        output[side] = {"joint_mdeg": raw,
                        "gripper_um": round(min(0.08, max(0.0, vector[6])) * 1_000_000)}
    return output


def checked_feedback(row, *, now, since_timestamp, previous=None):
    """Check individual CAN component times; a fresh ROS/aggregate stamp cannot substitute."""
    if row.get("ctrl_mode") != 1:
        raise RuntimeError("master is not in CAN control mode")
    if row.get("arm_status") != 0 or row.get("err_code") != 0 or row.get("gripper_mode") != 0:
        raise RuntimeError("master arm fault or non-WIDTH gripper feedback")
    enabled = row.get("driver_enabled", [])
    if len(enabled) != 6 or not all(type(value) is bool and value for value in enabled):
        raise RuntimeError("all six master drivers must be enabled")
    foc = row.get("driver_foc_bytes", [])
    if (len(foc) != 6 or not all(type(value) is int and 0 <= value <= 255 for value in foc)
            or any(value & 0xBF for value in foc)):
        raise RuntimeError("master motor driver fault or incomplete driver feedback")
    joints = finite_vector(row.get("joint_mdeg"), 6, "actual master joints")
    stamp = float(row.get("timestamp_sec", float("nan")))
    if not math.isfinite(stamp) or not -0.05 <= now - stamp <= MAX_AGE_S:
        raise RuntimeError("stale or future raw feedback sample")
    ages = row.get("component_age_s", {})
    stamps = {}
    for component in COMPONENTS:
        age = float(ages.get(component, float("nan")))
        component_stamp = stamp - age
        if (not math.isfinite(age) or not -0.05 <= age <= MAX_AGE_S
                or not -0.05 <= now - component_stamp <= MAX_AGE_S
                or component_stamp < since_timestamp):
            raise RuntimeError("missing/stale/pre-command CAN component " + component)
        if previous is not None and component_stamp <= previous[component]:
            raise RuntimeError("CAN component did not advance: " + component)
        stamps[component] = component_stamp
    gripper = float(row.get("gripper_um", float("nan")))
    if not math.isfinite(gripper):
        raise RuntimeError("invalid gripper feedback")
    return joints, gripper, stamps


def wait_for_master_alignment(targets, *, since_timestamp, publish=None,
                              reader=sample_feedback, clock=time.monotonic,
                              wall=time.time, pause=time.sleep,
                              shutdown=lambda: False, timeout=8.0,
                              tolerance_mdeg=1000, stable_seconds=0.3,
                              completion_ready=None, guard=None, initial_stable_since=None):
    """Verify the hold while an optional independent preparation completes.

    guard may raise at every sample boundary. completion_ready can delay success
    after the stable duration, while fresh feedback and target publication keep
    running. initial_stable_since is an already-verified arrival time, not an
    arbitrary wait before feedback validation.
    """
    report = {"passed": False, "sides": {}, "samples": 0,
              "gripper_matching_required": False, "hardware_api_calls": 0,
              "tolerance_mdeg": tolerance_mdeg, "stable_seconds": stable_seconds}
    previous = {}
    start = clock()
    stable_since = initial_stable_since
    deadline = start + timeout
    try:
        converted = alignment_targets(targets)
        report["targets"] = converted
        if (not math.isfinite(since_timestamp) or not math.isfinite(timeout) or timeout <= 0
                or not math.isfinite(tolerance_mdeg) or tolerance_mdeg < 0
                or not math.isfinite(stable_seconds) or stable_seconds < 0
                or initial_stable_since is not None and (
                    not math.isfinite(initial_stable_since) or initial_stable_since > start)):
            raise ValueError("Invalid alignment timing or tolerance")
        # Each production reader is bounded to .12 s. Reserve the pair's read
        # window instead of starting another pair at the timeout boundary.
        while clock() + 0.25 < deadline:
            if guard is not None:
                guard()
            if shutdown():
                raise RuntimeError("ROS shutdown during master alignment")
            if publish is not None:
                publish()
            all_close = True
            for side, channel in CHANNELS.items():
                if guard is not None:
                    guard()
                row = reader(channel, since_timestamp=since_timestamp)
                joints, grip, stamps = checked_feedback(
                    row, now=wall(), since_timestamp=since_timestamp,
                    previous=previous.get(side))
                previous[side] = stamps
                delta = [actual - target for actual, target in zip(joints, converted[side]["joint_mdeg"])]
                close = all(abs(value) <= tolerance_mdeg for value in delta)
                all_close = all_close and close
                report["sides"][side] = {
                    "actual_mdeg": joints, "target_mdeg": converted[side]["joint_mdeg"],
                    "error_mdeg": delta, "error_degrees": [value / 1000.0 for value in delta],
                    "within_tolerance": close,
                    "gripper_error_um": grip - converted[side]["gripper_um"],
                    "component_stamps": stamps,
                }
                if guard is not None:
                    guard()
            report["samples"] += 1
            observed = clock()
            if shutdown():
                raise RuntimeError("ROS shutdown during master alignment")
            if observed >= deadline:
                break
            if all_close:
                if stable_since is None:
                    stable_since = observed
                report["stable_elapsed_s"] = observed - stable_since
                if observed - stable_since >= stable_seconds:
                    if guard is not None:
                        guard()
                    ready = completion_ready is None or completion_ready()
                    report['completion_ready'] = bool(ready)
                    if ready:
                        report["passed"] = True
                        return report
            else:
                stable_since = None
                report["stable_elapsed_s"] = 0.0
            pause(min(0.02, max(0.0, deadline - clock())))
        report["error"] = ("Joint hold remained stable but parallel preparation was not ready before timeout"
                           if (completion_ready is not None and report.get('completion_ready') is False
                               and report.get('stable_elapsed_s', 0.0) >= stable_seconds)
                           else "Both masters' six joints did not converge before the alignment timeout")
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["elapsed_s"] = clock() - start
    return report
