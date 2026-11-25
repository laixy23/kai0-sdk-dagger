#!/usr/bin/env python3
"""Explicit, supported-master recovery from an old FA teaching session.

Default: passive plan only. --apply briefly removes motor torque, so the master
must be physically supported. All ROS arm/collector processes must be gone.
Never home, command a gripper, or use FA control frames as measured pose.
One soft reset while disabled requires the separate explicit --allow-reset flag;
it is never selected automatically after a failed transition.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import re
import time

from prepare_follower_mode import make_sdk, require, sample_feedback


CHANNELS = ("can_left_mas", "can_right_mas")
CONTROLLER_SCRIPTS = frozenset(("piper_start_ms_node_new.py",
                                "agilex_openpi_dagger_collect.py", "run_first_live_cycle.py"))


def script_argument(argv):
    """Inspect the executable/script slot, not shell command text or -c strings."""
    if not argv:
        return None
    name = Path(argv[0]).name
    if name in CONTROLLER_SCRIPTS:
        return name
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name):
        return None
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in ("-c", "-m"):
            return None
        if item == "--":
            return Path(argv[index + 1]).name if index + 1 < len(argv) else None
        if item in ("-W", "-X"):
            index += 2
            continue
        if not item.startswith("-"):
            return Path(item).name
        index += 1
    return None


def active_controllers(proc_root=Path("/proc")):
    """Stopped controllers are still owners and must not be resumed here."""
    require(proc_root.is_dir(), "Cannot verify Linux controller processes")
    result = []
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            argv = [part.decode(errors="replace") for part in
                    (directory / "cmdline").read_bytes().split(b"\0") if part]
            state = next(line.split()[1] for line in (directory / "status").read_text().splitlines()
                         if line.startswith("State:"))
        except FileNotFoundError:
            continue  # Process exited while being inspected.
        except (OSError, StopIteration, IndexError) as exc:
            raise RuntimeError(f"Cannot inspect controller ownership: PID {directory.name}") from exc
        script = script_argument(argv)
        if state not in ("Z", "X", "x") and script in CONTROLLER_SCRIPTS:
            result.append({"pid": int(directory.name), "state": state, "script": script})
    return result


def recover(channel, apply=False, allow_reset=False, *, reader=sample_feedback, sdk_factory=make_sdk,
            controller_reader=active_controllers, clock=time.monotonic, wall=time.time,
            pause=time.sleep, step_timeout=3.0, settle_span_mdeg=500, enable_drift_mdeg=2000):
    require(channel in CHANNELS, "Only explicitly mapped left/right MASTER channels are allowed")
    require(not allow_reset or apply, "--allow-reset requires --apply")
    report = {"channel": channel, "apply": apply, "plan_only": not apply, "passed": False,
              "sdk_instances_created": 0, "sdk_command_log": [], "gripper_commands_requested": 0,
              "automatic_disable_after_enable": False, "reset_selected": allow_reset,
              "reset_commands_requested": int(allow_reset),
              "physical_support_required": True,
              "plan": ["require all ROS arm/collector processes gone and master physically supported",
                       "send three disable-all commands before FC role change",
                       "set FC role with zero offsets; immediately repeat disable until fresh six-driver disabled acknowledgment",
                       "if explicitly selected, reset once while disabled and require fresh disabled feedback afterward",
                       "exit teach and wait for three fresh settled disabled actual poses",
                       "seed measured actual target; select CAN joint mode at speed 10",
                       "confirm fresh disabled CAN mode; reseed settled actual holding target",
                       "enable with immediate reseeding, fresh six-driver enable acknowledgment and bounded drift"]}
    sdk = None
    enable_started = False

    def read(since=0.0):
        return reader(channel, since_timestamp=since)

    def command(name, *args, **kwargs):
        report["sdk_command_log"].append({"method": name, "args": list(args),
                                           "kwargs": kwargs, "timestamp_sec": wall()})
        return getattr(sdk, name)(*args, **kwargs)

    def disabled(snapshot):
        require(len(snapshot["driver_enabled"]) == 6 and not any(snapshot["driver_enabled"]),
                "Expected fresh all-six-driver disabled acknowledgment")

    def settled(since, mode=None):
        poses = deque(maxlen=3)
        deadline = clock() + step_timeout
        while clock() < deadline:
            snapshot = read(since)
            disabled(snapshot)
            require(mode is None or snapshot["ctrl_mode"] == mode,
                    "Master changed mode before enabling")
            poses.append(snapshot["joint_mdeg"])
            if len(poses) == 3 and all(max(row[j] for row in poses) - min(row[j] for row in poses)
                                      <= settle_span_mdeg for j in range(6)):
                return snapshot
        raise TimeoutError("Supported master actual pose did not settle while disabled")

    try:
        report["active_controllers"] = controller_reader()
        if not apply:
            try:
                report["initial_feedback"] = read()
            except Exception as exc:
                report["initial_feedback_error"] = repr(exc)
            report.update(passed=True, hardware_control_writes=0, hardware_ready_not_verified=True)
            return report
        require(not report["active_controllers"], "ROS arm/collector processes must all be stopped before recovery")
        # FA teaching-input roles can be silent. Only missing frames permit the
        # explicit bootstrap; a decoded fault or wrong gripper unit aborts it.
        try:
            report["initial_feedback"] = read()
        except RuntimeError as exc:
            require(str(exc).startswith("Missing required CAN feedback frames"),
                    "Initial feedback unsafe; refusing bootstrap: " + str(exc))
            report["initial_feedback_missing"] = repr(exc)
        sdk = sdk_factory(channel)
        report["sdk_instances_created"] = 1
        sdk.ConnectPort(piper_init=False)
        for _ in range(3):
            command("DisableArm", 7)
            pause(0.03)
        phase = wall()
        command("MasterSlaveConfig", linkage_config=0xFC, feedback_offset=0,
                ctrl_offset=0, linkage_offset=0)
        deadline = clock() + step_timeout
        while clock() < deadline:
            command("DisableArm", 7)  # First command after FC; never enable blindly.
            try:
                snapshot = read(phase)
            except RuntimeError as exc:
                if not str(exc).startswith("Missing required CAN feedback frames"):
                    raise
                report["disabled_feedback_wait_error"] = repr(exc)
                pause(0.03)
                continue
            if len(snapshot["driver_enabled"]) == 6 and not any(snapshot["driver_enabled"]):
                break
        else:
            raise TimeoutError("No fresh all-six-driver disabled acknowledgment after FC; no teach/mode/target/enable commands sent")
        report["disabled_feedback"] = snapshot
        if allow_reset:
            # Explicit operator-selected branch, never a retry or fallback.
            # Reset may change the actual supported pose and discard the target.
            command("ResetPiper")
            phase = wall()
            deadline = clock() + step_timeout
            while clock() < deadline:
                try:
                    snapshot = read(phase)
                except RuntimeError as exc:
                    if not str(exc).startswith("Missing required CAN feedback frames"):
                        raise
                    report["reset_feedback_wait_error"] = repr(exc)
                    pause(0.03)
                    continue
                disabled(snapshot)
                break
            else:
                raise TimeoutError("No fresh disabled feedback after explicit reset; no enable requested")
            report["reset_disabled_feedback"] = snapshot
        phase = wall()
        command("MotionCtrl_1", grag_teach_ctrl=0x02)
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
            raise TimeoutError("No fresh disabled CAN mode acknowledgment; master left disabled")
        snapshot = settled(phase, mode=0x01)
        target = list(snapshot["joint_mdeg"])
        report["holding_target_mdeg"] = target
        command("JointCtrl", *target)
        phase = wall()
        enable_started = True
        deadline = clock() + step_timeout
        drifts = []
        while clock() < deadline:
            command("JointCtrl", *target)
            command("EnableArm", 7)
            command("JointCtrl", *target)
            snapshot = read(phase)
            require(snapshot["ctrl_mode"] == 1, "Master lost CAN mode while enabling")
            drift = max(abs(a - b) for a, b in zip(snapshot["joint_mdeg"], target))
            drifts.append(drift)
            require(drift <= enable_drift_mdeg, "Master enable drift exceeds the limit")
            if len(snapshot["driver_enabled"]) == 6 and all(snapshot["driver_enabled"]):
                break
        else:
            raise TimeoutError("All six master drivers did not acknowledge enabled state")
        for _ in range(3):
            snapshot = read(phase)
            require(snapshot["ctrl_mode"] == 1 and len(snapshot["driver_enabled"]) == 6
                    and all(snapshot["driver_enabled"]), "Master did not retain enabled CAN mode")
            drift = max(abs(a - b) for a, b in zip(snapshot["joint_mdeg"], target))
            drifts.append(drift)
            require(drift <= enable_drift_mdeg, "Master drifted after enabling")
        report.update(passed=True, final_feedback=snapshot, enable_pose_max_drift_mdeg=max(drifts))
    except (Exception, KeyboardInterrupt) as exc:
        report.update(error=repr(exc), enable_started=enable_started)
        try:
            snapshot = read()
            report["failure_feedback"] = snapshot
            if sdk is not None and enable_started and snapshot["ctrl_mode"] == 1:
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
    parser.add_argument("--apply", action="store_true", help="Explicitly recover this physically supported master")
    parser.add_argument("--allow-reset", action="store_true", help="Explicitly select one soft reset while all six drivers are disabled; requires --apply")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.allow_reset and not args.apply:
        parser.error("--allow-reset requires --apply")
    report = recover(args.channel, apply=args.apply, allow_reset=args.allow_reset)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
