"""Pure-SDK 3-camera + 4-arm hardware interface (zero ROS) and Mock rig for testing.

Reuses inference-origin's RealSenseRig & PiperDualArm for cameras and follower arms,
and kai0-official-dagger's pure-Python SocketCAN master alignment & NativeMasterReader
for safe FC/FA master-follower teaching handoff.
"""
from __future__ import annotations

from collections import deque
import datetime
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "setup"))

from sdk_dagger.collision import HandleCollisionConfig, find_minimal_safe_pose, plan_safe_master_trajectory, project_pose_to_handle_safe
from sdk_dagger.policy import ensure_inference_origin_paths
from sdk_dagger.storage import CAMERA_NAMES, FrameSample

RAD_TO_MDEG = 1000.0 * 180.0 / math.pi
MDEG_TO_RAD = (math.pi / 180.0) / 1000.0
ARM_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)


def _read_piper_arm_full_state(piper_arm) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read (qpos7, qvel7, effort7, timestamp) directly from piper_sdk without ROS."""
    iface = piper_arm._piper
    joint_msg = iface.GetArmJointMsgs()
    high_spd = iface.GetArmHighSpdInfoMsgs()
    gripper_msg = iface.GetArmGripperMsgs()

    js = joint_msg.joint_state
    gs = gripper_msg.gripper_state
    qpos = np.asarray([
        js.joint_1 * MDEG_TO_RAD,
        js.joint_2 * MDEG_TO_RAD,
        js.joint_3 * MDEG_TO_RAD,
        js.joint_4 * MDEG_TO_RAD,
        js.joint_5 * MDEG_TO_RAD,
        js.joint_6 * MDEG_TO_RAD,
        gs.grippers_angle / 1_000_000.0,
    ], dtype=np.float32)
    qvel = np.asarray([
        high_spd.motor_1.motor_speed / 1000.0,
        high_spd.motor_2.motor_speed / 1000.0,
        high_spd.motor_3.motor_speed / 1000.0,
        high_spd.motor_4.motor_speed / 1000.0,
        high_spd.motor_5.motor_speed / 1000.0,
        high_spd.motor_6.motor_speed / 1000.0,
        0.0,
    ], dtype=np.float32)
    effort = np.asarray([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        gs.grippers_effort / 1000.0,
    ], dtype=np.float32)

    stamps = []
    for msg in (joint_msg, high_spd):
        st = getattr(msg, "time_stamp", None)
        if isinstance(st, (int, float)) and st > 0:
            stamps.append(float(st))
    stamp = max(stamps) if stamps else time.time()
    return qpos, qvel, effort, stamp


class SDKDaggerRig:
    """Zero-ROS 3-camera + 2-follower + 2-master SDK hardware controller."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        ensure_inference_origin_paths(cfg["runtime"]["inference_origin_root"])
        from robot_io import PiperArm, PiperDualArm, RealSenseRig

        self._RealSenseRig = RealSenseRig
        self._PiperArm = PiperArm
        self._PiperDualArm = PiperDualArm

        self.arm_cfg = cfg["arm"]
        self.can_cfg = cfg["can"]
        self.cam_cfg = cfg["camera"]
        self._need_full_arm_state = (
            str((cfg.get("recording") or {}).get("format", "lerobot")).strip().lower() == "hdf5"
        )
        self.right_offset = float(self.arm_cfg.get("right_offset", 0.0))
        bias_cfg = self.arm_cfg.get("joint_bias_deg") or {}
        if not isinstance(bias_cfg, dict):
            raise ValueError("arm.joint_bias_deg must be a dict with left/right 6-element arrays")
        left_bias = np.asarray(bias_cfg.get("left", [0.0] * 6), dtype=float).reshape(6)
        right_bias = np.asarray(bias_cfg.get("right", [0.0] * 6), dtype=float).reshape(6)
        bias_deg = np.concatenate([left_bias, right_bias], axis=0)
        if not np.isfinite(bias_deg).all():
            raise ValueError("arm.joint_bias_deg must contain finite numbers")
        self._joint_bias_rad = np.deg2rad(bias_deg)
        self.handle_collision_cfg = HandleCollisionConfig.from_dict(self.arm_cfg.get("master_handle_collision"))
        self.last_collision_avoidance: dict[str, Any] | None = None
        self._clip_log_last_t: float = 0.0
        self._clip_total_steps: int = 0
        self._clip_prev_left6: np.ndarray | None = None
        self._clip_prev_right6: np.ndarray | None = None

        self.cameras = None
        self.followers = None
        self.master_left = None
        self.master_right = None

        self._fa_stop_event = threading.Event()
        self._fa_thread: threading.Thread | None = None
        self._fa_readers: dict[str, Any] = {}
        self._fa_slew_lock = threading.Lock()
        self._fa_last_cmd_mdeg: dict[str, list[float] | None] = {"left": None, "right": None}
        self._fa_last_cmd_t: dict[str, float | None] = {"left": None, "right": None}
        self._following_enabled = threading.Event()
        self._teaching_active = False
        self._mode_lock = threading.RLock()

        # Pre-roll ring buffer (populated by background camera+follower sampler during transitions)
        self._preroll_lock = threading.Lock()
        self._preroll_buffer: deque[FrameSample] = deque(maxlen=120)
        self._preroll_stop = threading.Event()
        self._preroll_thread: threading.Thread | None = None

    def start(self) -> None:
        # Build follower DualArm reusing inference-origin's PiperDualArm (including 200Hz high_follow)
        dual_can_cfg = {
            "left": {"name": self.can_cfg["follower_left"]["name"]},
            "right": {"name": self.can_cfg["follower_right"]["name"]},
        }
        self.followers = self._PiperDualArm.from_config(self.arm_cfg, dual_can_cfg)
        self.followers.connect()
        self.followers.start_state_stream()

        # Connect master arms (already safely prepared in FC CAN1 by prepare_four_arms_current_pose)
        self.master_left = self._PiperArm(
            can_name=self.can_cfg["master_left"]["name"],
            auto_enable=False,
            gripper_exist=True,
        )
        self.master_right = self._PiperArm(
            can_name=self.can_cfg["master_right"]["name"],
            auto_enable=False,
            gripper_exist=True,
        )
        self.master_left.connect()
        self.master_right.connect()

        # Start 3 RealSense cameras directly via pyrealsense2
        self.cameras = self._RealSenseRig(self.cam_cfg)
        self.cameras.start()

    def move_followers_to_init(self) -> None:
        init_cfg = self.arm_cfg["init_action"]
        left = list(init_cfg["left"])
        right = list(init_cfg["right"])
        steps = int(self.arm_cfg.get("init_steps", 100))
        sleep_s = float(self.arm_cfg.get("init_sleep_s", 0.01))
        print("[归位] 双从臂正在平滑移动到官方初始姿态……", flush=True)
        self.followers.move_to(left=left, right=right, steps=steps, sleep_s=sleep_s)
        self._sync_clip_prev_from_followers()
        print("[归位] 双从臂已到达初始姿态。", flush=True)

    def read_sample(self) -> FrameSample:
        """Read timestamp-aligned 3-camera RGB + 200Hz synced follower state (matching AgilexRobotIO)."""
        frame_time = None
        while frame_time is None:
            frame_time = self.cameras.latest_frame_time()
            if frame_time is None:
                time.sleep(0.002)
        images_by_model_key, image_timestamps = self.cameras.read_images_at_or_after(frame_time)
        img_ts = min(image_timestamps.values()) if image_timestamps else float(frame_time)
        images_bgr = {
            "cam_high": images_by_model_key["top_head"].copy(),
            "cam_left_wrist": images_by_model_key["hand_left"].copy(),
            "cam_right_wrist": images_by_model_key["hand_right"].copy(),
        }
        synced_qpos, _ = self.followers.get_synced_state(frame_time)
        qpos = np.asarray(synced_qpos, dtype=np.float32).reshape(14)
        if getattr(self, "_need_full_arm_state", False):
            _, lv, le, _ = _read_piper_arm_full_state(self.followers.left)
            _, rv, re_, _ = _read_piper_arm_full_state(self.followers.right)
            qvel = np.concatenate([lv, rv], axis=0)
            effort = np.concatenate([le, re_], axis=0)
        else:
            qvel = np.zeros(14, dtype=np.float32)
            effort = np.zeros(14, dtype=np.float32)
        return FrameSample(
            timestamp=float(img_ts),
            qpos=qpos,
            qvel=qvel,
            effort=effort,
            images_bgr=images_bgr,
        )

    def get_camera_health(self) -> dict[str, dict[str, Any]]:
        """Return per-camera latest frame age & stale flag (O(1) read-only check)."""
        role_map = (
            ("front", "cam_high", "正面相机"),
            ("left", "cam_left_wrist", "左腕相机"),
            ("right", "cam_right_wrist", "右腕相机"),
        )
        now = time.time()
        out: dict[str, dict[str, Any]] = {}
        cams = getattr(self.cameras, "cameras", None) if self.cameras is not None else None
        for role, rec_key, label in role_map:
            serial = str((self.cam_cfg.get(role) or {}).get("serial", "")) if isinstance(self.cam_cfg, dict) else ""
            cam_obj = cams.get(role) if isinstance(cams, dict) else None
            ts = None
            if cam_obj is not None:
                try:
                    ts = cam_obj.latest_timestamp()
                except Exception:
                    ts = None
            age_s = max(0.0, now - float(ts)) if ts is not None else None
            out[rec_key] = {
                "role": role,
                "label": label,
                "serial": serial,
                "timestamp": float(ts) if ts is not None else None,
                "age_s": age_s,
                "stale": bool(age_s is not None and age_s > 2.0),
            }
        return out

    def read_follower_qpos(self) -> np.ndarray:
        lq, _, _, _ = _read_piper_arm_full_state(self.followers.left)
        rq, _, _, _ = _read_piper_arm_full_state(self.followers.right)
        return np.concatenate([lq, rq], axis=0)

    def _sync_clip_prev_from_followers(self) -> None:
        """Anchor stateful boundary-slide clipper to current measured follower pose."""
        if self.followers is None:
            return
        try:
            q = self.read_follower_qpos().astype(np.float64)
            self._clip_prev_left6 = q[:6].copy()
            self._clip_prev_right6 = q[7:13].copy()
        except Exception:
            self._clip_prev_left6 = None
            self._clip_prev_right6 = None

    def _clip_follower_action_for_handle_safety(self, action14: np.ndarray) -> np.ndarray:
        """Real-time stateful smooth boundary projection keeping follower actions inside handle-safe space."""
        if not self.handle_collision_cfg.enabled or not self.handle_collision_cfg.clip_follower_actions:
            return action14
        out = action14.copy()
        safe_cfg = self.arm_cfg.get("master_safe_pose") or {}
        safe_l = safe_cfg.get("left", [0.0, 0.30, -0.30, 0.0, 0.0, 0.0])[:6]
        safe_r = safe_cfg.get("right", [0.0, 0.30, -0.30, 0.0, 0.0, 0.0])[:6]

        prev_l = getattr(self, "_clip_prev_left6", None)
        prev_r = getattr(self, "_clip_prev_right6", None)
        ql, info_l = project_pose_to_handle_safe(
            out[:6], self.handle_collision_cfg, q_prev6=prev_l, safe_fallback6=safe_l
        )
        qr, info_r = project_pose_to_handle_safe(
            out[7:13], self.handle_collision_cfg, q_prev6=prev_r, safe_fallback6=safe_r
        )
        out[:6] = ql
        out[7:13] = qr
        self._clip_prev_left6 = ql.copy()
        self._clip_prev_right6 = qr.copy()
        if info_l.get("clipped") or info_r.get("clipped"):
            self._clip_total_steps += 1
            now = time.monotonic()
            if now - self._clip_log_last_t >= 5.0:
                self._clip_log_last_t = now
                parts = []
                if info_l.get("clipped"):
                    parts.append(f"左臂 maxΔ={info_l.get('max_delta_deg', 0.0):.1f}°")
                if info_r.get("clipped"):
                    parts.append(f"右臂 maxΔ={info_r.get('max_delta_deg', 0.0):.1f}°")
                summary = ", ".join(parts)
                print(
                    f"🛡️ [实时防碰撞限幅] 已平滑约束从臂动作避开示教器手柄碰撞区（{summary}）",
                    flush=True,
                )
        return out

    def apply_policy_action(self, action14: np.ndarray, *, action_step: dict[str, Any] | None = None) -> None:
        with self._mode_lock:
            if self._teaching_active:
                return
            action = np.asarray(action14, dtype=float).reshape(-1).copy()
            if action.size < 14:
                raise ValueError(f"dual arm action must have 14 values, got {action.size}")
            action[list(ARM_JOINT_INDICES)] += self._joint_bias_rad
            action = self._clip_follower_action_for_handle_safety(action)
            self.followers.apply_action(action, action_step=action_step)

    def _anchor_followers_at_measured_pose(self) -> None:
        """Send each follower its own current measured pose so firmware halts in place with zero snap-back."""
        if self.followers is None:
            return
        try:
            qpos = self.read_follower_qpos().astype(np.float64)
        except Exception:
            return
        for arm, q7 in ((self.followers.left, qpos[:7]), (self.followers.right, qpos[7:14])):
            try:
                joints_mdeg = [round(float(v) * RAD_TO_MDEG) for v in q7[:6]]
                grip_um = max(0, min(80000, round(float(q7[6]) * 1_000_000)))
                with arm._lock:
                    arm._piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                    arm._piper.JointCtrl(*joints_mdeg)
                    if getattr(arm, "gripper_exist", True):
                        arm._piper.GripperCtrl(grip_um, int(self.arm_cfg.get("gripper_effort", 1000)), 0x01, 0)
                    arm._high_follow_mode = False
            except Exception:
                pass

    def hold_followers(self) -> None:
        if self.followers is None:
            return
        # Never call hold_current_position() while in DAgger or when high_follow is paused:
        # doing so would restart high_follow and replay the stale pre-DAgger _high_follow_last_ref!
        hf_enabled = bool(getattr(getattr(self.followers, "high_follow_config", None), "enabled", False))
        hf_running = bool(getattr(self.followers, "_high_follow_running", False))
        if self._teaching_active or (hf_enabled and not hf_running):
            self._following_enabled.clear()
            self._anchor_followers_at_measured_pose()
            return
        self.followers.hold_current_position()

    def _pause_high_follow_for_teleop(self) -> None:
        """Stop the 200Hz interpolator loop cleanly before direct FA forwarding writes to CAN."""
        if self.followers is None:
            return
        if not self.followers.high_follow_config.enabled:
            return
        self.followers._high_follow_running = False
        thread = self.followers._high_follow_thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=1.0)
        self.followers._high_follow_thread = None
        with self.followers._high_follow_lock:
            self.followers._high_follow_queue.clear()
            self.followers._high_follow_metadata_queue.clear()
            self.followers._reset_high_follow_segment_locked(reset_anchor=True)
            self.followers._high_follow_prev_waypoint = None
            # Clear stale pre-DAgger ref so nothing can ever replay an old inference pose
            self.followers._high_follow_last_ref = None
            self.followers._high_follow_anchor_waypoint = None

    def _resume_high_follow_after_teleop(self) -> None:
        """Re-enter high_follow anchored at the followers' NEW post-teleop measured pose."""
        if self.followers is None:
            return
        if not self.followers.high_follow_config.enabled:
            return
        cur_qpos = self.read_follower_qpos().astype(float)
        self.followers.left.enter_high_follow_mode()
        self.followers.right.enter_high_follow_mode()
        with self.followers._high_follow_lock:
            self.followers._high_follow_queue.clear()
            self.followers._high_follow_metadata_queue.clear()
            self.followers._reset_high_follow_segment_locked(reset_anchor=False)
            self.followers._high_follow_last_ref = cur_qpos.copy()
            self.followers._high_follow_prev_waypoint = None
            self.followers._high_follow_anchor_waypoint = cur_qpos.copy()
        self.followers.start_high_follow_control()

    def _get_fa_control_mode(self) -> str:
        raw = self.arm_cfg.get("fa_control_mode", self.arm_cfg.get("control_mode", "high_follow"))
        mode = str(raw).strip().lower()
        if mode not in ("high_follow", "position"):
            raise ValueError(f"Unsupported arm.fa_control_mode: {raw!r} (expected 'high_follow' or 'position')")
        return mode

    def _get_fa_max_joint_vel_rad_s(self) -> list[float]:
        hf_cfg = self.arm_cfg.get("high_follow") if isinstance(self.arm_cfg, dict) else None
        raw = (hf_cfg or {}).get("max_joint_vel", [2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
        if isinstance(raw, (int, float)):
            return [max(0.1, float(raw))] * 6
        vals = [max(0.1, float(v)) for v in raw]
        if len(vals) != 6:
            return [2.0, 2.0, 2.0, 2.5, 2.5, 2.5]
        return vals

    def _ensure_fa_slew_state(self) -> None:
        if not hasattr(self, "_fa_slew_lock"):
            self._fa_slew_lock = threading.Lock()
        if not hasattr(self, "_fa_last_cmd_mdeg"):
            self._fa_last_cmd_mdeg = {"left": None, "right": None}
        if not hasattr(self, "_fa_last_cmd_t"):
            self._fa_last_cmd_t = {"left": None, "right": None}

    def _reset_fa_slew_state(self) -> None:
        self._ensure_fa_slew_state()
        with self._fa_slew_lock:
            self._fa_last_cmd_mdeg["left"] = None
            self._fa_last_cmd_mdeg["right"] = None
            self._fa_last_cmd_t["left"] = None
            self._fa_last_cmd_t["right"] = None

    def _slew_limit_fa_joints_mdeg(
        self, side: str, target_mdeg: list[int], now: float | None = None
    ) -> tuple[list[int], bool]:
        """Velocity-cap FA joint commands in high_follow (0xAD) mode to prevent resume jumps.

        Normal human teleop (< 115-143 deg/s) passes through 1:1 untouched; any large gap
        (e.g., after pausing following in DAGGER_PAUSED and moving masters) ramps smoothly.
        """
        self._ensure_fa_slew_state()
        now_t = time.monotonic() if now is None else float(now)
        target = [float(v) for v in target_mdeg[:6]]
        max_vel = self._get_fa_max_joint_vel_rad_s()

        with self._fa_slew_lock:
            prev = self._fa_last_cmd_mdeg.get(side)
            prev_t = self._fa_last_cmd_t.get(side)
            if prev is None:
                seeded = None
                try:
                    qpos = self.read_follower_qpos()
                    sl = qpos[:6] if side == "left" else qpos[7:13]
                    seeded = [float(v) * RAD_TO_MDEG for v in sl]
                except Exception:
                    seeded = None
                prev = seeded if seeded is not None else list(target)
                dt = 0.005
            else:
                dt = 0.005 if prev_t is None else min(max(now_t - prev_t, 0.002), 0.05)

            out: list[float] = []
            ramping = False
            for i in range(6):
                max_step = max(1.0, max_vel[i] * RAD_TO_MDEG * dt)
                delta = target[i] - prev[i]
                if abs(delta) > max_step:
                    out.append(prev[i] + (max_step if delta > 0.0 else -max_step))
                    ramping = True
                else:
                    out.append(target[i])

            self._fa_last_cmd_mdeg[side] = out
            self._fa_last_cmd_t[side] = now_t
            return [round(v) for v in out], ramping

    def _apply_follower_direct_mdeg(self, side: str, joints_mdeg: list[int], gripper_um: int) -> None:
        arm = self.followers.left if side == "left" else self.followers.right
        mode = self._get_fa_control_mode()
        speed = int(self.arm_cfg.get("fa_forward_velocity", 100))
        effort = int(self.arm_cfg.get("fa_gripper_effort", 1000))
        clamped_grip = max(0, min(80000, int(gripper_um)))
        with arm._lock:
            if mode == "high_follow":
                if not getattr(arm, "_high_follow_mode", False):
                    arm._piper.MotionCtrl_2(0x01, 0x01, speed, 0xAD)
                    arm._high_follow_mode = True
                arm._piper.JointCtrl(*[int(v) for v in joints_mdeg[:6]])
                if getattr(arm, "gripper_exist", True):
                    arm._piper.GripperCtrl(clamped_grip, effort, 0x01, 0)
            else:
                arm._piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
                arm._piper.JointCtrl(*[int(v) for v in joints_mdeg[:6]])
                if getattr(arm, "gripper_exist", True):
                    arm._piper.GripperCtrl(clamped_grip, effort, 0x01, 0)
                arm._high_follow_mode = False

    def _move_masters_and_verify(self, left_target7: np.ndarray, right_target7: np.ndarray, *,
                                 label: str, duration_s: float = 3.0, hold_seconds: float = 0.0,
                                 shutdown_fn=lambda: False) -> bool:
        from master_alignment_feedback import wait_for_master_alignment

        speed = int(self.arm_cfg.get("master_align_velocity", 30))
        tol_mdeg = int(self.arm_cfg.get("master_align_tolerance_mdeg", 1000))
        targets = {"left": list(left_target7), "right": list(right_target7)}

        def publish_pose(vec_l7: np.ndarray, vec_r7: np.ndarray, *, vel: int = speed) -> None:
            for arm, vec in ((self.master_left, vec_l7), (self.master_right, vec_r7)):
                joints = [round(float(v) * RAD_TO_MDEG) for v in vec[:6]]
                grip = round(min(0.08, max(0.0, float(vec[6]))) * 1_000_000)
                with arm._lock:
                    arm._piper.MotionCtrl_2(0x01, 0x01, int(vel), 0x00)
                    arm._piper.JointCtrl(*joints)
                    arm._piper.GripperCtrl(grip, 1000, 0x01, 0)

        def publish_targets() -> None:
            publish_pose(left_target7, right_target7, vel=speed)

        # Read actual current master poses and plan synchronized collision-avoiding trajectories
        # so the 100mm teach handle never sweeps into the base/table during Step 1/2 OR Step 2/2.
        cur_l7 = None
        cur_r7 = None
        try:
            ql_cur, _, _, _ = _read_piper_arm_full_state(self.master_left)
            qr_cur, _, _, _ = _read_piper_arm_full_state(self.master_right)
            cur_l7 = ql_cur.astype(np.float64)
            cur_r7 = qr_cur.astype(np.float64)
        except Exception:
            cur_l7 = cur_r7 = None

        if cur_l7 is not None and cur_r7 is not None and self.handle_collision_cfg.enabled:
            dt = 0.04  # 25 Hz synchronized waypoint stream
            base_steps = max(15, int(round(float(duration_s) / dt)))
            traj_l, info_l = plan_safe_master_trajectory(
                cur_l7[:6], left_target7[:6], self.handle_collision_cfg, num_steps=base_steps
            )
            traj_r, info_r = plan_safe_master_trajectory(
                cur_r7[:6], right_target7[:6], self.handle_collision_cfg, num_steps=base_steps
            )
            for side_zh, info in (("左", info_l), ("右", info_r)):
                if info.get("detour"):
                    init_mm = float(info.get("initial_clearance_m", 0.0)) * 1000.0
                    print(
                        f"🛡️ [主臂移动避障] 检测到{side_zh}主臂移动路径（{label}）将靠近/碰撞底座或桌面"
                        f"（安全余量 {init_mm:.1f} mm），已自动规划绕行轨迹同步避让……",
                        flush=True,
                    )
            n_total = max(len(traj_l), len(traj_r))
            track_vel = max(speed, 45)
            for k in range(n_total):
                if shutdown_fn():
                    return False
                il = min(k, len(traj_l) - 1)
                ir = min(k, len(traj_r) - 1)
                frac = float(k + 1) / float(n_total)
                gl = (1.0 - frac) * float(cur_l7[6]) + frac * float(left_target7[6])
                gr = (1.0 - frac) * float(cur_r7[6]) + frac * float(right_target7[6])
                wl = np.concatenate([traj_l[il], [gl]])
                wr = np.concatenate([traj_r[ir], [gr]])
                publish_pose(wl, wr, vel=track_vel)
                time.sleep(dt)
            publish_targets()
        else:
            end_t = time.monotonic() + float(duration_s)
            while time.monotonic() < end_t and not shutdown_fn():
                publish_targets()
                time.sleep(0.05)
        if shutdown_fn():
            return False

        report = wait_for_master_alignment(
            targets,
            since_timestamp=time.time(),
            publish=publish_targets,
            shutdown=shutdown_fn,
            timeout=8.0,
            tolerance_mdeg=tol_mdeg,
            stable_seconds=0.3,
        )
        if not report.get("passed"):
            print(f"❌ 主臂对齐失败（{label}）：{report.get('error')}", flush=True)
            return False
        print(f"✅ 双主臂六关节已对齐到位（{label}）", flush=True)

        if hold_seconds > 0:
            print(f"⏳ 主臂已到位，保持位置 {hold_seconds:.1f} 秒后将自动释放进入可拖动示教（请准备扶稳主臂）……", flush=True)
            hold_report = wait_for_master_alignment(
                targets,
                since_timestamp=time.time(),
                publish=publish_targets,
                shutdown=shutdown_fn,
                timeout=hold_seconds + 2.0,
                tolerance_mdeg=tol_mdeg,
                stable_seconds=hold_seconds,
            )
            if not hold_report.get("passed"):
                print(f"❌ 主臂位置保持失败：{hold_report.get('error')}", flush=True)
                return False
            print(f"✅ 主臂已完成 {hold_seconds:.1f} 秒位置保持验证，正在释放主臂……", flush=True)
        return True

    def start_preroll_capture(self) -> None:
        self.stop_preroll_capture()
        with self._preroll_lock:
            self._preroll_buffer.clear()
        self._preroll_stop.clear()

        def _loop():
            period = 1.0 / float(self.cfg["recording"].get("frame_rate", 30))
            while not self._preroll_stop.is_set():
                t0 = time.monotonic()
                try:
                    sample = self.read_sample()
                    with self._preroll_lock:
                        self._preroll_buffer.append(sample)
                except Exception:
                    pass
                dt = period - (time.monotonic() - t0)
                if dt > 0:
                    self._preroll_stop.wait(dt)

        self._preroll_thread = threading.Thread(target=_loop, daemon=True)
        self._preroll_thread.start()

    def stop_preroll_capture(self, *, window_s: float = 0.0) -> list[FrameSample]:
        self._preroll_stop.set()
        if self._preroll_thread is not None and self._preroll_thread.is_alive():
            self._preroll_thread.join(timeout=1.0)
        self._preroll_thread = None
        with self._preroll_lock:
            items = list(self._preroll_buffer)
            self._preroll_buffer.clear()
        if window_s <= 0 or not items:
            return []
        cutoff = items[-1].timestamp - float(window_s)
        return [s for s in items if s.timestamp >= cutoff]

    def _adjust_followers_for_handle_collision(
        self,
        target_l: np.ndarray,
        target_r: np.ndarray,
        safe_l: np.ndarray,
        safe_r: np.ndarray,
        *,
        shutdown_fn=lambda: False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check 100mm teach-gripper handle vs base/table collision; gently nudge follower if needed."""
        self.last_collision_avoidance = None
        if not self.handle_collision_cfg.enabled:
            return target_l, target_r

        adj_l, info_l = find_minimal_safe_pose(target_l, safe_l, self.handle_collision_cfg)
        adj_r, info_r = find_minimal_safe_pose(target_r, safe_r, self.handle_collision_cfg)
        if not info_l.get("adjusted") and not info_r.get("adjusted"):
            return target_l, target_r

        for side_zh, info in (("左", info_l), ("右", info_r)):
            if info.get("adjusted"):
                init_mm = info.get("initial_clearance_m", 0.0) * 1000.0
                print(
                    f"🛡️ [防碰撞] 检测到{side_zh}主臂示教器手柄将靠近/碰撞底座或桌面（安全余量 {init_mm:.1f} mm），"
                    f"正在微调{side_zh}从臂（最大调整 {info.get('max_delta_deg', 0.0):.1f}°：{info.get('delta_deg')}）至安全对齐位……",
                    flush=True,
                )

        if shutdown_fn():
            return adj_l, adj_r

        steps = int(self.arm_cfg.get("collision_avoid_steps", 60))
        sleep_s = float(self.arm_cfg.get("collision_avoid_sleep_s", 0.01))
        hf_cfg = getattr(self.followers, "high_follow_config", None)
        hf_enabled = bool(getattr(hf_cfg, "enabled", False))
        was_paused = hf_enabled and not bool(getattr(self.followers, "_high_follow_running", True))
        if was_paused:
            self._resume_high_follow_after_teleop()
            self._sync_clip_prev_from_followers()
        try:
            if hf_enabled and hasattr(self.followers, "move_to_high_follow"):
                timeout_s = max(5.0, float(steps) * float(sleep_s) + 3.0)
                self.followers.move_to_high_follow(
                    left=adj_l.tolist(),
                    right=adj_r.tolist(),
                    timeout_s=timeout_s,
                )
            else:
                self.followers.move_to(
                    left=adj_l.tolist(),
                    right=adj_r.tolist(),
                    steps=steps,
                    sleep_s=sleep_s,
                )
        finally:
            if was_paused:
                self._pause_high_follow_for_teleop()
        # Re-read actual achieved follower pose so masters align exactly to followers
        refreshed = self.read_follower_qpos()
        out_l = refreshed[:7].astype(np.float64)
        out_r = refreshed[7:14].astype(np.float64)

        sides_meta: dict[str, Any] = {}
        for side_key, info, before7, after7 in (
            ("left", info_l, target_l, out_l),
            ("right", info_r, target_r, out_r),
        ):
            if info.get("adjusted"):
                sides_meta[side_key] = {
                    "reason": info.get("reason", "base"),
                    "initial_clearance_mm": round(float(info.get("initial_clearance_m", 0.0)) * 1000.0, 2),
                    "final_clearance_mm": round(float(info.get("final_clearance_m", 0.0)) * 1000.0, 2),
                    "max_delta_deg": round(float(info.get("max_delta_deg", 0.0)), 2),
                    "delta_deg": list(info.get("delta_deg") or []),
                    "before_qpos7": [round(float(x), 5) for x in before7],
                    "after_qpos7": [round(float(x), 5) for x in after7],
                }
        self.last_collision_avoidance = {
            "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "safety_margin_mm": round(self.handle_collision_cfg.safety_margin_m * 1000.0, 1),
            "sides": sides_meta,
        }
        return out_l, out_r

    def _read_master_actual_gripper_widths_um(
        self, *, fallback_left_m: float, fallback_right_m: float, timeout_s: float = 0.4
    ) -> dict[str, int]:
        """Read each master's fresh actual gripper width (um) in FC before releasing to FA."""
        widths: dict[str, int] = {}
        deadline = time.monotonic() + float(timeout_s)
        while len(widths) < 2 and time.monotonic() < deadline:
            for side, reader in self._fa_readers.items():
                if side in widths:
                    continue
                try:
                    snap = reader.snapshot()
                except Exception:
                    snap = None
                if snap is not None and "command_gripper_um" in snap:
                    widths[side] = max(0, min(80000, int(snap["command_gripper_um"])))
            if len(widths) < 2:
                time.sleep(0.005)
        for side, arm, fb_m in (
            ("left", self.master_left, fallback_left_m),
            ("right", self.master_right, fallback_right_m),
        ):
            if side not in widths:
                try:
                    q, _, _, _ = _read_piper_arm_full_state(arm)
                    widths[side] = max(0, min(80000, round(float(q[6]) * 1_000_000)))
                except Exception:
                    widths[side] = max(0, min(80000, round(float(fb_m) * 1_000_000)))
        return widths

    def _release_masters_to_fa(self, actual_widths_um: dict[str, int]) -> None:
        """Release each master gripper (effort=0, code=0x00) at its actual width, then send 0xFA."""
        for side, arm in (("left", self.master_left), ("right", self.master_right)):
            grip_um = max(0, min(80000, int(actual_widths_um.get(side, 0))))
            with arm._lock:
                arm._piper.GripperCtrl(grip_um, 0, 0x00, 0)
                arm._piper.MasterSlaveConfig(
                    linkage_config=0xFA, feedback_offset=0, ctrl_offset=0, linkage_offset=0
                )

    def enter_teaching_mode(self, *, shutdown_fn=lambda: False) -> bool:
        """Align masters to followers, hold 3s, release to 0xFA, and start direct CAN FA forwarding."""
        from master_native_forwarding import NativeMasterReader

        with self._mode_lock:
            self.hold_followers()

            safe_cfg = self.arm_cfg["master_safe_pose"]
            safe_l = np.asarray(safe_cfg["left"], dtype=np.float64)
            safe_r = np.asarray(safe_cfg["right"], dtype=np.float64)
            align_dur = float(self.arm_cfg.get("master_align_duration_s", 3.0))
            hold_s = float(self.arm_cfg.get("master_hold_seconds", 3.0))

            print("🛡️ 步骤 1/2：主臂先移动到中间安全姿态（请勿触碰主臂）……", flush=True)
            if not self._move_masters_and_verify(safe_l, safe_r, label="中间姿态",
                                                 duration_s=align_dur, hold_seconds=0.0,
                                                 shutdown_fn=shutdown_fn):
                return False

            follower_qpos = self.read_follower_qpos()
            target_l = follower_qpos[:7].astype(np.float64)
            target_r = follower_qpos[7:14].astype(np.float64)
            target_l, target_r = self._adjust_followers_for_handle_collision(
                target_l, target_r, safe_l, safe_r, shutdown_fn=shutdown_fn
            )
            if shutdown_fn():
                return False
            print("🎯 步骤 2/2：主臂正在对齐当前从臂姿态……", flush=True)
            if not self._move_masters_and_verify(target_l, target_r, label="对齐从臂",
                                                 duration_s=align_dur, hold_seconds=hold_s,
                                                 shutdown_fn=shutdown_fn):
                return False

            # Arm native SocketCAN readers and capture each master's actual FC gripper width
            # before releasing grippers and switching 0xFC -> 0xFA (matching kai0-official-dagger).
            self._stop_fa_forwarding()
            self._fa_readers = {
                "left": NativeMasterReader(self.can_cfg["master_left"]["name"]),
                "right": NativeMasterReader(self.can_cfg["master_right"]["name"]),
            }
            actual_since = time.time()
            for reader in self._fa_readers.values():
                reader.start()
                reader.wait_started()
                reader.begin_phase("actual", actual_since)

            actual_widths_um = self._read_master_actual_gripper_widths_um(
                fallback_left_m=float(target_l[6]),
                fallback_right_m=float(target_r[6]),
            )

            fa_since = time.time()
            for reader in self._fa_readers.values():
                reader.begin_phase("fa", fa_since)

            # Pause 200Hz high_follow right before direct FA forwarding takes over follower CAN.
            # Keeping it running until here ensures handle collision avoidance can smoothly move
            # followers and that aborted/failed alignment never leaves high_follow paused.
            self._pause_high_follow_for_teleop()

            # Official Kai0 FC->FA gripper handoff: release each master gripper at its actual width
            # (effort=0, code=0x00, set_zero=0) right before MasterSlaveConfig(0xFA) to prevent snap-shut.
            self._release_masters_to_fa(actual_widths_um)

            fa_mode = self._get_fa_control_mode()
            # Force a clean MotionCtrl_2 handshake on first FA frame (sets 0xAD + fa_forward_velocity)
            if self.followers is not None:
                for arm in (self.followers.left, self.followers.right):
                    with arm._lock:
                        arm._high_follow_mode = False
            self._reset_fa_slew_state()
            self._following_enabled.set()
            self._fa_stop_event.clear()
            self._teaching_active = True
            mode_desc = "high_follow (0xAD，与推理一致)" if fa_mode == "high_follow" else "position (0x00)"
            print(f"🎮 人工示教 FA 直转已启动：从臂控制模式 = {mode_desc}", flush=True)

            def _fa_forward_loop():
                last_stamps = {"left": None, "right": None}
                latest_targets: dict[str, tuple[list[int], int] | None] = {"left": None, "right": None}
                ramping_state = {"left": False, "right": False}
                use_slew = (self._get_fa_control_mode() == "high_follow")
                while not self._fa_stop_event.is_set():
                    did_work = False
                    any_ramping = False
                    for side, reader in self._fa_readers.items():
                        snap = None
                        try:
                            snap = reader.snapshot(after=last_stamps[side])
                        except Exception as exc:
                            print(f"⚠️ 原生主臂 FA 帧异常（{side}）：{exc}", flush=True)
                        if snap is not None:
                            last_stamps[side] = snap["component_stamps"]
                            latest_targets[side] = (list(snap["joint_mdeg"]), int(snap["command_gripper_um"]))

                        if not self._following_enabled.is_set():
                            ramping_state[side] = False
                            continue

                        should_send = (snap is not None) or (use_slew and ramping_state[side])
                        target = latest_targets[side]
                        if not should_send or target is None:
                            continue

                        joints_mdeg, grip_um = target
                        if use_slew:
                            joints_mdeg, is_ramping = self._slew_limit_fa_joints_mdeg(side, joints_mdeg)
                            ramping_state[side] = is_ramping
                            if is_ramping:
                                any_ramping = True
                        self._apply_follower_direct_mdeg(side, joints_mdeg, grip_um)
                        did_work = True
                    if any_ramping:
                        time.sleep(0.005)
                    elif not did_work:
                        time.sleep(0.002)

            self._fa_thread = threading.Thread(target=_fa_forward_loop, daemon=True)
            self._fa_thread.start()
            return True

    def set_following(self, enabled: bool) -> None:
        if enabled:
            if not self._following_enabled.is_set():
                # Re-seed slew limiter from current measured follower pose so resuming
                # after pausing in DAGGER_PAUSED can never jump in 0xAD high_follow mode.
                self._reset_fa_slew_state()
            self._following_enabled.set()
        else:
            self._following_enabled.clear()
            self._reset_fa_slew_state()

    def _stop_fa_forwarding(self) -> None:
        self._following_enabled.clear()
        self._reset_fa_slew_state()
        self._fa_stop_event.set()
        if self._fa_thread is not None and self._fa_thread.is_alive():
            self._fa_thread.join(timeout=1.5)
        self._fa_thread = None
        for reader in self._fa_readers.values():
            try:
                reader.stop()
            except Exception:
                pass
        self._fa_readers = {}

    def exit_teaching_mode(self, *, shutdown_fn=lambda: False) -> None:
        """Stop FA forwarding and return both masters to CAN1 (0xFC) anchored at current measured pose."""
        from prepare_follower_mode import sample_feedback
        from prepare_follower_enabled import checked_enabled_feedback

        with self._mode_lock:
            self._stop_fa_forwarding()
            self._teaching_active = False

            for side, arm, iface in (
                ("left", self.master_left, self.can_cfg["master_left"]["name"]),
                ("right", self.master_right, self.can_cfg["master_right"]["name"]),
            ):
                since = time.time()
                with arm._lock:
                    arm._piper.MasterSlaveConfig(linkage_config=0xFC, feedback_offset=0, ctrl_offset=0, linkage_offset=0)
                deadline = time.monotonic() + 3.5
                previous = None
                anchored = False
                confirmed = False
                while time.monotonic() < deadline and not shutdown_fn():
                    try:
                        row = sample_feedback(iface, since_timestamp=since)
                    except RuntimeError:
                        time.sleep(0.01)
                        continue
                    joints, previous = checked_enabled_feedback(
                        row, now=time.time(), since_timestamp=since, previous=previous
                    )
                    if not anchored:
                        with arm._lock:
                            arm._piper.JointCtrl(*joints)
                            arm._piper.MotionCtrl_1(grag_teach_ctrl=0x02)
                            arm._piper.JointCtrl(*joints)
                            since = time.time()
                            arm._piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01, move_spd_rate_ctrl=10)
                            arm._piper.JointCtrl(*joints)
                        anchored = True
                        continue
                    if row.get("ctrl_mode") == 1 and all(row.get("driver_enabled", [])):
                        confirmed = True
                        break
                    time.sleep(0.01)
                if not confirmed:
                    raise RuntimeError(f"{side} 主臂未在超时内确认回到 CAN1（0xFC）控制模式")

            self._resume_high_follow_after_teleop()

    def close(self) -> None:
        """Stop forwarding and release SDK handles without disabling physical motors or replaying stale poses."""
        self.stop_preroll_capture()
        self._stop_fa_forwarding()
        self._teaching_active = False
        if self.followers is not None:
            try:
                self._pause_high_follow_for_teleop()
                self._anchor_followers_at_measured_pose()
                self.followers.close()
            except Exception:
                pass
        for arm in (self.master_left, self.master_right):
            if arm is not None:
                try:
                    arm.close()
                except Exception:
                    pass
        if self.cameras is not None:
            try:
                self.cameras.close()
            except Exception:
                pass


class MockDaggerRig:
    """Zero-hardware mock rig for full end-to-end testing of the SDK DAgger runtime."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        init_cfg = cfg["arm"]["init_action"]
        self._qpos = np.concatenate([
            np.asarray(init_cfg["left"], dtype=np.float32),
            np.asarray(init_cfg["right"], dtype=np.float32),
        ])
        self._lock = threading.Lock()
        self.teaching_active = False
        self.following_enabled = False
        self.events: list[str] = []
        self._preroll_samples: list[FrameSample] = []
        self._frame_seq = 0
        self.enter_teaching_should_fail = False
        self.exit_teaching_should_fail = False

    def start(self) -> None:
        self.events.append("rig_start")

    def move_followers_to_init(self) -> None:
        self.events.append("move_to_init")
        print("[归位][Mock] 双从臂已到达初始姿态。", flush=True)

    def read_sample(self) -> FrameSample:
        with self._lock:
            self._frame_seq += 1
            if self.teaching_active and self.following_enabled:
                self._qpos[0] += 0.001
                self._qpos[7] += 0.001
            qpos = self._qpos.copy()
            seq = self._frame_seq
        h, w = 120, 160
        imgs = {}
        for idx, cam in enumerate(CAMERA_NAMES):
            frame = np.full((h, w, 3), (seq * 7 + idx * 40) % 255, dtype=np.uint8)
            imgs[cam] = frame
        return FrameSample(
            timestamp=time.time(),
            qpos=qpos,
            qvel=np.zeros(14, dtype=np.float32),
            effort=np.zeros(14, dtype=np.float32),
            images_bgr=imgs,
        )

    def apply_policy_action(self, action14: np.ndarray, *, action_step: dict[str, Any] | None = None) -> None:
        with self._lock:
            if not self.teaching_active:
                self._qpos = np.asarray(action14, dtype=np.float32).copy()

    def hold_followers(self) -> None:
        self.events.append("hold_followers")

    def start_preroll_capture(self) -> None:
        self.events.append("preroll_start")
        self._preroll_samples = [self.read_sample() for _ in range(4)]

    def stop_preroll_capture(self, *, window_s: float = 0.0) -> list[FrameSample]:
        self.events.append("preroll_stop")
        out = list(self._preroll_samples)
        self._preroll_samples = []
        return out if window_s > 0 else []

    def enter_teaching_mode(self, *, shutdown_fn=lambda: False) -> bool:
        self.events.append("enter_teaching")
        if self.enter_teaching_should_fail:
            return False
        self.teaching_active = True
        self.following_enabled = True
        return True

    def set_following(self, enabled: bool) -> None:
        self.following_enabled = bool(enabled)
        self.events.append(f"following_{'on' if enabled else 'off'}")

    def exit_teaching_mode(self, *, shutdown_fn=lambda: False) -> None:
        self.events.append("exit_teaching")
        if self.exit_teaching_should_fail:
            raise RuntimeError("模拟主臂交接超时")
        self.following_enabled = False
        self.teaching_active = False

    def close(self) -> None:
        self.events.append("rig_close")
        self.following_enabled = False
        self.teaching_active = False
