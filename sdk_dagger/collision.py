"""Master 100mm teach-gripper handle collision checker & minimal safe-pose solver.

Kinematics match Piper official URDF (assets/piper/piper.urdf) to machine precision.
Handle geometry is measured directly from 100mm示教器总.STEP (M-JXB2-130107-M):
  - Mounting flange offset: +4.5 mm along link6 +Z (gripper_base_joint)
  - Handle axial center: +30.0 mm forward of mounting face (+34.5 mm along link6 +Z)
  - Handle rod + circular end-cap protrude radially along link6 +X from r=30 mm to r=130 mm
    (rod radius 12 mm, end-cap radius 18 mm).
Obstacles in each arm's base_link frame:
  - Pedestal cylinder (base_link + link1 + shoulder): centered at (-6 mm, 0), radius 55 mm, z in [-20, 165] mm
  - Tabletop plane: z <= 15 mm
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

# Exact Piper URDF joint chain (base_link -> link6), units: meters & radians
_PIPER_JOINTS = (
    # (xyz, rpy, axis, lower, upper)
    ((0.0, 0.0, 0.123), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), -2.6179938779914944, 2.6179938779914944),
    ((0.0, 0.0, 0.0), (1.5707963, -0.1357866, -3.1415926), (0.0, 0.0, 1.0), 0.0, 3.141592653589793),
    ((0.28503, 0.0, 0.0), (0.0, 0.0, -1.7938494), (0.0, 0.0, 1.0), -2.9670597283903604, 0.0),
    ((-0.02198, -0.25075, 0.0), (1.5707963, 0.0, 0.0), (0.0, 0.0, 1.0), -1.7453292519943295, 1.7453292519943295),
    ((0.0, 0.0, 0.0), (-1.5707963, 0.0, 0.0), (0.0, 0.0, 1.0), -1.2217304763960306, 1.2217304763960306),
    ((8.8259e-05, -0.091, 0.0), (1.5707963, 0.0, 0.0), (0.0, 0.0, 1.0), -3.141592653589793, 3.141592653589793),
)

JOINT_LOWER = np.array([j[3] for j in _PIPER_JOINTS], dtype=np.float64)
JOINT_UPPER = np.array([j[4] for j in _PIPER_JOINTS], dtype=np.float64)

_AXIS_VECS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
}


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _fixed_transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rot_z(rpy[2]) @ _rot_y(rpy[1]) @ _rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


_FIXED_TF = tuple(_fixed_transform(j[0], j[1]) for j in _PIPER_JOINTS)


def piper_link6_fk(q6: np.ndarray | list[float]) -> np.ndarray:
    """Compute 4x4 transform of link6 in base_link frame (meters)."""
    q = np.asarray(q6, dtype=np.float64).reshape(6)
    T = np.eye(4, dtype=np.float64)
    for idx in range(6):
        T = T @ _FIXED_TF[idx]
        c, s = math.cos(float(q[idx])), math.sin(float(q[idx]))
        # All 6 Piper revolute joints rotate around local +Z
        R_step = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T[:3, :3] = T[:3, :3] @ R_step
    return T


@dataclass(frozen=True)
class HandleCollisionConfig:
    enabled: bool = True
    handle_axis: str = "+x"
    handle_axial_z_m: float = 0.0345        # 4.5mm flange + 30mm handle axial center
    handle_length_m: float = 0.130          # 130mm radial protrusion (100mm示教器总.STEP)
    handle_rod_radius_m: float = 0.012
    handle_tip_radius_m: float = 0.018
    base_center_xy_m: tuple[float, float] = (-0.006, 0.0)
    base_radius_m: float = 0.055            # base_link + link1 + rear connector
    base_z_min_m: float = -0.020
    base_z_max_m: float = 0.165             # base_link (86mm) + link1 (148mm) + shoulder margin
    table_z_m: float = 0.015                # tabletop plane clearance
    safety_margin_m: float = 0.005          # 5mm alignment safety buffer
    path_samples: int = 10
    clip_follower_actions: bool = True      # real-time smooth projection of policy actions into handle-safe zone
    clip_margin_extra_m: float = 0.003      # extra 3mm cushion -> 8mm online clip boundary

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "HandleCollisionConfig":
        if not raw:
            return cls()
        axis = str(raw.get("handle_axis", "+x")).strip().lower()
        if axis not in _AXIS_VECS:
            raise ValueError(f"Unsupported handle_axis={axis!r}; expected one of {sorted(_AXIS_VECS)}")
        bxy = raw.get("base_center_xy_m", (-0.006, 0.0))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            handle_axis=axis,
            handle_axial_z_m=float(raw.get("handle_axial_z_m", 0.0345)),
            handle_length_m=float(raw.get("handle_length_m", 0.130)),
            handle_rod_radius_m=float(raw.get("handle_rod_radius_m", 0.012)),
            handle_tip_radius_m=float(raw.get("handle_tip_radius_m", 0.018)),
            base_center_xy_m=(float(bxy[0]), float(bxy[1])),
            base_radius_m=float(raw.get("base_radius_m", 0.055)),
            base_z_min_m=float(raw.get("base_z_min_m", -0.020)),
            base_z_max_m=float(raw.get("base_z_max_m", 0.165)),
            table_z_m=float(raw.get("table_z_m", 0.015)),
            safety_margin_m=float(raw.get("safety_margin_m", 0.005)),
            path_samples=max(2, int(raw.get("path_samples", 10))),
            clip_follower_actions=bool(raw.get("clip_follower_actions", True)),
            clip_margin_extra_m=max(0.0, float(raw.get("clip_margin_extra_m", 0.003))),
        )


def _handle_spheres_in_link6(cfg: HandleCollisionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers (K,3), radii (K,)) of collision spheres representing the handle in link6 frame."""
    d = _AXIS_VECS[cfg.handle_axis]
    z_vec = np.array([0.0, 0.0, cfg.handle_axial_z_m], dtype=np.float64)
    # 6 sample spheres from motor edge (30mm) out to handle end-cap (130mm)
    rs = np.linspace(0.030, cfg.handle_length_m, 6)
    centers = np.array([z_vec + r * d for r in rs], dtype=np.float64)
    radii = np.full(len(rs), cfg.handle_rod_radius_m, dtype=np.float64)
    radii[-1] = cfg.handle_tip_radius_m
    return centers, radii


def _sphere_to_cylinder_dist(pts: np.ndarray, radii: np.ndarray, cfg: HandleCollisionConfig) -> np.ndarray:
    """Signed clearance from each sphere to the vertical base/link1 cylinder (positive = clear)."""
    dx = pts[:, 0] - cfg.base_center_xy_m[0]
    dy = pts[:, 1] - cfg.base_center_xy_m[1]
    r_xy = np.hypot(dx, dy) - cfg.base_radius_m
    dz = np.maximum(0.0, np.maximum(cfg.base_z_min_m - pts[:, 2], pts[:, 2] - cfg.base_z_max_m))
    # Outside distance when either r_xy > 0 or dz > 0; inside negative distance when both <= 0
    outside = np.hypot(np.maximum(r_xy, 0.0), dz)
    inside = np.minimum(0.0, np.maximum(r_xy, np.maximum(cfg.base_z_min_m - pts[:, 2], pts[:, 2] - cfg.base_z_max_m)))
    surf_dist = np.where((r_xy > 0.0) | (dz > 0.0), outside, inside)
    return surf_dist - radii


def evaluate_handle_clearance(q6: np.ndarray | list[float], cfg: HandleCollisionConfig) -> dict[str, Any]:
    """Compute signed clearance (meters) of the master handle at joint pose q6.

    clearance < 0 means collision or safety-margin violation against base cylinder or tabletop.
    """
    T6 = piper_link6_fk(q6)
    local_pts, radii = _handle_spheres_in_link6(cfg)
    world_pts = (local_pts @ T6[:3, :3].T) + T6[:3, 3]

    cyl_clear = _sphere_to_cylinder_dist(world_pts, radii, cfg) - cfg.safety_margin_m
    tbl_clear = (world_pts[:, 2] - radii - cfg.table_z_m) - cfg.safety_margin_m

    min_cyl = float(np.min(cyl_clear))
    min_tbl = float(np.min(tbl_clear))
    min_clear = min(min_cyl, min_tbl)
    reason = "base" if min_cyl <= min_tbl else "table"
    return {
        "safe": min_clear >= 0.0,
        "min_clearance_m": min_clear,
        "base_clearance_m": min_cyl,
        "table_clearance_m": min_tbl,
        "reason": reason,
        "tip_xyz_m": world_pts[-1].copy(),
    }


def evaluate_path_clearance(
    q_start6: np.ndarray | list[float],
    q_end6: np.ndarray | list[float],
    cfg: HandleCollisionConfig,
) -> dict[str, Any]:
    """Check handle clearance along the linear joint-space path from q_start6 to q_end6."""
    qs = np.asarray(q_start6, dtype=np.float64).reshape(6)
    qe = np.asarray(q_end6, dtype=np.float64).reshape(6)
    worst = None
    for alpha in np.linspace(0.0, 1.0, cfg.path_samples):
        q = (1.0 - alpha) * qs + alpha * qe
        res = evaluate_handle_clearance(q, cfg)
        if worst is None or res["min_clearance_m"] < worst["min_clearance_m"]:
            worst = dict(res, alpha=float(alpha))
    assert worst is not None
    return worst


def find_minimal_safe_pose(
    target7: np.ndarray | list[float],
    safe_start7: np.ndarray | list[float],
    cfg: HandleCollisionConfig,
    *,
    max_joint_shift_rad: float = 0.70,
) -> tuple[np.ndarray, dict[str, Any]]:
    """If target7 (or path safe_start7->target7) collides, find minimal joint nudge keeping handle safe.

    Returns (adjusted_target7, info_dict). Gripper width (index 6) is always preserved untouched.
    """
    tgt = np.asarray(target7, dtype=np.float64).reshape(7).copy()
    start6 = np.asarray(safe_start7, dtype=np.float64).reshape(7)[:6]
    q0 = tgt[:6].copy()

    if not cfg.enabled:
        return tgt, {"adjusted": False, "reason": "disabled", "clearance_m": 1.0}

    initial = evaluate_path_clearance(start6, q0, cfg)
    if initial["safe"]:
        return tgt, {
            "adjusted": False,
            "reason": "already_safe",
            "initial_clearance_m": initial["min_clearance_m"],
            "final_clearance_m": initial["min_clearance_m"],
            "max_delta_deg": 0.0,
        }

    # Joint preference weights (prefer gently adjusting J2 forward, J3 elbow, J5 wrist pitch, J6 roll)
    weights = np.array([2.5, 1.0, 1.1, 1.8, 0.9, 1.4], dtype=np.float64)
    # Candidate single-axis and coupled escape directions in joint space
    directions = [
        np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0]),      # pitch wrist up (lifts +X handle forward/away from base!)
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),       # extend shoulder forward (increases r_xy from base)
        np.array([0.0, 0.7, 0.0, 0.0, -0.7, 0.0]),      # extend J2 + pitch J5 up
        np.array([0.0, 0.6, -0.5, 0.0, -0.6, 0.0]),     # extend J2 + open J3 + pitch J5 up
        np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]),      # open elbow J3
        np.array([0.0, 0.8, 0.5, 0.0, 0.0, 0.0]),       # lift forearm up
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),       # roll J6 + (rotates handle sideways)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),      # roll J6 -
        np.array([0.0, 0.5, 0.0, 0.0, -0.5, 0.6]),      # combined J2 + J5 + J6+
        np.array([0.0, 0.5, 0.0, 0.0, -0.5, -0.6]),     # combined J2 + J5 + J6-
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),       # pitch wrist down (if colliding with table from other side)
    ]

    best_q: np.ndarray | None = None
    best_cost = float("inf")
    best_clear = initial["min_clearance_m"]

    def _consider(cand: np.ndarray) -> bool:
        nonlocal best_q, best_cost, best_clear
        cand_c = np.clip(cand, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
        if np.max(np.abs(cand_c - q0)) > max_joint_shift_rad + 1e-9:
            return False
        res = evaluate_path_clearance(start6, cand_c, cfg)
        if not res["safe"]:
            return False
        cost = float(np.linalg.norm(weights * (cand_c - q0)))
        if cost < best_cost:
            best_cost = cost
            best_q = cand_c.copy()
            best_clear = res["min_clearance_m"]
        return True

    # 1. Line search + binary refinement along each physical escape direction
    for d in directions:
        d_unit = d / np.linalg.norm(d)
        hit_s = None
        for s in np.linspace(0.05, max_joint_shift_rad, 14):
            if _consider(q0 + s * d_unit):
                hit_s = float(s)
                break
        if hit_s is not None:
            lo, hi = max(0.0, hit_s - 0.05), hit_s
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                if _consider(q0 + mid * d_unit):
                    hi = mid
                else:
                    lo = mid

    # 2. Fallback: wider J2/J5/J6 search if still colliding (e.g., deeply tucked pose)
    if best_q is None:
        for dj2 in (0.2, 0.4, 0.6, 0.85, 1.1):
            for dj5 in (-0.8, -0.5, -0.25, 0.0, 0.25):
                for dj6 in (0.0, 0.7, -0.7, 1.3, -1.3):
                    cand = q0 + np.array([0.0, dj2, 0.0, 0.0, dj5, dj6])
                    cand_c = np.clip(cand, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
                    res = evaluate_path_clearance(start6, cand_c, cfg)
                    if res["safe"]:
                        cost = float(np.linalg.norm(weights * (cand_c - q0)))
                        if cost < best_cost:
                            best_cost = cost
                            best_q = cand_c.copy()
                            best_clear = res["min_clearance_m"]

    if best_q is None:
        # Last resort: stay at safe_start7 (guaranteed safe)
        fallback = safe_start7.copy()
        fallback[6] = tgt[6]
        res = evaluate_handle_clearance(fallback[:6], cfg)
        return fallback, {
            "adjusted": True,
            "reason": f"{initial['reason']}_fallback_safe_pose",
            "initial_clearance_m": initial["min_clearance_m"],
            "final_clearance_m": res["min_clearance_m"],
            "max_delta_deg": float(np.degrees(np.max(np.abs(fallback[:6] - q0)))),
            "delta_deg": np.round(np.degrees(fallback[:6] - q0), 1).tolist(),
        }

    # 3. Local coordinate shrink toward q0 to make the nudge as small as possible
    cur = best_q.copy()
    for step in (0.04, 0.02, 0.01):
        for j in range(6):
            diff = q0[j] - cur[j]
            if abs(diff) < 1e-4:
                continue
            cand = cur.copy()
            cand[j] += math.copysign(min(abs(diff), step), diff)
            res = evaluate_path_clearance(start6, cand, cfg)
            if res["safe"]:
                cur = cand
                best_clear = res["min_clearance_m"]

    out = tgt.copy()
    out[:6] = cur
    return out, {
        "adjusted": True,
        "reason": initial["reason"],
        "initial_clearance_m": initial["min_clearance_m"],
        "final_clearance_m": best_clear,
        "max_delta_deg": float(np.degrees(np.max(np.abs(cur - q0)))),
        "delta_deg": np.round(np.degrees(cur - q0), 1).tolist(),
    }


# Inverse-cost metric for real-time smooth boundary projection:
# prefers adjusting wrist pitch (J5), shoulder (J2), elbow (J3), wrist roll (J6)
# while barely touching yaw (J1) and forearm roll (J4).
_CLIP_INV_WEIGHTS = np.array([0.05, 1.0, 0.7, 0.05, 1.3, 0.35], dtype=np.float64)
_CLIP_ACTIVE_JOINTS = (1, 2, 4, 5)


def _clearance_gradient(
    q6: np.ndarray,
    cfg: HandleCollisionConfig,
    c0: float | None = None,
    eps: float = 1e-3,
) -> tuple[np.ndarray, float]:
    if c0 is None:
        c0 = float(evaluate_handle_clearance(q6, cfg)["min_clearance_m"])
    grad = np.zeros(6, dtype=np.float64)
    for j in _CLIP_ACTIVE_JOINTS:
        qp = q6.copy()
        qp[j] += eps
        grad[j] = (float(evaluate_handle_clearance(qp, cfg)["min_clearance_m"]) - c0) / eps
    return grad, c0


def project_pose_to_handle_safe(
    q6: np.ndarray | list[float],
    cfg: HandleCollisionConfig,
    *,
    q_prev6: np.ndarray | list[float] | None = None,
    safe_fallback6: np.ndarray | list[float] | None = None,
    max_iters: int = 12,
    extra_margin_m: float | None = None,
    max_step_rad: float = 0.07853981633974483,  # 4.5 deg per 30Hz step
) -> tuple[np.ndarray, dict[str, Any]]:
    """Smoothly project a 6-DOF joint command onto the handle-safe boundary for real-time 30Hz clipping.

    When q_prev6 (a safe previous pose) is provided, uses ray-clip + tangential boundary sliding
    with strict slew-rate limiting (<= max_step_rad) so the arm NEVER flips across the base
    cylinder medial axis (eliminating branch-flip jumps completely).
    """
    q_orig = np.asarray(q6, dtype=np.float64).reshape(6)
    if not cfg.enabled or not cfg.clip_follower_actions:
        return q_orig.copy(), {"clipped": False, "clearance_m": 1.0}

    margin_buf = cfg.clip_margin_extra_m if extra_margin_m is None else max(0.0, float(extra_margin_m))
    res0 = evaluate_handle_clearance(q_orig, cfg)
    c0 = float(res0["min_clearance_m"])

    # Fast path: target is already safe
    if c0 >= margin_buf:
        if q_prev6 is None:
            return q_orig.copy(), {
                "clipped": False,
                "initial_clearance_m": c0,
                "final_clearance_m": c0,
                "max_delta_deg": 0.0,
            }
        qp = np.asarray(q_prev6, dtype=np.float64).reshape(6)
        diff = q_orig - qp
        md = float(np.max(np.abs(diff)))
        if md <= max_step_rad:
            return q_orig.copy(), {
                "clipped": False,
                "initial_clearance_m": c0,
                "final_clearance_m": c0,
                "max_delta_deg": 0.0,
            }
        # Slew smoothly from q_prev6 toward safe q_orig (prevents teleport if policy emerged on far side)
        q_target = qp + diff * (max_step_rad / md)
        ct = float(evaluate_handle_clearance(q_target, cfg)["min_clearance_m"])
        if ct >= margin_buf:
            delta_deg = np.degrees(q_target - q_orig)
            return q_target, {
                "clipped": True,
                "reason": "slew_recovery",
                "initial_clearance_m": c0,
                "final_clearance_m": ct,
                "max_delta_deg": float(np.max(np.abs(delta_deg))),
                "delta_deg": [round(float(x), 1) for x in delta_deg],
            }
    else:
        q_target = q_orig

    # Stateful branch-safe projection: start from q_prev6 (already on the safe side)
    if q_prev6 is not None:
        qp = np.asarray(q_prev6, dtype=np.float64).reshape(6)
        cp = float(evaluate_handle_clearance(qp, cfg)["min_clearance_m"])
        if cp >= margin_buf - 0.001:
            diff = q_target - qp
            md = float(np.max(np.abs(diff)))
            if md > max_step_rad:
                q_target = qp + diff * (max_step_rad / md)

            # 1) Binary search ray clip along qp -> q_target (stays strictly on the same side)
            lo, hi = 0.0, 1.0
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                qm = qp + mid * (q_target - qp)
                if float(evaluate_handle_clearance(qm, cfg)["min_clearance_m"]) >= margin_buf:
                    lo = mid
                else:
                    hi = mid
            q_cur = qp + lo * (q_target - qp)

            # 2) Smooth tangential boundary slide toward q_orig within max_step_rad of qp
            for _ in range(2):
                dq_rem = q_orig - q_cur
                if float(np.max(np.abs(dq_rem))) < 1e-4:
                    break
                grad, c_cur = _clearance_gradient(q_cur, cfg)
                g_norm2 = float(np.dot(grad, grad))
                if g_norm2 < 1e-10:
                    break
                dot = float(np.dot(dq_rem, grad))
                dq_tan = dq_rem - (dot / g_norm2) * grad if dot < 0.0 else dq_rem.copy()
                if c_cur < margin_buf:
                    dq_tan += ((margin_buf + 0.0005 - c_cur) / g_norm2) * grad
                m = float(np.max(np.abs(dq_tan)))
                if m > max_step_rad * 0.55:
                    dq_tan *= (max_step_rad * 0.55) / m
                q_try = np.clip(q_cur + dq_tan, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
                c_try = float(evaluate_handle_clearance(q_try, cfg)["min_clearance_m"])
                if c_try < margin_buf:
                    for _ in range(2):
                        gt, ct = _clearance_gradient(q_try, cfg, c_try)
                        wg = _CLIP_INV_WEIGHTS * gt
                        denom = float(np.dot(gt, wg))
                        if denom < 1e-10:
                            break
                        push = ((margin_buf + 0.0005 - ct) / denom) * wg
                        pm = float(np.max(np.abs(push)))
                        if pm > 0.03:
                            push *= 0.03 / pm
                        q_try = np.clip(q_try + push, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
                        c_try = float(evaluate_handle_clearance(q_try, cfg)["min_clearance_m"])
                        if c_try >= margin_buf:
                            break
                if c_try >= margin_buf - 0.0005 and float(np.max(np.abs(q_try - qp))) <= max_step_rad + 1e-9:
                    q_cur = q_try

            c_final = float(evaluate_handle_clearance(q_cur, cfg)["min_clearance_m"])
            delta_deg = np.degrees(q_cur - q_orig)
            return q_cur, {
                "clipped": True,
                "reason": res0["reason"],
                "initial_clearance_m": c0,
                "final_clearance_m": c_final,
                "max_delta_deg": float(np.max(np.abs(delta_deg))),
                "delta_deg": [round(float(x), 1) for x in delta_deg],
            }

    # Stateless fallback (only when q_prev6 is absent or already unsafe)
    q = np.clip(q_orig.copy(), JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
    c = float(evaluate_handle_clearance(q, cfg)["min_clearance_m"])
    target_c = margin_buf + 0.0005

    for _ in range(max_iters):
        if c >= margin_buf:
            break
        grad, c = _clearance_gradient(q, cfg, c)
        wgrad = _CLIP_INV_WEIGHTS * grad
        denom = float(np.dot(grad, wgrad))
        if denom < 1e-10:
            step = np.array([0.0, 0.04, -0.02, 0.0, -0.04, 0.0], dtype=np.float64)
        else:
            alpha = (target_c - c) / denom
            step = alpha * wgrad
            max_s = float(np.max(np.abs(step)))
            if max_s > 0.12:
                step *= 0.12 / max_s
        q = np.clip(q + step, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
        c = float(evaluate_handle_clearance(q, cfg)["min_clearance_m"])

    if c < 0.0:
        fb6 = (
            np.asarray(safe_fallback6, dtype=np.float64).reshape(6)
            if safe_fallback6 is not None
            else np.array([0.0, 0.30, -0.30, 0.0, 0.0, 0.0], dtype=np.float64)
        )
        adj7, info = find_minimal_safe_pose(
            np.concatenate([q_orig, [0.0]]),
            np.concatenate([fb6, [0.0]]),
            cfg,
        )
        q = adj7[:6]
        c = float(info.get("final_clearance_m", 0.0))

    delta_deg = np.degrees(q - q_orig)
    return q, {
        "clipped": True,
        "reason": res0["reason"],
        "initial_clearance_m": c0,
        "final_clearance_m": c,
        "max_delta_deg": float(np.max(np.abs(delta_deg))),
        "delta_deg": [round(float(x), 1) for x in delta_deg],
    }


def _minimum_jerk(u: np.ndarray | float) -> np.ndarray | float:
    """Fifth-order minimum-jerk S-curve s(u) = 10u^3 - 15u^4 + 6u^5 (zero vel & accel at endpoints)."""
    u2 = u * u
    u3 = u2 * u
    return u3 * (10.0 + u * (-15.0 + 6.0 * u))


def _resample_polyline(waypoints: list[np.ndarray], total_steps: int) -> np.ndarray:
    """Resample a joint-space polyline into total_steps minimum-jerk S-curve samples (N, 6)."""
    pts = [np.asarray(p, dtype=np.float64).reshape(6) for p in waypoints]
    if len(pts) <= 1 or total_steps <= 1:
        return np.asarray([pts[-1]], dtype=np.float64)
    seg_lens = [float(np.max(np.abs(pts[i + 1] - pts[i]))) for i in range(len(pts) - 1)]
    total_len = sum(seg_lens)
    if total_len < 1e-6:
        return np.repeat(pts[-1][None, :], total_steps, axis=0)

    out: list[np.ndarray] = []
    remaining = int(total_steps)
    for i, slen in enumerate(seg_lens):
        if i == len(seg_lens) - 1:
            n_seg = max(1, remaining)
        else:
            n_seg = max(3, int(round(total_steps * (slen / total_len))))
            n_seg = min(n_seg, max(1, remaining - (len(seg_lens) - 1 - i)))
        remaining -= n_seg
        # Fifth-order minimum-jerk S-curve on each leg: zero velocity AND zero acceleration at endpoints
        ts = np.linspace(0.0, 1.0, n_seg + 1, dtype=np.float64)[1:]
        alphas = _minimum_jerk(ts)
        p0, p1 = pts[i], pts[i + 1]
        for a in alphas:
            out.append((1.0 - a) * p0 + a * p1)
    return np.asarray(out, dtype=np.float64)


def _dense_path_safe(q0: np.ndarray, q1: np.ndarray, cfg: HandleCollisionConfig, samples: int = 20) -> bool:
    for a in np.linspace(0.0, 1.0, samples):
        if not evaluate_handle_clearance((1.0 - a) * q0 + a * q1, cfg)["safe"]:
            return False
    return True


def _escape_from_interior(q_start6: np.ndarray, cfg: HandleCollisionConfig) -> np.ndarray:
    """Move handle strictly outward along +grad(clearance) to exit collision zone safely."""
    q = np.clip(q_start6.copy(), JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
    target_c = max(0.005, cfg.clip_margin_extra_m + 0.002)
    c = float(evaluate_handle_clearance(q, cfg)["min_clearance_m"])
    for _ in range(18):
        if c >= target_c:
            break
        grad, c = _clearance_gradient(q, cfg, c)
        wgrad = _CLIP_INV_WEIGHTS * grad
        denom = float(np.dot(grad, wgrad))
        if denom < 1e-10:
            # Physical lift away from base: pitch wrist up (J5-) + extend shoulder (J2+)
            step = np.array([0.0, 0.06, -0.04, 0.0, -0.08, 0.0], dtype=np.float64)
        else:
            step = ((target_c + 0.002 - c) / denom) * wgrad
            m = float(np.max(np.abs(step)))
            if m > 0.10:
                step *= 0.10 / m
        q = np.clip(q + step, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02)
        c = float(evaluate_handle_clearance(q, cfg)["min_clearance_m"])
    if c < 0.0:
        adj7, _ = find_minimal_safe_pose(
            np.concatenate([q_start6, [0.0]]),
            np.array([0.0, 0.45, -0.45, 0.0, -0.15, 0.0, 0.0]),
            cfg,
        )
        q = adj7[:6]
    return q


def plan_safe_master_trajectory(
    q_start6: np.ndarray | list[float],
    q_goal6: np.ndarray | list[float],
    cfg: HandleCollisionConfig,
    *,
    num_steps: int = 75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Plan a synchronized, collision-avoiding joint trajectory for a master arm moving q_start6 -> q_goal6.

    Handles both:
      1) Start pose already tucked near/inside base collision zone (first escapes outward along +grad)
      2) Mid-path collision along straight line q_start -> q_goal (inserts minimal safe lift/extend via-point
         or stateful boundary-sliding intermediate path)
      3) Synchronized streaming waypoints so all 6 joints move in lockstep (J5 never races ahead of J2).
    """
    qs = np.asarray(q_start6, dtype=np.float64).reshape(6)
    qg = np.asarray(q_goal6, dtype=np.float64).reshape(6)
    steps = max(10, int(num_steps))

    if not cfg.enabled:
        traj = _resample_polyline([qs, qg], steps)
        return traj, {"detour": False, "escaped_start": False, "via_count": 0, "min_clearance_m": 1.0}

    start_res = evaluate_handle_clearance(qs, cfg)
    raw_path = evaluate_path_clearance(qs, qg, cfg)

    # Fast path: straight line is already completely safe -> synchronized lockstep interpolation
    if start_res["safe"] and raw_path["safe"] and _dense_path_safe(qs, qg, cfg, samples=21):
        traj = _resample_polyline([qs, qg], steps)
        return traj, {
            "detour": False,
            "escaped_start": False,
            "via_count": 0,
            "initial_clearance_m": float(raw_path["min_clearance_m"]),
            "min_clearance_m": float(raw_path["min_clearance_m"]),
            "reason": raw_path["reason"],
        }

    keyframes: list[np.ndarray] = [qs]
    escaped_start = False
    q_curr = qs
    if not start_res["safe"]:
        q_esc = _escape_from_interior(qs, cfg)
        keyframes.append(q_esc)
        q_curr = q_esc
        escaped_start = True

    via_count = 0
    if not _dense_path_safe(q_curr, qg, cfg, samples=21):
        # Search for minimal-detour 1-via-point safe lift/extend waypoint
        q_mid = 0.5 * (q_curr + qg)
        candidates: list[np.ndarray] = []
        for j2_min in (0.45, 0.70, 0.95):
            for j3_max in (-0.35, -0.60):
                for j5_max in (-0.30, -0.10, 0.05):
                    v = q_mid.copy()
                    v[1] = max(v[1], j2_min)
                    v[2] = min(v[2], j3_max)
                    v[4] = min(v[4], j5_max)
                    candidates.append(np.clip(v, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02))
        # Also try pure wrist-up at start / midpoint / goal
        for base_q in (q_curr, q_mid, qg):
            for j5_v in (-0.35, -0.15):
                v = base_q.copy()
                v[4] = min(v[4], j5_v)
                candidates.append(np.clip(v, JOINT_LOWER + 0.02, JOINT_UPPER - 0.02))

        best_via = None
        best_len = float("inf")
        for v in candidates:
            if not evaluate_handle_clearance(v, cfg)["safe"]:
                continue
            if _dense_path_safe(q_curr, v, cfg, samples=16) and _dense_path_safe(v, qg, cfg, samples=16):
                plen = float(np.linalg.norm(v - q_curr) + np.linalg.norm(qg - v))
                if plen < best_len:
                    best_len = plen
                    best_via = v
        if best_via is not None:
            keyframes.append(best_via)
            via_count = 1
        else:
            # Fallback: walk statefully along the safe boundary from q_curr to qg
            prev = q_curr.copy()
            for a in np.linspace(0.05, 0.95, 18):
                raw_wp = (1.0 - a) * q_curr + a * qg
                proj, _ = project_pose_to_handle_safe(raw_wp, cfg, q_prev6=prev, max_step_rad=0.14)
                keyframes.append(proj)
                prev = proj
            via_count = len(keyframes) - (2 if escaped_start else 1)

    keyframes.append(qg)
    # Give detour trajectories slightly more steps for gentle, unhurried motion
    eff_steps = steps if (not escaped_start and via_count == 0) else int(round(steps * 1.35))
    traj = _resample_polyline(keyframes, eff_steps)
    # Evaluate minimum clearance along planned trajectory (excluding initial frame if started inside)
    check_slice = traj[max(1, eff_steps // 5):] if escaped_start else traj
    min_c = min(float(evaluate_handle_clearance(q, cfg)["min_clearance_m"]) for q in check_slice)
    return traj, {
        "detour": bool(escaped_start or via_count > 0),
        "escaped_start": escaped_start,
        "via_count": via_count,
        "initial_clearance_m": float(raw_path["min_clearance_m"]),
        "min_clearance_m": min_c,
        "reason": raw_path["reason"],
    }
