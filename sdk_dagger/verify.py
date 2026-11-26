"""LeRobot v2.1 read-back dataset verification and durable save receipts.

Aligned with piper_ms_teleop's verify_episode / validate_local_files / verify_dataset:
- Validates all metadata-referenced Parquet & 3-camera MP4 files exist locally.
- Reads back Parquet rows to verify shapes, finite joint/action values, strictly
  increasing timestamps, valid is_dagger labels (when present), and large-jump warnings.
- Decodes first, middle, and last RGB frames from each MP4 camera video.
- Writes atomic, fsync'd verification receipts under <dataset_root>/save_receipts/.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
from pathlib import Path
import sys
import threading
from typing import Any

import subprocess
import numpy as np
import pyarrow.parquet as pq

RECEIPT_DIR_NAME = "save_receipts"
_VIDEO_DECODE_LOCK = threading.Lock()
SCHEMA_VERSION = 1


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def _fsync_files(paths: list[Path]) -> None:
    for p in paths:
        if not p.is_file():
            continue
        try:
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass


def _load_meta(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info_path = root / "meta/info.json"
    ep_path = root / "meta/episodes.jsonl"
    if not info_path.is_file():
        raise RuntimeError(f"缺少元数据文件：{info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes: list[dict[str, Any]] = []
    if ep_path.is_file():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(json.loads(line))
    return info, episodes


def _video_keys_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features") or {}
    return [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video"]


def _episode_file_paths(root: Path, info: dict[str, Any], ep_idx: int, *, include_videos: bool = True) -> dict[str, Any]:
    chunk_size = max(1, int(info.get("chunks_size", 1000)))
    chunk_idx = ep_idx // chunk_size
    data_tpl = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    video_tpl = str(info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"))
    parquet_path = root / data_tpl.format(episode_chunk=chunk_idx, episode_index=ep_idx)
    video_paths: dict[str, Path] = {}
    if include_videos:
        for vkey in _video_keys_from_info(info):
            video_paths[vkey] = root / video_tpl.format(
                episode_chunk=chunk_idx, video_key=vkey, episode_index=ep_idx
            )
    return {"parquet": parquet_path, "videos": video_paths}


def validate_local_files(root: str | Path, *, require_videos: bool | None = None) -> dict[str, Any]:
    """Check every referenced metadata, Parquet, and video file exists and is non-empty."""
    ds_root = Path(root).expanduser().resolve()
    info, episodes = _load_meta(ds_root)
    if str(info.get("codebase_version", "")) != "v2.1":
        raise RuntimeError(f"不支持的数据集版本：{info.get('codebase_version')!r}（预期 v2.1）")

    expected_ep = int(info.get("total_episodes", 0))
    expected_frames = int(info.get("total_frames", 0))
    if expected_ep <= 0 or expected_frames <= 0:
        raise RuntimeError(f"数据集为空（total_episodes={expected_ep}, total_frames={expected_frames}）")
    if len(episodes) != expected_ep:
        raise RuntimeError(f"episodes.jsonl 段数（{len(episodes)}）与 info.json（{expected_ep}）不一致")

    if require_videos is None:
        require_videos = int(info.get("total_videos", 0)) > 0

    missing: list[str] = []
    empty: list[str] = []
    sum_frames = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        sum_frames += int(ep.get("length", 0))
        paths = _episode_file_paths(ds_root, info, ep_idx, include_videos=require_videos)
        pq_path = paths["parquet"]
        if not pq_path.is_file():
            missing.append(str(pq_path.relative_to(ds_root)))
        elif pq_path.stat().st_size == 0:
            empty.append(str(pq_path.relative_to(ds_root)))
        for vp in paths["videos"].values():
            if not vp.is_file():
                missing.append(str(vp.relative_to(ds_root)))
            elif vp.stat().st_size == 0:
                empty.append(str(vp.relative_to(ds_root)))

    if sum_frames != expected_frames:
        raise RuntimeError(f"episodes.jsonl 总帧数（{sum_frames}）与 info.json（{expected_frames}）不一致")
    if missing:
        raise RuntimeError(f"本地数据文件缺失（{len(missing)} 个）：{', '.join(missing[:10])}")
    if empty:
        raise RuntimeError(f"本地数据文件为空（{len(empty)} 个）：{', '.join(empty[:10])}")

    return {"info": info, "episodes": episodes, "require_videos": require_videos}


def _decode_rgb_frame_ffmpeg(video_path: Path, *, width: int, height: int, seek_args: list[str], take_last: bool = False) -> np.ndarray:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", *seek_args, "-i", str(video_path)]
    if not take_last:
        cmd += ["-frames:v", "1"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg 解码失败：{video_path.name}: {err or proc.returncode}")
    frame_bytes = width * height * 3
    buf = proc.stdout
    if len(buf) < frame_bytes:
        raise RuntimeError(f"ffmpeg 未解码到完整 RGB 帧：{video_path.name}（got {len(buf)} bytes < {frame_bytes}）")
    raw = buf[-frame_bytes:] if take_last else buf[:frame_bytes]
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def _verify_video_via_python_subprocess(
    video_path: Path, *, expected_frames: int, expected_hw: tuple[int, int] | None
) -> dict[str, Any]:
    """Isolated child-process fallback using cv2/av when system ffprobe/ffmpeg are absent."""
    script = """
import json, sys
import numpy as np
vpath, exp_frames = sys.argv[1], int(sys.argv[2])
exp_h, exp_w = int(sys.argv[3]), int(sys.argv[4])
sample_indices = sorted({0, max(0, (exp_frames - 1) // 2), max(0, exp_frames - 1)})
try:
    import cv2
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError("cv2 cannot open video")
    try:
        reported = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if reported > 0 and abs(reported - exp_frames) > 1:
            raise RuntimeError(f"视频帧数与 Parquet 不一致（视频={reported}，Parquet={exp_frames}）")
        means, shape = [], None
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, fr = cap.read()
            if not ok or fr is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
                seq_fr, cnt = None, 0
                while True:
                    ok2, fr2 = cap.read()
                    if not ok2 or fr2 is None:
                        break
                    if cnt == idx:
                        seq_fr = fr2
                    cnt += 1
                if seq_fr is None:
                    raise RuntimeError(f"视频第 {idx} 帧解码失败（实际 {cnt} 帧，预期 {exp_frames} 帧）")
                fr = seq_fr
                reported = cnt
            h, w, c = fr.shape
            if c != 3:
                raise RuntimeError(f"视频帧通道异常：{fr.shape}")
            if exp_h > 0 and exp_w > 0 and (h, w) != (exp_h, exp_w):
                raise RuntimeError(f"视频分辨率不匹配：实测={(h, w)} 预期={(exp_h, exp_w)}")
            shape = [int(h), int(w), 3]
            means.append(float(fr.mean()))
        print(json.dumps({"reported_frames": reported or exp_frames, "sampled_indices": sample_indices, "shape": shape, "mean_luma": round(float(np.mean(means)), 2), "all_dark": bool(means and all(m < 2.0 for m in means))}))
    finally:
        cap.release()
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"""
    eh, ew = expected_hw if expected_hw is not None else (0, 0)
    proc = subprocess.run(
        [sys.executable, "-c", script, str(video_path), str(int(expected_frames)), str(eh), str(ew)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    out = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        try:
            err = json.loads(out).get("error") if out else None
        except Exception:
            err = None
        err = err or proc.stderr.decode("utf-8", errors="replace").strip() or f"exit={proc.returncode}"
        raise RuntimeError(f"视频校验失败：{video_path.name}: {err}")
    res = json.loads(out)
    res["path"] = str(video_path)
    return res


def _verify_video_file(video_path: Path, *, expected_frames: int, expected_hw: tuple[int, int] | None, fps: float = 30.0) -> dict[str, Any]:
    """Verify MP4 packet count, dimensions, and first/middle/last RGB decode via isolated ffprobe/ffmpeg."""
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"视频文件缺失或为空：{video_path}")

    if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
        return _verify_video_via_python_subprocess(
            video_path, expected_frames=expected_frames, expected_hw=expected_hw
        )

    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=width,height,nb_read_packets,nb_frames,codec_name",
        "-of", "json", str(video_path),
    ]
    probe = subprocess.run(probe_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if probe.returncode != 0:
        err = probe.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffprobe 检测视频失败：{video_path.name}: {err or probe.returncode}")
    streams = (json.loads(probe.stdout.decode("utf-8")) or {}).get("streams") or []
    if not streams:
        raise RuntimeError(f"视频文件不含视频流：{video_path.name}")
    st = streams[0]
    w = int(st.get("width") or 0)
    h = int(st.get("height") or 0)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"视频分辨率异常：{video_path.name} ({w}x{h})")
    if expected_hw is not None and (h, w) != expected_hw:
        raise RuntimeError(f"视频分辨率不匹配：{video_path.name} 实测={(h, w)} 预期={expected_hw}")

    pkt_cnt = int(st.get("nb_read_packets") or st.get("nb_frames") or 0)
    if abs(pkt_cnt - expected_frames) > 1:
        raise RuntimeError(
            f"视频帧数与 Parquet 不一致：{video_path.name}（视频={pkt_cnt}，Parquet={expected_frames}）"
        )

    means: list[float] = []
    sampled: list[int] = [0]
    f0 = _decode_rgb_frame_ffmpeg(video_path, width=w, height=h, seek_args=[], take_last=False)
    means.append(float(f0.mean()))

    if expected_frames >= 3:
        mid_idx = (expected_frames - 1) // 2
        mid_s = max(0.0, mid_idx / max(fps, 1.0))
        fm = _decode_rgb_frame_ffmpeg(video_path, width=w, height=h, seek_args=["-ss", f"{mid_s:.3f}"], take_last=False)
        means.append(float(fm.mean()))
        sampled.append(mid_idx)

    if expected_frames >= 2:
        tail_s = min(0.25, max(0.05, 3.0 / max(fps, 1.0)))
        fl = _decode_rgb_frame_ffmpeg(video_path, width=w, height=h, seek_args=["-sseof", f"-{tail_s:.3f}"], take_last=True)
        means.append(float(fl.mean()))
        sampled.append(expected_frames - 1)

    return {
        "path": str(video_path),
        "reported_frames": pkt_cnt,
        "sampled_indices": sampled,
        "shape": [h, w, 3],
        "mean_luma": round(float(np.mean(means)), 2) if means else 0.0,
        "all_dark": bool(means and all(m < 2.0 for m in means)),
    }


def verify_episode_v21(
    root: str | Path,
    ep_idx: int,
    *,
    expected_frames: int | None = None,
    check_video: bool | None = None,
    jump_warn_deg: float = 35.0,
) -> dict[str, Any]:
    """Read back and verify one LeRobot v2.1 episode (Parquet + 3-camera RGB MP4s)."""
    ds_root = Path(root).expanduser().resolve()
    info, episodes = _load_meta(ds_root)
    ep_row = next((e for e in episodes if int(e["episode_index"]) == int(ep_idx)), None)
    if ep_row is None:
        raise RuntimeError(f"episodes.jsonl 中未找到 episode_{ep_idx:06d}")

    ep_len = int(ep_row.get("length", 0))
    if expected_frames is not None and ep_len != int(expected_frames):
        raise RuntimeError(f"episode_{ep_idx:06d} 帧数不一致：meta={ep_len} vs expected={expected_frames}")
    if ep_len <= 0:
        raise RuntimeError(f"episode_{ep_idx:06d} 帧数为 0")

    if check_video is None:
        check_video = int(info.get("total_videos", 0)) > 0

    paths = _episode_file_paths(ds_root, info, int(ep_idx), include_videos=check_video)
    parquet_path: Path = paths["parquet"]
    if not parquet_path.is_file() or parquet_path.stat().st_size == 0:
        raise RuntimeError(f"Parquet 文件缺失或为空：{parquet_path}")

    table = pq.read_table(parquet_path, use_threads=False)
    if len(table) != ep_len:
        raise RuntimeError(
            f"episode_{ep_idx:06d} Parquet 行数（{len(table)}）与元数据（{ep_len}）不一致"
        )

    required_cols = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index")
    for col in required_cols:
        if col not in table.column_names:
            raise RuntimeError(f"episode_{ep_idx:06d} Parquet 缺少必要列：{col}")

    def _list_col_2d(col) -> np.ndarray:
        arr = col.combine_chunks()
        n = len(arr)
        if n == 0:
            return np.empty((0, 0), dtype=np.float64)
        vals = np.asarray(arr.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        if vals.size % n != 0:
            raise RuntimeError(f"episode_{ep_idx:06d} 变长向量列异常（rows={n}, vals={vals.size}）")
        return vals.reshape(n, vals.size // n)

    state = _list_col_2d(table["observation.state"])
    action = _list_col_2d(table["action"])
    ts = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1)
    fidx = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1)
    eidx = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1)

    features = info.get("features") or {}
    exp_state_dim = int((features.get("observation.state", {}).get("shape") or [14])[0])
    exp_act_dim = int((features.get("action", {}).get("shape") or [14])[0])
    if state.ndim != 2 or state.shape != (ep_len, exp_state_dim):
        raise RuntimeError(f"episode_{ep_idx:06d} observation.state 形状异常：{state.shape}")
    if action.ndim != 2 or action.shape != (ep_len, exp_act_dim):
        raise RuntimeError(f"episode_{ep_idx:06d} action 形状异常：{action.shape}")
    if not np.isfinite(state).all():
        raise RuntimeError(f"episode_{ep_idx:06d} observation.state 包含 NaN/Inf")
    if not np.isfinite(action).all():
        raise RuntimeError(f"episode_{ep_idx:06d} action 包含 NaN/Inf")
    if not np.isfinite(ts).all():
        raise RuntimeError(f"episode_{ep_idx:06d} timestamp 包含 NaN/Inf")

    fps = max(1.0, float(info.get("fps", 30)))
    warnings: list[str] = []
    if ep_len >= 2:
        dt = np.diff(ts)
        if np.any(dt <= 0):
            raise RuntimeError(f"episode_{ep_idx:06d} timestamp 非严格递增（min_dt={float(dt.min()):.6f}）")
        max_dt_thresh = max(0.25, 2.5 / fps)
        if float(dt.max()) > max_dt_thresh:
            warnings.append(f"时间戳间隔偏大（max_dt={float(dt.max()):.3f}s）")
    if not np.array_equal(fidx, np.arange(ep_len, dtype=np.int64)):
        raise RuntimeError(f"episode_{ep_idx:06d} frame_index 不连续")
    if not np.all(eidx == int(ep_idx)):
        raise RuntimeError(f"episode_{ep_idx:06d} episode_index 列值与编号不一致")

    if "is_dagger" in table.column_names:
        is_dag = np.asarray(table["is_dagger"].to_numpy(zero_copy_only=False), dtype=np.int64).reshape(-1)
        if not np.isin(is_dag, (0, 1)).all():
            raise RuntimeError(f"episode_{ep_idx:06d} is_dagger 包含非法值（仅允许 0/1）")

    # Joint step sanity check (arm joints 0..5 and 7..12)
    arm_cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12] if exp_state_dim >= 13 else list(range(exp_state_dim))
    max_step_deg = 0.0
    if ep_len >= 2:
        max_step_rad = float(np.max(np.abs(np.diff(state[:, arm_cols], axis=0))))
        max_step_deg = round(math.degrees(max_step_rad), 2)
        if max_step_deg > float(jump_warn_deg):
            warnings.append(f"单帧关节跳变较大（{max_step_deg:.1f}°）")

    video_reports: dict[str, Any] = {}
    checked_paths: list[Path] = [parquet_path]
    if check_video:
        for vkey, vp in paths["videos"].items():
            shape = (features.get(vkey) or {}).get("shape")
            exp_hw = (int(shape[0]), int(shape[1])) if isinstance(shape, list) and len(shape) >= 2 else None
            vrep = _verify_video_file(vp, expected_frames=ep_len, expected_hw=exp_hw, fps=fps)
            video_reports[vkey] = vrep
            checked_paths.append(vp)
            if vrep.get("all_dark"):
                warnings.append(f"相机 {vkey} 采样帧全黑（mean_luma={vrep['mean_luma']}）")

    return {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "dataset": ds_root.name,
        "episode_index": int(ep_idx),
        "frames": ep_len,
        "fps": int(round(fps)),
        "checked_video": bool(check_video),
        "max_joint_step_deg": max_step_deg,
        "warnings": warnings,
        "videos": {k: {"reported_frames": v["reported_frames"], "shape": v["shape"], "mean_luma": v["mean_luma"]}
                   for k, v in video_reports.items()},
        "files": [str(p.relative_to(ds_root)) for p in checked_paths],
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


def receipt_path_for(root: str | Path, ep_idx: int) -> Path:
    return Path(root) / RECEIPT_DIR_NAME / f"episode_{int(ep_idx):06d}.json"


def write_save_receipt(root: str | Path, ep_idx: int, verification: dict[str, Any]) -> Path:
    ds_root = Path(root).expanduser().resolve()
    rel_files = verification.get("files") or []
    _fsync_files([ds_root / f for f in rel_files] + [ds_root / "meta/info.json", ds_root / "meta/episodes.jsonl"])
    rpath = receipt_path_for(ds_root, int(ep_idx))
    return _atomic_write_json(rpath, verification)


def delete_save_receipt(root: str | Path, ep_idx: int) -> None:
    rpath = receipt_path_for(root, int(ep_idx))
    if rpath.is_file():
        try:
            rpath.unlink()
        except OSError:
            pass


def read_save_receipts(root: str | Path) -> dict[int, dict[str, Any]]:
    rdir = Path(root) / RECEIPT_DIR_NAME
    if not rdir.is_dir():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for fp in sorted(rdir.glob("episode_*.json")):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and "episode_index" in doc:
                out[int(doc["episode_index"])] = doc
        except Exception:
            continue
    return out


def dataset_receipt_status(root: str | Path, total_episodes: int | None = None) -> dict[str, Any]:
    ds_root = Path(root)
    if total_episodes is None:
        info_path = ds_root / "meta/info.json"
        try:
            total_episodes = int(json.loads(info_path.read_text(encoding="utf-8")).get("total_episodes", 0))
        except Exception:
            total_episodes = 0
    receipts = read_save_receipts(ds_root)
    passed_cnt = sum(1 for r in receipts.values() if r.get("passed") is True)
    failed_cnt = sum(1 for r in receipts.values() if r.get("passed") is False)
    warn_cnt = sum(len(r.get("warnings") or []) for r in receipts.values())
    if failed_cnt > 0:
        state = "failed"
    elif total_episodes > 0 and passed_cnt >= total_episodes:
        state = "verified"
    elif passed_cnt > 0:
        state = "partial"
    else:
        state = "unverified"
    return {
        "state": state,
        "total_episodes": int(total_episodes),
        "passed_episodes": passed_cnt,
        "failed_episodes": failed_cnt,
        "warnings": warn_cnt,
        "receipts": receipts,
    }


def verify_dataset_v21(
    root: str | Path,
    *,
    write_receipts: bool = False,
    check_video: bool | None = None,
) -> dict[str, Any]:
    """Validate all local files and read-back verify every episode in a LeRobot v2.1 dataset."""
    ds_root = Path(root).expanduser().resolve()
    val = validate_local_files(ds_root, require_videos=check_video)
    episodes = val["episodes"]
    req_vid = val["require_videos"]

    ep_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        ep_len = int(ep.get("length", 0))
        try:
            rep = verify_episode_v21(
                ds_root, ep_idx, expected_frames=ep_len, check_video=req_vid
            )
            ep_reports.append(rep)
            warnings.extend(f"ep{ep_idx}: {w}" for w in (rep.get("warnings") or []))
            if write_receipts:
                write_save_receipt(ds_root, ep_idx, rep)
        except Exception as exc:
            err_msg = f"episode_{ep_idx:06d}: {exc}"
            errors.append(err_msg)
            fail_rep = {
                "schema_version": SCHEMA_VERSION,
                "passed": False,
                "dataset": ds_root.name,
                "episode_index": ep_idx,
                "frames": ep_len,
                "error": str(exc),
                "warnings": [],
                "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            }
            ep_reports.append(fail_rep)
            if write_receipts:
                write_save_receipt(ds_root, ep_idx, fail_rep)

    return {
        "passed": len(errors) == 0 and len(ep_reports) > 0,
        "dataset": ds_root.name,
        "root": str(ds_root),
        "episodes": len(ep_reports),
        "frames": int(val["info"].get("total_frames", 0)),
        "warnings": warnings,
        "errors": errors,
        "episode_reports": ep_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读回验证本地 LeRobot v2.1 数据集（对齐 piper_ms_teleop 核查标准）")
    parser.add_argument("target", type=Path, help="单个数据集目录，或配合 --all 传入 data/ 根目录")
    parser.add_argument("--all", action="store_true", help="扫描目录下全部 LeRobot v2.1 数据集")
    parser.add_argument("--merged-only", action="store_true", help="配合 --all 仅核查 *_merged 数据集")
    parser.add_argument("--write-receipts", action="store_true", help="将校验结果写入 save_receipts/ 回执文件")
    parser.add_argument("--no-video", action="store_true", help="跳过 MP4 视频帧解码检查")
    args = parser.parse_args(argv)

    targets: list[Path] = []
    if args.all:
        if not args.target.is_dir():
            print(f"[失败] 目录不存在：{args.target}", file=sys.stderr)
            return 2
        for d in sorted(args.target.iterdir()):
            if not d.is_dir() or not (d / "meta/info.json").is_file():
                continue
            if args.merged_only and not d.name.endswith("_merged"):
                continue
            targets.append(d)
    else:
        targets.append(args.target)

    if not targets:
        print("未找到任何 LeRobot v2.1 数据集。")
        return 0

    ok_cnt = 0
    fail_cnt = 0
    warn_datasets = 0
    for ds in targets:
        try:
            rep = verify_dataset_v21(
                ds,
                write_receipts=args.write_receipts,
                check_video=(False if args.no_video else None),
            )
            if rep["passed"]:
                ok_cnt += 1
                wtxt = f" [警告: {'; '.join(rep['warnings'])}]" if rep["warnings"] else ""
                if rep["warnings"]:
                    warn_datasets += 1
                print(f"OK   {ds.name}: {rep['episodes']} 段 / {rep['frames']} 帧{wtxt}")
            else:
                fail_cnt += 1
                print(f"FAIL {ds.name}: {'; '.join(rep['errors'])}")
        except Exception as exc:
            fail_cnt += 1
            print(f"FAIL {ds.name}: {exc}")

    print(f"\n汇总：共检查 {len(targets)} 个数据集，通过 {ok_cnt} 个，失败 {fail_cnt} 个，含警告 {warn_datasets} 个。")
    return 0 if fail_cnt == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
