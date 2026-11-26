"""Mutable episode review tags sidecar, separate from raw data and save_receipts.

Mirrors piper_ms_teleop/episode_tags.py:
- Quality values: 'usable' (可用) / 'unusable' (不可用)
- Preset issue tags: 画面异常 / 机械臂异常 / 任务未完成 / 操作失误 / 夹爪异常 / 其他问题
- Optimistic revision locking + fcntl file lock + atomic os.replace + audit history.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Any

QUALITY = {"usable": "可用", "unusable": "不可用"}
LEGACY_QUALITY = {"unreviewed": "usable", "review": "unusable", "problem": "unusable"}
PRESETS = ("画面异常", "机械臂异常", "任务未完成", "操作失误", "夹爪异常", "其他问题")
NAME = "episode_tags.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _read_raw(path: Path) -> dict[str, Any]:
    if path.is_symlink() or (path.name == NAME and path.stat().st_size > 4 * 1024 * 1024):
        raise ValueError("标签文件不安全或过大")
    return json.loads(path.read_text(encoding="utf-8"))


def read_tags(root: str | Path, *, normalize: bool = True) -> dict[str, Any]:
    ds_root = Path(root)
    path = ds_root / NAME
    if not path.exists() and not path.is_symlink():
        path = ds_root / "meta" / NAME
    if not path.exists() and not path.is_symlink():
        return {"schema_version": 1, "default_quality": "usable", "episodes": {}, "history": []}

    doc = _read_raw(path)
    if (
        not isinstance(doc, dict)
        or doc.get("schema_version") != 1
        or not isinstance(doc.get("episodes"), dict)
        or not isinstance(doc.get("history"), list)
    ):
        raise ValueError("标签文件格式无效")

    valid_qualities = set(QUALITY.keys()) | set(LEGACY_QUALITY.keys())
    for key, row in doc["episodes"].items():
        if (
            not str(key).isdigit()
            or not isinstance(row, dict)
            or str(int(key)) != str(key)
            or int(row.get("episode_index", -1)) != int(key)
            or row.get("quality") not in valid_qualities
            or not isinstance(row.get("tags"), list)
            or any(not isinstance(t, str) for t in row["tags"])
            or not isinstance(row.get("note", ""), str)
            or not isinstance(row.get("actor", ""), str)
            or not isinstance(row.get("revision", 0), int)
        ):
            raise ValueError(f"标签记录格式无效：episode {key}")
        if normalize and row["quality"] in LEGACY_QUALITY:
            row["quality"] = LEGACY_QUALITY[row["quality"]]
    return doc


@contextmanager
def _lock_tags(root: Path):
    lock_path = root / f".{NAME}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _normalize_tags(raw_tags: list[str] | tuple[str, ...] | None) -> list[str]:
    if not raw_tags:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in raw_tags:
        s = str(t).strip()
        if not s or s in seen:
            continue
        if len(s) > 64:
            raise ValueError(f"单个标签过长（超过 64 字符）：{s[:20]}…")
        cleaned.append(s)
        seen.add(s)
    return cleaned


def save_tags(
    root: str | Path,
    episode_index: int,
    *,
    quality: str = "usable",
    tags: list[str] | tuple[str, ...] | None = None,
    note: str = "",
    actor: str = "",
    expected_revision: int | None = None,
) -> dict[str, Any]:
    ds_root = Path(root).expanduser().resolve()
    if not (ds_root / "meta/info.json").is_file():
        raise ValueError(f"目标目录不是有效的 LeRobot v2.1 数据集：{ds_root}")
    if (ds_root / "delivery_snapshot.json").exists():
        raise ValueError("不可变上传快照上的标签只读，请在原采集目录标记")
    if quality not in QUALITY:
        raise ValueError(f"不支持的质量值：{quality!r}（仅允许 {list(QUALITY.keys())}）")

    ep_idx = int(episode_index)
    if ep_idx < 0:
        raise ValueError("episode_index 必须为非负整数")
    norm_tags = _normalize_tags(tags)
    note_str = str(note or "").strip()
    if len(note_str) > 2000:
        raise ValueError("备注长度不能超过 2000 字符")
    actor_str = str(actor or "").strip() or "operator"
    if len(actor_str) > 128:
        raise ValueError("标记人名称过长")

    with _lock_tags(ds_root):
        doc = read_tags(ds_root, normalize=True)
        key = str(ep_idx)
        prev = doc["episodes"].get(key)
        cur_rev = int(prev["revision"]) if prev else 0
        if expected_revision is not None and int(expected_revision) != cur_rev:
            raise ValueError(f"标签已被其他操作更新（当前版本 {cur_rev}，预期 {expected_revision}），请先刷新标签")

        now_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {
            "episode_index": ep_idx,
            "quality": quality,
            "tags": norm_tags,
            "note": note_str,
            "actor": actor_str,
            "revision": cur_rev + 1,
            "updated_at": now_utc,
        }
        doc["episodes"][key] = row
        doc["history"].append(dict(row))
        doc["history"] = doc["history"][-500:]
        _atomic_json(ds_root / NAME, doc)
        return row


def summarize_dataset_quality(root: str | Path, total_episodes: int) -> dict[str, Any]:
    """Summarize usable/unusable counts across all episodes of a dataset."""
    try:
        doc = read_tags(root, normalize=True)
    except Exception:
        doc = {"episodes": {}}
    ep_map = doc.get("episodes") or {}
    unusable = 0
    tagged = 0
    for ep_idx in range(max(0, int(total_episodes))):
        row = ep_map.get(str(ep_idx))
        if row:
            tagged += 1
            if row.get("quality") == "unusable":
                unusable += 1
    usable = max(0, int(total_episodes) - unusable)
    return {
        "total_episodes": int(total_episodes),
        "usable_episodes": usable,
        "unusable_episodes": unusable,
        "tagged_episodes": tagged,
        "all_unusable": bool(total_episodes > 0 and unusable >= total_episodes),
        "episodes": ep_map,
    }
