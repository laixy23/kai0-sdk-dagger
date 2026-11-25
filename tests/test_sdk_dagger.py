"""Zero-hardware end-to-end tests for kai0-sdk-dagger (zero ROS, LeRobot v2.1 default + HDF5 fallback)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "operator"))

LEROBOT_VENDOR = Path("/home/ubuntu/dev/kai0-task-a/vendor/lerobot")
if LEROBOT_VENDOR.is_dir() and str(LEROBOT_VENDOR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_VENDOR))

from sdk_dagger.app import SDKDaggerApp
from sdk_dagger.policy import TemporalSmoothingScheduler
from sdk_dagger.prepare import load_config
from sdk_dagger.storage import (
    AsyncVideoExporter,
    CAMERA_NAMES,
    FrameSample,
    LEROBOT_CAMERA_KEYS,
    LeRobotV21DatasetWriter,
    delete_episode,
    next_episode_idx,
    save_episode,
)


def _fast_mock_cfg(tmp_dir: Path) -> dict:
    cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
    cfg["recording"]["dataset_dir"] = str(tmp_dir)
    cfg["recording"]["inference_tail_s"] = 0.08
    cfg["recording"]["dagger_preroll_s"] = 0.08
    cfg["recording"]["save_confirm_timeout_s"] = 1.5
    cfg["inference"]["publish_rate"] = 50
    cfg["inference"]["inference_rate"] = 10
    return cfg


def _make_samples(count: int, offset: float = 0.0, *, bgr_color: tuple[int, int, int] | None = None) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for i in range(count):
        q = np.full(14, offset + float(i) * 0.01, dtype=np.float32)
        if bgr_color is None:
            imgs = {cam: np.full((64, 80, 3), (i * 20) % 255, dtype=np.uint8) for cam in CAMERA_NAMES}
        else:
            frame = np.zeros((64, 80, 3), dtype=np.uint8)
            frame[:, :, 0], frame[:, :, 1], frame[:, :, 2] = bgr_color
            imgs = {cam: frame.copy() for cam in CAMERA_NAMES}
        samples.append(FrameSample(time.time() + i * 0.03, q, np.zeros(14, np.float32), np.zeros(14, np.float32), imgs))
    return samples


class SchedulerTests(unittest.TestCase):
    def test_temporal_smoothing_blends_and_reset_invalidates_old_generation(self):
        sched = TemporalSmoothingScheduler(chunk_size=10, latency_k=2, min_smooth_steps=3)
        gen0 = sched.generation
        c1 = np.zeros((10, 14), dtype=np.float64)
        self.assertTrue(sched.update(c1, generation=gen0))
        self.assertEqual(sched.remaining(), 10)
        _ = sched.pop()  # k=1 (< latency_k=2) -> StreamActionBuffer drops min(1, 2)=1 step, NOT 2!
        c2 = np.ones((10, 14), dtype=np.float64)
        self.assertTrue(sched.update(c2, generation=gen0))
        # Exact StreamActionBuffer blends across full 9-step overlap: w_old = linspace(1.0, 0.0, 9) -> 0.0, 0.125, 0.25
        s0 = sched.pop()
        s1 = sched.pop()
        s2 = sched.pop()
        self.assertAlmostEqual(s0[0], 0.0, places=6)
        self.assertAlmostEqual(s1[0], 0.125, places=6)
        self.assertAlmostEqual(s2[0], 0.25, places=6)

        gen1 = sched.reset()
        self.assertEqual(sched.remaining(), 0)
        self.assertFalse(sched.update(c2, generation=gen0))
        self.assertTrue(sched.update(c2, generation=gen1))

    def test_rejects_nan_or_wrong_shape(self):
        sched = TemporalSmoothingScheduler(chunk_size=10)
        bad = np.zeros((10, 14))
        bad[3, 2] = np.nan
        with self.assertRaises(ValueError):
            sched.update(bad, generation=sched.generation)
        with self.assertRaises(ValueError):
            sched.update(np.zeros((8, 14)), generation=sched.generation)


class StorageTests(unittest.TestCase):
    def test_lerobot_v21_schema_multi_episode_delete_and_official_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cloth_lerobot_dagger"
            exporter = AsyncVideoExporter()
            try:
                writer = LeRobotV21DatasetWriter(
                    root, prompt="Flatten and fold the cloth.", fps=30, video_exporter=exporter
                )
                # Episode 0 (5 frames, pure BGR red -> RGB (1, 0, 0))
                res0 = writer.save_episode(
                    _make_samples(6, offset=0.0, bgr_color=(0, 0, 255)),
                    export_video=True, after_inference_episode=0,
                )
                # Episode 1 (7 frames)
                res1 = writer.save_episode(_make_samples(8, offset=0.2), export_video=True, after_inference_episode=1)
                # Attempt 2 & Attempt 3 -> both deleted consecutively!
                res2 = writer.save_episode(_make_samples(5, offset=0.5), export_video=True, after_inference_episode=1)
                writer.delete_last_episode(res2)
                res3 = writer.save_episode(_make_samples(5, offset=0.6), export_video=True, after_inference_episode=1)
                writer.delete_last_episode(res3)

                self.assertFalse(res2["data_path"].exists())
                self.assertFalse(res3["data_path"].exists())
                for vp in res2["video_paths"] + res3["video_paths"]:
                    self.assertFalse(vp.exists())
                self.assertEqual(writer.next_episode_index, 2)
                self.assertEqual(writer.total_frames, 12)

                # Re-instantiate writer from disk to verify timeline recovery across restarts
                writer = LeRobotV21DatasetWriter(
                    root, prompt="Flatten and fold the cloth.", fps=30, video_exporter=exporter
                )
                # Save Attempt 4 as contiguous Episode 2 (3 frames) after Inference 2
                res2_new = writer.save_episode(_make_samples(4, offset=0.8), export_video=True, after_inference_episode=2)
                exporter.wait_all(announce=False)
            finally:
                exporter.close(wait=True)

            self.assertTrue(res0["saved"] and res1["saved"] and res2_new["saved"])
            self.assertEqual(res2_new["episode_idx"], 2)
            self.assertEqual(res2_new["attempt_idx"], 4)
            self.assertEqual(res2_new["deleted_attempts_before"], [2, 3])

            # Verify episodes.jsonl & session_timeline.jsonl record the exact deleted gap
            eps_rows = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([r["episode_index"] for r in eps_rows], [0, 1, 2])
            self.assertEqual([r["attempt_index"] for r in eps_rows], [0, 1, 4])
            self.assertEqual(eps_rows[2]["deleted_attempts_before"], [2, 3])
            self.assertEqual(eps_rows[2]["after_inference_episode"], 2)

            timeline_rows = [json.loads(line) for line in (root / "meta/session_timeline.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(
                [(r["event"], r["attempt_index"], r["episode_index"]) for r in timeline_rows],
                [
                    ("saved", 0, 0),
                    ("saved", 1, 1),
                    ("saved", 2, 2),
                    ("deleted", 2, 2),
                    ("saved", 3, 2),
                    ("deleted", 3, 2),
                    ("saved", 4, 2),
                ],
            )

            info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["robot_type"], "agilex")
            self.assertEqual(info["total_episodes"], 3)
            self.assertEqual(info["total_frames"], 15)
            self.assertEqual(info["total_videos"], 9)
            self.assertEqual(info["splits"], {"train": "0:3"})

            # Check parquet schema, action[t] == qpos[t+1], and global index continuity
            t0 = pq.read_table(root / "data/chunk-000/episode_000000.parquet")
            t1 = pq.read_table(root / "data/chunk-000/episode_000001.parquet")
            t2 = pq.read_table(root / "data/chunk-000/episode_000002.parquet")
            self.assertEqual(t0["index"].to_pylist(), list(range(0, 5)))
            self.assertEqual(t1["index"].to_pylist(), list(range(5, 12)))
            self.assertEqual(t2["index"].to_pylist(), list(range(12, 15)))
            s0 = np.asarray(t0["observation.state"].to_pylist(), dtype=np.float32)
            a0 = np.asarray(t0["action"].to_pylist(), dtype=np.float32)
            self.assertEqual(s0.shape, (5, 14))
            self.assertTrue(np.allclose(a0[:-1], s0[1:], atol=1e-6))

            # Check all 3x3 MP4s exist with exact frame counts
            for ep_idx, expected_n in ((0, 5), (1, 7), (2, 3)):
                for lr_key in LEROBOT_CAMERA_KEYS.values():
                    vp = root / f"videos/chunk-000/{lr_key}/episode_{ep_idx:06d}.mp4"
                    self.assertTrue(vp.exists(), f"Missing {vp}")
                    cap = cv2.VideoCapture(str(vp))
                    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    self.assertEqual(cnt, expected_n)

            # Verify real LeRobot v2.1 loader compatibility (Kai0 vendor lerobot)
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

            meta = LeRobotDatasetMetadata(repo_id="local/unit_test", root=root, force_cache_sync=False)
            self.assertEqual(meta.total_episodes, 3)
            self.assertEqual(meta.total_frames, 15)
            self.assertEqual(meta.tasks, {0: "Flatten and fold the cloth."})

            ds = LeRobotDataset(
                repo_id="local/unit_test",
                root=root,
                video_backend="pyav",
                delta_timestamps={"action": [i / 30.0 for i in range(4)]},
            )
            self.assertEqual(len(ds), 15)
            item0 = ds[0]
            self.assertEqual(tuple(item0["observation.state"].shape), (14,))
            self.assertEqual(tuple(item0["action"].shape), (4, 14))
            self.assertEqual(item0["task"], "Flatten and fold the cloth.")
            img_rgb = item0["observation.images.top_head"]
            self.assertEqual(tuple(img_rgb.shape), (3, 64, 80))
            # Pure BGR red -> RGB R channel near 1.0, G/B near 0.0
            rgb_means = img_rgb.mean(dim=(1, 2)).tolist()
            self.assertGreater(rgb_means[0], 0.9)
            self.assertLess(rgb_means[1], 0.1)
            self.assertLess(rgb_means[2], 0.1)
            self.assertEqual(int(ds[14]["episode_index"]), 2)
            self.assertEqual(int(ds[14]["index"]), 14)

    def test_hdf5_and_three_mp4_schema_and_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "ds_dagger_hdf5/aloha_mobile_dummy"
            samples = _make_samples(6)

            self.assertEqual(next_episode_idx(out_dir), 0)
            res = save_episode(samples, out_dir, 0, export_video=True, video_fps=30)
            self.assertTrue(res["saved"])
            self.assertEqual(res["frames"], 5)
            self.assertEqual(next_episode_idx(out_dir), 1)

            with h5py.File(res["hdf5_path"], "r") as hf:
                qpos = hf["observations/qpos"][:]
                action = hf["action"][:]
                self.assertEqual(qpos.shape, (5, 14))
                self.assertEqual(action.shape, (5, 14))
                self.assertTrue(np.allclose(action[:-1], qpos[1:], atol=1e-6))

            self.assertEqual(len(res["video_paths"]), 3)
            for vp in res["video_paths"]:
                cap = cv2.VideoCapture(str(vp))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                self.assertEqual(count, 5)

            delete_episode(res["hdf5_path"], res["video_paths"])
            self.assertFalse(res["hdf5_path"].exists())
            self.assertEqual(next_episode_idx(out_dir), 0)

    def test_async_video_exporter_completes_and_cancel_delete_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "async_dagger_hdf5/aloha_mobile_dummy"
            samples = _make_samples(10)

            exporter = AsyncVideoExporter()
            try:
                res0 = save_episode(samples, out_dir, 0, export_video=True, video_fps=30, video_exporter=exporter)
                self.assertTrue(res0["saved"])
                self.assertIsNotNone(res0["video_job"])
                exporter.wait_all(announce=False)
                for vp in res0["video_paths"]:
                    self.assertTrue(vp.exists())

                res1 = save_episode(samples, out_dir, 1, export_video=True, video_fps=30, video_exporter=exporter)
                delete_episode(res1["hdf5_path"], res1["video_paths"], video_job=res1["video_job"])
                self.assertFalse(res1["hdf5_path"].exists())
                for vp in res1["video_paths"]:
                    self.assertFalse(vp.exists())
            finally:
                exporter.close(wait=True)

    def test_merge_round_lerobot_v21_stitches_continuous_and_splits_on_collision_or_delete(self):
        from sdk_dagger.storage import merge_round_lerobot_v21
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            inf_dir = base / "round_inference"
            dag_dir = base / "round_dagger"
            merged_dir = base / "round_merged"

            inf_w = LeRobotV21DatasetWriter(inf_dir, prompt="Organize the clothes.", fps=30)
            dag_w = LeRobotV21DatasetWriter(dag_dir, prompt="Organize the clothes.", fps=30)

            inf_w.save_episode(_make_samples(6, offset=0.0, bgr_color=(0, 0, 255)), export_video=True)   # 5f
            dag_w.save_episode(_make_samples(8, offset=0.2, bgr_color=(0, 255, 0)), export_video=True)   # 7f
            inf_w.save_episode(_make_samples(5, offset=0.5, bgr_color=(255, 0, 0)), export_video=True)   # 4f
            dag_w.save_episode(_make_samples(7, offset=0.7, bgr_color=(0, 255, 255)), export_video=True) # 6f

            # Case A: continuous [inf0, dag0, inf1] with no breaks -> 1 single episode (5+7+4 = 16 frames)
            res_cont = merge_round_lerobot_v21(
                inference_dir=inf_dir, dagger_dir=dag_dir, merged_dir=base / "round_cont_merged",
                stitch_events=[
                    {"kind": "segment", "source": "inference", "episode_index": 0},
                    {"kind": "segment", "source": "dagger", "episode_index": 0},
                    {"kind": "segment", "source": "inference", "episode_index": 1},
                ],
                prompt="Organize the clothes.", fps=30,
            )
            self.assertTrue(res_cont["merged"])
            self.assertEqual(res_cont["episodes"], 1)
            self.assertEqual(res_cont["frames"], 16)
            t_cont = pq.read_table(base / "round_cont_merged/data/chunk-000/episode_000000.parquet")
            st_c = np.asarray(t_cont["observation.state"].to_pylist(), dtype=np.float32)
            ac_c = np.asarray(t_cont["action"].to_pylist(), dtype=np.float32)
            # Seam frames (4->5 and 11->12) have exact action[seam] == state[seam+1]
            self.assertTrue(np.allclose(ac_c[4], st_c[5], atol=1e-6))
            self.assertTrue(np.allclose(ac_c[11], st_c[12], atol=1e-6))
            ds_cont = LeRobotDataset(
                repo_id="local/cont_merged",
                root=base / "round_cont_merged",
                video_backend="pyav",
                delta_timestamps={"action": [i / 30.0 for i in range(4)]},
            )
            self.assertEqual(len(ds_cont), 16)
            # Check seam RGB transition: frame 4 is red, frame 5 is green!
            self.assertGreater(ds_cont[4]["observation.images.top_head"][0].mean().item(), 0.9)
            self.assertGreater(ds_cont[5]["observation.images.top_head"][1].mean().item(), 0.9)
            # Check per-frame is_dagger column (0=inference [0..4, 12..15], 1=dagger [5..11])
            is_dag_col = np.asarray(t_cont["is_dagger"].to_pylist(), dtype=np.int64)
            self.assertEqual(is_dag_col.tolist(), [0]*5 + [1]*7 + [0]*4)
            self.assertEqual(int(ds_cont[4]["is_dagger"].item()), 0)
            self.assertEqual(int(ds_cont[5]["is_dagger"].item()), 1)
            self.assertEqual(int(ds_cont[11]["is_dagger"].item()), 1)
            self.assertEqual(int(ds_cont[12]["is_dagger"].item()), 0)

            # Case B: breaks on collision_avoidance AND dagger_deleted -> splits into 3 episodes: [inf0], [dag0, inf1], [dag1]
            res_split = merge_round_lerobot_v21(
                inference_dir=inf_dir, dagger_dir=dag_dir, merged_dir=merged_dir,
                stitch_events=[
                    {"kind": "segment", "source": "inference", "episode_index": 0},
                    {"kind": "break", "reason": "collision_avoidance", "sides": {"left": {"max_delta_deg": 9.9}}},
                    {"kind": "segment", "source": "dagger", "episode_index": 0},
                    {"kind": "segment", "source": "inference", "episode_index": 1},
                    {"kind": "break", "reason": "dagger_deleted", "attempt_index": 1},
                    {"kind": "segment", "source": "dagger", "episode_index": 1},
                ],
                prompt="Organize the clothes.", fps=30,
            )
            self.assertTrue(res_split["merged"])
            self.assertEqual(res_split["episodes"], 3)
            self.assertEqual([r["length"] for r in res_split["runs"]], [5, 11, 6])
            ds_split = LeRobotDataset(
                repo_id="local/split_merged",
                root=merged_dir,
                video_backend="pyav",
                delta_timestamps={"action": [i / 30.0 for i in range(4)]},
            )
            self.assertEqual(len(ds_split), 22)
            self.assertEqual(int(ds_split[21]["episode_index"]), 2)


class FullMockAppStateMachineTests(unittest.TestCase):
    def test_full_dagger_cycle_delete_re_record_resume_and_exit_save_lerobot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_mock_cfg(Path(tmp))
            self.assertEqual(cfg["recording"]["format"], "lerobot")
            app = SDKDaggerApp(cfg, dataset_name="unit_round", dataset_dir=tmp, mock=True)
            rc_holder = []

            t = threading.Thread(target=lambda: rc_holder.append(app.run()), daemon=True)
            t.start()

            # Wait until inference has collected a few frames
            deadline = time.monotonic() + 5.0
            while len(app._inference_samples) < 5 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertGreaterEqual(len(app._inference_samples), 5)

            # 1. Enter DAgger (left single) -> saves inference episode_000000, enters DAGGER_RECORDING
            app.handle_operator_action("left", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "DAGGER_RECORDING" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(app.mode, "DAGGER_RECORDING")
            self.assertTrue((app.inference_dir / "data/chunk-000/episode_000000.parquet").exists())

            # Collect some DAgger frames
            time.sleep(0.15)
            # 2. Save DAgger -> enters CONFIRM_SAVE -> delete with 'w'
            app.handle_operator_action("save", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "CONFIRM_SAVE" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(app.mode, "CONFIRM_SAVE")
            app.handle_operator_action("w", source="keyboard")
            deadline = time.monotonic() + 5.0
            while app.mode != "DAGGER_PAUSED" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse((app.dagger_dir / "data/chunk-000/episode_000000.parquet").exists())
            self.assertEqual(app._dagger_episode_idx, 0)
            dagger_info_after_del = json.loads((app.dagger_dir / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(dagger_info_after_del["total_episodes"], 0)

            # 3. Start new segment (left single) -> record -> save -> confirm with pedal
            app.handle_operator_action("left", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "DAGGER_RECORDING" and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.15)
            app.handle_operator_action("save", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "CONFIRM_SAVE" and time.monotonic() < deadline:
                time.sleep(0.02)
            app.handle_operator_action("resume", source="pedal")  # first click only confirms!
            deadline = time.monotonic() + 5.0
            while app.mode != "DAGGER_PAUSED" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue((app.dagger_dir / "data/chunk-000/episode_000000.parquet").exists())
            self.assertEqual(app._dagger_episode_idx, 1)

            # 4. Resume inference (left double)
            app.handle_operator_action("resume", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "INFERENCE" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(app.mode, "INFERENCE")
            time.sleep(0.15)

            # 5. Stop round -> auto-saves inference episode_000001 and flushes all async videos
            app.request_stop(signal.SIGINT)
            t.join(timeout=8.0)
            self.assertFalse(t.is_alive())
            self.assertEqual(rc_holder, [130])
            self.assertTrue((app.inference_dir / "data/chunk-000/episode_000001.parquet").exists())

            # Validate both inference and dagger datasets load cleanly via LeRobotDatasetMetadata
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

            inf_meta = LeRobotDatasetMetadata(repo_id="local/unit_inf", root=app.inference_dir, force_cache_sync=False)
            dag_meta = LeRobotDatasetMetadata(repo_id="local/unit_dag", root=app.dagger_dir, force_cache_sync=False)
            self.assertEqual(inf_meta.total_episodes, 2)
            self.assertEqual(dag_meta.total_episodes, 1)
            self.assertEqual(dag_meta.episodes[0]["attempt_index"], 1)
            self.assertEqual(dag_meta.episodes[0]["deleted_attempts_before"], [0])
            self.assertEqual(dag_meta.episodes[0]["after_inference_episode"], 0)
            dag_timeline = [
                json.loads(line)
                for line in (app.dagger_dir / "meta/session_timeline.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [(r["event"], r["attempt_index"], r["episode_index"]) for r in dag_timeline],
                [("saved", 0, 0), ("deleted", 0, 0), ("saved", 1, 0)],
            )
            for lr_key in LEROBOT_CAMERA_KEYS.values():
                self.assertTrue((app.inference_dir / f"videos/chunk-000/{lr_key}/episode_000000.mp4").exists())
                self.assertTrue((app.inference_dir / f"videos/chunk-000/{lr_key}/episode_000001.mp4").exists())
                self.assertTrue((app.dagger_dir / f"videos/chunk-000/{lr_key}/episode_000000.mp4").exists())

            # Round had inf0 -> deleted dag -> kept dag0 -> inf1 => merged_dir has 2 continuous episodes: [inf0] and [dag0 + inf1]!
            merged_meta = LeRobotDatasetMetadata(repo_id="local/unit_merged", root=app.merged_dir, force_cache_sync=False)
            self.assertEqual(merged_meta.total_episodes, 2)
            self.assertEqual(len(merged_meta.episodes[0]["source_segments"]), 1)
            self.assertEqual(len(merged_meta.episodes[1]["source_segments"]), 2)
            self.assertEqual(merged_meta.episodes[1]["breaks_before"][0]["reason"], "dagger_deleted")

    def test_direct_resume_from_recording_auto_saves_unsaved_dagger_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_mock_cfg(Path(tmp))
            app = SDKDaggerApp(cfg, dataset_name="autosave_round", dataset_dir=tmp, mock=True)
            rc_holder = []
            t = threading.Thread(target=lambda: rc_holder.append(app.run()), daemon=True)
            t.start()
            deadline = time.monotonic() + 5.0
            while len(app._inference_samples) < 4 and time.monotonic() < deadline:
                time.sleep(0.02)

            app.handle_operator_action("left", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "DAGGER_RECORDING" and time.monotonic() < deadline:
                time.sleep(0.02)
            time.sleep(0.15)

            # Directly trigger left double (resume) WITHOUT pressing save first -> should auto-save episode_000000
            app.handle_operator_action("resume", source="pedal")
            deadline = time.monotonic() + 5.0
            while app.mode != "INFERENCE" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertEqual(app.mode, "INFERENCE")
            self.assertTrue((app.dagger_dir / "data/chunk-000/episode_000000.parquet").exists())
            self.assertEqual(app._dagger_episode_idx, 1)

            app.request_stop(signal.SIGINT)
            t.join(timeout=8.0)
            self.assertEqual(rc_holder, [130])

    def test_legacy_hdf5_format_option_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _fast_mock_cfg(Path(tmp))
            cfg["recording"]["format"] = "hdf5"
            app = SDKDaggerApp(cfg, dataset_name="legacy_round", dataset_dir=tmp, mock=True)
            rc_holder = []
            t = threading.Thread(target=lambda: rc_holder.append(app.run()), daemon=True)
            t.start()
            deadline = time.monotonic() + 5.0
            while len(app._inference_samples) < 4 and time.monotonic() < deadline:
                time.sleep(0.02)
            app.request_stop(signal.SIGINT)
            t.join(timeout=8.0)
            self.assertEqual(rc_holder, [130])
            self.assertTrue((app.inference_dir / "episode_0.hdf5").exists())


class TwoPedalStandbyMockPTYTests(unittest.TestCase):
    def test_two_rounds_via_real_pedal_supervisor_standby_and_mock_sdk_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = _fast_mock_cfg(tmp_path / "data")
            cfg_path = tmp_path / "mock.yaml"
            import yaml
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            pedal_cfg = {
                "version": 1,
                "device": {"vendor": "16c0", "product": "05e2", "serial": "030032217272"},
                "bindings": {
                    "left": {"interface": "kbd", "code": 64, "modifiers": [29]},
                    "right": {"interface": "kbd", "code": 66, "modifiers": [29]},
                },
            }
            pedal_cfg_path = tmp_path / "pedal.json"
            pedal_cfg_path.write_text(json.dumps(pedal_cfg), encoding="utf-8")
            fifo = tmp_path / "events.fifo"
            os.mkfifo(fifo, 0o600)
            writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)

            driver = tmp_path / "driver.py"
            driver.write_text(f"""
import json, os, sys
sys.path.insert(0, {str(ROOT / 'operator')!r})
from run_pedal_session import run_pedal_session

class PipePedal:
    def __init__(self, path):
        self.path = path
        self.fd = None
        self.buf = b""
    @property
    def filenos(self):
        return () if self.fd is None else (self.fd,)
    def open(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
    def read_events(self, fd):
        try:
            data = os.read(fd, 65536)
        except BlockingIOError:
            return []
        self.buf += data
        rows = []
        while b"\\n" in self.buf:
            line, self.buf = self.buf.split(b"\\n", 1)
            rows.append(tuple(json.loads(line)))
        return rows
    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

pedal = PipePedal({str(fifo)!r})
cmd = ["bash", {str(ROOT / 'run_dagger_sdk.sh')!r}, "--config", {str(cfg_path)!r}, "--mock"]
raise SystemExit(run_pedal_session(cmd, {str(pedal_cfg_path)!r}, device=pedal, standby_on_pedal_stop=True))
""", encoding="utf-8")

            master, slave = pty.openpty()
            proc = subprocess.Popen(
                [sys.executable, "-u", str(driver)],
                stdin=slave, stdout=slave, stderr=slave,
                close_fds=True, start_new_session=True,
            )
            os.close(slave)
            output = b""

            def send_click(code):
                for ev in (["kbd", 1, 29, 1], ["kbd", 1, code, 1]):
                    os.write(writer, (json.dumps(ev) + "\n").encode())
                time.sleep(0.01)
                for ev in (["kbd", 1, 29, 0], ["kbd", 1, code, 0]):
                    os.write(writer, (json.dumps(ev) + "\n").encode())

            def send_double(code):
                send_click(code)
                time.sleep(0.08)
                send_click(code)

            def wait_out(substr: str, count: int = 1, timeout: float = 12.0):
                nonlocal output
                needle = substr.encode("utf-8")
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if select.select([master], [], [], 0.05)[0]:
                        try:
                            output += os.read(master, 65536)
                        except OSError:
                            pass
                    if output.count(needle) >= count:
                        return
                self.fail(f"Timeout waiting for {count}x {substr!r}; tail={output[-800:].decode('utf-8', 'ignore')}")

            try:
                # Round 1 ready
                wait_out("[踏板] 控制已就绪", count=1)
                time.sleep(0.2)
                # Right double -> end Round 1 and enter standby
                send_double(66)
                wait_out("[待机] 本轮采集与保存已完成", count=1)

                # Single click in standby ignored
                send_click(64)
                time.sleep(0.6)
                self.assertEqual(output.count("[踏板] 控制已就绪".encode("utf-8")), 1)

                # Left double in standby starts Round 2!
                send_double(64)
                wait_out("[踏板] 控制已就绪", count=2)
                time.sleep(0.2)

                # Ctrl+C exits entire program cleanly
                os.killpg(proc.pid, signal.SIGINT)
                rc = proc.wait(timeout=10.0)
                self.assertEqual(rc, 130)

                # Verify both rounds produced valid LeRobot v2.1 inference datasets
                parquets = sorted((tmp_path / "data").glob("*_inference/data/chunk-000/episode_000000.parquet"))
                infos = sorted((tmp_path / "data").glob("*_inference/meta/info.json"))
                self.assertGreaterEqual(len(parquets), 2)
                self.assertGreaterEqual(len(infos), 2)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                os.close(master)
                os.close(writer)


class HardwareContractTests(unittest.TestCase):
    def test_real_sdk_imports_and_signatures_match(self):
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.prepare import load_config
        import inspect
        import prepare_follower_enabled
        import recover_master_enabled
        import master_native_forwarding

        cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
        rig = SDKDaggerRig(cfg)
        self.assertIsNotNone(rig._PiperDualArm)
        self.assertIsNotNone(rig._RealSenseRig)
        self.assertTrue(callable(prepare_follower_enabled.prepare_enabled))
        self.assertTrue(callable(recover_master_enabled.recover_enabled))
        fw_left = recover_master_enabled.cached_firmware_version(ROOT, "can_left_mas")
        fw_right = recover_master_enabled.cached_firmware_version(ROOT, "can_right_mas")
        self.assertIsNotNone(fw_left)
        self.assertIsNotNone(fw_right)
        self.assertIn("begin_phase", dir(master_native_forwarding.NativeMasterReader))
        self.assertIn("stop", dir(master_native_forwarding.NativeMasterReader))
        params = list(inspect.signature(rig._PiperDualArm.from_config).parameters)
        self.assertEqual(params[:2], ["arm_cfg", "can_cfg"])

    def test_high_follow_pause_and_resume_anchors_at_post_teleop_pose(self):
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.prepare import load_config
        from types import SimpleNamespace

        cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
        rig = SDKDaggerRig(cfg)
        entered = []

        def fake_arm(side):
            return SimpleNamespace(
                enter_high_follow_mode=lambda s=side: entered.append(s),
                apply_high_follow=lambda *a, **kw: None,
                close=lambda: None,
            )

        from robot_io import PiperHighFollowConfig
        dual = rig._PiperDualArm(
            left=fake_arm("left"),
            right=fake_arm("right"),
            high_follow_config=PiperHighFollowConfig.from_config(cfg["arm"]),
        )
        rig.followers = dual
        post_teleop_pose = np.linspace(0.1, 0.7, 14, dtype=np.float32)
        rig.read_follower_qpos = lambda: post_teleop_pose.copy()

        dual._high_follow_last_ref = np.zeros(14, dtype=float)
        rig._pause_high_follow_for_teleop()
        self.assertFalse(dual._high_follow_running)

        rig._resume_high_follow_after_teleop()
        try:
            self.assertEqual(entered, ["left", "right"])
            self.assertTrue(dual._high_follow_running)
            self.assertTrue(np.allclose(dual._high_follow_last_ref, post_teleop_pose))
            self.assertTrue(np.allclose(dual._high_follow_anchor_waypoint, post_teleop_pose))
        finally:
            rig._pause_high_follow_for_teleop()


    def test_apply_policy_action_applies_joint_bias_deg_only_to_arm_joints(self):
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.prepare import load_config
        from types import SimpleNamespace
        import math

        cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
        self.assertEqual(cfg["arm"]["joint_bias_deg"]["left"], [0, 0, 3, 0, 0, 0])
        self.assertEqual(cfg["arm"]["joint_bias_deg"]["right"], [0, 0, 3, 0, 0, 0])
        self.assertAlmostEqual(float(cfg["arm"]["right_offset"]), 0.003, places=6)

        rig = SDKDaggerRig(cfg)
        captured = []
        rig.followers = SimpleNamespace(apply_action=lambda act, action_step=None: captured.append(act.copy()))
        raw = np.zeros(14, dtype=np.float64)
        rig.apply_policy_action(raw)
        self.assertEqual(len(captured), 1)
        # Raw input not mutated
        self.assertTrue(np.allclose(raw, 0.0))
        # Left J3 (idx 2) and Right J3 (idx 9) shifted by +3 deg; grippers (6, 13) untouched
        expected = np.zeros(14, dtype=np.float64)
        expected[2] = math.radians(3.0)
        expected[9] = math.radians(3.0)
        self.assertTrue(np.allclose(captured[0], expected, atol=1e-9))

    def test_fc_to_fa_releases_master_grippers_before_0xfa(self):
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.prepare import load_config
        from types import SimpleNamespace

        cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
        rig = SDKDaggerRig(cfg)
        calls = {"left": [], "right": []}

        def make_master(side):
            piper = SimpleNamespace(
                GripperCtrl=lambda width, effort, code, zero, s=side: calls[s].append(("GripperCtrl", width, effort, code, zero)),
                MasterSlaveConfig=lambda **kw: calls[side].append(("MasterSlaveConfig", kw["linkage_config"])),
            )
            return SimpleNamespace(_lock=threading.Lock(), _piper=piper)

        rig.master_left = make_master("left")
        rig.master_right = make_master("right")
        rig._release_masters_to_fa({"left": 65000, "right": 42000})

        self.assertEqual(calls["left"], [("GripperCtrl", 65000, 0, 0x00, 0), ("MasterSlaveConfig", 0xFA)])
        self.assertEqual(calls["right"], [("GripperCtrl", 42000, 0, 0x00, 0), ("MasterSlaveConfig", 0xFA)])

    def test_handle_collision_matches_urdf_and_nudges_follower_on_real_crash_pose(self):
        import math
        from types import SimpleNamespace
        from sdk_dagger.collision import HandleCollisionConfig, evaluate_path_clearance, find_minimal_safe_pose, piper_link6_fk
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.prepare import load_config

        cfg = load_config(ROOT / "config/dagger_precision_5090.yaml")
        hcfg = HandleCollisionConfig.from_dict(cfg["arm"]["master_handle_collision"])
        self.assertTrue(hcfg.enabled)

        # Cross-check FK against piper_ms_teleop URDF if present
        urdf_mod_dir = Path("/home/ubuntu/dev/piper_ms_teleop")
        if (urdf_mod_dir / "robot_kinematics.py").is_file():
            sys.path.insert(0, str(urdf_mod_dir))
            from robot_kinematics import load_model
            urdf = load_model()
            q_sample = [0.2, 0.8, -0.9, -0.3, 0.5, -0.4]
            self.assertTrue(np.allclose(piper_link6_fk(q_sample), urdf.forward(q_sample)["link6"], atol=1e-12))

        safe_l = np.asarray(cfg["arm"]["master_safe_pose"]["left"], dtype=np.float64)
        safe_r = np.asarray(cfg["arm"]["master_safe_pose"]["right"], dtype=np.float64)
        # Real recorded pose when can_left_mas J5 hit the base vs safe right arm
        crash_l = np.array([math.radians(x / 1000.0) for x in [-2204, 23847, -16634, -11164, 38530, 6048]] + [0.05])
        ok_r = np.array([math.radians(x / 1000.0) for x in [-477, 106629, -69319, 315, 42606, 7470]] + [0.024])

        # Verify raw crash_l collides while ok_r is safe
        self.assertFalse(evaluate_path_clearance(safe_l[:6], crash_l[:6], hcfg)["safe"])
        self.assertTrue(evaluate_path_clearance(safe_r[:6], ok_r[:6], hcfg)["safe"])

        # Verify SDKDaggerRig gently nudges ONLY the colliding left follower and leaves right + grippers untouched
        rig = SDKDaggerRig(cfg)
        moved = []
        rig.followers = SimpleNamespace(move_to=lambda **kw: moved.append(kw))
        rig.read_follower_qpos = lambda: np.concatenate([np.asarray(moved[-1]["left"]), np.asarray(moved[-1]["right"])])

        new_l, new_r = rig._adjust_followers_for_handle_collision(crash_l, ok_r, safe_l, safe_r)
        self.assertEqual(len(moved), 1)
        self.assertTrue(evaluate_path_clearance(safe_l[:6], new_l[:6], hcfg)["safe"])
        self.assertAlmostEqual(float(new_l[6]), 0.05, places=6)
        self.assertTrue(np.allclose(new_r, ok_r, atol=1e-9))
        self.assertIsNotNone(rig.last_collision_avoidance)
        self.assertIn("left", rig.last_collision_avoidance["sides"])
        self.assertNotIn("right", rig.last_collision_avoidance["sides"])

        # Verify metadata persists cleanly into episodes.jsonl & session_timeline.jsonl and LeRobotDataset still loads!
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ca_test_dagger"
            w = LeRobotV21DatasetWriter(root, prompt="Organize the clothes.", fps=30)
            w.record_collision_avoidance(rig.last_collision_avoidance, after_inference_episode=0)
            w.save_episode(
                _make_samples(5),
                export_video=True,
                after_inference_episode=0,
                collision_avoidance_before=rig.last_collision_avoidance,
            )
            ep_row = json.loads((root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("left", ep_row["collision_avoidance_before"]["sides"])
            tl_rows = [json.loads(l) for l in (root / "meta/session_timeline.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual([r["event"] for r in tl_rows], ["collision_avoidance", "saved"])
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
            meta = LeRobotDatasetMetadata(repo_id="local/ca_test", root=root, force_cache_sync=False)
            self.assertEqual(meta.total_episodes, 1)


    def test_collision_avoidance_keeps_high_follow_running_and_resumes_if_paused(self):
        """Regression: collision follower move must work while high_follow is enabled (never timed out due to early pause)."""
        import types
        import numpy as np
        from sdk_dagger.hardware import SDKDaggerRig
        from sdk_dagger.collision import HandleCollisionConfig

        rig = object.__new__(SDKDaggerRig)
        rig.arm_cfg = {
            "collision_avoid_steps": 60,
            "collision_avoid_sleep_s": 0.01,
            "master_safe_pose": {
                "left": [0.0, 0.30, -0.30, 0.0, 0.0, 0.0, 0.0],
                "right": [0.0, 0.30, -0.30, 0.0, 0.0, 0.0, 0.0],
            },
        }
        rig.handle_collision_cfg = HandleCollisionConfig(safety_margin_m=0.010)
        rig.last_collision_avoidance = None

        events = []

        class DummyFollowers:
            high_follow_config = types.SimpleNamespace(enabled=True)
            _high_follow_running = False
            moved_to = None

            def move_to_high_follow(self, left, right, timeout_s=5.0):
                if not self._high_follow_running:
                    raise TimeoutError("high follow init target timed out: thread not running")
                events.append(("move_to_high_follow", round(timeout_s, 1)))
                self.moved_to = (np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))

        rig.followers = DummyFollowers()
        rig._resume_high_follow_after_teleop = lambda: (
            setattr(rig.followers, "_high_follow_running", True),
            events.append(("resume",)),
        )
        rig._pause_high_follow_for_teleop = lambda: (
            setattr(rig.followers, "_high_follow_running", False),
            events.append(("pause",)),
        )
        rig.read_follower_qpos = lambda: np.concatenate(rig.followers.moved_to, axis=0).astype(np.float32)

        # Real crash pose that triggers base collision on left arm
        crash_l = np.array([np.radians(-2.1), np.radians(31.9), np.radians(-14.8), np.radians(-15.5), np.radians(42.7), np.radians(14.6), 0.02], dtype=np.float64)
        safe_r = np.array([0.0, 0.6, -0.6, 0.0, 0.1, 0.0, 0.02], dtype=np.float64)
        safe_pose = np.array([0.0, 0.30, -0.30, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        out_l, out_r = rig._adjust_followers_for_handle_collision(crash_l, safe_r, safe_pose, safe_pose)
        self.assertEqual([e[0] for e in events], ["resume", "move_to_high_follow", "pause"])
        self.assertGreaterEqual(events[1][1], 5.0)
        self.assertIsNotNone(rig.last_collision_avoidance)
        self.assertIn("left", rig.last_collision_avoidance["sides"])
        self.assertFalse(rig.followers._high_follow_running)

        # Also verify enter_teaching_mode source calls _pause_high_follow_for_teleop AFTER _adjust_followers_for_handle_collision
        import inspect
        src = inspect.getsource(SDKDaggerRig.enter_teaching_mode)
        idx_adj = src.index("_adjust_followers_for_handle_collision")
        idx_pause = src.index("_pause_high_follow_for_teleop")
        idx_release = src.index("_release_masters_to_fa")
        self.assertLess(idx_adj, idx_pause)
        self.assertLess(idx_pause, idx_release)



    def test_realtime_follower_action_clip_prevents_handle_collision_smoothly(self):
        """Real-time policy action clip keeps followers inside handle-safe space smoothly while preserving safe poses & grippers."""
        import threading
        import numpy as np
        from sdk_dagger.collision import (
            HandleCollisionConfig,
            evaluate_handle_clearance,
            evaluate_path_clearance,
            project_pose_to_handle_safe,
        )
        from sdk_dagger.hardware import SDKDaggerRig

        cfg = HandleCollisionConfig(safety_margin_m=0.010, clip_follower_actions=True, clip_margin_extra_m=0.003)

        # 1. Already-safe pose (e.g., reaching forward) must pass through untouched
        safe_reach = np.array([0.1, 1.35, -0.85, -0.05, 0.70, 0.10], dtype=np.float64)
        out_safe, info_safe = project_pose_to_handle_safe(safe_reach, cfg)
        self.assertFalse(info_safe["clipped"])
        self.assertTrue(np.allclose(out_safe, safe_reach, atol=1e-12))

        # 2. Real crash pose must be projected safely with >=10mm+3mm clearance and path-safe from safe_pose
        crash_pose = np.radians([-2.1, 31.9, -14.8, -15.5, 42.7, 14.6])
        out_proj, info_proj = project_pose_to_handle_safe(crash_pose, cfg)
        self.assertTrue(info_proj["clipped"])
        self.assertGreaterEqual(evaluate_handle_clearance(out_proj, cfg)["min_clearance_m"], 0.0)
        safe_pose6 = np.array([0.0, 0.30, -0.30, 0.0, 0.0, 0.0], dtype=np.float64)
        self.assertTrue(evaluate_path_clearance(safe_pose6, out_proj, cfg)["safe"])

        # 3. End-to-end check inside SDKDaggerRig.apply_policy_action: preserves gripper & clips only colliding side
        rig = object.__new__(SDKDaggerRig)
        rig._mode_lock = threading.RLock()
        rig._teaching_active = False
        rig._joint_bias_rad = np.zeros(12, dtype=float)
        rig.arm_cfg = {"master_safe_pose": {"left": [0, 0.3, -0.3, 0, 0, 0, 0], "right": [0, 0.3, -0.3, 0, 0, 0, 0]}}
        rig.handle_collision_cfg = cfg
        rig._clip_log_last_t = 0.0
        rig._clip_total_steps = 0
        sent = []

        class CaptureFollowers:
            def apply_action(self, action, action_step=None):
                sent.append(np.asarray(action, dtype=float).copy())

        rig.followers = CaptureFollowers()
        cmd = np.concatenate([crash_pose, [0.042], safe_reach, [0.055]])
        rig.apply_policy_action(cmd)
        self.assertEqual(len(sent), 1)
        out_cmd = sent[0]
        # Left arm clipped safely, right arm untouched, both grippers preserved exactly
        self.assertTrue(evaluate_handle_clearance(out_cmd[:6], cfg)["safe"])
        self.assertTrue(np.allclose(out_cmd[7:13], safe_reach, atol=1e-12))
        self.assertAlmostEqual(out_cmd[6], 0.042, places=6)
        self.assertAlmostEqual(out_cmd[13], 0.055, places=6)
        self.assertEqual(rig._clip_total_steps, 1)



    def test_stateful_clip_eliminates_cylinder_branch_flip_jumps(self):
        """Regression: sweeping across the base-cylinder medial axis (J5=0..60 at J2=5,J3=-15) must never jump >4.5 deg."""
        import numpy as np
        from sdk_dagger.collision import HandleCollisionConfig, evaluate_handle_clearance, project_pose_to_handle_safe

        cfg = HandleCollisionConfig(safety_margin_m=0.005, clip_follower_actions=True, clip_margin_extra_m=0.003)
        prev = np.array([0.0, 0.32, -0.36, 0.0, 0.24, 0.0], dtype=np.float64)
        max_jump_deg = 0.0
        for j5_deg in range(0, 61, 1):
            q_raw = np.radians([0.0, 5.0, -15.0, 0.0, float(j5_deg), 0.0])
            out, _ = project_pose_to_handle_safe(q_raw, cfg, q_prev6=prev)
            self.assertGreaterEqual(evaluate_handle_clearance(out, cfg)["min_clearance_m"], 0.0)
            max_jump_deg = max(max_jump_deg, float(np.degrees(np.max(np.abs(out - prev)))))
            prev = out
        self.assertLessEqual(max_jump_deg, 4.55)



    def test_plan_safe_master_trajectory_escapes_tucked_start_and_avoids_midpath_collision(self):
        """Master trajectory planner escapes tucked-near-base start poses and avoids mid-flight collisions."""
        import numpy as np
        from sdk_dagger.collision import (
            HandleCollisionConfig,
            evaluate_handle_clearance,
            plan_safe_master_trajectory,
        )

        cfg = HandleCollisionConfig(safety_margin_m=0.005, clip_follower_actions=True, clip_margin_extra_m=0.003)
        safe_pose = np.array([0.0, 0.30, -0.30, 0.0, 0.0, 0.0], dtype=np.float64)

        # Case 1: Real tucked can_right_mas pose that collided during Step 1/2
        r_mas_tucked = np.radians([2.7, 0.0, -6.0, 4.8, 25.8, -0.8])
        self.assertFalse(evaluate_handle_clearance(r_mas_tucked, cfg)["safe"])
        traj1, info1 = plan_safe_master_trajectory(r_mas_tucked, safe_pose, cfg, num_steps=60)
        self.assertTrue(info1["detour"])
        self.assertTrue(info1["escaped_start"])
        self.assertTrue(np.allclose(traj1[-1], safe_pose, atol=1e-4))
        # Clearance must strictly monotonically increase out of tucked start and never re-enter collision zone
        clrs1 = [evaluate_handle_clearance(q, cfg)["min_clearance_m"] for q in traj1]
        first_safe = next(i for i, c in enumerate(clrs1) if c >= 0.0)
        for i in range(first_safe):
            self.assertGreaterEqual(clrs1[i + 1], clrs1[i] - 0.0005)
        for c in clrs1[first_safe:]:
            self.assertGreaterEqual(c, 0.0)

        # Case 3: Both start and goal are safe, but straight line sweeps through base -> detour keeps 100% safe
        q_left_side = np.radians([-70.0, 10.0, -5.0, 0.0, 35.0, 0.0])
        q_right_side = np.radians([70.0, 10.0, -5.0, 0.0, 35.0, 0.0])
        # Ensure start/goal are lifted just outside base on opposite sides of J1
        q_left_side = np.radians([-60.0, 22.0, -15.0, 0.0, 12.0, 0.0])
        q_right_side = np.radians([60.0, 22.0, -15.0, 0.0, 12.0, 0.0])
        traj3, info3 = plan_safe_master_trajectory(q_left_side, q_right_side, cfg, num_steps=50)
        self.assertTrue(np.allclose(traj3[-1], q_right_side, atol=1e-4))
        for q in traj3:
            self.assertTrue(evaluate_handle_clearance(q, cfg)["safe"])

        # Case 2: Already-safe Step 2/2 path stays synchronized (lockstep) and 100% safe
        r_slv = np.radians([23.7, 93.1, -23.9, 83.4, 32.9, -49.6])
        traj2, info2 = plan_safe_master_trajectory(safe_pose, r_slv, cfg, num_steps=60)
        self.assertFalse(info2["detour"])
        self.assertTrue(np.allclose(traj2[-1], r_slv, atol=1e-4))
        for q in traj2:
            self.assertTrue(evaluate_handle_clearance(q, cfg)["safe"])



    def test_hold_and_close_during_dagger_never_replays_stale_high_follow_pose(self):
        """Regression: stopping/closing during DAgger must hold measured post-DAgger pose, NEVER replay stale pre-DAgger high_follow_last_ref."""
        import threading, types
        import numpy as np
        from sdk_dagger.hardware import SDKDaggerRig

        rig = object.__new__(SDKDaggerRig)
        rig.arm_cfg = {"gripper_effort": 1000}
        rig._teaching_active = True
        rig._following_enabled = threading.Event()
        rig._following_enabled.set()
        rig._fa_stop_event = threading.Event()
        rig._fa_thread = None
        rig._fa_readers = {}
        rig._preroll_stop = threading.Event()
        rig._preroll_thread = None
        rig._preroll_lock = threading.Lock()
        rig._preroll_buffer = []
        rig.master_left = None
        rig.master_right = None
        rig.cameras = None

        sent_joints = {"left": [], "right": []}

        def _make_arm(side):
            arm = types.SimpleNamespace(_lock=threading.RLock(), gripper_exist=True, _high_follow_mode=True)
            piper = types.SimpleNamespace(
                MotionCtrl_2=lambda *a, **kw: None,
                JointCtrl=lambda *j: sent_joints[side].append(list(j)),
                GripperCtrl=lambda *a, **kw: None,
            )
            arm._piper = piper
            return arm

        class DummyFollowers:
            high_follow_config = types.SimpleNamespace(enabled=True)
            _high_follow_running = False
            _high_follow_thread = None
            _high_follow_lock = threading.Lock()
            _high_follow_queue = []
            _high_follow_metadata_queue = []
            _high_follow_last_ref = np.array([1.0] * 14, dtype=float)  # stale pre-DAgger pose!
            _high_follow_prev_waypoint = None
            _high_follow_anchor_waypoint = None
            left = _make_arm("left")
            right = _make_arm("right")
            hold_called = False
            closed = False

            def _reset_high_follow_segment_locked(self, reset_anchor=False):
                pass

            def hold_current_position(self):
                self.hold_called = True
                raise AssertionError("hold_current_position must NOT be called during DAgger/close!")

            def close(self):
                self.closed = True

        rig.followers = DummyFollowers()
        post_dagger_q = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.01, 0.1, 0.0, 0.0, -0.5, 0.5, 0.3, 0.02], dtype=np.float32)
        rig.read_follower_qpos = lambda: post_dagger_q.copy()

        rig.hold_followers()
        self.assertFalse(rig.followers.hold_called)
        self.assertFalse(rig._following_enabled.is_set())
        self.assertEqual(len(sent_joints["left"]), 1)
        # Verify anchored at measured post-DAgger pose (J2=0 deg), NOT stale 1.0 rad (~57.3 deg)
        self.assertEqual(sent_joints["left"][0][1], 0)

        rig.close()
        self.assertFalse(rig.followers.hold_called)
        self.assertTrue(rig.followers.closed)
        self.assertIsNone(rig.followers._high_follow_last_ref)
        self.assertEqual(sent_joints["left"][-1][1], 0)


    def test_fa_direct_forward_high_follow_and_position_modes(self):
        """Teleop FA forwarding must support high_follow (0xAD, sent once) and position (0x00) modes."""
        import threading, types
        from sdk_dagger.hardware import SDKDaggerRig

        def _make_rig(fa_mode: str):
            rig = object.__new__(SDKDaggerRig)
            rig.arm_cfg = {
                "fa_control_mode": fa_mode,
                "fa_forward_velocity": 100,
                "fa_gripper_effort": 1000,
            }
            calls = []
            arm = types.SimpleNamespace(
                _lock=threading.RLock(),
                gripper_exist=True,
                _high_follow_mode=False,
                _piper=types.SimpleNamespace(
                    MotionCtrl_2=lambda *a: calls.append(("MotionCtrl_2", a)),
                    JointCtrl=lambda *j: calls.append(("JointCtrl", j)),
                    GripperCtrl=lambda *g: calls.append(("GripperCtrl", g)),
                ),
            )
            rig.followers = types.SimpleNamespace(left=arm, right=arm)
            return rig, arm, calls

        # high_follow: sends MotionCtrl_2(..., 0xAD) once on entry, then only JointCtrl + GripperCtrl
        rig_hf, arm_hf, calls_hf = _make_rig("high_follow")
        rig_hf._apply_follower_direct_mdeg("left", [1000, 2000, -3000, 0, 500, -500], 35000)
        rig_hf._apply_follower_direct_mdeg("left", [1100, 2100, -3100, 0, 500, -500], 36000)
        self.assertTrue(arm_hf._high_follow_mode)
        mc_hf = [c for c in calls_hf if c[0] == "MotionCtrl_2"]
        jc_hf = [c for c in calls_hf if c[0] == "JointCtrl"]
        self.assertEqual(len(mc_hf), 1)
        self.assertEqual(mc_hf[0][1], (0x01, 0x01, 100, 0xAD))
        self.assertEqual(len(jc_hf), 2)
        self.assertEqual(jc_hf[1][1], (1100, 2100, -3100, 0, 500, -500))

        # position: sends MotionCtrl_2(..., 0x00) and keeps _high_follow_mode = False
        rig_pos, arm_pos, calls_pos = _make_rig("position")
        arm_pos._high_follow_mode = True
        rig_pos._apply_follower_direct_mdeg("left", [1000, 2000, -3000, 0, 500, -500], 35000)
        self.assertFalse(arm_pos._high_follow_mode)
        mc_pos = [c for c in calls_pos if c[0] == "MotionCtrl_2"]
        self.assertEqual(len(mc_pos), 1)
        self.assertEqual(mc_pos[0][1], (0x01, 0x01, 100, 0x00))

    def test_fa_high_follow_slew_guard_protects_pause_resume_and_preserves_normal_teleop(self):
        """Slew guard caps large pause->resume jumps in 0xAD mode while passing normal teleop 1:1."""
        import threading
        import numpy as np
        from sdk_dagger.hardware import SDKDaggerRig, RAD_TO_MDEG

        rig = object.__new__(SDKDaggerRig)
        rig.arm_cfg = {
            "fa_control_mode": "high_follow",
            "high_follow": {"max_joint_vel": [2.0, 2.0, 2.0, 2.5, 2.5, 2.5]},
        }
        rig._following_enabled = threading.Event()
        rig._following_enabled.set()
        # Follower currently at 0 deg on all joints
        rig.read_follower_qpos = lambda: np.zeros(14, dtype=np.float32)

        # 1. Sudden 40 deg (40000 mdeg) jump after pause -> resume
        rig.set_following(False)
        rig.set_following(True)
        cmd1, ramping1 = rig._slew_limit_fa_joints_mdeg("left", [40000, 0, 0, 0, 0, 0], now=10.000)
        self.assertTrue(ramping1)
        # First 5ms tick at 2.0 rad/s -> ~0.573 deg (~573 mdeg), nowhere near 40000 mdeg
        expected_step = round(2.0 * RAD_TO_MDEG * 0.005)
        self.assertAlmostEqual(cmd1[0], expected_step, delta=2)

        # After enough 5ms ticks (~0.40s), it converges smoothly to 40000 mdeg
        t = 10.000
        cmd = cmd1
        ramping = ramping1
        for _ in range(85):
            t += 0.005
            cmd, ramping = rig._slew_limit_fa_joints_mdeg("left", [40000, 0, 0, 0, 0, 0], now=t)
        self.assertFalse(ramping)
        self.assertEqual(cmd[0], 40000)

        # 2. Normal human teleop step (0.2 deg = 200 mdeg in 5ms -> 40 deg/s < 114.6 deg/s) passes 1:1
        cmd_norm, ramping_norm = rig._slew_limit_fa_joints_mdeg("left", [40200, -150, 100, 0, 250, -250], now=t + 0.005)
        self.assertFalse(ramping_norm)
        self.assertEqual(cmd_norm, [40200, -150, 100, 0, 250, -250])


if __name__ == "__main__":
    unittest.main()
