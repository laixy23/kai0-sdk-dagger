# Kai0 纯 SDK DAgger（零 ROS）

本目录为 `precision-5090` 上的**零 ROS 纯 SDK 版 DAgger 采集实现**（`/home/ubuntu/dev/kai0-sdk-dagger`），参考 `/home/ubuntu/dev/inference-origin` 的纯 SDK 架构，完全不依赖 `roscore`、`realsense2_camera` 或 Piper ROS 节点。

## 与 ROS 版（`kai0-official-dagger`）对比

| 模块 | 原 ROS 版（`kai0-official-dagger`） | 本目录纯 SDK 版（`kai0-sdk-dagger`） |
| --- | --- | --- |
| 相机采集 | `kai0-official-cameras.service`（ROS1 nodelet） | 直接 `pyrealsense2`（复用 `inference-origin` 的 `RealSenseRig`，独立 30 Hz 时间戳对齐采样线程） |
| 从臂推理控制 | ROS1 `piper` 节点 30 Hz | 直接 `piper_sdk`（复用 `inference-origin` 的 `PiperDualArm`，支持 200 Hz `high_follow` 三次插值） |
| 主臂示教接管 | ROS1 主臂节点 + 话题转发 | 进程内直接 `piper_sdk` + 原生 SocketCAN `NativeMasterReader`（零 ROS 跳数） |
| 模型推理客户端 | 独立 WebSocket 封装 | 复用 `inference-origin` `.venv-client` + `WebsocketClientPolicy` + `StreamActionBuffer(temporal)` |
| 数据格式 | `*_inference_hdf5` / `*_dagger_hdf5` + 三路 `.mp4` | **默认直接保存为 Kai0 兼容 LeRobot v2.1 格式**（Parquet + 三路后台异步并行 `.mp4` + `meta/`；可在配置中设 `recording.format: hdf5` 切回旧 HDF5） |
| 双踏板与待机循环 | 左单/右单/左双/右双 + 待机左双踩开新一轮 | **完全一致（全中文命令行提示）** |

## 默认输出结构（LeRobot v2.1）

每轮默认生成两个可直接被 Kai0 / LeRobot v2.1 加载的数据集目录（路径在 `/home/ubuntu/dev/kai0-sdk-dagger/data/` 下）：
- `<dataset_name>_inference/`（模型自主推理段）
- `<dataset_name>_dagger/`（人工纠正段，仅在有保存记录时创建）

目录内容：
```text
<dataset_name>_{inference|dagger}/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.top_head/episode_000000.mp4
│       ├── observation.images.hand_left/episode_000000.mp4
│       └── observation.images.hand_right/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── episodes_stats.jsonl
    └── session_timeline.jsonl   # 按时间顺序记录每段保存/删除事件与缺口
```

- `observation.state` / `action` 均为 `float32[14]`（`[left_j1..6, left_grip, right_j1..6, right_grip]`，且 `action[t] = qpos[t+1]`）。
- 按 `w` 删除刚保存的纠正段时，会取消/删除对应的 `.parquet` 和三路 `.mp4`，保持 `episode_XXXXXX` 连续（兼容 Kai0 `LeRobotDataset` 直接加载），同时在 `meta/episodes.jsonl`（`attempt_index`、`deleted_attempts_before`、`after_inference_episode`）与 `meta/session_timeline.jsonl` 中完整保留缺失记录。

> **注意：** 因为 `pyrealsense2` 需要直接访问三台 RealSense 相机，正式启动 SDK 版时会自动停止旧的 `kai0-official-cameras.service`（与 `inference-origin` 一致）。只读检查 `--check-startup` 不会停止它。

## 快速使用

```bash
cd /home/ubuntu/dev

# 1. 只读检查（不动机械臂、不停止 ROS 相机）
bash kai0-sdk-dagger/run_dagger_sdk_pedal.sh --check-startup

# 2. PyQt5 桌面工作台一键启动（实时三路画面 + 双臂读数 + 10秒删除倒计时 + 内置双踏板待机循环 + LeRobot v2.1 历史回放）
bash kai0-sdk-dagger/run_dagger_sdk_gui.sh

# 3. 命令行双踏板一键启动（默认开启两态待机循环：右双踩结束本轮进待机，左双踩开新一轮，终端 Ctrl+C 彻底退出）
bash kai0-sdk-dagger/run_dagger_sdk_pedal.sh

# 3. 无硬件模拟测试
bash kai0-sdk-dagger/run_dagger_sdk.sh --mock
```

## 运行自动化测试（零硬件）

```bash
cd /home/ubuntu/dev/kai0-sdk-dagger
/home/ubuntu/dev/inference-origin/.venv-client/bin/python -m unittest discover -s tests -v
```
