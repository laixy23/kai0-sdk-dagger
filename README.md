# Kai0 SDK DAgger for AgileX PiPER

A ROS-free DAgger collector for recording policy rollouts and human corrections on bimanual AgileX PiPER arms.

在双臂策略执行过程中，如果机械臂漏抓或跑偏，操作者可以随时暂停策略，用两只主臂拖动从臂完成人工纠正，再把控制权交回模型，同一轮内支持多次接管。

这一版去掉了 `roscore`、`realsense2_camera` 和 Piper ROS 节点依赖，直接通过 `piper_sdk`、`pyrealsense2` 和原生 SocketCAN 控制四只机械臂与三台 RealSense 相机，并默认导出兼容 Kai0 / LeRobot v2.1 的数据集。键盘、USB 双踏板和 PyQt5 桌面界面共用同一套状态机。

## 离线查看与模拟运行

没有连接机械臂和相机时，可以先查看各入口帮助，或用 `--mock` 跑通完整状态机和单元测试：

```bash
git clone https://github.com/laixy23/kai0-sdk-dagger.git
cd kai0-sdk-dagger

bash run_dagger_sdk.sh --help
bash run_dagger_sdk_pedal.sh --help
bash run_dagger_sdk_gui.sh --help

# 无硬件全流程模拟
bash run_dagger_sdk.sh --mock

# 离线单元测试
python3 -m unittest discover -s tests -v
```

## 实机启动

实机配置见 [config/dagger_precision_5090.yaml](config/dagger_precision_5090.yaml)，按自己的设备修改 CAN 适配器序列号、三台 RealSense 序列号以及策略服务地址（WebSocket）。

```bash
# 1. 只读检查 CAN、踏板、策略服务与三路相机（不动机械臂）
bash run_dagger_sdk_pedal.sh --check-startup

# 2. 启动 PyQt5 桌面工作台（三路实时画面、双臂读数、双踏板待机循环与历史回放）
bash run_dagger_sdk_gui.sh

# 3. 命令行 + USB 双踏板启动
bash run_dagger_sdk_pedal.sh
```

> 相机由 `pyrealsense2` 直接独占打开；如果机器上此前运行过 ROS 相机服务，正式启动时会自动先停掉它（`--check-startup` 只读检查不会停止）。

## 一轮采集怎么操作

| 目的 | 键盘 | USB 双踏板 |
| --- | --- | --- |
| 暂停策略，主臂对齐后开始录制人工纠正 | `d` | 左单踩 |
| 暂停跟随，保存当前纠正段 | `s` | 右单踩 |
| 删除刚保存的纠正段（保存后 10 秒窗口内） | `w` | —（界面按钮或键盘 `w`） |
| 确认保存后，继续录下一段人工纠正 | `空格` / `d` | 左单踩 |
| 把控制权交回策略（未保存时会自动先保存） | `r` | 左双踩 |
| 待机状态下开始新一轮 | — | 左双踩 |
| 结束本轮（双踏板进入待机；终端退出程序） | `Ctrl+C` | 右双踩 |

接管就绪后会自动开始录制。保存后有 10 秒确认窗口，窗口内按 `w` 会撤销刚保存的这一段，按其他键或踏板则提前确认保留。

## 输出数据（LeRobot v2.1）

默认会在 `data/` 下按轮次生成以下目录（如需旧版 HDF5，可在配置中把 `recording.format` 设为 `hdf5`）：

- `<dataset_name>_inference/`：策略自主执行段
- `<dataset_name>_dagger/`：人工纠正段（仅在有保存记录时创建）
- `<dataset_name>_merged/`：开启 `auto_merge_round` 时，本轮结束自动按时间顺序拼接的连续数据集

每个 LeRobot v2.1 目录结构如下：

```text
<dataset_name>_{inference|dagger}/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/
│   ├── observation.images.top_head/episode_000000.mp4
│   ├── observation.images.hand_left/episode_000000.mp4
│   └── observation.images.hand_right/episode_000000.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── tasks.jsonl
    ├── episodes_stats.jsonl
    └── session_timeline.jsonl
```

- `observation.state` 与 `action` 均为 `float32[14]`（先左后右，每侧 6 关节 + 1 夹爪），训练配对满足 `action[t] = qpos[t+1]`。
- 在确认窗口内删除纠正段时，会同步移除对应的 `.parquet` 和三路 `.mp4`，保持 `episode_XXXXXX` 编号连续以便直接加载；删除记录与缺口位置会保留在 `meta/episodes.jsonl` 和 `meta/session_timeline.jsonl` 中。

## 目录结构

- [sdk_dagger/](sdk_dagger/)：核心采集状态机、硬件驱动封装（`piper_sdk` / `pyrealsense2` / 200 Hz `high_follow`）、手柄防碰撞检测、LeRobot v2.1 / HDF5 存储、语音提示与 PyQt5 界面
- [operator/](operator/)：USB 双踏板识别、校准、权限配置与待机循环进程管理
- [setup/](setup/)：按序列号激活 CAN、示教识别、主从臂对齐与恢复脚本
- [config/](config/) 与 [_runtime/](_runtime/)：实机 YAML 配置与踏板映射配置
- [assets/voice/](assets/voice/)：采集状态中文语音提示音频
- [tests/](tests/)：无硬件状态机、LeRobot 导出、踏板待机循环与 GUI 单元测试
