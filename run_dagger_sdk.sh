#!/usr/bin/env bash
# 纯 SDK（零 ROS）DAgger 采集入口（键盘控制，或作为双踏板入口的子进程）
set -euo pipefail

SDK_DAGGER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SDK_DAGGER_CONFIG:-${SDK_DAGGER_ROOT}/config/dagger_precision_5090.yaml}"
PYTHON_BIN="${SDK_DAGGER_PYTHON:-/home/ubuntu/dev/inference-origin/.venv-client/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN=python3; fi

export PYTHONPATH="${SDK_DAGGER_ROOT}:${SDK_DAGGER_ROOT}/setup:/home/ubuntu/dev/inference-origin/packages/openpi-client/src:/home/ubuntu/dev/inference-origin/client/inference${PYTHONPATH:+:${PYTHONPATH}}"

MODE="run"
MOCK_ARGS=()
PASS_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'TEXT'
用法：bash run_dagger_sdk.sh [选项]

纯 SDK（零 ROS）DAgger 采集入口，参考 inference-origin 直接使用 pyrealsense2 + piper_sdk + 原生 SocketCAN。

常用选项：
  --config <yaml>    指定配置文件（默认 config/dagger_precision_5090.yaml）
  --check-startup    只读检查 CAN、模型服务与三路 RealSense 相机，不停止 ROS 相机服务、不动机械臂
  --prepare-only     准备 CAN、模型预热并释放旧 ROS 相机占用，不动机械臂
  --mock             无硬件模拟运行全套流程
  --dataset-name <n> 指定数据集名称（默认按时间戳自动新建）
  --dataset-dir <d>  指定数据保存目录（默认 data/）
TEXT
      exit 0 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --check-startup) MODE="check"; shift ;;
    --prepare-only) MODE="prepare"; shift ;;
    --mock) MOCK_ARGS=(--mock); shift ;;
    *) PASS_ARGS+=("$1"); shift ;;
  esac
done

case "${MODE}" in
  check)
    exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" --config "${CONFIG}" --check-only ;;
  prepare)
    exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" --config "${CONFIG}" --warmup-model ;;
esac

if [[ ${#MOCK_ARGS[@]} -eq 0 ]]; then
  "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" --config "${CONFIG}"
fi

# Do not pin 200Hz high_follow + 3x RealSense + ffmpeg onto 3 cores (matches inference-origin)
exec "${PYTHON_BIN}" -u "${SDK_DAGGER_ROOT}/sdk_dagger/app.py" --config "${CONFIG}" "${MOCK_ARGS[@]}" "${PASS_ARGS[@]}"
