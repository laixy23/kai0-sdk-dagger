#!/usr/bin/env bash
# 纯 SDK（零 ROS）PyQt5 桌面版 DAgger 工作台入口（内置双踏板待机循环 + LeRobot v2.1 实时/历史视图）
set -euo pipefail

SDK_DAGGER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SDK_DAGGER_CONFIG:-${SDK_DAGGER_ROOT}/config/dagger_precision_5090.yaml}"
PEDAL_CONFIG="${DAGGER_PEDAL_CONFIG:-${SDK_DAGGER_ROOT}/_runtime/foot-pedal.json}"
PYTHON_BIN="${SDK_DAGGER_PYTHON:-/home/ubuntu/dev/inference-origin/.venv-client/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN=python3; fi

export PYTHONPATH="${SDK_DAGGER_ROOT}:${SDK_DAGGER_ROOT}/setup:${SDK_DAGGER_ROOT}/operator:/home/ubuntu/dev/inference-origin/packages/openpi-client/src:/home/ubuntu/dev/inference-origin/client/inference${PYTHONPATH:+:${PYTHONPATH}}"

# Auto-detect local X11 display when launched from SSH on the workstation
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && -z "${QT_QPA_PLATFORM:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X1 ]]; then
    export DISPLAY=:1
  elif [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
  fi
fi

IS_MOCK=0
USE_PEDAL=1
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'TEXT'
用法：bash run_dagger_sdk_gui.sh [选项]

纯 SDK（零 ROS）PyQt5 桌面工作台：
  - 实时三路相机预览（双击放大）+ 双臂 6 轴/夹爪实时读数
  - DAgger 6 状态徽章 + 10 秒删除确认倒计时按钮 + 中文引导
  - 内置 USB 双踏板待机循环（免 PTY）：左单接管、右单保存、左双恢复/待机开新一轮、右双结束本轮进待机
  - 内置「历史数据与缺口 (v2.1)」Tab：查看 LeRobot v2.1 数据集、删除缺口并回放三路同步视频

常用选项：
  --check-startup      只读检查 CAN、踏板、模型服务与三路 RealSense（不动机械臂）
  --prepare-only       准备 CAN、踏板、模型预热并释放旧 ROS 相机占用（不动机械臂）
  --auto-start         打开界面后立即开始第一轮采集（默认先停在安全待机态，点按钮或左双踩开跑）
  --no-pedal           不连接 USB 双踏板（仅鼠标/键盘）
  --collector-name <n> 预选采集员姓名
  --mock               无硬件全链路模拟运行
  --config <yaml>      指定配置文件
TEXT
      exit 0 ;;
    --check-startup)
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" \
        --config "${CONFIG}" --check-only --with-pedal --pedal-config "${PEDAL_CONFIG}" ;;
    --prepare-only)
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" \
        --config "${CONFIG}" --with-pedal --pedal-config "${PEDAL_CONFIG}" --warmup-model ;;
    --mock)
      IS_MOCK=1
      USE_PEDAL=0
      FORWARD_ARGS+=(--mock --no-pedal)
      shift ;;
    --no-pedal)
      USE_PEDAL=0
      FORWARD_ARGS+=(--no-pedal)
      shift ;;
    --config)
      CONFIG="$2"
      FORWARD_ARGS+=(--config "$2")
      shift 2 ;;
    *)
      FORWARD_ARGS+=("$1")
      shift ;;
  esac
done

if [[ ${IS_MOCK} -eq 0 ]]; then
  PREP_ARGS=(--config "${CONFIG}")
  if [[ ${USE_PEDAL} -eq 1 ]]; then
    PREP_ARGS+=(--with-pedal --pedal-config "${PEDAL_CONFIG}")
  fi
  "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" "${PREP_ARGS[@]}"
fi

# Do not pin GUI + 200Hz high_follow + 3x RealSense + ffmpeg onto 3 cores (matches inference-origin)
exec "${PYTHON_BIN}" -u "${SDK_DAGGER_ROOT}/sdk_dagger/gui.py" \
  --config "${CONFIG}" --pedal-config "${PEDAL_CONFIG}" "${FORWARD_ARGS[@]}"
