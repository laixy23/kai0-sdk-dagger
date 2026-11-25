#!/usr/bin/env bash
# 纯 SDK（零 ROS）双踏板 DAgger 入口（默认带两态待机循环：右双踩结束本轮进入待机，左双踩开新一轮，终端 Ctrl+C 彻底退出）
set -euo pipefail

SDK_DAGGER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SDK_DAGGER_CONFIG:-${SDK_DAGGER_ROOT}/config/dagger_precision_5090.yaml}"
PEDAL_CONFIG="${DAGGER_PEDAL_CONFIG:-${SDK_DAGGER_ROOT}/_runtime/foot-pedal.json}"
PYTHON_BIN="${SDK_DAGGER_PYTHON:-/home/ubuntu/dev/inference-origin/.venv-client/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN=python3; fi

export PYTHONPATH="${SDK_DAGGER_ROOT}:${SDK_DAGGER_ROOT}/setup:${SDK_DAGGER_ROOT}/operator:/home/ubuntu/dev/inference-origin/packages/openpi-client/src:/home/ubuntu/dev/inference-origin/client/inference${PYTHONPATH:+:${PYTHONPATH}}"

STANDBY_ARGS=(--standby)
IS_MOCK=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'TEXT'
用法：bash run_dagger_sdk_pedal.sh [选项]

纯 SDK（零 ROS）双踏板 DAgger 采集入口：
  左单踩：示教接管 / 保存确认后开始下一段纠正
  右单踩：暂停跟随并保存本段（10 秒内按键盘 w 删除）
  左双踩：示教状态下恢复模型运行；待机状态下直接开启新一轮采集
  右双踩：结束本轮并保存剩余推理段，完成后进入待机状态
  终端 Ctrl+C：彻底退出整个程序

常用选项：
  --check-startup    只读检查 CAN、踏板权限、模型服务与三路 RealSense（不停止 ROS 相机、不动机械臂）
  --prepare-only     准备 CAN、踏板权限、模型预热并释放旧 ROS 相机占用，不动机械臂
  --test-pedal       仅测试双踏板按键识别，不动机器人
  --calibrate        校准双踏板（默认序列号 030032217272）
  --once             关闭待机循环（右双踩结束本轮后直接退出）
  --mock             无硬件模拟运行
  --config <yaml>    指定配置文件
TEXT
      exit 0 ;;
    --check-startup)
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" \
        --config "${CONFIG}" --check-only --with-pedal --pedal-config "${PEDAL_CONFIG}" ;;
    --prepare-only)
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" \
        --config "${CONFIG}" --with-pedal --pedal-config "${PEDAL_CONFIG}" --warmup-model ;;
    --test-pedal)
      shift
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/operator/foot_pedal_device.py" --test "${PEDAL_CONFIG}" "$@" ;;
    --calibrate)
      shift
      exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/operator/foot_pedal_device.py" --calibrate "${PEDAL_CONFIG}" "$@" ;;
    --once)
      STANDBY_ARGS=()
      shift ;;
    --mock)
      IS_MOCK=1
      FORWARD_ARGS+=(--mock)
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

if [[ ! -t 0 || ! -t 1 ]]; then
  echo "[错误] 需要交互终端；远程启动请使用 ssh -t。" >&2
  exit 2
fi

if [[ ${IS_MOCK} -eq 0 ]]; then
  "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/sdk_dagger/prepare.py" \
    --config "${CONFIG}" --with-pedal --pedal-config "${PEDAL_CONFIG}"
fi

exec "${PYTHON_BIN}" "${SDK_DAGGER_ROOT}/operator/run_pedal_session.py" \
  --config "${PEDAL_CONFIG}" "${STANDBY_ARGS[@]}" -- \
  bash "${SDK_DAGGER_ROOT}/run_dagger_sdk.sh" "${FORWARD_ARGS[@]}"
