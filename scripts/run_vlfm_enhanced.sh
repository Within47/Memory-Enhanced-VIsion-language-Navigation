#!/bin/bash
# VLFM运行脚本 - 最简版本

# 设置基本参数
CONFIG=${1:-"experiments/vlfm_objectnav_hm3d"}
MAX_STEPS=${2:-250}

# 设置必要的环境变量
export PYTHONUNBUFFERED=1
export VLFM_MAX_STEPS=$MAX_STEPS

# 设置功能模块环境变量
export VLFM_USE_LLM_GSP="false"
export VLFM_USE_MEMORY="true"
export VLFM_USE_UNCERTAINTY="false"
export VLFM_MEMORY_REPULSION="false"
export VLFM_CYCLE_DETECTION="false"
export VLFM_ADAPTIVE_EXPLORER="false"
export HYDRA_FULL_ERROR=1

# 创建日志目录和文件
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/vlfm_run_$(date +"%Y%m%d_%H%M%S").log"

# 运行程序并保存日志
python -m vlfm.run --config-name $CONFIG 2>&1 | tee "$LOG_FILE" 