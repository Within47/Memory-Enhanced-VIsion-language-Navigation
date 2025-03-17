#!/bin/bash
# VLFM 增强版运行脚本 - 完善的环境配置和错误处理

# 获取测试模式和步数
MAX_STEPS=${2:-250}     # 默认250步

# 设置配置文件
CONFIG="experiments/enhanced_vlfm_objectnav_hm3d"

# 根据测试模式设置环境变量

export VLFM_USE_MEMORY="true"
export VLFM_MEMORY_REPULSION="true"
export VLFM_CYCLE_DETECTION="true"


# 设置通用环境变量
export VLFM_MAX_STEPS=$MAX_STEPS
export VLFM_VERBOSE_LOG="true"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${TEST_NAME}_run_$(date +"%Y%m%d_%H%M%S").log"



# 显示当前配置结构并保存
python -m vlfm.run --config-name $CONFIG --info config &> config_structure.log

# 使用强制覆盖机制
echo "log will be save at: $LOG_FILE"

python -m vlfm.run --config-name $CONFIG \
  "+habitat_baselines.rl.policy.use_mm_nav=${VLFM_USE_MEMORY}" \
  "+habitat_baselines.rl.policy.enable_memory_repulsion=${VLFM_MEMORY_REPULSION}" \
  "+habitat_baselines.rl.policy.enable_cycle_detection=${VLFM_CYCLE_DETECTION}" \
  2>&1 | tee "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
exit $exit_code
