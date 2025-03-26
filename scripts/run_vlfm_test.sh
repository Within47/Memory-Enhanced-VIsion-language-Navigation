#!/bin/bash
# VLFM 增强版运行脚本 - 完善的环境配置和错误处理

# 设置彩色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 获取测试模式和步数
TEST_MODE=${1:-"full"}  # 默认为完整模式, MM-Nav, UA-Nav, LLM-GSP, MM-Nav_UA-Nav, MM-Nav_LLM-GSP, UA-Nav_LLM-GSP
MAX_STEPS=${2:-250}     # 默认250步

# 设置配置文件
CONFIG="experiments/enhanced_vlfm_objectnav_hm3d"  # 使用增强配置文件

# 根据测试模式设置环境变量
case "$TEST_MODE" in
  "baseline")
    # 基准模式 - 无任何增强功能
    export VLFM_USE_MEMORY="false"
    export VLFM_USE_UNCERTAINTY="false"
    export VLFM_MEMORY_REPULSION="false"
    export VLFM_CYCLE_DETECTION="false"
    export VLFM_ADAPTIVE_EXPLORER="false"
    TEST_NAME="baseline"
    ;;
  "MM-Nav")
    # 仅MM-Nav
    export VLFM_USE_MEMORY="true"
    export VLFM_USE_UNCERTAINTY="false"
    export VLFM_MEMORY_REPULSION="true"
    export VLFM_CYCLE_DETECTION="true"
    export VLFM_ADAPTIVE_EXPLORER="false"
    TEST_NAME="MM-Nav"
    ;;
  "UA-Nav")
    # 仅UA-Nav
    export VLFM_USE_MEMORY="false"
    export VLFM_USE_UNCERTAINTY="true"
    export VLFM_MEMORY_REPULSION="false"
    export VLFM_CYCLE_DETECTION="false"
    export VLFM_ADAPTIVE_EXPLORER="true"
    TEST_NAME="UA-Nav"
    ;;


  "full"|"all"|*)
    # 完整模式 - 所有功能
    export VLFM_USE_MEMORY="true"
    export VLFM_USE_UNCERTAINTY="true"
    export VLFM_MEMORY_REPULSION="true"
    export VLFM_CYCLE_DETECTION="true"
    export VLFM_ADAPTIVE_EXPLORER="true"


    TEST_NAME="full"
    ;;
esac

# 设置通用环境变量
export VLFM_MAX_STEPS=$MAX_STEPS
export VLFM_VERBOSE_LOG="true"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/${TEST_NAME}_run_$(date +"%Y%m%d_%H%M%S").log"


# 添加环境变量检查
echo "当前环境变量设置:"
echo "VLFM_USE_MEMORY: ${VLFM_USE_MEMORY}"
echo "VLFM_USE_UNCERTAINTY: ${VLFM_USE_UNCERTAINTY}"
echo "VLFM_MEMORY_REPULSION: ${VLFM_MEMORY_REPULSION}"
echo "VLFM_CYCLE_DETECTION: ${VLFM_CYCLE_DETECTION}"
echo "VLFM_ADAPTIVE_EXPLORER: ${VLFM_ADAPTIVE_EXPLORER}"

# 显示当前配置结构并保存
echo -e "\n${YELLOW}检查配置结构...${NC}"
python -m vlfm.run --config-name $CONFIG --info config &> config_structure.log

# 使用强制覆盖机制
echo -e "\n${GREEN}开始运行VLFM...${NC}"
echo "日志将保存至: $LOG_FILE"

python -m vlfm.run --config-name $CONFIG \
  "+habitat_baselines.rl.policy.use_mm_nav=${VLFM_USE_MEMORY}" \
  "+habitat_baselines.rl.policy.use_ua_explore=${VLFM_USE_UNCERTAINTY}" \
  "+habitat_baselines.rl.policy.use_llm_gsp=${VLFM_USE_LLM_GSP}" \
  "+habitat_baselines.rl.policy.enable_memory_repulsion=${VLFM_MEMORY_REPULSION}" \
  "+habitat_baselines.rl.policy.enable_cycle_detection=${VLFM_CYCLE_DETECTION}" \
  "+habitat_baselines.rl.policy.enable_adaptive_explorer=${VLFM_ADAPTIVE_EXPLORER}" \
  2>&1 | tee "$LOG_FILE"

exit_code=${PIPESTATUS[0]}
echo -e "\n${BLUE}===== VLFM运行结束 =====${NC}"
exit $exit_code
