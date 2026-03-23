#!/bin/bash
# =========================================================================
# 实验指挥部：单基线模型训练脚本
# 用法: bash scripts/run_single.sh <模型名字>
# 示例: bash scripts/run_single.sh resnet50
# =========================================================================

set -u -o pipefail

# 修正非法 OMP_NUM_THREADS，避免 libgomp 报错
if [ -z "${OMP_NUM_THREADS:-}" ] || ! [[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

# 接收终端输入的第一个词作为模型名
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
    echo "❌ 错误: 必须指定一个模型名称！"
    echo "💡 示例: bash scripts/run_single.sh resnet50"
    exit 1
fi

echo "======================================================="
echo "🚀 开始训练单模型: ${MODEL_NAME}"
echo "======================================================="

TRAIN_CSV=${TRAIN_CSV:-data/raw/train_split.csv}
VAL_CSV=${VAL_CSV:-data/raw/val_split.csv}
NUM_WORKERS=${NUM_WORKERS:-12}
IMAGE_SIZE=${IMAGE_SIZE:-224}
PRETRAINED=${PRETRAINED:-0}
PRETRAINED_PATH=${PRETRAINED_PATH:-}

COMMON_ARGS="--train-csv ${TRAIN_CSV} --val-csv ${VAL_CSV} --num-workers ${NUM_WORKERS} --image-size ${IMAGE_SIZE}"

if [ "${PRETRAINED}" = "1" ]; then
    COMMON_ARGS="${COMMON_ARGS} --pretrained"
else
    COMMON_ARGS="${COMMON_ARGS} --no-pretrained"
fi

if [ -n "${PRETRAINED_PATH}" ]; then
    if [ ! -f "${PRETRAINED_PATH}" ]; then
        echo "❌ 错误: PRETRAINED_PATH 指向的文件不存在: ${PRETRAINED_PATH}"
        exit 1
    fi
    COMMON_ARGS="${COMMON_ARGS} --pretrained-path ${PRETRAINED_PATH}"
fi

if [ ! -f "train.py" ]; then
    echo "❌ 错误: 当前目录未找到 train.py，请在项目根目录执行脚本。"
    exit 1
fi

if [ ! -f "${TRAIN_CSV}" ]; then
    echo "❌ 错误: 训练集 CSV 不存在: ${TRAIN_CSV}"
    exit 1
fi

if [ ! -f "${VAL_CSV}" ]; then
    echo "❌ 错误: 验证集 CSV 不存在: ${VAL_CSV}"
    exit 1
fi

if [ -f "scripts/autodl_preflight.sh" ]; then
    echo "🩺 先执行 AutoDL 预检..."
    bash scripts/autodl_preflight.sh || {
        echo "❌ 预检失败，请先修复后再训练。"
        exit 1
    }
fi

# 按模型自动匹配更合理的超参数，和 run_all_baselines.sh 保持一致
case "$MODEL_NAME" in
    vgg16)
        BATCH_SIZE=64
        LR=4e-4
        WEIGHT_DECAY=5e-4
        EPOCHS=30
        ;;
    resnet50|densenet121)
        BATCH_SIZE=64
        LR=1e-3
        WEIGHT_DECAY=1e-4
        EPOCHS=40
        ;;
    mobilenetv3_small)
        BATCH_SIZE=128
        LR=4e-3
        WEIGHT_DECAY=1e-4
        EPOCHS=50
        ;;
    mobilenetv3_large|shufflenetv2|ghostnet)
        BATCH_SIZE=128
        LR=2e-3
        WEIGHT_DECAY=1e-4
        EPOCHS=50
        ;;
    convnextv2)
        BATCH_SIZE=64
        LR=2e-3
        WEIGHT_DECAY=5e-2
        EPOCHS=50
        ;;
    efficientnet)
        BATCH_SIZE=64
        LR=2e-3
        WEIGHT_DECAY=1e-4
        EPOCHS=50
        ;;
    *)
        echo "❌ 错误: 不支持的模型名称 '$MODEL_NAME'"
        echo "💡 可选: vgg16 / resnet50 / densenet121 / mobilenetv3_small / mobilenetv3_large / shufflenetv2 / ghostnet / convnextv2 / efficientnet"
        exit 1
        ;;
esac

echo "🧪 当前超参数: epochs=${EPOCHS}, batch-size=${BATCH_SIZE}, lr=${LR}, weight-decay=${WEIGHT_DECAY}"

python train.py \
    --model-name ${MODEL_NAME} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    ${COMMON_ARGS}

STATUS=$?

if [ ${STATUS} -ne 0 ]; then
    echo "❌ ${MODEL_NAME} 训练失败，退出码: ${STATUS}"
    exit ${STATUS}
fi

echo ""
echo "🎉 ${MODEL_NAME} 训练结束！"