#!/bin/bash
# =========================================================================
# 挂机指挥部：全自动串行跑完所有基线 (Linux 纯净版)
# =========================================================================

set -u -o pipefail

echo "🚀 即将开始【9 大基线狂潮模式】挂机训练！"

TRAIN_CSV=${TRAIN_CSV:-data/raw/train_split.csv}
VAL_CSV=${VAL_CSV:-data/raw/val_split.csv}
NUM_WORKERS=${NUM_WORKERS:-12}
IMAGE_SIZE=${IMAGE_SIZE:-224}
AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-1}

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

# 🌟 核心补丁：把公共参数（数据路径、多线程、图片大小）提炼出来，不用每次都写
COMMON_ARGS="--train-csv ${TRAIN_CSV} --val-csv ${VAL_CSV} --num-workers ${NUM_WORKERS} --image-size ${IMAGE_SIZE}"
FAILED_REPORT="logs/baseline/FAILED_MODELS.txt"
mkdir -p logs/baseline
: > "${FAILED_REPORT}"

FAILED_MODELS=()

run_model() {
	local model_name="$1"
	shift

	echo ""
	echo "======================================================="
	echo "🚀 开始训练: ${model_name}"
	echo "======================================================="

	python train.py --model-name "${model_name}" "$@" ${COMMON_ARGS}
	local status=$?

	if [ ${status} -ne 0 ]; then
		echo "❌ 模型 ${model_name} 训练失败，退出码: ${status}，已跳过继续下一个。"
		FAILED_MODELS+=("${model_name}")
	else
		echo "✅ 模型 ${model_name} 训练完成。"
	fi
}

# 1. 重型网络组
echo "👉 1. VGG16 (4090大显存：Batch提至64)"
run_model vgg16 --batch-size 64 --lr 4e-4 --weight-decay 5e-4 --epochs 30

echo "👉 2. ResNet50 (标准测试台 - Batch提至64)"
run_model resnet50 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --epochs 40

echo "👉 3. DenseNet121"
run_model densenet121 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --epochs 40

# 2. 轻量化移动端
echo "👉 4. MobileNetV3 Small (轻量级模型 - Batch拉满至128)"
run_model mobilenetv3_small --batch-size 128 --lr 4e-3 --epochs 50

echo "👉 5. MobileNetV3 Large"
run_model mobilenetv3_large --batch-size 128 --lr 2e-3 --epochs 50

echo "👉 6. ShuffleNetV2"
run_model shufflenetv2 --batch-size 128 --lr 2e-3 --epochs 50

echo "👉 7. GhostNet"
run_model ghostnet --batch-size 128 --lr 2e-3 --epochs 50

# 3. 现代纯血网络
echo "👉 8. ConvNeXtV2 (需要极大的权重衰减 0.05 收敛)"
run_model convnextv2 --batch-size 64 --lr 2e-3 --weight-decay 5e-2 --epochs 50

echo "👉 9. EfficientNet"
run_model efficientnet --batch-size 64 --lr 2e-3 --weight-decay 1e-4 --epochs 50

echo ""
echo "====================== 训练汇总 ======================"
if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
	echo "✅ 全部模型训练成功。"
else
	echo "⚠️ 以下模型训练失败: ${FAILED_MODELS[*]}"
	echo "请检查 logs/baseline/<model>_training.log 与终端报错。"
	{
		echo "Run Time: $(date '+%Y-%m-%d %H:%M:%S')"
		echo "Failed Models: ${FAILED_MODELS[*]}"
		for model in "${FAILED_MODELS[@]}"; do
			echo "- ${model}"
		done
	} > "${FAILED_REPORT}"
	echo "📝 失败清单已写入: ${FAILED_REPORT}"
fi
echo "======================================================"

echo "🎉🎉🎉 全自动挂机结束！所有模型已跑完。"

if [ "${AUTO_SHUTDOWN}" = "1" ]; then
	echo "⏻ 30 秒后自动关机（AUTO_SHUTDOWN=1）..."
	sleep 30
	if command -v shutdown >/dev/null 2>&1; then
		shutdown -h now || poweroff
	else
		poweroff
	fi
else
	echo "ℹ️ 已禁用自动关机（AUTO_SHUTDOWN=${AUTO_SHUTDOWN}）。"
fi