#!/bin/bash
# AutoDL 训练前体检脚本：尽量在开跑前拦截常见环境问题

set -u -o pipefail

# 修正非法 OMP_NUM_THREADS，避免 libgomp 报错
if [ -z "${OMP_NUM_THREADS:-}" ] || ! [[ "${OMP_NUM_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS=1
fi

echo "================ AutoDL 预检开始 ================"

if ! command -v python >/dev/null 2>&1; then
    echo "❌ 未找到 python，请先激活正确环境。"
    exit 1
fi

echo "✅ Python: $(python -V 2>&1)"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ 检测到 GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️ 未找到 nvidia-smi，可能不是 GPU 实例。"
fi

echo "🧪 检查关键 Python 依赖..."
python - <<'PY'
import importlib
mods = [
    'torch', 'torchvision', 'timm', 'pandas', 'numpy', 'cv2',
    'PIL', 'tqdm', 'tensorboard', 'huggingface_hub', 'safetensors'
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print('❌ 缺少依赖:', ', '.join(missing))
    raise SystemExit(1)
print('✅ 依赖检查通过')
PY

if [ ! -f "data/raw/train_split.csv" ]; then
    echo "⚠️ 未找到 data/raw/train_split.csv"
fi

if [ ! -f "data/raw/val_split.csv" ]; then
    echo "⚠️ 未找到 data/raw/val_split.csv"
fi

if [ ! -d "data/raw/train_images" ]; then
    echo "⚠️ 未找到 data/raw/train_images"
fi

mkdir -p logs/baseline
mkdir -p weights

echo "📦 当前磁盘占用："
df -h .

echo "================ AutoDL 预检完成 ================"
