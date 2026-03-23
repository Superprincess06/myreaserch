@echo off
chcp 65001 >nul
:: =========================================================================
:: 实验指挥部：单基线模型训练脚本 (Windows 专用)
:: 用法: 双击运行，或者在终端输入 .\scripts\run_single.bat <模型名字>
:: 示例: .\scripts\run_single.bat resnet50
:: =========================================================================

:: 接收终端输入的第一个词作为模型名
set MODEL_NAME=%1

if "%MODEL_NAME%"=="" (
    echo ❌ 错误: 必须指定一个模型名称！
    echo 💡 示例: .\scripts\run_single.bat resnet50
    pause
    exit /b 1
)

echo =======================================================
echo 🚀 开始训练单模型: %MODEL_NAME%
echo =======================================================

:: 🌟 核心补丁：把公共参数（数据路径、多线程、图片大小）提炼出来，不用每次都写
set COMMON_ARGS=--train-csv data/raw/train_split.csv --val-csv data/raw/val_split.csv --num-workers 8 --image-size 224

python train.py ^
    --model-name %MODEL_NAME% ^
    --epochs 40 ^
    --batch-size 32 ^
    --lr 5e-4 ^
    --weight-decay 1e-4 ^
    %COMMON_ARGS%

echo.
echo 🎉 %MODEL_NAME% 训练结束！
pause
