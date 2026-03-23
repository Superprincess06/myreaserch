@echo off
chcp 65001 >nul
:: =========================================================================
:: 挂机指挥部：全自动串行跑完所有 9 个基线并记录日志
:: 注意事项：请确保您的显卡有持续供电。中间如果有模型 Out of Memory，
::           脚本会自动跳过报错的直接跑下一个模型。
:: =========================================================================

echo 🚀 即将开始【9 大基线狂潮模式】挂机训练！请不要关闭终端...
echo ⏳ 启动倒计时：3秒...
timeout /t 3 >nul

:: 🌟 核心补丁：把公共参数（数据路径、多线程、图片大小）提炼出来，不用每次都写
set COMMON_ARGS=--train-csv data/raw/train_split.csv --val-csv data/raw/val_split.csv --num-workers 8 --image-size 224

:: ==========================
:: 1. 重型/经典网络组 (VGG / ResNet / DenseNet)
:: ==========================
echo 👉 正在开始第1组：经典重型骨干网络
:: VGG16 (显存占用极大，非常容易过拟合，给 16 批次，小学习率)
call python train.py --model-name vgg16 --batch-size 16 --lr 1e-4 --weight-decay 5e-4 --epochs 30 %COMMON_ARGS%
:: ResNet50 (绝对标杆，兼容性好)
call python train.py --model-name resnet50 --batch-size 32 --lr 5e-4 --weight-decay 1e-4 --epochs 40 %COMMON_ARGS%
:: DenseNet121 (参数少但激活层大，收敛快，特征复用强)
call python train.py --model-name densenet121 --batch-size 32 --lr 5e-4 --weight-decay 1e-4 --epochs 40 %COMMON_ARGS%

:: ==========================
:: 2. 轻量化移动端组合 (MobileNet / ShuffleNet / GhostNet)
:: ==========================
echo 👉 正在开始第2组：轻量化实时推理网络
:: MobileNetV3 Small (轻如鸿毛，学习率可放大)
call python train.py --model-name mobilenetv3_small --batch-size 64 --lr 2e-3 --epochs 50 %COMMON_ARGS%
:: MobileNetV3 Large
call python train.py --model-name mobilenetv3_large --batch-size 64 --lr 1e-3 --epochs 50 %COMMON_ARGS%
:: ShuffleNetV2 (算子优化极致，通常收敛比较曲折，需要稍长 epochs)
call python train.py --model-name shufflenetv2 --batch-size 64 --lr 1e-3 --epochs 50 %COMMON_ARGS%
:: GhostNet (华为提出的极简生成冗余特征图方案)
call python train.py --model-name ghostnet --batch-size 64 --lr 1e-3 --epochs 50 %COMMON_ARGS%

:: ==========================
:: 3. 现代高效混合网络 (ConvNeXt / EfficientNet)
:: ==========================
echo 👉 正在开始第3组：现代高效 SOTA网络
:: ConvNeXtV2 (卷积中的 Transformer 大满贯, 需要强正则化，也就是大 weight decay，这里用 5e-2 即 0.05)
call python train.py --model-name convnextv2 --batch-size 32 --lr 1e-3 --weight-decay 5e-2 --epochs 50 %COMMON_ARGS%
:: EfficientNet B0 (复合缩放王者，容易过拟合且占用显存较大)
call python train.py --model-name efficientnet --batch-size 32 --lr 1e-3 --weight-decay 1e-4 --epochs 50 %COMMON_ARGS%

echo.
echo 🎉🎉🎉 全部 9 个基线模型已全部挂机执行完毕！
echo 👉 日志和参数信息请查看 logs/baseline 文件夹。
pause
