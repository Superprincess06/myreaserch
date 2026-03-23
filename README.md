# FGVC8 苹果叶片多标签分类实验工程

本项目是一个面向 FGVC8 的多标签分类训练框架，支持 9 个主流 CNN 基线统一训练、自动评估、断点续训、离线预训练权重自动匹配与批量实验记录。

当前代码主打目标：
- 快速复现实验
- 统一管理多模型结果
- 便于后续替换自定义网络

## 1. 项目特性

- 支持 9 个基线：vgg16、resnet50、densenet121、mobilenetv3_small、mobilenetv3_large、shufflenetv2、ghostnet、convnextv2、efficientnet
- 支持本地离线预训练权重自动匹配（优先 safetensors）
- 训练阶段内置：Warmup + Cosine 学习率调度、AMP 混合精度、TensorBoard 记录
- 自动保存两类权重：
  - 最优权重：weights/<model>_best.pth
  - 断点权重：weights/<model>_last.pth
- 训练结束可自动触发 evaluate.py 评估
- 评估结果自动落盘到 logs/eval_results（JSON 明细 + CSV 汇总）
- 提供数据切分、切分质量分析、mini 数据集制作、结构导出等工具脚本

## 2. 目录说明

核心目录：
- train.py：训练入口
- evaluate.py：评估入口
- models/baseline/builder.py：基线模型工厂
- utils/dataset.py：多标签数据集读取
- utils/augment.py：训练/评估增强
- utils/metrics.py：Macro Precision/Recall/MCC/F1、多标签准确率
- utils/split_data.py：多标签分层切分
- tools/analyze_split_quality.py：切分质量分析
- tools/make_mini_dataset.py：构建 mini 调试集
- tools/download_all_pretrained.py：下载并保存离线预训练权重
- scripts/run_single.sh(.bat)：单模型训练脚本
- scripts/run_all_baselines.sh(.bat)：批量基线训练脚本

## 3. 环境安装

建议 Python 3.10+，并在虚拟环境中安装依赖。

```bash
pip install -r requirements.txt
```

如果你在 AutoDL/Linux 使用脚本，建议训练前先做环境体检：

```bash
bash scripts/autodl_preflight.sh
```

## 4. 数据准备与格式

### 4.1 原始数据要求

默认数据位置：
- 原始标签：data/raw/train.csv
- 图片目录：data/raw/train_images

train.csv 需包含：
- image：图像文件名
- labels：空格分隔的多标签字符串（例如 "rust scab"）

### 4.2 生成训练/验证/测试切分

```bash
python utils/split_data.py
```

输出：
- data/raw/train_split.csv
- data/raw/val_split.csv
- data/raw/test_split.csv

切分后的 CSV 格式为：
- 第 1 列：image
- 后续列：各标签 0/1（当前 6 类：complex、frog_eye_leaf_spot、healthy、powdery_mildew、rust、scab）

### 4.3 生成 mini 调试集（可选）

```bash
python tools/make_mini_dataset.py
```

输出：
- data/raw/mini_data/mini_train_split.csv
- data/raw/mini_data/mini_val_split.csv

### 4.4 检查切分质量（可选）

```bash
python tools/analyze_split_quality.py
```

输出目录：logs/split_quality
- split_quality_report.json
- label_prevalence.png
- prevalence_gap_heatmap.png
- label_sample_counts.png

## 5. 预训练权重（离线推荐）

项目支持自动从 weights/pretrained 按模型名匹配本地预训练权重。

先下载：

```bash
python tools/download_all_pretrained.py
```

下载后，训练时默认会自动匹配对应文件；你也可以手动指定：

```bash
python train.py --model-name resnet50 --pretrained-path weights/pretrained/resnet50.safetensors
```

## 6. 训练

### 6.1 单模型训练（推荐从这里开始）

```bash
python train.py \
  --model-name mobilenetv3_small \
  --train-csv data/raw/train_split.csv \
  --val-csv data/raw/val_split.csv \
  --image-dir data/raw/train_images \
  --epochs 50 \
  --batch-size 128 \
  --num-workers 8 \
  --image-size 224 \
  --lr 4e-3 \
  --weight-decay 1e-4
```

常用参数：
- --model-name：基线模型名（9 选 1）
- --pretrained / --no-pretrained：开关预训练
- --pretrained-path：手动指定本地预训练文件
- --pretrained-dir：自动搜索目录（默认 weights/pretrained）
- --resume：从上次中断继续
- --resume-path：手动指定断点文件
- --auto-evaluate / --no-auto-evaluate：训练后自动评估开关（默认开启）

### 6.2 断点续训

默认会写入 weights/<model>_last.pth。

```bash
python train.py --model-name resnet50 --resume
```

或指定断点文件：

```bash
python train.py --model-name resnet50 --resume --resume-path weights/resnet50_last.pth
```

### 6.3 使用脚本启动训练

Windows（单模型）：

```bat
scripts\run_single.bat resnet50
```

Linux（单模型）：

```bash
bash scripts/run_single.sh resnet50
```

## 7. 批量跑 9 个基线

Linux：

```bash
bash scripts/run_all_baselines.sh
```

Windows：

```bat
scripts\run_all_baselines.bat
```

训练汇总会自动追加到：
- logs/baseline/ALL_MODELS_SUMMARY.csv

## 8. 模型评估

### 8.1 手动评估

```bash
python evaluate.py \
  --checkpoint weights/ghostnet_best.pth \
  --test-csv data/raw/test_split.csv \
  --image-dir data/raw/train_images \
  --batch-size 64 \
  --num-workers 8 \
  --result-dir logs/eval_results
```

### 8.2 自动评估

train.py 默认开启训练后自动评估，评估命令会读取训练权重内保存的配置（例如 model_name、image_size）。

## 9. 输出文件说明

训练阶段：
- 权重：weights/<model>_best.pth、weights/<model>_last.pth
- 日志：logs/baseline/<model>_training.log
- TensorBoard：logs/baseline/<model>/events...
- 训练汇总：logs/baseline/ALL_MODELS_SUMMARY.csv

评估阶段：
- 单次评估明细：logs/eval_results/<model>_<ckpt>_<time>.json
- 评估总表：logs/eval_results/EVAL_SUMMARY.csv

结构导出：
- logs/architectures/<model>_structure.txt

## 10. 当前基线结果快照

基于 logs/eval_results/EVAL_SUMMARY.csv 的当前结果（按 Macro F1 观察）：

| 模型 | Macro F1 | Multi-label Acc |
|---|---:|---:|
| ghostnet | 0.898750 | 0.966061 |
| densenet121 | 0.898167 | 0.965526 |
| resnet50 | 0.894285 | 0.965170 |
| mobilenetv3_large | 0.893351 | 0.965705 |
| efficientnet | 0.892817 | 0.966061 |
| shufflenetv2 | 0.888451 | 0.963923 |
| vgg16 | 0.881600 | 0.961162 |
| convnextv2 | 0.874445 | 0.959202 |
| mobilenetv3_small | 0.871566 | 0.958667 |

说明：这是当前仓库日志的历史快照，后续训练会继续刷新结果。

## 11. 常见问题

1) 找不到图片文件
- 检查 --image-dir 是否与 CSV 的 image 列路径拼接一致。

2) 预训练加载失败
- 网络环境受限时建议先执行 tools/download_all_pretrained.py。
- 或使用 --no-pretrained 先验证流程。

3) 显存不足（OOM）
- 降低 --batch-size。
- 适当降低 --image-size。
- 优先尝试 mobilenetv3_small、shufflenetv2、ghostnet。

4) 自动评估不需要
- 训练时加 --no-auto-evaluate。

## 12. 后续扩展建议

- 在 models/mynet.py 中实现自定义网络，并在 models/baseline/builder.py 增加统一入口。
- 基于 logs/architectures 的结构导出结果，定位插入注意力模块的位置。
- 在当前 best baseline 上增加阈值搜索、类别重加权、TTA 等策略继续提升 Macro F1。
