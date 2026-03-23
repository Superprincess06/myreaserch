# FGVC8 Multi-Label Baseline

This project provides a multi-label training baseline with timm ResNet-50.

## Project Layout

- data/raw: raw images and CSV labels
- data/synthetic: synthetic weather-degraded images (future use)
- models/baseline/resnet50.py: ResNet-50 baseline model
- models/mynet.py: placeholder custom model
- utils/dataset.py: dataset and CSV parsing
- utils/augment.py: image augmentation
- utils/metrics.py: macro F1 and accuracy metrics
- utils/losses.py: BCE and focal loss
- train.py: training entry
- evaluate.py: evaluation entry

## CSV Format

Your CSV should include:

1. one image path column (default uses the first column, or pass --image-col)
2. one or more numeric label columns (0/1)

Example:

| image | rust | mildew | scab |
|---|---:|---:|---:|
| images/0001.jpg | 1 | 0 | 1 |

## Install

```bash
pip install -r requirements.txt
```

## Train (ResNet-50)

```bash
python train.py \
  --train-csv data/raw/train.csv \
  --image-root data/raw \
  --model-name resnet50 \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4
```

Optional args:

- --val-csv data/raw/val.csv
- --label-cols rust,mildew,scab
- --image-col image
- --loss bce or --loss focal
- --no-pretrained

## Evaluate

```bash
python evaluate.py \
  --checkpoint weights/resnet50_xxx_best.pth \
  --csv-path data/raw/val.csv \
  --image-root data/raw
```

## Output

- Best checkpoint: weights/*.pth
- TensorBoard logs: logs/*
- Evaluation predictions: predictions.csv
