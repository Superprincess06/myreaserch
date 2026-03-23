import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline.builder import build_baseline_model, SUPPORTED_BASELINES
from utils.dataset import FGVC8Dataset
from utils.augment import get_eval_transforms
from utils.metrics import compute_multilabel_metrics

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """在测试集上计算损失、Macro Precision/Recall/MCC/F1 与 Multi-label Accuracy。"""
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for images, labels in tqdm(loader, ncols=100, desc="🔍 正在评估"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 开启混合精度加速评估（更快）
        amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    # 调用指标计算，由于 metrics.py 里包含 sigmoid 操作，这里阈值应该用概率空间的 0.5
    metrics = compute_multilabel_metrics(logits=logits_cat, targets=targets_cat, threshold=0.5)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="多标签病害识别基线测试")
    # 只需要指定权重文件和测试集路径，其他的统统自动读取！
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 .pth 权重文件路径")
    parser.add_argument("--test-csv", type=str, default="data/raw/test_split.csv")
    parser.add_argument("--image-dir", type=str, default="data/raw/train_images")
    parser.add_argument("--batch-size", type=int, default=64) # 测试时不需要反向传播，Batch Size 可以开大点
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--result-dir", type=str, default="logs/eval_results", help="评估结果保存目录")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"📦 正在加载权重文件: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    # 🌟 核心优化：从当初保存的字典中恢复训练时的参数
    train_args = ckpt.get("args", {})
    model_name = train_args.get("model_name", "mobilenetv3_small")
    image_size = train_args.get("image_size", 224)
    
    label_cols = ckpt.get("label_cols", None)

    print(f"🔧 自动读取训练配置: 模型={model_name}, 分辨率={image_size}x{image_size}")

    test_dataset = FGVC8Dataset(
        csv_file=args.test_csv,
        img_dir=args.image_dir,
        transform=get_eval_transforms(image_size=image_size),
    )

    if label_cols is not None and test_dataset.label_cols != label_cols:
        raise ValueError(
            "checkpoint 与测试集标签列不一致，请确认 test_csv 与训练任务匹配。"
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_baseline_model(model_name=model_name, num_classes=len(test_dataset.label_cols), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model=model, loader=test_loader, criterion=criterion, device=device)

    os.makedirs(args.result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    run_name = f"{model_name}_{ckpt_stem}_{file_ts}"

    # 每次评估保存一份完整明细，方便复盘与对比
    detail_file = os.path.join(args.result_dir, f"{run_name}.json")
    detail = {
        "time": timestamp,
        "model": model_name,
        "checkpoint": args.checkpoint,
        "test_csv": args.test_csv,
        "image_dir": args.image_dir,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "metrics": {
            "loss": float(metrics["loss"]),
            "macro_precision": float(metrics["macro_precision"]),
            "macro_recall": float(metrics["macro_recall"]),
            "macro_mcc": float(metrics["macro_mcc"]),
            "macro_f1": float(metrics["macro_f1"]),
            "multilabel_acc": float(metrics["multilabel_acc"]),
        },
    }
    with open(detail_file, "w", encoding="utf-8") as f:
        import json
        json.dump(detail, f, ensure_ascii=False, indent=2)

    # 追加到统一汇总表，方便快速看所有模型的横向表现
    summary_file = os.path.join(args.result_dir, "EVAL_SUMMARY.csv")
    summary_exists = os.path.isfile(summary_file)
    with open(summary_file, "a", encoding="utf-8") as f:
        if not summary_exists:
            f.write(
                "Time,Model,Checkpoint,TestCSV,ImageDir,ImageSize,BatchSize,NumWorkers,Loss,MacroPrecision,MacroRecall,MacroMCC,MacroF1,MultilabelAcc\n"
            )
        f.write(
            f"{timestamp},{model_name},{args.checkpoint},{args.test_csv},{args.image_dir},{image_size},{args.batch_size},{args.num_workers},"
            f"{metrics['loss']:.6f},{metrics['macro_precision']:.6f},{metrics['macro_recall']:.6f},{metrics['macro_mcc']:.6f},{metrics['macro_f1']:.6f},{metrics['multilabel_acc']:.6f}\n"
        )

    print("\n" + "="*40)
    print("🏆 测试集最终评估结果:")
    print("="*40)
    print(f"➤ Test Loss:             {metrics['loss']:.6f}")
    print(f"➤ Macro Precision:       {metrics['macro_precision']:.6f}")
    print(f"➤ Macro Recall:          {metrics['macro_recall']:.6f}")
    print(f"➤ Macro MCC:             {metrics['macro_mcc']:.6f}")
    print(f"➤ Macro F1-Score:        {metrics['macro_f1']:.6f}")
    print(f"➤ Multi-label Accuracy:  {metrics['multilabel_acc']:.6f}")
    print(f"📁 详细结果已保存: {detail_file}")
    print(f"📊 汇总表已更新:   {summary_file}")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()