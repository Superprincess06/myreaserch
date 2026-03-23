import argparse
import os
import random
import logging
import time
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.baseline.builder import build_baseline_model, SUPPORTED_BASELINES
from tools.export_all_structures import export_model_structure
from utils.dataset import FGVC8Dataset
from utils.augment import get_train_transforms, get_eval_transforms
from utils.metrics import compute_multilabel_metrics


def _auto_find_pretrained_path(model_name: str, pretrained_dir: str) -> str:
    """按模型名自动匹配本地预训练权重路径。找不到时返回空字符串。"""
    if not os.path.isdir(pretrained_dir):
        return ""

    valid_exts = (".safetensors", ".pth", ".pt")
    candidates = []
    for name in os.listdir(pretrained_dir):
        lower = name.lower()
        if lower.endswith(valid_exts):
            candidates.append(name)

    if not candidates:
        return ""

    model_key = model_name.lower()
    mapped_name = SUPPORTED_BASELINES.get(model_name, "").lower()

    def score(filename: str) -> tuple[int, int, str]:
        lower = filename.lower()
        s = 0
        if mapped_name and mapped_name in lower:
            s += 120
        if model_key in lower:
            s += 100
        if lower.endswith(".safetensors"):
            s += 20
        if "in1k" in lower:
            s += 5
        if "lamb" in lower:
            s += 4
        # 次级排序: 更短通常更通用，最后用文件名稳定排序
        return (s, -len(filename), filename)

    ranked = sorted(candidates, key=score, reverse=True)
    best = ranked[0]

    # 兜底: 至少要命中模型名或映射名，否则视为未找到
    best_lower = best.lower()
    if not ((mapped_name and mapped_name in best_lower) or (model_key in best_lower)):
        return ""

    return os.path.join(pretrained_dir, best)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证卷积运算的可复现性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None, # 🌟 接收 AMP 放大器
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_logits = []
    all_targets = []

    for images, labels in tqdm(loader, ncols=100, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            # 🌟 开启混合精度加速上下文 (如果传了 scaler 就启用)
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    # 🌟 AMP 专属反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 普通反向传播 (本地 CPU 用)
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_multilabel_metrics(logits=logits_cat, targets=targets_cat, threshold=0.5)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description="多标签病害识别基线训练")
    parser.add_argument("--train-csv", type=str, default="data/raw/mini_data/mini_train_split.csv")
    parser.add_argument("--val-csv", type=str, default="data/raw/mini_data/mini_val_split.csv")
    parser.add_argument("--image-dir", type=str, default="data/raw/train_images")
    parser.add_argument("--epochs", type=int, default=30) # 基线跑 30-50 轮即可
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="mobilenetv3_small", choices=list(SUPPORTED_BASELINES.keys()))
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True, help="是否加载预训练权重（离线环境建议 --no-pretrained）")
    parser.add_argument("--pretrained-path", type=str, default="", help="本地预训练权重路径（.safetensors/.pth/.pt）；设置后优先于 --pretrained")
    parser.add_argument("--pretrained-dir", type=str, default="weights/pretrained", help="自动匹配本地预训练权重的目录")
    parser.add_argument("--log-dir", type=str, default="logs/baseline")
    parser.add_argument("--save-path", type=str, default="weights/best_model.pth")
    parser.add_argument("--resume", action="store_true", help="是否从最近一次中断处继续训练")
    parser.add_argument("--resume-path", type=str, default="", help="手动指定断点权重路径（默认: weights/<model>_last.pth）")
    parser.add_argument("--auto-evaluate", action=argparse.BooleanOptionalAction, default=True, help="训练结束后是否自动运行 evaluate.py")
    parser.add_argument("--eval-csv", type=str, default="data/raw/test_split.csv", help="自动评估时使用的测试集 CSV")
    parser.add_argument("--eval-image-dir", type=str, default="data/raw/train_images", help="自动评估时使用的图片目录")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="自动评估 batch size")
    parser.add_argument("--eval-num-workers", type=int, default=8, help="自动评估 num_workers")
    parser.add_argument("--eval-result-dir", type=str, default="logs/eval_results", help="自动评估结果保存目录")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs("weights", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 🌟 配置日志记录器（同时输出到终端和文件）
    log_file = os.path.join(args.log_dir, f"{args.model_name}_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    train_dataset = FGVC8Dataset(
        csv_file=args.train_csv,
        img_dir=args.image_dir,
        transform=get_train_transforms(image_size=args.image_size),
    )
    val_dataset = FGVC8Dataset(
        csv_file=args.val_csv,
        img_dir=args.image_dir,
        transform=get_eval_transforms(image_size=args.image_size),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"========== 正在使用的训练设备: {device} ==========")

    # 若用户未手动指定 pretrained_path，则按模型名自动匹配本地预训练文件
    if args.pretrained and not args.pretrained_path:
        auto_pretrained_path = _auto_find_pretrained_path(args.model_name, args.pretrained_dir)
        if auto_pretrained_path:
            args.pretrained_path = auto_pretrained_path
            logger.info(f"🔎 自动匹配到本地预训练权重: {args.pretrained_path}")
        else:
            logger.info("ℹ️ 未在本地目录匹配到预训练权重，将按原逻辑使用远端预训练或回退随机初始化。")

    num_classes = len(train_dataset.label_cols)
    model = build_baseline_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
    ).to(device)

    arch_log_dir = os.path.join(args.log_dir, args.model_name)
    os.makedirs(arch_log_dir, exist_ok=True)
    current_arch_path = export_model_structure(
        model_name=args.model_name, model=model, save_dir=arch_log_dir, input_size=(1, 3, args.image_size, args.image_size),
    )
    logger.info(f"当前结构文件: {current_arch_path}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 🌟 组合调度器：前 5 轮 Warmup (极小学习率爬升)，之后余弦退火
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # 🌟 初始化 AMP 加速器 (只有在 GPU 下才启用)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ✅ 修复 2：为每个模型建立独立的 TensorBoard 子文件夹，防止曲线互相覆盖打架
    tb_dir = os.path.join(args.log_dir, args.model_name)
    writer = SummaryWriter(log_dir=tb_dir)
    best_f1 = -1.0
    start_epoch = 1

    # 断点续训：默认从 weights/<model>_last.pth 恢复
    last_ckpt_path = os.path.join("weights", f"{args.model_name}_last.pth")
    if args.resume:
        resume_path = args.resume_path.strip() if args.resume_path else last_ckpt_path
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"未找到 resume 文件: {resume_path}")

        logger.info(f"♻️ 正在从断点恢复: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)

        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in resume_ckpt and torch.cuda.is_available():
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])

        best_f1 = float(resume_ckpt.get("best_val_f1", -1.0))
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1

        if start_epoch > args.epochs:
            logger.info(f"✅ 已完成训练：checkpoint epoch={start_epoch - 1}，目标 epochs={args.epochs}。")
            writer.close()
            return

        logger.info(f"✅ 恢复成功，将从 Epoch {start_epoch} 继续训练，当前 best_f1={best_f1:.4f}")

    train_start_time = time.time() # 记录整个训练的开始时间

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time() # 记录当前 epoch 的开始时间
        
        train_metrics = run_one_epoch(
            model=model, loader=train_loader, criterion=criterion, device=device, optimizer=optimizer, scaler=scaler
        )

        val_metrics = run_one_epoch(
            model=model, loader=val_loader, criterion=criterion, device=device, optimizer=None, scaler=scaler
        )

        scheduler.step()

        writer.add_scalar("Train Loss", train_metrics["loss"], epoch)
        writer.add_scalar("Val Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val Precision", val_metrics["macro_precision"], epoch)
        writer.add_scalar("Val Recall", val_metrics["macro_recall"], epoch)
        writer.add_scalar("Val MCC", val_metrics["macro_mcc"], epoch)
        writer.add_scalar("Val F1-Score", val_metrics["macro_f1"], epoch)
        writer.add_scalar("Val Accuracy", val_metrics["multilabel_acc"], epoch)

        # 动态获取当前真实的学习率打出日志
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time # 计算当前 epoch 耗时
        epoch_time_str = f"{int(epoch_time // 60)}m {int(epoch_time % 60):02d}s"
        
        logger.info(f"[Epoch {epoch:03d}/{args.epochs:03d} | LR: {current_lr:.6f} | Time: {epoch_time_str}] "
                    f"Train Loss={train_metrics['loss']:.4f} | "
                    f"Val Loss={val_metrics['loss']:.4f} | "
                    f"Val Precision={val_metrics['macro_precision']:.4f} | "
                    f"Val Recall={val_metrics['macro_recall']:.4f} | "
                    f"Val MCC={val_metrics['macro_mcc']:.4f} | "
                    f"Val Macro F1={val_metrics['macro_f1']:.4f} | "
                    f"Val Acc={val_metrics['multilabel_acc']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            # 动态修改保存名字，比如 weights/mobilenetv3_small_best.pth，防止不同基线互相覆盖
            save_name = os.path.join("weights", f"{args.model_name}_best.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_cols": train_dataset.label_cols,
                "best_val_f1": best_f1,
                "args": vars(args),
            }, save_name)
            logger.info(f"🌟 发现更优模型，已保存到: {save_name}")

        # 每个 epoch 都存一份可恢复断点，避免长任务中断后从头再来
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if torch.cuda.is_available() else None,
            "label_cols": train_dataset.label_cols,
            "best_val_f1": best_f1,
            "args": vars(args),
        }, last_ckpt_path)

    total_train_time = time.time() - train_start_time
    total_time_str = f"{int(total_train_time // 3600)}h {int((total_train_time % 3600) // 60)}m {int(total_train_time % 60)}s"

    writer.close()
    logger.info(f"🎉 训练完成 [{args.model_name}]，总耗时: {total_time_str}，最佳验证集 Macro F1: {best_f1:.4f}")

    # 🌟 核心：训练结束后，把结果追加写入一个公共的汇总 CSV 文件
    summary_file = os.path.join(args.log_dir, "ALL_MODELS_SUMMARY.csv")
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode="a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Time,Model,Best_Macro_F1\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{args.model_name},{best_f1:.4f}\n")
    logger.info(f"📊 本次模型成绩已自动汇总至: {summary_file}")

    if args.auto_evaluate:
        best_ckpt = os.path.join("weights", f"{args.model_name}_best.pth")
        if not os.path.isfile(best_ckpt):
            logger.warning(f"⚠️ 自动评估跳过：未找到最佳权重 {best_ckpt}")
            return

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--checkpoint",
            best_ckpt,
            "--test-csv",
            args.eval_csv,
            "--image-dir",
            args.eval_image_dir,
            "--batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.eval_num_workers),
            "--result-dir",
            args.eval_result_dir,
        ]
        logger.info("🧪 开始自动评估（evaluate.py）...")
        result = subprocess.run(eval_cmd, check=False)
        if result.returncode != 0:
            logger.warning(f"⚠️ 自动评估失败，退出码: {result.returncode}")
        else:
            logger.info("✅ 自动评估完成，结果已写入评估结果目录。")

if __name__ == "__main__":
    main()