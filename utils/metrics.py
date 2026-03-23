import torch


@torch.no_grad()
def compute_multilabel_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """计算多标签任务常用指标。

    说明：
    - 严格使用 sigmoid + 阈值，不使用 argmax。
    - Macro Precision / Macro Recall: 先按类别算，再在类别维度取平均。
    - Macro MCC: 先按类别算 Matthews Correlation Coefficient，再在类别维度取平均。
    - Macro F1: 先按类别算 F1，再在类别维度取平均。
    - Multi-label Accuracy: 所有样本-标签位点整体正确率（也称 Hamming accuracy）。
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).to(dtype=torch.float32)
    targets = targets.to(dtype=torch.float32)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1.0 - targets)).sum(dim=0)
    fn = ((1.0 - preds) * targets).sum(dim=0)
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=0)

    eps = 1e-8
    precision_per_class = tp / (tp + fp + eps)
    recall_per_class = tp / (tp + fn + eps)
    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()

    mcc_num = tp * tn - fp * fn
    mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc_per_class = mcc_num / mcc_den
    macro_mcc = mcc_per_class.mean().item()

    f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    macro_f1 = f1_per_class.mean().item()

    multilabel_acc = (preds == targets).float().mean().item()

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_mcc": macro_mcc,
        "macro_f1": macro_f1,
        "multilabel_acc": multilabel_acc,
    }
