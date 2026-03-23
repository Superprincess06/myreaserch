from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_split_quality(data_dir: str = "data/raw", out_dir: str = "logs/split_quality") -> dict:
    data_root = Path(data_dir)
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_root / "train_split.csv")
    val = pd.read_csv(data_root / "val_split.csv")
    test = pd.read_csv(data_root / "test_split.csv")
    full = pd.concat([train, val, test], axis=0, ignore_index=True)

    if "image" not in train.columns:
        raise ValueError("train_split.csv 中未找到 image 列。")

    label_cols = [c for c in train.columns if c != "image"]
    if not label_cols:
        raise ValueError("未检测到标签列，请确认切分文件包含多标签 0/1 列。")

    sizes = {
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "full": int(len(full)),
    }
    ratios = {k: float(v / sizes["full"]) for k, v in sizes.items() if k != "full"}

    prevalence = {
        "full": full[label_cols].mean(),
        "train": train[label_cols].mean(),
        "val": val[label_cols].mean(),
        "test": test[label_cols].mean(),
    }

    abs_gaps = pd.DataFrame(
        {
            "train_abs_gap": (prevalence["train"] - prevalence["full"]).abs(),
            "val_abs_gap": (prevalence["val"] - prevalence["full"]).abs(),
            "test_abs_gap": (prevalence["test"] - prevalence["full"]).abs(),
        }
    )

    cardinality = {
        "full": float(full[label_cols].sum(axis=1).mean()),
        "train": float(train[label_cols].sum(axis=1).mean()),
        "val": float(val[label_cols].sum(axis=1).mean()),
        "test": float(test[label_cols].sum(axis=1).mean()),
    }

    prevalence_df = pd.DataFrame(
        {
            "full": prevalence["full"],
            "train": prevalence["train"],
            "val": prevalence["val"],
            "test": prevalence["test"],
        }
    )
    ax = prevalence_df.plot(kind="bar", figsize=(11, 5), rot=25)
    ax.set_title("Label prevalence by split")
    ax.set_ylabel("Positive ratio")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    prevalence_plot = output_root / "label_prevalence.png"
    plt.savefig(prevalence_plot, dpi=180)
    plt.close()

    gap_matrix = abs_gaps.T.values
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(gap_matrix, aspect="auto")
    ax.set_yticks(np.arange(abs_gaps.shape[1]))
    ax.set_yticklabels(abs_gaps.columns)
    ax.set_xticks(np.arange(len(label_cols)))
    ax.set_xticklabels(label_cols, rotation=25, ha="right")
    ax.set_title("Absolute prevalence gap vs full set")
    threshold = float(gap_matrix.max()) * 0.55 if gap_matrix.size > 0 else 0.0
    for i in range(gap_matrix.shape[0]):
        for j in range(gap_matrix.shape[1]):
            cell_val = float(gap_matrix[i, j])
            color = "white" if cell_val > threshold else "black"
            ax.text(j, i, f"{cell_val:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    gap_plot = output_root / "prevalence_gap_heatmap.png"
    plt.savefig(gap_plot, dpi=180)
    plt.close()

    # Plot 3: Label sample counts by split (absolute numbers)
    label_counts = pd.DataFrame(
        {
            "full": full[label_cols].sum(),
            "train": train[label_cols].sum(),
            "val": val[label_cols].sum(),
            "test": test[label_cols].sum(),
        }
    )
    ax = label_counts.plot(kind="bar", figsize=(11, 5), rot=25)
    ax.set_title("Label sample counts by split (absolute)")
    ax.set_ylabel("Positive sample count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    count_plot = output_root / "label_sample_counts.png"
    plt.savefig(count_plot, dpi=180)
    plt.close()

    report = {
        "sizes": sizes,
        "ratios": ratios,
        "label_cols": label_cols,
        "max_abs_gap_vs_full": {k: float(v) for k, v in abs_gaps.max().to_dict().items()},
        "mean_abs_gap_vs_full": {k: float(v) for k, v in abs_gaps.mean().to_dict().items()},
        "cardinality": cardinality,
        "per_label_abs_gap": abs_gaps.round(6).to_dict(orient="index"),
        "label_sample_counts": label_counts.astype(int).to_dict(orient="index"),
        "plot_files": [str(prevalence_plot), str(gap_plot), str(count_plot)],
    }

    report_path = output_root / "split_quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    result = analyze_split_quality()

    print("=== Split Quality Summary ===")
    print("样本量:", result["sizes"])
    print("占比:", {k: round(v, 4) for k, v in result["ratios"].items()})
    print("最大标签占比偏差:", {k: round(v, 4) for k, v in result["max_abs_gap_vs_full"].items()})
    print("平均标签占比偏差:", {k: round(v, 4) for k, v in result["mean_abs_gap_vs_full"].items()})
    print("标签基数(每图平均标签数):", {k: round(v, 4) for k, v in result["cardinality"].items()})
    print("报告与图像已保存到 logs/split_quality")