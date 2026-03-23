import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

# 添加项目根目录到 Python 路径（便于导入 models）
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from models.baseline.builder import build_baseline_model, SUPPORTED_BASELINES

def _format_shape(output) -> str:
    if isinstance(output, torch.Tensor):
        return str(tuple(output.shape))
    if isinstance(output, (list, tuple)):
        shapes = []
        for item in output:
            if isinstance(item, torch.Tensor):
                shapes.append(str(tuple(item.shape)))
            else:
                shapes.append(type(item).__name__)
        return "[" + ", ".join(shapes) + "]"
    return type(output).__name__

def export_model_structure(
    model_name: str,
    model: nn.Module,
    save_dir: str,
    input_size: tuple[int, int, int, int] = (1, 3, 224, 224),
) -> str:
    """生成并保存模型结构的文本文档。"""
    model.eval()
    save_path = Path(save_dir) / f"{model_name}_structure.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 88)
    lines.append(f"Model Structure Snapshot: {model_name}")
    lines.append(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 88)
    lines.append("")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"Total Params: {total_params:,}")
    lines.append(f"Trainable Params: {trainable_params:,}")
    lines.append("")

    lines.append("[1] Full Model Print (PyTorch Native)")
    lines.append("-" * 88)
    lines.append(str(model))
    lines.append("")

    lines.append("[2] Named Modules (Flattened)")
    lines.append("-" * 88)
    for name, module in model.named_modules():
        module_name = name if name else "<root>"
        lines.append(f"{module_name:60s} -> {module.__class__.__name__}")
    lines.append("")

    lines.append("[3] Leaf Module Output Shapes (Forward Hook)")
    lines.append("-" * 88)

    hooks = []
    shape_records = []

    def register_hook(mod_name: str, mod: nn.Module):
        def hook(_all_mod, _inputs, output):
            shape_records.append(f"{mod_name:60s} -> {_format_shape(output)}")
        hooks.append(mod.register_forward_hook(hook))

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            mod_name = name if name else "<root>"
            register_hook(mod_name, module)

    try:
        dummy_input = torch.randn(*input_size)
        with torch.no_grad(), torch.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            _ = model(dummy_input)
    except Exception as e:
        shape_records.append(f"Error during forward pass tracing: {e}")
    finally:
        for h in hooks:
            h.remove()

    lines.extend(shape_records)
    lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    return str(save_path)

def main():
    print("开始生成所有支持基线模型的结构文档...")
    save_dir = project_root / "logs" / "architectures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = 6 # FGVC8 有6个病害标签
    
    for model_name in SUPPORTED_BASELINES.keys():
        print(f"  正在解析 {model_name} ...", end="", flush=True)
        try:
            # 设 pretrained=False 加快实例化速度，仅为了看结构
            model = build_baseline_model(model_name, num_classes=num_classes, pretrained=False)
            export_path = export_model_structure(model_name, model, save_dir=str(save_dir))
            print(f" [成功] 保存至 {export_path}")
        except Exception as e:
            print(f" [失败] 报错: {e}")
            
    print(f"\n所有可用模型结构均已导出到文件夹: {save_dir}")
    print("你可以去这个文件夹里查阅具体模型的分支和层级名，用于后续添加注意力机制或魔改！")

if __name__ == "__main__":
    main()
