import timm
import torch.nn as nn
import torchvision.models as tv_models
import warnings
import os
import torch
from safetensors.torch import load_file as load_safetensors_file

# 定义支持的基线模型映射表，方便统一管理和后续自动导出结构
SUPPORTED_BASELINES = {
    "vgg16": "vgg16",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "mobilenetv3_small": "mobilenetv3_small_100",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "shufflenetv2": "shufflenet_v2_x1_0", # 使用 torchvision
    "ghostnet": "ghostnet_100",
    "convnextv2": "convnextv2_atto", # 使用 atto/femto 等适合做基线的轻量版
    "efficientnet": "efficientnet_b0"
}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("无法从权重文件中解析 state_dict。")


def _load_local_weights(model: nn.Module, weight_path: str) -> None:
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"本地预训练权重不存在: {weight_path}")

    if weight_path.endswith(".safetensors"):
        state_dict = load_safetensors_file(weight_path, device="cpu")
    else:
        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)

    model_state = model.state_dict()
    filtered_state = {}
    skipped = 0
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            skipped += 1

    if not filtered_state:
        raise RuntimeError("本地预训练权重与当前模型结构不匹配，未加载任何参数。")

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    warnings.warn(
        "本地预训练加载完成: "
        f"loaded={len(filtered_state)}, skipped={skipped}, "
        f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}",
        RuntimeWarning,
    )

class BaselineModel(nn.Module):
    """统一的基线模型封装套壳"""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, pretrained_path: str = ""):
        super().__init__()
        
        if model_name not in SUPPORTED_BASELINES:
            raise ValueError(f"不支持的基线模型: {model_name}。请选择: {list(SUPPORTED_BASELINES.keys())}")
        
        use_local_pretrained = bool(pretrained_path and pretrained_path.strip())

        if model_name == "shufflenetv2":
            # torchvision 的 shufflenet_v2_x1_0
            if use_local_pretrained:
                self.model = tv_models.shufflenet_v2_x1_0(weights=None)
            else:
                weights = tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
                try:
                    self.model = tv_models.shufflenet_v2_x1_0(weights=weights)
                except Exception as e:
                    if not pretrained:
                        raise
                    warnings.warn(
                        f"加载 shufflenetv2 预训练权重失败，自动回退为随机初始化。原始错误: {e}",
                        RuntimeWarning,
                    )
                    self.model = tv_models.shufflenet_v2_x1_0(weights=None)

            # 替换最后的分类头
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            if use_local_pretrained:
                _load_local_weights(self.model, pretrained_path)
        else:
            timm_name = SUPPORTED_BASELINES[model_name]

            # 使用 timm 直接构建，并自动修改最终的全连接层为 num_classes 输出
            if use_local_pretrained:
                self.model = timm.create_model(
                    timm_name,
                    pretrained=False,
                    num_classes=num_classes
                )
                _load_local_weights(self.model, pretrained_path)
            else:
                try:
                    self.model = timm.create_model(
                        timm_name,
                        pretrained=pretrained,
                        num_classes=num_classes
                    )
                except Exception as e:
                    if not pretrained:
                        raise
                    warnings.warn(
                        f"加载 {model_name} 预训练权重失败，自动回退为随机初始化。原始错误: {e}",
                        RuntimeWarning,
                    )
                    self.model = timm.create_model(
                        timm_name,
                        pretrained=False,
                        num_classes=num_classes
                    )

    def forward(self, x):
        return self.model(x)

def build_baseline_model(model_name: str, num_classes: int, pretrained: bool = True, pretrained_path: str = "") -> nn.Module:
    """统一的模型构建接口。"""
    return BaselineModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
    )
