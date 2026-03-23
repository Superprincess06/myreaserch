from torch import nn

from .baseline.builder import build_baseline_model, SUPPORTED_BASELINES
from .mynet import MyNet

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = model_name.lower().strip()

    if name in SUPPORTED_BASELINES:
        return build_baseline_model(model_name=name, num_classes=num_classes, pretrained=pretrained)
    if name == "mynet":
        return MyNet(num_classes=num_classes)

    raise ValueError(f"Unsupported model_name: {model_name}")

__all__ = ["build_model"]
