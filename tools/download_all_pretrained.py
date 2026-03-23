import json
from pathlib import Path

import timm
import torch
import torchvision.models as tv_models
from safetensors.torch import save_file

OUT_DIR = Path("weights/pretrained")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# model_key -> timm model name in this repo
TIMM_MODELS = {
    "vgg16": "vgg16",
    "resnet50": "resnet50",
    "densenet121": "densenet121",
    "mobilenetv3_small": "mobilenetv3_small_100",
    "mobilenetv3_large": "mobilenetv3_large_100",
    "ghostnet": "ghostnet_100",
    "convnextv2": "convnextv2_atto",
    "efficientnet": "efficientnet_b0",
}

results = {
    "ok": {},
    "failed": {},
}


def _save_safetensors(state_dict: dict, path: Path) -> None:
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_state, str(path))


for key, timm_name in TIMM_MODELS.items():
    out_path = OUT_DIR / f"{timm_name}.safetensors"
    try:
        print(f"[timm] downloading {timm_name}...")
        model = timm.create_model(timm_name, pretrained=True)
        _save_safetensors(model.state_dict(), out_path)
        results["ok"][key] = str(out_path).replace("\\", "/")
        print(f"[ok] {key} -> {out_path}")
    except Exception as exc:
        results["failed"][key] = f"{type(exc).__name__}: {exc}"
        print(f"[fail] {key}: {exc}")

# shufflenetv2 is from torchvision in this repo
try:
    print("[torchvision] downloading shufflenetv2...")
    model = tv_models.shufflenet_v2_x1_0(weights=tv_models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
    out_path = OUT_DIR / "shufflenet_v2_x1_0.safetensors"
    _save_safetensors(model.state_dict(), out_path)
    results["ok"]["shufflenetv2"] = str(out_path).replace("\\", "/")
    print(f"[ok] shufflenetv2 -> {out_path}")
except Exception as exc:
    results["failed"]["shufflenetv2"] = f"{type(exc).__name__}: {exc}"
    print(f"[fail] shufflenetv2: {exc}")

manifest_path = OUT_DIR / "manifest.json"
manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[done] manifest -> {manifest_path}")
print(json.dumps(results, ensure_ascii=False, indent=2))
