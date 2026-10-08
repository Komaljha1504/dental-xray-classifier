"""ResNet-18 transfer learning: ImageNet-pretrained backbone, new classification head."""
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, n_classes))
    return model


FREEZE_STAGES = {
    "none": [],
    "layer1": ["conv1", "bn1", "layer1"],
    "layer2": ["conv1", "bn1", "layer1", "layer2"],
}


def freeze_early(model: nn.Module, upto: str = "layer2") -> None:
    """Freeze the generic low-level feature layers (edges, textures) of the ImageNet backbone.

    This is standard transfer learning and, on a CPU-only machine, roughly doubles training speed. The frozen
    BatchNorm layers also stay in eval mode so their ImageNet statistics are not disturbed (see set_train_mode).
    """
    prefixes = FREEZE_STAGES[upto]
    for name, p in model.named_parameters():
        p.requires_grad = not any(name.startswith(f) for f in prefixes)
    model._frozen_prefixes = prefixes


def set_train_mode(model: nn.Module, train: bool) -> None:
    model.train(train)
    if train:
        for name, module in model.named_modules():
            if any(name == f or name.startswith(f + ".") for f in getattr(model, "_frozen_prefixes", [])):
                module.eval()


def load_trained(path, n_classes: int, device="cpu") -> nn.Module:
    model = build_model(n_classes, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()
