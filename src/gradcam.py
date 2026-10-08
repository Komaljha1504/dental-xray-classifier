"""Grad-CAM (Selvaraju et al., 2017) implemented from scratch for the ResNet-18 backbone.

It shows which regions of the X-ray pushed the model toward its prediction. It is an interpretability aid,
not proof the model is looking at the right anatomy.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model.eval()
        self._acts = None
        self._grads = None
        layer.register_forward_hook(self._save_acts)
        layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, out):
        self._acts = out.detach()

    def _save_grads(self, _m, _gi, go):
        self._grads = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: 1 x 3 x H x W. Returns (cam in [0,1] with shape H x W, class probabilities, class used)."""
        self.model.zero_grad()
        logits = self.model(x.requires_grad_(True))
        probs = F.softmax(logits, dim=1)[0].detach().numpy()
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].numpy()
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam, probs, class_idx


def overlay(gray_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap over the grayscale X-ray. Returns an RGB uint8 image."""
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)
