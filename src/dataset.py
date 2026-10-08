"""PyTorch dataset over cached uint8 X-ray arrays, with medical-imaging-safe augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as T

from config import IMAGENET_MEAN, IMAGENET_STD

_normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# Augmentation is mild on purpose: small rotations/shifts, left-right flip (a mirrored panoramic is still
# anatomically plausible), and brightness/contrast jitter to mimic different X-ray machines and exposures.
_train_aug = T.Compose(
    [
        T.RandomHorizontalFlip(0.5),
        T.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.25, contrast=0.25),
    ]
)


class XrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool):
        assert len(images) == len(labels)
        self.images = images  # N, H, W uint8 (grayscale)
        self.labels = labels.astype(np.int64)
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        gray = torch.from_numpy(self.images[i])  # H, W
        x = tv_tensors.Image(gray.unsqueeze(0).repeat(3, 1, 1))  # 3, H, W (ResNet expects 3 channels)
        if self.train:
            x = _train_aug(x)
        x = T.functional.to_dtype(x, torch.float32, scale=True)
        x = _normalize(x).as_subclass(torch.Tensor)
        return x, int(self.labels[i])


def preprocess_gray(gray_uint8: np.ndarray) -> torch.Tensor:
    """Single grayscale image (H, W, uint8) to a normalised 1 x 3 x H x W tensor, for inference."""
    x = torch.from_numpy(gray_uint8).unsqueeze(0).repeat(3, 1, 1)
    x = T.functional.to_dtype(tv_tensors.Image(x), torch.float32, scale=True)
    return _normalize(x).as_subclass(torch.Tensor).unsqueeze(0)
