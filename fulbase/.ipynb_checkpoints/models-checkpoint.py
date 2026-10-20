from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flat_dim = self._forward_conv(dummy).view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z)


class ResNet18ForSmallImage(nn.Module):
    """
    适配 CIFAR-10 / Fashion-MNIST 的 ResNet18。

    注意：
    1. 保留 features(x) 接口，方便 DP 类原型统计；
    2. 保留 classifier 属性，方便 prototype_lda.py 重建分类头；
    3. 对 32x32 小图像，使用 3x3 conv、stride=1，并去掉原始 maxpool。
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()

        backbone = resnet18(weights=None)

        # 原始 ResNet18 是给 ImageNet 224x224 设计的：
        # conv1 = 7x7, stride=2, maxpool
        # 对 CIFAR-10 这种 32x32 图像不太合适，所以改成小图像版本。
        backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        backbone.maxpool = nn.Identity()

        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z)


def build_model(dataset: str = "fashionmnist", num_classes: int = 10):
    dataset = dataset.lower()

    if dataset in {"fashionmnist", "mnist"}:
        # Fashion-MNIST / MNIST 是 1 通道 28x28。
        # 可以跑 ResNet，但会明显比 SmallCNN 慢。
        return ResNet18ForSmallImage(in_channels=1, num_classes=num_classes)

    if dataset == "cifar10":
        # CIFAR-10 是 3 通道 32x32，更适合用 ResNet18。
        return ResNet18ForSmallImage(in_channels=3, num_classes=num_classes)

    raise ValueError(f"Unsupported dataset: {dataset}")