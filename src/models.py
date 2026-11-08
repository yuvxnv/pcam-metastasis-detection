"""The two models: a small CNN trained from scratch, and a pretrained ResNet18."""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SimpleCNN(nn.Module):
    """4 conv blocks (conv -> batchnorm -> ReLU -> maxpool), then a small classifier.
    Outputs ONE number per image (a 'logit'); sigmoid turns it into a probability."""

    def __init__(self):
        super().__init__()
        chans = [3, 32, 64, 128, 128]
        blocks = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            blocks += [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                       nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*blocks)          # 96 -> 48 -> 24 -> 12 -> 6
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)


def pretrained_resnet18():
    """ResNet18 pretrained on ImageNet; final layer swapped for a 1-output head.
    All layers are fine-tuned (not frozen)."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return _SqueezeOut(m)


class _SqueezeOut(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).squeeze(1)


def build_model(name):
    if name == "cnn":
        return SimpleCNN()
    if name == "resnet18":
        return pretrained_resnet18()
    raise ValueError(name)
