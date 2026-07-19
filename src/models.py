import torch
import torch.nn as nn
from torchvision import models


def build_resnet18_classifier(num_classes=10):
    """
    Build ResNet-18 with a custom classifier head.
    Used to load the fine-tuned checkpoint for classification.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ResNet18EmbeddingExtractor(nn.Module):
    """
    Strip the classifier head from a fine-tuned ResNet-18
    to extract 512-dimensional embeddings.
    """

    def __init__(self, fine_tuned_model):
        super().__init__()
        self.features = nn.Sequential(*list(fine_tuned_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)