
# ============================================================
# model.py
# Builds ResNet, EfficientNet, and Vision Transformer (ViT)
# ============================================================

import torch
import torch.nn as nn
from torchvision import models

def build_model(model_name="resnet", num_classes=1716):
    """
    Builds a pretrained model (ResNet, EfficientNet, or ViT)
    and replaces the final layer with a new classifier
    for the given number of classes.
    """

    if model_name == "resnet":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "vit":
        model = models.vit_b_16(pretrained=True)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Invalid model name. Choose 'resnet', 'efficientnet', or 'vit'.")

    return model


