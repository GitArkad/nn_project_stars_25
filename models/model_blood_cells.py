from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn

NUM_CLASSES = 4


def build_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model(model_path):
    model = build_model()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
