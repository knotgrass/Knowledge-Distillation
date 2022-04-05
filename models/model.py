import torch
import torch.nn as nn
import torchvision.models as models

from data import num_classes  # , class_to_idx, idx_to_class

# import timm



teacher = models.resnet101(pretrained=True, progress=True)
teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                       out_features=num_classes, bias=True)
teacher.load_state_dict(
    torch.load("weights/teacher.pth"))

student = models.resnet18(pretrained=True, progress=True)
student.fc = nn.Linear(in_features=student.fc.in_features,
                       out_features=num_classes, bias=True)
student.load_state_dict(
    torch.load("weights/student.pth"))
