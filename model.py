import torch
import torch.nn as nn
import torchvision.models as models
# import timm

from data import num_classes  # , class_to_idx, idx_to_class


teacher = models.resnet101(pretrained=True, progress=True)
teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                       out_features=num_classes, bias=True)
# teacher.load_state_dict(
#     torch.load("/content/gdrive/MyDrive/Classify_pytorch/Weights/teacher.pth", 
#     map_location=torch.device('cpu')))

student = models.resnet18(pretrained=True, progress=True)
student.fc = nn.Linear(in_features=student.fc.in_features,
                       out_features=num_classes, bias=True)
