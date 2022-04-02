import torch.nn as nn
import torchvision.models as models
# import timm

from data import num_classes  # , class_to_idx, idx_to_class


teacher = models.resnet152(pretrained=False, progress=True)
teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                       out_features=num_classes, bias=True)


student = models.resnet18(pretrained=False, progress=True)
student.fc = nn.Linear(in_features=student.fc.in_features,
                       out_features=num_classes, bias=True)
