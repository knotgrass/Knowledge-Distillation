import torch
import torch.nn as nn
import torchvision.models as models

# import timm


teacher = models.resnet152(pretrained= False, progress= True)
teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                       out_features=100, bias= True)


student = models.resnet18(pretrained= False, progress= True)
student.fc = nn.Linear(in_features=student.fc.in_features,
                       out_features=100, bias= True)

