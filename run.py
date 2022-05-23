from prepare_dataloader import loaders, dataset_sizes, num_classes
from distiller.teacher_train import training
# from models.model import teacher#, student

import torch.nn as nn
from torchvision.models import resnet18

if __name__ == "__main__":
    teacher = resnet18(pretrained=True, progress=True)
    teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                       out_features=num_classes, bias=True)
    teacher = training(loaders, dataset_sizes, 2, 2, teacher, r"weights/teacher.pth")
    print(teacher)