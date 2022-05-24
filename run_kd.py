import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34

from distiller.distiller import Distiller
from distiller.loss import KDLoss
from distiller.teacher_train import training
from prepare_dataloader import dataset_sizes, loaders, num_classes

# from models.model import teacher#, student


if __name__ == "__main__":
    teacher = resnet34(pretrained=False, progress=True)
    teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                           out_features=num_classes, bias=True)
    teacher.load_state_dict(torch.load('weights/teacher.pth'))
    # teacher = training(loaders, dataset_sizes, 2, 2, teacher, r"weights/teacher.pth")
    # print(teacher)
    
    student = resnet18(pretrained=True, progress=True)
    student.fc = nn.Linear(in_features=student.fc.in_features,
                           out_features=num_classes, bias=True)
    
    distiller = Distiller(
        teacher=teacher,
        student=student,
        criterion= KDLoss(T=6, alpha=0.1, reduction='batchmean'))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epochs_warmup = 2
    epochs = 2
    save_ckpt = 20
    distiller.training_student(loaders, dataset_sizes, device,
                               epochs_warmup, epochs, 'resnet18', save_ckpt)
