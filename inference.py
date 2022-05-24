import torch, timm
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore
import numpy as np
import cv2
# from scipy.special import softmax
from distiller.loss import softmax
from typing import Tuple
# from prepare_dataloader import loaders

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Student(object):
    def __init__(self, input_size:tuple, num_classes:int, 
                 mean:tuple=(0.485, 0.456, 0.406), 
                 std:tuple=(0.229, 0.224, 0.225)) -> None:
        
        # create model
        self.model = models.resnet18(pretrained=False, progress=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                  out_features=num_classes, bias=True)
        
        # device and load weight
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.model.load_state_dict(torch.load('weights/student.pth'))
        else:
            self.device = torch.device('cpu')
            self.model.load_state_dict(
                torch.load('weights/student.pth', map_location=self.device))
        self.model.to(self.device).eval()
        
        # tranform datas
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        
        self.albumen_transform = A.Compose([
            A.Resize(*input_size, always_apply=True),
            A.Normalize(mean=mean, std=std, always_apply=True),
            ToTensorV2(always_apply=True),
        ], p=1.0)

    @torch.no_grad
    def inference(self, img:np.ndarray) -> Tuple[int, float]:
        img = self.transform(img)
        out = self.model(img.to(self.device)).detach().cpu().numpy()
        idx = out.argmax()
        dist = softmax(out)
        score = dist.max()
        
        return idx, score
        
