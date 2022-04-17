import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from colorama import Fore
import numpy as np
import cv2
# from scipy.special import softmax
from distiller.loss import softmax

from dataloader import loaders
from models.model import teacher
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Student(object):
    def __init__(self, input_size:tuple, mean, std, num_classes) -> None:
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
        self.tranform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean, std)
    ])

    @torch.no_grad
    def inference(self, img:np.ndarray) -> tuple:
        img = self.tranform(img)
        out = self.model(img.to(self.device)).detach().cpu().numpy()
        idx = out.argmax()
        dist = softmax(out)
        score = dist.max()
        
        return idx, score
        