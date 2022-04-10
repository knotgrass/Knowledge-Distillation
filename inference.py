import torch
# import torch.nn as nn

# import os
# import copy
from time import time
from colorama import Fore

from dataloader import loaders, transformers
from models.model import student
# from loss import loss_fn_kd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# student.load_state_dict(torch.load("weights/student.pth"))

datas, targets = iter(loaders['test']).next()
print(datas[0].shape, targets[0])
