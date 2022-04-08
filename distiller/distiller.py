import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
Tensor = torch.Tensor

class Distiller(nn.Module):
    def __init__(self, teacher:nn.Module, student:nn.Module):
        super().__init__()
        self.teacher = teacher
        self.student = student
    
    @staticmethod
    def distillation_loss(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T, alpha:float):
        return T * T * alpha * F.kl_div(F.log_softmax(preds / T, dim=1), 
                                        F.softmax(teacher_preds / T, dim=1),
                                        reduction='batchmean') + (1. - alpha) * F.cross_entropy(preds, labels)
    
    def train_and_valid(self, 
                        criterion:nn.Module, 
                        optimizer:optim.Optimizer, 
                        scheduler,
                        train_loader:DataLoader, 
                        val_loader:DataLoader,
                        n_epochs:int = 30):
        
        ...
