import copy
from time import time
from typing import Any
from colorama import Fore

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import KDLoss, loss_fn_kd

criterion = KDLoss(T=6, alpha=0.1, reduction='batchmean')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def desc(epoch:int, n_epoch:int, phase:str, loss:float, acc:float):
    phase_str = '{:<5}'.format(phase.capitalize())
    epoch_str = Fore.RED +'Epoch'+ Fore.RESET +' {:>2d}/{:<2d}'.format(epoch, n_epoch)
    loss_str = Fore.LIGHTMAGENTA_EX + 'loss' + Fore.RESET + ' = {:.6f}'.format(loss)
    acc_str = Fore.LIGHTCYAN_EX + 'acc'+ Fore.RESET + ' = {:.3f}'.format(acc)
    
    return '{} - {} - {} - {}'.format(phase_str, epoch_str, loss_str, acc_str)


def train_kd(student:nn.Module, teacher:nn.Module, best_acc:float=0.0,
          criterion:_Loss=criterion,optimizer:optim.Optimizer=...,
          scheduler:lr_scheduler.ReduceLROnPlateau=..., 
          epochs:int= 20, loaders:dict=..., dataset_sizes:dict=...,
          device:torch.device = device, path_save_weight:str= ...
          ) -> tuple:
    
    since = time()
    best_student = copy.deepcopy(student)
    
    for epoch in range(1, epochs + 1): 
        for phase in ('train', 'val'):
            student.train(phase == 'train')
            
            running_loss = 0.0
            running_corrects = 0.0
            
            with tqdm(loaders[phase], ncols=128, colour='YELLOW', 
                      desc=desc(epoch, epochs, phase, 0.0, best_acc)) as progress:
                for idx, (datas, targets, outp_Teacher) in enumerate(loaders[phase]):
                    datas= datas.to(device)
                    targets = targets.to(device)
                    outp_Teacher = outp_Teacher.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outp_Student = student(datas)
                        
                        loss = criterion(outp_Student, targets, outp_Teacher)
                        _, pred = torch.max(outp_Student, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item()*datas.size(0)
                    running_corrects += torch.sum(pred == targets.data)
                    
                    
                    num_iter_data = (idx + 1 )*loaders[phase].batch_size
                    progress.set_description(
                        desc(epoch, epochs, phase, 
                             loss=running_loss/num_iter_data,
                             acc= running_corrects / num_iter_data
                             ))
                    progress.update()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                progress.set_description(
                    desc(epoch, epochs, phase, epoch_loss, epoch_acc))
                
            if phase == 'train':
                scheduler.step(100. * epoch_acc) #acc

            if phase == 'val':
                time_elapsed = time() - since
                print('Time: {}m {:.3f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_student = copy.deepcopy(student)
                torch.save(student.state_dict(), path_save_weight)
        
    return best_student, best_acc
class Distiller(nn.Module):
    def __init__(self, teacher:Any, student:nn.Module, loss_fn:_Loss):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_fn = loss_fn
    
    @staticmethod
    def distillation_loss(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T, alpha:float):
        return T * T * alpha * F.kl_div(F.log_softmax(preds / T, dim=1), 
                                        F.softmax(teacher_preds / T, dim=1),
                                        reduction='batchmean') + (1. - alpha) * F.cross_entropy(preds, labels)
    
    def training_student(self, 
                        optimizer:optim.Optimizer, 
                        scheduler,
                        train_loader:DataLoader, 
                        val_loader:DataLoader,
                        n_epochs:int = 30):
        
        ...
