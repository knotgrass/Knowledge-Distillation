import copy
from time import time
from typing import Any, Tuple, Dict
from colorama import Fore
import os.path as osp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import KDLoss
from distiller.print_utils import print_msg, print_time, desc

# class Loaders(TypedDict):
#     # https://peps.python.org/pep-0589/#using-typeddict-types
#     'train':DataLoader
#     'val':DataLoader
#     'test':DataLoader


folder_save = 'weights'
if not osp.isdir(folder_save): os.makedirs(folder_save)
batch_num:int = 0
#TODO move train teacher here, remove best teacher
def train(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
          teacher:nn.Module, best_teacher:nn.Module, best_acc:float, 
          criterion:_Loss, optimizer:optim.Optimizer, scheduler, 
          epochs:int, ckpt:int=20
          ) -> Tuple[nn.Module, float]:
    
    global batch_num
    since = time()
    
    for epoch in range(1, epochs+1):
        for phase in ('train', 'val'):
            if phase == 'train': 
                teacher.train()
                print(Fore.RED); print('Epoch : {:>2d}/{:<2d}'.format(
                    epoch, epochs), Fore.RESET, ' {:>48}'.format('='*46))
            else:
                teacher.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for datas, targets in tqdm(loaders[phase], ncols=64, colour='green', 
                                       desc='{:6}'.format(phase).capitalize()):
                
                if phase == 'train': batch_num += 1
                
                datas, targets = datas.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outp = teacher(datas)
                    _, pred = torch.max(outp, 1)
                    loss = criterion(outp, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*datas.size(0)
                running_corrects += torch.sum(pred == targets.data)
                
                #save checkpoint
                if not batch_num % ckpt:
                    path_save = osp.join(folder_save, '{}_{}.pth'.format(
                        teacher.__class__.__name__, batch_num))
                    torch.save(teacher.state_dict(), path_save)
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                scheduler.step(100. * epoch_acc) #acc
                
            print('{} - loss = {:.6f}, accuracy = {:.3f}'.format(
                '{:5}'.format(phase).capitalize(), epoch_loss, 100*epoch_acc))

            if phase == 'val':
                time_elapsed = time() - since
                print('Time: {}m {:.3f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_teacher = copy.deepcopy(teacher)
                path_save = osp.join(folder_save, '{}_best.pth'.format(
                    teacher.__class__.__name__))
                torch.save(teacher.state_dict(), path_save)
    return best_teacher, best_acc


def train_kd(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
             student:nn.Module, best_acc:float,
             criterion:_Loss, optimizer:optim.Optimizer, scheduler:Any, 
             epochs:int, model_name:str, ckpt:int=20
             ) -> Tuple[nn.Module, float]:
    
    global batch_num
    student.to(device)
    best_student = copy.deepcopy(student)
    since = time()
    
    for epoch in range(1, epochs + 1): 
        for phase in ('train', 'val'):
            if phase == 'train': 
                student.train()
                print(Fore.RED); print('Epoch : {:>2d}/{:<2d}'.format(
                    epoch, epochs), Fore.RESET, ' {:>48}'.format('='*46))
            else:
                student.eval()
            
            running_loss = 0.0
            running_corrects = 0.0
            
            with tqdm(loaders[phase], ncols=128, colour='YELLOW', 
                      desc=desc(epoch, epochs, phase, 0.0, best_acc)) as progress:
                for idx, (datas, targets, soft_label) in enumerate(loaders[phase], start=1):
                    if phase == 'train': batch_num += 1
                    
                    datas= datas.to(device)
                    targets = targets.to(device)
                    # if is_aug: soft_label = teacher(datas)
                    soft_label = soft_label.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outp_S = student(datas) # forward
                        _, pred = torch.max(outp_S, 1)
                        loss = criterion(outp_S, targets, soft_label)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #save checkpoint
                    if not batch_num % ckpt:
                        path_save = osp.join(folder_save, '{}_{}.pth'.format(model_name, batch_num))
                        torch.save(student.state_dict(), path_save)
                        
                    running_loss += loss.item()*datas.size(0)
                    running_corrects += torch.sum(pred == targets.data)
                    
                    num_iter_data = idx*loaders[phase].batch_size
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
                
            print('{} - loss = {:.6f}, accuracy = {:.3f}'.format(
                '{:5}'.format(phase).capitalize(), epoch_loss, 100*epoch_acc))
            
            if phase == 'train':
                scheduler.step(100. * epoch_acc) #acc
            else:# phase == 'val'
                time_elapsed = time() - since
                print('Time: {}m {:.3f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_student = copy.deepcopy(student)
                path_save = osp.join(folder_save, '{}_best.pth'.format(model_name))
                torch.save(student.state_dict(), path_save)
        
    return best_student, best_acc


class Distiller(object):
    def __init__(self, teacher:Any, student:nn.Module, criterion:_Loss) -> None:
        # super().__init__()
        self.teacher = teacher
        self.student = student
        self.criterion = criterion # KDLoss(T=6, alpha=0.1, reduction='batchmean')

    @staticmethod
    def distillation_loss(preds:Tensor, labels:Tensor, teacher_preds:Tensor, T, alpha:float) -> Tensor:
        return T * T * alpha * F.kl_div(F.log_softmax(preds / T, dim=1), 
                                        F.softmax(teacher_preds / T, dim=1),
                                        reduction='batchmean') + (1. - alpha) * F.cross_entropy(preds, labels)

    def training_student(self, device:torch.device,
                         loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int],
                         epochs_warmup:int, epochs:int, model_name:str, ckpt:int,
                         ) -> nn.Module:
        
        assert len(loaders) >= 2 and len(dataset_sizes) >= 2, 'please check loaders'
        
        #TODO write desc of input and output
        #TODO https://tinyurl.com/8wnknv9p
        # all param unless classify layer must freeze at the first train
        # for param in teacher.parameters():
        #     param.requires_grad = False
        
        self.student.to(device)
        optimizer = optim.Adam(list(self.student.children())[-1].parameters(), lr=0.001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, 
                                                   patience=3, verbose=True)
        best_acc = 0.0
        since = time()
        
        #NOTE train only classify/fullyconnected/head layers
        self.student, best_acc = train_kd(loaders, dataset_sizes, device,
                                          self.student, best_acc, 
                                          self.criterion, optimizer, scheduler,
                                          epochs_warmup, model_name, ckpt)
        print(end='\n')
        print_time('FREEZE TRAINING TIME', time() - since)
        print_msg("Unfreeze all layers", model_name)
        
        # unfrezz all layer
        for param in self.student.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(self.student.parameters(), lr=0.0001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, 
                                                   patience=2, verbose=True)
        
        #NOTE train all layers of model
        self.student, best_acc = train_kd(loaders, dataset_sizes, device,
                                          self.student, best_acc, 
                                          self.criterion, optimizer, scheduler,
                                          epochs, model_name, ckpt)
        last_student = osp.join(folder_save, '{}_last.pth'.format(model_name))
        torch.save(self.student.state_dict(), last_student)
        time_elapsed = time() - since
        print('ALL NET TRAINING TIME {} m {:.3f}s'.format(
            time_elapsed//60, time_elapsed % 60))
        
        return self.student
    
    
class Distiller(object):
    def __init__(self, device:torch.device,
                 teacher:nn.Module, teacher_name:str,
                 student:nn.Module, student_name:str,
                 loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int],
                 KDcriterion:KDLoss, criterion:_Loss=nn.CrossEntropyLoss) -> None:
        self.teacher = teacher.to(device)
        self.teacher.__class__.__name__ = teacher_name
        self.student = student.to(device)
        self.student.__class__.__name__ = student_name
        self.device = device
        assert len(loaders) >= 2; self.loaders = loaders
        assert len(dataset_sizes) >=2; self.dataset_sizes = dataset_sizes
        self.S_criterion = KDcriterion.to(device)
        self.T_criterion = criterion.to(device)
        
        print(Fore.RED)
        print('Device name {}'.format(torch.cuda.get_device_name(0)), Fore.RESET)
        
        
    def training_teacher(self, epochs_warmup:int, epochs:int, ckpt:int):
        optimizer = optim.Adam(list(self.teacher.children())[-1].parameters(), lr=0.001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, 
                                                   patience=3, verbose=True)
        since = time()
        best_teacher = copy.deepcopy(self.teacher)
        best_acc = 0.0
        best_teacher, best_acc = train(self.loaders, self.dataset_sizes, self.device,
                                       self.teacher, best_teacher, best_acc, 
                                       self.T_criterion, optimizer, scheduler, 
                                       epochs_warmup, ckpt)
        
        time_elapsed = time() - since
        print('CLASSIFIER TRAINING TIME {} : {:.3f}'.format(
            time_elapsed//60, time_elapsed % 60))
        print_msg("Unfreeze all layers", self.teacher.__class__.__name__)
        
        self.teacher.load_state_dict(best_teacher.state_dict())
        
        # unfrezz all layer
        for param in self.teacher.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(self.teacher.parameters(), lr=0.0001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, 
                                                   patience=2, verbose=True)
        best_teacher, best_acc = train(self.loaders, self.dataset_sizes,
                                       self.teacher, best_teacher, best_acc, 
                                       self.T_criterion, optimizer, scheduler, 
                                       epochs, ckpt)
        
        last_teacher = osp.join(folder_save, '{}_last.pth'.format(
            self.teacher.__class__.__name__))
        torch.save(best_teacher.state_dict(), last_teacher)
        time_elapsed = time() - since
        print('ALL NET TRAINING TIME {} m {:.3f}s'.format(
            time_elapsed//60, time_elapsed % 60))
