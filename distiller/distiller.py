from typing import Any, Tuple, Dict
import os, copy, os.path as osp
from colorama import Fore
from time import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import Tensor, nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from .loss import KDLoss
from distiller.print_utils import print_msg, print_time, desc


folder_save = osp.realpath(osp.join(__file__, '..', '..', 'weights'))
if not osp.isdir(folder_save): os.makedirs(folder_save)
print('Model will save in ', Fore.MAGENTA, folder_save)
batch_num:int = 0

def reset_batch_num(): global batch_num; batch_num = 0


def train(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
          teacher:nn.Module, best_acc:float, 
          criterion:_Loss, optimizer:optim.Optimizer, scheduler, 
          epochs:int, ckpt:int=20
          ) -> Tuple[nn.Module, float]:
    
    global batch_num
    best_teacher = copy.deepcopy(teacher)
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


def train_kd_4(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
               teacher:nn.Module, student:nn.Module, best_acc:float,
               criterion:KDLoss, optimizer:optim.Optimizer, scheduler:Any, 
               epochs:int, ckpt:int ) -> Tuple[nn.Module, float]:
    
    global batch_num
    model_name = student.__class__.__name__
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
                      desc=desc(epoch, epochs, phase, 0.0, best_acc)) as stream:
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
                    
                    # num_iter_data = idx*loaders[phase].batch_size
                    num_iter_data = idx*datas.size(0)
                    stream.set_description(
                        desc(epoch, epochs, phase, 
                             loss=running_loss/num_iter_data,
                             acc= running_corrects / num_iter_data))
                    stream.update()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                stream.set_description(
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


def train_kd(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
             teacher:nn.Module, student:nn.Module, best_acc:float,
             criterion:KDLoss, optimizer:optim.Optimizer, scheduler:Any, 
             epochs:int, ckpt:int ) -> Tuple[nn.Module, float]:
    
    global batch_num
    model_name = student.__class__.__name__
    student.to(device)
    teacher.to(device).eval()
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
                      desc=desc(epoch, epochs, phase, 0.0, best_acc)) as stream:
                for idx, (datas, targets) in enumerate(loaders[phase], start=1):
                    if phase == 'train': batch_num += 1
                    
                    datas= datas.to(device)
                    targets = targets.to(device)
                    with torch.no_grad():
                        outp_T = teacher(datas).detach()
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outp_S = student(datas) # forward
                        _, pred = torch.max(outp_S, 1)
                        loss = criterion(outp_S, targets, outp_T)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    #save checkpoint
                    if not batch_num % ckpt:
                        path_save = osp.join(folder_save, '{}_{}.pth'.format(model_name, batch_num))
                        torch.save(student.state_dict(), path_save)
                        
                    running_loss += loss.item()*datas.size(0)
                    running_corrects += torch.sum(pred == targets.data)
                    
                    # num_iter_data = idx*loaders[phase].batch_size
                    num_iter_data = idx*datas.size(0)
                    stream.set_description(
                        desc(epoch, epochs, phase, 
                             loss=running_loss/num_iter_data,
                             acc= running_corrects / num_iter_data))
                    stream.update()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                stream.set_description(
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


class _Distiller(object):
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
        reset_batch_num()
        
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
    r"""
    # create teacher, resnet34
    teacher = resnet34(pretrained=True, progress=True)
    teacher.fc = nn.Linear(in_features=teacher.fc.in_features,
                           out_features=num_classes, bias=True)
    teacher.load_state_dict(torch.load('path/to/teacher.pth'))
    
    # create student, resnet18
    student = resnet18(pretrained=True, progress=True)
    student.fc = nn.Linear(in_features=teacher.fc.in_features,
                           out_features=num_classes, bias=True)
    student.load_state_dict(torch.load('path/to/student.pth'))
    
    
    # create loss function
    kd_loss = KDLoss(T=6., alpha=0.1,reduction='batchmean')
    
    distiller = Distiller(
        device = torch.device('cuda:0' if torch.cuda.is_available()else 'cpu'),
        teacher= teacher, teacher_name= 'resnet34',
        student= student, student_name= 'resnet18',
        loaders= dict_of_3_dataloaders,
        dataset_sizes= dict_of_3_dataset_size,
        S_criterion=kd_loss,
        T_criterion=nn.CrossEntropyLoss(),
    )
    
    # train_teacher
    distiller.training_teacher(0, 10, 20)
    
    # train_student
    distiller.training_student(20, 30, 20)
    """
    
    def __init__(self, device:torch.device,
                 teacher:nn.Module, teacher_name:str,
                 student:nn.Module, student_name:str,
                 loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int],
                 S_criterion:KDLoss, T_criterion:_Loss=nn.CrossEntropyLoss()) -> None:
        self.teacher = teacher.to(device)
        self.teacher.__class__.__name__ = teacher_name
        self.student = student.to(device)
        self.student.__class__.__name__ = student_name
        self.device = device
        assert len(loaders) >= 2; self.loaders = loaders
        assert len(dataset_sizes) >=2; self.dataset_sizes = dataset_sizes
        self.S_criterion = S_criterion.to(device)
        self.T_criterion = T_criterion.to(device)
        
        print(Fore.RED)
        print('Device name {}'.format(torch.cuda.get_device_name(0)), Fore.RESET)
        
    def training_teacher(self, epochs_freeze:int, epochs_unfreeze:int, ckpt:int):
        optimizer = optim.Adam(list(self.teacher.children())[-1].parameters(), lr=0.001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, 
                                                   patience=3, verbose=True)
        since = time()

        best_acc = 0.0
        reset_batch_num()
        self.teacher, best_acc = train(self.loaders, self.dataset_sizes, self.device,
                                       self.teacher, best_acc, 
                                       self.T_criterion, optimizer, scheduler, 
                                       epochs_freeze, ckpt)
        
        time_elapsed = time() - since
        print('CLASSIFIER TRAINING TIME {} : {:.3f}'.format(
            time_elapsed//60, time_elapsed % 60))
        print_msg("Unfreeze all layers", self.teacher.__class__.__name__)
        
        # unfrezz all layer
        for param in self.teacher.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(self.teacher.parameters(), lr=0.0001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, 
                                                   patience=2, verbose=True)
        self.teacher, best_acc = train(self.loaders, self.dataset_sizes,
                                       self.teacher, best_acc, 
                                       self.T_criterion, optimizer, scheduler, 
                                       epochs_unfreeze, ckpt)
        
        last_teacher = osp.join(folder_save, '{}_last.pth'.format(
            self.teacher.__class__.__name__))
        torch.save(self.teacher.state_dict(), last_teacher)
        time_elapsed = time() - since
        print('TEACHER TRAINING TIME {} m {:.3f}s'.format(
            time_elapsed//60, time_elapsed % 60))
        
    def training_student(self, epochs_freeze:int, epochs_unfreeze:int, ckpt:int):
        optimizer = optim.Adam(list(self.student.children())[-1].parameters(), lr=0.001, 
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, 
                                                   patience=3, verbose=True)
        best_acc = 0.0
        model_name = self.student.__class__.__name__
        reset_batch_num()
        since = time()
        #NOTE train only classify/fullyconnected/head layers
        self.student, best_acc = train_kd(self.loaders, self.dataset_sizes, self.device, 
                                          self.teacher, self.student, best_acc,
                                          self.S_criterion, optimizer, scheduler,
                                          epochs_freeze, ckpt)
        
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
        self.student, best_acc = train_kd(self.loaders, self.dataset_sizes, self.device, 
                                          self.teacher, self.student, best_acc,
                                          self.S_criterion, optimizer, scheduler,
                                          epochs_unfreeze, ckpt)
        
        last_student = osp.join(folder_save, '{}_last.pth'.format(model_name))
        torch.save(self.student.state_dict(), last_student)
        time_elapsed = time() - since
        print('STUDENT TRAINING TIME {} m {:.3f}s'.format(
            time_elapsed//60, time_elapsed % 60))


