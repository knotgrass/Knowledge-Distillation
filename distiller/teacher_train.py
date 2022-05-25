import copy
import os
import os.path as osp
from time import time
from colorama import Fore
from tqdm import tqdm
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from distiller.print_utils import print_msg

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
folder_save = 'weights'
if not osp.isdir(folder_save): os.makedirs(folder_save)
batch_num:int = 0

def train(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
          teacher:nn.Module, best_teacher:nn.Module, best_acc:float, 
          criterion:_Loss, optimizer:optim.Optimizer, scheduler, 
          epochs:int, model_name:str, ckpt:int=20
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
                    path_save = osp.join(folder_save, '{}_{}.pth'.format(model_name, batch_num))
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
                path_save = osp.join(folder_save, '{}_best.pth'.format(model_name))
                torch.save(teacher.state_dict(), path_save)
    return best_teacher, best_acc


def training(loaders:Dict[str, DataLoader], dataset_sizes:Dict[str, int], device:torch.device,
             epochs_freeze:int, epochs_unfreeze:int, 
             teacher:nn.Module, model_name:str, ckpt:int=20
             ) -> nn.Module:
    
    assert len(loaders) >=2 and len(dataset_sizes) >=2, 'please check loaders'
    print('Training {} on {}'.format(model_name, torch.cuda.get_device_name(0)))

    teacher.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(teacher.children())[-1].parameters(), lr=0.001, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                               patience=3, verbose=True)
    
    since = time()
    best_teacher = copy.deepcopy(teacher)
    best_acc = 0.0
    
    best_teacher, best_acc = train(loaders, dataset_sizes, device,
                                   teacher, best_teacher, best_acc, 
                                   criterion, optimizer, scheduler, 
                                   epochs_freeze, model_name, ckpt)

    time_elapsed = time() - since
    print('CLASSIFIER TRAINING TIME {} : {:.3f}'.format(
        time_elapsed//60, time_elapsed % 60))
    print_msg("Unfreeze all layers", teacher.__class__.__name__)

    teacher.load_state_dict(best_teacher.state_dict())

    # unfrezz all layer
    for param in teacher.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(teacher.parameters(), lr=0.0001, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                                               patience=2, verbose=True)
    
    best_teacher, best_acc = train(loaders, dataset_sizes,
                                   teacher, best_teacher, best_acc, 
                                   criterion, optimizer, scheduler, 
                                   epochs_unfreeze, model_name, ckpt)
    
    last_teacher = osp.join(folder_save, '{}_last.pth'.format(model_name))
    torch.save(best_teacher.state_dict(), last_teacher)
    time_elapsed = time() - since
    print('ALL NET TRAINING TIME {} m {:.3f}s'.format(
        time_elapsed//60, time_elapsed % 60))

    return best_teacher
