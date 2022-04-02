import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import copy
from colorama import Fore
from tqdm import tqdm
from time import time

from data import loaders, dataset_sizes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(student, best_student, best_acc, 
          criterion, optimizer, scheduler, 
          epochs, path_save_weight:str):
    since = time()
    
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train': student.train()
            else:                student.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for datas, targets in tqdm(loaders[phase]):
                datas, targets = datas.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outp = student(datas)
                    _, pred = torch.max(outp, 1)
                    loss = criterion(outp, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()*datas.size(0)
                running_corrects += torch.sum(pred == targets.data)

            if phase == 'train':
                acc = 100. * running_corrects.double() / dataset_sizes[phase]
                scheduler.step(acc)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                print(Fore.RED)
                print('Epoch: {}/{}'.format(epoch+1, epochs), Fore.RESET)
            print('{} - loss = {:.6f}, accuracy = {:.3f}'.format(
                phase, epoch_loss, 100*epoch_acc))

            if phase == 'val':
                print('Time: {}m {:.3f}s'.format(
                    int((time() - since)//60), (time() - since) % 60))
                print('=='*22)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_student = copy.deepcopy(student.state_dict())

                torch.save(student.state_dict(), path_save_weight)
        # scheduler.step()
        
    return best_student, best_acc


def training(epochs_freeze:int, epochs_unfreeze:int, 
             teacher:nn.Module, path_save_weight:str=None):
    if path_save_weight is None:
        if os.path.isdir('Weights'):
            os.makedirs('Weights')
        path_save_weight = os.path.join(
            'Weights', teacher.__class__.__name__ + '.pth')
    print('Training {} using {}'.format(teacher.__class__.__name__, device))

    teacher.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(teacher.children())[-1].parameters(), lr=0.001, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                               factor=0.5, patience=3, verbose=True)
    
    since = time()
    best_teacher = copy.deepcopy(teacher.state_dict())
    best_acc = 0.0
    
    best_teacher, best_acc = train(teacher, best_teacher, best_acc, 
                                   criterion, optimizer, scheduler, 
                                   epochs_freeze, path_save_weight)

    time_elapsed = time() - since
    print('CLASSIFIER TRAINING TIME {} : {:.3f}'.format(
        time_elapsed//60, time_elapsed % 60))
    print(Fore.RED, '\n', 'Unfreeze all layers of {} model'.format(
        teacher.__class__.__name__), '\n', '=='*22, Fore.RESET, '\n')

    teacher.load_state_dict(best_teacher)

    # unfrezz all layer
    for param in teacher.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(teacher.parameters(), lr=0.0001, 
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, 
                                               patience=2, verbose=True)
    
    best_teacher, best_acc = train(teacher, best_teacher, best_acc, 
                                   criterion, optimizer, scheduler, 
                                   epochs_unfreeze, path_save_weight)
    
    torch.save(best_teacher.state_dict(), path_save_weight)
    time_elapsed = time() - since
    print('ALL NET TRAINING TIME {} m {:.3f}s'.format(
        time_elapsed//60, time_elapsed % 60))

    return best_teacher
