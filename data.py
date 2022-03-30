import os
import torch
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100, ImageFolder
import torchvision.transforms as T

transformers = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean= (0.4915, 0.4823, 0.4468),
        std=(0.2470, 0.2435, 0.2616)
    )
])

cifar100 = CIFAR100('dataset', train=True, download=True, transform=transformers)

dataset_size = len(cifar100)

test_size = round(0.2*dataset_size)
val_size = round(0.1*dataset_size)
train_size = dataset_size - val_size - test_size
generator = Generator()
generator.manual_seed(0)

train, val, test = random_split(cifar100, [train_size, val_size, test_size], 
                                generator=generator)

batch_size = 64
n_workers = os.cpu_count()
loader = {}
loader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True, 
                          num_workers=n_workers, generator=generator, pin_memory= True)
loader['val'] = DataLoader(val, batch_size=batch_size, shuffle=True, 
                          num_workers=n_workers, generator=generator, pin_memory= True)
loader['test'] = DataLoader(val, batch_size=batch_size, shuffle=False, 
                          num_workers=n_workers, generator=generator, pin_memory= True)

del generator
