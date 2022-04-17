import os
from colorama import Fore

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100#, ImageFolder

from config import cfg


cifar100 = CIFAR100('dataset', train=True, download=True,
                    transform=cfg.transformers['original'])

dataset_size = len(cifar100)

test_size = round(0.2*dataset_size)
val_size = round(0.1*dataset_size)
train_size = dataset_size - val_size - test_size

generator = cfg.generator

train, val, test = random_split(cifar100, [train_size, val_size, test_size],
                                generator=generator)

batch_size = cfg.batch_size
n_workers = os.cpu_count()
loaders = {}
loaders['train'] = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, generator=generator, pin_memory=True)
loaders['val'] = DataLoader(val, batch_size=batch_size, shuffle=True,
                            num_workers=n_workers, generator=generator, pin_memory=True)
loaders['test'] = DataLoader(test, batch_size=batch_size, shuffle=True,
                             num_workers=n_workers, generator=generator, pin_memory=True)
del generator

dataset_sizes = {
    'train': train.__len__(),
    'val': val.__len__(),
    'test': test.__len__(),
}
num_classes = len(cifar100.class_to_idx)

# for classname, index in cifar100.class_to_idx.items():
#     print(index, Fore.RED, classname, Fore.RESET)
class_to_idx = cifar100.class_to_idx
idx_to_class = {y: x for x, y in cifar100.class_to_idx.items()}
# for k, v in index_to_class.items():
#     print(k, v)

