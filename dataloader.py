import os
from typing import Any, Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
from colorama import Fore
from torch.utils.data import ConcatDataset, DataLoader, random_split
# from torchvision.datasets import CIFAR100, ImageFolder, DatasetFolder
from torchvision.datasets.folder import (IMG_EXTENSIONS, DatasetFolder,
                                         ImageFolder)

from config.cfg import batch_size, dataset_root, generator, transformers

n_workers = os.cpu_count()

class AlbumDatasetFolder(DatasetFolder):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    

def albumen_loader(path:str) -> np.ndarray:
    # fpath = os.path.join(image_dir_path, fn)
    # img = cv2.imdecode(np.fromfile(fpath), cv2.IMREAD_COLOR)
    # img = cv2.imread(fpath)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

class AlbumImageFolder(AlbumDatasetFolder):
    
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = albumen_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(AlbumImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


# class AlbumImageFolder(ImageFolder):
#     def __init__(
#             self,
#             root: str,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = albumen_loader,
#             is_valid_file: Optional[Callable[[str], bool]] = None,
#     ):
#         super(AlbumImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
#                                           transform=transform,
#                                           target_transform=target_transform,
#                                           is_valid_file=is_valid_file)
#         self.imgs = self.samples

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]
#         sample = self.loader(path)
#         if self.transform is not None:
#             sample = self.transform(image=sample)['image']
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target


origin = ImageFolder(root=dataset_root, transform=transformers['original'])
light = AlbumImageFolder(root=dataset_root, transform=transformers['light'])
medium = AlbumImageFolder(root=dataset_root, transform=transformers['medium'])
strong = AlbumImageFolder(root=dataset_root, transform=transformers['strong'])


test_size = round(0.2*len(origin))
train_val, test = random_split(origin, [len(origin)-test_size, test_size], generator=generator)

train_val = ConcatDataset([train_val, light, medium, strong])
num_classes = len(origin.class_to_idx)
class_to_idx = origin.class_to_idx
classes = origin.classes

del light, medium, strong, origin
val_size = round(0.1*len(train_val))
train_size = len(train_val) - val_size

train, val = random_split(train_val, [train_size, val_size], generator=generator)

loaders = {
'train':DataLoader(train, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, generator=generator),
'val':DataLoader(val, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, generator=generator),
'test':DataLoader(test, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, generator=generator)
}

dataset_sizes = {
    'train': train_size,
    'val': val_size,
    'test': test_size,
}

if __name__ == '__main__':
    for idx, cls in enumerate(classes):
        print(Fore.RED,idx,Fore.RESET, cls)

