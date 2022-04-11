import os.path
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable, Optional, Tuple
import pickle
import numpy as np


import torch
import torch.nn as nn
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as T

from ..dataloader import loaders
from ..config import cfg
from models.model import teacher
from .pseudo_teacher import PseudoTeacher


#TODO
# Write custom dataset 
# kế thừa class CIFAR100
# viết lại hàm __init__ và __getitem__ để lấy được outp_teacher
# lưu lại outpTeacher , outpTeacher là vector 
    # outpT must be vecto of Probability distribution of class
    # ex [0.1, 0.2, 0.3, 0.4, 0.0, ] 5class
    # one hot vector is ok but largely isn't one-hot vecto

device = cfg.device
original_transform = cfg.transformer['original']

class CIFAR100_ForKD(CIFAR100):
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transformS: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            teacher:Any = PseudoTeacher()
    ) -> None:

        super(CIFAR100_ForKD, self).__init__(root, transform=transformS,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        self.teacher = teacher  # teacher
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        # self.idx_to_class = {y: x for x, y in self.class_to_idx.items()}
        
        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.fetch_outp_teacher()

    def fetch_outp_teacher(self) -> None:
        self.outps_teacher = []
        
        if isinstance(self.teacher, nn.Module):
            self.teacher.to(device).eval()
            
            for img in self.data:
                img = Image.fromarray(img)
                img = original_transform(img)
                img.to(device)
                with torch.no_grad:
                    outp = self.teacher(img).detach().cpu().numpy()
                self.outp_teacher.extend(outp)
        
        else:   # use pseudo teacher
            for target in self.targets:
                self.outps_teacher.append(self.teacher(target))

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, outpT) 
            where target is index of the target class.
            outpT is index of predict of teacher models
        """
        img, target , outpT = self.data[idx], self.targets[idx], self.outps_teacher[idx]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, outpT