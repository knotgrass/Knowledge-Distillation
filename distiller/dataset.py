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

from ..data import loaders
from ..config import cfg
from models.model import teacher



#TODO
# Write custom dataset 
# kế thừa class CIFAR100
# viết lại hàm __init__ và __getitem__ để lấy được outp_teacher
# lưu lại outpTeacher với format giống target(label/gt)

phase = 'train'

for datas, targets , outp_Teacher in tqdm(loaders[phase], ncols=64, colour='black', 
                            desc='{:6}'.format(phase).capitalize()):
    ...
# loader sẽ chạy trước 1 lượt, và save output ra ssd, khi cần sẽ load lên



loaderT = DataLoader(
    dataset=CIFAR100(root=cfg.dataset_root, 
                     train=True, download=True, 
                     transform=cfg.transformers['original']), 
    batch_size=cfg.batch_size, 
    shuffle=True,
    num_workers=cfg.n_workers, 
    generator=cfg.generator, 
    pin_memory=True)

device = cfg.device

            
        

    


class CIFAR100_ForKD(CIFAR100):
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transformS: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            teacher:nn.Module = ...
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
        self.idx_to_class = {y: x for x, y in self.class_to_idx.items()}
        
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
        self.teacher.to(device).eval()
        self.outp_teacher = []
        
        for img in self.data:
            img = Image.fromarray(img)
            img = cfg.transformers['original'](img)
            img.to(device)
            with torch.no_grad:
                outp = self.teacher(img).detach().cpu().numpy()
            self.outp_teacher.extend(outp)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, outpT) 
            where target is index of the target class.
            outpT is index of predict of teacher models
        """
        img, target , outpT = self.data[index], self.targets[index], self.outp_teacher

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            outpT = self.target_transform(outpT)

        return img, target, outpT