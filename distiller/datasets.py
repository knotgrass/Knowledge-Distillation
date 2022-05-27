import os
import os.path as osp
import pickle, re
from tqdm import tqdm
from PIL import Image
from typing import Any, Callable, Optional, Tuple, Dict, List, cast
from torch import Tensor
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import albumentations as A
from torchvision.datasets import CIFAR100
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder

from .pseudo_teacher import PseudoTeacher

from torchvision import set_image_backend
set_image_backend('accimage')

#TODO
# Write custom dataset 
# kế thừa class CIFAR100
# viết lại hàm __init__ và __getitem__ để lấy được outp_teacher
# lưu lại outpTeacher/soft_label , outpTeacher/soft_label là vector 
    # outpT must be vecto of Probability distribution of class
    # ex [0.1, 0.2, 0.3, 0.4, 0.0, ] 5class
    # one hot vector is ok but in largely case (99.99999999 %) isn't one-hot vector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input_size = (224, 224)
original_transform = T.Compose([
    T.Resize(size=input_size),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# album_transform = A.Compose([
#     A.Resize(*input_size, always_apply=True),
#     A.Normalize(always_apply=True),
#     ToTensorV2(always_apply=True),
# ], p=1.0)


class AlbumDatasetFolder(DatasetFolder):
    def __getitem__(self, index: int) -> Tuple[Tensor, Any]:
        r"""
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

"""
class AlbumImageFolder(ImageFolder):
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        # Args:
        #     index (int): Index

        # Returns:
        #     tuple: (sample, target) where target is class_index of the target class.
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
"""

class AlbumImageFolder_forKD(DatasetFolder):
    r"""
    giảm thiểu việc teacher inference lặp đi lặp lại với data ko augument
    bằng cách chạy inference trước 1 lượt và lưu lại trong object dataset
    có thể save lại các soft_label ở dạng numpy .npz, 
    khi cần sẽ load lên, các file npz sẽ trùng tên và cùng chung folder với ảnh
    ví dụ folder1/img1.jpg , folder1/img1.npz
    
    #NOTE prepare dataset
    dataset = AlbumImageFolder_forKD(root=root, transform=transform)
    dataset.create_softlabel(teacher, is_loader, device)
    
    #NOTE prepare dataloader
    loader = DataLoader(dataset, batch_size, shuffle, n_workers, pin_memory, generator)
    
    #NOTE training
    for (datas, targets, soft_label, is_aug) in loader:
        datas = datas.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        if is_aug:
            # with torch.no_grad:
                soft_label = teacher(datas).detach()
        outp = student(datas)
        loss = criterion(outp, targets, soft_label)
        loss.backward()
        optimizer.step()
    """
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = albumen_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(AlbumImageFolder_forKD, self).__init__(
            root, loader, 
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file)
        
        self.imgs = self.samples

    def _imgp_to_softlabel(self, teacher:nn.Module, imgp:str, 
                           device:torch.device) -> Tensor:
        img = self.loader(imgp)
        #NOTE apply same transform with student
        img = self.transform(img).to(device)
        soft_label = teacher(img).detach().cpu()
        return soft_label
        
    @torch.no_grad
    def create_softlabel(self, teacher:nn.Module, 
                         device:torch.device=torch.device('cpu')):

        teacher.to(device)
        self.soft_labels = []   #NOTE output of teacher
        with tqdm(self.samples, ncols=128, colour='YELLOW') as progress:
            for imgp, target in self.samples:
                ext = '.' + osp.basename(imgp).split('.')[-1]
                # https://tinyurl.com/yeyvz728
                soft_label_path = re.sub('{}$'.format(ext), '.npz', imgp)
                if osp.isfile(soft_label_path):
                    soft_label = np.load(soft_label_path)   #FIXME
                    self.soft_labels.append(soft_label)
                else:
                    soft_label = self._imgp_to_softlabel(teacher, imgp, device)
                    np.save(soft_label_path, soft_label)    #FIXME
                    self.soft_labels.append(soft_label)
                progress.update()
        
    def __getitem__(self, index: int) -> Tuple[Tensor, Any, Tensor]:
        r"""
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, soft_label, is_aug) where target is class_index of the target class.
            is_aug:True data aug, run teacher predict
            is_aug:False data origin, load from Dataloader
        """

        path, target = self.samples[index]
        soft_label = self.soft_labels[index]
        sample = self.loader(path)
        is_aug:bool = False
        
        if self.transform is not None:
            sample = self.transform(image=sample)['image']
            if len(self.transform.transforms) > 3: 
                is_aug = True

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target, soft_label, is_aug
    

class MixupImageFolder(DatasetFolder):
    # TODO implement dataset to using mixup aug
    # REF https://tinyurl.com/4dyv2sy3
    
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = albumen_loader,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        ...
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]: 
        ...
    
    
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
            file_path = osp.join(self.root, self.base_folder, file_name)
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
                with torch.no_grad:
                    outp = self.teacher(img.to(device)).detach().cpu().numpy()
                self.outp_teacher.extend(outp)
        
        else:   # use pseudo teacher
            raise NotImplementedError(
                "only support torch.nn.Module model")
            # for target in self.targets:
            #     self.outps_teacher.append(self.teacher(target))

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
    
