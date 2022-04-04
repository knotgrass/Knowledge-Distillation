import os
from colorama import Fore

from torch._C import Generator
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR100#, ImageFolder
import torchvision.transforms as T

mean = (0.4915, 0.4823, 0.4468)
std = (0.2470, 0.2435, 0.2616)

transformers = T.Compose([
    T.Resize(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

cifar100 = CIFAR100('.dataset', train=True,
                    download=True, transform=transformers)

dataset_size = len(cifar100)

test_size = round(0.2*dataset_size)
val_size = round(0.1*dataset_size)
train_size = dataset_size - val_size - test_size
generator = Generator()
generator.manual_seed(0)

train, val, test = random_split(cifar100, [train_size, val_size, test_size],
                                generator=generator)

batch_size = 32
n_workers = os.cpu_count()
loaders = {}
loaders['train'] = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, generator=generator, pin_memory=True)
loaders['val'] = DataLoader(val, batch_size=batch_size, shuffle=True,
                            num_workers=n_workers, generator=generator, pin_memory=True)
loaders['test'] = DataLoader(test, batch_size=batch_size, shuffle=False,
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


#TODO
# Write custom dataset 
# kế thừa class CIFAR100
# viết lại hàm __init__ và __getitem__ để lấy được outp_teacher
# lưu lại outpTeacher với format giống target(label/gt)

phase = 'train'
from tqdm import tqdm
for datas, targets , outp_Teacher in tqdm(loaders[phase], ncols=64, colour='black', 
                            desc='{:6}'.format(phase).capitalize()):
    ...
# loader sẽ chạy trước 1 lượt, và save output ra ssd, khi cần sẽ load lên

from PIL import Image
from typing import Any, Callable, Optional, Tuple
import pickle
import numpy as np
class CustomKDDataset(CIFAR100):
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CustomKDDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

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
        
    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, outp_Teacher) 
            where target is index of the target class.
            outp_Teacher is predict of teacher models
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target