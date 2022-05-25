import os
from colorama import Fore
from torch.utils.data import ConcatDataset, DataLoader, random_split
# from torchvision.datasets import CIFAR100, ImageFolder, DatasetFolder
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch._C import Generator
from distiller.datasets import AlbumImageFolder
#https://github.com/pytorch/accimage#accimage
from torchvision import set_image_backend
set_image_backend('accimage')

# class AlbumDatasetFolder(DatasetFolder):
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         r"""
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
    

# def albumen_loader(path:str) -> np.ndarray:
#     # fpath = os.path.join(image_dir_path, fn)
#     # img = cv2.imdecode(np.fromfile(fpath), cv2.IMREAD_COLOR)
#     # img = cv2.imread(fpath)
#     return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


# class AlbumImageFolder(AlbumDatasetFolder):
    
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

#TODO config for each dataset
dataset_root = 'dataset/intel-image-classification/seg_train'
batch_size = 8

n_workers = os.cpu_count()
lr = 0.001
temperature = 6
alpha = 0.1
seed = 111

generator = Generator()
generator.manual_seed(seed)

input_size = (224, 224)
# mean = (0.4915, 0.4823, 0.4468)
# std = (0.2470, 0.2435, 0.2616)
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

transformers = {
    'original': T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),

    'dataset1': T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomRotation(2),
        # T.RandomAffine(degrees=11, translate=(0.1, 0.1), scale=(0.8, 0.8)),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]),

    'dataset2': T.Compose([
        T.RandomHorizontalFlip(p=0.68),
        T.RandomErasing(inplace=True, scale=(0.01, 0.23)),
        # T.RandomAffine(translate=(0.05, 0.05), degrees=0),
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),

    'dataset3': T.Compose([
        T.Resize(input_size),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomRotation(15),
        # T.RandomAffine(translate=(0.08, 0.1), degrees=15),
        T.ToTensor(),
        T.Normalize(mean, std)
    ]),
    
    'light': A.Compose([
        A.RandomBrightnessContrast(p=0.5),    
        A.RandomGamma(p=0.5),    
        A.CLAHE(p=0.5),
        A.HorizontalFlip(p=0.9),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0),
    
    'medium': A.Compose([
        A.CLAHE(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
        A.HorizontalFlip(p=0.6),
        A.MedianBlur(p=0.3),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0),
        
    'strong' : A.Compose([
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25),
        A.ChannelShuffle(p=0.4),
        A.HorizontalFlip(p=0.6),
        A.JpegCompression(p=0.5),
        A.Resize(*input_size, always_apply=True),
        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True),
        ], p=1.0)
}


origin = ImageFolder(root=dataset_root, transform=transformers['original'])
light = AlbumImageFolder(root=dataset_root, transform=transformers['light'])
medium = AlbumImageFolder(root=dataset_root, transform=transformers['medium'])
strong = AlbumImageFolder(root=dataset_root, transform=transformers['strong'])


test_size = round(0.2*len(origin))
train_val, test = random_split(origin, [len(origin)-test_size, test_size], generator=generator)

train_val = ConcatDataset([train_val, light, medium, strong])
class_to_idx = origin.class_to_idx
num_classes = len(class_to_idx)
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
del generator

dataset_sizes = {
    'train': train_size,
    'val': val_size,
    'test': test_size,
}

if __name__ == '__main__':
    for idx, cls in enumerate(classes):
        print(Fore.RED,idx,Fore.RESET, cls)

